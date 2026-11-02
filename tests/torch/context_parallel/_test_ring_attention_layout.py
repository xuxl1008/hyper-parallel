# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Distributed numerical validation for dense and packed balanced Ring attention."""

from datetime import timedelta
from functools import partial
from itertools import accumulate
import os

import pytest
import torch
import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.distributed.context_parallel.ring_attention import ulysses_ring_attention
from tests.torch.context_parallel._test_ring_attention_distributed import _check_memory
from tests.torch.context_parallel._test_ring_attention_npu import (
    _Case, _compare, _inputs, _math_reference, _native_reference,
)


@pytest.fixture(name="environment", scope="module")
def _environment():
    size = int(os.environ["WORLD_SIZE"])
    if not torch.npu.is_available() or torch.npu.device_count() < size:
        pytest.skip(f"requires {size} NPUs")
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    deterministic = torch.are_deterministic_algorithms_enabled()
    # Native TND backward can atomically reduce dK/dV; exact schedule comparison
    # requires the deterministic kernel, just as deterministic Trainer runs do.
    torch.use_deterministic_algorithms(True)
    try:
        yield device, init_device_mesh("npu", (size,), mesh_dim_names=("cp",))
    finally:
        torch.use_deterministic_algorithms(deterministic)
        dist.destroy_process_group()


def _packed_reference(inputs, lengths, causal, native):
    query, key, value = (tensor.detach().clone().requires_grad_() for tensor in inputs[:3])
    if native:
        ends = tuple(accumulate(lengths))
        mask = torch.ones(2048, 2048, dtype=torch.bool, device=query.device).triu_(1) if causal else None
        output = torch.ops.npu.npu_fusion_attention(
            query, key, value, head_num=query.shape[1], input_layout="TND", scale=query.shape[-1] ** -0.5,
            actual_seq_qlen=ends, actual_seq_kvlen=ends, atten_mask=mask, sparse_mode=3 if causal else 0,
            pre_tockens=2147483647, next_tockens=0 if causal else 2147483647, keep_prob=1.,
        )[0]
    else:
        qmath = query.transpose(0, 1)
        kmath, vmath = (tensor.transpose(0, 1).repeat_interleave(query.shape[1] // key.shape[1], 0)
                        for tensor in (key, value))
        documents = torch.tensor([doc for doc, length in enumerate(lengths) for _ in range(length)])
        allowed = documents[:, None] == documents[None, :]
        if causal:
            allowed &= torch.arange(sum(lengths))[:, None] >= torch.arange(sum(lengths))[None, :]
        logits = (qmath @ kmath.transpose(-1, -2)) * query.shape[-1] ** -0.5
        output = (logits.masked_fill(~allowed, -float("inf")).softmax(-1) @ vmath).transpose(0, 1)
    grads = torch.autograd.grad(output, (query, key, value), inputs[3])
    return (output.detach(), *grads)


def _check(environment, degree, dtype, layout, mode, causal=True, *, owner=None, strided=False,
           needed=(True, True, True), side_stream=False, remainder=False, tiny=False, mha=False):
    device, mesh = environment
    total = (2 if tiny else 33 if remainder else 64) * mesh.size()
    case = _Case(dtype, "BSND" if layout == "TND" else layout, 4, causal, mesh.size(),
                 sequence_length=total, query_heads=4 if mha else 8, batch_size=1 if layout == "TND" else 2)
    lengths = ((1,) * total if tiny else (1, 7, 67, total - 75)) if layout == "TND" else None
    cpu = _inputs(case)
    if layout == "TND":
        cpu = tuple(tensor.squeeze(0) for tensor in cpu)
    axis = {"TND": 0, "BSND": 1, "BNSD": 2}[layout]
    local = total // mesh.size()
    if owner is not None:
        gradient = torch.zeros_like(cpu[3])
        gradient.narrow(axis, owner * local, local).copy_(cpu[3].narrow(axis, owner * local, local))
        cpu = (*cpu[:3], gradient)
    full = tuple(tensor.to(device) for tensor in cpu)
    if layout == "TND":
        expected = _packed_reference(full, lengths, causal, True)
        mathematical = _packed_reference(tuple(tensor.double() for tensor in cpu), lengths, causal, False)
    else:
        mask = torch.ones(2048, 2048, device=device, dtype=torch.bool).triu_(1)
        expected = _native_reference(*full, case, mask)
        mathematical = _math_reference(cpu, case)
    rank = mesh.get_local_rank()
    inputs = tuple(tensor.narrow(axis, rank * local, local).clone().detach() for tensor in full)
    if strided:
        inputs = tuple(tensor.transpose(-1, -2).contiguous().transpose(-1, -2) for tensor in inputs)
    for tensor, requires_grad in zip(inputs[:3], needed):
        tensor.requires_grad_(requires_grad)
    snapshots = tuple(tensor.detach().clone() for tensor in inputs[:3])
    options = {"ulysses_degree": degree, "input_layout": layout, "causal": causal, "load_balance": mode,
               "document_lengths": lengths,
               "balance_block_size": min(8, local) if mode in ("greedy", "hybrid") else None,
               "zigzag_min_length": 64 if mode == "hybrid" else None}

    def _run(overlap):
        output = ulysses_ring_attention(*inputs[:3], mesh, overlap=overlap, **options)
        differentiable = tuple(tensor for tensor in inputs[:3] if tensor.requires_grad)
        grads = iter(torch.autograd.grad(output, differentiable, inputs[3]))
        return (output, *(next(grads) if tensor.requires_grad else None for tensor in inputs[:3]))

    synchronous = _run(False)
    stream = torch.npu.Stream(device=device) if side_stream else torch.npu.current_stream(device)
    stream.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(stream):
        actual = _run(True)
    torch.npu.current_stream(device).wait_stream(stream)
    errors = []
    for name, value, sync, reference, math_value in zip(("O", "dQ", "dK", "dV"),
                                                       actual, synchronous, expected, mathematical):
        if value is not None:
            torch.testing.assert_close(value, sync, atol=0, rtol=0,
                                       msg=lambda message: f"{name}: sync/overlap: {message}")
            errors.append(_compare(value, reference.narrow(axis, rank * local, local), case, name + " native"))
            if tiny and name in ("dQ", "dK"):
                # A single-key softmax has exactly zero Q/K derivatives. Native
                # FP32 reduction can leave roundoff, so relative L2 is undefined.
                for gradient in (value, reference.narrow(axis, rank * local, local)):
                    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-6, rtol=0)
            else:
                _compare(value, math_value.narrow(axis, rank * local, local), case, name + " FP64")
    for tensor, snapshot in zip(inputs[:3], snapshots):
        torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)
    print(f"rank={rank} U={degree} layout={layout} mode={mode} causal={causal} errors={errors}", flush=True)


@pytest.mark.parametrize("degree", [1, 2, 4] if int(os.environ.get("WORLD_SIZE", "1")) >= 8 else [1, 2])
class TestRingLayouts:
    """Preserve document isolation, owner gradients and exact sync/overlap agreement."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    def test_dense_zigzag(self, environment, degree, dtype, layout):
        """Validate the one-kernel-per-round dense Zigzag path."""
        _check(environment, degree, dtype, layout, "zigzag")

    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    @pytest.mark.parametrize("mode", ["greedy", "hybrid"])
    @pytest.mark.parametrize("causal", [False, True])
    def test_dense_task_batches(self, environment, degree, layout, mode, causal):
        """Fold independent dense batch rows into native TND heads without mixing samples."""
        _check(environment, degree, torch.bfloat16, layout, mode, causal)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("mode", ["none", "zigzag", "greedy", "hybrid"])
    def test_packed(self, environment, degree, dtype, causal, mode):
        """Different documents, short fragments and variable worker counts remain independent."""
        _check(environment, degree, dtype, "TND", mode, causal)

    @pytest.mark.parametrize("mode", ["zigzag", "greedy", "hybrid"])
    @pytest.mark.parametrize("last", [False, True])
    def test_gradient_owner(self, environment, degree, mode, last):
        """Return all dKV when only the first or last source rank contributes loss."""
        _check(environment, degree, torch.bfloat16, "TND", mode,
               owner=environment[1].size() - 1 if last else 0)

    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    def test_remainders(self, environment, degree, layout):
        """Odd local lengths use real-token remainder tasks, including strided inputs."""
        _check(environment, degree, torch.bfloat16, layout, "zigzag", remainder=True, strided=True)

    @pytest.mark.parametrize("needed", [(False, True, True), (True, False, False)])
    def test_partial_gradients(self, environment, degree, needed):
        """Reordered tensors with frozen projections keep collective order valid."""
        _check(environment, degree, torch.bfloat16, "TND", "greedy", needed=needed)

    def test_stream_and_memory(self, environment, degree):
        """Preserve nondefault stream ordering and release route/transport activations."""
        _check(environment, degree, torch.bfloat16, "TND", "hybrid", side_stream=True)
        device, mesh = environment
        case = _Case(torch.bfloat16, "BNSD", 4, True, 1, sequence_length=128, query_heads=8)
        attention = partial(ulysses_ring_attention, ulysses_degree=degree, overlap=True, load_balance="zigzag")
        _check_memory(device, mesh, case, attention)

    @pytest.mark.parametrize("mode", ["zigzag", "greedy"])
    def test_single_token_documents(self, environment, degree, mode):
        """No cross-document attention, including empty remote row exchanges and MHA."""
        for _ in range(3):
            _check(environment, degree, torch.bfloat16, "TND", mode, tiny=True, mha=True)

    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    def test_dense_mha(self, environment, degree, layout):
        """Check MHA with two independent batch rows under internal Zigzag routing."""
        _check(environment, degree, torch.bfloat16, layout, "zigzag", mha=True)
