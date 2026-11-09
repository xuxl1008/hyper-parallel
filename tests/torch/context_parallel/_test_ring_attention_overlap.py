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
"""Real NPU checks for pipelined Ring and Ulysses + Ring forward/backward."""

from datetime import timedelta
from functools import partial
import os

import pytest
import torch
import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.distributed.context_parallel.ring_attention import ulysses_ring_attention
from tests.torch.context_parallel._test_ring_attention_distributed import _check_case, _check_memory
from tests.torch.context_parallel._test_ring_attention_npu import _Case, _inputs


@pytest.fixture(name="environment", scope="module")
def _environment():
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.npu.is_available() or torch.npu.device_count() < size:
        pytest.skip(f"Ring overlap tests require {size} visible NPU devices")
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        mesh = init_device_mesh("npu", (size,), mesh_dim_names=("cp",))
        yield device, mesh
    finally:
        dist.destroy_process_group()


def _attention(degree):
    return partial(ulysses_ring_attention, ulysses_degree=degree, overlap=True)


def _case(degree, dtype, layout, kv_heads, causal, parts, **kwargs):
    # U=4 needs at least four compact KV heads; retain MHA/GQA ratios.
    factor = max(1, degree // 2)
    return _Case(dtype, layout, kv_heads * factor, causal, parts, query_heads=4 * factor, **kwargs)


def _values(inputs, mesh, degree, overlap):
    output = ulysses_ring_attention(*inputs[:3], mesh, ulysses_degree=degree, causal=True, overlap=overlap)
    gradients = torch.autograd.grad(output, inputs[:3], grad_outputs=inputs[3])
    return (output, *gradients)


@pytest.mark.parametrize("degree", [1, 2, 4] if int(os.environ.get("WORLD_SIZE", "1")) >= 8 else [1, 2],
                         ids=lambda degree: f"ulysses{degree}")
class TestRingAttentionOverlap:
    """The overlapped schedule must retain sync numerics and stream-safe lifetimes."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    @pytest.mark.parametrize("kv_heads", [4, 2], ids=["mha", "gqa"])
    @pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
    def test_precision(self, environment, degree, dtype, layout, kv_heads, causal) -> None:
        """Compare prefetch and delayed dKV accumulation against NPU and CPU FP64 references."""
        device, mesh = environment
        case = _case(degree, dtype, layout, kv_heads, causal, mesh.size(), sequence_length=128 * mesh.size())
        _check_case(device, mesh, case, attention=_attention(degree))

    @pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
    @pytest.mark.parametrize("last_owner", [False, True], ids=["first_owner", "last_owner"])
    def test_gradient_owner(self, environment, degree, causal, last_owner) -> None:
        """Ranks with zero upstream gradients must still forward KV and dKV in matching order."""
        device, mesh = environment
        case = _case(degree, torch.bfloat16, "BNSD", 2, causal, mesh.size(), sequence_length=128 * mesh.size())
        _check_case(device, mesh, case, active_owner=mesh.size() - 1 if last_owner else 0,
                    attention=_attention(degree))

    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    def test_unaligned_noncontiguous(self, environment, degree, layout) -> None:
        """Strided input snapshots remain unchanged with unaligned local sequence lengths."""
        device, mesh = environment
        case = _case(degree, torch.bfloat16, layout, 2, True, mesh.size(), sequence_length=33 * mesh.size())
        _check_case(device, mesh, case, noncontiguous=True, attention=_attention(degree))

    @pytest.mark.parametrize("needed", [(False, True, True), (True, False, False)], ids=["kv_only", "q_only"])
    def test_partial_requires_grad(self, environment, degree, needed) -> None:
        """Frozen projections keep the remaining overlapped backward collective order valid."""
        device, mesh = environment
        case = _case(degree, torch.float16, "BNSD", 2, True, mesh.size(), sequence_length=64 * mesh.size())
        _check_case(device, mesh, case, requires_grad=needed, attention=_attention(degree))

    def test_repeated_calls_release_buffers(self, environment, degree) -> None:
        """Outstanding work handles and payload references must be released after backward."""
        device, mesh = environment
        case = _case(degree, torch.bfloat16, "BNSD", 2, True, 1, sequence_length=512, head_dim=128)
        _check_memory(device, mesh, case, attention=_attention(degree))

    @pytest.mark.parametrize("side_stream", [False, True], ids=["default_stream", "side_stream"])
    def test_sync_parity_and_stream_order(self, environment, degree, side_stream) -> None:
        """Preserve exact sync results even when the whole call uses a nondefault compute stream."""
        device, mesh = environment
        case = _case(degree, torch.bfloat16, "BNSD", 2, True, 1, sequence_length=128)
        inputs = tuple(tensor.to(device).requires_grad_(index < 3) for index, tensor in enumerate(_inputs(case)))
        expected = _values(inputs, mesh, degree, False)
        stream = torch.npu.Stream(device=device) if side_stream else torch.npu.current_stream(device)
        stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(stream):
            actual = _values(inputs, mesh, degree, True)
        torch.npu.current_stream(device).wait_stream(stream)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
