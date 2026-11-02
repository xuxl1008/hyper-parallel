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
"""Real multi-card workers for the synchronous NPU Ring Attention baseline."""

from datetime import timedelta
import gc
import os

import pytest
import torch
import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.distributed.context_parallel.ring_attention import ring_attention
from tests.torch.context_parallel._test_ring_attention_npu import (
    _Case, _compare, _inputs, _math_reference, _native_reference,
)


@pytest.fixture(name="environment", scope="module")
def _environment():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.npu.is_available() or torch.npu.device_count() < world_size:
        pytest.skip(f"Ring Attention tests require {world_size} visible NPU devices")
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("cp",))
        yield device, mesh
    finally:
        dist.destroy_process_group()


def _check_case(
    device, mesh, case, active_owner=None, noncontiguous=False, requires_grad=(True, True, True),
    attention=ring_attention,
):
    rank = 0 if mesh is None else mesh.get_local_rank()
    size = 1 if mesh is None else mesh.size()
    local_length = case.sequence_length // size
    cpu_inputs = _inputs(case)
    if mesh is not None:
        # Distinct data per subgroup catches accidental communication through the default group.
        offset = dist.get_global_rank(mesh.get_group(), 0) / 8
        cpu_inputs = tuple(tensor + offset for tensor in cpu_inputs)
    if active_owner is not None:
        grad_output = torch.zeros_like(cpu_inputs[3])
        grad_output.narrow(case.seq_dim, active_owner * local_length, local_length).copy_(
            cpu_inputs[3].narrow(case.seq_dim, active_owner * local_length, local_length),
        )
        cpu_inputs = (*cpu_inputs[:3], grad_output)

    full_inputs = tuple(tensor.to(device) for tensor in cpu_inputs)
    mask = torch.ones((2048, 2048), dtype=torch.bool, device=device).triu(1)
    expected = _native_reference(*full_inputs, case, mask)
    math_expected = _math_reference(cpu_inputs, case)
    local = tuple(tensor.narrow(case.seq_dim, rank * local_length, local_length).detach().clone()
                  for tensor in full_inputs)
    if noncontiguous:
        local = tuple(tensor.transpose(1, 2).contiguous().transpose(1, 2) for tensor in local)
    for tensor, needed in zip(local[:3], requires_grad):
        tensor.requires_grad_(needed)
    snapshots = tuple(tensor.detach().clone() for tensor in local[:3])
    saved = []

    def _pack(tensor):
        saved.append((tuple(tensor.shape), tensor.numel() * tensor.element_size()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(_pack, lambda tensor: tensor):
        actual = attention(*local[:3], mesh, causal=case.causal, input_layout=case.layout)
    actual.backward(local[3])
    actual_values = (actual, *(tensor.grad for tensor in local[:3]))
    errors = []
    for name, actual_value, reference, math_reference in zip(
        ("output", "dQ", "dK", "dV"), actual_values, expected, math_expected,
    ):
        if actual_value is None:
            continue
        reference = reference.narrow(case.seq_dim, rank * local_length, local_length)
        math_reference = math_reference.narrow(case.seq_dim, rank * local_length, local_length)
        errors.append(_compare(actual_value, reference, case, f"rank={rank}, attention/{name} vs native"))
        _compare(actual_value, math_reference, case, f"rank={rank}, attention/{name} vs CPU FP64")
    for tensor, snapshot in zip(local[:3], snapshots):
        torch.testing.assert_close(tensor.detach(), snapshot, atol=0, rtol=0)
    expected_saved_bytes = (
        sum(tensor.numel() * tensor.element_size() for tensor in (*local[:3], actual))
        + 2 * case.batch_size * case.query_heads * local_length * 8 * 4
    )
    actual_saved_bytes = sum(byte_count for _, byte_count in saved)
    assert len(saved) == 6 and actual_saved_bytes == expected_saved_bytes, (
        f"saved tensors={saved}, bytes={actual_saved_bytes}; expected 6 local tensors, bytes={expected_saved_bytes}"
    )
    print(f"rank={rank}, CP={size}, owner={active_owner}, {case}, errors={errors}, saved_bytes={actual_saved_bytes}")


def _memory_iteration(inputs, mesh, attention):
    output = attention(*inputs[:3], mesh, causal=True)
    gradients = torch.autograd.grad(output, inputs[:3], grad_outputs=inputs[3])
    del output, gradients


def _check_memory(device, mesh, case, attention=ring_attention):
    inputs = tuple(tensor.to(device).requires_grad_(index < 3) for index, tensor in enumerate(_inputs(case)))
    for _ in range(3):
        _memory_iteration(inputs, mesh, attention)
    torch.npu.synchronize()
    gc.collect()
    baseline = torch.npu.memory_allocated(device)
    torch.npu.reset_peak_memory_stats(device)
    measurements = []
    for _ in range(8):
        _memory_iteration(inputs, mesh, attention)
        torch.npu.synchronize()
        gc.collect()
        measurements.append(torch.npu.memory_allocated(device))
    peak_extra = torch.npu.max_memory_allocated(device) - baseline
    growth = max(measurements) - baseline
    assert growth <= 65536, (
        f"live memory grew by {growth} bytes; expected <=65536, baseline={baseline}, samples={measurements}"
    )
    print(f"CP={mesh.size()}, rank={mesh.get_local_rank()}, live_growth={growth}, peak_extra={peak_extra}")


class TestRingAttention:
    """Validate real communication, owner gradients and bounded saved activations."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    @pytest.mark.parametrize("kv_heads", [4, 2, 1], ids=["mha", "gqa", "mqa"])
    @pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
    def test_precision(self, environment, dtype, layout, kv_heads, causal) -> None:
        """Match full-sequence native and CPU math references for output and every input gradient."""
        device, mesh = environment
        case = _Case(dtype, layout, kv_heads, causal, mesh.size(), sequence_length=128 * mesh.size())
        _check_case(device, mesh, case)

    @pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
    @pytest.mark.parametrize("last_owner", [False, True], ids=["first_owner", "last_owner"])
    def test_gradient_owner(self, environment, causal, last_owner) -> None:
        """Only one Q owner contributes loss; other ranks must still return its dKV contributions."""
        device, mesh = environment
        case = _Case(torch.bfloat16, "BNSD", 2, causal, mesh.size(), sequence_length=128 * mesh.size())
        _check_case(device, mesh, case, active_owner=mesh.size() - 1 if last_owner else 0)

    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    def test_unaligned_noncontiguous(self, environment, layout) -> None:
        """Accept strided local QKV and 33-token blocks without writing to caller-owned storage."""
        device, mesh = environment
        case = _Case(torch.bfloat16, layout, 2, True, mesh.size(), sequence_length=33 * mesh.size())
        _check_case(device, mesh, case, noncontiguous=True)

    def test_partial_requires_grad(self, environment) -> None:
        """The gradient ring remains correct when only K and V require gradients."""
        device, mesh = environment
        case = _Case(torch.float16, "BNSD", 1, True, mesh.size(), sequence_length=64 * mesh.size())
        _check_case(device, mesh, case, requires_grad=(False, True, True))

    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    def test_single_rank(self, environment, layout) -> None:
        """The public autograd entry also works without any CP mesh or communication."""
        device, _ = environment
        _check_case(device, None, _Case(torch.bfloat16, layout, 2, True, 1))

    def test_noncontiguous_subgroup(self, environment) -> None:
        """Use group ranks, rather than assuming CP peers are adjacent global ranks."""
        device, world_mesh = environment
        if world_mesh.size() < 4:
            pytest.skip("noncontiguous CP subgroups require four ranks")
        mesh = init_device_mesh("npu", (world_mesh.size() // 2, 2), mesh_dim_names=("cp", "replica"))["cp"]
        case = _Case(torch.bfloat16, "BNSD", 1, True, mesh.size(), sequence_length=128 * mesh.size())
        _check_case(device, mesh, case, active_owner=mesh.size() - 1)

    def test_repeated_calls_release_buffers(self, environment) -> None:
        """Completed forward/backward calls release their saved tensors and reusable P2P buffers."""
        device, mesh = environment
        case = _Case(torch.bfloat16, "BNSD", 2, True, 1, sequence_length=512, head_dim=128)
        _check_memory(device, mesh, case)
