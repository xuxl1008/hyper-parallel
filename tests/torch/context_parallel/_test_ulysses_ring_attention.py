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
"""Four-card workers for Ulysses + Ring composition and its degenerate modes."""

from datetime import timedelta
from functools import partial
import os

import pytest
import torch
import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.distributed.context_parallel.collectives import (
    _build_hybrid_cp_submeshes, ulysses_head_to_seq, ulysses_seq_to_head,
)
from hyper_parallel.distributed.context_parallel.ring_attention import ulysses_ring_attention
from tests.torch.context_parallel._test_ring_attention_distributed import _check_case, _check_memory
from tests.torch.context_parallel._test_ring_attention_npu import _Case, _inputs

_HYBRID = partial(ulysses_ring_attention, ulysses_degree=2)


@pytest.fixture(name="environment", scope="module")
def _environment():
    if not torch.npu.is_available() or torch.npu.device_count() < 4:
        pytest.skip("Ulysses + Ring 2x2 tests require four visible NPU devices")
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        assert dist.get_world_size() == 4, "launch the 2x2 worker with exactly four processes"
        mesh = init_device_mesh("npu", (4,), mesh_dim_names=("cp",))
        yield device, mesh
    finally:
        dist.destroy_process_group()


class TestUlyssesRingAttention:
    """Check head/sequence ownership, composed autograd and constant live memory."""

    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    def test_layout_and_subgroups(self, environment, layout) -> None:
        """Check real process-group peers and labeled tensor layout before entering attention."""
        device, mesh = environment
        rank = mesh.get_local_rank()
        ulysses_mesh, ring_mesh = _build_hybrid_cp_submeshes(mesh, 2)
        assert dist.get_process_group_ranks(ulysses_mesh.get_group()) == [rank // 2 * 2, rank // 2 * 2 + 1]
        assert dist.get_process_group_ranks(ring_mesh.get_group()) == [rank % 2, rank % 2 + 2]
        assert ulysses_mesh.get_local_rank() == rank % 2
        assert ring_mesh.get_local_rank() == rank // 2
        cached = _build_hybrid_cp_submeshes(mesh, 2)
        assert cached[0] is ulysses_mesh and cached[1] is ring_mesh, "submeshes must be reused across calls"

        full = torch.arange(2 * 4 * 12 * 8, dtype=torch.float32, device=device).reshape(2, 4, 12, 8)
        seq_dim, head_dim = (2, 1) if layout == "BNSD" else (1, 2)
        if layout == "BSND":
            full = full.transpose(1, 2)
        local = full.narrow(seq_dim, rank * 3, 3).clone()
        actual = ulysses_seq_to_head(local, seq_dim, head_dim, ulysses_mesh)
        expected = full.narrow(seq_dim, rank // 2 * 6, 6).narrow(head_dim, rank % 2 * 2, 2)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        restored = ulysses_head_to_seq(actual, seq_dim, head_dim, ulysses_mesh)
        torch.testing.assert_close(restored, local, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    @pytest.mark.parametrize("kv_heads", [4, 2], ids=["mha", "gqa"])
    @pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
    def test_precision(self, environment, dtype, layout, kv_heads, causal) -> None:
        """Compare restored local output/dQ/dK/dV to full-sequence NPU and CPU FP64 attention."""
        device, mesh = environment
        case = _Case(dtype, layout, kv_heads, causal, 4, sequence_length=512)
        _check_case(device, mesh, case, attention=_HYBRID)

    @pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
    @pytest.mark.parametrize("owner", [0, 3], ids=["first_owner", "last_owner"])
    def test_gradient_owner(self, environment, causal, owner) -> None:
        """A single original CP rank supplies loss; dKV must cross both parallel dimensions."""
        device, mesh = environment
        case = _Case(torch.bfloat16, "BNSD", 2, causal, 4, sequence_length=512)
        _check_case(device, mesh, case, active_owner=owner, attention=_HYBRID)

    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    def test_unaligned_noncontiguous(self, environment, layout) -> None:
        """Gather two strided 33-token shards into a 66-token Ring block without padding."""
        device, mesh = environment
        case = _Case(torch.bfloat16, layout, 2, True, 4, sequence_length=132)
        _check_case(device, mesh, case, noncontiguous=True, attention=_HYBRID)

    @pytest.mark.parametrize("needed", [(False, True, True), (True, False, False)], ids=["kv_only", "q_only"])
    def test_partial_requires_grad(self, environment, needed) -> None:
        """Uniform frozen Q or KV projections preserve the remaining collective backward paths."""
        device, mesh = environment
        case = _Case(torch.float16, "BNSD", 2, True, 4, sequence_length=256)
        _check_case(device, mesh, case, requires_grad=needed, attention=_HYBRID)

    @pytest.mark.parametrize("degree", [1, 4], ids=["pure_ring", "pure_ulysses"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    @pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
    def test_degenerate_modes(self, environment, degree, dtype, layout, causal) -> None:
        """The same entry supports U=1 and R=1, including GQA without head replication."""
        device, mesh = environment
        case = _Case(dtype, layout, 4, causal, 4, sequence_length=256, query_heads=8)
        _check_case(device, mesh, case, attention=partial(ulysses_ring_attention, ulysses_degree=degree))

    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    def test_single_rank(self, environment, layout) -> None:
        """No mesh and U=1 executes single-rank GQA through the same public entry."""
        device, _ = environment
        case = _Case(torch.bfloat16, layout, 2, True, 1)
        _check_case(device, None, case, attention=ulysses_ring_attention)

    @pytest.mark.parametrize("layout", ["BNSD", "BSND"])
    @pytest.mark.parametrize("heads", [(4, 1), (3, 3)], ids=["kv_not_divisible", "q_and_kv_not_divisible"])
    def test_unshardable_heads(self, environment, layout, heads) -> None:
        """Reject invalid local head counts on every rank before any all-to-all."""
        device, mesh = environment
        case = _Case(torch.float16, layout, heads[1], True, 1, query_heads=heads[0])
        inputs = tuple(tensor.to(device) for tensor in _inputs(case)[:3])
        with pytest.raises(ValueError, match="local Q and KV head counts"):
            _HYBRID(*inputs, mesh, input_layout=layout)

    def test_repeated_calls_release_buffers(self, environment) -> None:
        """Repeated composed forward/backward calls release both A2A and Ring storage."""
        device, mesh = environment
        case = _Case(torch.bfloat16, "BNSD", 2, True, 1, sequence_length=512, head_dim=128)
        _check_memory(device, mesh, case, attention=_HYBRID)
