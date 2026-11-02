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
"""Distributed numerical and buffer-lifetime checks for the experimental chunk pipeline."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel import init_device_mesh
from hyper_parallel.distributed.context_parallel.ring_attention import ring_attention
from hyper_parallel.distributed.context_parallel.ring_chunk import chunked_ring_attention
from hyper_parallel.distributed.context_parallel.ring_layout import RingAttentionTrace
from tests.torch.context_parallel._benchmark_ring_attention_layout import _full_reference


VARIANTS = {
    "pruned_core": {"prune_documents": True}, "pruned_core_sync": {"prune_documents": True},
    "stream1": {}, "head2": {"chunks": 2}, "head4": {"chunks": 4},
    "seq2": {"chunk_axis": "sequence", "chunks": 2},
    "seq4": {"chunk_axis": "sequence", "chunks": 4},
    "pruned": {"prune_documents": True},
    "pruned_head2": {"prune_documents": True, "chunks": 2},
    "pruned_seq2": {"prune_documents": True, "chunk_axis": "sequence", "chunks": 2},
}


def _make_inputs(local, rank, dtype=torch.bfloat16, heads=40, kv_heads=8, dimension=128):
    """Generate distinct local Q/K/V/dO reproducibly for independent gradient checks."""
    generator = torch.Generator().manual_seed(20261108 + rank)
    return tuple(torch.randn(local, count, dimension, generator=generator).to(device="npu", dtype=dtype)
                 .requires_grad_(index < 3) for index, count in enumerate((heads, kv_heads, kv_heads, heads)))


def _attention(inputs, mesh, lengths, variant, causal=True, trace=None):
    if variant in ("baseline", "sync", "pruned_core", "pruned_core_sync"):
        return ring_attention(*inputs[:3], mesh, input_layout="TND", document_lengths=lengths,
                              causal=causal, overlap=variant in ("baseline", "pruned_core"),
                              prune_documents=variant.startswith("pruned_core"), trace=trace)
    return chunked_ring_attention(*inputs[:3], mesh, document_lengths=lengths, causal=causal,
                                  trace=trace, **VARIANTS[variant])


def _iteration(inputs, mesh, lengths, variant, causal=True, trace=None):
    """Run a complete local-output/global-KV-gradient attention step."""
    output = _attention(inputs, mesh, lengths, variant, causal, trace)
    return (output, *torch.autograd.grad(output, inputs[:3], inputs[3]))


def _errors(actual, expected):
    """Report finite values and norm/elementwise error without hiding diagnostic outliers."""
    result = {}
    for name, observed, reference in zip(("O", "dQ", "dK", "dV"), actual, expected):
        a, b = observed.detach().float(), reference.detach().float()
        diff = (a - b).abs()
        item = {'relative_l2': float((a - b).norm() / b.norm().clamp_min(1e-12)),
                    'max_abs': float(diff.max()), 'finite': bool(torch.isfinite(a).all()),
                    'historical_outliers': int((diff > 0.015625 + 0.06 * b.abs()).sum())}
        if not item['finite'] or item['relative_l2'] > 0.015:
            raise AssertionError(f"{name}: actual error={item}, expected finite relative_l2 <= 0.015")
        result[name] = item
    return result


def _run_checks(mesh, quick=False):
    """Check FP32 oracle parity, local-document zero traffic, and stream-safe reuse."""
    local, rank = 64, mesh.get_local_rank()
    total = local * mesh.size()
    inputs = _make_inputs(local, rank)
    cases = [(total,), (3, 61, total - 64)] if mesh.size() > 1 else [(64,), (3, 17, 44)]
    cases.append((local,) * mesh.size())
    records = []
    for lengths in cases:
        expected = _full_reference(inputs, mesh, lengths, "TND", "fp32")
        for variant in VARIANTS:
            actual = _iteration(inputs, mesh, lengths, variant)
            records.append({'lengths': lengths, 'variant': variant, 'errors': _errors(actual, expected)})
    if not quick:
        for dtype in (torch.float16, torch.bfloat16):
            inputs = _make_inputs(65, rank, dtype)
            lengths = (7, 65 * mesh.size() - 7)
            for causal in (False, True):
                expected = _iteration(inputs, mesh, lengths, "baseline", causal)
                for variant in VARIANTS:
                    records.append({'dtype': str(dtype), 'causal': causal, 'variant': variant,
                                    'errors': _errors(_iteration(inputs, mesh, lengths, variant, causal), expected)})
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for variant in ("head2", "seq4", "pruned", "pruned_core", "pruned_core_sync"):
                for _ in range(3):
                    _errors(_iteration(inputs, mesh, lengths, variant), _iteration(inputs, mesh, lengths, "baseline"))
        torch.npu.current_stream().wait_stream(stream)
        for _ in range(2):
            _iteration(inputs, mesh, lengths, "pruned_core")
        torch.npu.synchronize()
        before = torch.npu.memory_allocated()
        for _ in range(6):
            _iteration(inputs, mesh, lengths, "pruned_core")
        torch.npu.synchronize()
        growth = torch.npu.memory_allocated() - before
        assert growth == 0, f"live memory growth actual={growth}, expected=0"
    return records


@pytest.fixture(name="cp_mesh", scope="module")
def _cp_mesh():
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.npu.is_available() or torch.npu.device_count() < size:
        pytest.skip(f"chunked Ring requires {size} NPU devices")
    torch_npu.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        yield init_device_mesh("npu", (size,), mesh_dim_names=("cp",))
    finally:
        dist.destroy_process_group()


class TestRingChunk:
    """Validate the experimental schedule against independent attention references."""

    def test_numerics_and_lifetimes(self, cp_mesh) -> None:
        """Cover causal/noncausal GQA, uneven cuts, skipped traffic and side streams.

        Args:
            cp_mesh: Initialized CP mesh fixture.
        """
        _run_checks(cp_mesh)

    @pytest.mark.parametrize("needed", [(True, True, True), (True, False, False), (False, True, True)])
    @pytest.mark.parametrize("variant", ["head2", "seq4", "pruned", "pruned_core", "pruned_core_sync"])
    def test_strides_and_owner_gradient(self, cp_mesh, needed, variant) -> None:
        """Frozen projections, strided inputs and a sole active query owner preserve gradients.

        Args:
            cp_mesh: Initialized CP mesh fixture.
            needed: Requires-grad flags for Q, K and V.
            variant: Ring transport or chunking variant.
        """
        inputs = _make_inputs(130, cp_mesh.get_local_rank())
        inputs = tuple(tensor[::2].detach().requires_grad_(needed[index] if index < 3 else False)
                       for index, tensor in enumerate(inputs))
        lengths = (7, 65 * cp_mesh.size() - 7)
        if cp_mesh.get_local_rank() != cp_mesh.size() - 1:
            inputs[3].zero_()
        selected = tuple(tensor for tensor in inputs[:3] if tensor.requires_grad)
        expected = ring_attention(*inputs[:3], cp_mesh, input_layout="TND", document_lengths=lengths,
                                  causal=True, overlap=True)
        reference = (expected, *torch.autograd.grad(expected, selected, inputs[3]))
        output = _attention(inputs, cp_mesh, lengths, variant)
        actual = (output, *torch.autograd.grad(output, selected, inputs[3]))
        _errors(actual, reference)

    @pytest.mark.parametrize("heads,kv_heads,variant", [(8, 8, "head4"), (8, 8, "pruned"),
                                                       (8, 1, "seq4"), (8, 1, "pruned"),
                                                       (8, 8, "pruned_core"), (8, 1, "pruned_core")])
    def test_mha_and_mqa(self, cp_mesh, heads, kv_heads, variant) -> None:
        """The same transport plan preserves MHA and unsplit MQA head groups.

        Args:
            cp_mesh: Initialized CP mesh fixture.
            heads: Number of query heads.
            kv_heads: Number of compact KV heads.
            variant: Ring transport or chunking variant.
        """
        inputs = _make_inputs(33, cp_mesh.get_local_rank(), heads=heads, kv_heads=kv_heads)
        lengths = (11, 33 * cp_mesh.size() - 11)
        expected = _full_reference(inputs, cp_mesh, lengths, "TND", "fp32")
        _errors(_iteration(inputs, cp_mesh, lengths, variant), expected)

    def test_strided_process_groups(self, cp_mesh) -> None:
        """Use subgroup-local ownership when global neighbor ranks are not consecutive.

        Args:
            cp_mesh: World CP mesh used to construct noncontiguous subgroups.
        """
        if cp_mesh.size() < 4:
            pytest.skip("strided two-rank subgroups require at least four ranks")
        mesh = init_device_mesh("npu", (2, cp_mesh.size() // 2), mesh_dim_names=("cp", "replica"))["cp"]
        inputs = _make_inputs(33, dist.get_rank())
        lengths = (11, 55)
        expected = _full_reference(inputs, mesh, lengths, "TND", "fp32")
        for variant in ("head2", "seq4", "pruned", "pruned_core", "pruned_core_sync"):
            _errors(_iteration(inputs, mesh, lengths, variant), expected)

    def test_no_remote_consumers_and_round_trace(self, cp_mesh) -> None:
        """All-local documents need zero sends/receives even with backward and round tracing.

        Args:
            cp_mesh: Initialized CP mesh fixture.
        """
        inputs = _make_inputs(33, cp_mesh.get_local_rank())
        trace = RingAttentionTrace()
        lengths = (33,) * cp_mesh.size()
        expected = _iteration(inputs, cp_mesh, lengths, "baseline")
        with patch.object(dist, "isend", wraps=dist.isend) as send, \
                patch.object(dist, "irecv", wraps=dist.irecv) as recv:
            actual = _iteration(inputs, cp_mesh, lengths, "pruned_core", trace=trace)
        send.assert_not_called()
        recv.assert_not_called()
        _errors(actual, expected)
        assert len(trace.timings()) == 2, (
            f"trace rounds actual={len(trace.events)}, expected=2")
        _errors(_iteration(inputs, None, (7, 26), "pruned_core"),
                _iteration(inputs, None, (7, 26), "baseline"))


def main() -> None:
    """Initialize one CP group and write rank-wise diagnostics only when requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("cp",))
        result = _run_checks(mesh, args.quick)
        gathered = [None] * mesh.size()
        dist.all_gather_object(gathered, result)
        if mesh.get_local_rank() == 0:
            if args.output:
                Path(args.output).write_text(json.dumps(gathered, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({'passed': True, 'ranks': mesh.size(), 'checks_per_rank': len(result)}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
