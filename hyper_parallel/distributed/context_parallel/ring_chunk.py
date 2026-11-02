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
"""Experimental chunk pipelines for contiguous TND Ring, with explicit stream ownership.

The existing Ring API remains the baseline. This module reuses its native FA and
FP32 online merge/gradient kernels, changing only block plans and P2P scheduling.
Raw P2P is enclosed by a custom autograd function with an explicit reverse ring.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import accumulate
from typing import Any

# Native Torch is required; the legacy lint rule predates the Torch-only migration.
import torch  # pylint: disable=forbidden-backend-import
from torch.autograd.function import once_differentiable  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.context_parallel.ring_attention import (
    _RingContext, _backward_blocks, _empty_state, _forward_blocks, _ring_context, _validate_inputs,
)
from hyper_parallel.distributed.context_parallel.ring_layout import AttentionBlock, _block_batches, _document_hops
from hyper_parallel.distributed.context_parallel.ring_transport import (
    _PackedRingExchange as _ChunkExchange, _communication_stream, _pack_ring_kv, _return_ring_gradient,
)


@dataclass(frozen=True)
class ChunkPlan:
    """One head/token slice, its transport rows and exact per-owner attention blocks."""

    query_heads: tuple[int, int]
    key_heads: tuple[int, int]
    token_range: tuple[int, int]
    spans: tuple[tuple[tuple[int, int], ...], ...]
    blocks: tuple[tuple[AttentionBlock, ...], ...]
    batches: tuple
    local_blocks: tuple[AttentionBlock, ...]
    local_batches: tuple
    counts: tuple[int, ...]
    capacity: int
    hops: tuple[int, ...]


def _rectangles(query_range, key_spans, documents, causal):
    result, offset = [], 0
    for start, end in key_spans:
        for doc_start, doc_end in documents:
            qstart, qend = max(query_range[0], doc_start), min(query_range[1], doc_end)
            kstart, kend = max(start, doc_start), min(end, doc_end)
            if causal:
                qstart = max(qstart, kstart)
            if qstart >= qend or kstart >= kend:
                continue
            key_start = offset + kstart - start
            if not causal:
                result.append(AttentionBlock(qstart - query_range[0], qend - query_range[0],
                                             key_start, key_start + kend - kstart))
                continue
            diagonal_end = min(qend, kend)
            if qstart < diagonal_end:
                result.append(AttentionBlock(qstart - query_range[0], diagonal_end - query_range[0],
                                             key_start, key_start + diagonal_end - kstart, True))
            tail_start = max(qstart, kend)
            if tail_start < qend:
                result.append(AttentionBlock(tail_start - query_range[0], qend - query_range[0],
                                             key_start, key_start + kend - kstart))
        offset += end - start
    return tuple(result)


def _remote_spans(documents, owner, local, bounds, causal, prune):
    begin, end = owner * local + bounds[0], owner * local + bounds[1]
    if not prune:
        return ((begin, end),)
    spans = []
    for doc_start, doc_end in documents:
        start, stop = max(begin, doc_start), min(end, doc_end)
        remote = doc_end > (owner + 1) * local or (not causal and doc_start < owner * local)
        if start < stop and remote:
            spans.append((start, stop))
    return tuple(spans)


def _positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_plan(lengths, size, rank, query_heads, key_heads, axis, chunks, causal, prune):
    if not all(_positive_integer(n) for n in (size, query_heads, key_heads, chunks)):
        raise ValueError("size, head counts and chunks must be positive integers")
    if not isinstance(lengths, tuple) or not lengths or not all(_positive_integer(n) for n in lengths):
        raise ValueError("document lengths must be a nonempty tuple of positive integers")
    if sum(lengths) % size or query_heads % key_heads:
        raise ValueError("document tokens must divide CP and query heads must divide into KV groups")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < size:
        raise ValueError("rank must belong to the CP group")
    _validate_partition(axis, chunks, causal, prune, key_heads, sum(lengths) // size)


def _validate_partition(axis, chunks, causal, prune, key_heads, local):
    if axis not in ("head", "sequence") or not isinstance(causal, bool) or not isinstance(prune, bool):
        raise ValueError("axis must be head/sequence; causal and prune must be bool")
    if (axis == "head" and key_heads % chunks) or (axis == "sequence" and chunks > local):
        raise ValueError("head chunks must divide KV heads and sequence chunks must be nonempty")


@lru_cache(maxsize=128)
def chunk_plans(lengths: tuple[int, ...], size: int, rank: int, query_heads: int, key_heads: int,
                axis: str = "head", chunks: int = 1, causal: bool = True,
                prune: bool = False) -> tuple[ChunkPlan, ...]:
    """Partition heads or KV rows while preserving document-local causal coordinates.

    Args:
        lengths: Positive global document lengths in packed order.
        size: CP group size; global tokens must divide it.
        rank: Local rank in the CP group.
        query_heads: Number of local query heads.
        key_heads: Number of compact KV heads, dividing query_heads.
        axis: Either head (whole GQA groups) or sequence (KV token intervals).
        chunks: Head divisor or number of nonempty KV token intervals.
        causal: Whether keys after the original query position are masked.
        prune: Send only document intervals needed by a different rank.

    Returns:
        Immutable chunk plans with compact transport coordinates and local blocks.
    """
    _validate_plan(lengths, size, rank, query_heads, key_heads, axis, chunks, causal, prune)
    local = sum(lengths) // size
    boundaries = (0, *accumulate(lengths))
    documents = tuple(zip(boundaries, boundaries[1:]))
    plans = []
    for chunk in range(chunks):
        qheads = (chunk * query_heads // chunks, (chunk + 1) * query_heads // chunks)
        kheads = (chunk * key_heads // chunks, (chunk + 1) * key_heads // chunks)
        bounds = (0, local)
        if axis == "sequence":
            qheads, kheads = (0, query_heads), (0, key_heads)
            bounds = (chunk * local // chunks, (chunk + 1) * local // chunks)
        spans = tuple(_remote_spans(documents, owner, local, bounds, causal, prune) for owner in range(size))
        counts = tuple(sum(end - start for start, end in parts) for parts in spans)
        capacity = max(1, *counts)
        query_range = (rank * local, (rank + 1) * local)
        blocks = tuple(_rectangles(query_range, parts, documents, causal) for parts in spans)
        local_spans = ((rank * local + bounds[0], rank * local + bounds[1]),)
        local_blocks = _rectangles(query_range, local_spans, documents, causal)
        plans.append(ChunkPlan(qheads, kheads, bounds, spans, blocks,
                               tuple(_block_batches(parts, capacity) for parts in blocks),
                               local_blocks, _block_batches(local_blocks, local), counts, capacity,
                               _document_hops(spans, lengths, size, causal) if prune else (size - 1,) * size))
    return tuple(plans)


@dataclass
class ChunkTrace:
    """Opt-in events on the consumer stream; wait spans measure exposed dependencies."""

    events: list = field(default_factory=list)

    @contextmanager
    def measure(self, phase: str, step: int, chunk: int, stage: str) -> Any:
        """Record an event pair without synchronizing inside attention.

        Args:
            phase: Forward or backward.
            step: Ring hop.
            chunk: Chunk index.
            stage: Pack, receive wait, kernel/merge, gradient wait, or accumulation.

        Yields:
            None while the measured work is enqueued.
        """
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        yield
        end.record()
        self.events.append((phase, step, chunk, stage, begin, end))

    def timings(self) -> list[dict]:
        """Synchronize completed events and return per-stage device milliseconds."""
        result = []
        for phase, step, chunk, stage, begin, end in self.events:
            end.synchronize()
            result.append({'phase': phase, 'step': step, 'chunk': chunk, 'stage': stage,
                               'milliseconds': begin.elapsed_time(end)})
        return result


@contextmanager
def _measure(trace, phase, step, chunk, stage):
    if trace is None:
        yield
    else:
        with trace.measure(phase, step, chunk, stage):
            yield


def _head_view(tensor, bounds):
    return tensor[:, bounds[0]:bounds[1]]


def _pack(tensors, plan, ring, out=None):
    return _pack_ring_kv(tuple(_head_view(tensor, plan.key_heads) for tensor in tensors),
                         plan.spans, ring, plan.counts, out=out)


def _local_kv(tensors, plan):
    return tuple(_head_view(tensor, plan.key_heads)[slice(*plan.token_range)] for tensor in tensors)


def _forward(query, key, value, ring, plans, scale, trace):
    state = _empty_state(query, "TND")
    stream = _communication_stream(query.device, False)
    with _measure(trace, "forward", -1, -1, "pack"):
        initial = [_pack((key, value), plan, ring) for plan in plans]
        exchanges = [_ChunkExchange(_head_view(key, plan.key_heads), plan, ring, stream) for plan in plans]
    packets = initial
    last_step = max(max(plan.hops) for plan in plans)
    for step in range(last_step + 1):
        next_packets = [exchange.launch(packet, step) for exchange, packet in zip(exchanges, packets)] \
            if step < last_step else []
        for chunk, (plan, packet) in enumerate(zip(plans, packets)):
            blocks = plan.local_blocks if step == 0 else plan.blocks[packet.owner]
            if not blocks:
                continue
            with _measure(trace, "forward", step, chunk, "receive_wait"):
                packet.ready()
            keys, values = _local_kv((key, value), plan) if step == 0 else packet.tensors()
            batches = plan.local_batches if step == 0 else plan.batches[packet.owner]
            with _measure(trace, "forward", step, chunk, "fa_merge"):
                _forward_blocks(_head_view(query, plan.query_heads), keys, values,
                                tuple(_head_view(tensor, plan.query_heads) for tensor in state),
                                blocks, scale, "TND", batches)
            packet.consumed_now()
        packets = next_packets
    for exchange, packet in zip(exchanges, initial):
        exchange.finish(packet)
    return state[0].to(query.dtype), state[1], state[2]


def _accumulate_local(target, delta, plan):
    _head_view(target, plan.key_heads)[slice(*plan.token_range)].add_(delta)


def _replace_owned(target, source, plan, ring):
    offset = 0
    for start, end in plan.spans[ring.rank]:
        length = end - start
        _head_view(target, plan.key_heads)[start - ring.rank * target.shape[0]:end - ring.rank * target.shape[0]].copy_(
            source[offset:offset + length])
        offset += length


def _backward_chunk(saved, dout, accumulators, ring, plan, packet, gradient, scale, trace, step, chunk):
    query, key, value = saved[:3]
    dq, dk, dv = accumulators
    blocks = plan.local_blocks if step == 0 else plan.blocks[packet.owner]
    if not blocks:
        return
    with _measure(trace, "backward", step, chunk, "receive_wait"):
        packet.ready()
    keys, values = _local_kv((key, value), plan) if step == 0 else packet.tensors()
    batches = plan.local_batches if step == 0 else plan.batches[packet.owner]
    subset = tuple(_head_view(tensor, plan.query_heads) for tensor in (saved[0], *saved[3:]))
    with _measure(trace, "backward", step, chunk, "fa_prepare"):
        delta_key, delta_value = _backward_blocks(
            (subset[0], keys, values, *subset[1:]), _head_view(dout, plan.query_heads), keys, values,
            _head_view(dq, plan.query_heads), blocks, scale, "TND", batches)
    packet.consumed_now()
    with _measure(trace, "backward", step, chunk, "gradient_wait"):
        gradient.ready()
    with _measure(trace, "backward", step, chunk, "gradient_add"):
        if step == 0:
            _accumulate_local(dk, delta_key, plan)
            _accumulate_local(dv, delta_value, plan)
            _pack((dk, dv), plan, ring, out=gradient)
        else:
            for target, delta in zip(gradient.tensors(), (delta_key, delta_value)):
                target.add_(delta)
            gradient.produced_now()


def _return_gradients(owned, gradients, plans, ring, exchanges, initials, trace):
    dk, dv = owned
    with _measure(trace, "backward", ring.size, -1, "owner_return"):
        for plan, gradient in zip(plans, gradients):
            gradient.ready()
            for target, source in zip((dk, dv), gradient.tensors()):
                _replace_owned(target, source, plan, ring)
            gradient.consumed_now()
        for exchange, packet in zip(exchanges, initials):
            exchange.finish(packet)


def _backward(saved, dout, ring, plans, scale, trace):
    query, key, value = saved[:3]
    dq, dk, dv = (torch.zeros_like(tensor, dtype=torch.float32) for tensor in (query, key, value))
    kv_stream, grad_stream = (_communication_stream(query.device, gradient) for gradient in (False, True))
    with _measure(trace, "backward", -1, -1, "pack"):
        initial = [_pack((key, value), plan, ring) for plan in plans]
        exchanges = [_ChunkExchange(_head_view(key, plan.key_heads), plan, ring, kv_stream) for plan in plans]
        grad_initial = [_pack((dk, dv), plan, ring) for plan in plans]
        grad_exchanges = [_ChunkExchange(_head_view(dk, plan.key_heads), plan, ring, grad_stream) for plan in plans]
    packets, gradients = initial, grad_initial
    last_step = max(max(plan.hops) for plan in plans)
    for step in range(last_step + 1):
        next_packets = [exchange.launch(packet, step) for exchange, packet in zip(exchanges, packets)] \
            if step < last_step else []
        next_gradients = []
        for chunk, (plan, packet, gradient, exchange) in enumerate(zip(plans, packets, gradients, grad_exchanges)):
            _backward_chunk(saved, dout, (dq, dk, dv), ring, plan, packet, gradient, scale, trace, step, chunk)
            next_gradients.append(exchange.launch(gradient, step) if step < last_step else None)
            _return_ring_gradient(gradient, grad_initial[chunk], plan, ring, grad_stream, step)
        packets, gradients = next_packets, next_gradients
    _return_gradients((dk, dv), grad_initial, plans, ring, (*exchanges, *grad_exchanges),
                      (*initial, *grad_initial), trace)
    return tuple(gradient.to(tensor.dtype) for gradient, tensor in zip((dq, dk, dv), (query, key, value)))


class _ChunkedRing(torch.autograd.Function):
    """Recompute each native FA block using the final softmax statistics in backward."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, ring: _RingContext,
                plans: tuple[ChunkPlan, ...], scale: float, trace: ChunkTrace | None) -> torch.Tensor:
        """Save final online softmax statistics for native block recomputation.

        Args:
            ctx: Autograd context for local tensors and chunk plans.
            query: Local TND query tensor.
            key: Local TND compact-head key tensor.
            value: Local value tensor matching key.
            ring: Ordered CP process group and neighbor metadata.
            plans: Per-chunk head, token and transport plans.
            scale: Attention score multiplier.
            trace: Optional chunk-stage timing collector.
        """
        output, maximum, denominator = _forward(query, key, value, ring, plans, scale, trace)
        ctx.save_for_backward(query, key, value, output, maximum, denominator)
        ctx.ring, ctx.plans, ctx.scale, ctx.trace = ring, plans, scale, trace
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, dout: torch.Tensor) -> tuple:
        """Accumulate dQ locally and return each dKV buffer to its original owner.

        Args:
            ctx: Context containing the final attention statistics and plans.
            dout: Gradient of the local attention output.
        """
        gradients = _backward(ctx.saved_tensors, dout.contiguous(), ctx.ring, ctx.plans, ctx.scale, ctx.trace)
        return (*gradients, None, None, None, None)


def chunked_ring_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, cp_mesh: Any = None,
                           *, document_lengths: tuple[int, ...], causal: bool = True, scale: float | None = None,
                           chunk_axis: str = "head", chunks: int = 1, prune_documents: bool = False,
                           trace: ChunkTrace | None = None) -> torch.Tensor:
    """Run an opt-in NPU TND pipeline with unchanged contiguous source ownership.

    Args:
        query: Local TND query tensor, FP16 or BF16.
        key: Local compact GQA/MHA keys with the same token count.
        value: Values matching key shape and dtype.
        cp_mesh: One-dimensional CP mesh in process-group rank order.
        document_lengths: Positive global document lengths, identical on every rank.
        causal: Apply document-local causal attention.
        scale: Positive attention scale; defaults to inverse square-root head dimension.
        chunk_axis: Split complete KV head groups or KV sequence intervals.
        chunks: Number of head groups or sequence intervals.
        prune_documents: Omit transport rows never used by another rank; local FA still covers all rows.
        trace: Optional exposed-wait and compute device events, outside benchmark throughput runs.

    Returns:
        Output in original local TND order. All ranks must participate in both forward and backward.
    """
    _validate_inputs(query, key, value, causal, scale, "TND", 0.0, True)
    ring = _ring_context(cp_mesh)
    plans = chunk_plans(document_lengths, ring.size, ring.rank, query.shape[1], key.shape[1],
                        chunk_axis, chunks, causal, prune_documents)
    if sum(document_lengths) != query.shape[0] * ring.size:
        raise ValueError("document lengths must match the equal local source shards")
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    return _ChunkedRing.apply(query, key, value, ring, plans, scale, trace)
