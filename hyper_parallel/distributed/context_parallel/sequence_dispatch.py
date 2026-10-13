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
"""Differentiable all-to-all-v dispatch of contiguous sequence bank intervals."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import


@dataclass(frozen=True)
class SequenceDispatchRoute:
    """Row splits for a compact interval assembled from contiguous owner shards.

    Self rows stay local. Remote rows are ordered by owner rank, so the local
    shard fits between the received prefix and suffix without tensor indices.
    Backward reverses the exchange and SUMs overlapping consumer contributions.
    """

    local_length: int
    global_start: int
    global_end: int
    local_output_start: int
    send_offsets: tuple[int, ...]
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    needs_exchange: bool


def interval_route(
        local_length: int, rank: int, intervals: tuple[tuple[int, int], ...],
) -> SequenceDispatchRoute:
    """Derive owner/consumer intersections; all ranks agree whether to exchange."""
    local_start = rank * local_length
    global_start, global_end = intervals[rank]
    offsets, sends, receives = [], [], []
    for peer, (start, end) in enumerate(intervals):
        first = max(local_start, start)
        count = max(0, min(local_start + local_length, end) - first) if peer != rank else 0
        offsets.append(first - local_start if count else 0)
        sends.append(count)
        peer_start = peer * local_length
        receives.append(max(0, min(peer_start + local_length, global_end) - max(peer_start, global_start))
                        if peer != rank else 0)
    needs_exchange = any(start != peer * local_length or end != (peer + 1) * local_length
                         for peer, (start, end) in enumerate(intervals))
    return SequenceDispatchRoute(local_length, global_start, global_end, local_start - global_start,
                                 tuple(offsets), tuple(sends), tuple(receives), needs_exchange)


class _SequenceDispatchWait(torch.autograd.Function):
    """Attach remote bank rows to their owners without retaining send buffers."""

    @staticmethod
    def forward(ctx: Any, local: torch.Tensor, received: torch.Tensor, work: Any,
                route: SequenceDispatchRoute, group: Any) -> torch.Tensor:
        """Wait before exposing received data and insert the uncommunicated local rows."""
        ctx.route = route
        ctx.group = group
        work.wait()
        before = route.local_output_start
        return torch.cat((received[:before], local, received[before:]), dim=0)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple:
        """Reverse dispatch, SUMming consumers in rank order, including empty ranks."""
        route = ctx.route
        first, last = route.local_output_start, route.local_output_start + route.local_length
        send = torch.cat((grad[:first], grad[last:]), dim=0)
        received = grad.new_empty((sum(route.input_splits), *grad.shape[1:]))
        work = dist.all_to_all_single(
            received, send, output_split_sizes=list(route.input_splits),
            input_split_sizes=list(route.output_splits), group=ctx.group, async_op=True,
        )
        local = grad[first:last].contiguous().clone()
        work.wait()
        cursor = 0
        for offset, count in zip(route.send_offsets, route.input_splits):
            if count:
                local.narrow(0, offset, count).add_(received.narrow(0, cursor, count))
                cursor += count
        return local, None, None, None, None


@dataclass
class AsyncSequenceDispatch:
    """Single-use launch handle owning send storage until the communication completes."""

    local: torch.Tensor
    send: torch.Tensor | None
    received: torch.Tensor | None
    work: Any
    route: SequenceDispatchRoute
    group: Any
    sequence_dim: int
    waited: bool = False

    def wait(self) -> torch.Tensor:
        """Finish the exchange and return the compact bank in its original layout."""
        if self.waited:
            raise RuntimeError("a sequence dispatch handle cannot be waited twice")
        self.waited = True
        output = self.local
        if self.received is not None:
            output = _SequenceDispatchWait.apply(self.local, self.received, self.work, self.route, self.group)
        self.send = None
        self.received = None
        return output.movedim(0, self.sequence_dim).contiguous()


def async_sequence_dispatch(tensor: torch.Tensor, sequence_dim: int,
                            route: SequenceDispatchRoute, group: Any) -> AsyncSequenceDispatch:
    """Launch variable row exchange with an autograd edge attached at wait.

    Args:
        tensor: Contiguous local owner shard of a raw, compressed or Index K bank.
        sequence_dim: Sequence dimension; feature dimensions are transported unchanged.
        route: Immutable per-batch routing agreed by every member of the CP group.
        group: Existing framework-owned CP process group, in sequence-shard rank order.
    """
    if not -tensor.ndim <= sequence_dim < tensor.ndim:
        raise ValueError("sequence dispatch dimension is outside the input tensor dimensions")
    sequence_dim %= tensor.ndim
    if tensor.shape[sequence_dim] != route.local_length:
        raise ValueError("sequence dispatch owner length does not match the bank's physical layout")
    local = tensor.movedim(sequence_dim, 0).contiguous()
    if not route.needs_exchange:
        return AsyncSequenceDispatch(local, None, None, None, route, group, sequence_dim)
    parts = [local.detach().narrow(0, offset, count)
             for offset, count in zip(route.send_offsets, route.input_splits) if count]
    send = torch.cat(parts, dim=0) if parts else local.detach()[:0]
    received = local.new_empty((sum(route.output_splits), *local.shape[1:]))
    work = dist.all_to_all_single(
        received, send, output_split_sizes=list(route.output_splits),
        input_split_sizes=list(route.input_splits), group=group, async_op=True,
    )
    return AsyncSequenceDispatch(local, send, received, work, route, group, sequence_dim)


@lru_cache(maxsize=128)
def sliding_window_route(boundaries: tuple[int, ...], local_length: int, size: int,
                         rank: int, window: int) -> SequenceDispatchRoute:
    """Plan the same-document left halo; every rank agrees whether to exchange."""
    if local_length <= 0 or size <= 0 or not 0 <= rank < size or window <= 0:
        raise ValueError("SWA dispatch requires positive dimensions and a valid CP rank")
    if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != local_length * size
            or any(end <= start for start, end in zip(boundaries, boundaries[1:]))):
        raise ValueError("SWA document boundaries must span the full contiguous CP sequence")
    intervals = tuple(
        (max(boundaries[bisect_right(boundaries, peer * local_length) - 1], peer * local_length - window + 1),
         (peer + 1) * local_length)
        for peer in range(size)
    )
    return interval_route(local_length, rank, intervals)


@lru_cache(maxsize=128)
def document_prefix_route(boundaries: tuple[int, ...], local_length: int, size: int,
                          rank: int, ratio: int) -> SequenceDispatchRoute:
    """Dispatch complete compressed prefixes for contiguous document-local queries.

    Only the first intersecting document can need remote preceding blocks.
    Later documents begin within the local shard, so the union is contiguous.
    Compression retains its existing aligned physical layout.
    """
    if ratio < 1 or local_length <= 0 or size <= 0 or not 0 <= rank < size:
        raise ValueError("Document prefix dispatch requires positive lengths, size and compression ratio")
    if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != local_length * size
            or any(end <= start for start, end in zip(boundaries, boundaries[1:]))
            or local_length % ratio or any(boundary % ratio for boundary in boundaries)):
        raise ValueError("Document prefix dispatch requires compression-aligned document and CP boundaries")
    intervals = tuple(
        (boundaries[bisect_right(boundaries, peer * local_length) - 1] // ratio,
         (peer + 1) * local_length // ratio)
        for peer in range(size)
    )
    return interval_route(local_length // ratio, rank, intervals)
