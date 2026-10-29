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
"""Ordered cached tree with native Torch P2P and invocation-owned transfers.

The balanced postorder schedule and matmul/add order match the original research
control. Forward sends packed S/M upward; backward communicates gradients only.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TreeEdge:
    """One local up/down action, with the right endpoint owning the parent."""

    peer: int
    owner: bool


def ordered_tree_edges(rank: int, size: int) -> tuple[TreeEdge, ...]:
    """Build a postorder communication schedule, including non-power-of-two CP."""
    if size < 1 or not 0 <= rank < size:
        raise ValueError(f"Invalid ordered-tree rank {rank} for size {size}.")
    edges = []

    def _visit(start: int, end: int) -> None:
        if end - start == 1:
            return
        middle = (start + end) // 2
        # Only the branch containing this rank can contribute local actions.
        if rank < middle:
            _visit(start, middle)
        else:
            _visit(middle, end)
        if rank == middle - 1:
            edges.append(TreeEdge(end - 1, False))
        elif rank == end - 1:
            edges.append(TreeEdge(middle - 1, True))

    _visit(0, size)
    return tuple(edges)


class CachedOrderedTree:
    """Bidirectional affine scan that communicates no transfer matrices in backward.

    Forward composes ``M = M_right M_left``. Saved child matrices give
    ``G = G_left + M_left.T G_right`` in backward. The schedule and process
    group are reusable; every invocation owns a separate tensor cache.
    """

    def __init__(self, group: dist.ProcessGroup, rank: int, ranks: tuple[int, ...]) -> None:
        """Bind the chronological process-group order without creating a group."""
        if len(set(ranks)) != len(ranks) or tuple(dist.get_process_group_ranks(group)) != tuple(ranks):
            raise ValueError("Ordered-tree sequence order must match its process group order")
        if not 0 <= rank < len(ranks) or ranks[rank] != dist.get_rank():
            raise ValueError("Ordered-tree local rank must identify the calling process")
        self.group = group
        self.rank = rank
        self.ranks = tuple(ranks)
        self.edges = ordered_tree_edges(rank, len(ranks))
        self.cache_size = 2 * sum(edge.owner for edge in self.edges)

    def _send(self, tensor: torch.Tensor, peer: int) -> None:
        packet = tensor.contiguous()
        dist.isend(packet, dst=self.ranks[peer], group=self.group).wait()

    def _receive(self, like: torch.Tensor, peer: int, shape: tuple | None = None) -> torch.Tensor:
        packet = torch.empty_like(like) if shape is None else like.new_empty(shape)
        dist.irecv(packet, src=self.ranks[peer], group=self.group).wait()
        return packet

    def forward(
        self, state: torch.Tensor, transition: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return the exclusive prefix state and child transfers for backward.

        Args:
            state: Local affine offset [B,H,K,V].
            transition: Local transfer [B,H,K,K].
            initial_state: Global initial boundary, supplied on the root owner.
        """
        if (state.ndim != 4 or state.shape != transition.shape or state.shape[-2:] != (128, 128)
                or state.dtype != torch.float32 or transition.dtype != torch.float32
                or state.device != transition.device or not state.is_contiguous() or not transition.is_contiguous()):
            raise ValueError("Cached tree requires matching contiguous FP32 B/H/K128/V128 summaries")
        value_dim = state.shape[-1]
        aggregate = torch.cat((state, transition), dim=-1)
        left_offsets = []
        cache = []
        for edge in self.edges:
            if edge.owner:
                left = self._receive(aggregate, edge.peer)
                right_transfer = aggregate[..., value_dim:].contiguous()
                left_transfer = left[..., value_dim:].contiguous()
                cache.extend((left_transfer, right_transfer))
                left_offsets.append(left[..., :value_dim].contiguous())
                composed = right_transfer @ left
                composed[..., :value_dim].add_(aggregate[..., :value_dim])
                aggregate = composed
            else:
                self._send(aggregate, edge.peer)
        boundary = torch.zeros_like(state) if initial_state is None else initial_state
        owner_index = len(left_offsets)
        for edge in reversed(self.edges):
            if edge.owner:
                owner_index -= 1
                right_boundary = cache[2 * owner_index] @ boundary + left_offsets[owner_index]
                self._send(boundary, edge.peer)
                boundary = right_boundary
            else:
                boundary = self._receive(state, edge.peer)
        return boundary, tuple(cache)

    def backward(
        self, gradient: torch.Tensor, cache: tuple[torch.Tensor, ...],
        final_gradient: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the exclusive suffix adjoint using only cached child transfers."""
        if len(cache) != self.cache_size:
            raise ValueError("Ordered-tree backward received an incompatible forward cache.")
        aggregate = gradient
        right_gradients = []
        owner_index = 0
        for edge in self.edges:
            if edge.owner:
                left_gradient = self._receive(gradient, edge.peer)
                right_gradients.append(aggregate)
                aggregate = left_gradient + cache[2 * owner_index].transpose(-1, -2) @ aggregate
                owner_index += 1
            else:
                self._send(aggregate, edge.peer)
        boundary = torch.zeros_like(gradient) if final_gradient is None else final_gradient
        for edge in reversed(self.edges):
            if edge.owner:
                owner_index -= 1
                left_boundary = (
                    right_gradients[owner_index]
                    + cache[2 * owner_index + 1].transpose(-1, -2) @ boundary
                )
                self._send(left_boundary, edge.peer)
            else:
                boundary = self._receive(gradient, edge.peer)
        return boundary
