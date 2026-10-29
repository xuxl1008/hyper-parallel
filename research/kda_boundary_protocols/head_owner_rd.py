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
"""Research summary-head ownership with cached RD between chronological groups."""
from __future__ import annotations

import torch
import torch.distributed as dist

from hyper_parallel.distributed.context_parallel.kimi_delta_attention_rd import KDAStateRecursiveDoubling


class KDAStateHeadOwnerRD:
    """Exchange summaries by head while all token tensors remain on their rank.

    This executor is called inside the existing custom KDA autograd function;
    its explicit backward supplies the state adjoint. Raw collectives therefore
    do not bypass an expected torch autograd graph. All groups are supplied
    explicitly; protocol calls never create groups or move token tensors.
    """

    requires_contiguous_summaries = True

    def __init__(
        self, group: dist.ProcessGroup, inner_ranks: tuple[int, ...], outer: KDAStateRecursiveDoubling,
    ) -> None:
        """Bind precreated chronological owner groups and one outer head lane.

        Args:
            group: Inner process group ordered by consecutive sequence ranks.
            inner_ranks: Global ranks in the process group's actual order.
            outer: Current RD executor for the corresponding head slice.
        """
        peers = tuple(inner_ranks)
        if len(peers) < 2 or len(set(peers)) != len(peers):
            raise ValueError("Summary owner group requires at least two unique ranks")
        if tuple(dist.get_process_group_ranks(group)) != peers or dist.get_rank() not in peers:
            raise ValueError("Summary owner sequence order must match its process group order")
        self.group = group
        self.group_size = len(peers)
        self.inner_ranks = peers
        self.outer = outer

    def _validate(self, state: torch.Tensor, matrix: torch.Tensor) -> None:
        """Validate the dense summary contract before any collective is issued."""
        if (state.ndim != 4 or state.shape != matrix.shape or state.shape[-2:] != (128, 128)
                or state.shape[0] < 1 or state.shape[1] < 1 or state.shape[1] % self.group_size
                or state.dtype != torch.float32 or matrix.dtype != torch.float32
                or state.device != matrix.device or not state.is_contiguous() or not matrix.is_contiguous()):
            raise ValueError("Head-owner RD requires matching contiguous FP32 B/H/K128/V128 summaries")
        if state.device != self.outer.exchange.device:
            raise ValueError("Summary device must match the initialized communication backend")

    def _exchange(self, packet: torch.Tensor) -> torch.Tensor:
        """Wait before exposing received data; packet remains alive through send completion."""
        packet = packet.contiguous()
        result = torch.empty_like(packet)
        handle = dist.all_to_all_single(result, packet, group=self.group, async_op=True)
        handle.wait()
        return result

    def _scatter_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Collect every source rank's assigned head slice on its inner owner."""
        batch, heads, key_dim, columns = tensor.shape
        lanes = tensor.reshape(batch, self.group_size, heads // self.group_size, key_dim, columns)
        return self._exchange(lanes.permute(1, 0, 2, 3, 4))

    def _restore_heads(self, states: torch.Tensor) -> torch.Tensor:
        """Return the appropriate state slice to each token-owning rank."""
        received = self._exchange(states)
        _, batch, heads, key_dim, columns = received.shape
        return received.permute(1, 0, 2, 3, 4).reshape(batch, self.group_size * heads, key_dim, columns).contiguous()

    def forward(
        self, state: torch.Tensor, matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return the full-head input state and a local-M plus outer-RD cache.

        Args:
            state: Immutable local S, shaped [B, H, 128, 128].
            matrix: Immutable local M with the same shape and dtype.
        """
        self._validate(state, matrix)
        with torch.profiler.record_function("transport/head_owner/forward/scatter_SM"):
            maps = self._scatter_heads(torch.cat((state, matrix), dim=-1))
        offsets = maps[..., :128]
        transfers = maps[..., 128:].contiguous()
        aggregate = maps[0]
        with torch.profiler.record_function("transport/head_owner/forward/compose_group"):
            for index in range(1, self.group_size):
                composed = transfers[index] @ aggregate
                composed[..., :128].add_(offsets[index])
                aggregate = composed
        with torch.profiler.record_function("transport/head_owner/forward/outer_RD"):
            current, outer_cache = self.outer.forward(
                aggregate[..., :128].contiguous(), aggregate[..., 128:].contiguous())
        boundaries = []
        with torch.profiler.record_function("transport/head_owner/forward/local_prefix"):
            for index in range(self.group_size):
                boundaries.append(current)
                if index + 1 < self.group_size:
                    current = transfers[index] @ current + offsets[index]
        with torch.profiler.record_function("transport/head_owner/forward/restore_H"):
            incoming = self._restore_heads(torch.stack(boundaries))
        # Retaining a producer view must not keep its larger allocation alive.
        local_matrix = matrix
        if matrix.untyped_storage().nbytes() != matrix.numel() * matrix.element_size():
            local_matrix = matrix.clone()
        return incoming, (local_matrix, *outer_cache)

    def backward(self, gradient: torch.Tensor, cache: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Regather G/local M by head, propagate outer gradients and restore D.

        Args:
            gradient: Immutable local G, shaped [B, H, 128, 128].
            cache: Invocation-owned local M followed by the outer RD transfers.
        """
        expected = 1 + sum(self.outer.rank >= distance for distance in self.outer.distances)
        if not isinstance(cache, tuple) or len(cache) != expected or not isinstance(cache[0], torch.Tensor):
            raise ValueError("Head-owner RD received an incompatible invocation cache")
        local_matrix, *outer_cache = cache
        self._validate(gradient, local_matrix)
        expected_shape = (gradient.shape[0], gradient.shape[1] // self.group_size, 128, 128)
        if any(not isinstance(matrix, torch.Tensor) or matrix.shape != expected_shape or matrix.dtype != gradient.dtype
               or matrix.device != gradient.device or not matrix.is_contiguous() for matrix in outer_cache):
            raise ValueError("Head-owner RD outer cache does not match this invocation")
        with torch.profiler.record_function("transport/head_owner/backward/scatter_GM"):
            maps = self._scatter_heads(torch.cat((gradient, local_matrix), dim=-1))
        gradients = maps[..., :128]
        transfers = maps[..., 128:].contiguous()
        aggregate = gradients[-1]
        with torch.profiler.record_function("transport/head_owner/backward/compose_group"):
            for index in range(self.group_size - 2, -1, -1):
                aggregate = gradients[index] + transfers[index].transpose(-1, -2) @ aggregate
        with torch.profiler.record_function("transport/head_owner/backward/outer_RD"):
            current = self.outer.backward(aggregate.contiguous(), tuple(outer_cache))
        boundaries = []
        with torch.profiler.record_function("transport/head_owner/backward/local_suffix"):
            for index in range(self.group_size - 1, -1, -1):
                boundaries.append(current)
                if index > 0:
                    current = gradients[index] + transfers[index].transpose(-1, -2) @ current
        with torch.profiler.record_function("transport/head_owner/backward/restore_D"):
            return self._restore_heads(torch.stack(list(reversed(boundaries))))
