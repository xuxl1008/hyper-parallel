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
"""Invocation-owned lifetime and direct-restore variants of summary-head RD.

All workspaces belong to one invocation. Arithmetic order, groups, collectives,
outer RD and persistent caches are unchanged. These explicit forward/backward
protocols run inside a custom autograd function; out= does not build an autograd
trace and is not an alternative differentiable torch operator.
"""
from __future__ import annotations

import torch

from .head_owner_rd import KDAStateHeadOwnerRD


class HeadOwnerRDLifetime(KDAStateHeadOwnerRD):
    """Drop dead summaries before restore; optionally produce its send layout."""

    direct_restore = False

    def forward(self, state: torch.Tensor, matrix: torch.Tensor) -> tuple:
        """Keep the exact baseline composition order and invocation-owned cache."""
        self._validate(state, matrix)
        with torch.profiler.record_function("transport/head_owner/forward/scatter_SM"):
            maps = self._scatter_heads(torch.cat((state, matrix), dim=-1))
        offsets = maps[..., :128]
        transfers = maps[..., 128:].contiguous()
        aggregate = maps[0]
        with torch.profiler.record_function("transport/head_owner/forward/compose_group"):
            for index in range(1, self.group_size):
                # No second `composed` reference may retain the final aggregate.
                aggregate = transfers[index] @ aggregate
                aggregate[..., :128].add_(offsets[index])
        with torch.profiler.record_function("transport/head_owner/forward/outer_RD"):
            # Both packed halves need distinct contiguous copies. Once they
            # exist, the packed aggregate is dead even before outer RD starts.
            outer_state = aggregate[..., :128].contiguous()
            outer_matrix = aggregate[..., 128:].contiguous()
            del aggregate
            current, outer_cache = self.outer.forward(outer_state, outer_matrix)
            del outer_state, outer_matrix
        with torch.profiler.record_function("transport/head_owner/forward/local_prefix"):
            if self.direct_restore:
                # Each destination block is contiguous, including B > 1.
                boundaries = torch.empty((self.group_size, *current.shape),
                                         dtype=current.dtype, device=current.device)
                boundaries[0].copy_(current)
                del current
                for index in range(1, self.group_size):
                    torch.matmul(transfers[index - 1], boundaries[index - 1], out=boundaries[index])
                    torch.add(boundaries[index], offsets[index - 1], out=boundaries[index])
            else:
                boundaries = []
                for index in range(self.group_size):
                    boundaries.append(current)
                    if index + 1 < self.group_size:
                        current = transfers[index] @ current + offsets[index]
                del current
        # offsets aliases maps: both references must end after the final add.
        del offsets, transfers, maps
        with torch.profiler.record_function("transport/head_owner/forward/restore_H"):
            if not self.direct_restore:
                packet = torch.stack(boundaries)
                del boundaries
            else:
                packet = boundaries
                del boundaries
            incoming = self._restore_heads(packet)
        del packet
        local_matrix = matrix
        if matrix.untyped_storage().nbytes() != matrix.numel() * matrix.element_size():
            local_matrix = matrix.clone()
        return incoming, (local_matrix, *outer_cache)

    def backward(self, gradient: torch.Tensor, cache: tuple) -> torch.Tensor:
        """Preserve gradient-first additions, reverse order and immutable cache."""
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
        del aggregate
        with torch.profiler.record_function("transport/head_owner/backward/local_suffix"):
            if self.direct_restore:
                boundaries = torch.empty((self.group_size, *current.shape),
                                         dtype=current.dtype, device=current.device)
                boundaries[-1].copy_(current)
                del current
                for index in range(self.group_size - 2, -1, -1):
                    torch.matmul(transfers[index + 1].transpose(-1, -2), boundaries[index + 1],
                                 out=boundaries[index])
                    # Keep the baseline's operand order: gradient + product.
                    torch.add(gradients[index + 1], boundaries[index], out=boundaries[index])
            else:
                boundaries = []
                for index in range(self.group_size - 1, -1, -1):
                    boundaries.append(current)
                    if index > 0:
                        current = gradients[index] + transfers[index].transpose(-1, -2) @ current
                del current
        del gradients, transfers, maps
        with torch.profiler.record_function("transport/head_owner/backward/restore_D"):
            if not self.direct_restore:
                packet = torch.stack(list(reversed(boundaries)))
                del boundaries
            else:
                packet = boundaries
                del boundaries
            return self._restore_heads(packet)


class HeadOwnerRDDirectRestore(HeadOwnerRDLifetime):
    """Write local prefix/suffix results directly into the AllToAll send packet."""

    direct_restore = True
