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
"""Immutable reset-aware RD plans derived from packed document metadata."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class RDResetPlan:
    """Active edges of the established inclusive scan and exclusive shift.

    The same plan is transposed by backward. Matrices are never inspected on
    device: only true document boundaries can remove an edge.
    """

    size: int
    distances: tuple[int, ...]
    receivers: tuple[frozenset[int], ...]
    shift_receivers: frozenset[int]
    transfer_active: tuple[bool, ...]

    def peers(self, stage: int, rank: int) -> tuple[int | None, int | None]:
        """Return forward destination/source, both in process-group order."""
        distance = self.distances[stage]
        source = rank - distance if rank in self.receivers[stage] else None
        destination = rank + distance if rank + distance in self.receivers[stage] else None
        return destination, source

    def active_stages(self, rank: int) -> tuple[int, ...]:
        """Return only rounds in which this rank sends or receives."""
        return tuple(stage for stage in range(len(self.distances))
                     if any(peer is not None for peer in self.peers(stage, rank)))

    def cache_count(self, rank: int) -> int:
        """Count invocation-owned pre-round transfers on one rank."""
        return sum(rank in receivers for receivers in self.receivers)

    @property
    def messages_per_direction(self) -> int:
        """Count application-level sends, excluding communication setup."""
        return sum(map(len, self.receivers)) + len(self.shift_receivers)


def _transfer_flags(offsets: tuple[int, ...], local_length: int, size: int) -> tuple[tuple, tuple]:
    """Describe document continuation and exact entry-to-exit connectivity."""
    if (size < 1 or local_length < 1 or len(offsets) < 2 or offsets[0] != 0
            or offsets[-1] != size * local_length
            or any(left >= right for left, right in zip(offsets, offsets[1:]))):
        raise ValueError("RD reset planning requires ordered documents covering equal state-rank intervals")
    cuts = frozenset(offsets)
    continuation = tuple(rank > 0 and rank * local_length not in cuts for rank in range(size))
    transfer = tuple(
        continuation[rank]
        and bisect_right(offsets, rank * local_length) == bisect_left(offsets, (rank + 1) * local_length)
        for rank in range(size)
    )
    return continuation, transfer


def _stage_connectivity(transfer: tuple[bool, ...], distances: tuple[int, ...]) -> list:
    """Propagate structural nonzero flags without looking at tensor values."""
    nonzero = [transfer]
    for distance in distances:
        previous = nonzero[-1]
        nonzero.append(tuple(previous[rank] and (rank < distance or previous[rank - distance])
                             for rank in range(len(transfer))))
    return nonzero


@lru_cache(maxsize=32)
def build_reset_plan(offsets: tuple[int, ...], local_length: int, size: int) -> RDResetPlan:
    """Prune zero-transfer edges and intermediates with no state consumer.

    Args:
        offsets: Global document offsets, including zero and the token total.
        local_length: Equal token interval length after any Ulysses exchange.
        size: Number of chronological state ranks, independent of global rank IDs.

    Returns:
        An immutable CPU plan reusable across layers with identical metadata.
    """
    continuation, transfer = _transfer_flags(offsets, local_length, size)
    distances = tuple(1 << level for level in range((size - 1).bit_length()))
    nonzero = _stage_connectivity(transfer, distances)
    shift = frozenset(rank for rank in range(size) if continuation[rank])
    live = [rank + 1 in shift for rank in range(size)]
    receivers = []
    for stage in range(len(distances) - 1, -1, -1):
        distance = distances[stage]
        active = frozenset(rank for rank in range(distance, size) if live[rank] and nonzero[stage][rank])
        # Every active sum retains its right/identity operand. Left operands
        # make earlier nodes live; no gradient with respect to M is requested.
        for rank in active:
            live[rank - distance] = True
        receivers.append(active)
    return RDResetPlan(size, distances, tuple(reversed(receivers)), shift, transfer)
