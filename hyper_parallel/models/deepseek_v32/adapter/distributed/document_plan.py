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
"""Active packed documents shared by DSA selection, attention and KL."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate

import torch


@dataclass(frozen=True)
class DSADocumentPlan:
    """Positive Q/K prefixes in flattened BSND order, with document-local IDs."""

    lengths: tuple[int, ...]
    segments: tuple[tuple[int, int, int, int, int], ...]
    query_ends: tuple[int, ...]
    key_ends: tuple[int, ...]
    key_ranges: tuple[tuple[int, int], ...]

    def key(self, bank: torch.Tensor) -> torch.Tensor:
        """Pack complete visible prefixes; identical Q/K segment order is mandatory."""
        flat = bank.flatten(0, 1)
        parts = [flat[start:end] for start, end in self.key_ranges]
        packed = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return packed.unsqueeze(0)

    def sequence_lengths(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse immutable native cumulative lengths on their producing stream."""
        stream = None if device.type == "cpu" else torch.get_device_module(device).current_stream(device)
        return _device_lengths(self.query_ends, self.key_ends, device, stream)


@lru_cache(maxsize=128)
@torch.inference_mode(False)
def _device_lengths(query_ends, key_ends, device, stream):
    """Cache only address metadata, never per-step activations."""
    del stream
    return tuple(torch.tensor(ends, dtype=torch.int32, device=device) for ends in (query_ends, key_ends))


def _validate_document_boundaries(boundaries, batch, local, size, rank):
    """Reject incomplete or invalid global layouts before constructing cached routes."""
    global_length = local * size
    if batch < 1 or local < 1 or size < 1 or not 0 <= rank < size:
        raise ValueError("DSA document planning requires positive dimensions and a valid CP rank")
    if (not boundaries or boundaries[-1] != batch * global_length
            or any(end <= start for start, end in zip((0, *boundaries[:-1]), boundaries))):
        raise ValueError("DSA document ends must strictly increase and cover the global BSND input")


@lru_cache(maxsize=128)
def dsa_document_plan(boundaries: tuple[int, ...], batch: int, local: int,
                      size: int, rank: int) -> DSADocumentPlan:
    """Pair each nonempty local Q document with its causal global K prefix.

    Args:
        boundaries: Strict global cumulative document ends, without a leading zero.
        batch: Number of physical BSND rows. Documents cannot cross row boundaries.
        local: Equal sequence shard length per row.
        size: CP group size.
        rank: Sequence-ordered CP coordinate.

    Returns:
        A host-only immutable plan preserving flattened local Q order.
    """
    global_length = local * size
    _validate_document_boundaries(boundaries, batch, local, size, rank)
    segments, lengths = [], []
    for start, end in zip((0, *boundaries[:-1]), boundaries):
        row = start // global_length
        if (end - 1) // global_length != row:
            raise ValueError("DSA packed documents cannot cross physical batch rows")
        lengths.append(end - start)
        shard_start = row * global_length + rank * local
        q_start, q_end = max(start, shard_start), min(end, shard_start + local)
        if q_start < q_end:
            offset = row * local - shard_start
            segments.append((q_start + offset, q_end + offset, start, q_end, 0))
    query_ends = tuple(end for _, end, _, _, _ in segments)
    key_ends = tuple(accumulate(end - start for _, _, start, end, _ in segments))
    ranges = []
    for _, _, start, end, _ in segments:
        if ranges and ranges[-1][1] == start:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))
    return DSADocumentPlan(tuple(lengths), tuple(segments), query_ends, key_ends, tuple(ranges))


def pack_selection_prefixes(key: torch.Tensor, segments: tuple) -> tuple[torch.Tensor, tuple, tuple]:
    """Materialize LI task prefixes without changing document-local TopK coordinates.

    The selector has no backward. Owner KL uses its separate differentiable bank.
    Paging can replace this materialization independently after native parity checks.
    """
    parts = [key[:, start:end] for _, _, start, end, _ in segments]
    packed = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    q_ends = tuple(end for _, end, _, _, _ in segments)
    k_ends = tuple(accumulate(end - start for _, _, start, end, _ in segments))
    return packed, q_ends, k_ends
