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
"""No-grad query-task redistribution for compressed Indexer selection.

Main attention, teacher and KL retain the original sequence ownership. Selection
exchanges low-rank query residuals, merge weights and original rotary embeddings;
receivers project Indexer Q and return integer TopK results. The canonical Index K
bank remains differentiable and is never repacked by this communication layer.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import


def resolve_indexer_balance_block_size(mode: str, block_size: int | None) -> int | None:
    """Resolve the greedy-only query block option, defaulting to 2048 rows.

    None represents an omitted option so none/zigzag remain valid without it.
    Explicit values require greedy mode and must be positive integers.
    """
    if mode not in ("none", "zigzag", "greedy"):
        raise ValueError("indexer_load_balance must be 'none', 'zigzag' or 'greedy'")
    if block_size is not None:
        if mode != "greedy":
            raise ValueError("indexer_balance_block_size requires indexer_load_balance='greedy'")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
            raise ValueError("indexer_balance_block_size must be a positive integer")
    return (2048 if block_size is None else block_size) if mode == "greedy" else None


@dataclass(frozen=True, order=True)
class IndexerTask:
    """A contiguous query interval in the original document coordinates."""

    document: int
    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the number of real query rows, excluding any synthetic padding."""
        return self.end - self.start


def _merge_tasks(tasks: list[IndexerTask]) -> tuple[IndexerTask, ...]:
    merged = []
    for task in sorted(tasks):
        if merged and merged[-1].document == task.document and merged[-1].end == task.start:
            merged[-1] = IndexerTask(task.document, merged[-1].start, task.end)
        elif task.length:
            merged.append(task)
    return tuple(merged)


def _work_prefix(length: int, ratio: int) -> int:
    groups, tail = divmod(length, ratio)
    return ratio * groups * (groups - 1) // 2 + groups * (tail + 1)


def _task_work(task: IndexerTask, ratio: int) -> int:
    return _work_prefix(task.end, ratio) - _work_prefix(task.start, ratio)


def _zigzag_tasks(lengths: tuple[int, ...], size: int) -> list[list[IndexerTask]]:
    assignments = [[] for _ in range(size)]
    remainders = []
    for document, length in enumerate(lengths):
        chunk, remainder = divmod(length, 2 * size)
        if remainder:
            remainders.append(IndexerTask(document, 0, remainder))
        if chunk:
            for rank in range(size):
                for slot in (rank, 2 * size - 1 - rank):
                    start = remainder + slot * chunk
                    assignments[rank].append(IndexerTask(document, start, start + chunk))
    # All ranks already own equally many paired rows. Distribute the real
    # remainder rows evenly without inserting tokens into any document.
    capacity = sum(task.length for task in remainders) // size
    rank, used = 0, 0
    for task in remainders:
        start = task.start
        while start < task.end:
            if used == capacity:
                rank += 1
                used = 0
            end = min(task.end, start + capacity - used)
            assignments[rank].append(IndexerTask(task.document, start, end))
            used += end - start
            start = end
    return assignments


def _greedy_tasks(lengths: tuple[int, ...], size: int, ratio: int, block: int) -> list[list[IndexerTask]]:
    local = sum(lengths) // size
    tasks = [IndexerTask(document, start, min(start + block, length))
             for document, length in enumerate(lengths) for start in range(0, length, block)]
    assignments = [[] for _ in range(size)]
    costs, counts = [0] * size, [0] * size
    for task in sorted(tasks, key=lambda item: (-_task_work(item, ratio), item)):
        candidates = [rank for rank in range(size) if counts[rank] + task.length <= 2 * local]
        rank = min(candidates, key=lambda peer: (costs[peer], counts[peer], peer))
        assignments[rank].append(task)
        costs[rank] += _task_work(task, ratio)
        counts[rank] += task.length
    return assignments


@lru_cache(maxsize=128)
def indexer_task_plan(lengths: tuple[int, ...], size: int, ratio: int, mode: str,
                      block_size: int | None = None) -> tuple:
    """Assign real queries using zigzag with pooled remainders or block greedy.

    Args:
        lengths: Global document lengths in packed order.
        size: CP group size; total packed tokens must divide this value.
        ratio: Compression ratio used by the unchanged Indexer kernel.
        mode: 'zigzag' pairs document tails and heads after removing the leading
            remainder; 'greedy' weights blocks by the number of causal Q/K pairs.
        block_size: Maximum original query rows per greedy task, before adjacent
            tasks assigned to one rank are merged. Greedy defaults to 2048.
            Explicit values require greedy and must be positive and at most local Q
            to guarantee its per-worker row bound. Omit this option for zigzag.

    Returns:
        A sorted, disjoint task list for each rank. Greedy caps each worker at
        twice the original local query count; zigzag preserves equal row counts.
    """
    if size < 1 or ratio < 1 or not lengths or min(lengths) < 1 or sum(lengths) % size:
        raise ValueError("Indexer balance needs positive document lengths/ratio and an equal contiguous CP partition")
    block = resolve_indexer_balance_block_size(mode, block_size)
    if mode == "zigzag":
        assignments = _zigzag_tasks(lengths, size)
    elif mode == "greedy":
        local = sum(lengths) // size
        if block > local:
            raise ValueError(f"Indexer greedy block size {block} exceeds the local query count {local}")
        assignments = _greedy_tasks(lengths, size, ratio, block)
    else:
        raise ValueError("Indexer task mode must be 'zigzag' or 'greedy'")
    return tuple(_merge_tasks(tasks) for tasks in assignments)


def _segments(tasks: tuple, lengths: tuple[int, ...], ratio: int) -> tuple:
    key_bases = (0, *accumulate(length // ratio for length in lengths))
    result, cursor = [], 0
    for task in tasks:
        base = key_bases[task.document]
        result.append((cursor, cursor + task.length, base, base + task.end // ratio, task.end % ratio))
        cursor += task.length
    return tuple(result)


def _intersections(tasks: tuple, boundaries: tuple[int, ...], owner: int, local: int) -> tuple:
    result, cursor = [], 0
    for task in tasks:
        begin, end = boundaries[task.document] + task.start, boundaries[task.document] + task.end
        first, last = max(begin, owner * local), min(end, (owner + 1) * local)
        if first < last:
            result.append((first - owner * local, cursor + first - begin, last - first))
        cursor += task.length
    return tuple(result)


@dataclass(frozen=True)
class IndexerRoute:
    """Device-independent row ordering and matching variable collective splits."""

    segments: tuple
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    send_rows: tuple[int, ...]
    receive_rows: tuple[int, ...]
    local_sources: tuple[int, ...]
    local_targets: tuple[int, ...]
    query_count: int


@lru_cache(maxsize=128)
def indexer_route(lengths: tuple[int, ...], size: int, rank: int, ratio: int, mode: str,
                  block_size: int | None = None) -> IndexerRoute:
    """Keep self-owned tasks local and communicate the remaining rows in owner order."""
    if not 0 <= rank < size:
        raise ValueError("Indexer balance rank must be inside its CP group")
    assignments = indexer_task_plan(lengths, size, ratio, mode, block_size)
    boundaries, local = (0, *accumulate(lengths)), sum(lengths) // size
    send_rows, receive_rows, inputs, outputs = [], [], [], []
    for peer in range(size):
        sent, received = (), ()
        if peer != rank:
            sent = _intersections(assignments[peer], boundaries, rank, local)
            received = _intersections(assignments[rank], boundaries, peer, local)
        inputs.append(sum(count for _, _, count in sent))
        outputs.append(sum(count for _, _, count in received))
        send_rows.extend(row for start, _, count in sent for row in range(start, start + count))
        receive_rows.extend(row for _, start, count in received for row in range(start, start + count))
    kept = _intersections(assignments[rank], boundaries, rank, local)
    sources = tuple(row for start, _, count in kept for row in range(start, start + count))
    targets = tuple(row for _, start, count in kept for row in range(start, start + count))
    return IndexerRoute(_segments(assignments[rank], lengths, ratio), tuple(inputs), tuple(outputs),
                        tuple(send_rows), tuple(receive_rows), sources, targets,
                        sum(task.length for task in assignments[rank]))


@lru_cache(maxsize=32)
@torch.inference_mode(False)
def _device_rows(route: IndexerRoute, device: torch.device) -> tuple:
    return tuple(torch.tensor(rows, device=device, dtype=torch.long) for rows in
                 (route.send_rows, route.receive_rows, route.local_sources, route.local_targets))


def _exchange_rows(tensor: torch.Tensor, route: IndexerRoute, group: Any, rows: tuple) -> torch.Tensor:
    send_rows, receive_rows, local_sources, local_targets = rows
    send = tensor.index_select(0, send_rows).contiguous()
    received = tensor.new_empty((sum(route.output_splits), *tensor.shape[1:]))
    work = dist.all_to_all_single(received, send, list(route.output_splits), list(route.input_splits),
                                  group=group, async_op=True)
    output = tensor.new_empty((route.query_count, *tensor.shape[1:]))
    output.index_copy_(0, local_targets, tensor.index_select(0, local_sources))
    work.wait()
    output.index_copy_(0, receive_rows, received)
    return output


def _exchange_inputs(inputs: tuple[torch.Tensor, ...], route: IndexerRoute, group: Any, rows: tuple) -> tuple:
    """Pack fields of equal dtype together without rounding the original RoPE values."""
    by_dtype = {}
    for index, tensor in enumerate(inputs):
        by_dtype.setdefault(tensor.dtype, []).append(index)
    outputs = [None] * len(inputs)
    for indices in by_dtype.values():
        widths = [inputs[index].shape[-1] for index in indices]
        packed = torch.cat([inputs[index][0] for index in indices], dim=-1)
        received = _exchange_rows(packed, route, group, rows)
        for index, field in zip(indices, received.split(widths, dim=-1)):
            outputs[index] = field.unsqueeze(0).contiguous()
    return tuple(outputs)


@torch.no_grad()
def balanced_indexer_select(query_residual: torch.Tensor, weights: torch.Tensor,
                            position_embeddings: tuple[torch.Tensor, torch.Tensor], selector: Callable,
                            lengths: tuple[int, ...], ratio: int, mode: str, group: Any,
                            block_size: int | None = None) -> torch.Tensor:
    """Redistribute low-rank selection inputs and return TopK to the original owners.

    Args:
        query_residual: Local low-rank query residual [1, local tokens, q_lora_rank].
        weights: Local merge weights [1, local tokens, heads].
        position_embeddings: Original per-token cos/sin, each [1, local tokens, rotary/2].
            All payload dtypes are preserved; tensors of equal dtype share a collective.
        selector: Callable (qr, weights, (cos, sin), segments) that projects Indexer Q
            with the current layer parameters and selects against the full Index K bank.
            It runs without autograd; owner-side Q and weights retain their KL gradients.
        lengths: Global original document lengths; no per-document CP padding is added.
        ratio: Compression ratio used by this selection layer.
        mode: 'zigzag' or 'greedy', identical on all CP ranks.
        group: Original sequence-ordered CP process group.
        block_size: Greedy task length in original query rows; see indexer_task_plan.
    """
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    inputs = (query_residual, weights, *position_embeddings)
    if (len(position_embeddings) != 2 or query_residual.ndim != 3 or query_residual.shape[0] != 1
            or query_residual.shape[1] * size != sum(lengths)):
        raise ValueError("Indexer balance requires batch one and equal contiguous source query shards")
    if any(tensor.ndim != 3 or tensor.shape[:2] != query_residual.shape[:2]
           or tensor.device != query_residual.device for tensor in inputs):
        raise ValueError("Indexer qr, weights and rotary embeddings must have matching batch/token axes and device")
    route = indexer_route(lengths, size, rank, ratio, mode, block_size)
    rows = _device_rows(route, query_residual.device)
    balanced_qr, balanced_weights, cos, sin = _exchange_inputs(inputs, route, group, rows)
    selected = selector(balanced_qr, balanced_weights, (cos, sin), route.segments)[0]
    send_rows, receive_rows, local_sources, local_targets = rows
    send = selected.index_select(0, receive_rows).contiguous()
    received = selected.new_empty((sum(route.input_splits), selected.shape[-1]))
    work = dist.all_to_all_single(received, send, list(route.input_splits), list(route.output_splits),
                                  group=group, async_op=True)
    output = selected.new_empty((query_residual.shape[1], selected.shape[-1]))
    output.index_copy_(0, local_sources, selected.index_select(0, local_targets))
    work.wait()
    output.index_copy_(0, send_rows, received)
    return output.unsqueeze(0)
