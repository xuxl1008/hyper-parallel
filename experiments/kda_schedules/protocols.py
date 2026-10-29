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
"""Isolated WORLD-ordered state protocols; no buffers survive an invocation."""

import torch
import torch.distributed as dist

from hyper_parallel.distributed.context_parallel.kimi_delta_attention_state import (
    KDAStateAllGather, KDAStateAllGatherP2P, _scan_maps,
)


class PacketChain:
    """Post receives early and keep outgoing buffers until sends complete.

    Communication is deliberately two-sided HCCL/Gloo, not device-side All-Scan.
    A chain instance belongs to one forward or backward invocation only.
    """

    def __init__(self, template: torch.Tensor, blocks: int, reverse: bool) -> None:
        self.rank, self.size = dist.get_rank(), dist.get_world_size()
        self.source = self.rank + (1 if reverse else -1)
        self.target = self.rank + (-1 if reverse else 1)
        self.incoming, self.receives, self.sent, self.sends = [], [], [], []
        for _ in range(blocks):
            incoming = torch.zeros_like(template)
            work = dist.irecv(incoming, self.source) if 0 <= self.source < self.size else None
            self.incoming.append(incoming)
            self.receives.append(work)

    def apply(self, index: int, offset: torch.Tensor, matrix: torch.Tensor, reverse: bool) -> None:
        """Consume one ready summary and forward its applied state, not its map."""
        if self.receives[index] is not None:
            self.receives[index].wait()
        if 0 <= self.target < self.size:
            transfer = matrix.transpose(-1, -2) if reverse else matrix
            outgoing = (transfer @ self.incoming[index] + offset).contiguous()
            self.sent.append(outgoing)
            self.sends.append(dist.isend(outgoing, self.target))

    def drain(self) -> None:
        """Protect send-buffer reuse and checkpoint replay message order."""
        for work in self.sends:
            work.wait()


class ValuePackets:
    """Full S/M/G once; split only the state and affine apply along V columns."""

    def __init__(self, blocks: int) -> None:
        if blocks not in (1, 2, 4, 8):
            raise ValueError("Value packet count must be 1, 2, 4 or 8")
        self.blocks = blocks

    def exchange(self, offset: torch.Tensor, matrix: torch.Tensor, reverse: bool) -> torch.Tensor:
        """Run an exclusive forward prefix or reverse transposed suffix."""
        if offset.shape[-1] % self.blocks:
            raise ValueError("Value dimension must divide the packet count")
        parts = offset.chunk(self.blocks, dim=-1)
        chain = PacketChain(parts[0].contiguous(), self.blocks, reverse)
        for index, part in enumerate(parts):
            chain.apply(index, part, matrix, reverse)
        chain.drain()
        return torch.cat(chain.incoming, dim=-1)

    def forward(self, state: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Return this rank's incoming recurrent state."""
        return self.exchange(state, matrix, False)

    def backward(self, gradient: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Return this rank's outgoing state gradient."""
        return self.exchange(gradient, matrix, True)


class TriangularAllToAllV:
    """Negative/control experiment: send each affine map only to ranks needing it."""

    @staticmethod
    def exchange(offset: torch.Tensor, matrix: torch.Tensor, reverse: bool) -> torch.Tensor:
        """Pack, exchange and merge compact source-ordered affine intervals."""
        rank, size = dist.get_rank(), dist.get_world_size()
        packed = torch.cat((offset, matrix), dim=-1)
        count = packed.numel()
        send = [count if (peer < rank if reverse else peer > rank) else 0 for peer in range(size)]
        recv = [count if (peer > rank if reverse else peer < rank) else 0 for peer in range(size)]
        send_buffer = packed.flatten().repeat(sum(send) // count)
        receive_buffer = packed.new_empty(sum(recv))
        dist.all_to_all_single(receive_buffer, send_buffer, recv, send)
        maps = receive_buffer.view(sum(recv) // count, *packed.shape)
        incoming = torch.zeros_like(offset)
        indices = range(maps.shape[0] - 1, -1, -1) if reverse else range(maps.shape[0])
        if offset.device.type == "npu":
            # Use the current fused scan, including its transpose rule in backward.
            return _scan_maps(maps, incoming, indices, not reverse)[0]
        # CPU protocol oracle only; never substitute this loop in NPU benchmarks.
        for index in indices:
            state, transfer = maps[index].split(offset.shape[-1], dim=-1)
            incoming = (transfer.transpose(-1, -2) if reverse else transfer) @ incoming + state
        return incoming

    def forward(self, state: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Return a causal exclusive prefix."""
        return self.exchange(state, matrix, False)

    def backward(self, gradient: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Return a reverse suffix with transposed transfers."""
        return self.exchange(gradient, matrix, True)


def make_protocol(mode: str, blocks: int, width: int) -> object | None:
    """Construct WORLD-ordered metadata collectively, outside measured steps."""
    rank, size = dist.get_rank(), dist.get_world_size()
    if mode == "p2p":
        return None
    if mode == "v-packet":
        return ValuePackets(blocks)
    if mode == "alltoallv":
        return TriangularAllToAllV()
    if mode == "allgather":
        return KDAStateAllGather(dist.group.WORLD, rank, size)
    if mode != "grouped" or width < 2 or size % width:
        raise ValueError("Grouped mode requires width >= 2 dividing WORLD size")
    intra = None
    # All WORLD ranks create every group in the same order, including non-members.
    for start in range(0, size, width):
        ranks = list(range(start, start + width))
        group = dist.new_group(ranks)
        if rank in ranks:
            intra = group
    return KDAStateAllGatherP2P(dist.group.WORLD, intra, rank, tuple(range(size)), width)
