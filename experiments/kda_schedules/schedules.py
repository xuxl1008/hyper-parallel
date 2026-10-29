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
"""Research-only KDA core scheduling; fixed dense BTHD, WORLD sequence order."""

from dataclasses import fields, is_dataclass

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from .protocols import PacketChain
from .stages import (
    prepare, full_summary, finish_forward, prepare_backward, gradient_summary, finish_backward,
    transition_once, value_summary, value_gradient,
)


MODES = ("p2p", "allgather", "grouped", "alltoallv", "v-packet", "m-once",
         "heads-whole", "heads-single", "dual", "dual-drain", "dual-immediate")
_STREAMS = {}
_SAVED_COUNT = 13


def _record(value, stream):
    if isinstance(value, torch.Tensor):
        value.record_stream(stream)
    elif is_dataclass(value):
        for field in fields(value):
            _record(getattr(value, field.name), stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record(item, stream)


def _head_parts(inputs, blocks):
    heads = inputs[0].shape[2]
    width = heads // blocks
    for start in range(0, heads, width):
        yield tuple((tensor[:, :, start:start + width] if i < 5 else tensor[start:start + width]).contiguous()
                    for i, tensor in enumerate(inputs))


class HeadExchange:
    """Event-ordered producer/state streams with invocation-owned packet buffers."""

    def __init__(self, template: torch.Tensor, blocks: int, mode: str, reverse: bool) -> None:
        self.mode, self.reverse = mode, reverse
        self.parent = torch.npu.current_stream(template.device)
        if mode == "heads-single":
            self.producer = self.state = self.parent
        else:
            key = template.device.index
            if key not in _STREAMS:
                _STREAMS[key] = (torch.npu.Stream(device=template.device), torch.npu.Stream(device=template.device))
            self.producer, self.state = _STREAMS[key]
            self.producer.wait_stream(self.parent)
            self.state.wait_stream(self.parent)
        with torch.npu.stream(self.state):
            self.chain = PacketChain(template, blocks, reverse)
        self.ready, self.done, self.offsets, self.matrices = [], [], [], []

    def append(self, offset: torch.Tensor, matrix: torch.Tensor) -> None:
        """Record producer completion and optionally submit its consumer immediately."""
        event = torch.npu.Event()
        event.record(self.producer)
        self.ready.append(event)
        self.offsets.append(offset)
        self.matrices.append(matrix)
        if self.mode in ("heads-single", "dual-immediate"):
            self.enqueue(len(self.ready) - 1)

    def enqueue(self, index: int) -> None:
        """Enqueue receive-ready -> apply -> send on the state stream."""
        with torch.npu.stream(self.state):
            self.state.wait_event(self.ready[index])
            _record((self.offsets[index], self.matrices[index]), self.state)
            with torch.autograd.profiler.record_function(f"schedule/state/{index}"):
                self.chain.apply(index, self.offsets[index], self.matrices[index], self.reverse)
            done = torch.npu.Event()
            done.record(self.state)
            self.done.append(done)

    def produced(self) -> None:
        """Finish enqueueing state work; drain mode delays every local finish."""
        if self.mode not in ("heads-single", "dual-immediate"):
            for index in range(len(self.ready)):
                self.enqueue(index)
        self.parent.wait_stream(self.producer)
        if self.mode == "dual-drain":
            with torch.npu.stream(self.state):
                self.chain.drain()
            self.parent.wait_stream(self.state)

    def incoming(self, index: int) -> torch.Tensor:
        """Expose one state only after its receive/apply dependency is satisfied."""
        self.parent.wait_event(self.done[index])
        incoming = self.chain.incoming[index]
        incoming.record_stream(self.parent)
        return incoming

    def close(self) -> None:
        """Join sends before buffer release or a subsequent checkpoint replay."""
        with torch.npu.stream(self.state):
            self.chain.drain()
        self.parent.wait_stream(self.state)


class HeadFunction(torch.autograd.Function):
    """Symmetric full-core head preparation followed by state-first output."""

    @staticmethod
    def forward(ctx, query, key, value, raw_gate, raw_beta, alog, bias, mode, blocks):
        inputs = (query, key, value, raw_gate, raw_beta, alog, bias)
        template = query.new_empty((query.shape[0], query.shape[2] // blocks, 128, 128), dtype=torch.float32)
        exchange = HeadExchange(template, blocks, mode, False)
        stages = []
        with torch.npu.stream(exchange.producer):
            _record(inputs, exchange.producer)
            for index, part in enumerate(_head_parts(inputs, blocks)):
                with torch.autograd.profiler.record_function(f"schedule/forward/producer/{index}"):
                    stage = prepare(part)
                    offset, matrix = full_summary(stage)
                stages.append(stage)
                exchange.append(offset, matrix)
        exchange.produced()
        outputs, saved = [], []
        for index, stage in enumerate(stages):
            _record(stage, exchange.parent)
            h0 = exchange.incoming(index)
            with torch.autograd.profiler.record_function(f"schedule/forward/finish/{index}"):
                outputs.append(finish_forward(stage, h0))
            saved.extend((*stage.saved, exchange.matrices[index], h0))
        exchange.close()
        ctx.save_for_backward(*saved)
        ctx.mode, ctx.blocks = mode, blocks
        return torch.cat(outputs, dim=2)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        saved = ctx.saved_tensors
        parts = [saved[i * _SAVED_COUNT:(i + 1) * _SAVED_COUNT] for i in range(ctx.blocks)]
        exchange = HeadExchange(parts[0][-1], ctx.blocks, ctx.mode, True)
        stages = []
        with torch.npu.stream(exchange.producer):
            _record((parts, grad), exchange.producer)
            for index, (part, block_grad) in enumerate(zip(parts, grad.chunk(ctx.blocks, dim=2), strict=True)):
                with torch.autograd.profiler.record_function(f"schedule/backward/producer/{index}"):
                    stage = prepare_backward(part, block_grad)
                    offset = gradient_summary(stage)
                stages.append(stage)
                exchange.append(offset, part[-2])
        exchange.produced()
        gradients = []
        for index, stage in enumerate(stages):
            _record(stage, exchange.parent)
            with torch.autograd.profiler.record_function(f"schedule/backward/finish/{index}"):
                gradients.append(finish_backward(stage, exchange.incoming(index)))
        exchange.close()
        return (*[torch.cat([block[i] for block in gradients], dim=2 if i < 5 else 0)
                  for i in range(7)], None, None)


class ValueFunction(torch.autograd.Function):
    """M-once, S/G produced separately per V block; full-head prepare/finish."""

    @staticmethod
    def forward(ctx, query, key, value, raw_gate, raw_beta, alog, bias, blocks):
        stage = prepare((query, key, value, raw_gate, raw_beta, alog, bias))
        width = 128 // blocks
        matrix = transition_once(stage)
        template = matrix.new_empty((*matrix.shape[:-1], width))
        chain = PacketChain(template, blocks, False)
        for index in range(blocks):
            with torch.autograd.profiler.record_function(f"schedule/value/S/{index}"):
                offset = value_summary(stage, index * width, width)
                chain.apply(index, offset, matrix, False)
        h0 = torch.cat(chain.incoming, dim=-1)
        output = finish_forward(stage, h0)
        chain.drain()
        ctx.save_for_backward(*stage.saved, matrix, h0)
        ctx.blocks = blocks
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        saved = ctx.saved_tensors
        stage = prepare_backward(saved, grad)
        width, matrix = 128 // ctx.blocks, saved[-2]
        chain = PacketChain(matrix.new_empty((*matrix.shape[:-1], width)), ctx.blocks, True)
        for index in range(ctx.blocks):
            with torch.autograd.profiler.record_function(f"schedule/value/G/{index}"):
                offset = value_gradient(stage, index * width, width)
                chain.apply(index, offset, matrix, True)
        dht = torch.cat(chain.incoming, dim=-1)
        gradients = finish_backward(stage, dht)
        chain.drain()
        return (*gradients, None)


def run_core(inputs: tuple, mode: str, blocks: int = 2, protocol: object = None) -> torch.Tensor:
    """Run fixed dense KDA raw inputs; construct protocol metadata before this call.

    Args:
        inputs: q/k/v/gate [B,T,H,128], beta [B,T,H], FP32 A_log [H], dt_bias [H,128].
        mode: One of MODES; WORLD ranks must use the same mode and block count.
        blocks: Head or value blocks, depending on mode.
        protocol: Invocation-stateless metadata returned by make_protocol.

    Returns:
        Local output [B,T,H,128]. Global initial state is zero.
    """
    if mode not in MODES or blocks not in (1, 2, 4, 8) or len(inputs) != 7:
        raise ValueError("Unsupported mode/block count/input tuple")
    query, key, value, gate, beta, alog, bias = inputs
    if (query.ndim != 4 or query.shape[-1] != 128 or query.shape[1] % 64
            or min(query.shape[:3]) < 1):
        raise ValueError("Require nonempty BTH128, T divisible by 64")
    if (mode.startswith("heads-") or mode.startswith("dual")) and query.shape[2] % blocks:
        raise ValueError("Head schedules require H divisible by blocks")
    if (any(x.shape != query.shape for x in (key, value, gate)) or beta.shape != query.shape[:-1]
            or alog.shape != (query.shape[2],) or bias.shape != (query.shape[2], 128)):
        raise ValueError("This reference supports equal query/value heads and explicit parameter shapes only")
    if (any(x.device != query.device for x in inputs) or query.device.type != "npu"
            or any(x.dtype != torch.bfloat16 for x in inputs[:5])
            or any(x.dtype != torch.float32 for x in inputs[5:])):
        raise ValueError("Require BF16 token tensors and FP32 A_log/dt_bias on one NPU")
    if any(not x.is_contiguous() for x in inputs):
        raise ValueError("Pass contiguous inputs to make packing costs explicit")
    if mode == "m-once":
        return ValueFunction.apply(*inputs, blocks)
    if mode in ("heads-single", "dual", "dual-drain", "dual-immediate"):
        return HeadFunction.apply(*inputs, mode, blocks)
    rank, size = dist.get_rank(), dist.get_world_size()

    def formal(part, executor):
        # Formal fused entry imports optional Triton; CPU protocol checks do not need it.
        from hyper_parallel.components.functional.kimi_delta_attention import fused_chunk_kda_p2p
        return fused_chunk_kda_p2p(*part[:5], a_log=part[5], dt_bias=part[6],
                                   cp_group=dist.group.WORLD, prev_rank=rank - 1, next_rank=rank + 1,
                                   cp_rank=rank, cp_size=size, state_cp=executor)

    if mode == "heads-whole":
        return torch.cat([formal(part, None) for part in _head_parts(inputs, blocks)], dim=2)
    if mode != "p2p" and protocol is None:
        raise ValueError("This protocol mode requires precreated communication metadata")
    return formal(inputs, protocol)
