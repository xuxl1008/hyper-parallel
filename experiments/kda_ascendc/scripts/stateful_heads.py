# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Head-block schedules for the isolated AscendC KDA prototype.

The model projections, convolution, normalization and output projection are not
split. Only the normalized core is partitioned; no kernel implementation changes.
"""
import re

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from stateful_kda import StateCP, prepare, finish_forward, prepare_backward, dhu, finish_backward
from memory_options import retained_prepared


_STREAMS = {}


def split_inputs(inputs, blocks):
    heads = inputs[0].shape[1]
    if blocks < 1 or heads % blocks:
        raise ValueError("Head count must be divisible by the block count")
    width = heads // blocks
    return [tuple(x[:, start:start+width].contiguous() if i < 5
                  else x[start:start+width].contiguous() for i, x in enumerate(inputs))
            for start in range(0, heads, width)]


def head_core(inputs, mode):
    match = re.fullmatch(r"p2p_(whole|sf|dual)([1-9][0-9]*)", mode)
    if match is None:
        raise ValueError(mode)
    schedule, blocks = match[1], int(match[2])
    if schedule == "whole":
        return torch.cat([StateCP.apply(*part, "p2p") for part in split_inputs(inputs, blocks)], 1)
    return HeadStateCP.apply(*inputs, blocks, schedule == "dual")


class StatePackets:
    """One ordered packet exchange; buffers survive until all sends are drained."""

    def __init__(self, template, blocks, reverse, dual):
        self.rank, self.size = dist.get_rank(), dist.get_world_size()
        self.source = self.rank + (1 if reverse else -1)
        self.dest = self.rank + (-1 if reverse else 1)
        self.receives, self.works, self.sends, self.sent, self.events = [], [], [], [], []
        self.dual = dual
        self.producer = torch.npu.current_stream(template.device)
        if dual:
            key = template.device.index
            if key not in _STREAMS:
                _STREAMS[key] = torch.npu.Stream(device=template.device)
            self.consumer = _STREAMS[key]
            self.consumer.wait_stream(self.producer)
        else:
            self.consumer = self.producer
        for _ in range(blocks):
            state = torch.empty_like(template)
            if 0 <= self.source < self.size:
                work = dist.irecv(state, src=self.source)
            else:
                state.zero_()
                work = None
            self.receives.append(state)
            self.works.append(work)

    def enqueue(self, index, summary, matrix, apply):
        ready = torch.npu.Event()
        ready.record(self.producer)
        incoming = self.receives[index]
        with torch.npu.stream(self.consumer):
            self.consumer.wait_event(ready)
            if self.works[index] is not None:
                self.works[index].wait()
            if self.dual:
                for tensor in (summary, matrix, incoming):
                    tensor.record_stream(self.consumer)
            if 0 <= self.dest < self.size:
                outgoing = apply(summary, matrix, incoming).contiguous()
                self.sent.append(outgoing)
                self.sends.append(dist.isend(outgoing, dst=self.dest))
            forwarded = torch.npu.Event()
            forwarded.record(self.consumer)
            self.events.append(forwarded)

    def before_finish(self):
        # Only stream dependencies, not a device-wide synchronization. All local
        # state applies are enqueued before the token-local finish phase starts.
        for event in self.events:
            self.producer.wait_event(event)

    def drain(self):
        with torch.npu.stream(self.consumer):
            for work in self.sends:
                work.wait()
        if self.dual:
            self.producer.wait_stream(self.consumer)


class HeadStateCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, raw, beta, alog, bias, blocks, dual):
        from hyper_parallel.components.functional.kimi_delta_attention_state_summary import (
            kda_state_summary_forward_from_prepared, apply_kda_state_summary)
        inputs = split_inputs((q, k, v, raw, beta, alog, bias), blocks)
        template = torch.empty(q.shape[0], q.shape[1] // blocks, 128, 128,
                               device=q.device, dtype=torch.float32)
        packets = StatePackets(template, blocks, False, dual)
        prepared, matrices = [], []
        for index, part in enumerate(inputs):
            with torch.autograd.profiler.record_function(f"ascendc_heads/fwd/producer_{index}"):
                p = prepare(*part)
                gk, _, _, w, u, _, kg, *_ = p
                summary, matrix = kda_state_summary_forward_from_prepared(
                    *(x.transpose(1, 2).contiguous() for x in (kg, w, u, gk)), chunk_size=64)
            prepared.append(p)
            matrices.append(matrix)
            packets.enqueue(index, summary, matrix, apply_kda_state_summary)
        packets.before_finish()
        outputs, saved = [], []
        for index, (part, p, matrix) in enumerate(zip(inputs, prepared, matrices)):
            h0 = packets.receives[index]
            with torch.autograd.profiler.record_function(f"ascendc_heads/fwd/finish_{index}"):
                out, _, h, vn = finish_forward(p, h0)
            outputs.append(out)
            saved.extend((*part, h0, *retained_prepared(p), h, vn, matrix))
        packets.drain()
        ctx.save_for_backward(*saved)
        ctx.blocks, ctx.dual = blocks, dual
        return torch.cat(outputs, 1)

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        from hyper_parallel.components.functional.kimi_delta_attention_state_summary import (
            kda_state_gradient_summary_from_prepared, apply_kda_state_gradient_summary)
        xs = ctx.saved_tensors
        parts = [xs[i*19:(i+1)*19] for i in range(ctx.blocks)]
        packets = StatePackets(parts[0][7], ctx.blocks, True, ctx.dual)
        backward_parts = []
        for index, (part, grad) in enumerate(zip(parts, do.chunk(ctx.blocks, dim=1))):
            inputs, p, h, vn, matrix = part[:7], part[8:16], part[16], part[17], part[18]
            grad = grad.contiguous()
            with torch.autograd.profiler.record_function(f"ascendc_heads/bwd/producer_{index}"):
                da, dv, dqraw = prepare_backward(p[1], vn, grad, h)
                gk, _, _, w, _, qg, kg, *_ = p
                summary = kda_state_gradient_summary_from_prepared(
                    *(x.transpose(1, 2).contiguous() for x in (qg, kg, w, gk, grad, dv)),
                    .125, chunk_size=64)
            backward_parts.append((grad, da, dv, dqraw))
            packets.enqueue(index, summary, matrix, apply_kda_state_gradient_summary)
        packets.before_finish()
        grads = []
        for index, (part, stage) in enumerate(zip(parts, backward_parts)):
            inputs, h0, p, h, vn = part[:7], part[7], part[8:16], part[16], part[17]
            grad, da, dv, dqraw = stage
            with torch.autograd.profiler.record_function(f"ascendc_heads/bwd/finish_{index}"):
                dh, _, dvs = dhu(p, grad, dv, h0, packets.receives[index])
                grads.append(finish_backward(inputs, p, h, vn, dh, dvs, da, dqraw))
        packets.drain()
        return (*[torch.cat([part[i] for part in grads], dim=1 if i < 5 else 0)
                  for i in range(7)], None, None)
