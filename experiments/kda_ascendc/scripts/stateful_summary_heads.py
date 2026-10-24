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
"""Full-head AscendC stages with head-range summary production and State-P2P.

This diagnostic reuses the current production kernels without copying or editing
their source. Only chronological WORLD and dense D128/chunk64 are supported.
"""
import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.components.functional.kimi_delta_attention_state_summary import (
    apply_kda_state_summary,
    apply_kda_state_gradient_summary,
)
from hyper_parallel.components.functional._triton.kimi_delta_attention.state_summary import (
    kda_split_state_summary_kernel,
    kda_state_transition_kernel,
    kda_state_grad_ext_kernel,
)
from stateful_heads import StatePackets
from stateful_kda import prepare, finish_forward, prepare_backward, dhu, finish_backward
from memory_options import retained_prepared


def summary_range(tensors, start, count):
    """Read full BTHD storage, write one contiguous head range of S and M."""
    key, weight, value, gate = tensors
    batch, tokens, heads, dim = key.shape
    if not 0 <= start < start + count <= heads or dim != 128:
        raise ValueError("Invalid summary head range or dimension")
    state = torch.empty(batch, count, dim, dim, dtype=torch.float32, device=key.device)
    matrix = torch.empty_like(state)
    kda_split_state_summary_kernel[(1, batch * count)](
        key, weight, value, gate, state, matrix, tokens,
        H=count, H_TOTAL=heads, HEAD_OFFSET=start, K=dim, V=dim,
        BT=64, BV=128, OUTPUT_MODE=1, num_warps=2,
    )
    # The optimized M kernel has no HEAD_OFFSET. A per-batch pointer view keeps
    # its full-head token stride while limiting the grid to the selected heads.
    # Output head indices remain [0,count); no token slice is materialized.
    for index in range(batch):
        offset = (index * tokens * heads + start) * dim
        pointers = [x.view(-1)[offset:] for x in (key, weight, gate)]
        kda_state_transition_kernel[(2, count)](
            *pointers, matrix[index], tokens, H=heads, K=dim, BT=64,
            BV=64, num_warps=4, num_stages=2,
        )
    return state, matrix


def gradient_range(tensors, start, count, block_size=128):
    """Read full BTHD storage, produce only this range's reverse summary."""
    query, key, weight, gate, grad_output, grad_value = tensors
    batch, tokens, heads, dim = query.shape
    if not 0 <= start < start + count <= heads or dim != 128 or block_size not in (64, 128):
        raise ValueError("Invalid gradient-summary head range or dimension")
    result = torch.empty(batch, count, dim, dim, dtype=torch.float32, device=query.device)
    kda_state_grad_ext_kernel[(dim // block_size, batch * count)](
        query, key, weight, gate, grad_output, grad_value, result, .125, tokens,
        H=count, H_TOTAL=heads, HEAD_OFFSET=start, K=dim, V=dim,
        BT=64, BV=block_size, num_warps=2,
    )
    return result


class SummaryHeadCP(torch.autograd.Function):
    """One autograd node fixes packet order in both directions."""

    @staticmethod
    def forward(ctx, q, k, v, raw, beta, alog, bias, blocks):
        if blocks < 1 or q.shape[1] % blocks:
            raise ValueError("Summary block count must divide the head count")
        width = q.shape[1] // blocks
        inputs = (q, k, v, raw, beta, alog, bias)
        template = q.new_empty((q.shape[0], width, 128, 128), dtype=torch.float32)
        packets = StatePackets(template, blocks, False, False)
        prepared = prepare(*inputs)
        gate, _, _, weight, value, _, key, *_ = prepared
        summary_inputs = tuple(x.transpose(1, 2).contiguous() for x in (key, weight, value, gate))
        matrices = []
        for block in range(blocks):
            state, matrix = summary_range(summary_inputs, block * width, width)
            matrices.append(matrix)
            packets.enqueue(block, state, matrix, apply_kda_state_summary)
        # Match the baseline lifetime: layout conversions must not survive into
        # full-head native finish/workspace allocation.
        del summary_inputs, state, matrix
        packets.before_finish()
        h0 = torch.cat(packets.receives, dim=1)
        matrix = torch.cat(matrices, dim=1)
        out, _, hidden, value_new = finish_forward(prepared, h0)
        packets.drain()
        ctx.save_for_backward(*inputs, h0, *retained_prepared(prepared), hidden, value_new, matrix)
        ctx.blocks = blocks
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        inputs, h0, prepared = saved[:7], saved[7], saved[8:16]
        hidden, value_new, matrix = saved[16], saved[17], saved[18]
        width = h0.shape[1] // ctx.blocks
        packets = StatePackets(h0[:, :width].contiguous(), ctx.blocks, True, False)
        grad_output = grad_output.contiguous()
        da, dv, dqraw = prepare_backward(prepared[1], value_new, grad_output, hidden)
        gate, _, _, weight, _, query, key, *_ = prepared
        summary_inputs = tuple(x.transpose(1, 2).contiguous()
                               for x in (query, key, weight, gate, grad_output, dv))
        # Receive slots follow wire order, not head order: all ranks send the
        # last head block first in backward, matching the older summary path.
        for packet, block in enumerate(reversed(range(ctx.blocks))):
            start = block * width
            gradient = gradient_range(summary_inputs, start, width)
            packets.enqueue(packet, gradient, matrix[:, start:start + width], apply_kda_state_gradient_summary)
        del summary_inputs, gradient
        packets.before_finish()
        dht = torch.cat(list(reversed(packets.receives)), dim=1)
        dh, _, dvs = dhu(prepared, grad_output, dv, h0, dht)
        result = finish_backward(inputs, prepared, hidden, value_new, dh, dvs, da, dqraw)
        packets.drain()
        return (*result, None)
