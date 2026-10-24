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
"""Isolated dense AscendC KDA staged prototype, not a published backend.

Requires the separately built FP32-Dhu experimental wheel. Inputs are BNSD,
Q/K already normalized and beta already activated. No silent dtype fallback.
"""
import ctypes as c

import torch
from torch.autograd.function import once_differentiable
from fla_npu.ops.ascendc import _aclnn_ctypes as api
from memory_options import backward_blocks, retained_prepared


def nd(ctx, value, name):
    return ctx.tensor(value, name, acl_format_override=api.ACL_FORMAT_ND,
                      storage_shape_override=None if value is None else tuple(value.shape))


def prepare(q, k, v, raw, beta, alog, bias):
    if q.ndim != 4 or q.shape[-1] != 128 or q.shape[2] == 0 or q.shape[2] % 64:
        raise ValueError("Prototype requires dense BNSD, D=128 and nonempty T divisible by 64")
    if any(x.shape != q.shape for x in (k, v, raw)) or beta.shape != q.shape[:3]:
        raise ValueError("Prototype requires identical Q/K/V/gate shapes and beta [B,H,T]")
    if alog.shape != (q.shape[1],) or bias.shape != (q.shape[1], 128):
        raise ValueError("Prototype requires A_log [H] and dt_bias [H,128]")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v, raw, beta)):
        raise TypeError("Prototype supports BF16 token inputs only")
    if any(x.dtype != torch.float32 for x in (alog, bias)):
        raise TypeError("Prototype requires FP32 gate parameters")
    if q.device.type != "npu" or any(x.device != q.device or not x.is_contiguous()
                                    for x in (q, k, v, raw, beta, alog, bias)):
        raise ValueError("Prototype requires contiguous inputs on one NPU")
    return api.npu_chunk_kda_fwd_prepare(
        q, k, v, raw, beta, 0.125, layout="BNSD", chunk_size=64,
        A_log=alog, dt_bias=bias.flatten(), safe_gate=True, lower_bound=-5.,
        use_gate_in_kernel=True, use_exp2=True, backward_mode="save")


def finish_forward(prepared, h0):
    gk, aqk, _, w, u, _, kg, qgs, *_ = prepared
    if (h0.shape != (u.shape[0], u.shape[1], 128, 128) or h0.dtype != torch.float32
            or h0.device != u.device or not h0.is_contiguous()):
        raise ValueError("Prototype requires explicit contiguous FP32 K-first h0 on the token device")
    h, vn, ht = api.npu_chunk_fwd_h(kg, w, u, gk=gk, initial_state=h0,
                                  output_final_state=True, use_exp2=True)
    out = api.npu_chunk_kda_fwd_finalize(qgs, aqk, vn, h, output_layout="BNSD")
    return out, ht, h, vn


def prepare_backward(aqk, vn, do, h):
    da = torch.empty_like(aqk, dtype=torch.float32)
    dv = torch.empty_like(vn)
    dqraw = torch.empty_like(do, dtype=torch.float32)

    def args(ctx):
        return ([nd(ctx, t, n) for t, n in zip((aqk, vn, do, h), ("aqk", "vn", "do", "h"))]
                + [ctx.int_array(None), ctx.int_array(None), c.c_double(.125), c.c_int64(64), c.c_bool(False)]
                + [nd(ctx, t, n) for t, n in zip((da, dv, dqraw), ("da", "dv", "dqraw"))])
    return api._call_aclnn("aclnnChunkKdaBwdPrepare", args, (da, dv, dqraw))


def dhu(prepared, do, dv, h0, dht):
    gk, _, _, w, _, qg, kg, *_ = prepared
    assert h0.dtype == torch.float32
    assert dht is None or dht.dtype == torch.float32
    return api.npu_chunk_gated_delta_rule_bwd_dhu(
        qg, kg, w, do, dv, .125, 64, gK=gk, h0=h0, dht=dht, use_exp2=True)


def _finish_backward_full(inputs, prepared, h, vn, dh, dvs, da, dqraw):
    q, k, v, raw, beta, alog, bias = inputs
    gk, _, akk, *_ = prepared
    outputs = (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v),
               torch.empty_like(beta, dtype=torch.float32), torch.empty_like(raw, dtype=torch.float32),
               torch.empty_like(alog), torch.empty_like(bias))
    # A2 Finalize requires FP32 dBeta; ACLNN requires beta/dBeta matching.
    # Widen the already-quantized beta, without changing its numeric values.
    tensors = (q, k, v, gk, raw.float(), beta.float(), alog, bias, akk, vn, h, dh, dvs, da, dqraw, None, None)

    def args(ctx):
        return ([nd(ctx, t, f"in{i}") for i, t in enumerate(tensors)]
                + [ctx.int_array(None), ctx.int_array(None), c.c_double(.125), c.c_double(-5.),
                   c.c_int64(64), c.c_bool(True), c.c_bool(True), c.c_bool(True), c.c_bool(False)]
                + [nd(ctx, t, f"out{i}") for i, t in enumerate(outputs)])
    dq, dk, dv, db, dg, dalog, dbias = api._call_aclnn("aclnnChunkKdaBwdFinalize", args, outputs)
    return dq, dk, dv, dg.to(raw.dtype), db.to(beta.dtype), dalog, dbias


def finish_backward(inputs, prepared, h, vn, dh, dvs, da, dqraw):
    blocks = backward_blocks()
    if blocks == 1:
        return _finish_backward_full(inputs, prepared, h, vn, dh, dvs, da, dqraw)
    heads = inputs[0].shape[1]
    if heads % blocks:
        raise ValueError("Backward finish block count must divide the local head count")
    if h.ndim != 5 or h.shape[2] != heads or dh.shape != h.shape:
        raise ValueError("Native chunk states must use [B,chunk,H,K,V] layout")
    width, results = heads // blocks, []
    for start in range(0, heads, width):
        def part(value):
            return value[:, start:start + width].contiguous()
        xs = tuple(part(value) if i < 5 else value[start:start + width].contiguous()
                   for i, value in enumerate(inputs))
        # Finalize reads only gk and Akk from the prepared representation.
        p = (part(prepared[0]), None, part(prepared[2]))
        hidden = h[:, :, start:start + width].contiguous()
        d_hidden = dh[:, :, start:start + width].contiguous()
        results.append(_finish_backward_full(xs, p, hidden, part(vn), d_hidden,
                                             part(dvs), part(da), part(dqraw)))
    return tuple(torch.cat([result[i] for result in results], dim=1 if i < 5 else 0) for i in range(7))


class LocalKDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, raw, beta, alog, bias, h0):
        inputs = (q, k, v, raw, beta, alog, bias)
        prepared = prepare(*inputs)
        out, ht, h, vn = finish_forward(prepared, h0)
        ctx.save_for_backward(*inputs, h0, *retained_prepared(prepared), h, vn)
        ctx.set_materialize_grads(False)
        return out, ht

    @staticmethod
    @once_differentiable
    def backward(ctx, do, dht):
        saved = ctx.saved_tensors
        inputs, h0, prepared, h, vn = saved[:7], saved[7], saved[8:16], saved[16], saved[17]
        if do is None:
            do = torch.zeros_like(inputs[2])
        do = do.contiguous()
        da, dv, dqraw = prepare_backward(prepared[1], vn, do, h)
        dh, dh0, dvs = dhu(prepared, do, dv, h0, None if dht is None else dht.contiguous())
        return (*finish_backward(inputs, prepared, h, vn, dh, dvs, da, dqraw), dh0)


def cpu_reference(q, k, v, raw, beta, alog, bias, h0):
    """Independent FP64 token recurrence; avoids importing Hyper or FLA math."""
    gate = -5 * torch.sigmoid(alog.exp()[None, :, None, None] * (raw + bias[None, :, None, :]))
    state, outputs = h0, []
    for i in range(q.shape[2]):
        state = state * gate[:, :, i].exp().unsqueeze(-1)
        ki = k[:, :, i]
        residual = v[:, :, i] - (ki.unsqueeze(-1) * state).sum(-2)
        state = state + (ki * beta[:, :, i, None]).unsqueeze(-1) * residual.unsqueeze(-2)
        outputs.append((q[:, :, i, :, None] * state).sum(-2) * .125)
    return torch.stack(outputs, dim=2), state


class StateCP(torch.autograd.Function):
    """AscendC local stages plus existing Hyper summary and exchange kernels.

Only chronological WORLD is supported by this diagnostic prototype. Each call
owns its send buffers and drains Work handles before returning, including replay.
"""

    @staticmethod
    def forward(ctx, q, k, v, raw, beta, alog, bias, protocol):
        import torch.distributed as dist
        from hyper_parallel.components.functional.kimi_delta_attention_state_summary import (
            kda_state_summary_forward_from_prepared, apply_kda_state_summary)
        from hyper_parallel.distributed.context_parallel.kimi_delta_attention_state import KDAStateAllGather
        rank, size = dist.get_rank(), dist.get_world_size()
        if protocol not in ("p2p", "allgather"):
            raise ValueError(protocol)
        inputs = (q, k, v, raw, beta, alog, bias)
        p = prepare(*inputs)
        gk, _, _, w, u, _, kg, *_ = p
        s, matrix = kda_state_summary_forward_from_prepared(
            *(x.transpose(1, 2).contiguous() for x in (kg, w, u, gk)), chunk_size=64)
        exchange = KDAStateAllGather(dist.group.WORLD, rank, size) if protocol == "allgather" else None
        h0 = torch.zeros_like(s)
        send, sent = None, None
        if exchange is not None:
            h0 = exchange.forward(s, matrix)
        else:
            if rank:
                dist.irecv(h0, src=rank-1).wait()
            if rank < size-1:
                sent = apply_kda_state_summary(s, matrix, h0).contiguous()
                send = dist.isend(sent, dst=rank+1)
        out, _, h, vn = finish_forward(p, h0)
        if send is not None:
            send.wait()
        ctx.save_for_backward(*inputs, h0, *retained_prepared(p), h, vn, matrix)
        ctx.exchange, ctx.rank, ctx.size = exchange, rank, size
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        import torch.distributed as dist
        from hyper_parallel.components.functional.kimi_delta_attention_state_summary import (
            kda_state_gradient_summary_from_prepared, apply_kda_state_gradient_summary)
        xs = ctx.saved_tensors
        inputs, h0, p, h, vn, matrix = xs[:7], xs[7], xs[8:16], xs[16], xs[17], xs[18]
        do = do.contiguous()
        da, dv, dqraw = prepare_backward(p[1], vn, do, h)
        gk, _, _, w, _, qg, kg, *_ = p
        gradient = kda_state_gradient_summary_from_prepared(
            *(x.transpose(1, 2).contiguous() for x in (qg, kg, w, gk, do, dv)),
            .125, chunk_size=64)
        dht = torch.zeros_like(h0)
        send, sent = None, None
        if ctx.exchange is not None:
            dht = ctx.exchange.backward(gradient, matrix)
        else:
            if ctx.rank < ctx.size-1:
                dist.irecv(dht, src=ctx.rank+1).wait()
            if ctx.rank:
                sent = apply_kda_state_gradient_summary(gradient, matrix, dht).contiguous()
                send = dist.isend(sent, dst=ctx.rank-1)
        dh, _, dvs = dhu(p, do, dv, h0, dht)
        result = finish_backward(inputs, p, h, vn, dh, dvs, da, dqraw)
        if send is not None:
            send.wait()
        return (*result, None)
