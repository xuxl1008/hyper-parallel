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
"""Current FLA staged contract, separated solely for isolated schedule experiments."""

from dataclasses import dataclass

import torch

from hyper_parallel.components.functional.kimi_delta_attention_fla_adapter import get_fla_kda_staged_ops


_SCALE = 128 ** -0.5


@dataclass
class Prepared:
    """Forward tensors; only ``saved`` plus M/h0 survive into autograd."""

    saved: tuple
    gate: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    kg: torch.Tensor


@dataclass
class BackwardPrepared:
    """Local backward work independent of incoming remote dHT."""

    saved: tuple
    do: torch.Tensor
    beta: torch.Tensor
    gate: torch.Tensor
    w: torch.Tensor
    qg: torch.Tensor
    kg: torch.Tensor
    h: torch.Tensor
    vn: torch.Tensor
    da: torch.Tensor
    dv: torch.Tensor


def prepare(inputs: tuple) -> Prepared:
    """Normalize and prepare local tokens using the same calls as formal P2P."""
    ops = get_fla_kda_staged_ops()
    raw_q, raw_k, value, raw_gate, raw_beta, alog, bias = inputs
    query, qrstd = ops.l2norm_fwd(raw_q)
    key, krstd = ops.l2norm_fwd(raw_k)
    beta = ops.fused_beta_sigmoid(raw_beta)
    gate = ops.kda_gate_chunk_cumsum(g=raw_gate, A_log=alog, dt_bias=bias.flatten(),
                                    scale=1.4426950408889634, chunk_size=64, lower_bound=-5.)
    w, u, _, kg, aqk, akk = ops.chunk_kda_fwd_intra(
        q=query, k=key, v=value, gk=gate, beta=beta, scale=_SCALE, chunk_size=64,
        safe_gate=True, disable_recompute=True)
    saved = (query, qrstd, key, krstd, value, raw_gate, raw_beta, alog, bias, aqk, akk)
    return Prepared(saved, gate, w, u, kg)


def full_summary(stage: Prepared) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce full S/M exactly once."""
    # Optional Triton: do not load it for the CPU protocol oracle.
    from hyper_parallel.components.functional.kimi_delta_attention_state_summary import (
        kda_state_summary_forward_from_prepared,
    )
    return kda_state_summary_forward_from_prepared(stage.kg, stage.w, stage.u, stage.gate, chunk_size=64)


def finish_forward(stage: Prepared, incoming: torch.Tensor) -> torch.Tensor:
    """Finish local recurrence/output once the boundary state is ready."""
    ops = get_fla_kda_staged_ops()
    h, vn, _ = ops.chunk_gated_delta_rule_fwd_h(
        k=stage.kg, w=stage.w, u=stage.u, gk=stage.gate, initial_state=incoming,
        output_final_state=False, chunk_size=64)
    return ops.chunk_gla_fwd_o_gk(q=stage.saved[0], v=vn, g=stage.gate, A=stage.saved[9],
                                 h=h, scale=_SCALE, chunk_size=64)


def prepare_backward(saved: tuple, grad: torch.Tensor) -> BackwardPrepared:
    """Recompute local tensors, h/v and dA/dv before waiting for remote dHT."""
    ops = get_fla_kda_staged_ops()
    query, _, key, _, value, raw_gate, raw_beta, alog, bias, aqk, akk, _, h0 = saved
    grad = grad.contiguous()
    beta = ops.fused_beta_sigmoid(raw_beta)
    gate = ops.kda_gate_chunk_cumsum(g=raw_gate, A_log=alog, dt_bias=bias.flatten(),
                                    scale=1.4426950408889634, chunk_size=64, lower_bound=-5.)
    w, u, qg, kg = ops.recompute_w_u_fwd(q=query, k=key, v=value, beta=beta, A=akk, gk=gate)
    h, vn, _ = ops.chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=gate, initial_state=h0, output_final_state=False, chunk_size=64)
    da, dv = ops.chunk_kda_bwd_dav(q=query, k=key, v=vn, do=grad, A=aqk, scale=_SCALE, chunk_size=64)
    return BackwardPrepared(saved, grad, beta, gate, w, qg, kg, h, vn, da, dv)


def gradient_summary(stage: BackwardPrepared) -> torch.Tensor:
    """Produce the full G offset once, independent of remote dHT."""
    # Optional Triton: do not load it for the CPU protocol oracle.
    from hyper_parallel.components.functional.kimi_delta_attention_state_summary import (
        kda_state_gradient_summary_from_prepared,
    )
    return kda_state_gradient_summary_from_prepared(
        stage.qg, stage.kg, stage.w, stage.gate, stage.do, stage.dv, _SCALE, chunk_size=64)


def finish_backward(stage: BackwardPrepared, incoming: torch.Tensor) -> tuple:
    """Run the local full-gradient chain with nonzero remote dHT."""
    ops = get_fla_kda_staged_ops()
    query, qrstd, key, krstd, value, raw_gate, raw_beta, alog, bias, _, akk, _, h0 = stage.saved
    dh, _, dv = ops.chunk_gated_delta_rule_bwd_dhu(
        q=stage.qg, k=stage.kg, w=stage.w, gk=stage.gate, h0=h0, dht=incoming,
        do=stage.do, dv=stage.dv, scale=_SCALE, chunk_size=64)
    dq, dk, dv, db, dg, dakk = ops.chunk_kda_bwd_wy_dqkg_fused(
        q=query, k=key, v=value, v_new=stage.vn, g=stage.gate, beta=stage.beta, A=akk,
        h=stage.h, do=stage.do, dh=dh, dv=dv, scale=_SCALE, chunk_size=64)
    dq, dk, db, dg = ops.chunk_kda_bwd_intra(
        q=query, k=key, g=stage.gate, beta=stage.beta, dAqk=stage.da, dAkk=dakk,
        dq=dq, dk=dk, db=db, dg=dg, chunk_size=64, safe_gate=True)
    dg = ops.chunk_local_cumsum(dg, chunk_size=64, reverse=True)
    dg, da, dbias = ops.kda_gate_bwd(g=raw_gate, A_log=alog, dt_bias=bias.flatten(), dyg=dg, lower_bound=-5.)
    db = ops.fused_beta_sigmoid_bwd(raw_beta, db)
    return (ops.l2norm_bwd(query, qrstd, dq), ops.l2norm_bwd(key, krstd, dk), dv, dg, db,
            da.reshape_as(alog), dbias.reshape_as(bias))


def transition_once(stage: Prepared) -> torch.Tensor:
    """Reuse the production FP32 transition kernel, independent of V blocks."""
    # Optional Triton dependency: imported only inside the NPU experiment path.
    from hyper_parallel.components.functional._triton.kimi_delta_attention.state_summary import (
        kda_state_transition_kernel,
    )
    batch, tokens, heads, dim = stage.kg.shape
    matrix = stage.kg.new_empty((batch, heads, dim, dim), dtype=torch.float32)
    kda_state_transition_kernel[(2, batch * heads)](
        stage.kg, stage.w, stage.gate, matrix, tokens, H=heads, K=dim, BT=64, BV=64,
        num_warps=4, num_stages=2)
    return matrix


def value_summary(stage: Prepared, start: int, width: int) -> torch.Tensor:
    """Run an S producer per V block; its U packing cost is not hidden."""
    # Reuse existing kernel arithmetic, but explicitly repack the narrow V layout.
    from hyper_parallel.components.functional._triton.kimi_delta_attention.state_summary import (
        kda_split_state_summary_kernel,
    )
    batch, tokens, heads, dim = stage.kg.shape
    values = stage.u[..., start:start + width].contiguous()
    state = values.new_empty((batch, heads, dim, width), dtype=torch.float32)
    kda_split_state_summary_kernel[(1, batch * heads)](
        stage.kg, stage.w, values, stage.gate, state, state, tokens,
        H=heads, H_TOTAL=heads, HEAD_OFFSET=0, K=dim, V=width, BT=64, BV=width,
        OUTPUT_MODE=1, num_warps=2)
    return state


def value_gradient(stage: BackwardPrepared, start: int, width: int) -> torch.Tensor:
    """Run G per V block; the K/W/sequence scan is repeated for each block."""
    from hyper_parallel.components.functional._triton.kimi_delta_attention.state_summary import (
        kda_state_grad_ext_kernel,
    )
    batch, tokens, heads, dim = stage.kg.shape
    grad = stage.do[..., start:start + width].contiguous()
    dv = stage.dv[..., start:start + width].contiguous()
    state = grad.new_empty((batch, heads, dim, width), dtype=torch.float32)
    kda_state_grad_ext_kernel[(1, batch * heads)](
        stage.qg, stage.kg, stage.w, stage.gate, grad, dv, state, _SCALE, tokens,
        H=heads, H_TOTAL=heads, HEAD_OFFSET=0, K=dim, V=width, BT=64, BV=width, num_warps=2)
    return state
