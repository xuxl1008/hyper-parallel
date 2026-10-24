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
"""Optional fla-npu Ascend C local KDA backend for dense and packed Ulysses training."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import torch
from torch.autograd.function import once_differentiable


def _load_ops():
    """Keep the optional native runtime out of ordinary framework imports."""
    try:
        ops = importlib.import_module("fla_npu.ops.ascendc")
    except ImportError as error:
        raise RuntimeError("backend='ascendc' requires a compatible fla-npu installation.") from error
    required = ("chunk_kda_fwd_prepare", "chunk_fwd_h", "chunk_kda_fwd_finalize", "chunk_kda_bwd")
    if any(not callable(getattr(ops, name, None)) for name in required):
        raise RuntimeError("fla-npu lacks the public KDA prepare/state/finalize/backward APIs.")
    return ops


def _forward_kda(inputs, scale, lower_bound, metadata):
    """Run native stages, returning only the intermediates needed by autograd."""
    ops = _load_ops()
    query, key, value, gate, beta, a_log, dt_bias = inputs
    prepared = list(ops.chunk_kda_fwd_prepare(
        query, key, value, gate, beta, scale, layout="BNSD", chunk_size=64,
        safe_gate=True, lower_bound=lower_bound, use_gate_in_kernel=True,
        A_log=a_log, dt_bias=dt_bias.flatten(), epsilon=1e-6,
        use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        use_exp2=True, backward_mode="save", **metadata,
    ))
    gk, aqk, _, weights, residual, _, kg, qg_scaled, *_ = prepared
    hidden, v_new, _ = ops.chunk_fwd_h(
        kg, weights, residual, gk=gk, output_final_state=False, use_exp2=True, **metadata,
    )
    output = ops.chunk_kda_fwd_finalize(
        qg_scaled, aqk, v_new, hidden, output_layout="BSND", **metadata)
    # These buffers have no backward consumer and must not live across layers.
    prepared[4] = prepared[7] = None
    return output, prepared, hidden, v_new


@lru_cache(maxsize=1)
def _load_epilogue_ops():
    """Load optional FLA helpers without requiring its staged KDA backend."""
    try:
        importlib.import_module("triton.backends.ascend")
        norm = importlib.import_module("fla.modules.l2norm").l2norm_bwd
        sigmoid = importlib.import_module("fla.ops.common.gate").fused_beta_sigmoid_bwd
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "backend='ascendc' requires FLA with Triton-Ascend L2Norm and beta-sigmoid backward helpers."
        ) from error
    if not callable(norm) or not callable(sigmoid):
        raise RuntimeError("The installed FLA normalization/sigmoid backward APIs are incompatible.")
    return norm, sigmoid


class _AscendCKDA(torch.autograd.Function):
    """Own all native activations per invocation, including checkpoint replay."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                gate: torch.Tensor, beta: torch.Tensor, a_log: torch.Tensor,
                dt_bias: torch.Tensor, scale: float, lower_bound: float, metadata: dict) -> torch.Tensor:
        """Save native intermediates without retaining forward-only workspaces."""
        _load_epilogue_ops()
        inputs = (query, key, value, gate, beta, a_log, dt_bias)
        output, prepared, hidden, v_new = _forward_kda(inputs, scale, lower_bound, metadata)
        # FLA re-evaluates sigmoid from logits; norm backward uses native Q/K
        # and rstd auxiliaries, so the much larger raw Q/K need not be retained.
        ctx.save_for_backward(value, gate, beta, a_log, dt_bias, *prepared, hidden, v_new)
        ctx.dtypes = tuple(tensor.dtype for tensor in inputs)
        ctx.scale, ctx.lower_bound, ctx.native_metadata = scale, lower_bound, metadata
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Apply native local backward and the explicit raw-input chain rule."""
        value, gate, beta_raw, a_log, dt_bias, *saved = ctx.saved_tensors
        (gk, aqk, akk, weights, _, qg, kg, _, query, key,
         query_rstd, key_rstd, beta, hidden, v_new) = saved
        packed = bool(ctx.native_metadata)
        # Native whole backward uses NTD/chunk-major views for varlen,
        # while Prepare/FwdH keep the B=1 BNSD views of the same storage.
        def backward_view(tensor: torch.Tensor) -> torch.Tensor:
            """Expose the rank required by the native packed backward ABI."""
            return tensor.squeeze(0) if packed else tensor

        operands = (query, key, value, beta, gk, aqk, akk, weights, qg, kg, v_new, hidden,
                    grad_output.transpose(1, 2).contiguous())
        dq, dk, dv, dbeta, dgate, _, da_log, ddt_bias = _load_ops().chunk_kda_bwd(
            *(backward_view(tensor) for tensor in operands), ctx.scale,
            raw_g=backward_view(gate), A_log=a_log, dt_bias=dt_bias,
            initial_state=None, dht=None, chunk_size=64,
            safe_gate=True, lower_bound=ctx.lower_bound, use_gate_in_kernel=True,
            disable_recompute=True, use_exp2=True, state_v_first=False,
            implementation="legacy", **ctx.native_metadata,
        )
        if packed:
            dq, dk, dv, dbeta, dgate = (tensor.unsqueeze(0) for tensor in (dq, dk, dv, dbeta, dgate))
        norm_backward, sigmoid_backward = _load_epilogue_ops()
        # Keep native FP32 gradients until the helpers store raw-input grads.
        dq = norm_backward(query, query_rstd, dq)
        dk = norm_backward(key, key_rstd, dk)
        dbeta = sigmoid_backward(beta_raw, dbeta)
        gradients = (dq, dk, dv, dgate, dbeta, da_log, ddt_bias.reshape_as(dt_bias))
        return (*(grad.to(dtype) for grad, dtype in zip(gradients, ctx.dtypes)), None, None, None)


def _packed_metadata(cu_seqlens, tokens):
    """Snapshot validated host offsets and derive the canonical chunk order."""
    if cu_seqlens is None:
        return {}
    if isinstance(cu_seqlens, torch.Tensor):
        if (cu_seqlens.device.type != "cpu" or cu_seqlens.ndim != 1
                or cu_seqlens.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Ascend C cu_seqlens must be a one-dimensional CPU integer tensor or sequence.")
        cu_seqlens = cu_seqlens.tolist()
    offsets = tuple(cu_seqlens)
    if (len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != tokens
            or any(not isinstance(x, int) or isinstance(x, bool) for x in offsets)
            or any(begin >= end for begin, end in zip(offsets, offsets[1:]))):
        raise ValueError("Ascend C cu_seqlens must strictly increase from zero to the packed token count.")
    indices = tuple(value for seq, (begin, end) in enumerate(zip(offsets, offsets[1:]))
                    for chunk in range((end - begin + 63) // 64) for value in (seq, chunk))
    return {"cu_seqlens": offsets, "chunk_indices": indices}


def ascendc_chunk_kda(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    gate: torch.Tensor, beta: torch.Tensor, *, a_log: torch.Tensor,
    dt_bias: torch.Tensor, scale: float | None = None, lower_bound: float = -5.0,
    chunk_size: int = 64, cu_seqlens: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Run dense or packed KDA with native attention and FLA Triton epilogues.

    Args:
        query: Raw BF16 queries [B,T,H,128] or packed [T,H,128].
        key: Raw BF16 keys with the same shape.
        value: BF16 values with the same shape.
        gate: BF16 or FP32 gate logits with the same shape.
        beta: BF16 or FP32 beta logits [B,T,H] or packed [T,H].
        a_log: FP32 decay parameters with H elements.
        dt_bias: FP32 gate bias with H*128 elements.
        scale: Positive attention scale, default 1/sqrt(128).
        lower_bound: Safe gate lower bound in [-5,0).
        chunk_size: Must be 64 for this native kernel family.
        cu_seqlens: Strictly increasing host offsets for packed inputs, including
            zero and T. Packed rank-four inputs require B=1. Documents need not
            be multiples of 64; zero-length documents are rejected.

    Returns:
        BF16 output with the query shape. External recurrent states and double
        backward are not exposed. A2 native backward handles partial chunks
        through its public per-document padding path.
    """
    if query.ndim == 3:
        if cu_seqlens is None:
            raise ValueError("Packed TND inputs require cu_seqlens.")
        return ascendc_chunk_kda(
            *(tensor.unsqueeze(0) for tensor in (query, key, value, gate, beta)),
            a_log=a_log, dt_bias=dt_bias, scale=scale, lower_bound=lower_bound,
            chunk_size=chunk_size, cu_seqlens=cu_seqlens).squeeze(0)
    if query.ndim != 4 or query.shape[-1] != 128 or min(query.shape) <= 0:
        raise ValueError("Ascend C KDA requires nonempty dense [B,T,H,128] inputs.")
    metadata = _packed_metadata(cu_seqlens, query.shape[1])
    if metadata and query.shape[0] != 1:
        raise ValueError("Packed Ascend C KDA requires B=1.")
    if (not metadata and query.shape[1] % 64) or query.shape[2] > 128 or chunk_size != 64:
        raise ValueError("Ascend C KDA requires H<=128, chunk_size=64 and dense T divisible by 64.")
    if any(tensor.shape != query.shape or tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        raise ValueError("Ascend C KDA requires equal BF16 Q/K/V shapes.")
    if (gate.shape != query.shape or beta.shape != query.shape[:-1]
            or any(tensor.dtype not in (torch.bfloat16, torch.float32) for tensor in (gate, beta))):
        raise ValueError("Ascend C KDA requires matching BF16/FP32 gate and beta logits.")
    if (a_log is None or dt_bias is None or a_log.dtype != torch.float32 or dt_bias.dtype != torch.float32
            or a_log.numel() != query.shape[2] or dt_bias.numel() != query.shape[2] * 128):
        raise ValueError("Ascend C KDA requires FP32 a_log[H] and dt_bias[H,128].")
    tensors = (query, key, value, gate, beta, a_log, dt_bias)
    if query.device.type != "npu" or any(tensor.device != query.device for tensor in tensors):
        raise ValueError("Ascend C KDA requires all tensors on the same NPU.")
    scale = 128 ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0 or lower_bound is None or not -5 <= lower_bound < 0:
        raise ValueError("Ascend C KDA requires a positive finite scale and lower_bound in [-5,0).")
    tokens = tuple(tensor.transpose(1, 2).contiguous() for tensor in tensors[:5])
    # A head slice may be contiguous while retaining the full-head NPU storage
    # descriptor. Native tiling needs a descriptor for precisely this shard.
    return _AscendCKDA.apply(*tokens, a_log.reshape(-1).clone(),
                            dt_bias.reshape(-1, 128).clone(), scale, lower_bound, metadata)
