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
"""Experimental stateless KDA training adapter for external FlashServe AscendC.

No kernel copy is required. Fused normalization optionally imports fla_npu's
Triton implementation. State CP is deliberately not enabled: native terminal-state
gradients need an independently validated path.
"""
# pylint: disable=forbidden-backend-import
from __future__ import annotations

__all__ = ["ascendc_chunk_kda"]

import importlib
import math
from typing import Any

import torch
from torch.autograd.function import once_differentiable


def _load_ops():
    """Load the optional runtime only after the caller has selected AscendC."""
    try:
        ops = importlib.import_module("fla_npu.ops.ascendc")
    except ImportError as error:
        raise RuntimeError(
            "KDA backend='ascendc' requires a separately installed FlashServe "
            "fla_npu wheel built for this CANN/SoC; no fallback is performed."
        ) from error
    names = ("chunk_kda_fwd", "chunk_kda_bwd")
    if not all(callable(getattr(ops, name, None)) for name in names):
        raise RuntimeError("fla_npu is missing the required KDA forward/backward API.")
    return ops


def _normalize_qk(q: torch.Tensor, k: torch.Tensor, backend: str) -> tuple:
    """Keep the Torch baseline or explicitly select external fused L2Norm."""
    if backend == "torch":
        return tuple(x * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6).to(x.dtype)
                     for x in (q, k))
    if backend != "triton":
        raise ValueError("AscendC KDA normalization_backend must be 'torch' or 'triton'.")
    # The external backward schedules complete 16-row tiles only. The dense
    # T%64 contract satisfies this, but reject other callers instead of skipping rows.
    if any(x.shape[-1] != 128 or x.numel() // 128 % 16 for x in (q, k)):
        raise ValueError("fla_npu fused L2Norm requires D=128 and a multiple of 16 rows.")
    try:
        # Keep the optional Triton runtime out of ordinary Hyper imports.
        l2norm = importlib.import_module("fla_npu.ops.triton").l2norm
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Fused normalization requires fla_npu.ops.triton.l2norm; no fallback is performed.") from error
    return tuple(l2norm(x.contiguous(), eps=1e-6) for x in (q, k))


def _validate_inputs(q, k, v, gate, beta, a_log, dt_bias, scale, lower_bound, chunk_size):
    """Restrict the first training milestone to dense BF16 KDA with equal heads."""
    tensors = (q, k, v, gate, beta, a_log, dt_bias)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("AscendC KDA requires q/k/v/gate/beta/a_log/dt_bias tensors.")
    if any(tensor.ndim != 4 for tensor in (q, k, v, gate)) or beta.ndim != 3:
        raise ValueError("AscendC KDA expects BSND inputs and BSN beta.")
    batch, length, heads, width = q.shape
    if not (
        batch > 0 and length > 0 and 0 < heads <= 128
        and width == 128 and k.shape == v.shape == gate.shape == q.shape
        and beta.shape == q.shape[:3] and chunk_size == 64 and length % 64 == 0
        and a_log.numel() == heads and dt_bias.numel() == heads * width
    ):
        raise ValueError("AscendC KDA requires equal heads, Dk=Dv=128 and T%64=0.")
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, gate, beta)):
        raise ValueError("AscendC KDA token inputs must use BF16.")
    if a_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        raise ValueError("AscendC KDA a_log and dt_bias must use FP32.")
    if q.device.type != "npu" or any(tensor.device != q.device for tensor in tensors):
        raise ValueError("AscendC KDA requires all inputs on the same NPU.")
    if lower_bound is None or not -5.0 <= lower_bound < 0.0:
        raise ValueError("AscendC KDA requires a safe lower_bound in [-5,0).")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("AscendC KDA scale must be positive and finite.")


class _AscendCKDA(torch.autograd.Function):
    """Bridge the explicitly staged, non-autograd AscendC local operators."""

    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                gate: torch.Tensor, beta: torch.Tensor, a_log: torch.Tensor,
                dt_bias: torch.Tensor, scale: float, lower_bound: float) -> torch.Tensor:
        """Save the external operator's intermediates for stateless backward."""
        result = _load_ops().chunk_kda_fwd(
            q, k, v, gate, beta, scale, 64, layout="BNSD",
            safe_gate=True, lower_bound=lower_bound, use_gate_in_kernel=True,
            A_log=a_log, dt_bias=dt_bias.flatten(), disable_recompute=True,
            return_intermediate_states=True, use_exp2=True,
            use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False,
            initial_state=None, output_final_state=False, state_v_first=False,
        )
        if len(result) != 12:
            raise RuntimeError("Incompatible AscendC KDA forward: expected 12 outputs.")
        output, _, gk, aqk, akk, w, _, qg, kg, v_new, h, _ = result
        expected_shape = (q.shape[0], q.shape[2], q.shape[1], v.shape[-1])
        if tuple(output.shape) != expected_shape:
            raise RuntimeError("Incompatible AscendC KDA forward output: expected BSND.")
        ctx.save_for_backward(q, k, v, gate, beta, a_log, dt_bias,
                              gk, aqk, akk, w, qg, kg, v_new, h)
        ctx.scale, ctx.lower_bound = scale, lower_bound
        # The whole forward returns BSND even when its input layout is BNSD.
        return output.transpose(1, 2)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return token and gate-parameter gradients in the caller's dtypes."""
        (q, k, v, gate, beta, a_log, dt_bias,
         gk, aqk, akk, w, qg, kg, v_new, h) = ctx.saved_tensors
        result = _load_ops().chunk_kda_bwd(
            q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h,
            grad_output.contiguous(), ctx.scale, raw_g=gate,
            A_log=a_log, dt_bias=dt_bias.reshape(q.shape[1], q.shape[-1]),
            initial_state=None, dht=None, chunk_size=64,
            safe_gate=True, lower_bound=ctx.lower_bound, use_gate_in_kernel=True,
            disable_recompute=True, use_exp2=True, state_v_first=False,
            implementation="legacy",
        )
        if len(result) != 8:
            raise RuntimeError("Incompatible AscendC KDA backward: expected 8 outputs.")
        dq, dk, dv, dbeta, dgate, _, da_log, ddt_bias = result
        gradients = (dq, dk, dv, dgate, dbeta, da_log, ddt_bias)
        inputs = (q, k, v, gate, beta, a_log, dt_bias)
        if any(grad is None for grad in gradients):
            raise RuntimeError("AscendC KDA did not return all required training gradients.")
        return (*(grad.reshape_as(tensor).to(tensor.dtype)
                  for grad, tensor in zip(gradients, inputs)), None, None)


def ascendc_chunk_kda(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     gate: torch.Tensor, beta: torch.Tensor, *, a_log: torch.Tensor,
                     dt_bias: torch.Tensor, scale: float | None = None,
                     lower_bound: float = -5.0, chunk_size: int = 64,
                     normalization_backend: str = "torch") -> torch.Tensor:
    """Run stateless dense KDA with explicit normalization and Torch beta autograd.

    Args:
        normalization_backend: ``torch`` retains the original correctness
            baseline; ``triton`` imports fla_npu's fused forward/backward L2Norm.
            Both use eps=1e-6, but their BF16 rounding order differs.

    Note:
        Fused normalization is opt-in until the broader CP validation completes.
        Neither choice enables nonzero terminal-state gradients.
    """
    scale = q.shape[-1] ** -0.5 if scale is None and q.ndim else scale
    scale = float(scale) if scale is not None else float("nan")
    _validate_inputs(q, k, v, gate, beta, a_log, dt_bias, scale, lower_bound, chunk_size)
    q, k = _normalize_qk(q, k, normalization_backend)
    beta = beta.float().sigmoid().to(beta.dtype)
    # BNSD is also the backward API layout: do not retain duplicate BSND copies.
    tokens = tuple(tensor.transpose(1, 2).contiguous() for tensor in (q, k, v, gate, beta))
    output = _AscendCKDA.apply(*tokens, a_log.reshape(-1).contiguous(),
                              dt_bias.reshape(-1).contiguous(), scale, lower_bound)
    return output.transpose(1, 2).contiguous()
