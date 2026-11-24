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
"""Optional Ascend C GDN core with compact GVA and native norm/sigmoid."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import local_chunk_metadata


@lru_cache(maxsize=1)
def _load_ops():
    """Keep the optional native runtime out of ordinary framework imports."""
    try:
        ops = importlib.import_module("fla_npu.ops.ascendc")
    except ImportError as error:
        raise RuntimeError("GDN backend='ascendc' requires a compatible fla-npu installation.") from error
    required = {
        "chunk_gated_delta_rule_fwd": (
            "layout", "use_exp2", "cu_seqlens", "chunk_indices", "disable_recompute",
            "use_qk_l2norm_in_kernel", "use_beta_sigmoid_in_kernel", "use_gate_in_kernel",
        ),
        "chunk_gated_delta_rule_bwd": (
            "layout", "use_exp2", "cu_seqlens", "chunk_indices", "use_qk_l2norm_in_kernel",
            "use_beta_sigmoid_in_kernel", "use_gate_in_kernel", "q_rstd", "k_rstd", "beta_raw",
        ),
    }
    for name, parameters in required.items():
        try:
            signature = inspect.signature(getattr(ops, name, None))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"fla-npu lacks a compatible public {name} API.") from error
        missing = set(parameters).difference(signature.parameters)
        if missing:
            raise RuntimeError(f"fla-npu {name} lacks required parameters {sorted(missing)}.")
    return ops


class _AscendCGDN(torch.autograd.Function):
    """Own each invocation's native backward intermediates independently."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                gate: torch.Tensor, beta: torch.Tensor, metadata: dict,
                normalize_qk: bool, sigmoid_beta: bool) -> torch.Tensor:
        """Save only the tensors consumed by the native composite backward.

        Args:
            ctx: Autograd context owning this invocation's saved tensors.
            query: Raw queries when normalize_qk is enabled, otherwise effective queries.
            key: Keys with the same compact head count as query.
            value: Value tensor in the same dense or packed layout.
            gate: Effective log-decay before the chunk cumulative sum.
            beta: Raw logits when sigmoid_beta is enabled, otherwise effective coefficients.
            metadata: Host document boundaries and chunk indices, or an empty dict.
            normalize_qk: Normalize compact Q/K and save native backward auxiliaries.
            sigmoid_beta: Fuse sigmoid and its backward into the native operations.

        Returns:
            Attention output in the input layout.
        """
        scale = query.shape[-1] ** -0.5
        layout = "TND" if metadata else "BSND"
        result = _load_ops().chunk_gated_delta_rule_fwd(
            query, key, value, gate, beta, scale=scale, layout=layout, chunk_size=64,
            initial_state=None, output_final_state=False, disable_recompute=True,
            return_intermediate_states=False, state_v_first=False, use_exp2=False,
            use_qk_l2norm_in_kernel=normalize_qk, use_beta_sigmoid_in_kernel=sigmoid_beta,
            use_gate_in_kernel=False, **metadata,
        )
        # The ten-slot ABI exports normalized Q/K and rstd only when requested.
        if len(result) != 10 or any(not isinstance(result[index], torch.Tensor) for index in (0, 2, 3)):
            raise RuntimeError("fla-npu GDN forward must return ten slots with output, g_cumsum and A tensors.")
        output, _, gate_cumsum, matrix_a, beta_eff, _, q_hat, k_hat, q_rstd, k_rstd = result
        if normalize_qk:
            query, key = q_hat, k_hat
        # Native beta backward requires beta_raw to match FP32 beta_eff. Keep
        # the small scalar logits only; raw Q/K need not survive the forward.
        beta_raw = beta.float() if sigmoid_beta else None
        if sigmoid_beta:
            beta = beta_eff
        ctx.save_for_backward(query, key, value, gate_cumsum, beta, matrix_a, q_rstd, k_rstd, beta_raw)
        ctx.normalize_qk, ctx.sigmoid_beta = normalize_qk, sigmoid_beta
        ctx.native_metadata, ctx.layout, ctx.scale = metadata, layout, scale
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return raw-input gradients, applying enabled native epilogues exactly once.

        Args:
            ctx: Context saved by forward for this invocation.
            grad_output: Gradient of the attention output.

        Returns:
            Gradients for query, key, value, gate and beta; None for metadata and flags.
        """
        query, key, value, gate_cumsum, beta, matrix_a, q_rstd, k_rstd, beta_raw = ctx.saved_tensors
        gradients = _load_ops().chunk_gated_delta_rule_bwd(
            query, key, value, gate_cumsum, beta, matrix_a, grad_output.contiguous(), ctx.scale,
            layout=ctx.layout, chunk_size=64, initial_state=None, dht=None, use_exp2=False,
            use_qk_l2norm_in_kernel=ctx.normalize_qk, q_rstd=q_rstd, k_rstd=k_rstd,
            use_beta_sigmoid_in_kernel=ctx.sigmoid_beta, beta_raw=beta_raw,
            use_gate_in_kernel=False, state_v_first=False, **ctx.native_metadata,
        )
        if len(gradients) != 8 or any(not isinstance(gradient, torch.Tensor) for gradient in gradients[:5]):
            raise RuntimeError("fla-npu GDN backward must return eight slots with all five input gradients.")
        dq, dk, dv, dbeta, dgate, *_ = gradients
        return dq, dk, dv, dgate, dbeta, None, None, None


def chunk_gated_delta_rule_ascendc(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | Sequence[int] | None = None,
) -> tuple[torch.Tensor, None]:
    """Run the optional native GDN local core for dense or packed Ulysses.

    Args:
        query: Normalized or unnormalized BF16 query of shape [B,T,H,128].
        key: BF16 key with the same shape as query.
        value: BF16 values [B,T,HV,128], with HV a positive multiple of H.
        g: Effective FP32 log-decay of shape [B,T,HV], before chunk cumsum.
        beta: Delta coefficients or raw sigmoid logits of shape [B,T,HV].
        use_qk_l2norm_in_kernel: Normalize Q/K in the native prepare stage.
        use_beta_sigmoid_in_kernel: Apply sigmoid and its derivative inside native operators.
        cu_seqlens: Immutable document boundaries supplied as host integers.

    Returns:
        The output with the same shape as value, and None for final state.

    Raises:
        ValueError: Inputs violate the explicit native training contract.
        RuntimeError: The optional native API or device is unsupported.
    """
    if query.ndim != 4:
        raise ValueError("GDN inputs must have shape [B,T,H,K].")
    if cu_seqlens is not None and query.shape[0] != 1:
        raise ValueError("Packed Ascend C GDN requires physical batch size one.")
    metadata = local_chunk_metadata(cu_seqlens, query.shape[1])
    return _AscendCGDN.apply(
        query, key, value, g, beta, metadata, use_qk_l2norm_in_kernel, use_beta_sigmoid_in_kernel), None
