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
"""Opt-in Ring CP for Qwen3-MoE through the existing GQAAttention interface.

The model owns projection, normalization, RoPE and output projection. This
adapter only translates the attention call/return contract and asks the generic
forward rewriter to install it. Packed inputs use explicit global document lengths;
internal task redistribution restores the original sequence ownership.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

# Native Torch is required; the legacy lint rule predates the Torch-only migration.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.npu_fusion_attention import PACKED_SEQUENCE_ARGUMENTS
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.ring_attention import (
    _validate_pruning, _validate_ulysses_degree, ulysses_ring_attention,
)
from hyper_parallel.distributed.recipe_spec import inner_wrapper
from hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel import (
    _validate_qwen3_moe_flash_attention_cp_target,
)


def _validate_ring_call(attention_mask, dropout, kwargs):
    if attention_mask is not None:
        raise ValueError(
            "Qwen3-MoE Ring CP supports implicit causal attention without an explicit mask; "
            "use unpadded sequences, explicit global document_lengths for packing, and attention_mask=None"
        )
    if dropout != 0:
        raise ValueError("Qwen3-MoE Ring CP requires attention dropout=0")
    for name in (*sorted(PACKED_SEQUENCE_ARGUMENTS),
                 "sliding_window", "window_size", "softcap", "alibi_slopes", "attention_bias"):
        if kwargs.get(name) is not None:
            raise ValueError(f"Qwen3-MoE Ring CP does not support {name}")
    if kwargs.get("output_attentions", False):
        raise ValueError("Qwen3-MoE Ring CP does not return attention weights")


def _build_qwen3_moe_ring_interface(cp_mesh, ulysses_degree, overlap, balance_options):
    def attention_interface(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Accept prepared BNSD states and return the BSND output expected by GQA."""
        _validate_ring_call(attention_mask, dropout, kwargs)
        lengths = kwargs.get("document_lengths")
        layout = "BNSD"
        if lengths is not None:
            if query.shape[0] != 1:
                raise ValueError("packed Ring attention requires batch size one")
            query, key, value = (tensor.squeeze(0).transpose(0, 1).contiguous() for tensor in (query, key, value))
            layout = "TND"
        output = ulysses_ring_attention(
            query, key, value, cp_mesh, ulysses_degree=ulysses_degree,
            causal=kwargs.get("is_causal", getattr(module, "is_causal", True)),
            scale=scaling, input_layout=layout, dropout_p=dropout, overlap=overlap,
            document_lengths=lengths, **balance_options,
        )
        return (output.unsqueeze(0) if layout == "TND" else output.transpose(1, 2).contiguous()), None

    return attention_interface


@inner_wrapper
def qwen3_moe_ring_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    *,
    ulysses_degree: int = 1,
    overlap: bool = False,
    load_balance: str = "none",
    prune_documents: bool = False,
    balance_block_size: int | None = None,
    zigzag_min_length: int | None = None,
) -> _ForwardRewriteRequest:
    """Install Ring or Ulysses + Ring through the existing recipe method replacement.

    Args:
        target_module: GQAAttention produced by replace_qwen3_moe_flash_attention.
        mesh: Model mesh supplied by the recipe builder.
        tp_mesh: Tensor-parallel mesh supplied by the builder.
        cp_mesh: Active one-dimensional context-parallel mesh.
        ep_mesh: Expert-parallel mesh supplied by the builder.
        ulysses_degree: Positive divisor of CP size; 1 selects pure Ring.
        overlap: Prefetch Ring KV and pipeline dKV communication with compute.
        load_balance: Internal Ring task layout: none, zigzag, greedy or hybrid.
        prune_documents: Omit KV unused by remote queries; requires packed lengths and load_balance=none.
        balance_block_size: Optional query task granularity for greedy/hybrid.
        zigzag_min_length: Explicit raw-token long-document threshold for hybrid.

    Returns:
        Atomic rewrite request; no forward or companion attribute is mutated here.

    Raises:
        ValueError: If the target, degree, local heads or overlap option is unsupported.

    Note:
        Inputs retain equal contiguous source shards and original RoPE positions.
        Packed batch-one inputs supply document_lengths as a global host tuple in
        attention kwargs. No padding, explicit masks, dropout or KV cache is supported.
        Local cumulative lengths cannot substitute for global document boundaries.
    """
    del mesh, tp_mesh, ep_mesh
    original_forward = _validate_qwen3_moe_flash_attention_cp_target(
        target_module, cp_mesh, "qwen3_moe_ring_cp_wrapper",
    )
    _validate_ulysses_degree(cp_mesh, ulysses_degree)
    _validate_pruning(prune_documents, "TND", load_balance)
    if not isinstance(overlap, bool):
        raise ValueError("overlap must be bool")
    if load_balance not in ("none", "zigzag", "greedy", "hybrid"):
        raise ValueError("load_balance must be none, zigzag, greedy or hybrid")
    if balance_block_size is not None and (load_balance not in ("greedy", "hybrid")
            or not isinstance(balance_block_size, int) or isinstance(balance_block_size, bool)
            or balance_block_size < 1):
        raise ValueError("balance_block_size requires greedy/hybrid and a positive integer")
    if load_balance == "hybrid":
        if (not isinstance(zigzag_min_length, int) or isinstance(zigzag_min_length, bool)
                or zigzag_min_length < 1):
            raise ValueError("hybrid requires an explicit positive zigzag_min_length")
    elif zigzag_min_length is not None:
        raise ValueError("zigzag_min_length requires hybrid")
    balance_options = {"load_balance": load_balance, "prune_documents": prune_documents,
                       "balance_block_size": balance_block_size, "zigzag_min_length": zigzag_min_length}
    for name in ("num_heads", "num_key_value_heads"):
        count = getattr(target_module, name, None)
        if not isinstance(count, int) or count < 1 or count % ulysses_degree:
            raise ValueError(f"local {name} ({count}) must be positive and divisible by ulysses_degree")

    @wraps(original_forward)
    def cp_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        actual_seq_len: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Validate training-only semantics before any projection or cache mutation.

        Args:
            hidden_states: Local hidden states passed to the model attention.
            position_embeddings: Optional model-owned rotary position embeddings.
            attention_mask: Mask accepted by the Ring training contract.
            past_key_values: Cache argument, which must be absent during Ring training.
            actual_seq_len: Optional packed local sequence metadata.
            **kwargs: Additional model arguments, including global document_lengths.
        """
        if past_key_values is not None or kwargs.get("past_key_value") is not None:
            raise ValueError("Qwen3-MoE Ring CP requires training without a KV cache")
        _validate_ring_call(attention_mask, 0, {**kwargs, "actual_seq_len": actual_seq_len})
        if prune_documents and kwargs.get("document_lengths") is None:
            raise ValueError("prune_documents requires global document_lengths")
        return original_forward(
            hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
            past_key_values=past_key_values, actual_seq_len=actual_seq_len, **kwargs,
        )

    return _ForwardRewriteRequest(
        target_module, cp_forward,
        companion_attrs={
            "attention_interface": _build_qwen3_moe_ring_interface(cp_mesh, ulysses_degree, overlap, balance_options),
        },
    )
