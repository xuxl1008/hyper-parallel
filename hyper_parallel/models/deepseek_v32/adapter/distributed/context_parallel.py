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
"""DeepSeek-V3.2 DSA Context Parallel wrappers.

Main attention and KL retain the query shard while optional sparse selection
workers exchange low-rank qr and integer TopK. Main and Index K banks use either
all-gather or document-prefix all-to-all-v; the Index K route includes both
worker selection and owner KL needs. Dense warmup retains the owner layout.
The high-performance DSA operators remain in components.functional.
"""
# This model-specific adapter orchestrates PyTorch/NPU tensors directly.

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from itertools import accumulate
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.functional.dsa_indexer import dsa_indexer
from hyper_parallel.components.functional.dsa_sparse_attention import dsa_sparse_attention
from hyper_parallel.components.functional.rotary_embedding import _apply_interleaved_rope
from hyper_parallel.components.functional.npu_fusion_attention import (
    resolve_packed_sequence_lengths,
)
from hyper_parallel.components.modules.dsa_attention import (
    _apply_attention_rope,
    _deepseek_v32_dense_attention,
    _restore_attention_projection,
)
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.context_parallel.collectives import (
    async_cp_allgather_launch,
)
from hyper_parallel.distributed.context_parallel.indexer_balance import (
    launch_indexer_inputs, resolve_indexer_balance_block_size, validate_indexer_hybrid_threshold,
)
from hyper_parallel.distributed.context_parallel.index_key_dispatch import index_key_plan
from hyper_parallel.distributed.context_parallel.sequence_dispatch import (
    async_sequence_dispatch, document_prefix_route,
)
from hyper_parallel.distributed.recipe_spec import inner_wrapper
from hyper_parallel.models.deepseek_v32.adapter.distributed.document_plan import (
    dsa_document_plan, pack_selection_prefixes,
)


def _resolve_cp_document_plan(actual_seq_len, kwargs, batch_size, local_seq_len, cp_size, cp_rank):
    """Resolve packed aliases once and omit documents without local queries."""
    total = batch_size * local_seq_len * cp_size
    query_ends, key_ends = resolve_packed_sequence_lengths(
        {**kwargs, "actual_seq_len": actual_seq_len}, total, total,
    )
    if query_ends is None:
        query_ends = [(row + 1) * local_seq_len * cp_size for row in range(batch_size)]
        key_ends = query_ends
    if tuple(query_ends) != tuple(key_ends):
        raise ValueError("DSA CP self-attention requires identical global Q/K document boundaries")
    return dsa_document_plan(tuple(query_ends), batch_size, local_seq_len, cp_size, cp_rank)


def _apply_worker_query_rope(query, embeddings, rope_head_dim, interleaved):
    """Use the owner rotary convention without computing an unused companion key."""
    if embeddings is None:
        return query
    # Optional NPU dependency; CPU planning does not execute worker projections.
    from torch_npu import npu_rotary_mul  # pylint: disable=import-outside-toplevel

    cos, sin = embeddings
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    rotary, passthrough = query.split((rope_head_dim, query.shape[-1] - rope_head_dim), dim=-1)
    if interleaved:
        cos = cos[..., :rope_head_dim // 2].repeat_interleave(2, dim=-1).transpose(0, 1)
        sin = sin[..., :rope_head_dim // 2].repeat_interleave(2, dim=-1).transpose(0, 1)
        return _apply_interleaved_rope(rotary, passthrough, cos, sin, 2)
    rotated = npu_rotary_mul(rotary.clone(), cos.unsqueeze(2), sin.unsqueeze(2), rotary_mode="half")
    return torch.cat((rotated, passthrough), dim=-1)


@torch.no_grad()
def _worker_index_query(module, query_residual, embeddings, owner_rows):
    """Retain the owner's GEMM row shape before testing discrete selector parity."""
    projected = []
    for chunk in query_residual.split(owner_rows, dim=1):
        padded = torch.nn.functional.pad(chunk, (0, 0, 0, owner_rows - chunk.shape[1]))
        projected.append(module.indexer.wq_b(padded)[:, :chunk.shape[1]])
    query = projected[0] if len(projected) == 1 else torch.cat(projected, dim=1)
    query = query.view(
        1, query_residual.shape[1], module.num_index_heads, module.index_head_dim,
    )
    return _apply_worker_query_rope(
        query, embeddings, module.qk_rope_head_dim, module.index_rotary_interleaved,
    )


@torch.no_grad()
def _prepare_worker_query(module, exchange, apply_rope):
    """Consume worker inputs and project Q independently of the Index K transfer."""
    qr, weights, cos, sin = exchange.wait()
    query = None
    if exchange.route.query_count:
        query = _worker_index_query(module, qr, (cos, sin) if apply_rope else None, exchange.owner_queries)
    return query, weights


@torch.no_grad()
def _launch_worker_selection(module, exchange, selection_key, key_plan, worker_inputs):
    """Run LI after its K dependency and return document-local IDs to Q owners."""
    query, weights = worker_inputs
    if not exchange.route.query_count:
        selected = torch.empty((0, module.index_topk), device=weights.device, dtype=torch.int32)
    else:
        segments = exchange.route.segments
        if key_plan is not None:
            segments = key_plan.map_segments(segments)
        key, q_ends, k_ends = pack_selection_prefixes(selection_key, segments)
        lengths = tuple(torch.tensor(ends, device=query.device, dtype=torch.int32) for ends in (q_ends, k_ends))
        selected = dsa_indexer(
            query, key, weights, *lengths, module.index_topk, backend="cann", host_lengths=(q_ends, k_ends),
        )[0][:, 0]
    return exchange.launch_result(selected)


def _launch_selection_inputs(module, qr, weights, embeddings, plan, cp_mesh, options):
    """Submit the optional no-grad worker input before the KV bank collectives."""
    if options["indexer_load_balance"] == "none":
        return None
    if embeddings is None:
        shape = (1, qr.shape[1], module.qk_rope_head_dim)
        embeddings = (qr.new_ones(shape), qr.new_zeros(shape))
    else:
        embeddings = tuple(value.unsqueeze(0) if value.ndim == 2 else value for value in embeddings)
    return launch_indexer_inputs(
        qr, weights, embeddings, plan.lengths, 1, options["indexer_load_balance"], cp_mesh.get_group(),
        options["indexer_balance_block_size"], options["indexer_zigzag_min_length"],
    )


def _launch_cp_banks(index_key, kv_nope, k_rot, plan, cp_mesh, options):
    """Launch Index K first, then main KV, retaining independent gradient routes."""
    group, size, rank = cp_mesh.get_group(), cp_mesh.size(), cp_mesh.get_local_rank()
    key_plan = None
    if options["index_k_comm"] == "all_to_all_v":
        key_plan = index_key_plan(
            plan.lengths, size, rank, 1, options["indexer_load_balance"],
            options["indexer_balance_block_size"], options["indexer_zigzag_min_length"],
        )
        index_pending = async_sequence_dispatch(index_key, 1, key_plan.dispatch, group, gradient=key_plan.gradient)
    else:
        index_pending = async_cp_allgather_launch(index_key, 1, cp_mesh)
    if options["main_kv_comm"] == "all_to_all_v":
        route = document_prefix_route((0, *accumulate(plan.lengths)), index_key.shape[1], size, rank, 1)
        kv_pending = async_sequence_dispatch(kv_nope, 1, route, group)
        rope_pending = async_sequence_dispatch(k_rot, 1, route, group)
    else:
        kv_pending = async_cp_allgather_launch(kv_nope, 1, cp_mesh)
        rope_pending = async_cp_allgather_launch(k_rot, 1, cp_mesh)
    return key_plan, index_pending, kv_pending, rope_pending


def _validate_cp_target(target_module: nn.Module, cp_mesh: Any) -> None:
    """Validate the replacement surface before installing a CP forward."""
    wrapper_name = "deepseek_v32_dsa_attention_cp_wrapper"
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(f"{wrapper_name} requires an active CP mesh")
    if getattr(target_module, "_hp_deepseek_v32_dsa_cp_config", None) is not None:
        raise RuntimeError(
            "DeepSeek-V3.2 DSA Context Parallel has already been applied to this module"
        )

    required = (
        "_validate_forward_inputs",
        "_project_attention_states",
        "_project_index_states",
        "_apply_auxiliary_loss",
        "o_proj",
    )
    missing = [name for name in required if not hasattr(target_module, name)]
    if missing:
        raise TypeError(
            f"{wrapper_name} requires a DeepseekV32DSAAttention replacement; "
            f"missing attributes: {missing}"
        )


def _validate_cp_stage(batch_size, use_dense, options):
    """Restrict sparse worker scheduling and packed-only routes explicitly."""
    mode = options["indexer_load_balance"]
    if use_dense and mode != "none":
        raise ValueError("DSA Indexer load balance redistributes sparse TopK only; dense warmup requires mode='none'")
    if batch_size != 1 and (mode != "none" or options["main_kv_comm"] != "all_gather"
                           or options["index_k_comm"] != "all_gather"):
        raise ValueError("DSA query balance and document-prefix all_to_all_v require packed batch size one")


def _run_cp_dsa_kernels(
    module: nn.Module,
    index_states: tuple[torch.Tensor, ...],
    attention_states: tuple[torch.Tensor, ...],
    sequence_lengths: tuple[torch.Tensor, torch.Tensor],
    *,
    use_dense: bool,
    selected: torch.Tensor | None = None,
    host_lengths: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> torch.Tensor:
    """Run the selected sparse or dense DSA CANN path."""
    index_query, index_key, merge_weight = index_states
    absorbed_query, kv_nope, q_rot, k_rot = attention_states
    actual_q_len, actual_kv_len = sequence_lengths
    if use_dense:
        return _deepseek_v32_dense_attention(
            module, index_states, attention_states, sequence_lengths, host_lengths=host_lengths,
        )

    if selected is None:
        selected, *_ = dsa_indexer(
            index_query, index_key, merge_weight, actual_q_len, actual_kv_len, module.index_topk, backend="cann",
        )
    index_query_tnd, index_key_tnd, merge_weight_tnd = (
        tensor.flatten(0, 1) for tensor in index_states
    )
    topk_indices = selected
    attn_output, softmax_max, softmax_sum = dsa_sparse_attention(
        absorbed_query,
        kv_nope,
        q_rot,
        k_rot,
        topk_indices,
        module.scaling,
        actual_q_len,
        actual_kv_len,
        backend="cann",
        pad_output=False,
    )
    loss_lengths = sequence_lengths if host_lengths is None else host_lengths
    return module._apply_auxiliary_loss(  # pylint: disable=protected-access
        attn_output,
        (index_query_tnd, index_key_tnd, merge_weight_tnd),
        attention_states,
        topk_indices,
        (softmax_max, softmax_sum),
        *loss_lengths,
    )


def _deepseek_v32_dsa_cp_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    position_ids: torch.Tensor | None = None,
    actual_seq_len: torch.Tensor | Sequence[int] | None = None,
    use_dense: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run DSA with owner learning, optional selection workers and causal KV routes."""
    del attention_mask
    if hidden_states.ndim != 3:
        raise ValueError(
            "DeepseekV32DSAAttention CP expects hidden_states with shape [B, S, H]"
        )
    batch_size, local_seq_len = hidden_states.shape[:2]
    cp_mesh = getattr(module, "_hp_deepseek_v32_dsa_cp_mesh", None)
    if cp_mesh is None:
        raise RuntimeError("DeepSeek-V3.2 DSA CP forward is missing its CP mesh")
    cp_size = cp_mesh.size()
    cp_rank = cp_mesh.get_local_rank()
    use_dense = bool(module.dsa_dense_warm_up) if use_dense is None else use_dense
    options = module._hp_deepseek_v32_dsa_cp_config  # pylint: disable=protected-access
    _validate_cp_stage(batch_size, use_dense, options)
    plan = _resolve_cp_document_plan(actual_seq_len, kwargs, batch_size, local_seq_len, cp_size, cp_rank)
    module._validate_forward_inputs(  # pylint: disable=protected-access
        hidden_states,
        None,
        past_key_values,
        position_ids,
        dict(kwargs),
    )

    q_resid, absorbed_query, kv_nope, q_rot, k_rot, kv_weight = (
        module._project_attention_states(hidden_states, use_dense=use_dense)  # pylint: disable=protected-access
    )
    q_rot, k_rot = _apply_attention_rope(
        q_rot,
        k_rot,
        position_embeddings,
        interleaved=module.rotary_interleaved,
    )
    index_query, index_key, merge_weight = module._project_index_states(  # pylint: disable=protected-access
        hidden_states,
        q_resid,
        position_embeddings,
    )

    exchange = _launch_selection_inputs(
        module, q_resid, merge_weight, position_embeddings, plan, cp_mesh, options,
    )
    key_plan, index_key_gather, kv_nope_gather, k_rot_gather = _launch_cp_banks(
        index_key, kv_nope, k_rot, plan, cp_mesh, options,
    )
    sequence_lengths = plan.sequence_lengths(hidden_states.device)

    # Projection depends only on qr and RoPE inputs, so it can cover pending bank
    # communication. Collective launch order is identical on every CP rank.
    worker_inputs = None
    if exchange is not None:
        worker_inputs = _prepare_worker_query(module, exchange, position_embeddings is not None)
    if key_plan is None:
        selection_key = index_key_gather.wait()
        index_key = plan.key(selection_key)
    else:
        selection_key, index_key = index_key_gather.wait_with_gradient()
    index_query = index_query.flatten(0, 1).unsqueeze(0)
    merge_weight = merge_weight.flatten(0, 1).unsqueeze(0)
    selected = None
    if exchange is not None:
        pending_topk = _launch_worker_selection(
            module, exchange, selection_key.detach(), key_plan, worker_inputs,
        )
    elif not use_dense:
        selected = dsa_indexer(
            index_query, index_key, merge_weight, *sequence_lengths, module.index_topk, backend="cann",
            host_lengths=(plan.query_ends, plan.key_ends),
        )[0]

    # Main-bank waits occur after selection. A pending TopK return owns its buffers
    # until consumed; it cannot bypass an earlier collective on the same CP group.
    kv_nope, k_rot = kv_nope_gather.wait(), k_rot_gather.wait()
    if options["main_kv_comm"] == "all_gather":
        kv_nope, k_rot = plan.key(kv_nope), plan.key(k_rot)
    if exchange is not None:
        selected = pending_topk.wait().unsqueeze(1)
    attention_states = (absorbed_query.flatten(0, 1).unsqueeze(0), kv_nope,
                        q_rot.flatten(0, 1).unsqueeze(0), k_rot)
    attn_output = _run_cp_dsa_kernels(
        module, (index_query, index_key, merge_weight), attention_states, sequence_lengths,
        use_dense=use_dense, selected=selected, host_lengths=(plan.query_ends, plan.key_ends),
    ).view(batch_size, local_seq_len, module.num_heads, -1)
    if use_dense:
        return module.o_proj(attn_output.flatten(2)), None
    value_up_weight = kv_weight[:, module.qk_nope_head_dim :].transpose(1, 2)
    attn_output = _restore_attention_projection(
        attn_output,
        value_up_weight,
        num_heads=module.num_heads,
        batch_size=batch_size,
        seq_length=local_seq_len,
        kv_lora_rank=module.kv_lora_rank,
        value_head_dim=module.v_head_dim,
    )
    return module.o_proj(attn_output), None


@inner_wrapper
def deepseek_v32_dsa_attention_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    dense_warmup: bool | None = None,
    indexer_load_balance: str = "none",
    indexer_balance_block_size: int | None = None,
    indexer_zigzag_min_length: int | None = None,
    main_kv_comm: str = "all_gather",
    index_k_comm: str = "all_gather",
) -> _ForwardRewriteRequest:
    """Install CP and follow the source DSA dense-warm-up state by default.

    ``dense_warmup`` explicitly overrides the stage flag when supplied.
    Otherwise the forward reads ``target_module.dsa_dense_warm_up`` so training
    code can switch between dense warm-up and sparse DSA without reinstalling
    the wrapper. Set it to ``true`` or ``false`` in the recipe only to force a
    path for the whole run; leave it ``null`` to follow the model's stage flag.

    Sparse selection can use none/zigzag/greedy/hybrid. Dense warmup requires
    none: its fused dense KL learns from every visible key, not TopK workers.
    Both stages can independently select all_gather or all_to_all_v for main
    and Index K banks. The latter and balancing require packed batch size one.
    """
    del mesh, tp_mesh, ep_mesh
    block = resolve_indexer_balance_block_size(indexer_load_balance, indexer_balance_block_size)
    validate_indexer_hybrid_threshold(indexer_load_balance, indexer_zigzag_min_length)
    for name, value in (("main_kv_comm", main_kv_comm), ("index_k_comm", index_k_comm)):
        if value not in ("all_gather", "all_to_all_v"):
            raise ValueError(f"{name} must be 'all_gather' or 'all_to_all_v'")
    _validate_cp_target(target_module, cp_mesh)
    original_forward = target_module.forward

    @wraps(original_forward)
    def cp_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        position_ids: torch.Tensor | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Execute the CP DSA path using the target module's projections."""
        return _deepseek_v32_dsa_cp_forward(
            target_module,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            position_ids,
            actual_seq_len,
            use_dense=dense_warmup,
            **kwargs,
        )

    return _ForwardRewriteRequest(
        target_module,
        cp_forward,
        companion_attrs={
            "_hp_deepseek_v32_dsa_cp_mesh": cp_mesh,
            "_hp_deepseek_v32_dsa_cp_config": {
                "cp_size": cp_mesh.size(),
                "dense_warmup_override": dense_warmup,
                "indexer_load_balance": indexer_load_balance,
                "indexer_balance_block_size": block,
                "indexer_zigzag_min_length": indexer_zigzag_min_length,
                "main_kv_comm": main_kv_comm,
                "index_k_comm": index_k_comm,
            },
            "_hp_deepseek_v32_dsa_cp_dense_warmup": dense_warmup,
        },
    )


__all__ = [
    "deepseek_v32_dsa_attention_cp_wrapper",
]
