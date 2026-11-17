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
"""Opt-in DSA eager experiments; importing this module changes no training path.

The context manager is for a single-thread, single-module comparison. It uses
scoped replacements of private CP helpers, restored on exit. It is deliberately
not a Trainer recipe or an FSDP/checkpoint integration.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.components.modules.dsa_attention import _absorb_query_projection
from hyper_parallel.models.deepseek_v32.adapter.distributed import context_parallel as cp


@dataclass
class PendingGradient:
    """Own in-flight buffers until a downstream autograd gate consumes the sum."""

    pending: tuple | None = None
    expected: bool = False
    consumed: bool = False

    def launch(self, gradient: torch.Tensor, route: Any, group: Any) -> None:
        """Submit the reverse route without exposing unfinished receive storage."""
        if self.pending is not None or self.consumed:
            raise RuntimeError("The experimental gradient route supports one backward per forward")
        with torch.profiler.record_function("dsa/deferred_main_launch"):
            first = route.local_output_start
            last = first + route.local_length
            send = torch.cat((gradient[:first], gradient[last:]), dim=0)
            receive = gradient.new_empty((sum(route.input_splits), *gradient.shape[1:]))
            handle = dist.all_to_all_single(
                receive, send, output_split_sizes=list(route.input_splits),
                input_split_sizes=list(route.output_splits), group=group, async_op=True,
            )
            dtype = (torch.float32 if route.accumulate_fp32 and gradient.dtype in (torch.bfloat16, torch.float16)
                     else gradient.dtype)
            local = gradient[first:last].to(dtype=dtype, copy=True, memory_format=torch.contiguous_format)
            self.pending = handle, send, receive, local, gradient.dtype, route

    def finish(self) -> torch.Tensor:
        """Wait and reduce in the same peer order and dtype as the ordinary route."""
        if self.pending is None:
            raise RuntimeError("The Q gate reached main KV before its gradient communication was launched")
        with torch.profiler.record_function("dsa/deferred_main_wait_reduce"):
            handle, _send, receive, local, dtype, route = self.pending
            handle.wait()
            cursor = 0
            for offset, count in route.owner_ranges():
                if count:
                    local.narrow(0, offset, count).add_(receive.narrow(0, cursor, count))
                    cursor += count
            result = local.to(dtype)
            self.pending = None
            self.consumed = True
            return result


class _LatentGradientGate(torch.autograd.Function):
    """Make the local KV gradient available after the Q projection dependency."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor, state: PendingGradient) -> tuple:
        """Preserve both values and connect their backward dependencies."""
        ctx.state = state
        ctx.set_materialize_grads(False)
        return query, key

    @staticmethod
    def backward(ctx: Any, grad_query: torch.Tensor | None, grad_key: torch.Tensor | None) -> tuple:
        """Consume the owner sum only after every input edge of this gate is ready."""
        if ctx.state.expected:
            restored = ctx.state.finish().movedim(0, 1).contiguous()
            grad_key = restored if grad_key is None else grad_key + restored
        return grad_query, grad_key, None


class _DeferredBankWait(torch.autograd.Function):
    """Retain the forward bank and transfer its backward edge to the Q gate."""

    @staticmethod
    def forward(ctx: Any, local: torch.Tensor, receive: torch.Tensor, pending: Any,
                state: PendingGradient) -> torch.Tensor:
        """Wait for forward communication before assembling the bank."""
        pending.handle.wait()
        ctx.state, ctx.route, ctx.group = state, pending.route, pending.group
        state.expected = ctx.needs_input_grad[0]
        before = pending.route.local_output_start
        return torch.cat((receive[:before], local, receive[before:]), dim=0)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple:
        """Launch dKV; the gate supplies its real gradient after the independent Q edge."""
        ctx.state.launch(gradient, ctx.route, ctx.group)
        return None, None, None, None


@dataclass
class _DeferredPending:
    pending: Any
    state: PendingGradient

    def wait(self) -> torch.Tensor:
        """Attach the experimental autograd edge to an existing forward exchange."""
        pending = self.pending
        if pending.waited:
            raise RuntimeError("A sequence dispatch handle cannot be waited twice")
        if pending.gradient is not None or pending.sequence_dim != 1:
            raise ValueError("The experiment requires an unrestricted main KV route on sequence dimension 1")
        pending.waited = True
        output = _DeferredBankWait.apply(pending.local, pending.received, pending, self.state)
        pending.send = pending.received = None
        return output.movedim(0, 1).contiguous()


def _project_query(module: torch.nn.Module, query: torch.Tensor) -> tuple:
    batch, length = query.shape[:2]
    projected = module.q_b_proj(query).view(batch, length, module.num_heads, module.qk_head_dim)
    plain, rotary = projected.split((module.qk_nope_head_dim, module.qk_rope_head_dim), dim=-1)
    weight = module.kv_b_proj.weight.view(
        module.num_heads, module.qk_nope_head_dim + module.v_head_dim, module.kv_lora_rank,
    )
    absorbed = _absorb_query_projection(
        plain, weight[:, :module.qk_nope_head_dim], num_heads=module.num_heads,
        batch_size=batch, seq_length=length, input_dim=module.qk_nope_head_dim, output_dim=module.kv_lora_rank,
    )
    return absorbed, rotary, weight


@dataclass
class _ProjectionExperiment:
    module: torch.nn.Module
    defer_main: bool
    query_stream: Any = None
    state: PendingGradient | None = None

    def project(self, hidden: torch.Tensor, *, use_dense: bool = False) -> tuple:
        """Preserve projection arithmetic while changing only dependency boundaries."""
        if use_dense:
            raise ValueError("The backward scheduling experiment only supports sparse DSA")
        if self.state is not None and self.state.expected and not self.state.consumed:
            raise RuntimeError("Finish backward before the next experimental forward; checkpointing is not supported")
        module = self.module
        batch, length = hidden.shape[:2]
        query, key, rotary_key = module.linear_qkv(hidden).split(
            (module.q_lora_rank, module.kv_lora_rank, module.qk_rope_head_dim), dim=-1,
        )
        query = module.q_a_layernorm(query)
        key = module.kv_a_layernorm(key).view(batch, length, 1, module.kv_lora_rank)
        self.state = PendingGradient()
        if self.defer_main:
            query, key = _LatentGradientGate.apply(query, key, self.state)
        if self.query_stream is None:
            absorbed, rotary_query, weight = _project_query(module, query)
        else:
            current = torch.npu.current_stream()
            self.query_stream.wait_stream(current)
            with torch.npu.stream(self.query_stream):
                for tensor in (query, *module.q_b_proj.parameters(), *module.kv_b_proj.parameters()):
                    tensor.record_stream(self.query_stream)
                absorbed, rotary_query, weight = _project_query(module, query)
            current.wait_stream(self.query_stream)
            for tensor in (absorbed, rotary_query, weight):
                tensor.record_stream(current)
        return query, absorbed, key, rotary_query, rotary_key.view(batch, length, 1, -1), weight


@torch.no_grad()
def partial_worker_rope(query: torch.Tensor, embeddings: tuple | None, rope_head_dim: int,
                        interleaved: bool) -> torch.Tensor:
    """Compare the interleave-only partial op, including conversions to the owner layout.

    Half-pair inputs require two conversions and can be slower. Coefficients are
    rounded to query dtype to match the current owner rotary operator. This is a
    no-grad selection experiment; it must not replace the owner learning path.
    """
    if embeddings is None:
        return query
    if query.ndim != 4 or query.shape[0] != 1 or not 0 < rope_head_dim <= query.shape[-1] or rope_head_dim % 2:
        raise ValueError("Worker partial RoPE requires B=1 BSND Q and an even rotary prefix")
    # The experimental operator is an optional dependency; baseline imports must work without it.
    from cann_ops_transformer.ops import inplace_partial_rotary_mul  # pylint: disable=import-outside-toplevel

    cos, sin = embeddings
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    half = rope_head_dim // 2
    cos, sin = (value[..., :half].to(query.dtype).repeat_interleave(2, -1).unsqueeze(2).contiguous()
                for value in (cos, sin))
    if interleaved:
        output = query.clone(memory_format=torch.contiguous_format)
    else:
        paired = query[..., :rope_head_dim].unflatten(-1, (2, half)).transpose(-1, -2).flatten(-2)
        output = torch.cat((paired, query[..., rope_head_dim:]), dim=-1)
    inplace_partial_rotary_mul(output, cos, sin, rotary_mode="interleave", partial_slice=[0, rope_head_dim])
    # Both owner conventions return half-ordered rotary features in the current adapter.
    restored = output[..., :rope_head_dim].unflatten(-1, (half, 2)).transpose(-1, -2).flatten(-2)
    return torch.cat((restored, output[..., rope_head_dim:]), dim=-1)


@contextmanager
def dsa_experiment(module: torch.nn.Module, *, defer_main: bool = False, separate_query_stream: bool = False,
                   worker_partial_rope: bool = False) -> Iterator[None]:
    """Enable a scoped experiment around one eager forward/backward benchmark.

    The temporary CP helper replacement is process-wide: do not use concurrent
    forwards, checkpoint recomputation, FSDP or retained-graph second backward.
    Use identical options on every CP rank. Main-KV A2AV is required for defer.
    """
    options = module._hp_deepseek_v32_dsa_cp_config  # pylint: disable=protected-access
    if (defer_main or separate_query_stream) and module.dsa_dense_warm_up:
        raise ValueError("Backward scheduling experiments require sparse DSA")
    if defer_main and options["main_kv_comm"] != "all_to_all_v":
        raise ValueError("Deferred main gradient requires main_kv_comm=all_to_all_v")
    if separate_query_stream and module.linear_qkv.weight.device.type != "npu":
        raise ValueError("The separate Q stream experiment requires NPU")
    experiment = _ProjectionExperiment(module, defer_main,
                                       torch.npu.Stream() if separate_query_stream else None)
    original_launch = cp._launch_cp_banks  # pylint: disable=protected-access

    def launch(*args: Any, **kwargs: Any) -> tuple:
        results = original_launch(*args, **kwargs)
        pending = results[2]
        if pending.received is None:
            return results
        if experiment.state is None:
            raise RuntimeError("The experimental projection must run before the main KV launch")
        return *results[:2], _DeferredPending(pending, experiment.state), results[3]

    with ExitStack() as stack:
        if defer_main or separate_query_stream:
            stack.enter_context(patch.object(module, "_project_attention_states", experiment.project))
        if defer_main:
            stack.enter_context(patch.object(cp, "_launch_cp_banks", launch))
        if worker_partial_rope:
            stack.enter_context(patch.object(cp, "_apply_worker_query_rope", partial_worker_rope))
        yield
