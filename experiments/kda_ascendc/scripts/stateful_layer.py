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
"""Experimental full-layer wiring; no published adapter or backend is changed."""
from unittest.mock import patch

import torch
from fla_npu.ops.triton import l2norm
from hyper_parallel.distributed.context_parallel import kimi_delta_attention as cp

from stateful_kda import LocalKDA, StateCP
from memory_options import variant

MODES = ("p2p", "allgather", "ulysses", "p2p_whole2", "p2p_sf2", "p2p_dual2",
         "p2p_whole4", "p2p_sf4", "p2p_dual4", "p2p_summary2", "p2p_summary4")


def staged_core(query, key, value, gate, beta, *, a_log, dt_bias, protocol,
                scale=None, lower_bound=-5., chunk_size=64, safe_gate=True, **kwargs):
    if lower_bound != -5. or chunk_size != 64 or not safe_gate:
        raise ValueError("Full-layer prototype supports safe gate -5 and chunk64 only")
    if query.shape != value.shape:
        raise ValueError("Full-layer prototype requires equal Q/K/V heads and D128")
    q, k, v, raw, beta = [x.transpose(1, 2).contiguous() for x in (query, key, value, gate, beta)]
    q, k = l2norm(q, eps=1e-6), l2norm(k, eps=1e-6)
    beta = beta.float().sigmoid().to(beta.dtype)
    inputs = (q, k, v, raw, beta, a_log.contiguous(), dt_bias.reshape(q.shape[1], 128).contiguous())
    if protocol == "local":
        h0 = torch.zeros(q.shape[0], q.shape[1], 128, 128, dtype=torch.float32, device=q.device)
        result, _ = LocalKDA.apply(*inputs, h0)
    elif protocol.startswith("p2p_summary"):
        from stateful_summary_heads import SummaryHeadCP
        result = SummaryHeadCP.apply(*inputs, int(protocol.removeprefix("p2p_summary")))
    elif protocol.startswith("p2p_"):
        from stateful_heads import head_core
        result = head_core(inputs, protocol)
    else:
        result = StateCP.apply(*inputs, protocol)
    # The validated native prototype uses scale=.125. State is scale-independent;
    # rescaling output preserves the layer's default scale and its chain rule.
    effective_scale = query.shape[-1] ** -.5 if scale is None else float(scale)
    result = result * (effective_scale / .125)
    return result.transpose(1, 2).contiguous()


class ExperimentalStateCore(torch.nn.Module):
    def __init__(self, protocol):
        super().__init__()
        self.protocol = protocol

    def forward(self, *args, **kwargs):
        return staged_core(*args, protocol=self.protocol, **kwargs)


class ExperimentalUlyssesLayer(cp.KimiDeltaAttentionLayerUlyssesCP):
    def forward(self, *args, **kwargs):
        # Single-threaded experiment only: reuse the existing packed redistribution,
        # head-sliced convolutions and output path, changing just the core boundary.
        def run_local(*inputs, **options):
            return staged_core(*inputs, protocol="local", **options)
        with patch.object(cp, "_run_local_kda", run_local):
            return super().forward(*args, **kwargs)


def build_executor(layer, mesh, mode):
    if "norm" in variant():
        from fused_output_norm import ExperimentalGatedNorm
        if not isinstance(layer.o_norm, ExperimentalGatedNorm):
            layer.o_norm = ExperimentalGatedNorm(layer.o_norm)
    if mode == "ulysses":
        return ExperimentalUlyssesLayer(layer, mesh, backend="eager")
    if mode not in MODES:
        raise ValueError(mode)
    # Construct only the existing layer shell, then replace its core explicitly.
    # No eager or public AscendC fallback is used by the resulting executor.
    executor = cp.KimiDeltaAttentionLayerCP(layer, mesh, backend="eager")
    executor.core = ExperimentalStateCore(mode)
    return executor
