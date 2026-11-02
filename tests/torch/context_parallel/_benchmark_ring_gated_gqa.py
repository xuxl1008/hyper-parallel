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
"""Qwen3-14B-shaped GQA and local output-gate scheduling experiment.

The normal paths use the framework GQA modules, including QK norm and RoPE.
Gated paths use the same dimensions and add the framework's elementwise gate;
they are not the original Qwen3-14B architecture. Only an attention module and
local gradients are measured, without a model checkpoint, MLP, or grad sync.
"""

import argparse
from datetime import timedelta
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch
from torch import nn
import torch.distributed as dist
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from hyper_parallel import init_device_mesh
from hyper_parallel.components.modules.gqa_attention import (
    GatedGQAAttention,
    GQAAttention,
    _apply_gqa_rope,
)
from hyper_parallel.distributed.context_parallel.ring_layout import (
    _route_indices,
    redistribute_ring_qkv,
    ring_task_plan,
)
from tests.torch.context_parallel._benchmark_ring_projection_prepare import (
    RING,
    _WaitRoute,
    _error,
    _measure,
    _profile,
)


MODES = ("plain_none", "plain_zigzag", "gated_none", "gated_zigzag",
         "gated_restore_serial", "gated_sigmoid_overlap", "gated_split_serial", "gated_split_overlap")
LABELS = ("O", "dHidden", "dWqkv", "dWo", "dQnorm", "dKnorm", "dWgate")


def _source(args, gated):
    config = SimpleNamespace(hidden_size=args.hidden, num_attention_heads=args.query_heads,
                             num_key_value_heads=args.kv_heads, head_dim=args.head_dim,
                             attention_dropout=0.0, sliding_window=None)
    with torch.device("meta"):
        return SimpleNamespace(
            config=config, training=True, layer_idx=0,
            q_proj=nn.Linear(args.hidden, args.query_heads * args.head_dim * (2 if gated else 1), bias=False),
            k_proj=nn.Linear(args.hidden, args.kv_heads * args.head_dim, bias=False),
            v_proj=nn.Linear(args.hidden, args.kv_heads * args.head_dim, bias=False),
            o_proj=nn.Linear(args.query_heads * args.head_dim, args.hidden, bias=False),
            q_norm=Qwen3RMSNorm(args.head_dim, eps=1e-6), k_norm=Qwen3RMSNorm(args.head_dim, eps=1e-6),
        )


def _modules(args):
    plain = GQAAttention(module=_source(args, False)).to_empty(device="npu").to(torch.bfloat16)
    gated = GatedGQAAttention(module=_source(args, True)).to_empty(device="npu").to(torch.bfloat16)
    gate = nn.Linear(args.hidden, args.query_heads * args.head_dim, bias=False,
                     device="npu", dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for parameter in (plain.linear_qkv.weight, plain.o_proj.weight, gate.weight):
            random = torch.randn(parameter.shape, generator=generator) / parameter.shape[-1] ** 0.5
            parameter.copy_(random.to(device="npu", dtype=torch.bfloat16))
        grouped = plain.linear_qkv.weight.view(args.kv_heads, -1, args.hidden)
        gate_weight = gate.weight.view(args.kv_heads, -1, args.hidden)
        gated.linear_qkv.weight.copy_(torch.cat((grouped, gate_weight), dim=1).flatten(0, 1))
        gated.o_proj.weight.copy_(plain.o_proj.weight)
        for module in (plain, gated):
            module.q_norm.weight.fill_(1)
            module.k_norm.weight.fill_(1)
    return plain, gated, gate


def _positions(lengths, rank, size, head_dim):
    positions = torch.cat([torch.arange(length) for length in lengths]).chunk(size)[rank].float()
    inverse = 1.0 / (1000000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    frequencies = positions[:, None] * inverse[None, :]
    embedding = torch.cat((frequencies, frequencies), dim=-1).to("npu")
    return embedding.cos().bfloat16().unsqueeze(0), embedding.sin().bfloat16().unsqueeze(0)


def _attention_interface(mesh, lengths, balance, module, query, key, value, attention_mask, **kwargs):
    del module, attention_mask
    tensors = [tensor.squeeze(0).transpose(0, 1) for tensor in (query, key, value)]
    output = RING.ring_attention(*tensors, mesh, input_layout="TND", causal=True, overlap=True,
                                 scale=kwargs["scaling"], load_balance=balance, document_lengths=lengths)
    return output.unsqueeze(0), None


def _launch_restore(tensor, plan, group):
    send, receive, sources, targets = _route_indices(plan.route, tensor.device)
    source = tensor.index_select(0, receive).contiguous()
    output = source.new_empty((sum(plan.route.input_splits), *source.shape[1:]))
    inputs, outputs = list(plan.route.output_splits), list(plan.route.input_splits)
    # _WaitRoute supplies the inverse collective and retains the send storage.
    work = dist.all_to_all_single(output, source.detach(), output_split_sizes=outputs,
                                  input_split_sizes=inputs, group=group, async_op=True)
    local = tensor.index_select(0, targets)
    return (source, output, work, group, inputs, outputs), local, sources, send


def _finish_restore(pending, count):
    arguments, local, sources, receive = pending
    remote = _WaitRoute.apply(*arguments)
    output = local.new_zeros((count, *local.shape[1:]))
    output.index_put_((sources,), local)
    output.index_put_((receive,), remote)
    return output


def _experimental_forward(module, gate_projection, hidden, positions, plan, ring, mode):
    split = mode.startswith("gated_split")
    width = module.qkv_group_width if split else module.gated_qkv_group_width
    projected = module.linear_qkv(hidden).view(1, -1, module.num_key_value_heads, width)
    widths = module.qkv_split_sizes if split else (*module.qkv_split_sizes,
                                                  module.num_key_value_groups * module.qk_head_dim)
    parts = projected.split(widths, dim=-1)
    query, key, value = parts[:3]
    gate = None if split else parts[3].reshape(1, hidden.shape[1], -1)
    query = module.q_norm(query.reshape(1, -1, module.num_heads, module.qk_head_dim))
    key = module.k_norm(key)
    query, key = _apply_gqa_rope(query, key, positions, module.rotary_interleaved, module.qk_head_dim)
    query, key, value = redistribute_ring_qkv(
        query.squeeze(0), key.squeeze(0), value.squeeze(0), plan, ring.group, 0, 1,
    )
    output = RING._RingAttention.apply(query, key, value, ring, True, module.scaling, "TND", True, plan)
    pending = _launch_restore(output, plan, ring.group)
    if mode.endswith("serial"):
        output = _finish_restore(pending, plan.local_tokens)
    if split:
        gate = gate_projection(hidden)
    gate = torch.sigmoid(gate)
    if mode.endswith("overlap"):
        output = _finish_restore(pending, plan.local_tokens)
    return module.o_proj(output.reshape(1, hidden.shape[1], -1) * gate)


def _block(state, mode, backward=True, canonical=False):
    args, modules, hidden, grad_output, positions, mesh, lengths, plan, ring = state
    plain, gated, gate = modules
    split = mode.startswith("gated_split")
    is_plain = mode.startswith("plain")
    module = plain if split or is_plain else gated
    if mode in ("plain_none", "plain_zigzag", "gated_none", "gated_zigzag"):
        balance = "none" if mode.endswith("none") else "zigzag"
        module.attention_interface = partial(_attention_interface, mesh, lengths, balance)
        output = module(hidden, position_embeddings=positions)[0]
    else:
        output = _experimental_forward(module, gate, hidden, positions, plan, ring, mode)
    if not backward:
        return output
    parameters = (hidden, module.linear_qkv.weight, module.o_proj.weight,
                  module.q_norm.weight, module.k_norm.weight)
    if split:
        parameters += (gate.weight,)
    gradients = torch.autograd.grad(output, parameters, grad_output)
    if not canonical:
        return output, *gradients
    result = dict(zip(LABELS[:6], (output, *gradients[:5])))
    if split:
        result["dWgate"] = gradients[5]
    elif not is_plain:
        grouped = result["dWqkv"].view(args.kv_heads, -1, args.hidden)
        result["dWqkv"], result["dWgate"] = [part.flatten(0, 1) for part in grouped.split(
            (plain.qkv_group_width, args.query_heads // args.kv_heads * args.head_dim), dim=1)]
    return {name: value.detach() for name, value in result.items()}


def _check(state, modes, name):
    rank = dist.get_rank()
    references = {kind: _block(state, f"{kind}_zigzag", canonical=True) for kind in ("plain", "gated")}
    split_reference = _block(state, "gated_split_serial", canonical=True) if "gated_split_overlap" in modes else None
    errors = {}
    for mode in modes:
        if rank == 0:
            print(json.dumps({"check": name, "mode": mode}), flush=True)
        actual = _block(state, mode, canonical=True)
        expected = references["plain" if mode.startswith("plain") else "gated"]
        errors[mode] = {key: _error(value, expected[key]) for key, value in actual.items()}
        for key, metrics in errors[mode].items():
            if mode == "gated_split_overlap":
                torch.testing.assert_close(actual[key], split_reference[key], rtol=0, atol=0,
                                           msg=f"{name}/{mode}/{key} differs from the split serial control")
                metrics["split_schedule_bitwise_equal"] = True
            if not metrics["finite"] or metrics["relative_l2"] > 0.02:
                raise RuntimeError(f"{name}/{mode}/{key}: {metrics}")
            if mode in ("gated_restore_serial", "gated_sigmoid_overlap"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0,
                                           msg=f"{name}/{mode}/{key}: {metrics}")
    return errors


def _measure_modes(functions, repeats, iterations):
    measurements = {mode: [] for mode in functions}
    names = tuple(functions)
    for repeat in range(repeats):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for mode in order if repeat % 2 == 0 else reversed(order):
            measurements[mode].append(_measure(functions[mode], iterations))
    return measurements


def _summarize(workers, modes, repeats, forward_repeats):
    summary = {}
    for mode in modes:
        times = [max(worker["measurements"][mode][repeat]["ms"] for worker in workers)
                 for repeat in range(repeats)]
        errors = {}
        for name in workers[0]["errors"][mode]:
            items = [worker["errors"][mode][name] for worker in workers]
            errors[name] = {"bitwise_equal_all_ranks": all(item["bitwise_equal"] for item in items),
                            "max_abs": max(item["max_abs"] for item in items),
                            "max_relative_l2": max(item["relative_l2"] for item in items)}
        samples = [item for worker in workers for item in worker["measurements"][mode]]
        summary[mode] = {"median_slowest_rank_ms": median(times), "slowest_rank_samples_ms": times,
                         "max_peak_extra_bytes": max(item["peak_extra_bytes"] for item in samples),
                         "max_live_growth_bytes": max(item["live_growth_bytes"] for item in samples),
                         "errors": errors}
        if forward_repeats:
            times = [max(worker["forward"][mode][repeat]["ms"] for worker in workers)
                     for repeat in range(forward_repeats)]
            summary[mode]["forward_only"] = {"median_slowest_rank_ms": median(times),
                                             "slowest_rank_samples_ms": times}
    return summary


def _run_case(args, sample, mesh, modules):
    lengths = tuple(sample["document_lengths"])
    rank, size = mesh.get_local_rank(), mesh.size()
    local = sum(lengths) // size
    generator = torch.Generator().manual_seed(args.seed + rank)
    hidden = torch.randn(1, local, args.hidden, generator=generator).bfloat16().to("npu").requires_grad_()
    grad_output = torch.randn(hidden.shape, generator=generator).bfloat16().to("npu")
    positions = _positions(lengths, rank, size, args.head_dim)
    plan = ring_task_plan(lengths, size, rank, "zigzag", True)
    state = (args, modules, hidden, grad_output, positions, mesh, lengths, plan, RING._ring_context(mesh))
    torch.use_deterministic_algorithms(True)
    errors = _check(state, args.modes, sample["name"])
    torch.use_deterministic_algorithms(False)
    functions = {mode: partial(_block, state, mode) for mode in args.modes}
    for function in functions.values():
        for _ in range(args.warmup):
            function()
    measurements = _measure_modes(functions, args.repeats, args.iterations)
    forward = _measure_modes({mode: partial(_block, state, mode, False) for mode in args.modes},
                             args.forward_repeats, args.iterations)
    workers = [None] * size
    dist.all_gather_object(workers, {"rank": rank, "errors": errors, "measurements": measurements,
                                     "forward": forward})
    summary = _summarize(workers, args.modes, args.repeats, args.forward_repeats)
    if args.profile_dir and sample["name"] in args.profile_cases:
        _profile(functions, args.profile_modes, args.profile_dir, rank)
    return {"name": sample["name"], "document_lengths": lengths, "summary": summary, "workers": workers}


def main() -> None:
    """Measure framework attention modules and experimental gate scheduling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--query-heads", type=int, default=40)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--forward-repeats", type=int, default=5)
    parser.add_argument("--profile-dir")
    parser.add_argument("--profile-cases", nargs="+", default=["shuffled_000"])
    parser.add_argument("--profile-modes", nargs="+", choices=MODES,
                        default=["gated_restore_serial", "gated_sigmoid_overlap",
                                 "gated_split_serial", "gated_split_overlap"])
    args = parser.parse_args()
    if min(args.hidden, args.query_heads, args.kv_heads, args.head_dim, args.repeats, args.iterations) < 1:
        raise ValueError("Dimensions, repeat count, and iteration count must be positive")
    if args.query_heads % args.kv_heads or args.head_dim % 2:
        raise ValueError("Q heads must be divisible by KV heads and head_dim must be even")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=300))
    report = {"split_schedule_bitwise_checked": "gated_split_overlap" in args.modes,
              "parameters": vars(args), "results": [], "complete": False,
              "task_queue_enable": os.environ.get("TASK_QUEUE_ENABLE", "default"),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Framework GQA/GatedGQA with QK norm, document-relative RoPE, Ring and local gradients; "
                       "synthetic activations and weights, real document lengths; no MLP, full model, "
                       "checkpoint weights, optimizer, FSDP, or parameter gradient synchronization"}
    try:
        mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("cp",))
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if manifest["cp"] != mesh.size():
            raise ValueError("Manifest CP must match world size")
        selected = [sample for sample in manifest["samples"] if not args.cases or sample["name"] in args.cases]
        if not selected or (args.cases and len(selected) != len(set(args.cases))):
            raise ValueError("Every selected case must exist in the manifest")
        modules = _modules(args)
        for sample in selected:
            result = _run_case(args, sample, mesh, modules)
            report["results"].append(result)
            if dist.get_rank() == 0:
                print(json.dumps({"name": result["name"], "summary": result["summary"]}), flush=True)
                Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["complete"] = True
        if dist.get_rank() == 0:
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
