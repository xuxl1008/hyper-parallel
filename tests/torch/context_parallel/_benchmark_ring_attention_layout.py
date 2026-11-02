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
"""Opt-in balanced Ring benchmark with per-round, fragment, traffic and memory metrics.

Run with torchrun on 2/4/8 NPUs. End-to-end timing excludes instrumentation;
separate instrumented calls collect device events without per-round barriers.
Communication figures count planned sent payload bytes, not hardware counters.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.distributed.context_parallel.ring_attention import ulysses_ring_attention
from hyper_parallel.distributed.context_parallel.ring_layout import RingAttentionTrace, ring_task_plan
from tests.torch.context_parallel._test_ring_attention_layout import _packed_reference
from tests.torch.context_parallel._test_ring_attention_npu import _Case, _compare, _native_reference


def _iteration(inputs, mesh, options, trace=None):
    output = ulysses_ring_attention(*inputs[:3], mesh, trace=trace, **options)
    gradients = torch.autograd.grad(output, inputs[:3], inputs[3])
    return (output, *gradients)


@torch.no_grad()
def _chunked_reference(inputs, lengths, start, local, chunk=128):
    """Independent FP32 math for local Q rows and their contributions to global dKV.

    This reference materializes only chunk-by-document scores and uses the exact
    softmax derivative, with no fused attention or online Ring accumulation.
    """
    query, key, value, dout = (tensor.detach().float() for tensor in inputs)
    output, dq = (torch.zeros_like(query[start:start + local]) for _ in range(2))
    dk, dv = torch.zeros_like(key), torch.zeros_like(value)
    groups, scale = query.shape[1] // key.shape[1], query.shape[-1] ** -0.5
    offset = 0
    for length in lengths:
        stop = offset + length
        if max(start, offset) < min(start + local, stop):
            keys, values = (tensor[offset:stop].transpose(0, 1).repeat_interleave(groups, 0)
                            for tensor in (key, value))
            positions = torch.arange(length, device=query.device)
            for begin in range(max(start, offset), min(start + local, stop), chunk):
                end = min(begin + chunk, start + local, stop)
                q, dy = (tensor[begin:end].transpose(0, 1) for tensor in (query, dout))
                scores = (q @ keys.transpose(-1, -2)) * scale
                forbidden = positions[None, :] > torch.arange(begin - offset, end - offset,
                                                               device=query.device)[:, None]
                probability = scores.masked_fill_(forbidden, -float("inf")).softmax(-1)
                out = probability @ values
                ds = probability * (dy @ values.transpose(-1, -2) - (dy * out).sum(-1, keepdim=True))
                output[begin - start:end - start].copy_(out.transpose(0, 1))
                dq[begin - start:end - start].copy_(((ds @ keys) * scale).transpose(0, 1))
                for target, contribution in ((dk, (ds.transpose(-1, -2) @ q) * scale),
                                             (dv, probability.transpose(-1, -2) @ dy)):
                    reduced = contribution.unflatten(0, (key.shape[1], groups)).sum(1).transpose(0, 1)
                    target[offset:stop].add_(reduced)
        offset = stop
    return output, dq, dk, dv


def _full_reference(inputs, mesh, lengths, layout, kind):
    """Build native or chunked FP32 global references independently of Ring."""
    axis = 0 if layout == "TND" else 2
    full = []
    for tensor in inputs:
        gathered = [torch.empty_like(tensor) for _ in range(mesh.size())]
        dist.all_gather(gathered, tensor.detach().contiguous(), group=mesh.get_group())
        full.append(torch.cat(gathered, dim=axis))
    local = inputs[0].shape[axis]
    if kind == "fp32":
        packed = tuple(tensor if layout == "TND" else tensor.squeeze(0).transpose(0, 1) for tensor in full)
        previous = torch.npu.matmul.allow_hf32
        torch.npu.matmul.allow_hf32 = False
        try:
            output, dq, dk, dv = _chunked_reference(packed, lengths, mesh.get_local_rank() * local, local)
        finally:
            torch.npu.matmul.allow_hf32 = previous
        for gradient in (dk, dv):
            dist.all_reduce(gradient, group=mesh.get_group())
        expected = (output, dq, *(tensor.narrow(0, mesh.get_local_rank() * local, local).clone()
                                 for tensor in (dk, dv)))
        return tuple(tensor if layout == "TND" else tensor.transpose(0, 1).unsqueeze(0) for tensor in expected)
    if layout == "TND":
        expected = _packed_reference(tuple(full), lengths, True, True)
    else:
        case = _Case(torch.bfloat16, layout, 4, True, 1, sequence_length=sum(lengths),
                     query_heads=16, head_dim=128, batch_size=1)
        mask = torch.ones(2048, 2048, device=inputs[0].device, dtype=torch.bool).triu_(1)
        expected = _native_reference(full[0], full[1], full[2], full[3], case, mask)
    return tuple(tensor.detach().narrow(axis, mesh.get_local_rank() * local, local).clone() for tensor in expected)


def _timed(inputs, mesh, options, iterations):
    dist.barrier()
    torch.npu.synchronize()
    baseline = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    wall = perf_counter()
    begin.record()
    for _ in range(iterations):
        _iteration(inputs, mesh, options)
    end.record()
    end.synchronize()
    return {"device_ms": begin.elapsed_time(end) / iterations,
            "wall_ms": (perf_counter() - wall) * 1000 / iterations,
            "baseline_allocated_bytes": baseline, "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "peak_extra_bytes": torch.npu.max_memory_allocated() - baseline,
            "live_growth_bytes": torch.npu.memory_allocated() - baseline}


def _plan_metrics(lengths, mesh, degree, mode, block, threshold, causal):
    size, rank = mesh.size() // degree, mesh.get_local_rank() // degree
    plan = ring_task_plan(lengths, size, rank, mode, causal, block, threshold)
    q_heads, kv_heads, dim, element = 16 // degree, 4 // degree, 128, 2
    sent, received = sum(plan.route.input_splits), sum(plan.route.output_splits)
    if mode == "none":
        sent = received = 0
    forward = (sent * (q_heads + 2 * kv_heads) + received * q_heads) * dim * element
    backward = (received * (q_heads + 2 * kv_heads) + sent * q_heads) * dim * element
    ring_bytes = plan.capacity * kv_heads * dim * (4 * (size - 1) * element + 2 * size * 4)
    pairs, calls = [], []
    for step in range(size):
        blocks = plan.blocks[(rank - step) % size]
        batches = getattr(plan, "batches", None)
        calls.append(len(blocks) if batches is None else len(batches[(rank - step) % size]))
        pairs.append(sum((part.query_end - part.query_start) * (part.key_end - part.key_start)
                         - ((part.query_end - part.query_start) * (part.query_end - part.query_start - 1) // 2
                            if part.causal else 0) for part in blocks))
    ulysses_bytes = 2 * (2 * 16 + 2 * 4) * (sum(lengths) // mesh.size()) * dim * element * (degree - 1) // degree
    return {"ring_rank": rank, "ulysses_forward_backward_sent_bytes": ulysses_bytes,
            "worker_rows": plan.route.query_count, "capacity_rows": plan.capacity,
            "task_fragments": len(plan.assignments[rank]), "kernels_per_round": calls,
            "valid_qk_pairs_per_round": pairs, "fast_zigzag": plan.fast_zigzag,
            "redistribution_forward_sent_bytes": forward, "redistribution_backward_sent_bytes": backward,
            "ring_forward_backward_sent_bytes": ring_bytes if size > 1 else 0}


def _run_case(args, mesh, total, packed, degree):
    rank, size = mesh.get_local_rank(), mesh.size()
    local = total // size
    generator = torch.Generator().manual_seed(20261101 + rank)
    values = [torch.randn(local, heads, 128, generator=generator).bfloat16().to("npu")
              for heads in (16, 4, 4, 16)]
    inputs = tuple(tensor if packed else tensor.unsqueeze(0).transpose(1, 2).contiguous() for tensor in values)
    for tensor in inputs[:3]:
        tensor.requires_grad_()
    lengths = (33, 257, 3 * total // 4, total // 4 - 290) if packed else (total,)
    base = {"ulysses_degree": degree, "causal": True, "overlap": True,
            "input_layout": "TND" if packed else "BNSD", "document_lengths": lengths if packed else None}
    reference = (_full_reference(inputs, mesh, lengths, base["input_layout"], args.reference)
                 if args.reference in ("native", "fp32")
                 else _iteration(inputs, mesh, base))
    case = _Case(torch.bfloat16, base["input_layout"], 4, True, size, sequence_length=total,
                 query_heads=16, head_dim=128, batch_size=1)
    modes = tuple(args.modes)
    samples, options, errors = {}, {}, {}
    for mode in modes:
        options[mode] = {**base, "load_balance": mode,
                         "balance_block_size": args.block if mode in ("greedy", "hybrid") else None,
                         "zigzag_min_length": args.threshold if mode == "hybrid" else None}
        actual = _iteration(inputs, mesh, options[mode])
        errors[mode] = {}
        for name, value, expected in zip(("O", "dQ", "dK", "dV"), actual, reference):
            errors[mode][name] = _compare(value, expected, case, f"{mode} versus {args.reference} {name}")
        del actual
        for _ in range(args.warmup):
            _iteration(inputs, mesh, options[mode])
        samples[mode] = []
    del reference
    for repeat in range(args.repeats):
        for mode in modes if repeat % 2 == 0 else reversed(modes):
            samples[mode].append(_timed(inputs, mesh, options[mode], args.iterations))
    results = []
    for mode in modes:
        trace = RingAttentionTrace()
        for _ in range(args.repeats):
            _iteration(inputs, mesh, options[mode], trace)
        local_result = {"rank": rank, "errors": errors[mode],
                        "timing_samples": samples[mode], "round_timings": trace.timings(),
                        **_plan_metrics(lengths, mesh, degree, mode, options[mode]["balance_block_size"],
                                        options[mode]["zigzag_min_length"], True)}
        workers = [None] * size
        dist.all_gather_object(workers, local_result)
        round_times = []
        for phase in ("forward", "backward"):
            for step in range(size // degree):
                times = [median(item["milliseconds"] for item in worker["round_timings"]
                                if item["phase"] == phase and item["step"] == step) for worker in workers]
                round_times.append({"phase": phase, "step": step, "slowest_rank_ms": max(times),
                                    "slowest_rank": times.index(max(times)),
                                    "rank_milliseconds": times})
        result = {"global_tokens": total, "document_lengths": lengths, "layout": base["input_layout"],
                  "cp": size, "ulysses": degree, "ring": size // degree, "mode": mode,
                  "source_local_tokens": local, "ring_local_tokens": local * degree,
                  "block_size": options[mode]["balance_block_size"],
                  "zigzag_min_length": options[mode]["zigzag_min_length"],
                  "median_slowest_rank_ms": median(max(worker["timing_samples"][repeat]["device_ms"]
                                                       for worker in workers) for repeat in range(args.repeats)),
                  "median_slowest_wall_ms": median(max(worker["timing_samples"][repeat]["wall_ms"]
                                                       for worker in workers) for repeat in range(args.repeats)),
                  "max_peak_allocated_bytes": max(sample["peak_allocated_bytes"] for worker in workers
                                                  for sample in worker["timing_samples"]),
                  "max_peak_extra_bytes": max(sample["peak_extra_bytes"] for worker in workers
                                              for sample in worker["timing_samples"]),
                  "rounds": round_times, "workers": workers}
        if rank == 0:
            print(json.dumps({key: value for key, value in result.items() if key not in ("workers", "rounds")}),
                  flush=True)
        results.append(result)
    return results


def main() -> None:
    """Compare four strategies on dense/packed data with reusable JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    sequences = parser.add_mutually_exclusive_group()
    sequences.add_argument("--global-tokens", type=int, nargs="+")
    sequences.add_argument("--local-tokens", type=int, nargs="+")
    parser.add_argument("--layouts", nargs="+", choices=("BNSD", "TND"), default=["BNSD", "TND"])
    parser.add_argument("--reference", choices=("fp32", "native", "ring"), default="fp32")
    parser.add_argument("--ulysses-degrees", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--modes", nargs="+", choices=("none", "zigzag", "greedy", "hybrid"),
                        default=["none", "zigzag", "greedy", "hybrid"])
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("cp",))
        results = []
        totals = ([local * mesh.size() for local in args.local_tokens] if args.local_tokens
                  else args.global_tokens or [4096, 16384])
        for total in totals:
            for packed in (layout == "TND" for layout in args.layouts):
                for degree in args.ulysses_degrees:
                    results.extend(_run_case(args, mesh, total, packed, degree))
        if dist.get_rank() == 0:
            Path(args.output).write_text(json.dumps({
                "torch": torch.__version__, "device": torch.npu.get_device_name(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "parameters": vars(args), "results": results,
                "notes": ["NPU BF16 B=1 Hq=16 Hkv=4 D=128, dropout=0, causal=True",
                          "Uninstrumented end-to-end forward/backward timing includes all redistributions",
                          "Round events are measured separately without per-round barriers",
                          "Each reported round maximum uses the median duration for each rank",
                          "Traffic counts sent tensor payload, excluding transport protocol overhead",
                          "Peak memory counts Torch allocated bytes, including Ring padding",
                          "Extra peak memory subtracts warm input/metadata allocations",
                          "Plans are warmed and cached; changing document metadata is outside "
                          "this steady-state timing"]}, indent=2) + "\n",
                                         encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
