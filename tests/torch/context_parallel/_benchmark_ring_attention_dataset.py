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
"""Opt-in pure Ring benchmark driven by natural document boundaries in indexed data.

First run --prepare --dataset-prefix PREFIX --manifest PATH to record contiguous
packs from a seeded document shuffle, plus explicitly labelled ordered/long cases.
Then run this module with torchrun and --manifest PATH --output RESULTS. All modes
use identical random Q/K/V and an independent FP32 reference. This measures the
attention operator with real document lengths, not model training on token values.
"""

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel import init_device_mesh
from hyper_parallel.data.indexed.indexed_data_reader import IndexedDataReader
from hyper_parallel.distributed.context_parallel.query_balance import query_route, query_task_plan
from hyper_parallel.distributed.context_parallel.ring_attention import ring_attention
from hyper_parallel.distributed.context_parallel.ring_layout import RingAttentionTrace, ring_task_plan
from tests.torch.context_parallel._benchmark_ring_attention_layout import _full_reference, _plan_metrics
from tests.torch.context_parallel._test_ring_attention_npu import _Case, _compare


def _pack_documents(lengths, order, total, count, kind):
    position, offset, samples = 0, 0, []
    for number in range(count):
        remaining, segments = total, []
        while remaining:
            if position >= len(order):
                raise ValueError("Dataset exhausted before all complete packs could be formed")
            document = int(order[position])
            take = min(remaining, int(lengths[document]) - offset)
            if take:
                segments.append({"document": document, "offset": offset, "length": take})
                remaining -= take
                offset += take
            if offset == lengths[document]:
                position, offset = position + 1, 0
        pieces = [segment["length"] for segment in segments]
        samples.append({"name": f"{kind}_{number:03d}", "kind": kind, "segments": segments,
                        "document_lengths": pieces, "document_count": len(pieces),
                        "max_document_length": max(pieces),
                        "valid_qk_pairs": sum(length * (length + 1) // 2 for length in pieces)})
    return samples


def _prepare(args):
    if not args.dataset_prefix:
        raise ValueError("--prepare requires --dataset-prefix")
    reader = IndexedDataReader(args.dataset_prefix)
    index = reader.index
    cumulative = np.concatenate(([0], np.cumsum(index.sequence_lengths, dtype=np.int64)))
    lengths = np.diff(cumulative[index.document_indices])
    total = args.cp * args.local_tokens
    order = np.random.default_rng(args.seed).permutation(len(lengths))
    samples = _pack_documents(lengths, order, total, args.batches, "shuffled")
    samples += _pack_documents(lengths, np.arange(len(lengths)), total, 1, "ordered")
    long_documents = np.flatnonzero(lengths >= total)
    if len(long_documents):
        samples += _pack_documents(lengths, long_documents[:1], total, 1, "long")
    manifest = {"dataset_prefix": args.dataset_prefix,
                "index_sha256": hashlib.sha256(Path(args.dataset_prefix + ".idx").read_bytes()).hexdigest(),
                "token_dtype": str(index.dtype), "documents": len(lengths),
                "sequences": index.sequence_count, "total_dataset_tokens": int(lengths.sum()),
                "document_length_percentiles": dict(zip(("min", "p25", "p50", "p75", "p90", "p95", "p99",
                                                          "p999", "max"),
                                                         np.percentile(lengths, (0, 25, 50, 75, 90, 95, 99,
                                                                                 99.9, 100)).tolist())),
                "seed": args.seed, "cp": args.cp, "local_tokens": args.local_tokens,
                "global_tokens": total, "samples": samples,
                "packing": "Consecutive shuffled natural documents, splitting only at pack boundaries; "
                           "document attention resets at each boundary. No extra BOS/EOS or label shift.",
                "scope": "Random activations with real document lengths; no end-to-end model training."}
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": args.manifest, "documents": len(lengths), "samples": len(samples)}), flush=True)


def _iteration(inputs, mesh, options, trace=None):
    output = ring_attention(*inputs[:3], mesh, trace=trace, **options)
    return (output, *torch.autograd.grad(output, inputs[:3], inputs[3]))


def _measure(inputs, mesh, options, iterations):
    dist.barrier()
    torch.npu.synchronize()
    baseline = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
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


def _validate(actual, expected, case, label, record_errors):
    observed, target = actual.detach().cpu().double(), expected.detach().cpu().double()
    difference = (observed - target).abs()
    epsilon = torch.finfo(case.dtype).eps
    finite = bool(torch.isfinite(observed).all() and torch.isfinite(target).all())
    failed = int((difference > 2 * epsilon + 0.06 * target.abs()).sum())
    relative_l2 = float(difference.norm() / target.norm().clamp_min(1e-12))
    passed = finite and failed == 0 and relative_l2 <= epsilon
    if not record_errors:
        _compare(actual, expected, case, label)
    return {"passed": passed, "finite": finite, "failed_elements": failed,
            "elements": observed.numel(), "max_abs": float(difference.max()), "relative_l2": relative_l2}


def _options(args, lengths):
    return {f"{mode}_overlap{overlap}": {
        "input_layout": "TND", "document_lengths": lengths, "causal": True, "overlap": bool(overlap),
        "load_balance": mode, "balance_block_size": args.block if mode in ("greedy", "hybrid") else None,
        "zigzag_min_length": args.threshold if mode == "hybrid" else None,
    } for mode in args.modes for overlap in args.overlap}


def _trace(inputs, mesh, options, repeats):
    trace = RingAttentionTrace()
    for _ in range(repeats):
        _iteration(inputs, mesh, options, trace)
    return trace.timings()


def _profile(inputs, mesh, options, directory, name, iterations):
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=torch_npu.profiler._ExperimentalConfig(  # pylint: disable=protected-access
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                directory, worker_name=f"{name}_rank{mesh.get_local_rank()}")) as profile:
        dist.barrier()
        torch.npu.synchronize()
        for iteration in range(iterations):
            with torch.profiler.record_function(f"{name}_iteration_{iteration}"):
                _iteration(inputs, mesh, options)
            profile.step()
        torch.npu.synchronize()
        dist.barrier()


def _summarize(sample, options, workers, repeats):
    times = [max(worker["timing_samples"][repeat]["device_ms"] for worker in workers)
             for repeat in range(repeats)]
    walls = [max(worker["timing_samples"][repeat]["wall_ms"] for worker in workers)
             for repeat in range(repeats)]
    pairs = [sum(worker["valid_qk_pairs_per_round"]) for worker in workers]
    if sum(pairs) != sample["valid_qk_pairs"]:
        raise ValueError("Executable blocks must cover every valid document-local QK pair exactly once")
    rounds = []
    for phase in ("forward", "backward"):
        for step in range(len(workers)):
            durations = [median(item["milliseconds"] for item in worker["round_timings"]
                                if item["phase"] == phase and item["step"] == step) for worker in workers]
            rounds.append({"phase": phase, "step": step, "rank_milliseconds": durations,
                           "slowest_rank_ms": max(durations), "slowest_rank": durations.index(max(durations))})
    failures = {name: sum(worker["errors"][name]["failed_elements"] for worker in workers)
                for name in ("O", "dQ", "dK", "dV")}
    return {"numerical_pass": all(error["passed"] for worker in workers for error in worker["errors"].values()),
            "failed_elements": failures,
            "max_relative_l2": max(error["relative_l2"] for worker in workers for error in worker["errors"].values()),
            "sample": sample["name"], "kind": sample["kind"], "mode": options["load_balance"],
            "overlap": options["overlap"], "block": options["balance_block_size"],
            "threshold": options["zigzag_min_length"], "document_count": sample["document_count"],
            "max_document_length": sample["max_document_length"],
            "slowest_rank_samples_ms": times, "median_slowest_rank_ms": median(times),
            "slowest_wall_samples_ms": walls, "median_slowest_wall_ms": median(walls),
            "valid_qk_imbalance_max_over_mean": max(pairs) * len(pairs) / sum(pairs),
            "max_peak_extra_bytes": max(item["peak_extra_bytes"] for worker in workers
                                        for item in worker["timing_samples"]),
            "max_peak_allocated_bytes": max(item["peak_allocated_bytes"] for worker in workers
                                            for item in worker["timing_samples"]),
            "max_live_growth_bytes": max(item["live_growth_bytes"] for worker in workers
                                         for item in worker["timing_samples"]),
            "rounds": rounds, "workers": workers}


def _run_sample(args, mesh, sample):
    lengths = tuple(sample["document_lengths"])
    local, rank = sum(lengths) // mesh.size(), mesh.get_local_rank()
    generator = torch.Generator().manual_seed(args.seed + rank)
    inputs = tuple(torch.randn(local, heads, 128, generator=generator).bfloat16().to("npu")
                   .requires_grad_(index < 3) for index, heads in enumerate((16, 4, 4, 16)))
    reference = _full_reference(inputs, mesh, lengths, "TND", "fp32")
    case = _Case(torch.bfloat16, "TND", 4, True, mesh.size(), sequence_length=sum(lengths),
                 query_heads=16, head_dim=128, batch_size=1)
    native_errors = {}
    if args.native_check:
        native = _full_reference(inputs, mesh, lengths, "TND", "native")
        native_errors = {name: _validate(value, expected, case, f"{sample['name']}/native/{name}", args.record_errors)
                         for name, value, expected in zip(("O", "dQ", "dK", "dV"), native, reference)}
        del native
    options, errors, metrics = _options(args, lengths), {}, {}
    for key, option in options.items():
        mode = option["load_balance"]
        if mode not in metrics:
            for cached in (ring_task_plan, query_route, query_task_plan):
                cached.cache_clear()
            begin = perf_counter()
            plan_metrics = _plan_metrics(lengths, mesh, 1, mode, option["balance_block_size"],
                                         option["zigzag_min_length"], True)
            metrics[mode] = {**plan_metrics, "cold_host_plan_ms": (perf_counter() - begin) * 1000}
        actual = _iteration(inputs, mesh, option)
        errors[key] = {name: _validate(value, expected, case, f"{sample['name']}/{key}/{name}", args.record_errors)
                       for name, value, expected in zip(("O", "dQ", "dK", "dV"), actual, reference)}
        del actual
    del reference
    for option in options.values():
        for _ in range(args.warmup):
            _iteration(inputs, mesh, option)
    samples = {key: [] for key in options}
    keys = tuple(options)
    # Rotate and reverse the order deterministically to distribute temporal drift.
    for repeat in range(args.repeats):
        shifted = keys[repeat % len(keys):] + keys[:repeat % len(keys)]
        for key in shifted if repeat % 2 == 0 else reversed(shifted):
            samples[key].append(_measure(inputs, mesh, options[key], args.iterations))
    results = []
    for key, option in options.items():
        worker = {"rank": rank, "errors": errors[key], "native_errors": native_errors, "timing_samples": samples[key],
                  "round_timings": _trace(inputs, mesh, option, args.trace_repeats),
                  **metrics[option["load_balance"]]}
        workers = [None] * mesh.size()
        dist.all_gather_object(workers, worker)
        result = _summarize(sample, option, workers, args.repeats)
        results.append(result)
        if rank == 0:
            print(json.dumps({key: value for key, value in result.items() if key not in ("workers", "rounds")}),
                  flush=True)
    if args.profile_dir:
        for key, option in options.items():
            _profile(inputs, mesh, option, args.profile_dir, f"{sample['name']}_{key}", args.profile_iterations)
    return results


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dataset-prefix")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cp", type=int, default=8)
    parser.add_argument("--local-tokens", type=int, default=8192)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=("none", "zigzag", "greedy", "hybrid"),
                        default=["none", "zigzag", "greedy", "hybrid"])
    parser.add_argument("--overlap", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    parser.add_argument("--block", type=int, default=2048)
    parser.add_argument("--threshold", type=int, default=16384)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trace-repeats", type=int, default=2)
    parser.add_argument("--profile-dir")
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument("--output")
    parser.add_argument("--record-errors", action="store_true",
                        help="Diagnostic only: record failed original accuracy gates and continue timing")
    parser.add_argument("--native-check", action="store_true")
    args = parser.parse_args()
    for field in ("cp", "local_tokens", "batches", "block", "threshold", "iterations", "warmup", "repeats",
                  "trace_repeats", "profile_iterations"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if not args.prepare and not args.output:
        parser.error("--output is required for distributed measurements")
    return args


def main() -> None:
    """Prepare reproducible natural-document packs or benchmark their pure Ring execution."""
    args = _arguments()
    if args.prepare:
        _prepare(args)
        return
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    selected = [sample for sample in manifest["samples"] if not args.cases or sample["name"] in args.cases]
    if not selected or (args.cases and set(args.cases) != {sample["name"] for sample in selected}):
        raise ValueError("--cases must select existing manifest samples")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=300))
    try:
        mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("cp",))
        if mesh.size() != manifest["cp"]:
            raise ValueError("The manifest CP degree must match the launched world size")
        report = {"parameters": vars(args), "manifest": manifest, "results": [], "complete": False,
                  "cp": mesh.size(), "ulysses": 1, "device": torch.npu.get_device_name(),
                  "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                  "task_queue_enable": os.environ.get("TASK_QUEUE_ENABLE", "default"),
                  "notes": ["TND BF16 Hq=16 Hkv=4 D=128, causal, dropout=0; direct ring_attention API",
                            "Independent FP32 softmax/derivative reference with HF32 disabled",
                            "End-to-end operator timing includes forward, backward and all row redistribution",
                            "Warm plans/indices; cold CPU plan time is reported separately",
                            "Round events and optional profiler are outside throughput measurements",
                            "Payload estimates count sent bytes, not wire overhead",
                            "BF16 accuracy gates unchanged: atol=0.015625, rtol=0.06, relative L2<=0.0078125",
                            "With --record-errors, numerical_pass=False remains a failed validation"]}
        for sample in selected:
            if sum(sample["document_lengths"]) != manifest["global_tokens"]:
                raise ValueError("All sample lengths must sum to the manifest's global token count")
            if mesh.get_local_rank() == 0:
                print(json.dumps({"starting": sample["name"], "documents": sample["document_count"]}), flush=True)
            report["results"].extend(_run_sample(args, mesh, sample))
            report["complete"] = sample == selected[-1]
            if mesh.get_local_rank() == 0:
                Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
