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
"""Portable research workers; launch explicitly with torchrun's module mode."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist

from . import benchmark
from .protocol_sources import build_protocols


def summarize(ranks: list[dict], suite: str) -> dict:
    """Reduce every iteration across ranks before reporting its median."""
    result = {}
    for case in ranks[0]["cases"]:
        result[case] = {}
        for name in ranks[0]["cases"][case]:
            rows = [rank["cases"][case][name] for rank in ranks]
            times = [row["timing"]["median"]["total_ms"] for row in rows]
            cache_bytes = [row["correctness"]["cache"]["unique_storage_bytes"] for row in rows]
            iteration_max = [max(row["timing"]["samples"][index]["total_ms"] for row in rows)
                             for index in range(len(rows[0]["timing"]["samples"]))]
            passed = all(row["correctness"]["pass"] for row in rows)
            passed = passed and all(metric["pass"] for row in rows for metric in row["vs_first"].values())
            if suite == "g2":
                passed = passed and all(row["baseline_bitwise_equal"] for row in rows)
            result[case][name] = {
                "pass": passed,
                "cache_bytes_by_rank": cache_bytes, "cache_sum_bytes": sum(cache_bytes),
                "cache_max_bytes": max(cache_bytes), "median_total_ms_by_rank": times,
                "slowest_rank": max(range(len(times)), key=times.__getitem__),
                "max_rank_median_total_ms": max(times),
                "iteration_max_rank_total_ms": iteration_max,
                "median_iteration_max_rank_total_ms": statistics.median(iteration_max),
                "peak_allocated_delta_bytes_by_rank": [row["timing"]["npu_peak_allocated_delta_bytes"]
                                                       for row in rows],
                "live_after_forward_delta_bytes_by_rank": [row["timing"]["npu_live_after_forward_delta_bytes"]
                                                           for row in rows],
                "max_relative_l2": {interface: max(row["correctness"]["interfaces"][interface]["relative_l2"]
                                                    for row in rows)
                                    for interface in ("H", "D", "second_H", "second_D")},
            }
            if suite == "g2":
                result[case][name]["baseline_bitwise_equal"] = all(row["baseline_bitwise_equal"] for row in rows)
    return result


def parse_arguments(suite: str) -> argparse.Namespace:
    """Validate run configuration before creating process groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--variants", nargs="+", choices=("baseline", "lifetime", "direct"),
                        default=("baseline", "lifetime", "direct"))
    parser.add_argument("--protocols", nargs="+", choices=("flat", "head", "tree"), default=("flat", "head", "tree"))
    parser.add_argument("--group-width", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--source-dir", type=Path, help="External rankN_boundaries.pt inputs containing S, M and G")
    parser.add_argument("--source-ranks", type=int, default=8)
    parser.add_argument("--repeat-captures", action="store_true")
    parser.add_argument("--force-public", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--cases", choices=("long", "reset", "both"), default="both")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-dir", type=Path, help="Optional separate aligned NPU Level0 export directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.heads < 1 or args.batch < 1 or args.source_ranks < 1:
        parser.error("iterations, heads, batch, source-ranks must be positive; warmup must be nonnegative")
    if args.profile_dir is not None and args.device != "npu":
        parser.error("--profile-dir requires --device npu")
    if args.profile_only and (args.profile_dir is None or args.device != "npu"):
        parser.error("--profile-only requires --device npu and --profile-dir")
    if args.profile_only and args.output.exists():
        parser.error("--profile-only requires a new output path")
    if suite == "g2" and (args.variants[0] != "baseline" or len(set(args.variants)) != len(args.variants)):
        parser.error("--variants must begin with baseline and contain no duplicates")
    if len(set(args.protocols)) != len(args.protocols):
        parser.error("--protocols must contain no duplicates")
    return args


def run_case(protocols: dict, args: argparse.Namespace, suite: str, case: str, rank: int, size: int) -> tuple:
    """Check two live invocations, then time protocols on the same local inputs."""
    values = benchmark.make_inputs(args, size, reset=case == "reset")
    if (suite == "g2" or "head" in args.protocols) and values[0][0].shape[1] % args.group_width:
        raise ValueError("Summary head count must be divisible by owner width")
    local = tuple(pieces[rank].to(args.device).contiguous() for pieces in values)
    rows, first_outputs = {}, None
    for name, protocol in protocols.items():
        correctness, outputs = benchmark.run_correctness(protocol, values, rank, args.device)
        if first_outputs is None:
            first_outputs = outputs
        difference = {key: benchmark.metrics(actual, expected) for key, actual, expected
                      in zip(("H", "D", "second_H", "second_D"), outputs[:4], first_outputs[:4])}
        row = {"correctness": correctness, "vs_first": difference}
        if suite == "g2":
            equal = [torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                     for actual, expected in zip(outputs, first_outputs)]
            row["baseline_bitwise_equal"] = len(outputs) == len(first_outputs) and all(equal)
            row["compared_tensor_equal"] = equal
        row["timing"] = benchmark.run_timing(protocol, local, args)
        rows[name] = row
    return rows, list(values[0][0].shape)


def run(suite: str) -> None:
    """Run one suite with explicit groups and no production configuration writes.

    Args:
        suite: ``protocols`` compares flat RD, head RD and the cached tree;
            ``g2`` checks baseline/lifetime/direct including every cache byte.
    """
    args = parse_arguments(suite)
    torch.set_num_threads(1)
    if args.device == "npu":
        importlib.import_module("torch_npu")
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
        torch.npu.matmul.allow_hf32 = False
        torch.npu.config.allow_internal_format = False
    dist.init_process_group("hccl" if args.device == "npu" else "gloo", timeout=timedelta(seconds=240))
    rank, size = dist.get_rank(), dist.get_world_size()
    protocols, sources = build_protocols(args.group_width, args.force_public, suite,
                                         tuple(args.variants), tuple(args.protocols))
    row = {"rank": rank, "cases": {}}
    cases = () if args.profile_only else (("long", "reset") if args.cases == "both" else (args.cases,))
    for case in cases:
        row["cases"][case], row["input_shape"] = run_case(protocols, args, suite, case, rank, size)
    row["profiles"] = benchmark.run_profiles(protocols, args, rank, size)
    gathered = [None] * size
    dist.all_gather_object(gathered, row)
    summary = {} if args.profile_only else summarize(gathered, suite)
    passed = all(protocol["pass"] for case in summary.values() for protocol in case.values())
    if rank == 0:
        config = {key: value for key, value in vars(args).items()
                  if key not in ("source_dir", "profile_dir", "output")}
        config.update({"suite": suite, "input_kind": "external_capture" if args.source_dir else "synthetic",
                       "profiles_requested": args.profile_dir is not None})
        result = {"config": config, "world_size": size, "sources": sources, "ranks": gathered, "summary": summary,
                  "pass": None if args.profile_only else passed, "validation_performed": not args.profile_only,
                  "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "benchmark_sha256": hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest(),
                  "scope": "Same S/M/G, FP32 protocols vs sequential FP64 affine reference; not full KDA or training.",
                  "constructor_scope": "Explicit natural WORLD groups; no DeviceMesh qualification.",
                  "reset_scope": "Structural M=0 algebra only; no reset plan or message pruning.",
                  "gate": "norm(error)<=2e-6*norm(reference)+1e-11; not a model training tolerance."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"pass": result["pass"], "summary": summary}), flush=True)
    dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)
