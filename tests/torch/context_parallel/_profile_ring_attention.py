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
"""Opt-in pure Ring jitter diagnosis; pacing experiments never alter production code.

Queued, local-sync and barrier pacing expose rank drift separately from raw
throughput. Optional all-rank CPU/NPU traces run after unprofiled measurements.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel import init_device_mesh
from hyper_parallel.distributed.context_parallel.ring_attention import ring_attention


def _iteration(inputs, mesh, overlap):
    with torch.profiler.record_function("ring_forward"):
        output = ring_attention(*inputs[:3], mesh, causal=True, overlap=overlap)
    with torch.profiler.record_function("ring_backward"):
        torch.autograd.grad(output, inputs[:3], inputs[3])


def _measure(inputs, mesh, overlap, pacing, iterations):
    dist.barrier()
    torch.npu.synchronize()
    events, submissions = [], []
    start = perf_counter()
    for _ in range(iterations):
        begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        host = perf_counter()
        begin.record()
        _iteration(inputs, mesh, overlap)
        end.record()
        submissions.append((perf_counter() - host) * 1000)
        events.append((begin, end))
        if pacing != "queued":
            end.synchronize()
        if pacing == "barrier":
            dist.barrier()
            torch.npu.synchronize()
    events[-1][1].synchronize()
    return {"rank": mesh.get_local_rank(), "batch_wall_ms": (perf_counter() - start) * 1000,
            "iteration_device_ms": [begin.elapsed_time(end) for begin, end in events],
            "submission_ms": submissions}


def _profile(inputs, mesh, overlap, directory, iterations, pacing):
    # Every rank is profiled; tracing only one rank would introduce asymmetric host load.
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=torch_npu.profiler._ExperimentalConfig(  # pylint: disable=protected-access
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                directory, worker_name=f"ring_overlap{int(overlap)}_rank{mesh.get_local_rank()}")) as profile:
        dist.barrier()
        torch.npu.synchronize()
        events = []
        for iteration in range(iterations):
            begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            begin.record()
            with torch.profiler.record_function(f"ring_iteration_{iteration}"):
                _iteration(inputs, mesh, overlap)
            end.record()
            events.append((begin, end))
            if pacing != "queued":
                end.synchronize()
            if pacing == "barrier":
                dist.barrier()
                torch.npu.synchronize()
            profile.step()
        torch.npu.synchronize()
        dist.barrier()
    Path(directory, f"device_timings_overlap{int(overlap)}_rank{mesh.get_local_rank()}.json").write_text(
        json.dumps({"pacing": pacing, "milliseconds": [begin.elapsed_time(end) for begin, end in events]}) + "\n",
        encoding="utf-8")


def main() -> None:
    """Collect per-iteration duration and host-submission evidence for pure Ring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-tokens", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--pacing", nargs="+", choices=("queued", "sync", "barrier"),
                        default=["queued", "sync", "barrier"])
    parser.add_argument("--overlap", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--profile-dir")
    parser.add_argument("--profile-iterations", type=int, default=4)
    parser.add_argument("--profile-pacing", choices=("queued", "sync", "barrier"), default="queued")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("cp",))
        generator = torch.Generator().manual_seed(20261101 + mesh.get_local_rank())
        inputs = tuple(torch.randn(args.local_tokens, heads, 128, generator=generator).bfloat16()
                       .unsqueeze(0).transpose(1, 2).contiguous().to("npu").requires_grad_(index < 3)
                       for index, heads in enumerate((16, 4, 4, 16)))
        results = []
        combinations = [(bool(overlap), pacing) for overlap in args.overlap for pacing in args.pacing]
        for overlap in args.overlap:
            for _ in range(args.warmup):
                _iteration(inputs, mesh, bool(overlap))
        for repeat in range(args.repeats):
            for overlap, pacing in combinations if repeat % 2 == 0 else reversed(combinations):
                measured = _measure(inputs, mesh, overlap, pacing, args.iterations)
                workers = [None] * mesh.size()
                dist.all_gather_object(workers, measured)
                result = {"repeat": repeat, "overlap": overlap, "pacing": pacing, "workers": workers}
                results.append(result)
                if mesh.get_local_rank() == 0:
                    print(json.dumps({"repeat": repeat, "overlap": overlap, "pacing": pacing,
                                      "slowest_wall_per_iteration_ms": max(item["batch_wall_ms"]
                                                                           for item in workers) / args.iterations}),
                          flush=True)
        if mesh.get_local_rank() == 0:
            Path(args.output).write_text(json.dumps({"parameters": vars(args), "cp": mesh.size(),
                "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                "task_queue_enable": os.environ.get("TASK_QUEUE_ENABLE", "default"), "results": results,
                "notes": ["Device duration excludes pacing barriers; batch wall time includes them",
                          "CPU submission includes backpressure but does not wait for device completion",
                          "Profiler runs are separate and are not throughput samples"]}, indent=2) + "\n",
                                        encoding="utf-8")
        if args.profile_dir:
            for overlap in args.overlap:
                _profile(inputs, mesh, bool(overlap), args.profile_dir, args.profile_iterations, args.profile_pacing)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
