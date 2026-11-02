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
"""Opt-in torchrun benchmark; timing is reported, never used as a CI assertion.

Example: python -m torch.distributed.run --standalone --nproc-per-node=4
    tests/torch/context_parallel/_benchmark_ring_attention_overlap.py
    --sequences 512 2048 --profile-dir /tmp/ring-profile --output /tmp/ring-timing.json
"""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel import init_device_mesh
from hyper_parallel.distributed.context_parallel.ring_attention import ulysses_ring_attention
from tests.torch.context_parallel._test_ring_attention_npu import _Case, _compare, _inputs


def _iteration(inputs, mesh, degree, overlap):
    output = ulysses_ring_attention(
        *inputs[:3], mesh, ulysses_degree=degree, causal=True, overlap=overlap,
    )
    gradients = torch.autograd.grad(output, inputs[:3], grad_outputs=inputs[3])
    return (output, *gradients)


def _measure(inputs, mesh, degree, overlap, iterations):
    dist.barrier()
    torch.npu.synchronize()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    wall_start = perf_counter()
    begin.record()
    for _ in range(iterations):
        _iteration(inputs, mesh, degree, overlap)
    end.record()
    end.synchronize()
    elapsed = torch.tensor(
        [begin.elapsed_time(end) / iterations, (perf_counter() - wall_start) * 1000 / iterations],
        dtype=torch.float32, device=inputs[0].device,
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.cpu().tolist()


def _profile(inputs, mesh, degree, overlap, directory):
    rank = mesh.get_local_rank()
    context = nullcontext()
    if rank == mesh.size() - 1:
        context = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=torch_npu.profiler._ExperimentalConfig(  # pylint: disable=protected-access
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                directory, worker_name=f"cp{mesh.size()}_u{degree}_overlap{int(overlap)}_rank{rank}",
            ),
        )
    with context:
        dist.barrier()
        for _ in range(2):
            _iteration(inputs, mesh, degree, overlap)
        torch.npu.synchronize()
        dist.barrier()


def _run_case(arguments, mesh, sequence, degree):
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    case = _Case(torch.bfloat16, "BNSD", 4, True, 1, sequence_length=sequence,
                 query_heads=16, head_dim=128, batch_size=1)
    inputs = tuple((tensor + mesh.get_local_rank() / 16).to(device).requires_grad_(index < 3)
                   for index, tensor in enumerate(_inputs(case)))
    expected = _iteration(inputs, mesh, degree, False)
    actual = _iteration(inputs, mesh, degree, True)
    for name, value, reference in zip(("output", "dQ", "dK", "dV"), actual, expected):
        _compare(value, reference, case, f"overlap vs sync: {name}")
    del actual, expected
    samples = {False: [], True: []}
    for overlap in (False, True):
        for _ in range(arguments.warmup):
            _iteration(inputs, mesh, degree, overlap)
    for repeat in range(arguments.repeats):
        for overlap in ((False, True) if repeat % 2 == 0 else (True, False)):
            samples[overlap].append(_measure(inputs, mesh, degree, overlap, arguments.iterations))
    result = {
        "cp_degree": mesh.size(), "ulysses_degree": degree, "ring_degree": mesh.size() // degree,
        "local_sequence": sequence, "global_sequence": sequence * mesh.size(),
        "batch": 1, "query_heads": 16, "kv_heads": 4, "head_dim": 128, "dtype": "bfloat16",
        "causal": True, "iterations": arguments.iterations, "repeats": arguments.repeats,
        "sync_ms": median(sample[0] for sample in samples[False]),
        "overlap_ms": median(sample[0] for sample in samples[True]),
        "sync_wall_ms": median(sample[1] for sample in samples[False]),
        "overlap_wall_ms": median(sample[1] for sample in samples[True]),
        "sync_samples_ms": samples[False], "overlap_samples_ms": samples[True],
    }
    result["speedup"] = result["sync_ms"] / result["overlap_ms"]
    if mesh.get_local_rank() == 0:
        print(json.dumps(result), flush=True)
    if arguments.profile_dir and sequence == max(arguments.sequences):
        for overlap in (False, True):
            _profile(inputs, mesh, degree, overlap, arguments.profile_dir)
    return result


def main() -> None:
    """Measure alternating sync/overlap runs and optionally collect device timelines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--ulysses-degrees", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-dir")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("cp",))
        results = [_run_case(arguments, mesh, sequence, degree)
                   for degree in arguments.ulysses_degrees for sequence in arguments.sequences]
        if arguments.output and dist.get_rank() == 0:
            Path(arguments.output).write_text(json.dumps({
                "torch": torch.__version__, "torch_npu": torch_npu.__version__, "results": results,
            }, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
