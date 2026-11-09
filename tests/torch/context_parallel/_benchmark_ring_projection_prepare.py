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
"""Opt-in projection/preparation overlap experiment; does not change Ring defaults.

Run with torchrun and an explicit dataset length manifest. The measured block is
Q/K/V projections, causal packed Ring, output projection, and all local gradients.
Weights are replicated without parameter-gradient synchronization. This is not a
Trainer benchmark and deliberately excludes model-specific QK norm and RoPE.
"""

import argparse
from datetime import timedelta
from functools import lru_cache, partial
import importlib
import json
import os
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.functional import linear
import torch_npu  # pylint: disable=unused-import

from hyper_parallel import init_device_mesh
from hyper_parallel.core.utils.communication import (
    differentiable_all_to_all_single,
)
from hyper_parallel.distributed.context_parallel.ring_layout import (
    _route_indices,
    redistribute_ring_rows,
    ring_task_plan,
)


RING = importlib.import_module("hyper_parallel.distributed.context_parallel.ring_attention")
MODES = ("none", "baseline", "prepack_serial", "prepack_overlap", "qcomm_serial", "qcomm_overlap")


@lru_cache(maxsize=32)
def _packing_indices(route, device):
    return torch.tensor(route.send_rows + route.local_sources, dtype=torch.long, device=device)


def _prepare(tensor, plan, stream):
    indices = _packing_indices(plan.route, tensor.device)
    if stream is None:
        return tensor.index_select(0, indices), None
    current = torch.npu.current_stream()
    stream.wait_stream(current)
    tensor.record_stream(stream)
    indices.record_stream(stream)
    with torch.npu.stream(stream):
        result = tensor.index_select(0, indices)
        ready = torch.npu.Event()
        ready.record()
    result.record_stream(current)
    return result, ready


def _wait_prepared(parts):
    for _, ready in parts:
        if ready is not None:
            ready.wait(torch.npu.current_stream())
    return [tensor for tensor, _ in parts]


def _assemble(remote, local, plan):
    _, receive, _, targets = _route_indices(plan.route, local.device)
    result = local.new_zeros((plan.capacity, *local.shape[1:]))
    result.index_put_((targets,), local)
    result.index_put_((receive,), remote)
    return result


def _route_prepared(parts, plan, group):
    tensors = _wait_prepared(parts)
    count = len(plan.route.send_rows)
    send = torch.cat([tensor[:count] for tensor in tensors], dim=1)
    local = torch.cat([tensor[count:] for tensor in tensors], dim=1)
    remote = differentiable_all_to_all_single(
        send, list(plan.route.input_splits), list(plan.route.output_splits), group,
    )
    return _assemble(remote, local, plan)


class _WaitRoute(torch.autograd.Function):
    """Attach a pre-launched row exchange with an explicit inverse in backward."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, output: torch.Tensor, work: Any, group: Any,
                input_splits: list[int], output_splits: list[int]) -> torch.Tensor:
        """Keep the source gradient edge and wait before exposing received rows."""
        del tensor
        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        work.wait()
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return each received row gradient to its original source."""
        send = grad_output.contiguous()
        output = send.new_empty((sum(ctx.input_splits), *send.shape[1:]))
        work = dist.all_to_all_single(
            output, send, output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits, group=ctx.group, async_op=True,
        )
        work.wait()
        return output, None, None, None, None, None, None


def _launch_route(tensor, plan, group):
    packed, _ = _prepare(tensor, plan, None)
    count = len(plan.route.send_rows)
    source = packed[:count]
    output = source.new_empty((sum(plan.route.output_splits), *source.shape[1:]))
    inputs, outputs = list(plan.route.input_splits), list(plan.route.output_splits)
    work = dist.all_to_all_single(
        output, source.detach(), output_split_sizes=outputs, input_split_sizes=inputs,
        group=group, async_op=True,
    )
    return (source, output, work, group, inputs, outputs), packed[count:]


def _finish_route(pending, plan):
    arguments, local = pending
    remote = _WaitRoute.apply(*arguments)
    return _assemble(remote, local, plan)


# PyTorch exports this native callable through _add_docstr; pylint infers its return incorrectly.
# pylint: disable=not-callable
def _block(inputs, mesh, lengths, plan, ring, stream, mode, backward=True):
    hidden, query_weight, key_weight, value_weight, output_weight, grad_output = inputs
    query = linear(hidden, query_weight).view(-1, 16, 128)
    parts, pending = [], None
    if mode.startswith("prepack"):
        parts.append(_prepare(query, plan, stream if mode == "prepack_overlap" else None))
    elif mode.startswith("qcomm"):
        pending = _launch_route(query, plan, ring.group)
        if mode == "qcomm_serial":
            query = _finish_route(pending, plan)
    key = linear(hidden, key_weight).view(-1, 4, 128)
    if mode.startswith("prepack"):
        parts.append(_prepare(key, plan, stream if mode == "prepack_overlap" else None))
    value = linear(hidden, value_weight).view(-1, 4, 128)
    if mode in ("baseline", "none"):
        output = RING.ring_attention(
            query, key, value, mesh, input_layout="TND", causal=True, overlap=True,
            load_balance="zigzag" if mode == "baseline" else "none", document_lengths=lengths,
        )
    else:
        if mode.startswith("prepack"):
            parts.append(_prepare(value, plan, stream if mode == "prepack_overlap" else None))
            query, key, value = _route_prepared(parts, plan, ring.group).split((16, 4, 4), dim=1)
        else:
            kv_pending = _launch_route(torch.cat((key, value), dim=1), plan, ring.group)
            key, value = _finish_route(kv_pending, plan).split((4, 4), dim=1)
            if mode == "qcomm_overlap":
                query = _finish_route(pending, plan)
        output = RING._RingAttention.apply(  # pylint: disable=protected-access
            query, key, value, ring, True, 128 ** -0.5, "TND", True, plan,
        )
        output = redistribute_ring_rows(output, plan, ring.group, 0, inverse=True)
    output = linear(output.flatten(1), output_weight)
    if not backward:
        return output
    return (output, *torch.autograd.grad(output, inputs[:5], grad_output))

# pylint: enable=not-callable


def _error(actual, expected):
    difference = actual.detach().float() - expected.detach().float()
    return {"bitwise_equal": bool(torch.equal(actual, expected)),
            "finite": bool(torch.isfinite(actual).all()),
            "max_abs": float(difference.abs().max()),
            "relative_l2": float(difference.norm() / expected.detach().float().norm().clamp_min(1e-12))}


def _measure(function, iterations):
    dist.barrier()
    torch.npu.synchronize()
    baseline = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    begin.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return {"ms": begin.elapsed_time(end) / iterations,
            "peak_extra_bytes": torch.npu.max_memory_allocated() - baseline,
            "live_growth_bytes": torch.npu.memory_allocated() - baseline}



def _profile(functions, modes, directory, rank):
    for mode in modes:
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                experimental_config=torch_npu.profiler._ExperimentalConfig(  # pylint: disable=protected-access
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    directory, worker_name=f"{mode}_rank{rank}")) as profile:
            dist.barrier()
            torch.npu.synchronize()
            for _ in range(3):
                functions[mode]()
                profile.step()
            torch.npu.synchronize()
            dist.barrier()


def _forward_measurements(args, inputs, mesh, lengths, plan, ring, stream):
    measurements = {mode: [] for mode in args.modes}
    names = tuple(args.modes)
    for repeat in range(args.forward_repeats):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for mode in order if repeat % 2 == 0 else reversed(order):
            function = partial(_block, inputs, mesh, lengths, plan, ring, stream, mode, False)
            measurements[mode].append(_measure(function, args.iterations))
    return measurements


def _run_case(args, sample, mesh):
    lengths = tuple(sample["document_lengths"])
    size, rank = mesh.size(), mesh.get_local_rank()
    local = sum(lengths) // size
    ring = RING._ring_context(mesh)  # pylint: disable=protected-access
    plan = ring_task_plan(lengths, size, rank, "zigzag", True)
    stream = torch.npu.Stream()
    generator = torch.Generator().manual_seed(args.seed + rank)
    hidden = torch.randn(local, args.hidden, generator=generator).bfloat16().to("npu").requires_grad_()
    generator = torch.Generator().manual_seed(args.seed)
    weights = tuple((torch.randn(out_width, in_width, generator=generator) / in_width ** 0.5)
                    .bfloat16().to("npu").requires_grad_()
                    for out_width, in_width in ((2048, args.hidden), (512, args.hidden),
                                                (512, args.hidden), (args.hidden, 2048)))
    generator = torch.Generator().manual_seed(args.seed + 100 + rank)
    grad_output = torch.randn(local, args.hidden, generator=generator).bfloat16().to("npu")
    inputs = (hidden, *weights, grad_output)
    functions = {mode: lambda mode=mode: _block(inputs, mesh, lengths, plan, ring, stream, mode)
                 for mode in args.modes}
    torch.use_deterministic_algorithms(True)
    expected = _block(inputs, mesh, lengths, plan, ring, stream, "baseline")
    errors = {}
    labels = ("O", "dHidden", "dWq", "dWk", "dWv", "dWo")
    for mode, function in functions.items():
        if rank == 0:
            print(json.dumps({"check": sample["name"], "mode": mode}), flush=True)
        actual = function()
        errors[mode] = {name: _error(value, target) for name, value, target in zip(labels, actual, expected)}
        if not all(item["finite"] for item in errors[mode].values()):
            raise RuntimeError(f"Non-finite result for {sample['name']}/{mode}")
        if mode != "none":
            for name, value, target in zip(labels, actual, expected):
                # The three branches can accumulate dHidden in different BF16 orders.
                # Every individual weight gradient and the output must be unchanged.
                if name != "dHidden":
                    torch.testing.assert_close(value, target, rtol=0, atol=0,
                                               msg=f"{sample['name']}/{mode}/{name}: {errors[mode][name]}")
        del actual
    del expected
    torch.use_deterministic_algorithms(False)
    for function in functions.values():
        for _ in range(args.warmup):
            function()
    measurements = {mode: [] for mode in functions}
    names = tuple(functions)
    for repeat in range(args.repeats):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for mode in order if repeat % 2 == 0 else reversed(order):
            measurements[mode].append(_measure(functions[mode], args.iterations))
    forward = _forward_measurements(args, inputs, mesh, lengths, plan, ring, stream)
    workers = [None] * size
    dist.all_gather_object(workers, {"rank": rank, "errors": errors, "measurements": measurements,
                                     "forward_measurements": forward})
    summary = {}
    for mode in names:
        times = [max(worker["measurements"][mode][repeat]["ms"] for worker in workers)
                 for repeat in range(args.repeats)]
        summary[mode] = {
            "median_slowest_rank_ms": median(times), "slowest_rank_samples_ms": times,
            "max_peak_extra_bytes": max(item["peak_extra_bytes"] for worker in workers
                                        for item in worker["measurements"][mode]),
            "max_live_growth_bytes": max(item["live_growth_bytes"] for worker in workers
                                         for item in worker["measurements"][mode]),
            "errors": {name: {"bitwise_equal_all_ranks": all(worker["errors"][mode][name]["bitwise_equal"]
                                                             for worker in workers),
                              "max_abs": max(worker["errors"][mode][name]["max_abs"] for worker in workers),
                              "max_relative_l2": max(worker["errors"][mode][name]["relative_l2"]
                                                     for worker in workers)} for name in labels},
        }
    if args.forward_repeats:
        for mode in names:
            times = [max(worker["forward_measurements"][mode][repeat]["ms"] for worker in workers)
                     for repeat in range(args.forward_repeats)]
            summary[mode]["forward_only"] = {
                "median_slowest_rank_ms": median(times), "slowest_rank_samples_ms": times,
            }
    if args.profile_dir and sample["name"] in args.profile_cases:
        _profile(functions, args.profile_modes, args.profile_dir, rank)
    return {"name": sample["name"], "document_lengths": lengths, "summary": summary, "workers": workers}


def main() -> None:
    """Run paired complete-block measurements using fixed manifest document lengths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--forward-repeats", type=int, default=0)
    parser.add_argument("--profile-dir")
    parser.add_argument("--profile-cases", nargs="+", default=["shuffled_000"])
    parser.add_argument("--profile-modes", nargs="+", choices=MODES,
                        default=["baseline", "prepack_overlap", "qcomm_overlap"])
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=300))
    report = {"parameters": vars(args), "results": [], "complete": False,
              "task_queue_enable": os.environ.get("TASK_QUEUE_ENABLE", "default"),
              "scope": "Q/K/V projection + Ring + output projection + local backward; no QK norm, RoPE, "
                       "parameter gradient synchronization, FSDP, optimizer, or Trainer"}
    try:
        mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("cp",))
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if manifest["cp"] != mesh.size():
            raise ValueError("Manifest CP size must match the launched world size")
        for sample in manifest["samples"]:
            if args.cases and sample["name"] not in args.cases:
                continue
            result = _run_case(args, sample, mesh)
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
