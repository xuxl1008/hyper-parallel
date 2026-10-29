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
"""Shared boundary-protocol correctness, timing and aligned Level0 helpers."""
from __future__ import annotations

import argparse
import gc
import importlib
import statistics
import time

import torch
import torch.distributed as dist


def make_inputs(args: argparse.Namespace, size: int, reset: bool) -> tuple[list, list, list]:
    """Use exact captured S/M/G or reproducible noncommuting synthetic summaries."""
    if args.source_dir:
        if args.source_ranks != size and not args.repeat_captures:
            raise ValueError("Capture rank count differs from WORLD; repeating requires --repeat-captures")
        pieces = [torch.load(args.source_dir / f"rank{index}_boundaries.pt", map_location="cpu",
                             weights_only=True) for index in range(args.source_ranks)]
        values = tuple([pieces[index % args.source_ranks][name].float().contiguous().clone()
                        for index in range(size)] for name in ("S", "M", "G"))
    else:
        generator = torch.Generator().manual_seed(args.seed)
        shape = (size, args.batch, args.heads, 128, 128)
        states = torch.randn(shape, generator=generator) * 0.03
        matrices = torch.eye(128) * 0.91 + torch.randn(shape, generator=generator) * 0.002
        gradients = torch.randn(shape, generator=generator) * 0.03
        values = tuple([tensor[index].contiguous().clone() for index in range(size)]
                       for tensor in (states, matrices, gradients))
    if reset:
        # A structural zero transfer models reset semantics only; no message pruning is claimed.
        for index in range(0, size, 2):
            values[1][index].zero_()
    return values


def reference(values: tuple, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ordered exclusive prefix and suffix directly in CPU FP64."""
    states, matrices, gradients = values
    incoming = torch.zeros_like(states[0], dtype=torch.float64)
    outgoing = torch.zeros_like(incoming)
    for index in range(rank):
        incoming = matrices[index].double() @ incoming + states[index].double()
    for index in range(len(states) - 1, rank, -1):
        outgoing = gradients[index].double() + matrices[index].double().transpose(-1, -2) @ outgoing
    return incoming, outgoing


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Report both absolute and relative errors, including reference magnitude."""
    actual, expected = actual.detach().cpu().double(), expected.double()
    difference = actual - expected
    error_norm, reference_norm = difference.norm().item(), expected.norm().item()
    head_error = difference.flatten(2).norm(dim=-1)
    head_norm = expected.flatten(2).norm(dim=-1)
    head_relative = head_error / head_norm.clamp_min(1e-30)
    worst = head_relative.flatten().argmax().item()
    finite = bool(torch.isfinite(actual).all())
    return {"relative_l2": error_norm / max(reference_norm, 1e-30), "max_abs": difference.abs().max().item(),
            "error_norm": error_norm, "reference_norm": reference_norm,
            "worst_head_relative_l2": head_relative.flatten()[worst].item(),
            "worst_head_reference_norm": head_norm.flatten()[worst].item(), "finite": finite,
            "pass": finite and error_norm <= 2e-6 * reference_norm + 1e-11}


def cache_info(cache: tuple) -> dict:
    """Count persisted tensor storage, including aliases and oversized allocations."""
    storages = {(tensor.device, tensor.untyped_storage().data_ptr()): tensor.untyped_storage().nbytes()
                for tensor in cache}
    return {"count": len(cache), "logical_bytes": sum(t.numel() * t.element_size() for t in cache),
            "unique_storage_bytes": sum(storages.values()),
            "compact": all(t.untyped_storage().nbytes() == t.numel() * t.element_size() for t in cache)}


def synchronize(device: str) -> None:
    """Complete device work before comparing outputs or taking host timestamps."""
    if device == "npu":
        torch.npu.synchronize()


def run_correctness(protocol: object, values: tuple, rank: int, device: str) -> tuple[dict, tuple]:
    """Check fixed inputs, two live calls, reverse backward and immutable caches."""
    local = [pieces[rank].to(device).contiguous() for pieces in values]
    snapshots = [tensor.clone() for tensor in local]
    outputs, cache = protocol.forward(local[0], local[1])
    saved = [tensor.clone() for tensor in cache]
    modified = tuple([tensor * (1.02 if which != 1 else 0.99) for tensor in pieces]
                     for which, pieces in enumerate(values))
    next_local = [pieces[rank].to(device).contiguous() for pieces in modified]
    next_snapshots = [tensor.clone() for tensor in next_local]
    next_output, next_cache = protocol.forward(next_local[0], next_local[1])
    next_saved = [tensor.clone() for tensor in next_cache]
    next_gradient = protocol.backward(next_local[2], next_cache)
    gradient = protocol.backward(local[2], cache)
    expected = reference(values, rank)
    next_expected = reference(modified, rank)
    cache_immutable = all(torch.equal(before, after) for before, after in zip(saved, cache))
    inputs_immutable = all(torch.equal(before, after) for before, after in zip(snapshots, local))
    second_inputs_immutable = all(torch.equal(before, after) for before, after in zip(next_snapshots, next_local))
    second_cache_immutable = all(torch.equal(before, after) for before, after in zip(next_saved, next_cache))
    checked = {"H": metrics(outputs, expected[0]), "D": metrics(gradient, expected[1]),
               "second_H": metrics(next_output, next_expected[0]),
               "second_D": metrics(next_gradient, next_expected[1])}
    info = cache_info(cache)
    row = {"interfaces": checked, "cache": info, "cache_immutable": cache_immutable,
           "inputs_immutable": inputs_immutable, "second_inputs_immutable": second_inputs_immutable,
           "second_cache_immutable": second_cache_immutable,
           "pass": all(item["pass"] for item in checked.values()) and cache_immutable and inputs_immutable
                   and second_inputs_immutable and second_cache_immutable and info["compact"]}
    compared = (outputs, gradient, next_output, next_gradient, *cache, *next_cache)
    # Independent snapshots also check both live invocations and every cache.
    return row, tuple(tensor.detach().cpu().clone() for tensor in compared)


def run_timing(protocol: object, local: tuple, args: argparse.Namespace) -> dict:
    """Measure identical isolated per-rank forward/backward, excluding input copies."""
    for _ in range(args.warmup):
        output, cache = protocol.forward(local[0], local[1])
        gradient = protocol.backward(local[2], cache)
    if args.warmup:
        del output, cache, gradient
    synchronize(args.device)
    gc.collect()
    base_bytes = None
    if args.device == "npu":
        base_bytes = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
    samples = []
    retained_bytes = None
    for _ in range(args.iterations):
        dist.barrier()
        synchronize(args.device)
        start = time.perf_counter()
        output, cache = protocol.forward(local[0], local[1])
        synchronize(args.device)
        forward_end = time.perf_counter()
        if args.device == "npu":
            retained_bytes = torch.npu.memory_allocated() - base_bytes
        gradient = protocol.backward(local[2], cache)
        synchronize(args.device)
        end = time.perf_counter()
        samples.append({"forward_ms": (forward_end - start) * 1000,
                        "backward_ms": (end - forward_end) * 1000, "total_ms": (end - start) * 1000})
        del output, cache, gradient
    peak_delta = torch.npu.max_memory_allocated() - base_bytes if args.device == "npu" else None
    return {"samples": samples,
            "median": {key: statistics.median(item[key] for item in samples) for key in samples[0]},
            "npu_peak_allocated_delta_bytes": peak_delta,
            "npu_live_after_forward_delta_bytes": retained_bytes,
            "timing_scope": "Protocol only; synchronized forward/backward; no summary production or token core."}


def run_profiles(protocols: dict, args: argparse.Namespace, rank: int, size: int) -> dict:
    """Capture one measured Level0 forward/backward per protocol after setup.

    The first selected case is measured once even when correctness covers both
    cases. Setup F/B and alignment are separately marked inside the profiler.
    """
    if args.profile_dir is None:
        return {}
    torch_npu = importlib.import_module("torch_npu")
    case = "reset" if args.cases == "reset" else "long"
    values = make_inputs(args, size, reset=case == "reset")
    local = tuple(pieces[rank].to(args.device).contiguous() for pieces in values)
    records = {}
    for name, protocol in protocols.items():
        output_path = args.profile_dir.resolve() / case / name / f"rank{rank}"
        output_path.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        synchronize(args.device)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False, profile_memory=False, with_stack=False,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output_path)),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0),
        ):
            # Profiler start-up varies across ranks. Warm up while every profiler
            # is active, then align devices before the single measured call.
            with torch.profiler.record_function(f"profile_setup/{name}"):
                warm_output, warm_cache = protocol.forward(local[0], local[1])
                warm_gradient = protocol.backward(local[2], warm_cache)
                synchronize(args.device)
                del warm_output, warm_cache, warm_gradient
                dist.barrier()
                synchronize(args.device)
            with torch.profiler.record_function(f"protocol/{name}/forward"):
                output, cache = protocol.forward(local[0], local[1])
            with torch.profiler.record_function(f"protocol/{name}/backward"):
                gradient = protocol.backward(local[2], cache)
            synchronize(args.device)
        records[name] = {"directory": f"{case}/{name}/rank{rank}", "case": case, "forward_backward_calls": 1,
                         "level": "Level0", "record_shapes": False, "with_stack": False,
                         "timing_separate": True, "profile_setup_calls": 1,
                         "alignment": "one setup F/B then barrier+synchronize inside profiler before measured markers"}
        if hasattr(protocol, "edges"):
            records[name]["tree_edges"] = [{"peer": edge.peer, "owner": edge.owner} for edge in protocol.edges]
        del output, cache, gradient
        dist.barrier()
    return records
