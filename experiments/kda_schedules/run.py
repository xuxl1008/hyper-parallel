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
"""Explicit research runner, outside automatic UT/ST collection."""

import argparse
from datetime import timedelta
from functools import partial
import importlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

import hyper_parallel

from .protocols import TriangularAllToAllV, ValuePackets, make_protocol
from .schedules import MODES, run_core


def _cpu_protocols():
    rank, size = dist.get_rank(), dist.get_world_size()
    generator = torch.Generator().manual_seed(20261030)
    matrices = torch.eye(16).reshape(1, 1, 1, 16, 16).repeat(size, 1, 2, 1, 1) * .9
    matrices += torch.randn(matrices.shape, generator=generator) * .01
    offsets = torch.randn((size, 1, 2, 16, 16), generator=generator)
    for mode, protocol in [(f"v{n}", ValuePackets(n)) for n in (1, 2, 4, 8)] + [
        ("alltoallv", TriangularAllToAllV())
    ]:
        for reverse in (False, True):
            expected = torch.zeros_like(offsets[0])
            indices = range(size - 1, rank, -1) if reverse else range(rank)
            for index in indices:
                matrix = matrices[index].transpose(-1, -2) if reverse else matrices[index]
                expected = matrix @ expected + offsets[index]
            actual = protocol.exchange(offsets[rank], matrices[rank], reverse)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        if rank == 0:
            print(json.dumps({"mode": mode, "world_size": size, "cpu_protocol": "passed"}), flush=True)


def _inputs(tokens, heads, seed):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, tokens, heads, 128)
    tokens_list = [torch.randn(shape, generator=generator).to(torch.bfloat16) for _ in range(4)]
    tokens_list[3] *= .2
    beta = torch.randn(shape[:-1], generator=generator).to(torch.bfloat16)
    alog = torch.linspace(-1., .2, heads)
    # Slow enough decay to exercise nonzero cross-rank state and dHT.
    bias = torch.full((heads, 128), -6.)
    return (*tokens_list, beta, alog, bias)


def _device_inputs(inputs, device, local_tokens=None):
    rank = dist.get_rank()
    result = []
    for index, tensor in enumerate(inputs):
        if local_tokens is not None and index < 5:
            tensor = tensor[:, rank * local_tokens:(rank + 1) * local_tokens]
        result.append(tensor.to(device).contiguous().detach().requires_grad_())
    return tuple(result)


def _loss(output, target):
    return (output.float() * target.float()).sum()


def _invoke_core(*values, mode, blocks, protocol):
    return run_core(values, mode, blocks, protocol)


def _validate(args, device):
    # Optional Triton is needed only for actual NPU numerical validation.
    from hyper_parallel.components.functional.kimi_delta_attention import fused_chunk_kda
    rank, size = dist.get_rank(), dist.get_world_size()
    references = []
    for index in range(args.inflight):
        sample = _inputs(args.local_tokens * size, args.heads, args.seed + index)
        full = _device_inputs(sample, device)
        generator = torch.Generator().manual_seed(args.seed + index + 100)
        target = torch.randn(full[0].shape, generator=generator).to(device, dtype=torch.bfloat16)
        output = fused_chunk_kda(*full[:5], a_log=full[5], dt_bias=full[6])
        gradients = torch.autograd.grad(_loss(output, target), full)
        local = slice(rank * args.local_tokens, (rank + 1) * args.local_tokens)
        expected = (output[:, local].detach(), *(g[:, local].detach() if i < 5 else g.detach()
                                                for i, g in enumerate(gradients)))
        references.append((sample, target[:, local].contiguous(), expected))
        del full, output, gradients
    for mode in args.modes:
        protocol = make_protocol(mode, args.blocks, args.group_width) if mode in (
            "p2p", "v-packet", "allgather", "grouped", "alltoallv") else None
        # Independent invocations plus reverse-order backward expose stale executor workspaces.
        pending = []
        for sample, target, expected in references:
            inputs = _device_inputs(sample, device, args.local_tokens)

            invoke = partial(_invoke_core, mode=mode, blocks=args.blocks, protocol=protocol)
            output = checkpoint(invoke, *inputs, use_reentrant=False) if args.checkpoint else invoke(*inputs)
            pending.append((inputs, output, target, expected))
        errors = torch.zeros(8, device=device)
        for inputs, output, target, expected in reversed(pending):
            gradients = torch.autograd.grad(_loss(output, target), inputs)
            for grad in gradients[5:]:
                dist.all_reduce(grad)
            for index, (actual, reference) in enumerate(zip((output, *gradients), expected, strict=True)):
                if not bool(torch.isfinite(actual).all()):
                    raise AssertionError(f"Nonfinite {mode} tensor {index}, rank {rank}")
                error = (actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-7)
                errors[index] = torch.maximum(errors[index], error)
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        values = errors.cpu().tolist()
        result = dict(mode=mode, cp=size, local_tokens=args.local_tokens, heads=args.heads, blocks=args.blocks,
                      checkpoint=args.checkpoint, inflight=args.inflight,
                      relative_l2=dict(zip(("output", "dq", "dk", "dv", "dgate", "dbeta", "dA_log", "ddt_bias"),
                                           values, strict=True)))
        if rank == 0:
            print(json.dumps(result), flush=True)
        if max(values) > args.tolerance:
            raise AssertionError(f"{mode} exceeds declared relative-L2 tolerance {args.tolerance}: {values}")
        del pending, inputs, output, gradients
        dist.barrier()


def _benchmark(args, device):
    rank, size = dist.get_rank(), dist.get_world_size()
    # Operator-level experiment only. The operator does not perform parameter gradient DP reduction.
    inputs = _device_inputs(_inputs(args.local_tokens, args.heads, args.seed + rank), device)
    target = torch.ones_like(inputs[0])
    for mode in args.modes:
        protocol = make_protocol(mode, args.blocks, args.group_width) if mode in (
            "p2p", "v-packet", "allgather", "grouped", "alltoallv") else None
        times, forward_peaks, step_peaks = [], [], []
        for iteration in range(args.warmup + args.repeat):
            dist.barrier()
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            output = run_core(inputs, mode, args.blocks, protocol)
            torch.npu.synchronize()
            forward_end = time.perf_counter()
            peak_forward = torch.npu.max_memory_allocated()
            gradients = torch.autograd.grad(_loss(output, target), inputs)
            torch.npu.synchronize()
            end = time.perf_counter()
            if iteration >= args.warmup:
                times.append(((forward_end - start) * 1e3, (end - start) * 1e3))
                forward_peaks.append(peak_forward)
                step_peaks.append(torch.npu.max_memory_allocated())
            del output, gradients
        summary = torch.tensor((*torch.tensor(times).mean(0).tolist(), max(forward_peaks) / 2**20,
                                max(step_peaks) / 2**20), device=device)
        dist.all_reduce(summary, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(json.dumps(dict(mode=mode, cp=size, local_tokens=args.local_tokens, heads=args.heads,
                                  blocks=args.blocks, warmup=args.warmup, repeat=args.repeat,
                                  metric="max-rank mean; live inputs included; not full attention layer",
                                  fwd_ms=summary[0].item(), step_ms=summary[1].item(),
                                  fwd_peak_mib=summary[2].item(), step_peak_mib=summary[3].item())), flush=True)


def main() -> None:
    """Run CPU protocol checks, full-gradient NPU comparisons, or explicit timing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-protocols", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--blocks", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--group-width", type=int, default=2)
    parser.add_argument("--local-tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--inflight", type=int, choices=(1, 2), default=1)
    parser.add_argument("--tolerance", type=float, default=.02)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--exclusive-devices-confirmed", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    if args.benchmark and (not args.exclusive_devices_confirmed or args.checkpoint or args.inflight != 1):
        parser.error("Timing requires audited idle devices, no checkpoint and a single in-flight call")
    if args.repeat < 1 or args.warmup < 0 or args.tolerance <= 0:
        parser.error("Invalid repeat/warmup/tolerance")
    root = Path(__file__).resolve().parents[2]
    if Path(hyper_parallel.__file__).resolve().parent.parent != root:
        raise RuntimeError("Install this checkout with pip install -e . before running")
    device = "cpu"
    if not args.cpu_protocols:
        # Keep the CPU communication oracle usable without an NPU runtime.
        importlib.import_module("torch_npu")
        device = f"npu:{int(os.environ['LOCAL_RANK'])}"
        torch.npu.set_device(device)
    dist.init_process_group("gloo" if args.cpu_protocols else "hccl", timeout=timedelta(minutes=15))
    try:
        if dist.get_rank() == 0:
            print(json.dumps({"torch": torch.__version__, "hyper_parallel": hyper_parallel.__file__,
                              "scope": "KDA core only; no projection/conv/output norm/optimizer/FSDP"}), flush=True)
        if args.cpu_protocols:
            _cpu_protocols()
        elif args.benchmark:
            _benchmark(args, device)
        else:
            _validate(args, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
