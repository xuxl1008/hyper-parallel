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
"""Full KimiDeltaAttention validation and benchmark with the staged AscendC CP prototype."""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

MODES = ("p2p", "allgather", "ulysses", "p2p_whole2", "p2p_sf2", "p2p_dual2",
         "p2p_whole4", "p2p_sf4", "p2p_dual4", "p2p_summary2", "p2p_summary4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--local-tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--compare-modes", nargs="+", choices=MODES, default=[])
    parser.add_argument("--memory-variants", nargs="+", default=[])
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    root = args.install.resolve()
    sys.path.insert(0, str(root))
    vendor = root / "fla_npu/opp/vendors/fla_npu_transformer"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{vendor}:{vendor}/op_api/lib"
    import torch
    import torch.distributed as dist
    import torch_npu
    import fla_npu
    import hyper_parallel
    from torch.utils.checkpoint import checkpoint
    from hyper_parallel.components.modules import kimi_delta_attention as model
    from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
    from stateful_layer import build_executor
    from memory_options import metadata as memory_metadata

    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    torch.set_num_threads(1)
    if args.memory_variants:
        # Use deterministic Conv reductions for tight A/A checks. This setting
        # is restricted to numerical checks, never performance runs.
        torch.use_deterministic_algorithms(True)
    dist.init_process_group("hccl")
    world = dist.get_world_size()
    mesh = DeviceMesh("npu", list(range(world)), mesh_dim_names=("cp",))

    def sample(r):
        gen = torch.Generator().manual_seed(args.seed + 1000 + r)
        shape = (1, args.local_tokens, args.hidden_size)
        return tuple(torch.randn(shape, generator=gen).bfloat16() for _ in range(2))

    try:
        if args.heads % world or args.local_tokens % 64 or min(args.repeat, args.warmup) < 1:
            raise ValueError("Require H divisible by CP, chunk-aligned T, positive warmup/repeat")
        if args.validate_only and (args.local_tokens * world > 1024 or args.heads > 8 or args.hidden_size > 64):
            raise ValueError("CPU full-layer reference is restricted to small shapes")
        torch.manual_seed(args.seed)
        layer = model.KimiDeltaAttention(hidden_size=args.hidden_size, num_heads=args.heads).bfloat16()
        # Gate parameters remain FP32 even though the projections use BF16.
        layer.A_log.data = layer.A_log.data.float()
        layer.dt_bias.data = layer.dt_bias.data.float()
        reference = copy.deepcopy(layer).float() if args.validate_only else None
        layer = layer.npu().train()
        original_norm = layer.o_norm
        executor = build_executor(layer, mesh, args.mode)
        x_cpu, do_cpu = sample(rank)
        x, do = x_cpu.npu().requires_grad_(), do_cpu.npu()
        names, params = zip(*layer.named_parameters())
        leaves = (x, *params)

        def run():
            y = checkpoint(executor, x, use_reentrant=False) if args.checkpoint else executor(x)
            return y[0] if isinstance(y, tuple) else y

        def evaluate():
            y = run()
            grads = torch.autograd.grad(y, leaves, do, allow_unused=False)
            finite = torch.stack([torch.isfinite(t).all() for t in (y, *grads)]).all().int()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise AssertionError("Non-finite full-layer output/gradient")
            return y, grads

        def reduced_cpu(y, grads):
            values = [y.detach().float().cpu(), grads[0].detach().float().cpu()]
            for grad in grads[1:]:
                g = grad.detach().float()
                dist.all_reduce(g)
                values.append(g.cpu())
            return values

        def compare(actual, expected):
            errors = [float((a-b.float()).norm()/b.float().norm().clamp_min(1e-12))
                      for a, b in zip(actual, expected)]
            errors = torch.tensor(errors, device="npu")
            dist.all_reduce(errors, op=dist.ReduceOp.MAX)
            result = dict(zip(("output", "dhidden", *names), errors.cpu().tolist()))
            if max(result.values()) >= .03:
                raise AssertionError(result)
            return result

        metadata = {**memory_metadata(), **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cp": world, "global_tokens": args.local_tokens * world, "batch": 1,
                    "dimension": 128, "chunk": 64, "scale": 128 ** -.5,
                    "native_core_scale": .125, "output_scale_correction": (128 ** -.5) / .125,
                    "scope": "full KimiDeltaAttention layer, hidden-to-hidden, all input/parameter grads, no parameter allreduce in timing",
                    "state_dtype": "float32", "token_dtype": "bfloat16", "gate_parameter_dtype": "float32",
                    "recompute": args.checkpoint, "forward_grad_enabled": True,
                    "normalization": "fla_npu Triton L2Norm",
                    "summary_gradient_value_block": 128,
                    "shortconv": "kernel4; state modes halo; Ulysses packed exchange then head-local conv",
                    "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                    "triton_ascend": importlib.metadata.version("triton-ascend"),
                    "fla_npu": fla_npu.__file__, "hyper_parallel": hyper_parallel.__file__,
                    "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                    "ub_override": os.environ.get("ASCEND_UB_CAPACITY_BITS"),
                    "parameter_count": sum(p.numel() for p in params),
                    "parameter_names": names,
                    "prototype_sha256": hashlib.sha256(Path(__file__).with_name("stateful_kda.py").read_bytes()).hexdigest(),
                    "layer_prototype_sha256": hashlib.sha256(Path(__file__).with_name("stateful_layer.py").read_bytes()).hexdigest(),
                    "head_schedule_sha256": hashlib.sha256(Path(__file__).with_name("stateful_heads.py").read_bytes()).hexdigest(),
                    "summary_head_sha256": hashlib.sha256(Path(__file__).with_name("stateful_summary_heads.py").read_bytes()).hexdigest(),
                    "opapi_sha256": hashlib.sha256((vendor / "op_api/lib/libcust_opapi.so").read_bytes()).hexdigest()}
        if rank == 0:
            print(json.dumps({"event": "start", **metadata}), flush=True)
        y, grads = evaluate()
        if args.validate_only or args.compare_modes or args.memory_variants:
            actual = reduced_cpu(y, grads)
        del y, grads
        if args.validate_only:
            full = [sample(r) for r in range(world)]
            rx = torch.cat([item[0] for item in full], 1).float().requires_grad_()
            rd = torch.cat([item[1] for item in full], 1).float()

            def recurrent(*inputs, **kwargs):
                kwargs.pop("chunk_size", None)
                return model.torch_recurrent_kda(*inputs, **kwargs)

            with patch.object(model, "torch_chunk_kda", recurrent):
                ry = reference(rx)
            rg = torch.autograd.grad(ry, (rx, *reference.parameters()), rd)
            begin, end = rank * args.local_tokens, (rank + 1) * args.local_tokens
            refs = [ry[:, begin:end].detach(), rg[0][:, begin:end], *rg[1:]]
            metadata["validation_rel_l2"] = compare(actual, refs)
            metadata["reference"] = "full layer CPU FP32 with token-recurrent KDA, same quantized weights"
            del ry, rg, rx, rd, refs, reference
        if args.compare_modes:
            metadata["cross_mode_rel_l2"] = {}
            for mode in args.compare_modes:
                executor = build_executor(layer, mesh, mode)
                y, grads = evaluate()
                metadata["cross_mode_rel_l2"][mode] = compare(reduced_cpu(y, grads), actual)
                del y, grads
            executor = build_executor(layer, mesh, args.mode)
        if args.memory_variants:
            if not (args.validate_only or args.smoke_only):
                raise ValueError("Cross-variant checks must not be mixed with timing")
            original_variant = os.environ.get("KDA_MEMORY_VARIANT", "baseline")
            metadata['cross_memory_variant_rel_l2'] = {}
            for candidate in args.memory_variants:
                os.environ['KDA_MEMORY_VARIANT'] = candidate
                layer.o_norm = original_norm
                executor = build_executor(layer, mesh, args.mode)
                y, grads = evaluate()
                errors = compare(reduced_cpu(y, grads), actual)
                limit = .005 if 'norm' in candidate else 1e-6
                if max(errors.values()) > limit:
                    raise AssertionError((candidate, limit, errors))
                metadata['cross_memory_variant_rel_l2'][candidate] = errors
                del y, grads
            os.environ['KDA_MEMORY_VARIANT'] = original_variant
            layer.o_norm = original_norm
            executor = build_executor(layer, mesh, args.mode)
        if args.validate_only or args.compare_modes or args.memory_variants:
            del actual

        def measure(training):
            for _ in range(args.warmup):
                out = run()
                gradients = torch.autograd.grad(out, leaves, do) if training else ()
                torch.npu.synchronize()
                del out, gradients
            gc.collect()
            torch.npu.empty_cache()
            dist.barrier()
            torch.npu.synchronize()
            baseline = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            times = []
            for _ in range(args.repeat):
                dist.barrier()
                torch.npu.synchronize()
                start = time.perf_counter()
                out = run()
                gradients = torch.autograd.grad(out, leaves, do) if training else ()
                torch.npu.synchronize()
                times.append(1000 * (time.perf_counter() - start))
                del out, gradients
            sizes = torch.tensor([baseline, torch.npu.max_memory_allocated(), torch.npu.max_memory_reserved()],
                                 device="npu", dtype=torch.int64)
            times = torch.tensor(times, device="npu")
            dist.all_reduce(sizes, op=dist.ReduceOp.MAX)
            dist.all_reduce(times, op=dist.ReduceOp.MAX)
            sizes, times = sizes.cpu().tolist(), times.cpu().tolist()
            return {"median_ms": statistics.median(times), "samples_rank_max_ms": times,
                    "min_ms": min(times), "max_ms": max(times), "baseline_allocated_mib": sizes[0] / 2**20,
                    "peak_allocated_mib": sizes[1] / 2**20, "peak_reserved_mib": sizes[2] / 2**20}

        if args.validate_only or args.smoke_only:
            metadata["status"] = "validation_passed" if args.validate_only else "finite_smoke_passed"
        else:
            metadata["forward"] = measure(False)
            metadata["forward_backward"] = measure(True)
        if rank == 0:
            args.output.write_text(json.dumps(metadata, indent=2) + "\n")
            print(json.dumps({"event": "result", **metadata}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
