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
"""Explicit DSA draft experiments with a supplied packed token sample.

Run as ``python -m torch.distributed.run ... --module examples.torch.dsa.benchmark``
from the repository root. Output is per-rank JSON; no activation dumps are saved.
"""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Attention, DeepseekV32RotaryEmbedding

from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context, bind_aux_loss_scale
from hyper_parallel.components.modules.dsa_attention import DeepseekV32DSAAttention
from hyper_parallel.models.deepseek_v32.adapter.distributed import context_parallel as cp
from examples.torch.dsa.runtime_experiments import dsa_experiment


class _Mesh:
    @staticmethod
    def size() -> int:
        return dist.get_world_size()

    @staticmethod
    def get_local_rank() -> int:
        return dist.get_rank()

    @staticmethod
    def get_group():
        return dist.group.WORLD


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="NPZ: 1D tokens and cumulative boundaries")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-tokens", type=int, default=16384)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--qr", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--mode", choices=("none", "zigzag", "greedy", "hybrid"), default="greedy")
    parser.add_argument("--experiment", choices=("baseline", "deferred-main", "q-stream", "q-stream-deferred",
                                               "partial-rope"), default="baseline")
    parser.add_argument("--interleaved", action="store_true", help="Opt into nondefault Indexer RoPE convention")
    parser.add_argument("--check", action="store_true", help="Compare full outputs/gradients against baseline, untimed")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--profile-rank", type=int, default=-1, help="Profile one extra step on this rank; -1 disables")
    args = parser.parse_args()
    if min(args.global_tokens, args.heads, args.qr, args.hidden, args.iterations) <= 0 or args.warmup < 0:
        parser.error("Dimensions and iterations must be positive; warmup must be nonnegative")
    if args.experiment == "partial-rope" and args.mode == "none":
        parser.error("partial-rope requires an enabled selection worker; choose zigzag, greedy or hybrid")
    if args.check and os.environ.get("DSA_SFAG_SCATTER_OVERRIDE", "auto") != "auto":
        parser.error("Deterministic --check does not exercise the experimental SFA scatter gate; use separate runs")
    return args


def _model_inputs(args: argparse.Namespace) -> tuple:
    config = DeepseekV32Config(
        hidden_size=args.hidden, intermediate_size=512, moe_intermediate_size=128,
        num_attention_heads=args.heads, num_key_value_heads=1, q_lora_rank=args.qr, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        index_n_heads=64, index_head_dim=128, index_topk=2048, num_hidden_layers=1, attention_bias=False,
    )
    config.dsa_loss_coeff, config.dsa_dense_warm_up = 0.001, False
    source = DeepseekV32Attention(config, 0).bfloat16()
    module = DeepseekV32DSAAttention(module=source)
    with torch.no_grad():
        module.linear_qkv.weight.copy_(torch.cat((source.q_a_proj.weight, source.kv_a_proj_with_mqa.weight)))
    del source
    module = module.to("npu").train()
    module.index_rotary_interleaved = args.interleaved
    module._hp_deepseek_v32_dsa_cp_mesh = _Mesh()  # pylint: disable=protected-access
    module._hp_deepseek_v32_dsa_cp_config = {  # pylint: disable=protected-access
        "indexer_load_balance": args.mode,
        "indexer_balance_block_size": 2048 if args.mode in ("greedy", "hybrid") else None,
        "indexer_zigzag_min_length": 32768 if args.mode == "hybrid" else None,
        "main_kv_comm": "all_to_all_v", "index_k_comm": "all_to_all_v",
    }
    with np.load(args.data, allow_pickle=False) as sample:
        if sample["tokens"].ndim != 1 or sample["tokens"].size < args.global_tokens:
            raise ValueError("Sample must contain enough one-dimensional tokens")
        tokens = torch.from_numpy(sample["tokens"][:args.global_tokens].astype(np.int64))
        ends = tuple(int(x) for x in sample["boundaries"] if 0 < x < args.global_tokens) + (args.global_tokens,)
    if any(right <= left for left, right in zip((0, *ends[:-1]), ends)) or int(tokens.min()) < 0:
        raise ValueError("Boundaries must be strictly increasing and tokens nonnegative")
    local = args.global_tokens // dist.get_world_size()
    start = dist.get_rank() * local
    table = torch.randn(int(tokens.max()) + 1, args.hidden) * 0.2
    hidden = table[tokens[start:start+local]].unsqueeze(0).to(device="npu", dtype=torch.bfloat16).requires_grad_()
    positions = torch.cat([torch.arange(end-begin) for begin, end in zip((0, *ends[:-1]), ends)])
    positions = positions[start:start+local].unsqueeze(0).to("npu")
    embeddings = DeepseekV32RotaryEmbedding(config).to("npu")(hidden, positions)
    return module, hidden, embeddings, ends


def _step(module: torch.nn.Module, hidden: torch.Tensor, embeddings: tuple, ends: tuple) -> tuple:
    module.zero_grad(set_to_none=True)
    hidden.grad = None
    torch.npu.synchronize()
    dist.barrier()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    begin, middle, end = (torch.npu.Event(enable_timing=True) for _ in range(3))
    begin.record()
    with aux_loss_scale_context():
        with torch.profiler.record_function("dsa/forward"):
            output, _ = cp._deepseek_v32_dsa_cp_forward(  # pylint: disable=protected-access
                module, hidden, embeddings, actual_seq_len=ends,
            )
            loss = output.float().square().mean()
            bind_aux_loss_scale(loss)
        middle.record()
        with torch.profiler.record_function("dsa/backward"):
            loss.backward()
    end.record()
    torch.npu.synchronize()
    return output, {
        "forward_ms": begin.elapsed_time(middle), "backward_ms": middle.elapsed_time(end),
        "total_ms": begin.elapsed_time(end), "peak_allocated_mib": torch.npu.max_memory_allocated() / 2**20,
    }


def _snapshot(module: torch.nn.Module, hidden: torch.Tensor, output: torch.Tensor) -> dict:
    tensors = {"output": output.detach().cpu(), "input_gradient": hidden.grad.detach().cpu().clone()}
    tensors.update({name: parameter.grad.detach().cpu().clone()
                    for name, parameter in module.named_parameters() if parameter.grad is not None})
    return tensors


def _checked_step(module: torch.nn.Module, hidden: torch.Tensor, embeddings: tuple, ends: tuple) -> dict:
    captured = {}
    original = cp._run_cp_dsa_kernels  # pylint: disable=protected-access

    def capture(*args, **kwargs):
        captured["topk"] = kwargs["selected"].detach().cpu().clone()
        return original(*args, **kwargs)

    with patch.object(cp, "_run_cp_dsa_kernels", capture):
        output, _ = _step(module, hidden, embeddings, ends)
    return {**_snapshot(module, hidden, output), **captured}


def main() -> None:
    """Run an explicit experiment and report evidence without saving large tensors."""
    args = _arguments()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(args.check)
    os.environ["HCCL_DETERMINISTIC"] = "true" if args.check else "false"
    torch.manual_seed(431)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    if dist.get_world_size() < 2 or args.global_tokens % dist.get_world_size():
        raise ValueError("Use CP>=2 and a global token count divisible by CP size")
    if not -1 <= args.profile_rank < dist.get_world_size():
        raise ValueError("profile-rank must be -1 or a valid CP rank")
    args.output.mkdir(parents=True, exist_ok=True)
    module, hidden, embeddings, ends = _model_inputs(args)
    report = {"experiment": args.experiment, "rank": dist.get_rank(), "cp": dist.get_world_size(),
              "global_tokens": args.global_tokens, "documents": len(ends), "heads": args.heads, "qr": args.qr,
              "mode": args.mode, "interleaved": args.interleaved, "deterministic": args.check,
              "scatter_gate": os.environ.get("DSA_SFAG_SCATTER_OVERRIDE", "auto"),
              "scope": "one attention forward/backward including Indexer and KL, no optimizer/FSDP/Trainer"}
    reference = None
    if args.check:
        reference = _checked_step(module, hidden, embeddings, ends)
    with dsa_experiment(
        module, defer_main=args.experiment in ("deferred-main", "q-stream-deferred"),
        separate_query_stream=args.experiment in ("q-stream", "q-stream-deferred"),
        worker_partial_rope=args.experiment == "partial-rope",
    ):
        if args.check:
            actual = _checked_step(module, hidden, embeddings, ends)
            if actual.keys() != reference.keys():
                raise AssertionError("Gradient names differ from the baseline")
            for name, tensor in actual.items():
                torch.testing.assert_close(tensor, reference[name], rtol=0, atol=0, msg=name)
            report["exact_output_topk_and_gradient_comparisons"] = len(actual)
            del actual, reference
        else:
            samples = []
            for iteration in range(args.warmup + args.iterations):
                output, sample = _step(module, hidden, embeddings, ends)
                del output
                if iteration >= args.warmup:
                    samples.append(sample)
            report["samples"] = samples
            report["median"] = {key: statistics.median(row[key] for row in samples) for key in samples[0]}
        if args.profile_rank >= 0:
            context = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True, profile_memory=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output / "profile")),
            ) if dist.get_rank() == args.profile_rank else nullcontext()
            with context:
                output, _ = _step(module, hidden, embeddings, ends)
                del output
    report["completed"] = True
    (args.output / f"rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
