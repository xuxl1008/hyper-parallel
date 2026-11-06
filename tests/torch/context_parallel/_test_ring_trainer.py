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
"""Real TextTrainer training with Ring CP, FSDP, checkpointing and accumulation.

The existing AllGather CP implementation is the control, with identical model,
data, FSDP topology and optimizer. Compare every optimizer step's loss, clipped
FP32 gradients and FP32 parameter updates, not just a successful forward call.
AllGather validates the first step at identical weights; synchronous Ring must
match the complete overlapped trajectory exactly. Later AllGather gradients are
reported because independent BF16/AdamW trajectories no longer share weights.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from hyper_parallel import DTensor, HSDPModule
from hyper_parallel.trainer.callbacks.base import Callback
from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.text_trainer import TextTrainer

_RECIPE = Path(__file__).with_name("ring_trainer.yaml")
_ADAPTER = "hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel"


def _local_cpu(tensor):
    value = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return value.detach().float().cpu().clone()


class _TrainingTrace(Callback):
    """Observe the public training lifecycle and optimizer pre-step hooks."""

    def __init__(self, trainer: Any, trace_path: Path) -> None:
        """Capture initial master shards and initialize per-step observations."""
        super().__init__(trainer)
        self.trace_path = trace_path
        self.steps = []
        self.gradients = []
        self.micro_steps = 0
        self.attention_calls = 0
        self.initial = self._parameters()
        self.replay_parameter = torch.nn.Parameter(self.initial.clone())
        self.replay_optimizer = torch.optim.AdamW([self.replay_parameter], lr=1e-4, weight_decay=0, foreach=False)

    def _parameters(self):
        return torch.cat([_local_cpu(param).flatten() for group in self.trainer.optimizer.param_groups
                          for param in group["params"]])

    def before_attention(self, module: Any, args: Any) -> None:
        """Count forward entries to prove the configured recomputation executes."""
        del module, args
        self.attention_calls += 1

    def on_micro_step_begin(self, state: Any, micro_batch: dict, **kwargs: Any) -> None:
        """Check global RoPE positions and mask-free complete local token intervals."""
        del state, kwargs
        length = 256 // self.mesh.cp_size
        expected = torch.arange(self.mesh.cp_rank * length, (self.mesh.cp_rank + 1) * length)
        torch.testing.assert_close(micro_batch["position_ids"].cpu(), expected.unsqueeze(0))
        assert micro_batch["attention_mask"] is None
        assert micro_batch["input_ids"].shape == (1, length)
        self.micro_steps += 1

    def before_optimizer(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        """Capture the actual FP32 gradient consumed by AdamW after FSDP/clipping."""
        del args, kwargs
        parameters = [param for group in optimizer.param_groups for param in group["params"]]
        assert all(param.grad is not None for param in parameters)
        self.gradients.extend(_local_cpu(param.grad).flatten() for param in parameters)

    def on_step_end(self, state: Any, loss: float, loss_dict: dict, grad_norm: float, **kwargs: Any) -> None:
        """Record metrics and local master-parameter updates after each full step."""
        del loss_dict, kwargs
        gradients = torch.cat(self.gradients)
        parameters = self._parameters()
        assert torch.isfinite(gradients).all() and torch.isfinite(parameters).all()
        assert gradients.norm() > 0 and (parameters - self.initial).norm() > 0
        self.replay_parameter.grad = gradients
        self.replay_optimizer.step()
        torch.testing.assert_close(parameters, _local_cpu(self.replay_parameter), atol=2e-8, rtol=1e-6)
        for main_param, model_param in self.trainer.optimizer.model_param_by_optimizer_param.items():
            torch.testing.assert_close(_local_cpu(model_param),
                                       _local_cpu(main_param).to(model_param.dtype).float(), atol=0, rtol=0)
        self.steps.append((loss, grad_norm, gradients, parameters - self.initial))
        self.gradients.clear()
        assert state.global_step == len(self.steps)

    def on_train_end(self, state: Any) -> None:
        """Save before Trainer's final barrier, so every rank can read all shards."""
        del state
        torch.save((self.initial, self.steps), self.trace_path)


def _run_training(cp_size, degree, shard_size):
    raw = yaml.safe_load(_RECIPE.read_text(encoding="utf-8"))
    raw["accelerator"]["cp_size"] = cp_size
    raw["training"]["global_batch_size"] = 2 * (8 // cp_size)
    raw["fsdp_config"]["dp_shard_size"] = shard_size
    raw["fsdp_config"]["edp_shard_size"] = shard_size
    implementation = raw["plan_overrides"][1]["replace"]["implementation"]
    if degree == 0:
        raw["plan_overrides"][1]["replace"]["implementation"] = {
            "_target_": _ADAPTER + ".qwen3_moe_flash_attention_cp_mask_wrapper",
        }
    else:
        implementation["ulysses_degree"] = degree
        implementation["overlap"] = os.environ["HYPER_CP_TRAINER_MODE"] == "ring"
    trainer = TextTrainer(resolve_config(raw))
    assert trainer.mesh.cp_size == cp_size and trainer.mesh.dp_size == 8 // cp_size
    assert trainer.mesh.dp_shard_size == shard_size
    assert trainer.base.num_micro_batches == 2
    assert isinstance(trainer.base.model, HSDPModule)
    assert sum(isinstance(module, HSDPModule) for module in trainer.base.model.modules()) > 1
    assert any(isinstance(param, DTensor) and param.to_local().numel() < param.numel()
               for param in trainer.base.model.parameters())
    name = f"rank_{os.environ['RANK']}.pt"
    if degree:
        name = os.environ["HYPER_CP_TRAINER_MODE"] + "_" + name
    trace = _TrainingTrace(trainer.base, Path(os.environ["HYPER_CP_TRAINER_REFERENCE_DIR"]) / name)
    # There is no public callback registration method; extend the existing callback list.
    trainer.base._callbacks.append(trace)  # pylint: disable=protected-access
    handles = [optimizer.register_step_pre_hook(trace.before_optimizer)
               for optimizer in trainer.base.optimizer.chained_optimizers]
    attention = next(module for module in trainer.base.model.modules()
                     if callable(getattr(module, "attention_interface", None)))
    handles.append(attention.register_forward_pre_hook(trace.before_attention))
    try:
        trainer.train()
        assert trainer.base.state.global_step == 3 and len(trace.steps) == 3
        assert trace.micro_steps == 6 and trace.attention_calls >= 12
        result = trace.initial, trace.steps
    finally:
        for handle in handles:
            handle.remove()
        trainer.base.destroy_distributed()
    del trace, trainer
    gc.collect()
    torch.npu.empty_cache()
    return result


def _global_gradient_errors(directory):
    difference = torch.zeros(3, dtype=torch.float64)
    norm = torch.zeros_like(difference)
    for rank in range(8):
        _, actual = torch.load(directory / f"ring_rank_{rank}.pt", map_location="cpu", weights_only=True)
        _, expected = torch.load(directory / f"rank_{rank}.pt", map_location="cpu", weights_only=True)
        for step, (values, reference) in enumerate(zip(actual, expected)):
            difference[step] += (values[2].double() - reference[2].double()).square().sum()
            norm[step] += reference[2].double().square().sum()
    return (difference / norm.clamp_min(1e-24)).sqrt()


@pytest.mark.parametrize("cp_size,degree,shard_size", [(8, 1, 8), (8, 2, 8), (8, 4, 8), (4, 2, 4)],
                         ids=["cp8_ring_fsdp8", "cp8_u2r4_fsdp8", "cp8_u4r2_fsdp8", "dp2_cp4_u2r2_hsdp2x4"])
def test_trainer_fsdp(cp_size, degree, shard_size) -> None:
    """Validate AllGather initial gradients and exact sync/overlap AdamW trajectories."""
    mode = os.environ["HYPER_CP_TRAINER_MODE"]
    assert mode in ("control", "sync", "ring")
    reference_path = Path(os.environ["HYPER_CP_TRAINER_REFERENCE_DIR"]) / f"rank_{os.environ['RANK']}.pt"
    if mode == "control":
        _run_training(cp_size, 0, shard_size)
        return
    actual_initial, actual = _run_training(cp_size, degree, shard_size)
    if mode == "sync":
        return
    expected_initial, expected = torch.load(reference_path, map_location="cpu", weights_only=True)
    sync_initial, sync_steps = torch.load(reference_path.with_name(f"sync_{reference_path.name}"),
                                         map_location="cpu", weights_only=True)
    torch.testing.assert_close(actual_initial, sync_initial, atol=0, rtol=0)
    for values, synchronous in zip(actual, sync_steps):
        for value, target in zip(values, synchronous):
            torch.testing.assert_close(value, target, atol=0, rtol=0)
    torch.testing.assert_close(actual_initial, expected_initial, atol=0, rtol=0)
    global_gradient_errors = _global_gradient_errors(reference_path.parent)
    for step, (values, reference) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(torch.tensor(values[0]), torch.tensor(reference[0]), atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(torch.tensor(values[1]), torch.tensor(reference[1]), atol=2e-4, rtol=0.01)
        gradient_error = (values[2] - reference[2]).norm() / reference[2].norm().clamp_min(1e-12)
        update_error = (values[3] - reference[3]).norm() / reference[3].norm().clamp_min(1e-12)
        parameter_error = (values[3] - reference[3]).norm() / (expected_initial + reference[3]).norm()
        print(f"TRAINER CP={cp_size}, U={degree}, shard={shard_size}, step={step + 1}, "
              f"loss={values[0]}, grad_norm={values[1]}, gradient_l2={gradient_error.item()}, "
              f"global_gradient_l2={global_gradient_errors[step].item()}, "
              f"parameter_l2={parameter_error.item()}, update_l2={update_error.item()}", flush=True)
        # FSDP partitions have unequal gradient norms. Compare the logical full
        # gradient, independent of which rank owns each parameter interval.
        if step == 0:
            assert global_gradient_errors[step] < 0.02, f"initial gradient L2={global_gradient_errors[step]}"
        # AdamW approaches sign(g) near zero and can amplify BF16 rounding in a
        # tiny update. Check the full parameter trajectory within BF16 precision;
        # each actual update is independently checked against CPU AdamW above.
        assert parameter_error < torch.finfo(torch.bfloat16).eps, f"step={step}, parameter L2={parameter_error}"
