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
"""Native Ulysses must preserve the full native sequence and its gradients."""
# pylint: disable=forbidden-backend-import
from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch_npu
from torch.utils.checkpoint import checkpoint

import hyper_parallel as hp
from hyper_parallel.components.functional.kimi_delta_attention_ascendc import ascendc_chunk_kda
from hyper_parallel.distributed.context_parallel.kimi_delta_attention import (
    KimiDeltaAttentionCoreCP,
    KimiDeltaAttentionUlyssesCP,
)


def test_kda_ascendc_ulysses() -> None:
    """Check raw input semantics, head-sliced parameters and checkpoint ownership."""
    torch.set_num_threads(1)
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.config.allow_internal_format = False
    dist.init_process_group("hccl", timeout=timedelta(seconds=240))
    try:
        size, rank = dist.get_world_size(), dist.get_rank()
        mesh = hp.init_device_mesh("npu", (size,), mesh_dim_names=("cp",))
        torch.manual_seed(83)
        shape = (1, 4096 * size, 4, 128)
        source = [torch.randn(shape).bfloat16() for _ in range(4)]
        source += [torch.randn(shape[:-1]).bfloat16(), torch.zeros(4), torch.full((4, 128), -5.)]
        full = [value.npu().requires_grad_() for value in source]
        grad_output = torch.randn(shape).bfloat16().npu()
        expected = ascendc_chunk_kda(*full[:5], a_log=full[5], dt_bias=full[6])
        expected_grads = torch.autograd.grad(expected, full, grad_output)
        local = [(value[:, rank * 4096:(rank + 1) * 4096] if index < 5 else value)
                 .detach().clone().requires_grad_() for index, value in enumerate(full)]
        executors = (KimiDeltaAttentionUlyssesCP(mesh, backend="ascendc"),
                     KimiDeltaAttentionCoreCP(mesh, ulysses_degree=size, backend="ascendc"))

        def run(*values: torch.Tensor) -> torch.Tensor:
            """Expose the raw-input core to checkpoint without changing parameters."""
            return executor(*values[:5], a_log=values[5], dt_bias=values[6])

        seed = grad_output[:, rank * 4096:(rank + 1) * 4096].contiguous()
        for executor, checkpointed in zip(executors, (False, True)):
            actual = checkpoint(run, *local, use_reentrant=False) if checkpointed else run(*local)
            grads = torch.autograd.grad(actual, local, seed)
            torch.testing.assert_close(actual, expected[:, rank * 4096:(rank + 1) * 4096], rtol=0, atol=0)
            for index, (grad, reference) in enumerate(zip(grads, expected_grads)):
                if index < 5:
                    reference = reference[:, rank * 4096:(rank + 1) * 4096]
                else:
                    dist.all_reduce(grad)
                torch.testing.assert_close(grad, reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
