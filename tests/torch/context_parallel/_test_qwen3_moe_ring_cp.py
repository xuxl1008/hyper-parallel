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
"""Qwen3-MoE attention-layer parity through the recipe forward-rewrite machinery."""

from copy import deepcopy
from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

from hyper_parallel import init_device_mesh
from hyper_parallel.distributed._builder.forward_rewriter import _wrap_inner_attention
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.qwen3_moe.adapter.conversion.module_replacement import replace_qwen3_moe_flash_attention
from hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel import qwen3_moe_flash_attention_cp_wrapper
from hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel_ring import qwen3_moe_ring_cp_wrapper
from hyper_parallel.trainer.config.target import Target
from tests.torch.context_parallel._test_ring_attention_npu import _Case, _compare

@pytest.fixture(name="environment", scope="module")
def _environment():
    if not torch.npu.is_available() or torch.npu.device_count() < 4:
        pytest.skip("Qwen3-MoE Ring integration requires four visible NPU devices")
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        mesh = init_device_mesh("npu", (4,), mesh_dim_names=("cp",))
        yield device, mesh
    finally:
        dist.destroy_process_group()


def _build_attention(device, dtype):
    torch.manual_seed(314159)
    config = Qwen3MoeConfig.from_dict({
        "hidden_size": 256, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 64, "num_hidden_layers": 1, "attention_dropout": 0.0,
    })
    source = Qwen3MoeAttention(config, layer_idx=0)
    return replace_qwen3_moe_flash_attention(
        module=source, module_fqn="model.layers.0.self_attn", context={},
    ).to(device=device, dtype=dtype).train()


def _run_layer(module, hidden, positions, checkpointed, **kwargs):
    if checkpointed:
        return checkpoint(module, hidden, position_embeddings=positions, use_reentrant=False, **kwargs)[0]
    return module(hidden, position_embeddings=positions, **kwargs)[0]


def _compare_parameter_gradient(gradient, reference, case, name):
    gradient, reference = gradient.cpu().double(), reference.detach().cpu().double()
    rms = reference.square().mean().sqrt().clamp_min(1e-12)
    gradient, reference = gradient / rms, reference / rms
    epsilon = torch.finfo(case.dtype).eps
    # Each CP rank rounds its own GEMM gradient before the sum. Use a CP-sized
    # cancellation allowance in reference-RMS units, calibrated by the existing
    # AllGather CP control. Keep the primitive test's stricter relative L2 bound.
    torch.testing.assert_close(gradient, reference, atol=case.parts * epsilon,
                               rtol=0.012 if case.dtype == torch.float16 else 0.06,
                               msg=lambda message: f"parameter {name}: {message}")
    difference = gradient - reference
    relative_l2 = difference.norm() / reference.norm().clamp_min(1e-12)
    assert relative_l2 <= epsilon, f"parameter {name}: relative L2={relative_l2}, limit={epsilon}"
    return difference.abs().max().item(), relative_l2.item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("mode", [(0, False), (1, False), (2, False), (1, True), (2, True)],
                         ids=["allgather_baseline", "ring_sync", "hybrid_sync", "ring_overlap", "hybrid_overlap"])
@pytest.mark.parametrize("load_balance", ["none", "zigzag"])
@pytest.mark.parametrize("checkpointed", [False, True], ids=["direct", "checkpoint"])
def test_attention_layer(environment, dtype, mode, checkpointed, load_balance) -> None:
    """Compare full projection/RoPE/output paths and CP-summed parameter gradients."""
    device, mesh = environment
    degree, overlap = mode
    reference = _build_attention(device, dtype)
    parallel = deepcopy(reference)
    parameters = dict(parallel.named_parameters())
    wrapper = qwen3_moe_flash_attention_cp_wrapper if degree == 0 else qwen3_moe_ring_cp_wrapper
    options = {} if degree == 0 else {"ulysses_degree": degree, "overlap": overlap, "load_balance": load_balance}
    spec = ModuleShardingSpec(
        inner_target="self", region_dispatch=False,
        inner_wrapper=Target(wrapper, target_path=wrapper.__module__ + "." + wrapper.__name__, **options),
    )
    _wrap_inner_attention(parallel, mesh, spec=spec, mesh=mesh, mesh_dim_names=("cp",))
    assert all(parameter is parameters[name] for name, parameter in parallel.named_parameters())

    generator = torch.Generator().manual_seed(20261102)
    hidden = torch.randn(2, 512, 256, generator=generator).to(device=device, dtype=dtype).requires_grad_()
    upstream = torch.randn(2, 512, 256, generator=generator).to(device=device, dtype=dtype)
    frequency = torch.arange(512).float().unsqueeze(-1) * (10000.0 ** (-torch.arange(0, 64, 2).float() / 64))
    angles = torch.cat((frequency, frequency), dim=-1).unsqueeze(0).expand(2, -1, -1)
    positions = tuple(tensor.to(device=device, dtype=dtype) for tensor in (angles.cos(), angles.sin()))
    expected = _run_layer(reference, hidden, positions, checkpointed)
    expected.backward(upstream)

    start, length = mesh.get_local_rank() * 128, 128
    local_hidden = hidden.detach().narrow(1, start, length).clone().requires_grad_()
    local_positions = tuple(tensor.narrow(1, start, length).contiguous() for tensor in positions)
    actual = _run_layer(parallel, local_hidden, local_positions, checkpointed)
    actual.backward(upstream.narrow(1, start, length).contiguous())
    case = _Case(dtype, "BNSD", 2, True, mesh.size(), sequence_length=512)
    parameter_gradients = []
    for (name, parameter), (reference_name, reference_parameter) in zip(
        parallel.named_parameters(), reference.named_parameters(),
    ):
        assert name == reference_name
        assert parameter.grad is not None, f"missing gradient: {name}"
        gradient = parameter.grad.float()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=mesh.get_group())
        parameter_gradients.append((name, gradient, reference_parameter.grad))
    # Finish every collective before local assertions so a rank's numerical
    # failure cannot strand its peers inside a later parameter all-reduce.
    errors = [_compare(actual, expected.narrow(1, start, length), case, "attention-layer output"),
              _compare(local_hidden.grad, hidden.grad.narrow(1, start, length), case, "attention-layer dInput")]
    errors.extend(_compare_parameter_gradient(gradient, reference_gradient, case, name)
                  for name, gradient, reference_gradient in parameter_gradients)
    print(f"rank={mesh.get_local_rank()}, dtype={dtype}, U={degree}, overlap={overlap}, "
          f"checkpoint={checkpointed}, errors={errors}")


@pytest.mark.parametrize("mode", ["none", "zigzag", "greedy", "hybrid", "pruned"])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_packed_attention_layer(environment, mode, degree, checkpointed) -> None:
    """Preserve per-document RoPE and projection gradients through packed Ring routing.

    Args:
        environment: Initialized NPU device and CP mesh fixture.
        mode: Packed layout or document-pruning variant.
        degree: Ulysses degree within the CP mesh.
        checkpointed: Whether to recompute attention during backward.
    """
    device, mesh = environment
    reference = _build_attention(device, torch.bfloat16)
    parallel = deepcopy(reference)
    options = {"ulysses_degree": degree, "overlap": True, "load_balance": "none" if mode == "pruned" else mode,
               "prune_documents": mode == "pruned",
               "balance_block_size": 16 if mode in ("greedy", "hybrid") else None,
               "zigzag_min_length": 128 if mode == "hybrid" else None}
    spec = ModuleShardingSpec(inner_target="self", region_dispatch=False, inner_wrapper=Target(
        qwen3_moe_ring_cp_wrapper, target_path=qwen3_moe_ring_cp_wrapper.__module__ +
        ".qwen3_moe_ring_cp_wrapper", **options))
    _wrap_inner_attention(parallel, mesh, spec=spec, mesh=mesh, mesh_dim_names=("cp",))
    generator = torch.Generator().manual_seed(314160)
    hidden = torch.randn(1, 256, 256, generator=generator).to(device=device, dtype=torch.bfloat16).requires_grad_()
    upstream = torch.randn(1, 256, 256, generator=generator).to(device=device, dtype=torch.bfloat16)
    lengths = (7, 65, 184)
    positions = torch.cat([torch.arange(length) for length in lengths]).float()
    frequency = positions[:, None] * (10000.0 ** (-torch.arange(0, 64, 2).float() / 64))
    angles = torch.cat((frequency, frequency), -1).unsqueeze(0)
    embeddings = tuple(tensor.to(device=device, dtype=torch.bfloat16) for tensor in (angles.cos(), angles.sin()))
    # The default Qwen interface ignores packed length kwargs. Its explicit
    # block mask supplies an independent, full-sequence document-isolated oracle.
    documents = torch.tensor([doc for doc, length in enumerate(lengths) for _ in range(length)], device=device)
    allowed = (documents[:, None] == documents[None, :]) & (
        torch.arange(256, device=device)[:, None] >= torch.arange(256, device=device)[None, :])
    expected = _run_layer(reference, hidden, embeddings, checkpointed, attention_mask=allowed[None, None])
    expected.backward(upstream)
    start, count = mesh.get_local_rank() * 64, 64
    local_hidden = hidden.detach().narrow(1, start, count).clone().requires_grad_()
    local_embeddings = tuple(tensor.narrow(1, start, count).contiguous() for tensor in embeddings)
    actual = _run_layer(parallel, local_hidden, local_embeddings, checkpointed, document_lengths=lengths)
    actual.backward(upstream.narrow(1, start, count).contiguous())
    gradients = []
    for (name, parameter), (_, expected_parameter) in zip(parallel.named_parameters(), reference.named_parameters()):
        gradient = parameter.grad.float()
        dist.all_reduce(gradient, group=mesh.get_group())
        gradients.append((gradient, expected_parameter.grad, name))
    case = _Case(torch.bfloat16, "BNSD", 2, True, mesh.size())
    _compare(actual, expected.narrow(1, start, count), case, "packed layer O")
    _compare(local_hidden.grad, hidden.grad.narrow(1, start, count), case, "packed layer dInput")
    for gradient, expected_gradient, name in gradients:
        _compare_parameter_gradient(gradient, expected_gradient, case, name)
