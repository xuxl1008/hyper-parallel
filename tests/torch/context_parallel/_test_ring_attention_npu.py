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
"""Prove the NPU forward/backward contract before adding ring communication.

This worker deliberately keeps the block traversal local. It validates the
kernel math, not P2P ordering, distributed memory use or training integration.
Each Q block visits compact KV blocks in ring order; causal future blocks are
skipped. Backward receives the final output and global softmax statistics for
that Q block, instead of the independently normalized block outputs.

The initial contract is fixed-length self-attention, equal Q/K/V head dimensions
and dropout=0. Packed sequences, external masks and MLA are separate contracts.
"""

import os
from dataclasses import dataclass
from typing import Tuple

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

_UNLIMITED_TOKENS = 2147483647


@dataclass(frozen=True)
class _Case:
    dtype: torch.dtype
    layout: str
    kv_heads: int
    causal: bool
    parts: int
    sequence_length: int = 128
    query_heads: int = 4
    head_dim: int = 64
    batch_size: int = 2

    @property
    def seq_dim(self) -> int:
        """Return the sequence axis in the native kernel layout."""
        return 2 if self.layout == "BNSD" else 1

    @property
    def scale(self) -> float:
        """Return the attention logit scale."""
        return self.head_dim ** -0.5


@dataclass
class _SoftmaxState:
    output: torch.Tensor
    maximum: torch.Tensor
    denominator: torch.Tensor


@pytest.fixture(name="npu_device", scope="module")
def _npu_device() -> torch.device:
    if not torch.npu.is_available():
        pytest.skip("Ring attention kernel validation requires an NPU")
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.set_device(device)
    return device


def _kernel_options(case: _Case, causal_mask: torch.Tensor, diagonal_causal: bool) -> dict:
    return {
        "head_num": case.query_heads,
        "input_layout": case.layout,
        "atten_mask": causal_mask if diagonal_causal else None,
        "sparse_mode": 2 if diagonal_causal else 0,
        "pre_tockens": _UNLIMITED_TOKENS,
        "next_tockens": 0 if diagonal_causal else _UNLIMITED_TOKENS,
        "keep_prob": 1.0,
    }


def _block_order(case: _Case, query_owner: int) -> Tuple[int, ...]:
    owners = tuple((query_owner - step) % case.parts for step in range(case.parts))
    return tuple(owner for owner in owners if not case.causal or owner <= query_owner)


def _merge(state: _SoftmaxState, block: _SoftmaxState, layout: str) -> _SoftmaxState:
    maximum = torch.maximum(state.maximum, block.maximum)
    previous_mass = state.denominator * torch.exp(state.maximum - maximum)
    block_mass = block.denominator * torch.exp(block.maximum - maximum)
    denominator = previous_mass + block_mass
    previous_weight = (previous_mass / denominator)[..., :1]
    block_weight = (block_mass / denominator)[..., :1]
    if layout == "BSND":
        previous_weight = previous_weight.transpose(1, 2)
        block_weight = block_weight.transpose(1, 2)
    output = state.output * previous_weight + block.output * block_weight
    return _SoftmaxState(output, maximum, denominator)


def _forward_query_block(query, key_blocks, value_blocks, query_owner, case, causal_mask):
    state = None
    for owner in _block_order(case, query_owner):
        options = _kernel_options(case, causal_mask, case.causal and owner == query_owner)
        output, maximum, denominator = torch_npu.npu_fusion_attention(
            query, key_blocks[owner], value_blocks[owner], scale=case.scale, **options,
        )[:3]
        expected_shape = (case.batch_size, case.query_heads, query.shape[case.seq_dim], 8)
        for name, statistic in (("maximum", maximum), ("denominator", denominator)):
            assert statistic.shape == expected_shape, (
                f"{name} shape: actual={statistic.shape}, expected={expected_shape}"
            )
            assert statistic.dtype == torch.float32, (
                f"{name} dtype: actual={statistic.dtype}, expected={torch.float32}"
            )
        block = _SoftmaxState(output.float(), maximum, denominator)
        state = block if state is None else _merge(state, block, case.layout)
    return state


@torch.no_grad()
def _blocked_attention(query, key, value, grad_output, case, causal_mask):
    query_blocks, key_blocks, value_blocks, grad_blocks = (
        tuple(block.contiguous() for block in tensor.chunk(case.parts, dim=case.seq_dim))
        for tensor in (query, key, value, grad_output)
    )
    outputs = []
    grad_query = [torch.zeros_like(block, dtype=torch.float32) for block in query_blocks]
    grad_key = [torch.zeros_like(block, dtype=torch.float32) for block in key_blocks]
    grad_value = [torch.zeros_like(block, dtype=torch.float32) for block in value_blocks]

    for query_owner, query_block in enumerate(query_blocks):
        state = _forward_query_block(query_block, key_blocks, value_blocks, query_owner, case, causal_mask)
        final_output = state.output.to(case.dtype)
        outputs.append(final_output)
        for owner in _block_order(case, query_owner):
            options = _kernel_options(case, causal_mask, case.causal and owner == query_owner)
            # Global normalization and final O account for every KV block in this gradient contribution.
            delta_query, delta_key, delta_value = torch_npu.npu_fusion_attention_grad(
                query_block, key_blocks[owner], value_blocks[owner], grad_blocks[query_owner],
                softmax_max=state.maximum,
                softmax_sum=state.denominator,
                attention_in=final_output,
                scale_value=case.scale,
                **options,
            )[:3]
            grad_query[query_owner].add_(delta_query.float())
            grad_key[owner].add_(delta_key.float())
            grad_value[owner].add_(delta_value.float())

    return tuple(torch.cat(blocks, dim=case.seq_dim).to(case.dtype)
                 for blocks in (outputs, grad_query, grad_key, grad_value))


def _native_reference(query, key, value, grad_output, case, causal_mask):
    inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in (query, key, value))
    options = _kernel_options(case, causal_mask, case.causal)
    output = torch_npu.npu_fusion_attention(*inputs, scale=case.scale, **options)[0]
    gradients = torch.autograd.grad(output, inputs, grad_outputs=grad_output)
    return (output.detach(), *gradients)


def _math_reference(tensors, case):
    query, key, value, grad_output = tensors
    inputs = tuple(tensor.detach().double().requires_grad_() for tensor in (query, key, value))
    math_query, math_key, math_value = inputs
    if case.layout == "BSND":
        math_query, math_key, math_value = (tensor.transpose(1, 2) for tensor in inputs)
    groups = case.query_heads // case.kv_heads
    math_key = math_key.repeat_interleave(groups, dim=1)
    math_value = math_value.repeat_interleave(groups, dim=1)
    logits = torch.matmul(math_query, math_key.transpose(-1, -2)) * case.scale
    if case.causal:
        mask = torch.ones(case.sequence_length, case.sequence_length, dtype=torch.bool).triu(1)
        logits = logits.masked_fill(mask, -float("inf"))
    output = torch.matmul(logits.softmax(dim=-1), math_value)
    if case.layout == "BSND":
        output = output.transpose(1, 2)
    gradients = torch.autograd.grad(output, inputs, grad_outputs=grad_output.double())
    return (output.detach(), *gradients)


def _inputs(case: _Case) -> Tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20261101)
    tensors = []
    for heads in (case.query_heads, case.kv_heads, case.kv_heads, case.query_heads):
        tensor = torch.randn(case.batch_size, heads, case.sequence_length, case.head_dim, generator=generator)
        if case.layout == "BSND":
            tensor = tensor.transpose(1, 2)
        tensors.append(tensor.to(case.dtype).contiguous())
    return tuple(tensors)


def _compare(actual, expected, case, label):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    # Fused backward rounds intermediate probabilities and gradients in the input dtype.
    # Near-zero cancellation needs an absolute bound; the L2 bound also limits aggregate error.
    epsilon = torch.finfo(case.dtype).eps
    atol, rtol = 2 * epsilon, 0.012 if case.dtype == torch.float16 else 0.06
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol, msg=lambda message: f"{label}, {case}: {message}",
    )
    relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    limit = epsilon
    assert relative_error <= limit, (
        f"{label}, {case}: relative L2 error={relative_error.item():.6g}, expected<={limit}"
    )
    return (actual - expected).abs().max().item(), relative_error.item()


def _run_case(case: _Case, device: torch.device) -> None:
    cpu_inputs = _inputs(case)
    npu_inputs = tuple(tensor.to(device) for tensor in cpu_inputs)
    causal_mask = torch.ones((2048, 2048), dtype=torch.bool, device=device).triu(1)
    math_result = _math_reference(cpu_inputs, case)
    native_result = _native_reference(*npu_inputs, case, causal_mask)
    blocked_result = _blocked_attention(*npu_inputs, case, causal_mask)
    errors = []
    for name, blocked, native, reference in zip(("output", "dQ", "dK", "dV"),
                                               blocked_result, native_result, math_result):
        _compare(native, reference, case, f"native/{name} vs CPU FP64")
        _compare(blocked, reference, case, f"blocked/{name} vs CPU FP64")
        error = _compare(blocked, native, case, f"blocked/{name} vs native")
        errors.append(f"{name}: max_abs={error[0]:.6g}, rel_l2={error[1]:.6g}")
    print(f"{case}: {'; '.join(errors)}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("layout", ["BNSD", "BSND"])
@pytest.mark.parametrize("kv_heads", [4, 2, 1], ids=["mha", "gqa", "mqa"])
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("parts", [1, 2, 4], ids=["parts1", "parts2", "parts4"])
def test_npu_block_attention(
    npu_device: torch.device,
    dtype: torch.dtype,
    layout: str,
    kv_heads: int,
    causal: bool,
    parts: int,
) -> None:
    """Compare full attention with local ring-order block traversal and explicit block backward."""
    _run_case(_Case(dtype, layout, kv_heads, causal, parts), npu_device)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("layout", ["BNSD", "BSND"])
@pytest.mark.parametrize("sequence_length", [132, 1024], ids=["unaligned", "multiple_tiles"])
def test_npu_block_boundaries(
    npu_device: torch.device, dtype: torch.dtype, layout: str, sequence_length: int,
) -> None:
    """Cover partial kernel tiles and causal blocks masked by their global positions."""
    _run_case(_Case(dtype, layout, kv_heads=2, causal=True, parts=4, sequence_length=sequence_length), npu_device)
