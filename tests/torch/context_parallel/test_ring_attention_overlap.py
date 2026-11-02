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
"""Lightweight launchers for overlapped NPU Ring and hybrid CP."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_ring_attention_overlap.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ring_overlap_2_cards() -> None:
    """
    Feature: Two-card Ring communication/computation overlap.
    Description: Verify prefetch, gradient return, stream dependencies and memory lifetimes.
    Expectation: Full-reference parity and bounded live memory.
    """
    torchrun_case(_WORKER, "TestRingAttentionOverlap", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ring_overlap_4_cards() -> None:
    """
    Feature: Four-card Ring and 2x2 Ulysses + Ring overlap.
    Description: Exercise complete pipelined gradient cycles and noncontiguous Ring subgroups.
    Expectation: Exact synchronous parity on default and nondefault compute streams.
    """
    torchrun_case(_WORKER, "TestRingAttentionOverlap", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ring_overlap_8_cards() -> None:
    """
    Feature: Eight-card Ring, 2x4 and 4x2 Ulysses + Ring overlap.
    Description: Check full-reference gradients, remote owners, strided inputs and buffer lifetimes.
    Expectation: Preserve strict NPU/FP64 accuracy and exact synchronous stream parity.
    """
    torchrun_case(_WORKER, "TestRingAttentionOverlap", num_proc=8)
