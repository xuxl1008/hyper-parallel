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
"""Lightweight launchers for document-aware and Zigzag Ring attention."""

from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_ring_attention_layout.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("cards", [2, 4, 8])
def test_ring_layouts(cards: int) -> None:
    """
    Feature: Dense and packed Ring load balancing.
    Description: Compare layouts and gradients against full NPU and FP64 references.
    Expectation: Document isolation, owner gradients, exact overlap parity and bounded memory.
    """
    torchrun_case(_WORKER, "TestRingLayouts", num_proc=cards)
