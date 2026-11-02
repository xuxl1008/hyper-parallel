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
"""Framework-free launcher for real four-card Ulysses + Ring validation."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ulysses_ring_attention_2x2() -> None:
    """
    Feature: Four-card Ulysses + Ring with two ranks along each dimension.
    Description: Compare output and gradients to full attention, validate subgroups and boundary modes.
    Expectation: Correct local sequence/head ownership, compact GQA, and bounded live memory.
    """
    worker = str(Path(__file__).with_name("_test_ulysses_ring_attention.py"))
    torchrun_case(worker, "TestUlyssesRingAttention", num_proc=4)
