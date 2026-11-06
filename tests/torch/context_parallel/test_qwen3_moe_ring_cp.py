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
"""Framework-free launcher for Qwen3-MoE Ring CP recipe integration."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_qwen3_moe_ring_cp() -> None:
    """
    Feature: Ring and hybrid CP installed through the Qwen3-MoE recipe wrapper.
    Description: Compare the complete GQA layer and checkpointed backward to a full-sequence reference.
    Expectation: Correct input and CP-summed parameter gradients with unchanged parameter identities.
    """
    worker = str(Path(__file__).with_name("_test_qwen3_moe_ring_cp.py"))
    torchrun_case(worker, "test_attention_layer", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_qwen3_moe_packed_ring_cp() -> None:
    """
    Feature: Packed TND through the Qwen3-MoE Ring wrapper.
    Description: Preserve document positions and parameter gradients with checkpoint replay.
    Expectation: Full-document native reference parity for all four task strategies.
    """
    torchrun_case(str(Path(__file__).with_name("_test_qwen3_moe_ring_cp.py")),
                  "test_packed_attention_layer", num_proc=4)
