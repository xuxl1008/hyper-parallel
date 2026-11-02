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
"""Single-card launchers for the NPU kernel contracts needed by Ring Attention."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_ring_attention_npu.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_ring_attention_npu_block_contract() -> None:
    """
    Feature: Blockwise fused attention with global softmax statistics.
    Description: Simulate ring block visits on one NPU and compare with full attention and CPU math.
    Expectation: MHA/GQA outputs and all Q/K/V gradients agree for both layouts, dtypes and causal modes.
    """
    torchrun_case(_WORKER, "test_npu_block_attention", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_ring_attention_npu_block_boundaries() -> None:
    """
    Feature: Causal block selection across kernel tile boundaries.
    Description: Split 132/1024 tokens into four blocks, including fully masked future blocks.
    Expectation: Skipping future blocks preserves the full attention output and Q/K/V gradients.
    """
    torchrun_case(_WORKER, "test_npu_block_boundaries", num_proc=1)
