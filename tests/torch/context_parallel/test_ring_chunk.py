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
"""Lightweight launchers for chunked TND Ring experiments."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_ring_chunk.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ring_chunks_2_cards() -> None:
    """
    Feature: 2-card chunked Ring pipelines.
    Description: Exercise head/token cuts, document pruning, gradients and stream lifetimes.
    Expectation: Match independent references and release all temporary allocations.
    """
    torchrun_case(_WORKER, "TestRingChunk", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ring_chunks_4_cards() -> None:
    """
    Feature: 4-card chunked Ring pipelines.
    Description: Exercise head/token cuts, document pruning, gradients and stream lifetimes.
    Expectation: Match independent references and release all temporary allocations.
    """
    torchrun_case(_WORKER, "TestRingChunk", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_ring_chunks_8_cards() -> None:
    """
    Feature: 8-card chunked Ring pipelines.
    Description: Exercise head/token cuts, document pruning, gradients and stream lifetimes.
    Expectation: Match independent references and release all temporary allocations.
    """
    torchrun_case(_WORKER, "TestRingChunk", num_proc=8)
