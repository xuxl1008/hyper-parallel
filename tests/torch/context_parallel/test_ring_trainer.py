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
"""Lightweight launcher for offline Trainer + Ring CP + FSDP training."""

from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("case", ["cp8_ring_fsdp8", "cp8_u2r4_fsdp8", "cp8_u4r2_fsdp8",
                                  "dp2_cp4_u2r2_hsdp2x4"])
def test_ring_trainer_8_cards(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Feature: Real TextTrainer with Ring/hybrid CP and FSDP/HSDP.
    Description: Train offline Qwen3-MoE with recomputation, accumulation and FP32 AdamW states.
    Expectation: AllGather CP initial-gradient parity and exact sync/overlap training trajectories.
    """
    monkeypatch.setenv("HYPER_CP_TRAINER_REFERENCE_DIR", str(tmp_path))
    # Trainer.train owns process-group destruction; separate runs avoid HCCL reinitialization.
    for mode in ("control", "sync", "ring"):
        monkeypatch.setenv("HYPER_CP_TRAINER_MODE", mode)
        torchrun_case(str(Path(__file__).with_name("_test_ring_trainer.py")),
                      f"test_trainer_fsdp[{case}]", num_proc=8)
