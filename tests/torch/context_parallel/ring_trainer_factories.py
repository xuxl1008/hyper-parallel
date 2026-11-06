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
"""Offline model and complete, equal-length samples for the Ring Trainer ST."""

from typing import Any

import torch
from transformers import Qwen3MoeConfig

from hyper_parallel.models import HyperAutoModelForCausalLM


def build_model(**kwargs: Any) -> torch.nn.Module:
    """Use the normal AutoModel build, recipe replacement and FSDP pipeline."""
    config = Qwen3MoeConfig.from_dict({
        "vocab_size": 256, "hidden_size": 512, "intermediate_size": 512,
        "moe_intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 64,
        "num_experts": 4, "num_experts_per_tok": 2, "max_position_embeddings": 1024,
        "attention_dropout": 0.0, "router_aux_loss_coef": 0.0,
        "tie_word_embeddings": False, "use_cache": False,
    })
    return HyperAutoModelForCausalLM.from_config(config, torch_dtype="bfloat16", **kwargs)


def build_dataset(data_config: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
    """Shift next-token labels globally before the standard CP batch sharder."""
    assert data_config["labels_are_shifted"] and not data_config["create_attention_mask_in_dataloader"]
    generator = torch.Generator().manual_seed(20261103)
    samples = torch.randint(1, 256, (32, 257), generator=generator)
    return [{"tokens": row[:-1].clone(), "labels": row[1:].clone()} for row in samples]
