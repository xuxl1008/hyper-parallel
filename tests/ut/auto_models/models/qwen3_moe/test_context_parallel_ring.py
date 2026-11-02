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
"""CPU contracts for opt-in Ring recipe injection into Qwen3-MoE GQA."""

from __future__ import annotations

from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from hyper_parallel.distributed._builder.forward_rewriter import _wrap_inner_attention
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel_ring import qwen3_moe_ring_cp_wrapper
from hyper_parallel.trainer.config.target import Target

_MODULE = "hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel_ring"
_WRAPPER = _MODULE + ".qwen3_moe_ring_cp_wrapper"


class _GQAStub(nn.Module):
    def __init__(self) -> None:
        """Create only CPU state and a recording attention callable."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.num_heads, self.num_key_value_heads = 4, 2
        self.is_causal = True
        self.attention_interface = Mock()

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Any = None,
                attention_mask: torch.Tensor | None = None, past_key_values: Any = None,
                actual_seq_len: Any = None, **kwargs: Any) -> Any:
        """Mimic the GQA call contract without any model kernel dependency."""
        del position_embeddings, past_key_values
        return self.attention_interface(
            self, hidden_states, hidden_states[:, :2], hidden_states[:, :2], attention_mask,
            scaling=0.125, actual_seq_len=actual_seq_len, **kwargs,
        )


class TestQwenRingWrapper(unittest.TestCase):
    """Exercise actual config binding and rewriting without kernels or process groups."""

    def setUp(self) -> None:
        """Use mocked mesh metadata without initializing distributed state."""
        self.mesh = Mock(ndim=1)
        self.mesh.size.return_value = 4
        self.target = _GQAStub()
        self.query = torch.randn(1, 4, 8, 64)

    def test_target_binding_and_atomic_rewrite(self) -> None:
        original_forward = self.target.forward
        original_interface = self.target.attention_interface
        original_weight = self.target.weight
        request = qwen3_moe_ring_cp_wrapper(self.target, None, None, self.mesh, None,
                                            ulysses_degree=2, overlap=True)
        self.assertIs(self.target.forward.__func__, original_forward.__func__)
        self.assertIs(self.target.attention_interface, original_interface)
        self.assertEqual(set(request.companion_attrs), {"attention_interface"})
        spec = ModuleShardingSpec(
            inner_target="self", region_dispatch=False,
            inner_wrapper=Target(qwen3_moe_ring_cp_wrapper, target_path=_WRAPPER,
                                 ulysses_degree=2, overlap=True),
        )
        _wrap_inner_attention(self.target, self.mesh, spec=spec, mesh=self.mesh, mesh_dim_names=("cp",))
        self.assertIs(self.target.weight, original_weight)
        with patch(_MODULE + ".ulysses_ring_attention", return_value=self.query + 1) as attention:
            output, weights = self.target(self.query)
        torch.testing.assert_close(output, (self.query + 1).transpose(1, 2))
        self.assertIsNone(weights)
        self.assertIs(attention.call_args.args[3], self.mesh)
        self.assertEqual(attention.call_args.kwargs, {
            "ulysses_degree": 2, "causal": True, "scale": 0.125, "input_layout": "BNSD",
            "dropout_p": 0.0, "overlap": True, "document_lengths": None,
            "load_balance": "none", "prune_documents": False, "balance_block_size": None, "zigzag_min_length": None,
        })

    def test_packed_uses_global_lengths_and_restores_shape(self) -> None:
        request = qwen3_moe_ring_cp_wrapper(self.target, None, None, self.mesh, None,
                                            load_balance="zigzag")
        interface = request.companion_attrs["attention_interface"]
        expected = self.query.squeeze(0).transpose(0, 1).contiguous()
        with patch(_MODULE + ".ulysses_ring_attention", return_value=expected) as attention:
            output, _ = interface(self.target, self.query, self.query[:, :2], self.query[:, :2], None,
                                  document_lengths=(7, 25))
        self.assertEqual(output.shape, (1, 8, 4, 64))
        self.assertEqual(attention.call_args.args[0].shape, (8, 4, 64))
        self.assertEqual(attention.call_args.kwargs["document_lengths"], (7, 25))
        self.assertEqual(attention.call_args.kwargs["load_balance"], "zigzag")

    def test_unsupported_semantics_fail_before_ring(self) -> None:
        request = qwen3_moe_ring_cp_wrapper(self.target, None, None, self.mesh, None)
        interface = request.companion_attrs["attention_interface"]
        cases = (
            ({"attention_mask": torch.ones(8, 32, dtype=torch.bool)}, "explicit mask"),
            ({"dropout": 0.1}, "dropout=0"),
            ({"actual_seq_len": [8]}, "actual_seq_len"),
            ({"cu_seq_lens_q": torch.tensor([0, 8])}, "cu_seq_lens_q"),
            ({"cu_seq_lens": torch.tensor([0, 8])}, "cu_seq_lens"),
            ({"packed_seq_params": object()}, "packed_seq_params"),
            ({"sliding_window": 128}, "sliding_window"),
            ({"output_attentions": True}, "attention weights"),
        )
        with patch(_MODULE + ".ulysses_ring_attention") as attention:
            for options, message in cases:
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                    interface(self.target, self.query, self.query[:, :2], self.query[:, :2],
                              **{"attention_mask": None, **options})
            attention.assert_not_called()

    def test_cache_and_packing_fail_before_original_forward(self) -> None:
        original = Mock()
        self.target.forward = original
        request = qwen3_moe_ring_cp_wrapper(self.target, None, None, self.mesh, None)
        with self.assertRaisesRegex(ValueError, "KV cache"):
            request.forward(self.query, None, None, object())
        with self.assertRaisesRegex(ValueError, "actual_seq_len"):
            request.forward(self.query, actual_seq_len=[8])
        original.assert_not_called()

    def test_invalid_configuration_keeps_target_unchanged(self) -> None:
        original = self.target.attention_interface
        for options in ({"ulysses_degree": 3}, {"overlap": "true"}, {"ulysses_degree": 4},
                        {"prune_documents": "true"}, {"prune_documents": True, "load_balance": "zigzag"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                qwen3_moe_ring_cp_wrapper(self.target, None, None, self.mesh, None, **options)
            self.assertIs(self.target.attention_interface, original)

    def test_pruning_binding_and_missing_documents(self) -> None:
        original = Mock()
        self.target.forward = original
        request = qwen3_moe_ring_cp_wrapper(self.target, None, None, self.mesh, None, prune_documents=True)
        with self.assertRaisesRegex(ValueError, "document_lengths"):
            request.forward(self.query)
        original.assert_not_called()
        interface = request.companion_attrs["attention_interface"]
        expected = self.query.squeeze(0).transpose(0, 1).contiguous()
        with patch(_MODULE + ".ulysses_ring_attention", return_value=expected) as attention:
            interface(self.target, self.query, self.query[:, :2], self.query[:, :2], None, document_lengths=(7, 25))
        self.assertIs(attention.call_args.kwargs["prune_documents"], True)
        self.assertEqual(attention.call_args.kwargs["input_layout"], "TND")

    def test_recipe_typo_is_rejected_without_mutation(self) -> None:
        original = self.target.attention_interface
        spec = ModuleShardingSpec(
            inner_target="self", region_dispatch=False,
            inner_wrapper=Target(qwen3_moe_ring_cp_wrapper, target_path=_WRAPPER, overlapp=True),
        )
        with self.assertRaisesRegex(ValueError, "undeclared keys"):
            _wrap_inner_attention(self.target, self.mesh, spec=spec, mesh=self.mesh, mesh_dim_names=("cp",))
        self.assertIs(self.target.attention_interface, original)
