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
"""CPU coverage for the CANN DSA sparse-attention autograd adapter."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional.dsa_sparse_attention import _cann_sparse_attention, dsa_sparse_attention


class TestSparseAttentionAutograd(unittest.TestCase):
    """Check lazy backward loading and the shared K/V gradient contract."""

    def test_backward_is_lazy_and_returns_gradients_for_all_inputs(self):
        """
        Feature: Sparse-attention autograd adapter.
        Description: Run forward before lazily loading and invoking the gradient operator.
        Expectation: Return query and shared key/value gradients for all tensor inputs.
        """
        query = torch.randn(2, 1, 3, requires_grad=True)
        key = torch.randn(2, 1, 3, requires_grad=True)
        indices = torch.zeros(2, 1, 1, dtype=torch.int32)
        query_lengths = torch.tensor([2], dtype=torch.int32)
        key_lengths = torch.tensor([2], dtype=torch.int32)
        attention_op = Mock(side_effect=lambda q, *_args, **_kwargs: (
            q.clone(), torch.zeros(1), torch.ones(1),
        ))
        expected_dq = torch.full_like(query, 2)
        expected_dkv = torch.full_like(key, 8)
        grad_op = Mock(return_value=(expected_dq, expected_dkv, torch.empty(0), None, None, torch.empty(0)))
        grad_loader = Mock(return_value=grad_op)

        output, _, _ = _cann_sparse_attention(
            query, key, indices, 0.5,
            query_rope=None,
            key_rope=None,
            query_lengths=query_lengths,
            key_lengths=key_lengths,
            layout="TND",
            attention_op=attention_op,
            grad_op=grad_loader,
        )

        grad_loader.assert_not_called()
        self.assertIs(attention_op.call_args.kwargs["actual_seq_lengths_query"], query_lengths)
        output.sum().backward()

        grad_loader.assert_called_once_with()
        torch.testing.assert_close(query.grad, expected_dq)
        torch.testing.assert_close(key.grad, expected_dkv)

    def test_unpadded_output_preserves_values_and_projection_gradient(self):
        """
        Feature: Optional unpadded sparse-attention output.
        Description: Consume value channels with a noncontiguous projection gradient.
        Expectation: Both output modes retain input gradients and native value width.
        """
        query = torch.randn(1, 3, 2, 4, requires_grad=True)
        key = torch.randn(1, 5, 1, 4, requires_grad=True)
        query_rope = torch.randn(1, 3, 2, 2, requires_grad=True)
        key_rope = torch.randn(1, 5, 1, 2, requires_grad=True)
        seed = torch.randn(1, 2, 3, 4).transpose(1, 2)
        backend = ModuleType("torch_npu")
        backend.npu_sparse_flash_attention = Mock(side_effect=lambda q, *_args, **_kwargs: (
            q.clone(), torch.zeros(1), torch.ones(1),
        ))

        def native_backward(q, k, _value, _indices, gradient, output, *_args, **_kwargs):
            """Check the gradient layout and value-only native output contract."""
            self.assertEqual(output.shape[-1], 4)
            self.assertTrue(gradient.is_contiguous())
            return (gradient.clone(), torch.ones_like(k), None,
                    torch.ones_like(q[..., :2]), torch.ones_like(k[..., :2]), None)

        outputs, gradients = [], []
        with patch.dict(sys.modules, {"torch_npu": backend}), patch(
            "hyper_parallel.components.functional.dsa_sparse_attention._load_sparse_attention_grad",
            return_value=native_backward,
        ):
            for padded in (True, False):
                output, _, _ = dsa_sparse_attention(
                    query, key, query_rope, key_rope, torch.zeros(3, 1, 2, dtype=torch.int32),
                    0.5, torch.tensor([3]), torch.tensor([5]), pad_output=padded,
                )
                self.assertEqual(output.shape[-1], 6 if padded else 4)
                if padded:
                    self.assertEqual(torch.count_nonzero(output[..., 4:]).item(), 0)
                outputs.append(output[..., :4].detach())
                gradients.append(torch.autograd.grad(output[..., :4], (query, key, query_rope, key_rope), seed))
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        for old, new in zip(*gradients):
            torch.testing.assert_close(old, new, atol=0, rtol=0)
