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
"""DSA batch boundaries must reach fused attention without CP slicing."""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.data.batching.get_batch import TextParallelBatch
from hyper_parallel.data.batching.runtime_input import AttentionRuntime, RuntimeInputContext
from hyper_parallel.models.deepseek_v32.adapter.data.runtime import DeepseekV32Runtime
from hyper_parallel.models.deepseek_v32.adapter.distributed.document_plan import dsa_document_plan


class TestDSARuntimeInput(unittest.TestCase):
    """Check the packed metadata contract on CPU without native operators."""

    def test_text_batch_forwards_global_ends_and_local_positions(self):
        """A CP shard retains global document ends and document-local positions."""
        runtime = TextParallelBatch.__new__(TextParallelBatch)
        runtime.parallel_context = SimpleNamespace(cp_rank=1, cp_world_size=2, tp_rank=0, tp_world_size=1)
        runtime.runtime_input_adapter = DeepseekV32Runtime()
        runtime.source_type = "online"
        runtime.causal = True
        runtime.sliding_window = None
        runtime.reset_position_ids = True
        runtime.labels_are_shifted = True
        runtime.attention_runtime = AttentionRuntime(mode="compressed", create_mask=True,
                                                     reset_mask=True, sliding_window=None)
        batch = {
            "input_ids": torch.ones(1, 4, dtype=torch.long), "labels": torch.ones(1, 4, dtype=torch.long),
            "loss_mask": torch.ones(1, 4), "cu_seq_lens": torch.tensor([0, 3, 8], dtype=torch.int32),
        }
        # Exercise the actual batch-to-model seam; metadata must survive input splitting.
        # pylint: disable=protected-access
        batch["position_ids"] = runtime._build_local_position_ids(batch["input_ids"], batch["cu_seq_lens"])
        extra = runtime._build_runtime_inputs(batch)
        model_inputs, _ = runtime._split_model_and_loss_inputs(batch, extra)
        self.assertEqual(model_inputs["actual_seq_len"], (3, 8))
        self.assertIsNone(model_inputs["attention_mask"])
        torch.testing.assert_close(model_inputs["position_ids"], torch.tensor([[1, 2, 3, 4]]))
        self.assertEqual(runtime.runtime_input_adapter.runtime_input_fields(), ("actual_seq_len",))

    def test_rejects_invalid_boundaries(self):
        """Do not silently convert invalid metadata to continuous causal attention."""
        context = RuntimeInputContext((2, 4), {"cp": 1}, {"cp": 2}, {})
        adapter = DeepseekV32Runtime()
        self.assertEqual(adapter.build_runtime_inputs(batch={"cu_seq_lens": [0, 3, 8, 16]}, context=context),
                         {"actual_seq_len": (3, 8, 16)})
        for boundaries in (None, [3, 8, 16], [0, 3, 3, 8, 16], [0, 3, 15], [0, 9, 16], [0, 8.0, 16]):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                adapter.build_runtime_inputs(batch={"cu_seq_lens": boundaries}, context=context)

    def test_document_plan_matches_visible_keys_and_gradients(self):
        """Omit absent documents while preserving each local query's causal key prefix."""
        for boundaries, batch, local, size in (((1, 5, 12, 16), 1, 4, 4), ((2, 8, 9, 16), 2, 4, 2)):
            global_length = local * size
            for rank in range(size):
                expected_keys, query_ends, key_ends = [], [], []
                query_count = 0
                for start, end in zip((0, *boundaries[:-1]), boundaries):
                    row = start // global_length
                    shard_start = row * global_length + rank * local
                    queries = range(max(start, shard_start), min(end, shard_start + local))
                    if not queries:
                        continue
                    expected_keys.extend(range(start, queries[-1] + 1))
                    query_count += len(queries)
                    query_ends.append(query_count)
                    key_ends.append(len(expected_keys))
                with self.subTest(boundaries=boundaries, rank=rank):
                    plan = dsa_document_plan(boundaries, batch, local, size, rank)
                    self.assertEqual(plan.query_ends, tuple(query_ends))
                    self.assertEqual(plan.key_ends, tuple(key_ends))
                    bank = torch.arange(boundaries[-1], dtype=torch.float32).view(batch, global_length, 1)
                    bank.requires_grad_(True)
                    packed = plan.key(bank)
                    torch.testing.assert_close(packed.flatten(), torch.tensor(expected_keys, dtype=torch.float32))
                    packed.sum().backward()
                    expected_grad = torch.zeros(boundaries[-1])
                    expected_grad[expected_keys] = 1
                    torch.testing.assert_close(bank.grad.flatten(), expected_grad)
