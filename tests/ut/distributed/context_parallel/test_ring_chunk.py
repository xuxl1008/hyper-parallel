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
"""Exhaustive original-coordinate oracles for chunked Ring masks and traffic pruning."""

from itertools import accumulate, product
import unittest

from hyper_parallel.distributed.context_parallel.ring_chunk import chunk_plans


def _actual_pairs(lengths, size, causal, axis, chunks, prune):
    local = sum(lengths) // size
    actual = []
    for rank in range(size):
        plans = chunk_plans(lengths, size, rank, 4, 2, axis, chunks, causal, prune)
        for plan in plans:
            for owner in range(size):
                spans = plan.spans[owner]
                blocks = plan.blocks[owner]
                if owner == rank:
                    spans = ((rank * local + plan.token_range[0],
                              rank * local + plan.token_range[1]),)
                    blocks = plan.local_blocks
                keys = [row for start, end in spans for row in range(start, end)]
                for block in blocks:
                    qcount = block.query_end - block.query_start
                    kcount = block.key_end - block.key_start
                    actual.extend((h, rank * local + block.query_start + qi,
                                   keys[block.key_start + ki])
                                  for h in range(*plan.query_heads)
                                  for qi in range(qcount) for ki in range(kcount)
                                  if not block.causal or ki <= qi + kcount - qcount)
    return sorted(actual)


class TestRingChunks(unittest.TestCase):
    """Compare generated blocks against independent document-local attention graphs."""

    def test_pairs_once(self):
        """Head/token cuts, tails and pruning preserve every legal pair exactly once."""
        cases = product(((24,), (3, 5, 16), (1,) * 24, (7, 13, 4)), (1, 2, 4, 8), (False, True),
                        (("head", 1), ("head", 2), ("sequence", 2), ("sequence", 3)), (False, True))
        for lengths, size, causal, (axis, chunks), prune in cases:
            documents = [(doc, pos) for doc, length in enumerate(lengths) for pos in range(length)]
            expected = sorted((h, qi, ki) for h in range(4)
                              for qi, q in enumerate(documents) for ki, k in enumerate(documents)
                              if q[0] == k[0] and (not causal or k[1] <= q[1]))
            with self.subTest(lengths=lengths, size=size, causal=causal, axis=axis, chunks=chunks, prune=prune):
                actual = _actual_pairs(lengths, size, causal, axis, chunks, prune)
                self.assertEqual(actual, expected, f"pair counts actual={len(actual)}, expected={len(expected)}")

    def test_pruning_is_exact(self):
        """Transport contains precisely the KV rows needed by at least one remote rank."""
        lengths, size = (3, 9, 2, 10), 4
        local = sum(lengths) // size
        ends = (0, *accumulate(lengths))
        for causal in (False, True):
            plan = chunk_plans(lengths, size, 0, 4, 2, causal=causal, prune=True)[0]
            for owner, spans in enumerate(plan.spans):
                actual = {row for start, end in spans for row in range(start, end)}
                expected = {key for start, end in zip(ends, ends[1:]) for key in range(start, end)
                            if key // local == owner and any(query // local != owner
                                                             and (not causal or key <= query)
                                                             for query in range(start, end))}
                self.assertEqual(actual, expected, f"transport actual={actual}, expected={expected}")

    def test_empty_remote_and_uneven_chunks(self):
        """Local documents need no remote payload; sequence tails are retained."""
        plans = chunk_plans((5, 5, 5, 5), 4, 0, 4, 2, "sequence", 3, True, True)
        self.assertEqual([p.token_range for p in plans], [(0, 1), (1, 3), (3, 5)])
        self.assertTrue(all(p.counts == (0, 0, 0, 0) for p in plans))

    def test_invalid_metadata(self):
        """Reject unsupported chunk configurations before any communication."""
        for args in [((7,), 2, 0, 4, 2), ((8,), 2, 2, 4, 2), ((8,), 2, 0, 3, 2),
                     ((8,), 2, 0, 4, 2, "head", 3), ((8,), 2, 0, 4, 2, "sequence", 5),
                     ((8,), 2, 0, 4, 2, "head", True), ((8,), 2, 0, 4, 2, "embedding", 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                chunk_plans(*args)
