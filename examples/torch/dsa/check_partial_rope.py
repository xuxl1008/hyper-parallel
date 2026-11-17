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
"""Worker-only partial RoPE comparison; run explicitly on one free NPU."""

import argparse
from functools import partial
import json
from pathlib import Path
import statistics

import torch
import torch_npu

from hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel import _apply_worker_query_rope
from examples.torch.dsa.runtime_experiments import partial_worker_rope


def _measure(function) -> float:
    for _ in range(3):
        function()
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        begin.record()
        for _ in range(5):
            function()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / 5)
    return statistics.median(samples)


def main() -> None:
    """Require exact owner-layout parity, then time the complete adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 63, 1024, 8192, 16384])
    args = parser.parse_args()
    if any(rows <= 0 for rows in args.rows):
        parser.error("rows must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.npu.set_device(0)
    torch.manual_seed(431)
    cases = []
    with torch.no_grad():
        for rows in args.rows:
            query = torch.randn((1, rows, 64, 128), device="npu", dtype=torch.bfloat16)
            angles = torch.randn((1, rows, 32), device="npu", dtype=torch.float32)
            for interleaved in (False, True):
                for dtype in (torch.bfloat16, torch.float32):
                    embeddings = tuple(torch.cat((value, value), -1).to(dtype)
                                       for value in (angles.cos(), angles.sin()))
                    original = query.clone()
                    baseline = partial(_apply_worker_query_rope, query, embeddings, 64, interleaved)
                    candidate = partial(partial_worker_rope, query, embeddings, 64, interleaved)
                    torch.testing.assert_close(candidate(), baseline(), rtol=0, atol=0)
                    torch.testing.assert_close(query, original, rtol=0, atol=0)
                    cases.append({"rows": rows, "interleaved": interleaved, "coefficient_dtype": str(dtype),
                                  "exact": True, "baseline_ms": _measure(baseline), "partial_ms": _measure(candidate)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"completed": True, "torch_npu": torch_npu.__version__, "cases": cases}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(cases), flush=True)


if __name__ == "__main__":
    main()
