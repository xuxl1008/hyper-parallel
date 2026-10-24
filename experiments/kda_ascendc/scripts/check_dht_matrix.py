# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local-only A2 Dhu qualification with an independent CPU recurrence."""
import argparse
import json
import os
from pathlib import Path
import sys


def reference(torch, inputs, lengths, batch, heads, gate_kind, exp2, h0, dht):
    q, k, w, do, dv, gate = [x.double() for x in inputs]
    group = heads // q.shape[1]
    nt = sum((n + 63) // 64 for n in lengths) if lengths else (q.shape[2] + 63) // 64
    dh = torch.zeros(batch, nt, heads, 128, 128, dtype=torch.float64)
    dh0 = torch.zeros_like(dht, dtype=torch.float64) if h0 else None
    dv2 = torch.zeros_like(dv)
    ranges = [(i, 0, q.shape[2]) for i in range(batch)]
    if lengths:
        start = 0
        ranges = []
        for length in lengths:
            ranges.append((0, start, start + length))
            start += length
    chunk_base = 0
    for seq, (b, begin, end) in enumerate(ranges):
        carry = dht[seq].double().clone()
        chunks = list(range(begin, end, 64))
        for ci in reversed(range(len(chunks))):
            start, stop = chunks[ci], min(chunks[ci] + 64, end)
            dh[b, chunk_base + ci] = carry
            for h in range(heads):
                qs, ks = q[b, h // group, start:stop], k[b, h // group, start:stop]
                update = ks @ carry[h]
                if gate_kind == "g":
                    gs = gate[b, h, start:stop]
                    exponential = torch.exp2 if exp2 else torch.exp
                    update *= exponential(gs[-1] - gs)[:, None]
                    next_carry = exponential(gs[-1]) * carry[h]
                    qs = qs * exponential(gs)[:, None]
                else:
                    next_carry = torch.exp2(gate[b, h, stop - 1])[:, None] * carry[h]
                update += dv[b, h, start:stop]
                dv2[b, h, start:stop] = update
                carry[h] = next_carry + 0.125 * qs.T @ do[b, h, start:stop] - w[b, h, start:stop].T @ update
        if dh0 is not None:
            dh0[seq] = carry
        if lengths:
            chunk_base += len(chunks)
    return dh, dh0, dv2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--install", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--stress", action="store_true", help="96 heads and the >=32-chunk synchronization path")
    p.add_argument("--fp32-state", action="store_true")
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()
    root = args.install.resolve()
    sys.path.insert(0, str(root))
    vendor = root / "fla_npu/opp/vendors/fla_npu_transformer"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{vendor}:{vendor}/op_api/lib"
    import torch
    import torch_npu  # noqa: F401
    import fla_npu
    from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd_dhu
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    assert str(root) in fla_npu.__file__, fla_npu.__file__
    old = torch.load(args.baseline, weights_only=True) if args.baseline else {}
    outputs, records = {}, []
    # Cover both gate conventions, GVA, odd heads, tail chunks and empty segments.
    cases = [(1, 1, 1, 128, None, "g", False),
             (2, 3, 3, 65, None, "g", True),
             (1, 2, 4, 129, None, "g", False),
             (1, 3, 3, 128, None, "gK", True),
             (1, 2, 4, 66, [1, 0, 65], "g", False),
             (1, 3, 3, 66, [0, 1, 65, 0], "gK", True)]
    if args.stress:
        cases = [(1, 96, 96, 128, None, "gK", True),
                 (1, 3, 3, 2048, None, "g", False)]
    for dtype in ((torch.bfloat16,) if args.stress else (torch.bfloat16, torch.float16)):
        for seed in ((7,) if args.stress else (7, 23)):
            for case, (b, hk, hv, t, lengths, kind, exp2) in enumerate(cases):
                torch.manual_seed(seed)
                ns = len(lengths) if lengths else b
                shapes = [(b, hk, t, 128)] * 2 + [(b, hv, t, 128)] * 3
                inputs = [(torch.randn(s) * 0.03).to(dtype) for s in shapes]
                gs = (b, hv, t) if kind == "g" else (b, hv, t, 128)
                inputs.append(-torch.rand(gs) * 0.2)
                state_dtype = torch.float32 if args.fp32_state else dtype
                h0 = torch.randn(ns, hv, 128, 128).to(state_dtype)
                boundary = (torch.randn_like(h0.float()) * 0.1).to(state_dtype)
                ni = [x.npu() for x in inputs]
                extra = {kind: ni[-1], "use_exp2": exp2}
                if lengths:
                    cu, chunks = [0], []
                    for seq, length in enumerate(lengths):
                        cu.append(cu[-1] + length)
                        chunks.extend(j for ci in range((length + 63) // 64) for j in (seq, ci))
                    extra.update(cu_seqlens=cu, chunk_indices=chunks)
                for mode in ("none", "zero", "nonzero", "no_h0"):
                    if args.fp32_state and mode == "no_h0":
                        continue  # FP32 prototype requires an explicit h0 descriptor.
                    if args.baseline_only and mode not in ("none", "zero"):
                        continue
                    key = f"{dtype}_{seed}_{case}_{mode}"
                    dht = None if mode == "none" else (torch.zeros_like(boundary) if mode == "zero" else boundary)
                    print("START", key, flush=True)
                    actual = chunk_gated_delta_rule_bwd_dhu(
                        *ni[:5], 0.125, 64, h0=h0.npu() if mode != "no_h0" else None,
                        dht=dht.npu() if dht is not None else None, **extra)
                    torch.npu.synchronize()
                    actual = tuple(x.cpu() if x is not None else None for x in actual)
                    if mode != "no_h0":
                        assert actual[1].dtype == state_dtype
                    expected = reference(torch, inputs, lengths, b, hv, kind, exp2,
                                         mode != "no_h0", dht if dht is not None else torch.zeros_like(boundary))
                    metrics = {}
                    for name, value, ref in zip(("dh", "dh0", "dv2"), actual, expected):
                        if ref is None:
                            assert value is None
                            continue
                        diff = value.double() - ref
                        rel = float(diff.norm() / ref.norm().clamp_min(1e-20))
                        metrics[name] = {"rel_l2": rel, "max_abs": float(diff.abs().max())}
                        if not torch.isfinite(value).all() or rel >= 0.02:
                            torch.save({"actual": actual, "expected": expected, "boundary": boundary,
                                        "inputs": inputs, "key": key}, args.output.with_suffix(".failure.pt"))
                            if name == "dh0":
                                print("FAIL_PER_SEQUENCE", diff.abs().flatten(1).max(1).values.tolist(), flush=True)
                        # Fixed pre-run tolerances, including the BF16 GEMM round trips.
                        assert torch.isfinite(value).all() and rel < 0.02, (key, name, metrics[name])
                        torch.testing.assert_close(value.double(), ref, rtol=0.03, atol=0.001)
                    if mode == "zero":
                        for a, v in zip(actual, outputs[key.replace("_zero", "_none")]):
                            if a is not None:
                                torch.testing.assert_close(a, v, rtol=0, atol=0)
                    if mode in ("nonzero", "no_h0"):
                        zero = outputs[key.rsplit("_", 1)[0] + "_none"] if mode == "nonzero" else outputs[key.replace("_no_h0", "_none")]
                        assert (actual[0].float() - zero[0].float()).norm() > 0.01
                        if lengths and mode == "nonzero":
                            for seq, length in enumerate(lengths):
                                if length == 0:
                                    torch.testing.assert_close(actual[1][seq], boundary[seq], rtol=0, atol=0)
                    if key in old:
                        for a, v in zip(actual, old[key]):
                            if a is not None:
                                torch.testing.assert_close(a, v, rtol=0, atol=0)
                    outputs[key] = actual
                    records.append({"case": key, "metrics": metrics})
                    print("PASS", key, metrics, flush=True)
    torch.save(outputs, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps({"install": str(root), "records": records}, indent=2))
    print("ALL PASSED", len(records), flush=True)


if __name__ == "__main__":
    main()
