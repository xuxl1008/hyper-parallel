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
"""Local-only full-gradient gate for the stateful AscendC prototype."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--install", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 23])
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--forward-only", action="store_true")
    args = p.parse_args()
    root = args.install.resolve()
    sys.path.insert(0, str(root))
    vendor = root / "fla_npu/opp/vendors/fla_npu_transformer"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{vendor}:{vendor}/op_api/lib"
    import torch
    import torch_npu  # noqa: F401
    import fla_npu
    from stateful_kda import LocalKDA, cpu_reference
    torch.npu.set_device(args.device)
    torch.set_num_threads(1)
    records = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        shape = (1, args.heads, args.tokens, 128)
        q, k = [torch.nn.functional.normalize(torch.randn(shape), dim=-1).bfloat16() for _ in range(2)]
        inputs = [q, k, (torch.randn(shape) * .2).bfloat16(),
                  (torch.randn(shape) * .3 - 5).bfloat16(),
                  torch.rand(shape[:3]).bfloat16(), torch.randn(args.heads) * .1,
                  torch.randn(args.heads, 128) * .1, torch.randn(1, args.heads, 128, 128) * .1]
        do = (torch.randn(shape) * .1).bfloat16()
        dht = torch.randn_like(inputs[-1]) * .1
        for mode in ("output", "state", "both"):
            ni = [x.npu().detach().requires_grad_() for x in inputs]
            ci = [x.double().requires_grad_() for x in inputs]
            print("START", seed, mode, flush=True)
            actual = LocalKDA.apply(*ni)
            expected = cpu_reference(*ci)
            metrics = {}
            pairs = list(zip(("output", "ht"), actual, expected))
            if not args.forward_only:
                if mode == "output":
                    actual[0].backward(do.npu())
                    expected[0].backward(do.double())
                elif mode == "state":
                    actual[1].backward(dht.npu())
                    expected[1].backward(dht.double())
                    # The final state is independent of q, so the CPU graph
                    # leaves q.grad absent; the staged kernel returns zeros.
                    if ci[0].grad is None:
                        ci[0].grad = torch.zeros_like(ci[0])
                else:
                    torch.autograd.backward(actual, (do.npu(), dht.npu()))
                    torch.autograd.backward(expected, (do.double(), dht.double()))
                pairs += list(zip(("dq", "dk", "dv", "dg", "db", "dA", "dbias", "dh0"),
                                  (x.grad for x in ni), (x.grad for x in ci)))
            torch.npu.synchronize()
            for name, a, e in pairs:
                assert a is not None, name
                a = a.detach().cpu().double()
                e = e.detach()
                rel = float((a-e).norm() / e.norm().clamp_min(1e-20))
                metrics[name] = {"rel_l2": rel, "max_abs": float((a-e).abs().max())}
                print(name, metrics[name], flush=True)
                assert torch.isfinite(a).all() and rel < .03, (seed, mode, name, metrics[name])
            assert actual[1].dtype == torch.float32
            if not args.forward_only:
                assert ni[-1].grad.dtype == torch.float32
            records.append({"seed": seed, "mode": mode, "metrics": metrics})
            args.output.write_text(json.dumps({"fla_npu": fla_npu.__file__, "records": records}, indent=2))
    print("ALL PASSED", len(records), flush=True)


if __name__ == "__main__":
    main()
