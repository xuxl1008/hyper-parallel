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
"""Full-layer CP validation and benchmark with selected-device occupancy auditing."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from benchmark_stateful_layer import MODES


ROOT = Path(__file__).resolve().parent


def snapshot():
    value = subprocess.check_output(["npu-smi", "info"], text=True)
    table = value.split("Process id", 1)[-1]
    processes = [(int(d), int(p)) for d, p in re.findall(r"\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|", table)]
    return value, processes


def belongs_to(pid, launcher):
    # Torch elastic starts each worker in its own session; SID is not ownership.
    chain = []
    while pid > 1 and pid not in chain:
        chain.append(pid)
        if pid == launcher:
            return True, chain
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except FileNotFoundError:
            return None, chain
        pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
    return False, chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "runs")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["p2p", "allgather", "ulysses"])
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--suffix", default="run")
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--allow-shared-validation", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--validate-only", action="store_true")
    phase.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--compare-modes", nargs="+", choices=MODES, default=[])
    parser.add_argument("--memory-variants", nargs="+", default=[])
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.memory_variants and not (args.validate_only or args.smoke_only):
        raise ValueError("Memory variant comparison requires full-layer numerical validation")
    if args.allow_shared_validation and not (args.validate_only or args.smoke_only):
        raise ValueError("Shared-device execution is only allowed for numerical checks, never timing")
    allow_shared = args.allow_shared_validation
    if len(set(args.devices)) != len(args.devices):
        raise ValueError("Physical devices must be distinct")
    env = os.environ.copy()
    env.update(ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, args.devices)), HCCL_NPU_SOCKET_PORT_RANGE="auto",
               FLA_NPU_INSTALL=str(args.install.resolve()))
    deadline = time.monotonic() + args.wait_seconds
    while True:
        before, processes = snapshot()
        occupied = [(d, p) for d, p in processes if d in args.devices]
        if not occupied or allow_shared:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Selected devices occupied: {occupied}; no timing started")
        print(json.dumps({"waiting": occupied}), flush=True)
        time.sleep(15)
    for tokens in args.tokens:
        for mode in args.modes:
            name = f"stateful_layer_cp{len(args.devices)}_t{tokens}_{mode}_{args.suffix}"
            if (output_dir / f"{name}.audit.json").exists():
                raise FileExistsError(f"Preserve existing run {name}; choose a new suffix")
            before, processes = snapshot()
            if any(d in args.devices for d, _ in processes) and not allow_shared:
                raise RuntimeError("Selected cards became occupied; stopping before next measurement")
            (output_dir / f"{name}.before.txt").write_text(before)
            command = ["bash", str(ROOT / "run_npu.sh"), "-m", "torch.distributed.run", "--standalone",
                       f"--nproc_per_node={len(args.devices)}",
                       str(ROOT / "benchmark_stateful_layer.py"),
                       "--install", str(args.install.resolve()),
                       "--local-tokens", str(tokens), "--heads", str(args.heads),
                       "--output", str(output_dir / f"{name}.json"),
                       "--mode", mode, "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                       "--hidden-size", str(args.hidden_size), "--seed", str(args.seed)]
            if args.checkpoint:
                command.append("--checkpoint")
            if args.validate_only:
                command.append("--validate-only")
            if args.smoke_only:
                command.append("--smoke-only")
            if args.compare_modes:
                command.extend(["--compare-modes", *args.compare_modes])
            if args.memory_variants:
                command.extend(["--memory-variants", *args.memory_variants])
            audit = {"devices": args.devices, "other_device_processes_before": processes,
                     "memory_variant": env.get("KDA_MEMORY_VARIANT", "baseline"),
                     "external_processes_during": [], "owned_process_chains": {},
                     "measurement_type": ("cpu_validation" if args.validate_only else
                                          "finite_or_crossmode" if args.smoke_only else "timing"),
                     "scope": ("shared devices permitted for numerical checks only" if args.allow_shared_validation
                               else "selected devices idle, other devices may be active")}
            print(json.dumps({"starting": name}), flush=True)
            with (output_dir / f"{name}.log").open("w") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                audit["launcher_pid"] = child.pid
                started = time.monotonic()
                while child.poll() is None:
                    text, live = snapshot()
                    for device, pid in live:
                        if device not in args.devices:
                            continue
                        ours, chain = belongs_to(pid, child.pid)
                        if ours is True:
                            audit["owned_process_chains"][str(pid)] = chain
                        elif ours is False:
                            audit["external_processes_during"].append([device, pid])
                    if time.monotonic() - started > 900:
                        # Only terminate the process group created above.
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        raise TimeoutError(name)
                    time.sleep(2)
            after, remaining = snapshot()
            (output_dir / f"{name}.after.txt").write_text(after)
            audit.update(returncode=child.returncode, processes_after=remaining)
            audit["timing_usable"] = (child.returncode == 0 and not audit["external_processes_during"]
                                      and not any(d in args.devices for d, _ in remaining))
            (output_dir / f"{name}.audit.json").write_text(json.dumps(audit, indent=2))
            print(json.dumps({"finished": name, "returncode": child.returncode,
                              "timing_usable": audit["timing_usable"],
                              "owned_workers": len(audit["owned_process_chains"]),
                              "external_workers": len({tuple(x) for x in audit["external_processes_during"]}),
                              "audit": str(output_dir / f"{name}.audit.json")}), flush=True)
            if child.returncode or (not audit["timing_usable"] and not allow_shared):
                raise RuntimeError("Failed or contaminated run; do not use timing")


if __name__ == "__main__":
    main()
