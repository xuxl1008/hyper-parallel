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
"""Explicit, process-local ablations; baseline remains the default."""
import hashlib
import os
from pathlib import Path


VARIANTS = ("baseline", "trim", "trim_norm", "trim_norm_bwd2", "trim_norm_bwd4")


def variant():
    value = os.environ.get("KDA_MEMORY_VARIANT", "baseline")
    if value not in VARIANTS:
        raise ValueError(f"Unknown KDA_MEMORY_VARIANT: {value}")
    return value


def retained_prepared(prepared):
    # Keep tuple positions stable: only forward consumes u and scaled qg.
    values = list(prepared[:8])
    if variant() != "baseline":
        values[4] = values[7] = None
    return tuple(values)


def backward_blocks():
    value = variant()
    return int(value[-1]) if "_bwd" in value else 1


def metadata():
    root = Path(__file__).parent
    names = ("memory_options.py", "fused_output_norm.py")
    return {"memory_variant": variant(),
            "memory_optimization_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                           for name in names}}
