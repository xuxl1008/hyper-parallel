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
"""Construct research protocols using the installed checkout's current RD API."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.distributed.context_parallel import kimi_delta_attention_rd as rd
from hyper_parallel.distributed.context_parallel import kimi_delta_attention_rd_plan as rd_plan

from . import cached_tree, head_owner_rd, head_rd_lifetime


def source_manifest() -> dict:
    """Record source hashes and module names without recording local paths."""
    modules = (rd, rd_plan, cached_tree, head_owner_rd, head_rd_lifetime)
    result = {
        module.__name__: {"sha256": hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()}
        for module in modules
    }
    if dist.get_backend() == "hccl":
        kernel_path = (Path(inspect.getfile(rd)).parents[2] / "components/functional/_triton"
                       / "kimi_delta_attention/rd_pointwise.py")
        result["hyper_parallel.components.functional._triton.kimi_delta_attention.rd_pointwise"] = {
            "sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest()}
    result[__name__] = {"sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return result


def build_protocols(
    group_width: int, force_public: bool, suite: str = "protocols",
    variants: tuple[str, ...] = ("baseline", "lifetime", "direct"),
    selected: tuple[str, ...] = ("flat", "head", "tree"),
) -> tuple[dict[str, Any], dict]:
    """Create every communicator collectively before any measured operation.

    Args:
        group_width: Consecutive sequence ranks sharing summary heads.
        force_public: Request the current RD public batch P2P transport.
        suite: Compare different protocols or the three g2 implementations.
        variants: Ordered g2 variants, beginning with baseline.
        selected: Protocol controls to construct; enables tree at odd CP sizes.

    Returns:
        Executors and a path-free source manifest. Only natural WORLD order is
        qualified here; no DeviceMesh construction is implied.
    """
    rank, size = dist.get_rank(), dist.get_world_size()
    if suite not in ("protocols", "g2"):
        raise ValueError("suite must be protocols or g2")
    if not selected or any(name not in ("flat", "head", "tree") for name in selected):
        raise ValueError("selected must contain flat, head or tree")
    if not variants or variants[0] != "baseline" or len(set(variants)) != len(variants):
        raise ValueError("g2 variants must start with baseline and contain no duplicates")
    classes = {"baseline": head_owner_rd.KDAStateHeadOwnerRD,
               "lifetime": head_rd_lifetime.HeadOwnerRDLifetime,
               "direct": head_rd_lifetime.HeadOwnerRDDirectRestore}
    if any(name not in classes for name in variants):
        raise ValueError("Unknown g2 variant")
    use_head = suite == "g2" or "head" in selected
    if use_head and (isinstance(group_width, bool) or group_width < 2 or size % group_width):
        raise ValueError("Owner width must be >=2 and divide WORLD size")
    flat = rd.KDAStateRecursiveDoubling(dist.group.WORLD, rank, tuple(range(size)), force_public)
    protocols = {}
    if suite == "protocols" and "flat" in selected:
        protocols["flat_RD"] = flat
    warm_groups = [dist.group.WORLD]
    if use_head:
        inner = inner_ranks = outer = None
        for start in range(0, size, group_width):
            peers = tuple(range(start, start + group_width))
            group = dist.new_group(list(peers))
            if rank in peers:
                inner, inner_ranks = group, peers
        for lane in range(group_width):
            peers = tuple(range(lane, size, group_width))
            group = dist.new_group(list(peers))
            if rank in peers:
                outer = rd.KDAStateRecursiveDoubling(group, peers.index(rank), peers, force_public)
        if inner is None or inner_ranks is None or outer is None:
            raise RuntimeError("Every WORLD rank must belong to an inner group and an outer lane")
        if suite == "g2":
            for name in variants:
                protocols[f"head_g{group_width}_RD_{name}"] = classes[name](inner, inner_ranks, outer)
        else:
            protocols[f"head_g{group_width}_RD"] = classes["baseline"](inner, inner_ranks, outer)
        warm_groups.extend((inner, outer.exchange.group))
    if suite == "protocols" and "tree" in selected:
        protocols["cached_tree"] = cached_tree.CachedOrderedTree(dist.group.WORLD, rank, tuple(range(size)))
    for group in warm_groups:
        token = torch.zeros(1, device=flat.exchange.device)
        dist.all_reduce(token, group=group)
    return protocols, source_manifest()
