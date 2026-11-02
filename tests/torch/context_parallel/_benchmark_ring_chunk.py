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
"""Opt-in paired measurements for chunked Ring; generated reports stay outside Git."""

import argparse
from contextlib import ExitStack
from datetime import timedelta
from functools import lru_cache, partial
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from threading import Lock
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.distributed.context_parallel.ring_attention import _planned_options, _ring_context, ring_attention
from hyper_parallel.distributed.context_parallel.ring_chunk import ChunkTrace, chunk_plans
from hyper_parallel.distributed.context_parallel.ring_layout import RingAttentionTrace
from tests.torch.context_parallel._test_ring_chunk import VARIANTS, _errors, _iteration, _make_inputs
from tests.torch.context_parallel._benchmark_ring_gated_gqa import _modules, _positions
from tests.torch.context_parallel._benchmark_ring_projection_prepare import _profile

RING = importlib.import_module("hyper_parallel.distributed.context_parallel.ring_attention")
CHUNK = importlib.import_module("hyper_parallel.distributed.context_parallel.ring_chunk")


def _measure(function, iterations):
    dist.barrier()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    wall = perf_counter()
    begin.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return {'ms': begin.elapsed_time(end) / iterations, 'wall_ms': (perf_counter() - wall) * 1000 / iterations,
                'extra_bytes': torch.npu.max_memory_allocated() - before,
                'live_growth': torch.npu.memory_allocated() - before}


def _paired(functions, args, mesh):
    for function in functions.values():
        for _ in range(args.warmup):
            function()
    names = tuple(functions)
    local = {name: [] for name in names}
    for repeat in range(args.repeats):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in order if repeat % 2 == 0 else reversed(order):
            local[name].append(_measure(functions[name], args.iterations))
    workers = [None] * mesh.size()
    dist.all_gather_object(workers, local)
    summary = {}
    for name in names:
        values = [max(worker[name][repeat]['ms'] for worker in workers) for repeat in range(args.repeats)]
        summary[name] = {'median_ms': median(values), 'slowest_samples_ms': values,
                             'peak_extra_bytes': max(row['extra_bytes'] for w in workers for row in w[name]),
                             'max_live_growth': max(row['live_growth'] for w in workers for row in w[name])}
    return {'summary': summary, 'workers': workers}


def _micro(args, mesh):
    ring = _ring_context(mesh)
    result = {'p2p': [], 'fa': []}
    for mib in (4, 8, 16, 32, 64):
        buffers = [torch.empty(mib * 1024 * 1024 // 2, dtype=torch.bfloat16, device="npu") for _ in range(2)]

        def _exchange():
            if ring.rank % 2:
                receive = dist.irecv(buffers[1], ring.previous_rank, group=ring.group)
                send = dist.isend(buffers[0], ring.next_rank, group=ring.group)
            else:
                send = dist.isend(buffers[0], ring.next_rank, group=ring.group)
                receive = dist.irecv(buffers[1], ring.previous_rank, group=ring.group)
            receive.wait()
            send.wait()
            buffers.reverse()

        measured = _paired({"p2p": _exchange}, args, mesh)
        result['p2p'].append({'mib': mib, **measured['summary']['p2p']})
    for qheads, kheads, kv_tokens in ((40, 8, 8192), (20, 4, 8192), (10, 2, 8192),
                                     (5, 1, 8192), (40, 8, 4096), (40, 8, 2048)):
        query, key, value, dout = _make_inputs(8192, ring.rank, heads=qheads, kv_heads=kheads)
        key, value = key[:kv_tokens], value[:kv_tokens]
        options = _planned_options(query, key, "TND", False)
        output, maximum, denominator = torch.ops.npu.npu_fusion_attention(
            query, key, value, scale=128 ** -0.5, **options)[:3]

        def _forward():
            return torch.ops.npu.npu_fusion_attention(query, key, value, scale=128 ** -0.5, **options)

        def _backward():
            return torch.ops.npu.npu_fusion_attention_grad(
                query, key, value, dout, softmax_max=maximum, softmax_sum=denominator,
                attention_in=output, scale_value=128 ** -0.5, **options)

        measured = _paired({'forward': _forward, 'backward': _backward}, args, mesh)
        result['fa'].append({'query_heads': qheads, 'kv_heads': kheads, 'kv_tokens': kv_tokens,
                                 'summary': measured['summary']})
    return result


class _BaselineTrace:
    def __init__(self) -> None:
        """Track the same stage labels without changing the baseline implementation."""
        self.phase = "forward"
        self.step = -1
        self.stages = ChunkTrace()

    def begin(self) -> None:
        """Advance the baseline hop counter without device work."""
        self.step += 1

    def end(self, phase: str, step: int, begin: None) -> None:
        """Implement the baseline trace hook; stages are collected by wrappers.

        Args:
            phase: Completed forward or backward phase.
            step: Completed Ring hop.
            begin: Unused token matching the baseline trace protocol.
        """
        del phase, step, begin


def _baseline_trace(inputs, mesh, lengths):
    trace = _BaselineTrace()

    def _phase_call(phase, function, *args, **kwargs):
        trace.phase, trace.step = phase, -1
        return function(*args, **kwargs)

    def _stage_call(stage, function, *args, **kwargs):
        if stage in ("submit", "wait"):
            stage = ("gradient_" if args[0]._buffers[0].dtype == torch.float32 else "kv_") + stage
        with trace.stages.measure(trace.phase, trace.step, 0, stage):
            return function(*args, **kwargs)

    with trace.stages.measure("step", -1, -1, "total"), ExitStack() as stack:
        for name, phase in (("_planned_forward", "forward"), ("_planned_backward", "backward")):
            stack.enter_context(patch.object(RING, name, partial(_phase_call, phase, getattr(RING, name))))
        for name, stage in (("_forward_blocks", "fa_merge"), ("_backward_blocks", "fa_prepare")):
            stack.enter_context(patch.object(RING, name, partial(_stage_call, stage, getattr(RING, name))))
        for name in ("launch", "wait"):
            original = getattr(RING._RingExchange, name)
            stage = "submit" if name == "launch" else "wait"
            def _wrapped(exchange, *args, function=original, label=stage):
                return _stage_call(label, function, exchange, *args)
            stack.enter_context(patch.object(RING._RingExchange, name, _wrapped))
        output = ring_attention(*inputs[:3], mesh, input_layout="TND", document_lengths=lengths,
                                causal=True, overlap=True, trace=trace)
        torch.autograd.grad(output, inputs[:3], inputs[3])
    return trace.stages.timings()


def _payload(lengths, mesh, variant):
    options = VARIANTS.get(variant, {})
    plans = chunk_plans(lengths, mesh.size(), mesh.get_local_rank(), 40, 8,
                        options.get('chunk_axis', 'head'), options.get('chunks', 1), True,
                        options.get('prune_documents', False))
    rank, size = mesh.get_local_rank(), mesh.size()
    forward = backward_kv = gradients = 0
    calls = 0
    for plan in plans:
        heads = plan.key_heads[1] - plan.key_heads[0]
        for step in range(size):
            count = plan.counts[(rank - step) % size]
            if count:
                gradients += 2 * count * heads * 128 * 4 if size > 1 else 0
                calls += int(size > 1)
                if step + 1 < size:
                    forward += 2 * count * heads * 128 * 2
                    backward_kv += 2 * count * heads * 128 * 2
                    calls += 2
    return {'forward_bytes': forward, 'backward_kv_bytes': backward_kv, 'dkv_bytes': gradients,
                'full_step_mib': (forward + backward_kv + gradients) / 1024 ** 2, 'send_calls': calls}


def _attention(args, mesh, samples):
    results = []
    for sample in samples:
        lengths = tuple(sample['document_lengths'])
        inputs = _make_inputs(sum(lengths) // mesh.size(), mesh.get_local_rank())
        reference = _iteration(inputs, mesh, lengths, "baseline")
        numerical = {name: _errors(_iteration(inputs, mesh, lengths, name), reference) for name in args.variants}
        del reference
        functions = {name: partial(_iteration, inputs, mesh, lengths, name) for name in args.variants}
        measured = _paired(functions, args, mesh)
        traces, payloads = {}, {name: _payload(lengths, mesh, name) for name in args.variants}
        if args.trace:
            for name in args.variants:
                if name == "sync":
                    continue
                rows = []
                for _ in range(3):
                    dist.barrier()
                    if name.startswith("pruned_core"):
                        trace = RingAttentionTrace()
                        _iteration(inputs, mesh, lengths, name, trace=trace)
                        rows.append([{**row, 'stage': 'round', 'chunk': 0} for row in trace.timings()])
                    elif name == "baseline":
                        rows.append(_baseline_trace(inputs, mesh, lengths))
                    else:
                        trace = ChunkTrace()
                        with trace.measure("step", -1, -1, "total"):
                            _iteration(inputs, mesh, lengths, name, trace=trace)
                        rows.append(trace.timings())
                traces[name] = rows
        workers = [None] * mesh.size()
        dist.all_gather_object(workers, {'rank': mesh.get_local_rank(), 'errors': numerical,
                                            'payloads': payloads, 'traces': traces})
        result = {'name': sample['name'], **measured, 'diagnostics': workers}
        results.append(result)
        if mesh.get_local_rank() == 0:
            print(json.dumps({'case': sample['name'], 'summary': measured['summary']}), flush=True)
            _save(args, results)
    return results


@lru_cache(maxsize=1)
def _frozen_reference(directory):
    if directory is None:
        return RING, CHUNK
    root = Path(directory).resolve()
    name = "_ring_before_hops"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                 submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return tuple(importlib.import_module(name + "." + module) for module in ("ring_attention", "ring_chunk"))


def _module_interface(mesh, lengths, variant, module, query, key, value, attention_mask, reference=None, **kwargs):
    del module, attention_mask
    modules = reference if variant.endswith("_before") else (RING, CHUNK)
    variant = variant.removesuffix("_before")
    tensors = tuple(tensor.squeeze(0).transpose(0, 1) for tensor in (query, key, value))
    if variant in ("baseline", "sync", "pruned_core", "pruned_core_sync"):
        output = modules[0].ring_attention(*tensors, mesh, input_layout="TND", document_lengths=lengths,
                                causal=True, overlap=variant in ('baseline', 'pruned_core'), scale=kwargs['scaling'],
                                prune_documents=variant.startswith('pruned_core'))
    else:
        output = modules[1].chunked_ring_attention(*tensors, mesh, document_lengths=lengths,
                                        scale=kwargs['scaling'], **VARIANTS[variant])
    return output.unsqueeze(0), None


def _module_step(module, hidden, dout, positions, interface):
    module.attention_interface = interface
    output = module(hidden, position_embeddings=positions)[0]
    parameters = (hidden, module.linear_qkv.weight, module.o_proj.weight, module.q_norm.weight, module.k_norm.weight)
    return (output, *torch.autograd.grad(output, parameters, dout))


def _host_measure(function, variant, args):
    rows, guard = {}, Lock()

    def _timed(label, original, *inputs, **kwargs):
        begin = perf_counter()
        result = original(*inputs, **kwargs)
        elapsed = (perf_counter() - begin) * 1000
        with guard:
            item = rows.setdefault(label, {"calls": 0, "host_ms": 0.0})
            item["calls"] += 1
            item["host_ms"] += elapsed
        return result

    modules = _frozen_reference(args.reference_root) if variant.endswith("_before") else (RING, CHUNK)
    variant = variant.removesuffix("_before")
    baseline = variant in ("baseline", "sync")
    core_pruned = variant.startswith("pruned_core")
    namespace = modules[0] if baseline or core_pruned else modules[1]
    exchange = modules[0]._RingExchange if baseline else modules[0]._PackedRingExchange
    targets = [(namespace, "_forward_blocks", "forward_FA_merge"),
               (namespace, "_backward_blocks", "backward_FA_prepare"),
               (exchange, "launch", "transport_submit"),
               (exchange, "wait" if baseline else "finish", "transport_wait_or_finish")]
    if core_pruned:
        targets.append((namespace, "_pack_ring_kv", "transport_pack"))
    elif not baseline:
        targets.append((namespace, "_pack", "transport_pack"))
    if not baseline and hasattr(namespace, "_return_ring_gradient"):
        targets.append((namespace, "_return_ring_gradient", "gradient_return_submit"))
    with ExitStack() as stack:
        for target, name, label in targets:
            original = getattr(target, name)

            def _wrapped(*inputs, function=original, stage=label, **kwargs):
                return _timed(stage, function, *inputs, **kwargs)

            stack.enter_context(patch.object(target, name, _wrapped))
        dist.barrier()
        torch.npu.synchronize()
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        wall = perf_counter()
        for _ in range(args.iterations):
            function()
        submission = (perf_counter() - wall) * 1000 / args.iterations
        end.record()
        end.synchronize()
        device = begin.elapsed_time(end) / args.iterations
    for item in rows.values():
        item['calls'] /= args.iterations
        item['host_ms'] /= args.iterations
    return {"submission_ms": submission, "device_ms": device, "stages": rows,
            "unclassified_submission_ms": submission - sum(row['host_ms'] for row in rows.values())}


def _host_diagnostics(functions, args, mesh):
    for repeat in range(3):
        local = {name: _host_measure(function, name, args) for name, function in functions.items()}
        workers = [None] * mesh.size()
        dist.all_gather_object(workers, local)
        yield {"repeat": repeat, "workers": workers}


def _module(args, mesh, samples):
    modules = _modules(args)
    module = modules[0]
    reference_modules = _frozen_reference(args.reference_root)
    del modules
    results = []
    for sample in samples:
        lengths = tuple(sample['document_lengths'])
        local, rank = sum(lengths) // mesh.size(), mesh.get_local_rank()
        generator = torch.Generator().manual_seed(args.seed + rank)
        hidden = torch.randn(1, local, args.hidden, generator=generator).bfloat16().to('npu').requires_grad_()
        dout = torch.randn(hidden.shape, generator=generator).bfloat16().to('npu')
        positions = _positions(lengths, rank, mesh.size(), args.head_dim)
        functions = {name: partial(_module_step, module, hidden, dout, positions,
                                   partial(_module_interface, mesh, lengths, name, reference=reference_modules))
                     for name in args.variants}
        reference = functions['baseline']()
        numerical = {}
        for name, function in functions.items():
            actual = function()
            metrics = []
            for a, b in zip(actual, reference):
                difference = a.detach().float() - b.detach().float()
                value = float(difference.norm() / b.detach().float().norm().clamp_min(1e-12))
                if not bool(torch.isfinite(a).all()) or value > 0.015:
                    raise AssertionError(f"module gradient {name}: actual relative_l2={value}, expected <=0.015")
                metrics.append(value)
            numerical[name] = metrics
        del reference, actual
        measured = _paired(functions, args, mesh)
        workers = [None] * mesh.size()
        dist.all_gather_object(workers, numerical)
        result = {'name': sample['name'], **measured, 'numerical_relative_l2': workers}
        if args.host_trace:
            result['host_diagnostics'] = list(_host_diagnostics(functions, args, mesh))
        results.append(result)
        if args.profile_dir:
            _profile(functions, args.variants, str(Path(args.profile_dir) / sample['name']), rank)
        if rank == 0:
            print(json.dumps({'case': sample['name'], 'summary': measured['summary']}), flush=True)
            _save(args, results)
    return results


def _source_hashes():
    paths = ['hyper_parallel/distributed/context_parallel/ring_chunk.py',
             'hyper_parallel/distributed/context_parallel/ring_attention.py',
             'hyper_parallel/distributed/context_parallel/ring_layout.py',
             'hyper_parallel/distributed/context_parallel/ring_transport.py',
             'tests/torch/context_parallel/_test_ring_chunk.py',
             'tests/torch/context_parallel/_benchmark_ring_gated_gqa.py', __file__]
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def _save(args, results):
    Path(args.output).write_text(json.dumps({'parameters': vars(args), 'results': results,
        'queue': os.environ.get('TASK_QUEUE_ENABLE', 'default'), 'torch': torch.__version__}, indent=2) + '\n',
        encoding='utf-8')


def _reference_metadata(args, parser):
    if not any(name.endswith('_before') for name in args.variants):
        return
    if args.suite != 'module' or not args.reference_root:
        parser.error('frozen reference variants require --suite module and --reference-root')
    args.reference_sha256 = json.loads((Path(args.reference_root) / 'original_sha256.json').read_text())


def main() -> None:
    """Run one hardware experiment at a time; report each repeat's slowest rank."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('micro', 'attention', 'module'), default='attention')
    parser.add_argument('--manifest', default='/tmp/ring_dataset_manifest.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cases', nargs='+', default=['shuffled_000', 'shuffled_005', 'long_000'])
    parser.add_argument('--variants', nargs='+',
                        choices=['baseline', 'sync', *VARIANTS, 'pruned_core_before', 'head2_before', 'seq2_before'],
                        default=['baseline', 'stream1', 'head2', 'head4', 'seq2', 'seq4', 'pruned'])
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--host-trace', action='store_true')
    parser.add_argument('--profile-dir')
    parser.add_argument('--reference-root',
                        help='Local frozen modules imported under _ring_before_hops for paired runs')
    parser.add_argument('--hidden', type=int, default=5120)
    parser.add_argument('--query-heads', type=int, default=40)
    parser.add_argument('--kv-heads', type=int, default=8)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261101)
    args = parser.parse_args()
    args.source_sha256 = _source_hashes()
    _reference_metadata(args, parser)
    if args.suite != "micro":
        args.manifest_sha256 = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if (args.query_heads, args.kv_heads, args.head_dim) != (40, 8, 128):
        parser.error("this experiment fixes the Qwen3-14B head shape at 40/8/128")
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=300))
    try:
        mesh = init_device_mesh('npu', (dist.get_world_size(),), mesh_dim_names=('cp',))
        if args.suite == 'micro':
            results = _micro(args, mesh)
        else:
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            if mesh.size() != manifest['cp']:
                raise ValueError('manifest CP degree must match the world size')
            samples = [sample for sample in manifest['samples'] if sample['name'] in args.cases]
            if {s['name'] for s in samples} != set(args.cases):
                raise ValueError('every requested case must exist in the manifest')
            function = _attention if args.suite == 'attention' else _module
            results = function(args, mesh, samples)
        if mesh.get_local_rank() == 0:
            _save(args, results)
            print(json.dumps({'complete': True, 'output': args.output}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
