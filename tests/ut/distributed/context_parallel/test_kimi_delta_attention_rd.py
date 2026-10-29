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
"""CPU checks for RD configuration, reset plans and summary consumers."""
from dataclasses import FrozenInstanceError
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from hyper_parallel.components.functional.linear_attention_packed import PackedSequenceInfo
from hyper_parallel.distributed.context_parallel import kimi_delta_attention_rd as rd
from hyper_parallel.models.kimi_k3.adapter.distributed import context_parallel as adapter
from tests.common.mark_utils import arg_mark


def _make_protocol(rank, size):
    """Create real executor metadata with an entirely mocked process group."""
    ranks = tuple(range(size))
    with patch.object(rd.dist, "get_backend", return_value="gloo"), \
            patch.object(rd.dist, "get_process_group_ranks", return_value=ranks):
        return rd.KDAStateRecursiveDoubling(None, rank, ranks)


def _load_summary_functional():
    """Load the actual producer helper without importing optional Triton kernels."""
    source = Path(rd.__file__).resolve().parents[2] / "components/functional/kimi_delta_attention.py"
    package = "hyper_parallel.components.functional"
    spec = importlib.util.spec_from_file_location(f"{package}._kda_summary_cpu_regression", source)
    module = importlib.util.module_from_spec(spec)
    backend = SimpleNamespace(**{name: Mock() for name in (
        "apply_kda_state_gradient_summary", "apply_kda_state_summary",
        "kda_state_gradient_summary_from_prepared", "kda_state_summary_forward_from_prepared",
    )})
    with patch.dict(sys.modules, {f"{package}.kimi_delta_attention_state_summary": backend}):
        spec.loader.exec_module(module)
    return module


class TestCachedDoubling(unittest.TestCase):
    """Keep environment gating and invalid-input checks independent of hardware."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_transport_qualification(self):
        """Feature: RD transport selection.

        Description: Vary the runtime versions, device and explicit public override.
        Expectation: Private coalescing is selected only for the validated environment.
        """
        args = ("2.10.0", "2.10.0", "9.1.0-beta.3", "Ascend910B3", True)
        self.assertTrue(rd.select_transport(*args).direct)
        self.assertFalse(rd.select_transport(*args, force_public=True).direct)
        for index, value in ((0, "2.11.0"), (1, "2.9.0"), (2, "9.2.0"), (3, "Ascend950"), (4, False)):
            changed = list(args)
            changed[index] = value
            self.assertFalse(rd.select_transport(*changed).direct)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_inputs_fail_before_exchange(self):
        """Feature: RD input validation.

        Description: Pass incompatible peers, shape, dtype and invocation caches.
        Expectation: Invalid calls fail before any peer exchange.
        """
        with patch.object(rd.dist, "get_backend", return_value="gloo"), \
                patch.object(rd.dist, "get_process_group_ranks", return_value=[0, 2, 4]):
            protocol = rd.KDAStateRecursiveDoubling(None, 2, (0, 2, 4))
            with self.assertRaisesRegex(ValueError, "group order"):
                rd.KDAStateRecursiveDoubling(None, 0, (2, 0, 4))
        state = torch.zeros(1, 2, 128, 128)
        with patch.object(protocol, "exchange") as exchange:
            for matrix in (state.double(), state[..., :64], state.transpose(-1, -2)):
                with self.assertRaises(ValueError):
                    protocol.forward(state, matrix)
            for cache in ((), (state,), [state, state], (state, state.double()),
                          (state, state.transpose(-1, -2))):
                with self.assertRaises(ValueError):
                    protocol.backward(state, cache)
            exchange.assert_not_called()
        with self.assertRaisesRegex(ValueError, "outside"):
            protocol.exchange(state, 3, None)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_adapter_passes_explicit_rd_configuration(self):
        """Feature: Unified KDA state CP configuration.

        Description: Select recursive doubling while retaining older wrapper defaults.
        Expectation: The executor records RD and fixed chunk64; legacy signatures are unchanged.
        """
        mesh = SimpleNamespace(size=lambda: 4)
        with patch.object(adapter, "KimiDeltaAttentionLayerCP") as executor:
            request = adapter.kimi_delta_attention_cp_wrapper(
                nn.Identity(), None, None, mesh, None, backend="triton",
                state_cp_method="recursive_doubling", ulysses_degree=2,
            )
        self.assertEqual(executor.call_args.kwargs["state_cp_method"], "recursive_doubling")
        self.assertEqual(executor.call_args.kwargs["chunk_size"], 64)
        self.assertEqual(executor.call_args.kwargs["ulysses_degree"], 2)
        self.assertNotIn("chunk_size", inspect.signature(adapter.kimi_delta_attention_cp_wrapper).parameters)
        for wrapper in (adapter.kimi_delta_attention_p2p_cp_wrapper, adapter.kimi_delta_attention_ulysses_cp_wrapper):
            parameters = inspect.signature(wrapper).parameters
            self.assertEqual(parameters["chunk_size"].default, 64)
            self.assertNotIn("state_cp_method", parameters)
        self.assertEqual(request.companion_attrs["_hp_kda_cp_config"]["state_cp_method"], "recursive_doubling")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mixed_reset_flags_and_invocation_binding(self):
        """Feature: Reset-aware RD invocation metadata.

        Description: Bind mixed and empty CP5 layouts, then reverse the order of two backward calls.
        Expectation: Producer flags match document connectivity and each immutable call retains its plan.
        """
        offsets = (0, 5, 8, 20)
        expected = ((True, False), (False, False), (True, False), (True, True), (False, False))
        calls = []
        for rank, flags in enumerate(expected):
            protocol = _make_protocol(rank, 5)
            call = protocol.bind_sequence(PackedSequenceInfo(offsets, rank * 4, (rank + 1) * 4))
            self.assertEqual((call.needs_state_summary, call.needs_transition_summary), flags)
            self.assertEqual(call.plan.cache_count(rank) > 0, flags[1])
            calls.append(call)
        call_a = calls[3]
        protocol = call_a.protocol
        call_b = protocol.bind_sequence(PackedSequenceInfo((0, 4, 8, 12, 16, 20), 12, 16))
        self.assertEqual(call_a.plan.receivers, (frozenset({3}), frozenset(), frozenset()))
        self.assertEqual(call_a.plan.active_stages(2), (0,))
        self.assertEqual(call_b.plan.messages_per_direction, 0)
        with patch.object(protocol, "forward", return_value=(None, ())) as forward, \
                patch.object(protocol, "backward", return_value=None) as backward:
            call_a.forward(None, None)
            call_b.forward(None, None)
            call_b.backward(None, ())
            call_a.backward(None, ())
        self.assertEqual([item.args[2] for item in forward.call_args_list], [call_a.plan, call_b.plan])
        self.assertEqual([item.args[2] for item in backward.call_args_list], [call_b.plan, call_a.plan])
        with self.assertRaises(FrozenInstanceError):
            call_a.plan = call_b.plan
        with self.assertRaises(FrozenInstanceError):
            call_a.plan.size = 6

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_empty_plan_omits_packing_and_rounds(self):
        """Feature: RD execution for an empty reset plan.

        Description: Run forward and backward with independent documents on CP1, CP2 and CP5.
        Expectation: No packed tensor or RD round is created, and both state boundaries are zero.
        """
        state = torch.ones(1, 1, 128, 128)
        for size in (1, 2, 5):
            with self.subTest(size=size):
                rank = size // 2
                protocol = _make_protocol(rank, size)
                info = PackedSequenceInfo(tuple(range(0, size * 4 + 1, 4)), rank * 4, (rank + 1) * 4)
                call = protocol.bind_sequence(info)
                with patch.object(rd.torch, "cat") as pack, \
                        patch.object(protocol, "exchange", return_value=None) as exchange:
                    incoming, cache = call.forward(state, state)
                    gradient = call.backward(state, cache)
                pack.assert_not_called()
                self.assertEqual(cache, ())
                self.assertEqual(exchange.call_count, 2)
                for item in exchange.call_args_list:
                    self.assertEqual(item.args[1:], (None, None))
                torch.testing.assert_close(incoming, torch.zeros_like(state), rtol=0, atol=0)
                torch.testing.assert_close(gradient, torch.zeros_like(state), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_early_receive_keeps_latest_state_for_later_sends(self):
        """Feature: RD state lifetime after an early final receive.

        Description: Rank 1 of CP8 receives once and then sends through later packed and narrow rounds.
        Expectation: The final narrow round and exclusive shift send its latest composed state.
        """
        protocol = _make_protocol(1, 8)
        call = protocol.bind_sequence(PackedSequenceInfo((0, 32), 4, 8))
        state = torch.full((1, 1, 128, 128), 2.0)
        matrix = (2 * torch.eye(128)).reshape(1, 1, 128, 128)
        preceding = torch.full_like(state, 3.0)
        received = torch.cat((preceding, torch.zeros_like(matrix)), dim=-1)
        with patch.object(protocol, "exchange", side_effect=(received, None, None, preceding)) as exchange:
            incoming, cache = call.forward(state, matrix)
        self.assertEqual(len(cache), 1)
        self.assertIs(cache[0], matrix)
        self.assertEqual([item.args[1:] for item in exchange.call_args_list],
                         [(2, 0), (3, None), (5, None), (2, 0)])
        for item in exchange.call_args_list[2:]:
            torch.testing.assert_close(item.args[0], torch.full_like(state, 8.0), rtol=0, atol=0)
        torch.testing.assert_close(incoming, preceding, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_reset_stops_transfer_copies_after_last_receive(self):
        """Feature: Reset-aware transfer copy pruning.

        Description: Run CP8 rank 5 with only its first RD receive active.
        Expectation: Only the original M is cached, no next M is copied, and the shift keeps the latest S.
        """
        protocol = _make_protocol(5, 8)
        call = protocol.bind_sequence(PackedSequenceInfo((0, 5, 8, 13, 16, 25, 32), 20, 24))
        state = torch.full((1, 1, 128, 128), 2.0)
        matrix = (2 * torch.eye(128)).reshape(1, 1, 128, 128)
        preceding = torch.full_like(state, 3.0)
        received = torch.cat((preceding, torch.zeros_like(matrix)), dim=-1)
        with patch.object(protocol, "exchange", side_effect=(received, preceding)) as exchange, \
                patch.object(rd, "_finish_round", wraps=rd._finish_round) as finish:
            _, cache = call.forward(state, matrix)
        self.assertEqual(call.plan.active_stages(5), (0,))
        self.assertEqual(len(cache), 1)
        self.assertIs(cache[0], matrix)
        self.assertFalse(finish.call_args.kwargs["save_m"])
        torch.testing.assert_close(exchange.call_args.args[0], torch.full_like(state, 8.0), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_producer_skips_dead_s_and_m_but_retains_head_gradient(self):
        """Feature: Independent forward summary and backward gradient consumers.

        Description: Mock S/M kernels at the first, middle, internally reset and last state ranks.
        Expectation: Only live summaries are produced, while the last rank still sends its head G.
        """
        functional = _load_summary_functional()
        kernel = functional.kda_state_summary_forward_from_prepared
        prepared = torch.ones(1, 4, 1, 128, dtype=torch.bfloat16)
        summary = torch.ones(1, 1, 128, 128, dtype=torch.float32)
        cases = ((0, (0, 20), False), (1, (0, 20), True), (2, (0, 9, 20), False))
        for rank, offsets, needs_matrix in cases:
            with self.subTest(rank=rank, offsets=offsets):
                info = PackedSequenceInfo(offsets, rank * 4, (rank + 1) * 4)
                call = _make_protocol(rank, 5).bind_sequence(info)
                kernel.reset_mock()
                kernel.return_value = (summary, summary if needs_matrix else None)
                state, matrix = functional._prepare_packed_kda_summary(
                    prepared, prepared, prepared, prepared, info, call, info.continues_next, 64)
                kernel.assert_called_once()
                self.assertEqual(kernel.call_args.kwargs["output_transition"], needs_matrix)
                self.assertEqual(kernel.call_args.args[0].shape[1], info.length - info.tail.start)
                self.assertIs(state, summary)
                torch.testing.assert_close(matrix, summary if needs_matrix else torch.zeros_like(summary))
        info = PackedSequenceInfo((0, 20), 16, 20)
        protocol = _make_protocol(4, 5)
        call = protocol.bind_sequence(info)
        kernel.reset_mock()
        state, matrix = functional._prepare_packed_kda_summary(
            prepared, prepared, prepared, prepared, info, call, False, 64)
        kernel.assert_not_called()
        self.assertTrue(info.continues_previous)
        torch.testing.assert_close(state, torch.zeros_like(summary), rtol=0, atol=0)
        torch.testing.assert_close(matrix, torch.zeros_like(summary), rtol=0, atol=0)
        with patch.object(protocol, "exchange", return_value=None) as exchange:
            call.backward(summary, ())
        exchange.assert_called_once_with(summary, 3, None)
