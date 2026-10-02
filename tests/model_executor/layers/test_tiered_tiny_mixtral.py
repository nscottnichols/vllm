"""Focused tests for the tiny-Mixtral vLLM integration."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_TIERED_WEIGHTS_SRC = (
    Path(__file__).resolve().parents[5] / "packages" / "tiered_weights" / "src"
)
if str(_TIERED_WEIGHTS_SRC) not in sys.path:
    sys.path.insert(0, str(_TIERED_WEIGHTS_SRC))

import vllm.model_executor.models.mixtral as mixtral
from tiered_weights.adapters.tiny_mixtral import TinyMixtralAdapter
from vllm.model_executor.layers.fused_moe.tiered_tiny_mixtral import (
    TieredTinyMixtralMoEMethod,
    TieredTinyMixtralRoutedExperts,
    TinyMixtralResidentExperts,
    clear_tiny_mixtral_tiered_experts,
    configure_tiny_mixtral_tiered_experts,
)
from vllm.model_executor.models.transformers.tiered_weights_hook import (
    tiered_routed_experts,
)


class FakeProvider:
    def request_experts(self, demands):
        raise AssertionError("provider should not be called during construction")


class RecordingProvider:
    def __init__(self):
        self.prefetch_calls = []
        self.request_calls = []

    def prefetch_experts(self, demands, *, depth):
        self.prefetch_calls.append((depth, tuple(demand.unit_id for demand in demands)))

    def request_experts(self, demands):
        demands = tuple(demands)
        self.request_calls.append(tuple(demand.unit_id for demand in demands))
        return TinyMixtralResidentExperts({0: {}}, lambda: None)

    def execution_callback(self, hidden_states, topk_weights, topk_ids, experts):
        return hidden_states * 2


def _tiny_mixtral_config():
    return SimpleNamespace(
        model_type="mixtral",
        hidden_size=1024,
        intermediate_size=3584,
        num_local_experts=8,
        num_experts_per_tok=2,
    )


def test_tiered_hook_supports_mixtral_block_sparse_moe_prefix():
    provider = FakeProvider()
    adapter = TinyMixtralAdapter()
    config = _tiny_mixtral_config()

    configure_tiny_mixtral_tiered_experts(provider, adapter=adapter)
    try:
        selected = tiered_routed_experts(config, "model.layers.1.block_sparse_moe")
    finally:
        clear_tiny_mixtral_tiered_experts()

    assert selected == (
        TieredTinyMixtralRoutedExperts,
        {"provider": provider, "adapter": adapter},
    )


def test_method_rejects_invalid_prefetch_depth():
    with pytest.raises(ValueError, match="prefetch_depth"):
        TieredTinyMixtralMoEMethod(
            SimpleNamespace(),
            provider=FakeProvider(),
            prefetch_depth=5,
        )


def test_mixtral_moe_passes_tiered_experts_to_factory(monkeypatch):
    provider = FakeProvider()
    routed_experts_cls = object()
    routed_experts_args = {"provider": provider}
    captured_kwargs = {}

    monkeypatch.setattr(
        mixtral,
        "tiered_routed_experts",
        lambda config, prefix: (routed_experts_cls, routed_experts_args),
    )
    monkeypatch.setattr(
        mixtral,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(hf_config=_tiny_mixtral_config()),
            parallel_config=SimpleNamespace(
                eplb_config=SimpleNamespace(num_redundant_experts=0)
            ),
        ),
    )
    monkeypatch.setattr(
        mixtral,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 1)),
    )
    monkeypatch.setattr(mixtral, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mixtral, "ReplicatedLinear", lambda *args, **kwargs: object())

    def fake_factory(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(mixtral, "FusedMoEFactory", fake_factory)

    mixtral.MixtralMoE(
        num_experts=8,
        top_k=2,
        hidden_size=1024,
        intermediate_size=3584,
        prefix="model.layers.0.block_sparse_moe",
    )

    assert captured_kwargs["routed_experts_cls"] is routed_experts_cls
    assert captured_kwargs["routed_experts_args"] is routed_experts_args


def test_method_prefetches_exact_selection_without_changing_output():
    def apply(provider, *, prefetch_depth=None):
        method = TieredTinyMixtralMoEMethod(
            SimpleNamespace(),
            provider=provider,
            prefetch_depth=prefetch_depth,
        )
        layer = SimpleNamespace(layer_name="model.layers.1.block_sparse_moe")
        hidden_states = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
        topk_weights = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        topk_ids = torch.tensor([[0, 1]], dtype=torch.int64)
        return method.apply(layer, hidden_states, topk_weights, topk_ids, None, None)

    disabled_provider = RecordingProvider()
    enabled_provider = RecordingProvider()
    enabled_provider.prefetch_depth = 3

    disabled_output = apply(disabled_provider, prefetch_depth=0)
    enabled_output = apply(enabled_provider)

    assert torch.equal(disabled_output, enabled_output)
    assert disabled_output.tolist() == [[2.0, 6.0]]
    assert disabled_provider.prefetch_calls == []
    expected_demand_ids = tuple(
        f"tiny-mixtral-layer-1-expert-{expert_id}-{projection}"
        for expert_id in (0, 1)
        for projection in ("gate_proj", "up_proj", "down_proj")
    )
    assert enabled_provider.prefetch_calls == [(3, expected_demand_ids)]
    assert enabled_provider.request_calls == [expected_demand_ids]


def test_method_treats_provider_prefetch_failure_as_advisory():
    class FailingPrefetchProvider(RecordingProvider):
        def prefetch_experts(self, demands, *, depth):
            raise RuntimeError("synthetic prefetch failure")

    provider = FailingPrefetchProvider()
    method = TieredTinyMixtralMoEMethod(
        SimpleNamespace(),
        provider=provider,
        prefetch_depth=2,
    )
    layer = SimpleNamespace(layer_name="model.layers.1.block_sparse_moe")
    hidden_states = torch.tensor([[4.0]], dtype=torch.float32)
    topk_weights = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int64)

    output = method.apply(layer, hidden_states, topk_weights, topk_ids, None, None)

    assert output.tolist() == [[8.0]]
    assert provider.request_calls == [
        tuple(
            f"tiny-mixtral-layer-1-expert-{expert_id}-{projection}"
            for expert_id in (0, 1)
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
    ]
