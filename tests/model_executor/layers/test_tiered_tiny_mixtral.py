"""Focused tests for the tiny-Mixtral vLLM integration."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_TIERED_WEIGHTS_SRC = (
    Path(__file__).resolve().parents[5] / "packages" / "tiered_weights" / "src"
)
if str(_TIERED_WEIGHTS_SRC) not in sys.path:
    sys.path.insert(0, str(_TIERED_WEIGHTS_SRC))

import vllm.model_executor.models.mixtral as mixtral
from vllm.model_executor.layers.fused_moe.tiered_tiny_mixtral import (
    TieredTinyMixtralRoutedExperts,
    clear_tiny_mixtral_tiered_experts,
    configure_tiny_mixtral_tiered_experts,
)
from vllm.model_executor.models.transformers.tiered_weights_hook import (
    tiered_routed_experts,
)


class FakeProvider:
    def request_experts(self, demands):
        raise AssertionError("provider should not be called during construction")


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
    config = _tiny_mixtral_config()

    configure_tiny_mixtral_tiered_experts(provider)
    try:
        selected = tiered_routed_experts(config, "model.layers.1.block_sparse_moe")
    finally:
        clear_tiny_mixtral_tiered_experts()

    assert selected == (TieredTinyMixtralRoutedExperts, {"provider": provider})


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
