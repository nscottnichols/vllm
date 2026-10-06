# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for DeepSeek MTP routed-expert weight loading."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.models.deepseek_mtp as deepseek_mtp
from vllm.model_executor.layers.fused_moe.tiered_glm import (
    clear_glm_53_tiered_experts,
    configure_glm_53_tiered_experts,
    glm_53_tiered_experts,
)
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP

pytestmark = pytest.mark.cpu_test


class FakeParameter:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[tuple[torch.Tensor, tuple, dict]] = []
        self.weight_loader = _record_weight_loader


def _record_weight_loader(
    parameter: FakeParameter,
    loaded_weight: torch.Tensor,
    *args: object,
    **kwargs: object,
) -> bool:
    parameter.calls.append((loaded_weight, args, kwargs))
    return bool(kwargs.get("return_success", False))


class FakeMTPModel:
    def __init__(self, use_tiered_routed_experts: bool):
        self.layers = {
            "78": SimpleNamespace(
                mtp_block=SimpleNamespace(
                    mlp=SimpleNamespace(
                        use_tiered_routed_experts=use_tiered_routed_experts,
                    ),
                ),
            ),
        }
        self.mtp_start_layer_idx = 78
        self.num_mtp_layers = 1


class FakeDeepSeekMTP:
    def __init__(
        self,
        config: SimpleNamespace,
        use_tiered_routed_experts: bool,
        parameter_names: set[str],
    ):
        self.config = config
        self.model = FakeMTPModel(use_tiered_routed_experts)
        self._parameters = {name: FakeParameter(name) for name in parameter_names}

    def named_parameters(self):
        return iter(self._parameters.items())

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        return DeepSeekMTP._rewrite_spec_layer_name(self, spec_layer, name)


def _glm_mtp_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="deepseek_mtp",
        num_hidden_layers=78,
        num_nextn_predict_layers=1,
        n_routed_experts=256,
        num_experts_per_tok=8,
        hidden_size=6144,
        moe_intermediate_size=2048,
        n_shared_experts=1,
        n_group=1,
    )


def _normal_deepseek_mtp_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="deepseek_mtp",
        num_hidden_layers=78,
        num_nextn_predict_layers=1,
        n_routed_experts=64,
        n_shared_experts=1,
        n_group=1,
    )


def _non_expert_parameter_names() -> set[str]:
    return {
        "model.layers.78.mtp_block.self_attn.q_proj.weight",
        "model.layers.78.mtp_block.post_attention_layernorm.weight",
        "model.layers.78.mtp_block.mlp.gate.weight",
        "model.layers.78.mtp_block.mlp.shared_experts.gate_up_proj.weight",
    }


def _patch_loader_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deepseek_mtp.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: False,
    )
    monkeypatch.setattr(deepseek_mtp, "get_pp_missing_layer_names", lambda _: [])
    monkeypatch.setattr(
        deepseek_mtp,
        "_try_load_fp8_indexer_wk",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        deepseek_mtp,
        "is_mtp_completeness_check_enabled",
        lambda: False,
    )


def _weights() -> list[tuple[str, torch.Tensor]]:
    return [
        (
            "model.layers.78.self_attn.q_proj.weight",
            torch.tensor([1.0]),
        ),
        (
            "model.layers.78.post_attention_layernorm.weight",
            torch.tensor([2.0]),
        ),
        (
            "model.layers.78.mlp.gate.weight",
            torch.tensor([3.0]),
        ),
        (
            "model.layers.78.mlp.shared_experts.gate_proj.weight",
            torch.tensor([4.0]),
        ),
        (
            "model.layers.78.mlp.experts.3.gate_proj.weight",
            torch.tensor([5.0]),
        ),
        (
            "model.layers.78.mlp.experts.3.gate_proj.weight_scale_inv",
            torch.tensor([6.0]),
        ),
        (
            "model.layers.78.mlp.experts.3.up_proj.weight",
            torch.tensor([7.0]),
        ),
        (
            "model.layers.78.mlp.experts.3.down_proj.weight",
            torch.tensor([8.0]),
        ),
        (
            "model.layers.78.mlp.experts.3.down_proj.weight_scale_inv",
            torch.tensor([9.0]),
        ),
    ]


def test_glm_mtp_tiered_dispatch_skips_layer_78_routed_expert_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_loader_helpers(monkeypatch)
    clear_glm_53_tiered_experts()
    configure_glm_53_tiered_experts(SimpleNamespace())
    try:
        config = _glm_mtp_config()
        assert glm_53_tiered_experts(config, "model.layers.78.mlp") is not None
        model = FakeDeepSeekMTP(
            config=config,
            use_tiered_routed_experts=True,
            parameter_names=_non_expert_parameter_names(),
        )

        loaded = DeepSeekMTP.load_weights(model, _weights())

        expert_parameters = {
            name: parameter
            for name, parameter in model._parameters.items()
            if ".mlp.experts." in name
        }
        assert not expert_parameters
        assert all(
            ".mlp.experts." not in name for name, parameter in model._parameters.items()
        )
        assert loaded == _non_expert_parameter_names()
        assert {
            name: parameter.calls[0][0].item()
            for name, parameter in model._parameters.items()
        } == {
            "model.layers.78.mtp_block.self_attn.q_proj.weight": 1.0,
            "model.layers.78.mtp_block.post_attention_layernorm.weight": 2.0,
            "model.layers.78.mtp_block.mlp.gate.weight": 3.0,
            "model.layers.78.mtp_block.mlp.shared_experts.gate_up_proj.weight": 4.0,
        }
    finally:
        clear_glm_53_tiered_experts()


def test_deepseek_mtp_eager_loading_maps_routed_expert_weights_and_scales(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_loader_helpers(monkeypatch)
    config = _normal_deepseek_mtp_config()
    parameter_names = _non_expert_parameter_names() | {
        "model.layers.78.mtp_block.mlp.experts.routed_experts.w13_weight",
        "model.layers.78.mtp_block.mlp.experts.routed_experts.w13_weight_scale_inv",
        "model.layers.78.mtp_block.mlp.experts.routed_experts.w2_weight",
        "model.layers.78.mtp_block.mlp.experts.routed_experts.w2_weight_scale_inv",
    }
    model = FakeDeepSeekMTP(
        config=config,
        use_tiered_routed_experts=False,
        parameter_names=parameter_names,
    )

    loaded = DeepSeekMTP.load_weights(model, _weights())

    expert_calls = {
        name: parameter.calls
        for name, parameter in model._parameters.items()
        if ".mlp.experts." in name
    }
    assert (
        len(
            expert_calls[
                "model.layers.78.mtp_block.mlp.experts.routed_experts.w13_weight"
            ]
        )
        == 2
    )
    assert (
        len(
            expert_calls[
                "model.layers.78.mtp_block.mlp.experts."
                "routed_experts.w13_weight_scale_inv"
            ]
        )
        == 1
    )
    assert (
        len(
            expert_calls[
                "model.layers.78.mtp_block.mlp.experts.routed_experts.w2_weight"
            ]
        )
        == 1
    )
    assert (
        len(
            expert_calls[
                "model.layers.78.mtp_block.mlp.experts."
                "routed_experts.w2_weight_scale_inv"
            ]
        )
        == 1
    )
    assert loaded == parameter_names
    w13_calls = expert_calls[
        "model.layers.78.mtp_block.mlp.experts.routed_experts.w13_weight"
    ]
    assert [call[2]["shard_id"] for call in w13_calls] == ["w1", "w3"]
    assert all(call[2]["expert_id"] == 3 for call in w13_calls)
