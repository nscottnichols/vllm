"""Focused tests for the DeepSeek-V2-Lite Tiered Weights vLLM path."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SUPERPROJECT_ROOT = Path(__file__).parents[5]
_TIERED_WEIGHTS_SRC = _SUPERPROJECT_ROOT / "packages" / "tiered_weights" / "src"
if str(_TIERED_WEIGHTS_SRC) not in sys.path:
    sys.path.insert(0, str(_TIERED_WEIGHTS_SRC))

from tiered_weights.adapters.deepseek_v2_lite.moe import expert_output  # noqa: E402
from tiered_weights.core.units import WeightDemand, WeightUnit  # noqa: E402
from tiered_weights.residency.manager import ResidencyManager  # noqa: E402
from tiered_weights.runtime.device import DeviceWeightRuntime  # noqa: E402
from tiered_weights.runtime.torch import TorchTensorBackend  # noqa: E402
from tiered_weights.storage.memory import MemoryWeightStore  # noqa: E402

import vllm.model_executor.models.deepseek_v2 as deepseek_v2  # noqa: E402
from vllm.model_executor.layers.fused_moe.tiered_deepseek_v2 import (  # noqa: E402
    DeviceWeightRuntimeDeepseekV2LiteTensorProvider,
    TieredDeepseekV2LiteMoEMethod,
    TieredDeepseekV2LiteRoutedExperts,
    clear_deepseek_v2_lite_tiered_experts,
    configure_deepseek_v2_lite_tiered_experts,
)
from vllm.model_executor.models.transformers.tiered_weights_hook import (  # noqa: E402
    tiered_routed_experts,
)

pytestmark = pytest.mark.skip_global_cleanup


class FakeRoutedExpertsLayer:
    layer_name = "model.layers.1.mlp.experts"
    global_num_experts = 3
    local_num_experts = 3


class FakeDeepseekV2LiteAdapter:
    def __init__(
        self,
        expert_ids: tuple[int, ...] = (1, 7, 9, 11, 13, 15),
        layer_id: int = 1,
    ) -> None:
        self.expert_ids = expert_ids
        self.layer_id = layer_id
        self.units: dict[str, WeightUnit] = {}
        self.payloads: dict[str, bytes] = {}
        self.weights: dict[str, torch.Tensor] = {}

        value = 1
        for expert_id in expert_ids:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                unit_id = (
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}."
                    f"{projection}.weight"
                )
                weight = torch.arange(value, value + 6, dtype=torch.bfloat16)
                if projection == "down_proj":
                    weight = weight.reshape(3, 2)
                else:
                    weight = weight.reshape(2, 3)
                value += 6
                payload = weight.contiguous().view(torch.uint8).numpy().tobytes()
                self.units[unit_id] = WeightUnit(
                    unit_id=unit_id,
                    storage_key=unit_id,
                    offset_bytes=0,
                    length_bytes=len(payload),
                    alignment_bytes=1,
                    storage_layout="safetensors:BF16:row-major",
                    target_layout="safetensors:BF16:row-major",
                    affinity_group="routed_experts",
                    checksum_sha256=hashlib.sha256(payload).hexdigest(),
                )
                self.payloads[unit_id] = payload
                self.weights[unit_id] = weight

    def get_model_name(self) -> str:
        return "deepseek-v2-lite"

    def claim_real_checkpoint(self) -> bool:
        return True

    def build_weight_units(self) -> dict[str, WeightUnit]:
        return dict(self.units)

    def native_quantization_metadata(self) -> dict[str, object]:
        return {"quantized": False, "weight_dtype": "BF16"}

    def native_metadata(self, unit_id: str) -> dict[str, object]:
        unit = self.units[unit_id]
        return {
            "unit_id": unit_id,
            "native_dtype": "BF16",
            "dtype": "BF16",
            "target_dtype": "BF16",
            "shape": list(self.weights[unit_id].shape),
            "tensor_kind": "unquantized_weight",
            "storage_layout": unit.storage_layout,
            "target_layout": unit.target_layout,
            "affinity_group": unit.affinity_group,
            "layer_id": self.layer_id,
            "source_file": unit.storage_key,
            "source_offset_bytes": unit.offset_bytes,
            "source_end_offset_bytes": unit.end_offset_bytes,
            "length_bytes": unit.length_bytes,
            "checksum_sha256": unit.checksum_sha256,
            "quantized": False,
        }

    def build_demands_for_top_k_routing(
        self,
        selected_expert_ids: list[int] | None = None,
        *,
        layer_id: int | None = None,
    ) -> list[WeightDemand]:
        selected = (
            self.expert_ids if selected_expert_ids is None else selected_expert_ids
        )
        resolved_layer_id = self.layer_id if layer_id is None else layer_id
        demands = []
        for expert_id in selected:
            demands.extend(
                WeightDemand(
                    unit_id=unit_id,
                    target_view_id="gpu-vram",
                    earliest_use_step=0,
                    deadline_step=1,
                    priority=10,
                    confidence=1.0,
                    exact=True,
                )
                for unit_id in self.expert_unit_ids(
                    expert_id,
                    layer_id=resolved_layer_id,
                )
            )
        return demands

    def expert_unit_ids(
        self,
        expert_id: int,
        *,
        layer_id: int | None = None,
    ) -> list[str]:
        if expert_id not in self.expert_ids:
            raise ValueError(f"unknown fake expert {expert_id}")
        return [
            f"model.layers.{self.layer_id}.mlp.experts.{expert_id}.{projection}.weight"
            for projection in ("gate_proj", "up_proj", "down_proj")
        ]


def _deepseek_v2_lite_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="deepseek_v2",
        hidden_size=2048,
        intermediate_size=10944,
        moe_intermediate_size=1408,
        n_routed_experts=64,
        n_shared_experts=2,
        num_experts_per_tok=6,
        num_hidden_layers=27,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        torch_dtype=torch.bfloat16,
        hidden_act="silu",
        norm_topk_prob=False,
        n_group=1,
        topk_group=1,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
    )


def _runtime_and_method():
    adapter = FakeDeepseekV2LiteAdapter()
    store = MemoryWeightStore(adapter.payloads)
    manager = ResidencyManager(slot_count=1, slot_capacity_bytes=12)
    runtime = DeviceWeightRuntime(
        adapter=adapter,
        manager=manager,
        store=store,
        tensor_backend=TorchTensorBackend(),
        device="cpu",
        device_budget_bytes=512,
        max_resident_experts=1,
    )
    provider = DeviceWeightRuntimeDeepseekV2LiteTensorProvider(
        runtime=runtime,
        adapter=adapter,
    )
    method = TieredDeepseekV2LiteMoEMethod(SimpleNamespace(), provider=provider)
    return adapter, runtime, provider, method


def test_deepseek_v2_lite_selection_is_exact() -> None:
    provider = object()
    configure_deepseek_v2_lite_tiered_experts(provider)
    try:
        selected = tiered_routed_experts(
            _deepseek_v2_lite_config(),
            "model.layers.1.mlp",
        )
        dense_layer = tiered_routed_experts(
            _deepseek_v2_lite_config(),
            "model.layers.0.mlp",
        )
        wrong_dtype_config = _deepseek_v2_lite_config()
        wrong_dtype_config.torch_dtype = torch.float32
        wrong_dtype = tiered_routed_experts(
            wrong_dtype_config,
            "model.layers.1.mlp",
        )
    finally:
        clear_deepseek_v2_lite_tiered_experts()

    assert selected == (
        TieredDeepseekV2LiteRoutedExperts,
        {"provider": provider},
    )
    assert dense_layer is None
    assert wrong_dtype is None


def test_deepseek_v2_lite_model_passes_tiered_experts_to_factory(monkeypatch) -> None:
    routed_experts_cls = object()
    routed_experts_args = {"provider": object()}
    captured_kwargs = {}

    class FakeGate:
        out_dtype = torch.float32
        e_score_correction_bias = None

        def set_out_dtype(self, dtype):
            self.out_dtype = dtype

    monkeypatch.setattr(
        deepseek_v2,
        "tiered_routed_experts",
        lambda config, prefix: (routed_experts_cls, routed_experts_args),
    )
    monkeypatch.setattr(
        deepseek_v2,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 1)),
    )
    monkeypatch.setattr(
        deepseek_v2,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        deepseek_v2,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        deepseek_v2,
        "rocm_aiter_ops",
        SimpleNamespace(
            is_fused_moe_enabled=lambda: False,
            is_fusion_moe_shared_experts_enabled=lambda: False,
        ),
    )
    monkeypatch.setattr(deepseek_v2, "GateLinear", lambda *args, **kwargs: FakeGate())
    monkeypatch.setattr(deepseek_v2, "DeepseekV2MLP", lambda *args, **kwargs: object())

    def fake_factory(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(deepseek_v2, "FusedMoEFactory", fake_factory)

    deepseek_v2.DeepseekV2MoE(
        config=_deepseek_v2_lite_config(),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0),
            enable_eplb=False,
            use_sequence_parallel_moe=False,
        ),
        prefix="model.layers.1.mlp",
    )

    assert captured_kwargs["routed_experts_cls"] is routed_experts_cls
    assert captured_kwargs["routed_experts_args"] is routed_experts_args


def test_deepseek_v2_lite_loading_skips_only_routed_experts() -> None:
    layer_ids = {1}
    routed_name = "model.layers.1.mlp.experts.7.gate_proj.weight"
    retained_names = (
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.1.input_layernorm.weight",
    )

    assert deepseek_v2._is_tiered_routed_expert_weight(
        routed_name,
        layer_ids,
        64,
    )
    assert not any(
        deepseek_v2._is_tiered_routed_expert_weight(
            name,
            layer_ids,
            64,
        )
        for name in retained_names
    )

    routed_experts = object.__new__(TieredDeepseekV2LiteRoutedExperts)
    loaded = list(
        routed_experts.load_weights(
            [(routed_name, torch.empty(2, 3, dtype=torch.bfloat16))]
        )
    )
    assert loaded == []


def test_deepseek_v2_lite_execution_uses_selected_bf16_experts() -> None:
    adapter, runtime, _, method = _runtime_and_method()
    generator = torch.Generator().manual_seed(1234)
    hidden_states = torch.randn(
        (2, 3),
        generator=generator,
        dtype=torch.bfloat16,
    )
    expert_ids = (1, 7, 9, 11, 13, 15)
    topk_weights = torch.full((2, 6), 1.0 / 6.0, dtype=torch.float32)
    topk_ids = torch.tensor(
        [list(expert_ids), list(reversed(expert_ids))],
        dtype=torch.int32,
    )
    expected = torch.zeros_like(hidden_states)
    for expert_id in expert_ids:
        rows, slots = torch.nonzero(topk_ids == expert_id, as_tuple=True)
        projection_ids = adapter.expert_unit_ids(expert_id, layer_id=1)
        weights = [adapter.weights[unit_id] for unit_id in projection_ids]
        contribution = expert_output(hidden_states[rows], *weights)
        contribution *= topk_weights[rows, slots].to(hidden_states.dtype).unsqueeze(1)
        expected.index_add_(0, rows, contribution)

    result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=object(),
        shared_experts_input=hidden_states.clone(),
    )

    assert torch.equal(result, expected)
    runtime_stats = runtime.stats()
    assert runtime_stats["active_leases"] == 0
    assert runtime_stats["resident_groups"] == 1
    assert runtime_stats["resident_units"] == 3
    assert runtime_stats["loads"] == 18
    assert runtime_stats["max_resident_experts"] == 1


def test_deepseek_v2_lite_ignores_negative_padding_ids() -> None:
    adapter, runtime, _, method = _runtime_and_method()
    hidden_states = torch.randn(
        (2, 3),
        generator=torch.Generator().manual_seed(2345),
        dtype=torch.bfloat16,
    )
    topk_weights = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [[1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]],
        dtype=torch.int32,
    )

    expected = torch.zeros_like(hidden_states)
    projection_ids = adapter.expert_unit_ids(1, layer_id=1)
    weights = [adapter.weights[unit_id] for unit_id in projection_ids]
    contribution = expert_output(hidden_states[0:1], *weights)
    expected[0:1] += contribution * topk_weights[0, 0].to(
        hidden_states.dtype
    ).reshape(1, 1)

    result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=object(),
        shared_experts_input=hidden_states.clone(),
    )

    assert torch.equal(result, expected)
    runtime_stats = runtime.stats()
    assert runtime_stats["active_leases"] == 0
    assert runtime_stats["loads"] == 3
    assert runtime_stats["resident_groups"] == 1


def test_deepseek_v2_lite_all_padding_returns_zero_output() -> None:
    _, runtime, _, method = _runtime_and_method()
    hidden_states = torch.ones((1, 3), dtype=torch.bfloat16)
    topk_weights = torch.zeros((1, 6), dtype=torch.float32)
    topk_ids = torch.full((1, 6), -1, dtype=torch.int32)

    result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=object(),
        shared_experts_input=hidden_states.clone(),
    )

    assert torch.equal(result, torch.zeros_like(hidden_states))
    runtime_stats = runtime.stats()
    assert runtime_stats["active_leases"] == 0
    assert runtime_stats["loads"] == 0
    assert runtime_stats["resident_groups"] == 0


def test_deepseek_v2_lite_lease_releases_on_success_and_failure() -> None:
    adapter, runtime, provider, _ = _runtime_and_method()
    demands = adapter.build_demands_for_top_k_routing([1], layer_id=1)

    with provider.request_experts(demands) as resident_experts:
        assert resident_experts.layer_id == 1
        projections = resident_experts.experts[1]
        assert set(projections) == {"gate_proj", "up_proj", "down_proj"}
        assert all(
            projection.weight.dtype == torch.bfloat16
            and projection.weight.shape
            == (
                (3, 2)
                if projection.projection == "down_proj"
                else (2, 3)
            )
            for projection in projections.values()
        )
        assert runtime.stats()["active_leases"] == 1
    assert runtime.stats()["active_leases"] == 0

    def failing_callback(*args):
        assert runtime.stats()["active_leases"] == 1
        raise RuntimeError("interrupt DeepSeek execution")

    provider.execution_callback = failing_callback
    method = TieredDeepseekV2LiteMoEMethod(SimpleNamespace(), provider=provider)
    hidden_states = torch.ones((1, 3), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="interrupt DeepSeek execution"):
        method.apply(
            layer=FakeRoutedExpertsLayer(),
            x=hidden_states,
            topk_weights=torch.ones((1, 6), dtype=torch.float32),
            topk_ids=torch.tensor([[1, 7, 9, 11, 13, 15]], dtype=torch.int32),
            shared_experts=object(),
            shared_experts_input=hidden_states.clone(),
        )

    runtime_stats = runtime.stats()
    assert runtime_stats["active_leases"] == 0
    assert runtime_stats["scratch_bytes"] == 0
