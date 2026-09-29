# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for the GLM 5.3 Tiered Weights MoE path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

_SUPERPROJECT_ROOT = Path(__file__).parents[5]
_TIERED_WEIGHTS_SRC = _SUPERPROJECT_ROOT / "packages" / "tiered_weights" / "src"
if str(_TIERED_WEIGHTS_SRC) not in sys.path:
    sys.path.insert(0, str(_TIERED_WEIGHTS_SRC))

from tiered_weights.adapters.glm_5_3 import Glm53Adapter  # noqa: E402
from tiered_weights.adapters.glm_5_3.manifest import native_manifest  # noqa: E402
from tiered_weights.residency.manager import ResidencyManager  # noqa: E402
from tiered_weights.runtime import (  # noqa: E402
    DeviceRuntimeBackpressureError,
    DeviceWeightRuntime,
    TensorSpec,
)
from tiered_weights.storage.memory import MemoryWeightStore  # noqa: E402

from vllm.model_executor.layers.fused_moe.tiered_glm import (  # noqa: E402
    DeviceWeightRuntimeGlmTensorProvider,
    EagerSelectedExpertsGlmTensorProvider,
    LazyGlm53TensorProvider,
    TieredGlm53MoEMethod,
    TieredGlm53RoutedExperts,
    clear_glm_53_tiered_experts,
    configure_glm_53_tiered_experts,
    glm_53_tiered_experts,
)
from vllm.model_executor.models.transformers.tiered_weights_hook import (  # noqa: E402
    TieredWeightsResidencyHook,
)

pytestmark = pytest.mark.skip_global_cleanup


class FakeRoutedExpertsLayer:
    layer_name = "model.layers.3.mlp.experts"


def tiny_glm_manifest(
    expert_count: int = 10,
    layer_ids: tuple[int, ...] = (3,),
) -> dict[str, object]:
    manifest = native_manifest(layer_ids, expert_count=expert_count)
    offset_bytes = 0
    for record in manifest["records"]:
        if record["tensor_kind"] == "quantized_weight":
            if record["unit_id"].endswith(".down_proj.weight"):
                record["shape"] = [3, 2]
            else:
                record["shape"] = [2, 3]
            record["length_bytes"] = 6
        else:
            record["shape"] = [1, 1]
            record["length_bytes"] = 4
        record["offset_bytes"] = offset_bytes
        offset_bytes += (record["length_bytes"] + 7) // 8 * 8
    return manifest


def manifest_payload(manifest: dict[str, object]) -> bytes:
    payload = bytearray(
        max(
            record["offset_bytes"] + record["length_bytes"]
            for record in manifest["records"]
        )
    )
    for record in manifest["records"]:
        offset = record["offset_bytes"]
        length = record["length_bytes"]
        if record["dtype"] == "F8_E4M3":
            for index in range(length):
                payload[offset + index] = (index % 8) + 1
        else:
            scale_values = (0.25, 0.5, 0.75, 1.0)
            payload[offset : offset + length] = (
                torch.tensor(scale_values[:length], dtype=torch.float32)
                .numpy()
                .tobytes()
            )
    return bytes(payload)


def payload_tensor_factory(records: dict[str, dict[str, object]], payload: bytes):
    created_tensors: dict[str, torch.Tensor] = {}

    def create(unit_id: str) -> torch.Tensor:
        record = records[unit_id]
        raw = payload[
            record["offset_bytes"] : record["offset_bytes"] + record["length_bytes"]
        ]
        if record["dtype"] == "F8_E4M3":
            tensor = (
                torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                .clone()
                .reshape(record["shape"])
                .view(torch.float8_e4m3fn)
            )
        else:
            tensor = (
                torch.frombuffer(bytearray(raw), dtype=torch.float32)
                .clone()
                .reshape(record["shape"])
            )
        created_tensors[unit_id] = tensor
        return tensor

    return create, created_tensors


def all_resident_experts(
    adapter: Glm53Adapter,
    records: dict[str, dict[str, object]],
    payload: bytes,
    expert_ids: tuple[int, ...],
    layer_id: int = 3,
) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    experts = {}
    for expert_id in expert_ids:
        projections = {}
        for unit_id in adapter.expert_unit_ids(expert_id, layer_id=layer_id):
            record = records[unit_id]
            raw = payload[
                record["offset_bytes"] : record["offset_bytes"] + record["length_bytes"]
            ]
            if record["dtype"] == "F8_E4M3":
                tensor = (
                    torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                    .clone()
                    .reshape(record["shape"])
                    .view(torch.float8_e4m3fn)
                )
            else:
                tensor = (
                    torch.frombuffer(bytearray(raw), dtype=torch.float32)
                    .clone()
                    .reshape(record["shape"])
                )
            projections[unit_id] = tensor
        experts[expert_id] = projections
    return experts


def dequant_projection(projections: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    dequantized = {}
    for weight_unit_id, weight in projections.items():
        if not weight_unit_id.endswith(".weight"):
            continue
        scale = projections[f"{weight_unit_id}_scale_inv"]
        block_scale = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[
            : weight.shape[0], : weight.shape[1]
        ]
        projection_name = weight_unit_id.rsplit(".", maxsplit=2)[-2]
        dequantized[projection_name] = weight.to(torch.float32) * block_scale
    return dequantized


def all_resident_reference(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts: dict[int, dict[str, torch.Tensor]],
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    selected_experts = {
        expert_id for routing_row in topk_ids.tolist() for expert_id in routing_row
    }
    for expert_id in sorted(selected_experts):
        rows, slots = torch.nonzero(topk_ids == expert_id, as_tuple=True)
        weights = dequant_projection(experts[expert_id])
        expert_input = hidden_states[rows].to(torch.float32)
        gate_output = expert_input @ weights["gate_proj"].T
        up_output = expert_input @ weights["up_proj"].T
        intermediate = F.silu(gate_output) * up_output
        expert_output = intermediate @ weights["down_proj"].T
        routing_weights = topk_weights[rows, slots].to(torch.float32)
        output.index_add_(
            0,
            rows,
            expert_output * routing_weights.unsqueeze(1),
        )
    return output.to(hidden_states.dtype)


class RecordingMemoryWeightStore(MemoryWeightStore):
    def __init__(self, objects: dict[str, bytes]) -> None:
        super().__init__(objects)
        self.reads: list[tuple[str, int, int]] = []

    def read(self, storage_key: str, offset_bytes: int, length_bytes: int) -> bytes:
        self.reads.append((storage_key, offset_bytes, length_bytes))
        return super().read(storage_key, offset_bytes, length_bytes)


class NativeTensorBackend:
    def dtype_byte_size(self, native_dtype: str) -> int:
        return {"F8_E4M3": 1, "F32": 4}[native_dtype]

    def create_tensor(
        self, payload: bytes | bytearray | memoryview, spec: TensorSpec, device: str
    ) -> torch.Tensor:
        if spec.native_dtype == "F8_E4M3":
            tensor = (
                torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                .clone()
                .reshape(spec.shape)
                .view(torch.float8_e4m3fn)
            )
        else:
            tensor = (
                torch.frombuffer(bytearray(payload), dtype=torch.float32)
                .clone()
                .reshape(spec.shape)
            )
        self.validate_tensor(tensor, spec, device)
        return tensor

    def validate_tensor(
        self, tensor: torch.Tensor, spec: TensorSpec, device: str
    ) -> None:
        assert tensor.device.type == device
        assert tuple(tensor.shape) == tuple(spec.shape)
        assert (
            tensor.dtype == torch.float8_e4m3fn
            if spec.native_dtype == "F8_E4M3"
            else torch.float32
        )


def test_router_demands_group_by_token_and_provider_materializes_only_selection():
    manifest = tiny_glm_manifest(layer_ids=(3, 77))
    payload = manifest_payload(manifest)
    adapter = Glm53Adapter(manifest)
    records = {record["unit_id"]: record for record in manifest["records"]}

    for layer_id in (3, 77):
        create_tensor, created_tensors = payload_tensor_factory(records, payload)
        provider = LazyGlm53TensorProvider(
            adapter=adapter,
            tensor_factory=create_tensor,
            max_resident_units=18,
        )
        method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
        generator = torch.Generator().manual_seed(1234)
        hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
        topk_weights = torch.tensor(
            [[0.25, 0.75], [0.125, 0.875]],
            dtype=torch.float32,
        )
        topk_ids = torch.tensor([[1, 7], [7, 9]], dtype=torch.int32)
        expected = all_resident_reference(
            hidden_states,
            topk_weights,
            topk_ids,
            all_resident_experts(
                adapter,
                records,
                payload,
                tuple(range(10)),
                layer_id=layer_id,
            ),
        )

        result = method.apply(
            layer=SimpleNamespace(layer_name=f"model.layers.{layer_id}.mlp.experts"),
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

        assert torch.allclose(result, expected, rtol=0.0, atol=0.0)

        expected_unit_ids = set()
        for expert_id in (1, 7, 9):
            expected_unit_ids.update(
                adapter.expert_unit_ids(expert_id, layer_id=layer_id)
            )
        assert set(created_tensors) == expected_unit_ids
        assert all(
            unit_id.startswith(f"model.layers.{layer_id}.")
            for unit_id in created_tensors
        )


def device_runtime_for_tiny_glm(
    layer_ids: tuple[int, ...] = (3,),
    *,
    max_resident_experts: int = 3,
    slot_count: int = 18,
    device_budget_bytes: int = 426,
):
    manifest = tiny_glm_manifest(layer_ids=layer_ids)
    payload = manifest_payload(manifest)
    adapter = Glm53Adapter(manifest)
    store = RecordingMemoryWeightStore({manifest["storage_key"]: payload})
    manager = ResidencyManager(slot_count=slot_count, slot_capacity_bytes=8)
    runtime = DeviceWeightRuntime(
        adapter=adapter,
        manager=manager,
        store=store,
        tensor_backend=NativeTensorBackend(),
        device="cpu",
        device_budget_bytes=device_budget_bytes,
        max_resident_experts=max_resident_experts,
    )
    records = {record["unit_id"]: record for record in manifest["records"]}
    return adapter, runtime, manifest, records, payload, store


def test_glm_hook_routes_layer_specific_demands():
    manifest = tiny_glm_manifest(layer_ids=(3, 77))
    adapter = Glm53Adapter(manifest)
    hook = TieredWeightsResidencyHook(adapter=adapter)
    hook.enable_for_model("glm-5.3")

    layer_demand_unit_ids = {}
    for layer_id in (3, 77):
        demands = hook.build_demands_from_router(
            router_topk_ids=[[1, 7], [7, 9]],
            layer_prefix=f"model.layers.{layer_id}.mlp.experts",
            target_view_id="gpu-vram",
        )
        expected_unit_ids = set()
        for expert_id in (1, 7, 9):
            expected_unit_ids.update(
                adapter.expert_unit_ids(expert_id, layer_id=layer_id)
            )
        layer_demand_unit_ids[layer_id] = expected_unit_ids
        assert {demand.unit_id for demand in demands} == expected_unit_ids

    assert layer_demand_unit_ids[3].isdisjoint(layer_demand_unit_ids[77])


class RecordingGlmProvider:
    def __init__(self, provider, runtime):
        self._provider = provider
        self._runtime = runtime
        self.experts = []
        self.runtime_stats = []

    def request_experts(self, demands):
        resident_experts = self._provider.request_experts(demands)
        self.experts.append(resident_experts.experts)
        self.runtime_stats.append(self._runtime.stats())
        return resident_experts

    def reserve_execution_scratch(self, scratch_bytes):
        return self._runtime.reserve_scratch(scratch_bytes)


class RecordingDeviceRuntime:
    def __init__(self, runtime):
        self._runtime = runtime
        self.acquire_calls = []

    def acquire_expert(self, expert_id, *, layer_id=None):
        self.acquire_calls.append((expert_id, layer_id))
        return self._runtime.acquire_expert(expert_id, layer_id=layer_id)

    def reserve_scratch(self, scratch_bytes):
        return self._runtime.reserve_scratch(scratch_bytes)


def test_vllm_glm_path_routes_and_materializes_layers_3_and_77():
    adapter, runtime, manifest, records, payload, store = device_runtime_for_tiny_glm(
        layer_ids=(3, 77)
    )
    recording_runtime = RecordingDeviceRuntime(runtime)
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=recording_runtime,
        adapter=adapter,
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)

    layer_unit_ids = {}
    for layer_id in (3, 77):
        store.reads.clear()
        recording_runtime.acquire_calls.clear()
        generator = torch.Generator().manual_seed(1234)
        hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
        topk_weights = torch.tensor(
            [[0.25, 0.75], [0.125, 0.875]],
            dtype=torch.float32,
        )
        topk_ids = torch.tensor([[1, 7], [7, 9]], dtype=torch.int32)
        expected = all_resident_reference(
            hidden_states,
            topk_weights,
            topk_ids,
            all_resident_experts(
                adapter,
                records,
                payload,
                tuple(range(10)),
                layer_id=layer_id,
            ),
        )

        result = method.apply(
            layer=SimpleNamespace(layer_name=f"model.layers.{layer_id}.mlp.experts"),
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

        assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
        assert recording_runtime.acquire_calls == [
            (1, layer_id),
            (7, layer_id),
            (7, layer_id),
            (9, layer_id),
        ]

        expected_unit_ids = set()
        for expert_id in (1, 7, 9):
            expected_unit_ids.update(
                adapter.expert_unit_ids(expert_id, layer_id=layer_id)
            )
        read_unit_ids = {
            unit_id
            for unit_id, record in records.items()
            if (
                manifest["storage_key"],
                record["offset_bytes"],
                record["length_bytes"],
            )
            in store.reads
        }
        layer_unit_ids[layer_id] = expected_unit_ids
        assert read_unit_ids == expected_unit_ids

    assert layer_unit_ids[3].isdisjoint(layer_unit_ids[77])


def test_vllm_glm_path_partitions_multi_token_expert_unions():
    adapter, runtime, manifest, records, payload, _ = device_runtime_for_tiny_glm(
        max_resident_experts=8,
        slot_count=1,
        device_budget_bytes=440,
    )

    class ChunkRecordingProvider:
        def __init__(self, provider, runtime):
            self._provider = provider
            self._runtime = runtime
            self.acquired_expert_counts = []

        def request_experts(self, demands):
            acquire_count_before = len(self._runtime.acquire_calls)
            resident_experts = self._provider.request_experts(demands)
            self.acquired_expert_counts.append(
                len(self._runtime.acquire_calls) - acquire_count_before
            )
            return resident_experts

        def reserve_execution_scratch(self, scratch_bytes):
            return self._provider.reserve_execution_scratch(scratch_bytes)

    recording_runtime = RecordingDeviceRuntime(runtime)
    provider = ChunkRecordingProvider(
        DeviceWeightRuntimeGlmTensorProvider(
            runtime=recording_runtime,
            adapter=adapter,
        ),
        recording_runtime,
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(2468)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
    topk_weights = torch.full((2, 8), 0.125, dtype=torch.float32)
    topk_ids = torch.tensor(
        [list(range(8)), list(range(1, 9))],
        dtype=torch.int32,
    )
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(
            adapter,
            records,
            payload,
            tuple(range(10)),
        ),
    )

    result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
    assert provider.acquired_expert_counts == [8, 8]
    assert runtime.stats()["active_leases"] == 0


def test_vllm_glm_path_reserves_execution_scratch_in_device_budget():
    adapter, runtime, _, _, _, _ = device_runtime_for_tiny_glm(
        max_resident_experts=8,
        slot_count=1,
        device_budget_bytes=300,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(runtime=runtime, adapter=adapter)
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)

    with pytest.raises(DeviceRuntimeBackpressureError, match="execution scratch"):
        method.apply(
            layer=FakeRoutedExpertsLayer(),
            x=torch.zeros((1, 3), dtype=torch.bfloat16),
            topk_weights=torch.ones((1, 8), dtype=torch.float32),
            topk_ids=torch.tensor([list(range(8))], dtype=torch.int32),
            shared_experts=None,
            shared_experts_input=None,
        )

    runtime_stats = runtime.stats()
    assert runtime_stats["active_leases"] == 0
    assert runtime_stats["scratch_bytes"] == 0


def test_vllm_glm_path_uses_device_runtime_and_selected_experts_only():
    adapter, runtime, manifest, records, payload, store = device_runtime_for_tiny_glm(
        device_budget_bytes=438,
    )
    provider = RecordingGlmProvider(
        DeviceWeightRuntimeGlmTensorProvider(runtime=runtime, adapter=adapter),
        runtime,
    )

    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(5678)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.float32)
    topk_weights = torch.tensor([[0.25, 0.75], [0.125, 0.875]], dtype=torch.float32)
    topk_ids = torch.tensor([[1, 7], [7, 9]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(
            adapter,
            records,
            payload,
            tuple(range(10)),
        ),
    )

    result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
    assert [set(experts) for experts in provider.experts] == [{1, 7}, {7, 9}]
    assert provider.runtime_stats[0]["active_leases"] == 2
    assert provider.runtime_stats[1]["active_leases"] == 2

    expected_unit_ids = set()
    for expert_id in (1, 7, 9):
        expected_unit_ids.update(adapter.expert_unit_ids(expert_id))
    read_unit_ids = {
        unit_id
        for unit_id, record in records.items()
        if (
            manifest["storage_key"],
            record["offset_bytes"],
            record["length_bytes"],
        )
        in store.reads
    }
    assert read_unit_ids == expected_unit_ids

    for experts in provider.experts:
        for expert_id, projections in experts.items():
            assert set(projections) == {"gate_proj", "up_proj", "down_proj"}
            for projection in projections.values():
                weight_record = records[projection.weight_unit_id]
                scale_record = records[projection.scale_unit_id]
                assert projection.weight_unit_id.endswith(".weight")
                assert (
                    weight_record["scale_unit_id"]
                    == projection.scale_unit_id
                    == scale_record["unit_id"]
                )
                assert projection.weight.dtype == torch.float8_e4m3fn
                assert projection.scale.dtype == torch.float32
                expected_weight_bytes = payload[
                    weight_record["offset_bytes"] : weight_record["offset_bytes"]
                    + weight_record["length_bytes"]
                ]
                expected_scale_bytes = payload[
                    scale_record["offset_bytes"] : scale_record["offset_bytes"]
                    + scale_record["length_bytes"]
                ]
                assert torch.equal(
                    projection.weight.view(torch.uint8).flatten(),
                    torch.frombuffer(
                        bytearray(expected_weight_bytes), dtype=torch.uint8
                    ),
                )
                assert torch.equal(
                    projection.scale.view(torch.uint8).flatten(),
                    torch.frombuffer(
                        bytearray(expected_scale_bytes), dtype=torch.uint8
                    ),
                )

    runtime_stats = runtime.stats()
    assert runtime_stats["active_leases"] == 0
    assert runtime_stats["staging_bytes"] == 0
    assert runtime_stats["resident_groups"] == 3
    assert runtime_stats["resident_units"] == 18
    assert runtime_stats["resident_bytes"] == 90
    assert runtime_stats["resident_groups"] <= runtime_stats["max_resident_experts"]

    unused_expert_id = 0
    for unit_id in adapter.expert_unit_ids(unused_expert_id):
        record = records[unit_id]
        assert (
            manifest["storage_key"],
            record["offset_bytes"],
            record["length_bytes"],
        ) not in store.reads


def test_vllm_glm_eager_provider_is_separate_and_does_not_page():
    manifest = tiny_glm_manifest()
    payload = manifest_payload(manifest)
    adapter = Glm53Adapter(manifest)
    records = {record["unit_id"]: record for record in manifest["records"]}
    create_tensor, created_tensors = payload_tensor_factory(records, payload)
    provider = EagerSelectedExpertsGlmTensorProvider(
        adapter=adapter,
        tensor_factory=create_tensor,
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(1357)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.125, 0.875]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[1, 7], [7, 9]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(
            adapter,
            records,
            payload,
            tuple(range(10)),
        ),
    )

    result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
    assert provider.stats()["paging"] is False
    assert provider.stats()["bounded_eviction"] is False
    assert provider.stats()["requests"] == 2
    assert provider.stats()["loads"] == 24
    assert provider.stats()["storage_read_bytes"] == 120
    assert provider.stats()["active_leases"] == 0
    assert provider.stats()["resident_units"] == 0
    assert len(created_tensors) == 18


def test_tiered_expert_weight_names_accept_loader_prefix_forms():
    from vllm.model_executor.models.deepseek_v2 import _is_tiered_routed_expert_weight

    layer_ids = {10}
    assert _is_tiered_routed_expert_weight(
        "model.layers.10.mlp.experts.0.gate_proj.weight", layer_ids
    )
    assert _is_tiered_routed_expert_weight(
        "layers.10.mlp.experts.0.gate_proj.weight", layer_ids
    )
    assert not _is_tiered_routed_expert_weight(
        "model.layers.9.mlp.experts.0.gate_proj.weight", layer_ids
    )
    assert not _is_tiered_routed_expert_weight(
        "model.layers.10.mlp.shared_experts.gate_proj.weight", layer_ids
    )


def test_vllm_glm_path_releases_device_runtime_leases_on_execution_failure():
    adapter, runtime, manifest, records, _, store = device_runtime_for_tiny_glm()

    class IncompleteProvider:
        def request_experts(self, demands):
            resident_experts = DeviceWeightRuntimeGlmTensorProvider(
                runtime=runtime,
                adapter=adapter,
            ).request_experts(demands)
            del resident_experts.experts[1]["down_proj"]
            return resident_experts

    method = TieredGlm53MoEMethod(
        SimpleNamespace(),
        provider=IncompleteProvider(),
    )
    topk_ids = torch.tensor([[1, 7], [7, 9]], dtype=torch.int32)

    with pytest.raises(KeyError, match="down_proj"):
        method.apply(
            layer=FakeRoutedExpertsLayer(),
            x=torch.zeros((2, 3), dtype=torch.bfloat16),
            topk_weights=torch.ones((2, 2), dtype=torch.float32),
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

    runtime_stats = runtime.stats()
    assert runtime_stats["active_leases"] == 0
    assert runtime_stats["staging_bytes"] == 0
    assert runtime_stats["resident_groups"] == 2
    assert runtime_stats["resident_units"] == 12

    expected_unit_ids = set()
    for expert_id in (1, 7):
        expected_unit_ids.update(adapter.expert_unit_ids(expert_id))
    read_unit_ids = {
        unit_id
        for unit_id, record in records.items()
        if (
            manifest["storage_key"],
            record["offset_bytes"],
            record["length_bytes"],
        )
        in store.reads
    }
    assert read_unit_ids == expected_unit_ids


def test_deepseek_glm_path_selects_lazy_routed_experts(monkeypatch: pytest.MonkeyPatch):
    import vllm.model_executor.models.deepseek_v2 as deepseek_v2

    manifest = tiny_glm_manifest()
    records = {record["unit_id"]: record for record in manifest["records"]}
    create_tensor, _ = payload_tensor_factory(records, manifest_payload(manifest))
    adapter = Glm53Adapter(manifest)
    provider = LazyGlm53TensorProvider(
        adapter=adapter,
        tensor_factory=create_tensor,
    )

    configure_glm_53_tiered_experts(provider)
    try:
        config = SimpleNamespace(
            model_type="glm_moe_dsa",
            hidden_size=6144,
            moe_intermediate_size=2048,
            n_routed_experts=256,
            n_shared_experts=None,
            num_experts_per_tok=8,
            norm_topk_prob=True,
            hidden_act="silu",
            n_group=1,
            topk_group=1,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
        )
        parallel_config = SimpleNamespace(
            use_sequence_parallel_moe=False,
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        )

        class FakeGate:
            def __init__(self, *args, **kwargs):
                self.out_dtype = None
                self.e_score_correction_bias = None

        class FakeExpertGroup:
            def size(self):
                return 1

        class FakeEPGroup:
            device_group = FakeExpertGroup()

        captured_factory_kwargs = {}

        def fake_factory(**kwargs):
            captured_factory_kwargs.update(kwargs)
            return SimpleNamespace()

        monkeypatch.setattr(deepseek_v2, "GateLinear", FakeGate)
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
        monkeypatch.setattr(deepseek_v2, "get_ep_group", lambda: FakeEPGroup())
        monkeypatch.setattr(
            deepseek_v2.rocm_aiter_ops,
            "is_fused_moe_enabled",
            lambda: False,
        )
        monkeypatch.setattr(
            deepseek_v2.rocm_aiter_ops,
            "is_fusion_moe_shared_experts_enabled",
            lambda: False,
        )
        monkeypatch.setattr(deepseek_v2, "FusedMoEFactory", fake_factory)

        moe = deepseek_v2.DeepseekV2MoE(
            config=config,
            parallel_config=parallel_config,
            quant_config=None,
            prefix="model.layers.3.mlp",
        )

        assert moe.use_tiered_routed_experts
        assert captured_factory_kwargs["routed_experts_cls"] is TieredGlm53RoutedExperts
        assert captured_factory_kwargs["routed_experts_args"] == {"provider": provider}
    finally:
        clear_glm_53_tiered_experts()


def test_glm_helper_is_focused_and_reversible():
    clear_glm_53_tiered_experts()
    glm_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        n_routed_experts=256,
        num_experts_per_tok=8,
        hidden_size=6144,
        moe_intermediate_size=2048,
    )
    other_config = SimpleNamespace(
        model_type="deepseek_v3",
        n_routed_experts=256,
        num_experts_per_tok=8,
        hidden_size=6144,
        moe_intermediate_size=2048,
    )

    try:
        configure_glm_53_tiered_experts(SimpleNamespace())
        sparse_layer_results = [
            glm_53_tiered_experts(glm_config, f"model.layers.{layer_index}.mlp")
            for layer_index in range(3, 78)
        ]
        assert len(sparse_layer_results) == 75
        assert all(result is not None for result in sparse_layer_results)
        assert glm_53_tiered_experts(glm_config, "model.layers.2.mlp") is None
        assert glm_53_tiered_experts(glm_config, "model.layers.78.mlp") is None
        assert glm_53_tiered_experts(other_config, "model.layers.3.mlp") is None
    finally:
        clear_glm_53_tiered_experts()
