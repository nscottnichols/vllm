# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for the GLM 5.3 Tiered Weights MoE path."""

from __future__ import annotations

import hashlib
import math
import sys
from concurrent.futures import Future
from contextlib import contextmanager
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

from vllm.model_executor.layers.fused_moe import (  # noqa: E402
    tiered_glm as tiered_glm_module,
)
from vllm.model_executor.layers.fused_moe.tiered_glm import (  # noqa: E402
    DeviceWeightRuntimeGlmTensorProvider,
    EagerSelectedExpertsGlmTensorProvider,
    LazyGlm53TensorProvider,
    TieredGlm53MoEMethod,
    TieredGlm53RoutedExperts,
    TieredGlmApplyTimings,
    TieredGlmCurrentRouterPredictor,
    TieredGlmExecutionTimings,
    TieredGlmPreviousTokenLayerPredictor,
    TieredGlmPreviousTokenPredictor,
    TieredGlmProjection,
    TieredGlmRouteController,
    _dequant_glm_53_fp8_block,
    _glm_53_execution_scratch_bytes,
    _partition_rows_by_expert_union,
    _timed_glm_apply_stage,
    _timed_glm_execution_stage,
    clear_glm_53_tiered_experts,
    configure_glm_53_tiered_experts,
    glm_53_tiered_execution_callback,
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
    expert_count: int = 10,
    max_resident_experts: int = 3,
    slot_count: int = 18,
    device_budget_bytes: int = 600,
    expert_cache_budget_bytes: int | None = None,
    prefetch_depth: int = 1,
):
    manifest = tiny_glm_manifest(
        layer_ids=layer_ids,
        expert_count=expert_count,
    )
    payload = manifest_payload(manifest)
    for record in manifest["records"]:
        record["checksum_sha256"] = hashlib.sha256(
            payload[
                record["offset_bytes"] : record["offset_bytes"] + record["length_bytes"]
            ]
        ).hexdigest()
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
        expert_cache_budget_bytes=expert_cache_budget_bytes,
        prefetch_depth=prefetch_depth,
    )
    records = {record["unit_id"]: record for record in manifest["records"]}
    return (
        adapter,
        FakeBatchedDeviceRuntime(runtime),
        manifest,
        records,
        payload,
        store,
    )


def glm_expert_demands(
    adapter: Glm53Adapter,
    expert_id: int,
    layer_id: int = 3,
):
    return [
        SimpleNamespace(unit_id=unit_id)
        for unit_id in adapter.expert_unit_ids(expert_id, layer_id=layer_id)
    ]


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


class ChunkRecordingProvider:
    def __init__(self, provider, runtime):
        self._provider = provider
        self._runtime = runtime
        self.acquired_expert_counts = []

    def request_experts(self, demands):
        acquire_count_before = len(self._runtime.acquire_experts_calls)
        resident_experts = self._provider.request_experts(demands)
        acquire_calls = self._runtime.acquire_experts_calls[acquire_count_before:]
        self.acquired_expert_counts.append(
            sum(len(selected_expert_ids) for selected_expert_ids, _ in acquire_calls)
        )
        return resident_experts

    def reserve_execution_scratch(self, scratch_bytes):
        return self._provider.reserve_execution_scratch(scratch_bytes)

    def stats(self):
        return self._provider.stats()


class FakeBatchedLeaseSet:
    def __init__(self, leases):
        self.leases = tuple(leases)
        self.released = False

    def __enter__(self):
        return self.leases

    def __exit__(self, exc_type, exc, traceback):
        for lease in reversed(self.leases):
            lease.release()
        self.released = True
        return False


class FakeBatchedDeviceRuntime:
    def __init__(self, runtime):
        self._runtime = runtime
        self.acquire_experts_calls = []
        self.lease_sets = []

    def acquire_experts(self, selected_expert_ids, *, layer_id=None):
        selected_expert_ids = tuple(selected_expert_ids)
        self.acquire_experts_calls.append((selected_expert_ids, layer_id))
        leases = []
        try:
            for expert_id in selected_expert_ids:
                leases.append(
                    self._runtime.acquire_expert(expert_id, layer_id=layer_id)
                )
        except BaseException:
            for lease in reversed(leases):
                lease.release()
            raise
        lease_set = FakeBatchedLeaseSet(leases)
        self.lease_sets.append(lease_set)
        return lease_set

    def acquire_expert(self, expert_id, *, layer_id=None):
        raise AssertionError("provider used per-expert acquisition")

    def prefetch_experts(self, selected_expert_ids, *, layer_id=None):
        return self._runtime.prefetch_experts(
            selected_expert_ids,
            layer_id=layer_id,
        )

    def submit_prefetch_experts(self, selected_expert_ids, *, layer_id=None):
        return self._runtime.submit_prefetch_experts(
            selected_expert_ids,
            layer_id=layer_id,
        )

    def submit_expert_cache_prefetch(self, selected_expert_ids, *, layer_id=None):
        return self._runtime.submit_expert_cache_prefetch(
            selected_expert_ids,
            layer_id=layer_id,
        )

    def reserve_scratch(self, scratch_bytes):
        return self._runtime.reserve_scratch(scratch_bytes)

    def stats(self):
        return self._runtime.stats()

    def close_prefetches(self, *, wait=True):
        return self._runtime.close_prefetches(wait=wait)

    def __getattr__(self, name):
        return getattr(self._runtime, name)


class RecordingDeviceRuntime:
    def __init__(self, runtime):
        self._runtime = runtime
        self.acquire_experts_calls = []
        self.prefetch_calls = []
        self.submit_calls = []
        self.events = []

    def acquire_experts(self, selected_expert_ids, *, layer_id=None):
        selected_expert_ids = tuple(selected_expert_ids)
        self.acquire_experts_calls.append((selected_expert_ids, layer_id))
        self.events.append(("acquire-experts", selected_expert_ids, layer_id))
        return self._runtime.acquire_experts(
            selected_expert_ids,
            layer_id=layer_id,
        )

    def acquire_expert(self, expert_id, *, layer_id=None):
        raise AssertionError("provider used per-expert acquisition")

    def prefetch_experts(self, selected_expert_ids, *, layer_id=None):
        selected_expert_ids = tuple(selected_expert_ids)
        self.prefetch_calls.append((selected_expert_ids, layer_id))
        self.events.append(("prefetch", selected_expert_ids, layer_id))
        return self._runtime.prefetch_experts(
            selected_expert_ids,
            layer_id=layer_id,
        )

    def submit_prefetch_experts(self, selected_expert_ids, *, layer_id=None):
        self.submit_calls.append((tuple(selected_expert_ids), layer_id))
        self.events.append(("submit-prefetch", tuple(selected_expert_ids), layer_id))
        return self._runtime.submit_prefetch_experts(
            selected_expert_ids,
            layer_id=layer_id,
        )

    def reserve_scratch(self, scratch_bytes):
        return self._runtime.reserve_scratch(scratch_bytes)

    def stats(self):
        return self._runtime.stats()


def test_vllm_glm_path_routes_and_materializes_layers_3_and_77():
    adapter, runtime, manifest, records, payload, store = device_runtime_for_tiny_glm(
        layer_ids=(3, 77),
        device_budget_bytes=600,
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
        recording_runtime.acquire_experts_calls.clear()
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
        assert recording_runtime.acquire_experts_calls == [((1, 7, 9), layer_id)]

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
        device_budget_bytes=600,
    )

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


def test_partition_rows_by_expert_union_defaults_to_routing_width():
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)

    assert _partition_rows_by_expert_union(topk_ids, 2) == [[0], [1], [2]]
    assert _partition_rows_by_expert_union(topk_ids, 2, max_expert_union=4) == [
        [0, 1, 2]
    ]


def test_glm_execution_scratch_matches_parent_n32_dequant_peak():
    projection = TieredGlmProjection(
        expert_id=1,
        projection="gate_proj",
        weight=torch.empty((2048, 6144), dtype=torch.float8_e4m3fn),
        weight_unit_id="gate_proj.weight",
        scale=torch.empty((16, 48), dtype=torch.float32),
        scale_unit_id="gate_proj.weight_scale_inv",
    )
    hidden_states = torch.empty((32, 6144), dtype=torch.bfloat16)

    full_scratch_bytes = _glm_53_execution_scratch_bytes(
        hidden_states,
        {1: {"gate_proj": projection}},
    )
    nested_scratch_bytes = _glm_53_execution_scratch_bytes(
        hidden_states,
        {1: {"gate_proj": projection}},
        include_output=False,
    )
    assert full_scratch_bytes == 264_634_368
    assert nested_scratch_bytes == 264_241_152
    assert (
        full_scratch_bytes - nested_scratch_bytes
        == hidden_states.numel() * hidden_states.element_size()
    )


def test_glm_execution_scratch_accounts_for_fixed_gemm_batches():
    projection = TieredGlmProjection(
        expert_id=1,
        projection="gate_proj",
        weight=torch.empty((2, 3), dtype=torch.float8_e4m3fn),
        weight_unit_id="gate_proj.weight",
        scale=torch.empty((1, 1), dtype=torch.float32),
        scale_unit_id="gate_proj.weight_scale_inv",
    )
    experts = {1: {"gate_proj": projection}}

    for batch_size in (1, 2, 3, 4):
        hidden_states = torch.empty((batch_size, 3), dtype=torch.bfloat16)
        padded_activation_bytes = 4 * ((2 * batch_size + 8) * 3 + 12 * 2)
        callback_activation_bytes = max(
            4 * batch_size * (5 * 3 + 3 * 2),
            padded_activation_bytes,
        )
        expected_scratch = max(
            callback_activation_bytes + 16 * 6,
            20 * batch_size * 3 + 16 * batch_size * 2 + 12 * 6,
            21 * 6,
        )

        assert (
            _glm_53_execution_scratch_bytes(
                hidden_states,
                experts,
                include_output=False,
            )
            == expected_scratch
        )


def test_vllm_glm_path_batches_mixed_rows_up_to_provider_max_union():
    adapter, runtime, manifest, records, payload, _ = device_runtime_for_tiny_glm(
        expert_count=18,
        max_resident_experts=12,
        slot_count=1,
        device_budget_bytes=810,
    )
    recording_runtime = RecordingDeviceRuntime(runtime)
    provider = ChunkRecordingProvider(
        DeviceWeightRuntimeGlmTensorProvider(
            runtime=recording_runtime,
            adapter=adapter,
        ),
        recording_runtime,
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(8642)
    hidden_states = torch.randn((3, 3), generator=generator, dtype=torch.bfloat16)
    topk_weights = torch.full((3, 8), 0.125, dtype=torch.float32)
    topk_ids = torch.tensor(
        [list(range(8)), list(range(4, 12)), list(range(10, 18))],
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
            tuple(range(18)),
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

    assert torch.equal(result, expected)
    assert provider.acquired_expert_counts == [12, 8]
    assert all(lease_set.released for lease_set in runtime.lease_sets)
    runtime_stats = runtime.stats()
    assert runtime_stats["active_leases"] == 0
    assert runtime_stats["scratch_bytes"] == 0
    assert runtime_stats["resident_groups"] <= runtime_stats["max_resident_experts"]


def test_vllm_glm_path_consumes_chunk_output_before_release():
    adapter, runtime, manifest, records, payload, _ = device_runtime_for_tiny_glm(
        expert_count=18,
        max_resident_experts=12,
        slot_count=1,
        device_budget_bytes=810,
    )
    events = []

    class CopyObservingTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func is torch.Tensor.index_copy_:
                events.append("output-copy")
            return super().__torch_function__(func, types, args, kwargs)

    class RecordedResidentExperts:
        def __init__(self, resident_experts):
            self._resident_experts = resident_experts

        def __enter__(self):
            return self._resident_experts

        def __exit__(self, exc_type, exc, traceback):
            events.append("release")
            return self._resident_experts.__exit__(exc_type, exc, traceback)

    class LifecycleRecordingProvider:
        def __init__(self, provider):
            self._provider = provider
            self._scratch_reservations = 0

        def request_experts(self, demands):
            events.append("request")
            return RecordedResidentExperts(self._provider.request_experts(demands))

        def reserve_execution_scratch(self, scratch_bytes):
            self._scratch_reservations += 1
            label = (
                "output-scratch"
                if self._scratch_reservations == 1
                else "chunk-scratch"
            )

            @contextmanager
            def recorded_reservation():
                events.append(f"{label}-enter")
                try:
                    with self._provider.reserve_execution_scratch(scratch_bytes):
                        yield
                finally:
                    events.append(f"{label}-exit")

            return recorded_reservation()

        def stats(self):
            return self._provider.stats()

        def execution_callback(self, hidden_states, topk_weights, topk_ids, experts):
            events.append("callback")
            result = glm_53_tiered_execution_callback(
                hidden_states,
                topk_weights,
                topk_ids,
                experts,
            )
            return result.as_subclass(CopyObservingTensor)

    provider = LifecycleRecordingProvider(
        DeviceWeightRuntimeGlmTensorProvider(
            runtime=RecordingDeviceRuntime(runtime),
            adapter=adapter,
        )
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(1357)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
    topk_weights = torch.full((2, 8), 0.125, dtype=torch.float32)
    topk_ids = torch.tensor(
        [list(range(8)), list(range(4, 12))],
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
            tuple(range(12)),
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

    assert torch.equal(result, expected)
    assert events == [
        "output-scratch-enter",
        "request",
        "chunk-scratch-enter",
        "callback",
        "output-copy",
        "chunk-scratch-exit",
        "release",
        "output-scratch-exit",
    ]
    runtime_stats = runtime.stats()
    assert runtime_stats["active_leases"] == 0
    assert runtime_stats["scratch_bytes"] == 0
    assert all(lease_set.released for lease_set in runtime.lease_sets)


def test_vllm_glm_path_reserves_execution_scratch_in_device_budget():
    adapter, runtime, _, _, _, _ = device_runtime_for_tiny_glm(
        expert_count=18,
        max_resident_experts=12,
        slot_count=1,
        device_budget_bytes=500,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(runtime=runtime, adapter=adapter)
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)

    with pytest.raises(DeviceRuntimeBackpressureError, match="execution scratch"):
        method.apply(
            layer=FakeRoutedExpertsLayer(),
            x=torch.zeros((2, 3), dtype=torch.bfloat16),
            topk_weights=torch.ones((2, 8), dtype=torch.float32),
            topk_ids=torch.tensor(
                [list(range(8)), list(range(4, 12))],
                dtype=torch.int32,
            ),
            shared_experts=None,
            shared_experts_input=None,
        )

    runtime_stats = runtime.stats()
    assert runtime_stats["active_leases"] == 0
    assert runtime_stats["scratch_bytes"] == 0


def test_vllm_glm_path_uses_device_runtime_and_selected_experts_only():
    adapter, runtime, manifest, records, payload, store = device_runtime_for_tiny_glm(
        device_budget_bytes=600,
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


def test_vllm_glm_device_cache_hit_avoids_runtime_lease_and_copy(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter, runtime, _, records, payload, store = device_runtime_for_tiny_glm(
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )
    recording_runtime = RecordingDeviceRuntime(runtime)
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=recording_runtime,
        adapter=adapter,
        device_cache_budget_bytes=30,
    )
    demands = glm_expert_demands(adapter, expert_id=1)
    generator = torch.Generator().manual_seed(2468)
    hidden_states = torch.randn((1, 3), generator=generator, dtype=torch.float32)
    topk_weights = torch.tensor([[1.0]], dtype=torch.float32)
    topk_ids = torch.tensor([[1]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, (1,)),
    )

    with provider.request_experts(demands) as first_experts:
        first_projections = first_experts.experts[1]
        first_result = glm_53_tiered_execution_callback(
            hidden_states,
            topk_weights,
            topk_ids,
            first_experts.experts,
        )
        inside_stats = provider.stats()
        assert inside_stats["device_cache_active_leases"] == 3
        assert inside_stats["active_leases"] == 0

    first_stats = provider.stats()
    read_count_after_first = len(store.reads)
    assert first_stats["device_cache_hits"] == 0
    assert first_stats["device_cache_misses"] == 3
    assert first_stats["device_cache_bytes"] == 30
    assert first_stats["device_cache_active_leases"] == 0
    assert first_stats["scratch_bytes"] == 30
    assert torch.equal(first_result, expected)

    def fail_clone(self, *args, **kwargs):
        raise AssertionError("Tiered GLM cache hit copied a tensor")

    monkeypatch.setattr(torch.Tensor, "clone", fail_clone)
    with provider.request_experts(demands) as second_experts:
        second_projections = second_experts.experts[1]
        second_result = glm_53_tiered_execution_callback(
            hidden_states,
            topk_weights,
            topk_ids,
            second_experts.experts,
        )
        for projection_name in first_projections:
            assert (
                second_projections[projection_name]
                is first_projections[projection_name]
            )
            assert (
                second_projections[projection_name].weight
                is first_projections[projection_name].weight
            )
            assert (
                second_projections[projection_name].scale
                is first_projections[projection_name].scale
            )

    second_stats = provider.stats()
    assert second_stats["device_cache_hits"] == 3
    assert second_stats["device_cache_misses"] == 3
    assert second_stats["device_cache_bytes"] == 30
    assert second_stats["device_cache_active_leases"] == 0
    assert second_stats["scratch_bytes"] == 30
    assert recording_runtime.prefetch_calls == []
    assert recording_runtime.acquire_experts_calls == [((1,), 3)]
    assert len(store.reads) == read_count_after_first
    assert torch.equal(second_result, expected)


def test_glm_execution_callback_timings_preserve_exact_output_and_accumulate():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        max_resident_experts=3,
        slot_count=18,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
    )
    demands = [
        *glm_expert_demands(adapter, expert_id=1),
        *glm_expert_demands(adapter, expert_id=2),
    ]
    generator = torch.Generator().manual_seed(8642)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.float32)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.75, 0.25]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, (1, 2)),
    )
    timings = TieredGlmExecutionTimings()

    with provider.request_experts(demands) as resident_experts:
        unprofiled_result = glm_53_tiered_execution_callback(
            hidden_states,
            topk_weights,
            topk_ids,
            resident_experts.experts,
        )
        first_result = glm_53_tiered_execution_callback(
            hidden_states,
            topk_weights,
            topk_ids,
            resident_experts.experts,
            timings=timings,
        )
        first_metrics = timings.metrics()
        second_result = glm_53_tiered_execution_callback(
            hidden_states,
            topk_weights,
            topk_ids,
            resident_experts.experts,
            timings=timings,
        )
        second_metrics = timings.metrics()

    assert torch.equal(unprofiled_result, expected)
    assert torch.equal(first_result, unprofiled_result)
    assert torch.equal(second_result, unprofiled_result)
    for metric_name in first_metrics:
        assert second_metrics[metric_name] >= first_metrics[metric_name]
    for count_name in (
        "routing_indexing_count",
        "fp8_block_dequantization_count",
        "matmul_activation_accumulation_count",
    ):
        assert first_metrics[count_name] > 0
        assert second_metrics[count_name] >= first_metrics[count_name]


def test_glm_execution_callback_uses_fixed_gemm_batch_size(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        max_resident_experts=3,
        slot_count=18,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
    )
    demands = [
        *glm_expert_demands(adapter, expert_id=1),
        *glm_expert_demands(adapter, expert_id=2),
    ]
    generator = torch.Generator().manual_seed(24680)
    hidden_states = torch.randn((4, 3), generator=generator, dtype=torch.float32)
    topk_weights = torch.full((4, 2), 0.5, dtype=torch.float32)
    topk_ids = torch.tile(
        torch.tensor([[1, 2]], dtype=torch.int32),
        (4, 1),
    )
    original_fixed_batch_input = tiered_glm_module._glm_53_fixed_batch_expert_input
    fixed_batch_shapes: list[tuple[int, ...]] = []
    results: dict[int, torch.Tensor] = {}

    def record_fixed_batch_input(expert_input: torch.Tensor) -> torch.Tensor:
        fixed_batch_input = original_fixed_batch_input(expert_input)
        fixed_batch_shapes.append(tuple(fixed_batch_input.shape))
        return fixed_batch_input

    monkeypatch.setattr(
        tiered_glm_module,
        "_glm_53_fixed_batch_expert_input",
        record_fixed_batch_input,
    )

    with provider.request_experts(demands) as resident_experts:
        for batch_size in (1, 2, 3, 4):
            batch_expected = all_resident_reference(
                hidden_states[:batch_size],
                topk_weights[:batch_size],
                topk_ids[:batch_size],
                all_resident_experts(adapter, records, payload, (1, 2)),
            )
            results[batch_size] = glm_53_tiered_execution_callback(
                hidden_states[:batch_size],
                topk_weights[:batch_size],
                topk_ids[:batch_size],
                resident_experts.experts,
            )
            assert torch.equal(results[batch_size], batch_expected)

    assert fixed_batch_shapes == [(4, 3)] * 8
    for batch_size in (2, 3, 4):
        assert torch.equal(results[1][0], results[batch_size][0])


def test_glm_execution_timing_synchronizes_cuda_stages(monkeypatch: pytest.MonkeyPatch):
    synchronized_devices = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: synchronized_devices.append(device),
    )
    timings = TieredGlmExecutionTimings()
    cuda_device = SimpleNamespace(type="cuda")

    with _timed_glm_execution_stage(
        timings,
        "routing_indexing",
        cuda_device,
    ):
        pass

    assert synchronized_devices == [cuda_device, cuda_device]
    assert timings.metrics()["routing_indexing_count"] == 1
    assert timings.metrics()["routing_indexing_seconds"] >= 0.0


def test_glm_apply_timings_cover_every_diagnostic_bucket():
    adapter, runtime, _, _, _, _ = device_runtime_for_tiny_glm(
        device_budget_bytes=600,
    )
    timings = TieredGlmApplyTimings()
    inner_provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        apply_timings=timings,
    )

    class PassthroughRouteController:
        def __init__(self):
            self.apply_calls = []

        def apply(self, layer_id, topk_ids, topk_weights):
            self.apply_calls.append(layer_id)
            return topk_ids, topk_weights

    class DiagnosticProvider:
        def __init__(self, provider, route_controller):
            self._provider = provider
            self.route_controller = route_controller
            self.apply_timings = timings
            self.callback_calls = 0

        def request_experts(self, demands):
            return self._provider.request_experts(demands)

        def reserve_execution_scratch(self, scratch_bytes):
            return self._provider.reserve_execution_scratch(scratch_bytes)

        def stats(self):
            return self._provider.stats()

        def execution_callback(self, hidden_states, topk_weights, topk_ids, experts):
            self.callback_calls += 1
            return hidden_states

    route_controller = PassthroughRouteController()
    provider = DiagnosticProvider(inner_provider, route_controller)
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(97531)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.float32)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.75, 0.25]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)

    result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert torch.equal(result, hidden_states)
    assert route_controller.apply_calls == [3]
    assert provider.callback_calls == 1
    metrics = provider.stats()
    for metric_name in TieredGlmApplyTimings._STAGE_METRIC_NAMES.values():
        assert metrics[f"{metric_name}_count"] == 1
        assert metrics[f"{metric_name}_seconds"] >= 0.0
    assert "provider_request_seconds" not in metrics
    assert "provider_request_count" not in metrics
    assert runtime.stats()["active_leases"] == 0


def test_glm_apply_timings_are_absent_when_diagnostics_are_disabled():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        device_budget_bytes=600,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(runtime=runtime, adapter=adapter)
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(13579)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.float32)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.75, 0.25]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, (1, 2)),
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
    assert provider.apply_timings is None
    stats = provider.stats()
    assert all(
        f"{metric_name}_seconds" not in stats
        for metric_name in TieredGlmApplyTimings._STAGE_METRIC_NAMES.values()
    )
    assert all(
        f"{metric_name}_count" not in stats
        for metric_name in TieredGlmApplyTimings._STAGE_METRIC_NAMES.values()
    )


def test_glm_apply_timing_synchronizes_cuda_boundaries_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    synchronized_devices = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: synchronized_devices.append(device),
    )
    timings = TieredGlmApplyTimings()
    cuda_device = SimpleNamespace(type="cuda")

    for stage_name in ("route_controller", "output_copy"):
        with _timed_glm_apply_stage(timings, stage_name, cuda_device):
            pass

    assert synchronized_devices == [cuda_device] * 4
    assert timings.metrics()["route_controller_count"] == 1
    assert timings.metrics()["output_copy_count"] == 1

    for stage_name in ("route_controller", "output_copy"):
        with _timed_glm_apply_stage(None, stage_name, cuda_device):
            pass

    assert synchronized_devices == [cuda_device] * 4


def test_provider_reserve_execution_scratch_falls_back_to_legacy_runtime():
    class LegacyRuntime:
        def __init__(self):
            self.reserve_scratch_calls = []

        def reserve_scratch(self, scratch_bytes):
            self.reserve_scratch_calls.append(scratch_bytes)
            return "legacy-reservation"

    runtime = LegacyRuntime()
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=SimpleNamespace(),
    )

    assert provider.reserve_execution_scratch(123) == "legacy-reservation"
    assert runtime.reserve_scratch_calls == [123]


def test_glm_fp8_block_dequantization_aligned_fast_path_is_exact():
    generator = torch.Generator().manual_seed(24680)
    weight = torch.randn(
        (256, 512),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    scale = torch.rand((2, 4), generator=generator, dtype=torch.float32) + 0.125
    projection = TieredGlmProjection(
        expert_id=1,
        projection="gate_proj",
        weight=weight,
        weight_unit_id="aligned.weight",
        scale=scale,
        scale_unit_id="aligned.weight_scale_inv",
    )

    result = _dequant_glm_53_fp8_block(projection)
    expanded_scale = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    expected = weight.to(torch.float32) * expanded_scale

    assert torch.equal(result, expected)


def test_glm_dequant_cache_preserves_exact_output_and_reuses_values():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        max_resident_experts=3,
        slot_count=18,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        dequant_cache_budget_bytes=72,
    )
    demands = glm_expert_demands(adapter, expert_id=1)
    generator = torch.Generator().manual_seed(135724)
    hidden_states = torch.randn((1, 3), generator=generator, dtype=torch.float32)
    topk_weights = torch.tensor([[1.0]], dtype=torch.float32)
    topk_ids = torch.tensor([[1]], dtype=torch.int32)
    expected_weights = dequant_projection(
        all_resident_experts(adapter, records, payload, (1,))[1]
    )
    expected_output = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, (1,)),
    )

    with provider.request_experts(demands) as first_experts:
        first_gate = first_experts.experts[1]["gate_proj"].dequantized_weight
        assert first_gate is not None
        assert torch.equal(first_gate, expected_weights["gate_proj"])
        first_output = glm_53_tiered_execution_callback(
            hidden_states,
            topk_weights,
            topk_ids,
            first_experts.experts,
        )
        first_stats = provider.stats()
        assert first_stats["dequant_cache_hits"] == 0
        assert first_stats["dequant_cache_misses"] == 3
        assert first_stats["dequant_cache_bytes"] == 72
        assert first_stats["dequant_cache_peak_bytes"] == 72
        assert first_stats["dequant_cache_active_leases"] == 3
        assert first_stats["scratch_bytes"] == 72

    assert torch.equal(first_output, expected_output)

    with provider.request_experts(demands) as second_experts:
        second_gate = second_experts.experts[1]["gate_proj"].dequantized_weight
        assert second_gate is first_gate
        second_output = glm_53_tiered_execution_callback(
            hidden_states,
            topk_weights,
            topk_ids,
            second_experts.experts,
        )

    second_stats = provider.stats()
    assert torch.equal(second_output, expected_output)
    assert second_stats["dequant_cache_hits"] == 3
    assert second_stats["dequant_cache_misses"] == 3
    assert second_stats["dequant_cache_bytes"] == 72
    assert second_stats["dequant_cache_active_leases"] == 0


def test_glm_dequant_cache_disabled_preserves_exact_output_without_derived_values():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        max_resident_experts=3,
        slot_count=18,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        dequant_cache_budget_bytes=None,
    )
    demands = glm_expert_demands(adapter, expert_id=1)
    generator = torch.Generator().manual_seed(135725)
    hidden_states = torch.randn((1, 3), generator=generator, dtype=torch.float32)
    topk_weights = torch.tensor([[1.0]], dtype=torch.float32)
    topk_ids = torch.tensor([[1]], dtype=torch.int32)
    expected_output = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, (1,)),
    )

    with provider.request_experts(demands) as resident_experts:
        assert all(
            projection.dequantized_weight is None
            for projection in resident_experts.experts[1].values()
        )
        output = glm_53_tiered_execution_callback(
            hidden_states,
            topk_weights,
            topk_ids,
            resident_experts.experts,
        )

    stats = provider.stats()
    assert torch.equal(output, expected_output)
    assert stats["dequant_cache_hits"] == 0
    assert stats["dequant_cache_misses"] == 0
    assert stats["dequant_cache_bytes"] == 0
    assert stats["dequant_cache_budget_bytes"] is None
    assert stats["dequant_cache_active_leases"] == 0


def test_glm_dequant_cache_is_bounded_and_evicts_lru_entries():
    adapter, runtime, _, _, _, _ = device_runtime_for_tiny_glm(
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        dequant_cache_budget_bytes=24,
    )

    for expert_id in range(3):
        with provider.request_experts(
            glm_expert_demands(adapter, expert_id)
        ) as resident_experts:
            cached_count = sum(
                projection.dequantized_weight is not None
                for projection in resident_experts.experts[expert_id].values()
            )
            assert cached_count == 1
            stats = provider.stats()
            assert stats["dequant_cache_bytes"] <= 24
            assert stats["dequant_cache_peak_bytes"] <= 24
            assert stats["dequant_cache_active_leases"] == 1

    final_stats = provider.stats()
    assert final_stats["dequant_cache_bytes"] == 24
    assert final_stats["dequant_cache_peak_bytes"] == 24
    assert final_stats["dequant_cache_budget_bytes"] == 24
    assert final_stats["dequant_cache_evictions"] == 2
    assert final_stats["dequant_cache_active_leases"] == 0


def test_glm_dequant_cache_active_lease_blocks_eviction_until_release():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        dequant_cache_budget_bytes=24,
    )
    reference_experts = all_resident_experts(adapter, records, payload, (0, 1))

    with provider.request_experts(
        glm_expert_demands(adapter, expert_id=0)
    ) as first_experts:
        first_gate = first_experts.experts[0]["gate_proj"].dequantized_weight
        assert first_gate is not None
        assert provider.stats()["dequant_cache_evictions"] == 0

        with provider.request_experts(
            glm_expert_demands(adapter, expert_id=1)
        ) as second_experts:
            second_gate = second_experts.experts[1]["gate_proj"].dequantized_weight
            assert second_gate is None
        stats = provider.stats()
        assert stats["dequant_cache_bytes"] == 24
        assert stats["dequant_cache_evictions"] == 0
        assert stats["dequant_cache_active_leases"] == 1

        assert torch.equal(
            first_gate,
            dequant_projection(reference_experts[0])["gate_proj"],
        )

    with provider.request_experts(glm_expert_demands(adapter, expert_id=1)):
        pass

    final_stats = provider.stats()
    assert final_stats["dequant_cache_evictions"] == 1
    assert final_stats["dequant_cache_bytes"] == 24
    assert final_stats["dequant_cache_active_leases"] == 0


def test_glm_dequant_cache_close_releases_scratch_accounting():
    adapter, runtime, _, _, _, _ = device_runtime_for_tiny_glm(
        max_resident_experts=3,
        slot_count=18,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        dequant_cache_budget_bytes=72,
    )

    with provider.request_experts(glm_expert_demands(adapter, expert_id=1)):
        pass

    stats = provider.stats()
    assert stats["dequant_cache_bytes"] == 72
    assert stats["scratch_bytes"] == 72

    provider.close_prefetches()

    closed_stats = provider.stats()
    assert closed_stats["dequant_cache_bytes"] == 0
    assert closed_stats["dequant_cache_active_leases"] == 0
    assert closed_stats["dequant_cache_hits"] == 0
    assert closed_stats["dequant_cache_misses"] == 3
    assert closed_stats["scratch_bytes"] == 0


def test_vllm_glm_device_cache_uses_lru_eviction():
    adapter, runtime, _, _, _, _ = device_runtime_for_tiny_glm(
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        device_cache_budget_bytes=60,
    )

    for expert_id in (0, 1):
        with provider.request_experts(glm_expert_demands(adapter, expert_id)):
            pass

    stats = provider.stats()
    assert stats["device_cache_bytes"] == 60
    assert stats["device_cache_evictions"] == 0

    with provider.request_experts(glm_expert_demands(adapter, expert_id=0)):
        pass

    with provider.request_experts(glm_expert_demands(adapter, expert_id=2)):
        pass

    stats = provider.stats()
    assert stats["device_cache_bytes"] == 60
    assert stats["device_cache_evictions"] == 3
    assert stats["device_cache_active_leases"] == 0

    expected_keys = set()
    for expert_id in (0, 2):
        expected_keys.update(
            (3, expert_id, projection_name)
            for projection_name in ("gate_proj", "up_proj", "down_proj")
        )
    assert set(provider._device_cache) == expected_keys


def test_vllm_glm_device_cache_active_lease_prevents_eviction():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        device_cache_budget_bytes=30,
    )
    reference_experts = all_resident_experts(adapter, records, payload, (0, 1))

    with provider.request_experts(glm_expert_demands(adapter, expert_id=0)) as first:
        first_gate = first.experts[0]["gate_proj"]
        first_stats = provider.stats()
        assert first_stats["device_cache_active_leases"] == 3
        assert first_stats["device_cache_bytes"] == 30
        assert first_stats["device_cache_evictions"] == 0

        with provider.request_experts(
            glm_expert_demands(adapter, expert_id=1)
        ) as second:
            second_stats = provider.stats()
            assert second_stats["device_cache_active_leases"] == 3
            assert second_stats["device_cache_bytes"] == 30
            assert second_stats["device_cache_evictions"] == 0
            assert second_stats["active_leases"] == 1
            for projection_name, projection in second.experts[1].items():
                weight_reference = reference_experts[1][projection.weight_unit_id]
                scale_reference = reference_experts[1][projection.scale_unit_id]
                assert torch.equal(
                    projection.weight.view(torch.uint8),
                    weight_reference.view(torch.uint8),
                )
                assert torch.equal(
                    projection.scale.view(torch.uint8),
                    scale_reference.view(torch.uint8),
                )

        assert (3, 0, "gate_proj") in provider._device_cache
        assert provider._device_cache[(3, 0, "gate_proj")].projection is first_gate
        assert provider.stats()["device_cache_active_leases"] == 3

    assert provider.stats()["device_cache_active_leases"] == 0
    assert provider.stats()["device_cache_evictions"] == 0
    assert torch.equal(
        first_gate.weight,
        reference_experts[0][first_gate.weight_unit_id],
    )

    with provider.request_experts(glm_expert_demands(adapter, expert_id=1)):
        pass

    final_stats = provider.stats()
    assert final_stats["device_cache_active_leases"] == 0
    assert final_stats["device_cache_evictions"] == 3
    assert final_stats["device_cache_bytes"] == 30
    assert (3, 0, "gate_proj") not in provider._device_cache
    assert (3, 1, "gate_proj") in provider._device_cache


def test_vllm_glm_device_cache_failure_releases_active_leases():
    adapter, runtime, _, _, _, _ = device_runtime_for_tiny_glm(
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )

    class FailingAcquireRuntime:
        def __init__(self, runtime):
            self._runtime = runtime

        def prefetch_experts(self, selected_expert_ids, *, layer_id=None):
            return self._runtime.prefetch_experts(
                selected_expert_ids,
                layer_id=layer_id,
            )

        def acquire_experts(self, selected_expert_ids, *, layer_id=None):
            if 1 in selected_expert_ids:
                raise RuntimeError("synthetic GLM acquire failure")
            return self._runtime.acquire_experts(
                selected_expert_ids,
                layer_id=layer_id,
            )

        def reserve_scratch(self, scratch_bytes):
            return self._runtime.reserve_scratch(scratch_bytes)

        def stats(self):
            return self._runtime.stats()

    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=FailingAcquireRuntime(runtime),
        adapter=adapter,
        device_cache_budget_bytes=30,
    )
    with provider.request_experts(glm_expert_demands(adapter, expert_id=0)):
        pass

    demands = [
        *glm_expert_demands(adapter, expert_id=0),
        *glm_expert_demands(adapter, expert_id=1),
    ]
    with pytest.raises(RuntimeError, match="synthetic GLM acquire failure"):
        provider.request_experts(demands)

    stats = provider.stats()
    assert stats["device_cache_active_leases"] == 0
    assert stats["device_cache_bytes"] == 30
    assert stats["device_cache_evictions"] == 0
    assert stats["active_leases"] == 0
    assert (3, 0, "gate_proj") in provider._device_cache


def test_vllm_glm_device_cache_byte_accounting_stays_bounded():
    adapter, runtime, _, _, _, _ = device_runtime_for_tiny_glm(
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        device_cache_budget_bytes=60,
    )

    for expert_id in range(4):
        with provider.request_experts(glm_expert_demands(adapter, expert_id)):
            stats = provider.stats()
            assert stats["device_cache_bytes"] <= 60
            assert stats["device_cache_peak_bytes"] <= 60
            assert stats["device_cache_active_leases"] == 3

    final_stats = provider.stats()
    assert final_stats["device_cache_bytes"] == 60
    assert final_stats["device_cache_peak_bytes"] == 60
    assert final_stats["device_cache_budget_bytes"] == 60
    assert final_stats["device_cache_evictions"] == 6
    assert final_stats["device_cache_active_leases"] == 0
    assert final_stats["scratch_bytes"] == 60


def test_vllm_glm_prefetch_receives_exact_router_expert_union():
    manifest = tiny_glm_manifest()
    payload = manifest_payload(manifest)
    adapter = Glm53Adapter(manifest)
    records = {record["unit_id"]: record for record in manifest["records"]}
    create_tensor, _ = payload_tensor_factory(records, payload)

    class PrefetchRecordingProvider:
        def __init__(self, provider):
            self._provider = provider
            self.enable_prefetch = True
            self.prefetch_unit_ids = []
            self.request_unit_ids = []

        def prefetch_experts(self, demands):
            self.prefetch_unit_ids.append([demand.unit_id for demand in demands])

        def request_experts(self, demands):
            self.request_unit_ids.append([demand.unit_id for demand in demands])
            return self._provider.request_experts(demands)

    provider = PrefetchRecordingProvider(
        LazyGlm53TensorProvider(
            adapter=adapter,
            tensor_factory=create_tensor,
            max_resident_units=18,
        )
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(8642)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.float32)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.125, 0.875]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[1, 7], [7, 9]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, tuple(range(10))),
    )
    result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    expected_unit_ids = set()
    for expert_id in (1, 7, 9):
        expected_unit_ids.update(adapter.expert_unit_ids(expert_id))
    assert set(provider.prefetch_unit_ids[0]) == expected_unit_ids
    assert len(provider.prefetch_unit_ids) == 1
    first_request_unit_ids = set()
    second_request_unit_ids = set()
    for expert_id in (1, 7):
        first_request_unit_ids.update(adapter.expert_unit_ids(expert_id))
    for expert_id in (7, 9):
        second_request_unit_ids.update(adapter.expert_unit_ids(expert_id))
    assert [set(unit_ids) for unit_ids in provider.request_unit_ids] == [
        first_request_unit_ids,
        second_request_unit_ids,
    ]
    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)


def test_vllm_glm_prefetch_failure_is_output_and_flow_neutral():
    manifest = tiny_glm_manifest()
    payload = manifest_payload(manifest)
    adapter = Glm53Adapter(manifest)
    records = {record["unit_id"]: record for record in manifest["records"]}
    create_tensor, _ = payload_tensor_factory(records, payload)

    class FailingPrefetchProvider:
        def __init__(self, provider):
            self._provider = provider
            self.enable_prefetch = True
            self.prefetch_calls = 0
            self.request_calls = 0

        def prefetch_experts(self, demands):
            self.prefetch_calls += 1
            raise RuntimeError("prefetch transport failed")

        def request_experts(self, demands):
            self.request_calls += 1
            return self._provider.request_experts(demands)

    provider = FailingPrefetchProvider(
        LazyGlm53TensorProvider(
            adapter=adapter,
            tensor_factory=create_tensor,
            max_resident_units=18,
        )
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(9753)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.float32)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.125, 0.875]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[1, 7], [7, 9]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, tuple(range(10))),
    )
    result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert provider.prefetch_calls == 1
    assert provider.request_calls == 2
    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)


def test_vllm_glm_provider_prefetch_is_opt_in_and_counts_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        device_budget_bytes=600,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(runtime=runtime, adapter=adapter)
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    hidden_states = torch.randn((2, 3), dtype=torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.125, 0.875]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [[1, 7], [7, 9]],
        dtype=torch.int32,
    )
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, tuple(range(10))),
    )

    def fail_prefetch(*args, **kwargs):
        raise RuntimeError("prefetch transport failed")

    default_result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )
    assert torch.allclose(default_result, expected, rtol=0.0, atol=0.0)
    assert provider.stats()["prefetch_successes"] == 0
    assert provider.stats()["prefetch_failures"] == 0

    provider.enable_prefetch = True
    successful_result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )
    assert torch.allclose(successful_result, expected, rtol=0.0, atol=0.0)
    assert provider.stats()["prefetch_successes"] == 1
    assert provider.stats()["prefetch_failures"] == 0

    monkeypatch.setattr(runtime, "prefetch_experts", fail_prefetch)
    failure_result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )
    assert torch.allclose(failure_result, expected, rtol=0.0, atol=0.0)
    assert provider.stats()["prefetch_successes"] == 1
    assert provider.stats()["prefetch_failures"] == 1
    assert runtime.stats()["active_leases"] == 0


def test_vllm_glm_next_layer_prefetch_runs_once_after_lease_and_scratch():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )
    recording_runtime = RecordingDeviceRuntime(runtime)

    class OrderedProvider:
        def __init__(self, provider):
            self._provider = provider
            self.enable_prefetch = True
            self.enable_next_layer_prefetch = True
            self.events = []
            self.next_layer_topk_ids = []

        def prefetch_experts(self, demands):
            self.events.append("same-layer-prefetch")
            self._provider.prefetch_experts(demands)

        def prefetch_next_layer_experts(self, layer_id, topk_ids):
            self.events.append(("next-layer-prefetch", layer_id))
            self.next_layer_topk_ids.append(topk_ids.tolist())
            self._provider.prefetch_next_layer_experts(layer_id, topk_ids)

        def request_experts(self, demands):
            self.events.append("request-experts")
            return self._provider.request_experts(demands)

        def reserve_execution_scratch(self, scratch_bytes):
            self.events.append("reserve-scratch")
            return self._provider.reserve_execution_scratch(scratch_bytes)

    provider = OrderedProvider(
        DeviceWeightRuntimeGlmTensorProvider(
            runtime=recording_runtime,
            adapter=adapter,
            enable_prefetch=True,
            enable_next_layer_prefetch=True,
            next_layer_prediction_provider=lambda layer_id, topk_ids: (0,),
        )
    )

    def execution_callback(hidden_states, topk_weights, topk_ids, experts):
        provider.events.append(("execution", tuple(sorted(experts))))
        return glm_53_tiered_execution_callback(
            hidden_states,
            topk_weights,
            topk_ids,
            experts,
        )

    provider.execution_callback = execution_callback
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(2468)
    hidden_states = torch.randn((4, 3), generator=generator, dtype=torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.25, 0.75]] * 4,
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [[1, 2], [1, 2], [3, 4], [3, 4]],
        dtype=torch.int32,
    )
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, tuple(range(10))),
    )

    try:
        result = method.apply(
            layer=FakeRoutedExpertsLayer(),
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

        assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
        assert [
            event[0] if isinstance(event, tuple) else event for event in provider.events
        ] == [
            "same-layer-prefetch",
            "reserve-scratch",
            "request-experts",
            "reserve-scratch",
            "next-layer-prefetch",
            "execution",
            "request-experts",
            "reserve-scratch",
            "execution",
        ]
        assert provider.events[4] == ("next-layer-prefetch", 3)
        assert provider.next_layer_topk_ids == [[[1, 2], [1, 2], [3, 4], [3, 4]]]
        assert recording_runtime.prefetch_calls == [((1, 2, 3, 4), 3)]
        assert recording_runtime.submit_calls == [((0,), 4)]
        assert runtime.stats()["active_leases"] == 0
    finally:
        runtime.close_prefetches(wait=True)


def test_vllm_glm_next_layer_prefetch_does_not_wait_before_execution_callback():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        layer_ids=(3, 77),
        device_budget_bytes=1000,
    )

    class PendingPrefetchFuture:
        def __init__(self):
            self.result_calls = 0

        def done(self):
            return False

        def result(self, timeout=None):
            self.result_calls += 1
            raise AssertionError("next-layer prefetch future was awaited")

        def add_done_callback(self, callback):
            pass

    class PendingSubmitRuntime:
        def __init__(self, runtime):
            self._runtime = runtime
            self.prefetch_calls = []
            self.submit_calls = []
            self.futures = []

        def acquire_experts(self, selected_expert_ids, *, layer_id=None):
            return self._runtime.acquire_experts(
                selected_expert_ids,
                layer_id=layer_id,
            )

        def prefetch_experts(self, selected_expert_ids, *, layer_id=None):
            self.prefetch_calls.append((tuple(selected_expert_ids), layer_id))
            return self._runtime.prefetch_experts(
                selected_expert_ids,
                layer_id=layer_id,
            )

        def submit_prefetch_experts(self, selected_expert_ids, *, layer_id=None):
            self.submit_calls.append((tuple(selected_expert_ids), layer_id))
            future = PendingPrefetchFuture()
            self.futures.append(future)
            return future

        def reserve_scratch(self, scratch_bytes):
            return self._runtime.reserve_scratch(scratch_bytes)

        def stats(self):
            return self._runtime.stats()

    pending_runtime = PendingSubmitRuntime(runtime)
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=pending_runtime,
        adapter=adapter,
        enable_next_layer_prefetch=True,
        next_layer_prediction_provider=lambda layer_id, topk_ids: (0, 5),
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)

    def execution_callback(hidden_states, topk_weights, topk_ids, experts):
        assert pending_runtime.futures
        assert pending_runtime.futures[0].result_calls == 0
        assert not pending_runtime.futures[0].done()
        return glm_53_tiered_execution_callback(
            hidden_states,
            topk_weights,
            topk_ids,
            experts,
        )

    provider.execution_callback = execution_callback
    generator = torch.Generator().manual_seed(1357)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.125, 0.875]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, tuple(range(10))),
    )

    result = method.apply(
        layer=SimpleNamespace(layer_name="model.layers.77.mlp.experts"),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
    assert pending_runtime.prefetch_calls == []
    assert pending_runtime.submit_calls == [((0, 5), 3)]
    assert provider.stats()["next_layer_prefetch_successes"] == 1
    assert provider.stats()["next_layer_prefetch_failures"] == 0
    assert [future.result_calls for future in pending_runtime.futures] == [0]
    assert all(not future.done() for future in pending_runtime.futures)
    assert runtime.stats()["active_leases"] == 0


def test_vllm_glm_previous_token_predictor_returns_multiple_future_layers():
    predictor = TieredGlmPreviousTokenPredictor()
    previous_layer_selections = {
        3: torch.tensor([[1, 2]], dtype=torch.int32),
        4: torch.tensor([[3, 4]], dtype=torch.int32),
        5: torch.tensor([[5, 6]], dtype=torch.int32),
    }
    for layer_id, topk_ids in previous_layer_selections.items():
        predictor.predict(layer_id, topk_ids, depth=3)

    predictions = predictor.predict(
        77,
        torch.tensor([[7, 8]], dtype=torch.int32),
        depth=3,
    )

    assert predictions == (
        (3, frozenset({1, 2})),
        (4, frozenset({3, 4})),
        (5, frozenset({5, 6})),
    )


def test_vllm_glm_current_router_predictor_returns_multiple_future_layers():
    predictor = TieredGlmCurrentRouterPredictor()

    predictions = predictor.predict(
        3,
        torch.tensor([[1, 2], [2, 1]], dtype=torch.int32),
        depth=4,
    )

    assert predictions == (
        (4, frozenset({1, 2})),
        (5, frozenset({1, 2})),
        (6, frozenset({1, 2})),
        (7, frozenset({1, 2})),
    )
    assert predictor.predict(77, torch.empty((0, 2)), depth=2) == (None, None)


def test_vllm_glm_current_router_predictor_bounds_batch_union_to_routing_width():
    predictor = TieredGlmCurrentRouterPredictor()

    predictions = predictor.predict(
        3,
        torch.tensor(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7],
            ],
            dtype=torch.int32,
        ),
        depth=1,
    )

    assert predictions == ((4, frozenset({2, 3, 4, 5})),)


def test_vllm_glm_previous_token_layer_predictor_fires_on_every_layer():
    predictor = TieredGlmPreviousTokenLayerPredictor()
    first_token_selections = {
        3: torch.tensor([[1, 2]], dtype=torch.int32),
        4: torch.tensor([[3, 4]], dtype=torch.int32),
        5: torch.tensor([[5, 6]], dtype=torch.int32),
        6: torch.tensor([[7, 8]], dtype=torch.int32),
    }
    for layer_id, topk_ids in first_token_selections.items():
        assert predictor.predict(layer_id, topk_ids, depth=2) == (None, None)

    assert predictor.predict(
        3,
        torch.tensor([[9, 10]], dtype=torch.int32),
        depth=2,
    ) == (
        (4, frozenset({3, 4})),
        (5, frozenset({5, 6})),
    )
    assert predictor.predict(
        4,
        torch.tensor([[11, 12]], dtype=torch.int32),
        depth=2,
    ) == (
        (5, frozenset({5, 6})),
        (6, frozenset({7, 8})),
    )


def test_vllm_glm_previous_token_layer_predictor_uses_layer_history_after_first_pass():
    predictor = TieredGlmPreviousTokenLayerPredictor()

    assert predictor.predict(
        3,
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        depth=1,
    ) == (None,)
    assert predictor.predict(
        4,
        torch.tensor([[5, 6]], dtype=torch.int32),
        depth=1,
    ) == (None,)

    assert predictor.predict(
        3,
        torch.tensor([[7, 8]], dtype=torch.int32),
        depth=1,
    ) == ((4, frozenset({5, 6})),)


def test_vllm_glm_previous_token_layer_predictor_aggregates_all_batch_rows():
    predictor = TieredGlmPreviousTokenLayerPredictor()
    predictor.predict(
        4,
        torch.tensor([[1, 2], [1, 2], [3, 4]], dtype=torch.int32),
        depth=1,
    )

    assert predictor.predict(
        3,
        torch.tensor([[5, 6]], dtype=torch.int32),
        depth=1,
    ) == ((4, frozenset({1, 2})),)


def test_vllm_glm_previous_token_layer_predictor_gates_low_confidence_history():
    predictor = TieredGlmPreviousTokenLayerPredictor()
    predictor.predict(
        4,
        torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            dtype=torch.int32,
        ),
        depth=1,
    )

    assert predictor.predict(
        3,
        torch.tensor([[9, 10]], dtype=torch.int32),
        depth=1,
    ) == (None,)


def test_vllm_glm_provider_uses_previous_token_layer_predictions_by_default():
    class PendingSubmitRuntime:
        def __init__(self):
            self.submit_calls = []

        def submit_prefetch_experts(self, selected_expert_ids, *, layer_id=None):
            self.submit_calls.append((tuple(selected_expert_ids), layer_id))
            return Future()

        def stats(self):
            return {}

    runtime = PendingSubmitRuntime()
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=object(),
        enable_next_layer_prefetch=True,
        prefetch_depth=3,
    )

    first_token_selections = {
        3: torch.tensor([[1, 2]], dtype=torch.int32),
        4: torch.tensor([[3, 4]], dtype=torch.int32),
        5: torch.tensor([[5, 6]], dtype=torch.int32),
        6: torch.tensor([[7, 8]], dtype=torch.int32),
    }
    for layer_id, topk_ids in first_token_selections.items():
        provider.prefetch_next_layer_experts(layer_id, topk_ids)
        assert not runtime.submit_calls

    provider.prefetch_next_layer_experts(
        3,
        torch.tensor([[9, 10]], dtype=torch.int32),
    )

    assert runtime.submit_calls == [
        ((3, 4), 4),
        ((5, 6), 5),
        ((7, 8), 6),
    ]


def test_vllm_glm_prefetch_queue_replaces_stale_predictions():
    class PendingSubmitRuntime:
        def __init__(self):
            self.submit_calls = []

        def submit_prefetch_experts(self, selected_expert_ids, *, layer_id=None):
            self.submit_calls.append((tuple(selected_expert_ids), layer_id))
            return Future()

        def stats(self):
            return {}

    runtime = PendingSubmitRuntime()
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=object(),
        enable_next_layer_prefetch=True,
        prefetch_depth=3,
    )
    provider._next_layer_predictor = TieredGlmCurrentRouterPredictor()

    original_submit_next_prefetch = provider._submit_next_prefetch
    provider._submit_next_prefetch = lambda: None
    try:
        provider.prefetch_next_layer_experts(
            3,
            torch.tensor([[1, 2]], dtype=torch.int32),
        )
        assert list(provider._prefetch_queue) == [
            (4, (1, 2)),
            (5, (1, 2)),
            (6, (1, 2)),
        ]

        provider.prefetch_next_layer_experts(
            4,
            torch.tensor([[3, 4]], dtype=torch.int32),
        )
        assert list(provider._prefetch_queue) == [
            (5, (3, 4)),
            (6, (3, 4)),
            (7, (3, 4)),
        ]
    finally:
        provider._submit_next_prefetch = original_submit_next_prefetch

    stats = provider.stats()
    assert stats["next_layer_prefetch_cancellations"] == 3
    assert stats["next_layer_prefetch_queued"] == 3
    assert stats["next_layer_prefetch_active"] == 0
    assert runtime.submit_calls == []


def test_vllm_glm_prefetch_depth_bounds_submissions_to_four_layers():
    class PendingSubmitRuntime:
        def __init__(self):
            self.submit_calls = []
            self.futures = []

        def submit_prefetch_experts(self, selected_expert_ids, *, layer_id=None):
            selected_expert_ids = tuple(selected_expert_ids)
            self.submit_calls.append((selected_expert_ids, layer_id))
            future = Future()
            self.futures.append(future)
            return future

        def stats(self):
            return {}

    runtime = PendingSubmitRuntime()
    topk_ids = torch.tensor([[1, 2]], dtype=torch.int32)
    with pytest.raises(ValueError, match="prefetch_depth must be an integer"):
        DeviceWeightRuntimeGlmTensorProvider(
            runtime=runtime,
            adapter=object(),
            prefetch_depth=5,
        )

    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=object(),
        enable_next_layer_prefetch=True,
        prefetch_depth=4,
        next_layer_prediction_provider=lambda layer_id, topk_ids: (layer_id - 4,),
    )

    provider.prefetch_next_layer_experts(3, topk_ids)
    assert len(runtime.submit_calls) == 4
    provider.prefetch_next_layer_experts(3, topk_ids)
    assert len(runtime.submit_calls) == 4
    while runtime.futures:
        future = runtime.futures.pop(0)
        future.set_result(None)

    assert len(runtime.submit_calls) == 4
    assert {layer_id for _, layer_id in runtime.submit_calls} == {4, 5, 6, 7}
    assert provider.stats()["next_layer_prefetch_successes"] == 4
    assert provider.stats()["next_layer_prefetch_failures"] == 0


def test_vllm_glm_drain_prefetches_drains_active_and_queued_work():
    class DrainingRuntime:
        def __init__(self):
            self.futures = []
            self.submit_calls = []
            self.drain_calls = 0
            self.close_calls = []

        def submit_expert_cache_prefetch(self, selected_expert_ids, *, layer_id=None):
            selected_expert_ids = tuple(selected_expert_ids)
            self.submit_calls.append((selected_expert_ids, layer_id))
            future = Future()
            self.futures.append(future)
            return future

        def drain_prefetches(self):
            self.drain_calls += 1
            while self.futures:
                self.futures.pop(0).set_result(None)

        def close_prefetches(self, *, wait=True):
            self.close_calls.append(wait)

        def stats(self):
            return {"expert_cache_budget_bytes": 1000}

    runtime = DrainingRuntime()
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=object(),
        enable_next_layer_prefetch=True,
        prefetch_depth=2,
        next_layer_prediction_provider=lambda layer_id, topk_ids: (1,)
        if layer_id == 4
        else (2,),
    )
    original_submit_next_prefetch = provider._submit_next_prefetch
    provider._submit_next_prefetch = lambda: None
    try:
        provider.prefetch_next_layer_experts(
            3,
            torch.tensor([[1, 2]], dtype=torch.int32),
        )
    finally:
        provider._submit_next_prefetch = original_submit_next_prefetch

    assert provider.stats()["next_layer_prefetch_queued"] == 2
    provider.drain_prefetches()

    stats = provider.stats()
    assert runtime.submit_calls == [((1,), 4), ((2,), 5)]
    assert runtime.drain_calls == 1
    assert stats["next_layer_prefetch_active"] == 0
    assert stats["next_layer_prefetch_queued"] == 0
    assert runtime.close_calls == []


def test_vllm_glm_drain_prefetches_samples_zero_staging_and_keeps_provider_usable():
    adapter, base_runtime, _, _, _, store = device_runtime_for_tiny_glm(
        layer_ids=(3,),
        expert_cache_budget_bytes=1000,
    )

    class DrainingRuntime:
        def __init__(self):
            self.futures = []
            self.staging_bytes = 0
            self.close_calls = []

        def submit_expert_cache_prefetch(self, selected_expert_ids, *, layer_id=None):
            future = Future()
            self.futures.append(future)
            self.staging_bytes += 12
            return future

        def drain_prefetches(self):
            while self.futures:
                self.futures.pop(0).set_result(None)
            self.staging_bytes = 0

        def acquire_experts(self, selected_expert_ids, *, layer_id=None):
            return base_runtime.acquire_experts(
                selected_expert_ids,
                layer_id=layer_id,
            )

        def reserve_scratch(self, scratch_bytes):
            return base_runtime.reserve_scratch(scratch_bytes)

        def stats(self):
            stats = dict(base_runtime.stats())
            stats["expert_cache_staging_bytes"] = self.staging_bytes
            return stats

        def close_prefetches(self, *, wait=True):
            self.close_calls.append(wait)

    runtime = DrainingRuntime()
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        enable_next_layer_prefetch=True,
        prefetch_depth=1,
        dequant_cache_budget_bytes=100,
        next_layer_prediction_provider=lambda layer_id, topk_ids: (1,),
    )
    runtime.submit_expert_cache_prefetch((0,), layer_id=77)
    assert provider.stats()["expert_cache_staging_bytes"] == 12

    provider.prefetch_next_layer_experts(
        3,
        torch.tensor([[1, 2]], dtype=torch.int32),
    )
    provider.drain_prefetches()

    stats = provider.stats()
    assert stats["expert_cache_staging_bytes"] == 0
    assert stats["next_layer_prefetch_active"] == 0
    assert stats["next_layer_prefetch_queued"] == 0

    store.reads.clear()
    with provider.request_experts(glm_expert_demands(adapter, expert_id=1)):
        pass

    stats = provider.stats()
    assert store.reads
    assert stats["active_leases"] == 0
    assert stats["dequant_cache_bytes"] > 0
    assert stats["dequant_cache_active_leases"] == 0
    assert runtime.close_calls == []


def test_vllm_glm_multi_layer_prefetch_selects_host_cache_when_configured():
    adapter, runtime, _, _, _, _ = device_runtime_for_tiny_glm(
        layer_ids=(3, 4, 5, 77),
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
        expert_cache_budget_bytes=1000,
        prefetch_depth=3,
    )

    class RecordingHostCacheRuntime:
        def __init__(self, cache_budget_bytes):
            self.cache_budget_bytes = cache_budget_bytes
            self.host_cache_calls = []
            self.device_calls = []
            self.host_cache_futures = []

        def submit_expert_cache_prefetch(self, selected_expert_ids, *, layer_id=None):
            selected_expert_ids = tuple(selected_expert_ids)
            self.host_cache_calls.append((selected_expert_ids, layer_id))
            future = runtime.submit_expert_cache_prefetch(
                selected_expert_ids,
                layer_id=layer_id,
            )
            self.host_cache_futures.append(future)
            return future

        def submit_prefetch_experts(self, selected_expert_ids, *, layer_id=None):
            selected_expert_ids = tuple(selected_expert_ids)
            self.device_calls.append((selected_expert_ids, layer_id))
            return runtime.submit_prefetch_experts(
                selected_expert_ids,
                layer_id=layer_id,
            )

        def stats(self):
            return runtime.stats()

    recording_runtime = RecordingHostCacheRuntime(None)
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=recording_runtime,
        adapter=object(),
        enable_next_layer_prefetch=True,
        prefetch_depth=3,
        next_layer_prediction_provider=lambda layer_id, topk_ids: (0, 9),
    )

    provider.prefetch_next_layer_experts(
        77,
        torch.tensor([[1, 2]], dtype=torch.int32),
    )
    assert recording_runtime.host_cache_calls == [
        ((0, 9), 3),
        ((0, 9), 4),
        ((0, 9), 5),
    ]
    assert recording_runtime.device_calls == []
    for future in recording_runtime.host_cache_futures:
        future.result()
    runtime_stats = runtime.stats()
    assert runtime_stats["expert_cache_budget_bytes"] == 1000
    assert runtime_stats["expert_cache_bytes"] <= 1000
    assert runtime_stats["resident_units"] == 0
    assert runtime_stats["scratch_bytes"] == 0
    assert runtime_stats["active_leases"] == 0
    assert provider.stats()["next_layer_prefetch_successes"] == 3
    assert provider.stats()["next_layer_prefetch_failures"] == 0


def test_vllm_glm_multi_layer_prefetch_falls_back_to_device_without_host_cache():
    class HostCacheRuntime:
        def __init__(self, cache_budget_bytes):
            self.cache_budget_bytes = cache_budget_bytes
            self.host_cache_calls = []
            self.device_calls = []

        def submit_expert_cache_prefetch(self, selected_expert_ids, *, layer_id=None):
            selected_expert_ids = tuple(selected_expert_ids)
            self.host_cache_calls.append((selected_expert_ids, layer_id))
            return Future()

        def submit_prefetch_experts(self, selected_expert_ids, *, layer_id=None):
            selected_expert_ids = tuple(selected_expert_ids)
            self.device_calls.append((selected_expert_ids, layer_id))
            return Future()

        def stats(self):
            return {"expert_cache_budget_bytes": self.cache_budget_bytes}

    runtime = HostCacheRuntime(None)
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=object(),
        enable_next_layer_prefetch=True,
        prefetch_depth=3,
        next_layer_prediction_provider=lambda layer_id, topk_ids: (0, 9),
    )

    provider.prefetch_next_layer_experts(
        77,
        torch.tensor([[1, 2]], dtype=torch.int32),
    )

    assert runtime.device_calls == [((0, 9), 3), ((0, 9), 4), ((0, 9), 5)]
    assert runtime.host_cache_calls == []
    assert provider.stats()["next_layer_prefetch_successes"] == 3
    assert provider.stats()["next_layer_prefetch_failures"] == 0


def test_vllm_glm_host_cache_prefetch_failure_is_output_neutral():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        layer_ids=(3, 4, 5, 77),
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
        expert_cache_budget_bytes=1000,
    )

    class FailingHostCacheRuntime:
        def __init__(self, runtime):
            self._runtime = runtime
            self.host_cache_calls = 0
            self.device_calls = 0

        def acquire_experts(self, selected_expert_ids, *, layer_id=None):
            return self._runtime.acquire_experts(
                selected_expert_ids,
                layer_id=layer_id,
            )

        def prefetch_experts(self, selected_expert_ids, *, layer_id=None):
            return self._runtime.prefetch_experts(
                selected_expert_ids,
                layer_id=layer_id,
            )

        def submit_expert_cache_prefetch(self, selected_expert_ids, *, layer_id=None):
            self.host_cache_calls += 1
            raise RuntimeError("host-cache prefetch failed")

        def submit_prefetch_experts(self, selected_expert_ids, *, layer_id=None):
            self.device_calls += 1
            return self._runtime.submit_prefetch_experts(
                selected_expert_ids,
                layer_id=layer_id,
            )

        def reserve_scratch(self, scratch_bytes):
            return self._runtime.reserve_scratch(scratch_bytes)

        def stats(self):
            return self._runtime.stats()

    failing_runtime = FailingHostCacheRuntime(runtime)
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=failing_runtime,
        adapter=adapter,
        enable_next_layer_prefetch=True,
        prefetch_depth=3,
        next_layer_prediction_provider=lambda layer_id, topk_ids: (0, 9),
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(6802)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.125, 0.875]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, tuple(range(10)), layer_id=77),
    )

    result = method.apply(
        layer=SimpleNamespace(layer_name="model.layers.77.mlp.experts"),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
    assert failing_runtime.host_cache_calls == 3
    assert failing_runtime.device_calls == 0
    assert provider.stats()["next_layer_prefetch_successes"] == 0
    assert provider.stats()["next_layer_prefetch_failures"] == 3
    assert runtime.stats()["active_leases"] == 0


def test_vllm_glm_multi_layer_prefetch_failure_is_output_neutral(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        layer_ids=(3, 4, 5, 77),
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        enable_next_layer_prefetch=True,
        prefetch_depth=3,
        next_layer_prediction_provider=lambda layer_id, topk_ids: (0, 9),
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(8642)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.125, 0.875]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, tuple(range(10)), layer_id=77),
    )

    def fail_prefetch(*args, **kwargs):
        raise RuntimeError("multi-layer prefetch failed")

    monkeypatch.setattr(runtime, "submit_prefetch_experts", fail_prefetch)
    result = method.apply(
        layer=SimpleNamespace(layer_name="model.layers.77.mlp.experts"),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
    assert provider.stats()["next_layer_prefetch_successes"] == 0
    assert provider.stats()["next_layer_prefetch_failures"] == 3
    assert runtime.stats()["active_leases"] == 0


def test_vllm_glm_multi_layer_prefetch_preserves_exact_output_without_blocking():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        layer_ids=(3, 4, 5, 77),
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )
    baseline_adapter, baseline_runtime, _, _, _, _ = device_runtime_for_tiny_glm(
        layer_ids=(3, 4, 5, 77),
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )

    class PendingSubmitRuntime:
        def __init__(self, runtime):
            self._runtime = runtime
            self.submit_calls = []
            self.futures = []

        def acquire_experts(self, selected_expert_ids, *, layer_id=None):
            return self._runtime.acquire_experts(
                selected_expert_ids,
                layer_id=layer_id,
            )

        def prefetch_experts(self, selected_expert_ids, *, layer_id=None):
            return self._runtime.prefetch_experts(
                selected_expert_ids,
                layer_id=layer_id,
            )

        def submit_prefetch_experts(self, selected_expert_ids, *, layer_id=None):
            self.submit_calls.append((tuple(selected_expert_ids), layer_id))
            future = Future()
            self.futures.append(future)
            return future

        def reserve_scratch(self, scratch_bytes):
            return self._runtime.reserve_scratch(scratch_bytes)

        def stats(self):
            return self._runtime.stats()

    pending_runtime = PendingSubmitRuntime(runtime)
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=pending_runtime,
        adapter=adapter,
        enable_next_layer_prefetch=True,
        prefetch_depth=3,
        next_layer_prediction_provider=lambda layer_id, topk_ids: (0, 9),
    )
    baseline_provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=baseline_runtime,
        adapter=baseline_adapter,
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    baseline_method = TieredGlm53MoEMethod(
        SimpleNamespace(),
        provider=baseline_provider,
    )

    def execution_callback(hidden_states, topk_weights, topk_ids, experts):
        assert pending_runtime.futures
        assert all(not future.done() for future in pending_runtime.futures)
        return glm_53_tiered_execution_callback(
            hidden_states,
            topk_weights,
            topk_ids,
            experts,
        )

    provider.execution_callback = execution_callback
    generator = torch.Generator().manual_seed(9753)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.125, 0.875]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, tuple(range(10)), layer_id=77),
    )

    result = method.apply(
        layer=SimpleNamespace(layer_name="model.layers.77.mlp.experts"),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )
    assert all(not future.done() for future in pending_runtime.futures)
    while pending_runtime.futures:
        future = pending_runtime.futures.pop(0)
        future.set_result(None)

    baseline_result = baseline_method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
    assert torch.allclose(result, baseline_result, rtol=0.0, atol=0.0)
    assert pending_runtime.submit_calls == [((0, 9), 3), ((0, 9), 4), ((0, 9), 5)]
    assert provider.stats()["next_layer_prefetch_successes"] == 3
    assert provider.stats()["next_layer_prefetch_failures"] == 0
    assert runtime.stats()["active_leases"] == 0
    assert baseline_runtime.stats()["active_leases"] == 0


def test_vllm_glm_current_router_predictor_fires_on_every_sparse_layer():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        layer_ids=(3, 77),
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
        prefetch_depth=4,
    )
    recording_runtime = RecordingDeviceRuntime(runtime)
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=recording_runtime,
        adapter=adapter,
        enable_next_layer_prefetch=True,
    )
    provider._next_layer_predictor = TieredGlmCurrentRouterPredictor()
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    topk_weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    try:
        outputs = []
        for token_id, layer_id, layer_topk_ids in (
            (0, 3, torch.tensor([[1, 2]], dtype=torch.int32)),
            (0, 77, torch.tensor([[3, 4]], dtype=torch.int32)),
            (1, 3, torch.tensor([[5, 6]], dtype=torch.int32)),
            (1, 77, torch.tensor([[7, 8]], dtype=torch.int32)),
        ):
            generator = torch.Generator().manual_seed(1000 + 100 * token_id + layer_id)
            hidden_states = torch.randn(
                (1, 3), generator=generator, dtype=torch.bfloat16
            )
            expected = all_resident_reference(
                hidden_states,
                topk_weights,
                layer_topk_ids,
                all_resident_experts(
                    adapter,
                    records,
                    payload,
                    tuple(range(10)),
                    layer_id=layer_id,
                ),
            )
            outputs.append(
                method.apply(
                    layer=SimpleNamespace(
                        layer_name=f"model.layers.{layer_id}.mlp.experts"
                    ),
                    x=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=layer_topk_ids,
                    shared_experts=None,
                    shared_experts_input=None,
                )
            )
            assert torch.allclose(outputs[-1], expected, rtol=0.0, atol=0.0)

        assert recording_runtime.prefetch_calls == []
        assert recording_runtime.submit_calls == [
            ((1, 2), 4),
            ((3, 4), 3),
            ((5, 6), 4),
            ((7, 8), 3),
        ]
        assert provider.stats()["next_layer_prefetch_successes"] == 4
        assert provider.stats()["next_layer_prefetch_failures"] == 0
        assert runtime.stats()["active_leases"] == 0
        assert provider._active_prefetches == {}
        assert provider._active_prefetch_layer_ids == {}
    finally:
        runtime.close_prefetches(wait=True)


def test_vllm_glm_current_router_prediction_is_output_neutral():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        layer_ids=(3, 77),
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )
    baseline_adapter, baseline_runtime, _, _, _, _ = device_runtime_for_tiny_glm(
        layer_ids=(3, 77),
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        enable_next_layer_prefetch=True,
    )
    provider._next_layer_predictor = TieredGlmCurrentRouterPredictor()
    baseline_provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=baseline_runtime,
        adapter=baseline_adapter,
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    baseline_method = TieredGlm53MoEMethod(
        SimpleNamespace(),
        provider=baseline_provider,
    )
    generator = torch.Generator().manual_seed(3579)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.125, 0.875]],
        dtype=torch.float32,
    )
    previous_topk_ids = torch.tensor([[0, 5], [5, 0]], dtype=torch.int32)
    current_topk_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        current_topk_ids,
        all_resident_experts(adapter, records, payload, tuple(range(10)), layer_id=77),
    )

    try:
        method.apply(
            layer=SimpleNamespace(layer_name="model.layers.3.mlp.experts"),
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=previous_topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )
        result = method.apply(
            layer=SimpleNamespace(layer_name="model.layers.77.mlp.experts"),
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=current_topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )
    finally:
        runtime.close_prefetches(wait=True)

    baseline_result = baseline_method.apply(
        layer=SimpleNamespace(layer_name="model.layers.77.mlp.experts"),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=current_topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
    assert torch.allclose(result, baseline_result, rtol=0.0, atol=0.0)
    assert provider.stats()["next_layer_prefetch_successes"] == 2
    assert provider.stats()["next_layer_prefetch_failures"] == 0
    assert runtime.stats()["active_leases"] == 0
    assert baseline_runtime.stats()["active_leases"] == 0


def test_vllm_glm_next_layer_prefetch_failure_is_neutral(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm(
        layer_ids=(3, 77),
        max_resident_experts=5,
        slot_count=30,
        device_budget_bytes=1000,
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=runtime,
        adapter=adapter,
        enable_next_layer_prefetch=True,
        next_layer_prediction_provider=lambda layer_id, topk_ids: (0, 5),
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(4680)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.125, 0.875]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, tuple(range(10)), layer_id=77),
    )

    def fail_prefetch(*args, **kwargs):
        raise RuntimeError("next-layer prefetch failed")

    monkeypatch.setattr(runtime, "submit_prefetch_experts", fail_prefetch)
    result = method.apply(
        layer=SimpleNamespace(layer_name="model.layers.77.mlp.experts"),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
    assert provider.stats()["next_layer_prefetch_successes"] == 0
    assert provider.stats()["next_layer_prefetch_failures"] == 1
    assert runtime.stats()["active_leases"] == 0


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


def test_vllm_glm_provider_uses_one_batched_expert_acquisition():
    adapter, runtime, _, records, payload, _ = device_runtime_for_tiny_glm()
    recording_runtime = RecordingDeviceRuntime(runtime)
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=recording_runtime,
        adapter=adapter,
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(8642)
    hidden_states = torch.randn((1, 3), generator=generator, dtype=torch.bfloat16)
    topk_weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    topk_ids = torch.tensor([[1, 7, 9]], dtype=torch.int32)
    expected = all_resident_reference(
        hidden_states,
        topk_weights,
        topk_ids,
        all_resident_experts(adapter, records, payload, tuple(range(10))),
    )
    resident_expert_sets = []

    def execution_callback(hidden_states, topk_weights, topk_ids, experts):
        resident_expert_sets.append(experts)
        lease_set = runtime.lease_sets[0]
        assert not lease_set.released
        for expert_id, projections in experts.items():
            lease = next(
                current_lease
                for current_lease in lease_set.leases
                if current_lease.expert_id == expert_id
            )
            assert set(projections) == {"gate_proj", "up_proj", "down_proj"}
            for projection in projections.values():
                assert (
                    projection.weight is lease.weight_tensor(projection.weight_unit_id)
                )
                assert projection.scale is lease.scale_tensor(projection.weight_unit_id)
                assert projection.scale_unit_id == lease.linked_scale_unit_id(
                    projection.weight_unit_id
                )
        return glm_53_tiered_execution_callback(
            hidden_states,
            topk_weights,
            topk_ids,
            experts,
        )

    provider.execution_callback = execution_callback

    result = method.apply(
        layer=FakeRoutedExpertsLayer(),
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
    assert recording_runtime.prefetch_calls == []
    assert recording_runtime.acquire_experts_calls == [((1, 7, 9), 3)]
    assert len(resident_expert_sets) == 1
    lease_set = runtime.lease_sets[0]
    assert recording_runtime.events == [("acquire-experts", (1, 7, 9), 3)]
    assert lease_set.released
    assert runtime.stats()["active_leases"] == 0


def test_vllm_glm_batched_acquisition_failure_has_no_partial_leases():
    adapter, runtime, _, _, _, _ = device_runtime_for_tiny_glm()

    class FailingBatchAcquisitionRuntime:
        def __init__(self, runtime):
            self._runtime = runtime
            self.acquire_experts_calls = []

        def acquire_experts(self, selected_expert_ids, *, layer_id=None):
            self.acquire_experts_calls.append((tuple(selected_expert_ids), layer_id))
            raise RuntimeError("batched acquisition failed")

        def stats(self):
            return self._runtime.stats()

    hook = TieredWeightsResidencyHook(adapter=adapter)
    hook.enable_for_model("glm-5.3")
    demands = hook.build_demands_from_router(
        router_topk_ids=[[1, 7, 9]],
        layer_prefix="model.layers.3.mlp.experts",
        target_view_id="gpu-vram",
    )
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=FailingBatchAcquisitionRuntime(runtime),
        adapter=adapter,
    )

    with pytest.raises(RuntimeError, match="batched acquisition failed"):
        provider.request_experts(demands)

    assert provider._runtime.acquire_experts_calls == [((1, 7, 9), 3)]
    assert runtime.stats()["active_leases"] == 0


def test_vllm_glm_batched_acquire_failure_releases_leases_and_cache_leases(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter, runtime, _, _, _, _ = device_runtime_for_tiny_glm()
    recording_runtime = RecordingDeviceRuntime(runtime)
    provider = DeviceWeightRuntimeGlmTensorProvider(
        runtime=recording_runtime,
        adapter=adapter,
        device_cache_budget_bytes=30,
    )

    with provider.request_experts(glm_expert_demands(adapter, expert_id=1)):
        pass

    assert runtime.lease_sets[0].released
    assert provider.stats()["device_cache_active_leases"] == 0

    demands = [
        *glm_expert_demands(adapter, expert_id=1),
        *glm_expert_demands(adapter, expert_id=7),
    ]

    def fail_build_experts(leases):
        raise RuntimeError("synthetic GLM projection build failure")

    monkeypatch.setattr(provider, "_build_experts", fail_build_experts)
    with pytest.raises(RuntimeError, match="synthetic GLM projection build failure"):
        provider.request_experts(demands)

    assert recording_runtime.acquire_experts_calls == [((1,), 3), ((7,), 3)]
    assert runtime.lease_sets[1].released
    stats = provider.stats()
    assert stats["device_cache_active_leases"] == 0
    assert stats["active_leases"] == 0
    assert (3, 1, "gate_proj") in provider._device_cache


def test_tiered_expert_weight_names_accept_loader_prefix_forms():
    from vllm.model_executor.models.deepseek_v2 import _is_tiered_routed_expert_weight

    layer_ids = {10}
    assert _is_tiered_routed_expert_weight(
        "model.layers.10.mlp.experts.0.gate_proj.weight", layer_ids
    )
    assert _is_tiered_routed_expert_weight(
        "layers.10.mlp.experts.0.gate_proj.weight", layer_ids
    )
    assert _is_tiered_routed_expert_weight(
        "model.layers.10.mlp.experts.0.down_proj.weight_scale_inv", layer_ids
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

    configure_glm_53_tiered_experts(provider, adapter=adapter)
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
        assert captured_factory_kwargs["routed_experts_args"] == {
            "provider": provider,
            "adapter": adapter,
        }
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


def test_clear_glm_53_tiered_experts_closes_prefetch_workers():
    class ClosingProvider:
        def __init__(self):
            self.closed = False

        def close_prefetches(self, *, wait: bool = True) -> None:
            self.closed = True
            assert wait is True

    provider = ClosingProvider()
    configure_glm_53_tiered_experts(provider)
    clear_glm_53_tiered_experts()

    assert provider.closed


def tiered_route_fixture(
    row_count: int = 4,
    layer_count: int = 75,
    top_k: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = (
        torch.arange(top_k, dtype=torch.int32)
        .repeat(row_count * layer_count, 1)
        .reshape(row_count, layer_count, top_k)
    )
    weight_bits = torch.tensor(
        [
            0x00000001,
            -0x80000000,
            0x007FFFFF,
            0x00800000,
            0x00800001,
            0x3F7FFFFF,
            0x3F800001,
            0x7F7FFFFF,
        ],
        dtype=torch.int32,
    ).repeat(row_count * layer_count, 1)
    weights = weight_bits.reshape(row_count, layer_count, top_k).view(torch.float32)
    return ids, weights


def test_tiered_glm_route_controller_captures_exact_router_bits():
    ids, weights = tiered_route_fixture()
    controller = TieredGlmRouteController.capture()
    controller.start_capture_sample()
    for layer_index, layer_id in enumerate(range(3, 78)):
        layer_ids = ids[:, layer_index]
        layer_weights = weights[:, layer_index]
        returned_ids, returned_weights = controller.apply(
            layer_id,
            layer_ids,
            layer_weights,
        )
        assert returned_ids is layer_ids
        assert returned_weights is layer_weights

    captured_ids, captured_weights = controller.finish_capture_sample()

    assert captured_ids.device.type == "cpu"
    assert captured_weights.device.type == "cpu"
    assert captured_ids.dtype == torch.int32
    assert captured_weights.dtype == torch.float32
    assert captured_ids.shape == ids.shape
    assert captured_weights.shape == weights.shape
    assert captured_ids.equal(ids)
    assert captured_weights.view(torch.int32).equal(weights.view(torch.int32))


def test_tiered_glm_route_controller_replays_exact_router_bits():
    ids, weights = tiered_route_fixture()
    live_ids = (
        torch.arange(8, 16, dtype=torch.int32)
        .repeat(4, 1)
        .reshape(4, 8)
    )
    live_weights = torch.full_like(live_ids, 0.5, dtype=torch.float32)
    controller = TieredGlmRouteController.replay(ids, weights)
    for layer_index, layer_id in enumerate(range(3, 78)):
        returned_ids, returned_weights = controller.apply(
            layer_id,
            live_ids,
            live_weights,
        )
        assert returned_ids is live_ids
        assert returned_weights is live_weights
        assert live_ids.equal(ids[:, layer_index])
        assert live_weights.view(torch.int32).equal(
            weights[:, layer_index].view(torch.int32)
        )


def test_tiered_glm_replay_custom_op_has_a_fake_implementation():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        target_ids = torch.ones((1, 2), dtype=torch.int32)
        target_weights = torch.ones((1, 2), dtype=torch.float32)
        source_ids = torch.zeros((1, 2), dtype=torch.int32)
        source_weights = torch.zeros((1, 2), dtype=torch.float32)

        torch.ops.vllm.tiered_glm_replay_router(
            target_ids,
            target_weights,
            source_ids,
            source_weights,
        )

        assert target_ids.shape == (1, 2)
        assert target_weights.shape == (1, 2)


def test_tiered_glm_route_controller_stats_are_finite_and_monotonic():
    capture_controller = TieredGlmRouteController.capture()
    ids, weights = tiered_route_fixture()

    initial_capture_stats = capture_controller.stats()
    assert set(initial_capture_stats) == {
        "capture_count",
        "capture_sample_count",
        "capture_layer_count",
        "replay_count",
        "replay_layer_count",
        "apply_count",
    }
    assert all(key.endswith("_count") for key in initial_capture_stats)
    assert all(
        isinstance(value, int) and math.isfinite(value)
        for value in initial_capture_stats.values()
    )

    capture_controller.start_capture_sample()
    for layer_id in range(3, 78):
        capture_controller.apply(
            layer_id,
            ids[:, layer_id - 3],
            weights[:, layer_id - 3],
        )
    capture_controller.finish_capture_sample()
    capture_stats = capture_controller.stats()

    replay_controller = TieredGlmRouteController.replay(ids, weights)
    initial_replay_stats = replay_controller.stats()
    for layer_id in range(3, 78):
        replay_controller.apply(
            layer_id,
            ids[:, layer_id - 3],
            weights[:, layer_id - 3],
        )
    replay_stats = replay_controller.stats()

    assert capture_stats["capture_count"] == 1
    assert capture_stats["capture_sample_count"] == 1
    assert capture_stats["capture_layer_count"] == 75
    assert capture_stats["replay_count"] == 0
    assert capture_stats["apply_count"] == 75
    assert replay_stats["replay_count"] == 1
    assert replay_stats["replay_layer_count"] == 75
    assert replay_stats["apply_count"] == 75
    assert all(
        capture_stats[key] >= initial_capture_stats[key]
        for key in initial_capture_stats
    )
    assert all(
        replay_stats[key] >= initial_replay_stats[key]
        for key in initial_replay_stats
    )


def test_tiered_glm_route_controller_supports_parent_cumulative_replays():
    ids, weights = tiered_route_fixture()
    controller = TieredGlmRouteController.replay(
        ids,
        weights,
        row_count=4,
        layer_count=75,
        top_k=8,
        device="cpu",
    )

    initial_stats = controller.stats()
    for layer_id in range(3, 78):
        controller.apply(
            layer_id,
            ids[:, layer_id - 3],
            weights[:, layer_id - 3],
        )
    warmup_stats = controller.stats()

    for layer_id in range(3, 78):
        controller.apply(
            layer_id,
            ids[:, layer_id - 3],
            weights[:, layer_id - 3],
        )

    final_stats = controller.stats()
    assert final_stats["replay_count"] == 1
    assert final_stats["replay_layer_count"] == 150
    assert final_stats["apply_count"] == 150
    assert (
        warmup_stats["replay_layer_count"] - initial_stats["replay_layer_count"]
        == 75
    )
    assert warmup_stats["apply_count"] - initial_stats["apply_count"] == 75
    assert (
        final_stats["replay_layer_count"] - warmup_stats["replay_layer_count"]
        == 75
    )
    assert final_stats["apply_count"] - warmup_stats["apply_count"] == 75


def test_tiered_glm_route_controller_exposes_missing_replay_layer_count():
    ids, weights = tiered_route_fixture()
    controller = TieredGlmRouteController.replay(ids, weights)

    for layer_id in range(3, 77):
        controller.apply(
            layer_id,
            ids[:, layer_id - 3],
            weights[:, layer_id - 3],
        )

    stats = controller.stats()
    assert stats["replay_count"] == 1
    assert stats["replay_layer_count"] == 74
    assert stats["apply_count"] == 74
    assert 75 - stats["replay_layer_count"] == 1
    with pytest.raises(ValueError, match="replayed more than once"):
        controller.apply(3, ids[:, 0], weights[:, 0])


def test_tiered_glm_route_controller_rejects_invalid_replay_inputs():
    cases = {
        "row width": lambda ids, weights: (
            ids[:3],
            weights[:3],
            {},
        ),
        "layer width": lambda ids, weights: (
            ids[:, :74],
            weights[:, :74],
            {},
        ),
        "top-k": lambda ids, weights: (
            ids[..., :7],
            weights[..., :7],
            {},
        ),
        "id dtype": lambda ids, weights: (
            ids.to(torch.int64),
            weights,
            {},
        ),
        "weight dtype": lambda ids, weights: (
            ids,
            weights.to(torch.float64),
            {},
        ),
        "nonfinite weight": lambda ids, weights: (
            ids,
            weights.clone().fill_(float("nan")),
            {},
        ),
        "negative weight": lambda ids, weights: (
            ids,
            weights.clone().fill_(-1.0),
            {},
        ),
        "duplicate ids": lambda ids, weights: (
            ids.clone().fill_(0),
            weights,
            {},
        ),
    }

    for case_name, prepare_inputs in cases.items():
        with pytest.raises(ValueError, match="Tiered GLM replay"):
            ids, weights = tiered_route_fixture()
            replay_ids, replay_weights, kwargs = prepare_inputs(ids, weights)
            TieredGlmRouteController.replay(
                replay_ids,
                replay_weights,
                **kwargs,
            )


def test_tiered_glm_route_controller_rejects_invalid_live_inputs():
    ids, weights = tiered_route_fixture()
    controller = TieredGlmRouteController.replay(ids, weights)
    cases = {
        "row width": lambda ids, weights: (ids[:3], weights[:3]),
        "top-k": lambda ids, weights: (ids[:, :7], weights[:, :7]),
        "id dtype": lambda ids, weights: (
            ids.to(torch.int64),
            weights,
        ),
        "weight dtype": lambda ids, weights: (
            ids,
            weights.to(torch.float64),
        ),
        "nonfinite weight": lambda ids, weights: (
            ids,
            weights.clone().fill_(float("inf")),
        ),
        "negative weight": lambda ids, weights: (
            ids,
            weights.clone().fill_(-0.25),
        ),
        "duplicate ids": lambda ids, weights: (
            ids.clone().fill_(0),
            weights,
        ),
    }

    for case_name, prepare_inputs in cases.items():
        with pytest.raises(ValueError, match="Tiered GLM route live"):
            live_ids, live_weights = prepare_inputs(
                ids[:, 0].clone(),
                weights[:, 0].clone(),
            )
            controller.apply(3, live_ids, live_weights)


def test_tiered_glm_route_controller_fails_closed_on_layer_errors():
    ids, weights = tiered_route_fixture()
    controller = TieredGlmRouteController.capture()
    controller.start_capture_sample()

    for layer_id in (2, 78):
        with pytest.raises(ValueError, match="outside the configured layer range"):
            controller.apply(layer_id, ids[:, 0], weights[:, 0])

    controller.apply(3, ids[:, 0], weights[:, 0])
    with pytest.raises(ValueError, match="captured more than once"):
        controller.apply(3, ids[:, 0], weights[:, 0])
    with pytest.raises(RuntimeError, match="missing layers"):
        controller.finish_capture_sample()

    controller = TieredGlmRouteController.replay(ids, weights)
    controller.apply(3, ids[:, 0], weights[:, 0])
    with pytest.raises(ValueError, match="replayed more than once"):
        controller.apply(3, ids[:, 0], weights[:, 0])
    with pytest.raises(RuntimeError, match="missing layers"):
        controller._configure_replay(
            ids,
            weights,
            row_count=4,
            layer_count=75,
            top_k=8,
            device=None,
        )


def test_vllm_glm_method_reads_provider_route_controller_before_demands():
    manifest = tiny_glm_manifest()
    payload = manifest_payload(manifest)
    adapter = Glm53Adapter(manifest)
    records = {record["unit_id"]: record for record in manifest["records"]}
    create_tensor, _ = payload_tensor_factory(records, payload)
    replay_ids = torch.tensor(
        [[[1, 2]], [[3, 4]]],
        dtype=torch.int32,
    )
    replay_weights = torch.tensor(
        [[[0.25, 0.75]], [[0.125, 0.875]]],
        dtype=torch.float32,
    )
    controller = TieredGlmRouteController.replay(
        replay_ids,
        replay_weights,
        row_count=2,
        layer_count=1,
        top_k=2,
    )

    class RouteControllerProvider:
        def __init__(self, provider):
            self._provider = provider
            self.route_controller = controller
            self.request_unit_ids = []

        def request_experts(self, demands):
            self.request_unit_ids.append([demand.unit_id for demand in demands])
            return self._provider.request_experts(demands)

    provider = RouteControllerProvider(
        LazyGlm53TensorProvider(
            adapter=adapter,
            tensor_factory=create_tensor,
            max_resident_units=24,
        )
    )
    method = TieredGlm53MoEMethod(SimpleNamespace(), provider=provider)
    generator = torch.Generator().manual_seed(2468)
    hidden_states = torch.randn((2, 3), generator=generator, dtype=torch.float32)
    expected = all_resident_reference(
        hidden_states,
        replay_weights[:, 0],
        replay_ids[:, 0],
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
        topk_weights=torch.ones((2, 2), dtype=torch.float32),
        topk_ids=torch.tensor([[1, 7], [7, 9]], dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=None,
    )

    requested_unit_ids = {
        unit_id for request in provider.request_unit_ids for unit_id in request
    }
    expected_unit_ids = set()
    for expert_id in (1, 2, 3, 4):
        expected_unit_ids.update(adapter.expert_unit_ids(expert_id))
    assert requested_unit_ids == expected_unit_ids
    assert torch.allclose(result, expected, rtol=0.0, atol=0.0)
