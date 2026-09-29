# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused Tiered Weights execution path for GLM 5.3 routed experts."""

from __future__ import annotations

import re
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.models.transformers.tiered_weights_hook import (
    TieredWeightsResidencyHook,
    clear_tiered_routed_experts,
    configure_tiered_routed_experts,
)


@dataclass(frozen=True, slots=True)
class TieredGlmProjection:
    expert_id: int
    projection: str
    weight: torch.Tensor
    weight_unit_id: str
    scale: torch.Tensor
    scale_unit_id: str


TieredGlmExpertProjections = dict[int, dict[str, TieredGlmProjection]]
TieredGlmExecutionCallback = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, TieredGlmExpertProjections],
    torch.Tensor,
]


class TieredGlmResidentExperts:
    def __init__(
        self,
        experts: dict[int, dict[str, TieredGlmProjection]],
        release: Callable[[], None],
    ) -> None:
        self.experts = experts
        self._release = release
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._release()
            self._released = True

    def __enter__(self) -> TieredGlmResidentExperts:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


class TieredGlmTensorProvider(Protocol):
    def request_experts(self, demands: Sequence[Any]) -> TieredGlmResidentExperts:
        """Make only the requested GLM routed-expert tensors resident."""
        ...

    def reserve_execution_scratch(self, scratch_bytes: int) -> Any:
        """Reserve callback scratch against a coherent device budget."""
        ...


_GLM_53_WEIGHT_BLOCK_SIZE = (128, 128)
_GLM_53_FIRST_SPARSE_LAYER = 3
_GLM_53_LAST_SPARSE_LAYER = 77
_GLM_EXPERT_UNIT_ID_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer_id>\d+)\.mlp\.experts\."
    r"(?P<expert_id>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<suffix>weight|weight_scale_inv)$"
)


def _parse_expert_location(unit_id: str) -> tuple[int, int] | None:
    match = _GLM_EXPERT_UNIT_ID_PATTERN.fullmatch(unit_id)
    if match is None:
        return None
    return int(match.group("layer_id")), int(match.group("expert_id"))


def _validate_exact_expert_demands(
    adapter: Any,
    selected_experts: set[tuple[int, int]],
    requested_units: set[str],
) -> int:
    layer_ids = {layer_id for layer_id, _ in selected_experts}
    if len(layer_ids) != 1:
        raise ValueError("Tiered GLM demands must target exactly one layer")

    expected_units: set[str] = set()
    for layer_id, expert_id in selected_experts:
        expert_units = set(adapter.expert_unit_ids(expert_id, layer_id=layer_id))
        actual_units = {
            unit_id
            for unit_id in requested_units
            if _parse_expert_location(unit_id) == (layer_id, expert_id)
        }
        if actual_units != expert_units:
            raise ValueError(
                f"Tiered GLM expert {expert_id} in layer {layer_id} has an "
                "incomplete unit demand set"
            )
        expected_units.update(expert_units)

    if requested_units != expected_units:
        raise ValueError("Tiered GLM demand set is not an exact expert union")
    return next(iter(layer_ids))


class LazyGlm53TensorProvider:
    """Bounded tensor-backed provider for exact routed-expert unit demands."""

    def __init__(
        self,
        *,
        adapter: Any,
        tensor_factory: Callable[[str], torch.Tensor],
        max_resident_units: int = 256 * 6,
    ) -> None:
        if max_resident_units < 6:
            raise ValueError("max_resident_units must hold one complete expert")
        self._adapter = adapter
        self._tensor_factory = tensor_factory
        self._max_resident_units = max_resident_units
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._active_units: Counter[str] = Counter()
        self._lock = threading.RLock()

    def request_experts(self, demands: Sequence[Any]) -> TieredGlmResidentExperts:
        if not demands:
            raise ValueError("Tiered GLM demands must not be empty")

        unit_ids = [demand.unit_id for demand in demands]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("Tiered GLM demands contain duplicate units")

        requested_units = set(unit_ids)
        selected_experts = self._selected_experts(requested_units)
        layer_id = _validate_exact_expert_demands(
            self._adapter,
            selected_experts,
            requested_units,
        )

        acquired_units: list[str] = []
        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            with self._lock:
                for unit_id in acquired_units:
                    self._active_units[unit_id] -= 1
                    if self._active_units[unit_id] == 0:
                        del self._active_units[unit_id]
                self._trim_cache()
            released = True

        try:
            with self._lock:
                self._reserve_capacity(requested_units)
                for unit_id in unit_ids:
                    self._acquire(unit_id, acquired_units)
                experts = self._build_experts(
                    selected_experts,
                    requested_units,
                    layer_id,
                )
        except Exception:
            release()
            raise

        return TieredGlmResidentExperts(experts, release)

    def _selected_experts(self, unit_ids: set[str]) -> set[tuple[int, int]]:
        selected_experts: set[tuple[int, int]] = set()
        for unit_id in unit_ids:
            expert_location = _parse_expert_location(unit_id)
            if expert_location is None:
                raise ValueError(f"demand is not a routed-expert unit: {unit_id!r}")
            selected_experts.add(expert_location)
        return selected_experts

    def _reserve_capacity(self, requested_units: set[str]) -> None:
        missing_count = sum(unit_id not in self._cache for unit_id in requested_units)
        while len(self._cache) + missing_count > self._max_resident_units:
            evictable_unit_id = next(
                (
                    unit_id
                    for unit_id in self._cache
                    if unit_id not in requested_units
                    and self._active_units[unit_id] == 0
                ),
                None,
            )
            if evictable_unit_id is None:
                raise RuntimeError(
                    "Tiered GLM tensor residency budget is too small for the "
                    "selected expert set"
                )
            self._cache.pop(evictable_unit_id)

    def _acquire(self, unit_id: str, acquired_units: list[str]) -> None:
        if unit_id not in self._cache:
            self._cache[unit_id] = self._materialize(unit_id)
        else:
            self._cache.move_to_end(unit_id)
        self._active_units[unit_id] += 1
        acquired_units.append(unit_id)

    def _build_experts(
        self,
        selected_experts: set[tuple[int, int]],
        requested_units: set[str],
        layer_id: int,
    ) -> dict[int, dict[str, TieredGlmProjection]]:
        experts: dict[int, dict[str, TieredGlmProjection]] = {}
        for _, expert_id in sorted(selected_experts):
            projections: dict[str, TieredGlmProjection] = {}
            for unit_id in self._adapter.expert_unit_ids(expert_id, layer_id=layer_id):
                if not unit_id.endswith(".weight") or unit_id not in requested_units:
                    continue
                metadata = self._adapter.native_metadata(unit_id)
                if metadata is None or metadata.get("scale_unit_id") is None:
                    raise ValueError(f"missing scale identity for unit {unit_id!r}")
                scale_unit_id = metadata["scale_unit_id"]
                if scale_unit_id not in requested_units:
                    raise ValueError(f"missing scale unit {scale_unit_id!r}")
                projection = unit_id.rsplit(".", maxsplit=2)[-2]
                projections[projection] = TieredGlmProjection(
                    expert_id=expert_id,
                    projection=projection,
                    weight=self._cache[unit_id],
                    weight_unit_id=unit_id,
                    scale=self._cache[scale_unit_id],
                    scale_unit_id=scale_unit_id,
                )
            if len(projections) != 3:
                raise ValueError(f"Tiered GLM expert {expert_id} is incomplete")
            experts[expert_id] = projections
        return experts

    def _materialize(self, unit_id: str) -> torch.Tensor:
        metadata = self._adapter.native_metadata(unit_id)
        if metadata is None:
            raise ValueError(f"Tiered GLM unit {unit_id!r} has no native metadata")
        expected_shape = tuple(metadata["shape"])
        expected_dtype = self._expected_dtype(metadata["dtype"])
        tensor = self._tensor_factory(unit_id)
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"Tiered GLM unit {unit_id!r} changed dtype: "
                f"{tensor.dtype} != {expected_dtype}"
            )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Tiered GLM unit {unit_id!r} changed shape: "
                f"{tuple(tensor.shape)} != {expected_shape}"
            )
        return tensor

    @staticmethod
    def _expected_dtype(native_dtype: str) -> torch.dtype:
        if native_dtype == "F8_E4M3":
            return torch.float8_e4m3fn
        if native_dtype == "F32":
            return torch.float32
        raise ValueError(f"unsupported native Tiered GLM dtype {native_dtype!r}")

    def _trim_cache(self) -> None:
        while len(self._cache) > self._max_resident_units:
            evictable_unit_id = next(
                (
                    unit_id
                    for unit_id in self._cache
                    if self._active_units[unit_id] == 0
                ),
                None,
            )
            if evictable_unit_id is None:
                break
            self._cache.pop(evictable_unit_id)


class EagerSelectedExpertsGlmTensorProvider:
    """Materialize each requested expert set without paging or eviction."""

    def __init__(
        self,
        *,
        adapter: Any,
        tensor_factory: Callable[[str], torch.Tensor],
        execution_callback: TieredGlmExecutionCallback | None = None,
    ) -> None:
        self._adapter = adapter
        self._tensor_factory = tensor_factory
        self.execution_callback = execution_callback
        self._lock = threading.RLock()
        self._active_unit_counts: Counter[str] = Counter()
        self._active_expert_counts: Counter[tuple[int, int]] = Counter()
        self._active_bytes = 0
        self._peak_bytes = 0
        self._requests = 0
        self._loads = 0
        self._read_bytes = 0

    def request_experts(self, demands: Sequence[Any]) -> TieredGlmResidentExperts:
        if not demands:
            raise ValueError("Tiered GLM demands must not be empty")

        unit_ids = [demand.unit_id for demand in demands]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("Tiered GLM demands contain duplicate units")

        requested_units = set(unit_ids)
        expert_locations = [
            _parse_expert_location(unit_id) for unit_id in requested_units
        ]
        if any(expert_location is None for expert_location in expert_locations):
            invalid_unit_ids = [
                unit_id
                for unit_id in requested_units
                if _parse_expert_location(unit_id) is None
            ]
            raise ValueError(
                f"demand is not a routed-expert unit: {invalid_unit_ids[0]!r}"
            )
        selected_experts = set(cast(tuple[int, int], expert_locations))
        layer_id = _validate_exact_expert_demands(
            self._adapter,
            selected_experts,
            requested_units,
        )

        tensors = {
            unit_id: self._materialize(unit_id, expected_shape, expected_dtype)
            for unit_id, expected_shape, expected_dtype in self._ordered_tensor_specs(
                selected_experts,
                layer_id,
            )
        }
        experts = self._build_experts(selected_experts, tensors, layer_id)
        with self._lock:
            self._requests += 1
            for unit_id in requested_units:
                self._active_unit_counts[unit_id] += 1
                if self._active_unit_counts[unit_id] == 1:
                    self._active_bytes += self._tensor_byte_size(tensors[unit_id])
            for expert_location in selected_experts:
                self._active_expert_counts[expert_location] += 1
            self._peak_bytes = max(self._peak_bytes, self._active_bytes)

        def release() -> None:
            with self._lock:
                for unit_id in requested_units:
                    self._active_unit_counts[unit_id] -= 1
                    if self._active_unit_counts[unit_id] == 0:
                        del self._active_unit_counts[unit_id]
                        self._active_bytes -= self._tensor_byte_size(tensors[unit_id])
                for expert_location in selected_experts:
                    self._active_expert_counts[expert_location] -= 1
                    if self._active_expert_counts[expert_location] == 0:
                        del self._active_expert_counts[expert_location]

        return TieredGlmResidentExperts(experts, release)

    def stats(self) -> dict[str, int | float | str | bool]:
        with self._lock:
            return {
                "provider": "eager-selected-experts",
                "paging": False,
                "bounded_eviction": False,
                "resident_units": len(self._active_unit_counts),
                "resident_groups": len(self._active_expert_counts),
                "resident_bytes": self._active_bytes,
                "peak_resident_bytes": self._peak_bytes,
                "active_leases": sum(self._active_expert_counts.values()),
                "requests": self._requests,
                "loads": self._loads,
                "storage_read_bytes": self._read_bytes,
            }

    def _ordered_tensor_specs(
        self,
        selected_experts: set[tuple[int, int]],
        layer_id: int,
    ) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
        tensor_specs: list[tuple[str, tuple[int, ...], torch.dtype]] = []
        for _, expert_id in sorted(selected_experts):
            for unit_id in self._adapter.expert_unit_ids(
                expert_id,
                layer_id=layer_id,
            ):
                metadata = self._adapter.native_metadata(unit_id)
                if metadata is None:
                    raise ValueError(f"Tiered GLM unit {unit_id!r} has no metadata")
                tensor_specs.append(
                    (
                        unit_id,
                        tuple(metadata["shape"]),
                        self._expected_dtype(metadata["dtype"]),
                    )
                )
        return tensor_specs

    def _materialize(
        self,
        unit_id: str,
        expected_shape: tuple[int, ...],
        expected_dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = self._tensor_factory(unit_id)
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"Tiered GLM unit {unit_id!r} changed dtype: "
                f"{tensor.dtype} != {expected_dtype}"
            )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Tiered GLM unit {unit_id!r} changed shape: "
                f"{tuple(tensor.shape)} != {expected_shape}"
            )
        with self._lock:
            self._loads += 1
            self._read_bytes += tensor.numel() * tensor.element_size()
        return tensor

    @staticmethod
    def _expected_dtype(native_dtype: str) -> torch.dtype:
        if native_dtype == "F8_E4M3":
            return torch.float8_e4m3fn
        if native_dtype == "F32":
            return torch.float32
        raise ValueError(f"unsupported native Tiered GLM dtype {native_dtype!r}")

    @staticmethod
    def _tensor_byte_size(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _build_experts(
        self,
        selected_experts: set[tuple[int, int]],
        tensors: dict[str, torch.Tensor],
        layer_id: int,
    ) -> dict[int, dict[str, TieredGlmProjection]]:
        experts: dict[int, dict[str, TieredGlmProjection]] = {}
        for _, expert_id in sorted(selected_experts):
            projections: dict[str, TieredGlmProjection] = {}
            for unit_id in self._adapter.expert_unit_ids(
                expert_id,
                layer_id=layer_id,
            ):
                if not unit_id.endswith(".weight"):
                    continue
                metadata = self._adapter.native_metadata(unit_id)
                scale_unit_id = metadata.get("scale_unit_id")
                if scale_unit_id is None or scale_unit_id not in tensors:
                    raise ValueError(f"missing scale identity for unit {unit_id!r}")
                projection = unit_id.rsplit(".", maxsplit=2)[-2]
                projections[projection] = TieredGlmProjection(
                    expert_id=expert_id,
                    projection=projection,
                    weight=tensors[unit_id],
                    weight_unit_id=unit_id,
                    scale=tensors[scale_unit_id],
                    scale_unit_id=scale_unit_id,
                )
            if len(projections) != 3:
                raise ValueError(f"Tiered GLM expert {expert_id} is incomplete")
            experts[expert_id] = projections
        return experts


class DeviceWeightRuntimeGlmTensorProvider:
    """Adapt the model-agnostic runtime to the GLM routed-expert API."""

    def __init__(
        self,
        *,
        runtime: Any,
        adapter: Any,
        execution_callback: TieredGlmExecutionCallback | None = None,
    ) -> None:
        self._runtime = runtime
        self._adapter = adapter
        self.execution_callback = execution_callback

    def request_experts(self, demands: Sequence[Any]) -> TieredGlmResidentExperts:
        if not demands:
            raise ValueError("Tiered GLM demands must not be empty")

        unit_ids = [demand.unit_id for demand in demands]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("Tiered GLM demands contain duplicate units")

        requested_units = set(unit_ids)
        selected_experts = self._selected_experts(requested_units)
        layer_id = _validate_exact_expert_demands(
            self._adapter,
            selected_experts,
            requested_units,
        )

        lease_stack = ExitStack()
        leases = []
        try:
            for _, expert_id in sorted(selected_experts):
                lease = self._runtime.acquire_expert(
                    expert_id,
                    layer_id=layer_id,
                )
                lease_stack.enter_context(lease)
                leases.append(lease)
            experts = self._build_experts(leases)
        except BaseException:
            lease_stack.close()
            raise

        return TieredGlmResidentExperts(experts, lease_stack.close)

    def reserve_execution_scratch(self, scratch_bytes: int):
        return self._runtime.reserve_scratch(scratch_bytes)

    def stats(self) -> dict[str, int | float | str]:
        return self._runtime.stats()

    def _selected_experts(self, unit_ids: set[str]) -> set[tuple[int, int]]:
        selected_experts: set[tuple[int, int]] = set()
        for unit_id in unit_ids:
            expert_location = _parse_expert_location(unit_id)
            if expert_location is None:
                raise ValueError(f"demand is not a routed-expert unit: {unit_id!r}")
            selected_experts.add(expert_location)
        return selected_experts

    def _build_experts(
        self, leases: Sequence[Any]
    ) -> dict[int, dict[str, TieredGlmProjection]]:
        experts: dict[int, dict[str, TieredGlmProjection]] = {}
        for lease in leases:
            projections: dict[str, TieredGlmProjection] = {}
            for unit_id in lease.weight_unit_ids:
                metadata = self._adapter.native_metadata(unit_id)
                scale_unit_id = lease.linked_scale_unit_id(unit_id)
                if metadata is None or scale_unit_id != metadata.get("scale_unit_id"):
                    raise ValueError(f"missing scale identity for unit {unit_id!r}")
                projection = unit_id.rsplit(".", maxsplit=2)[-2]
                projections[projection] = TieredGlmProjection(
                    expert_id=lease.expert_id,
                    projection=projection,
                    weight=lease.weight_tensor(unit_id),
                    weight_unit_id=unit_id,
                    scale=lease.scale_tensor(unit_id),
                    scale_unit_id=scale_unit_id,
                )
            if len(projections) != 3:
                raise ValueError(f"Tiered GLM expert {lease.expert_id} is incomplete")
            experts[lease.expert_id] = projections
        return experts


class TieredGlm53MoEMethod(FusedMoEMethodBase):
    """MoE method that loads only router-selected GLM 5.3 expert tensors."""

    def __init__(
        self,
        moe: FusedMoEConfig,
        *,
        provider: TieredGlmTensorProvider,
    ) -> None:
        super().__init__(moe)
        self._provider = provider
        self._residency_hook = TieredWeightsResidencyHook()
        if not self._residency_hook.enable_for_model("glm-5.3"):
            raise RuntimeError(
                "the registered GLM 5.3 Tiered Weights adapter is unavailable"
            )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        return None

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> None:
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if topk_ids.ndim != 2:
            raise ValueError("Tiered GLM routing IDs must be rank 2")
        experts_per_token = topk_ids.shape[1]
        if experts_per_token <= 0:
            raise ValueError("Tiered GLM routing selected no experts")

        reserve_execution_scratch = getattr(
            self._provider,
            "reserve_execution_scratch",
            None,
        )
        output_reservation = (
            reserve_execution_scratch(x.numel() * x.element_size())
            if callable(reserve_execution_scratch)
            else nullcontext()
        )
        with output_reservation:
            output = torch.zeros_like(x)
            for row_indices in _partition_rows_by_expert_union(
                topk_ids,
                experts_per_token,
            ):
                demands = self._residency_hook.build_demands_from_router(
                    router_topk_ids=topk_ids[row_indices],
                    layer_prefix=layer.layer_name,
                    target_view_id="gpu-vram",
                )
                if not demands:
                    raise RuntimeError(
                        "router produced no GLM 5.3 Tiered Weights demands"
                    )
                with self._provider.request_experts(demands) as resident_experts:
                    scratch_bytes = _glm_53_execution_scratch_bytes(
                        x[row_indices],
                        resident_experts.experts,
                    )
                    if callable(reserve_execution_scratch):
                        with reserve_execution_scratch(scratch_bytes):
                            chunk_output = self._execution_callback(
                                x[row_indices],
                                topk_weights[row_indices],
                                topk_ids[row_indices],
                                resident_experts.experts,
                            )
                    else:
                        chunk_output = self._execution_callback(
                            x[row_indices],
                            topk_weights[row_indices],
                            topk_ids[row_indices],
                            resident_experts.experts,
                        )
                output.index_copy_(
                    0,
                    torch.as_tensor(row_indices, device=x.device, dtype=torch.long),
                    chunk_output,
                )
        return output

    @property
    def _execution_callback(self) -> TieredGlmExecutionCallback:
        callback = getattr(self._provider, "execution_callback", None)
        return glm_53_tiered_execution_callback if callback is None else callback

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Tiered GLM 5.3 requires modular router execution")


class TieredGlm53RoutedExperts(RoutedExperts):
    """Routed-experts container without eager fused expert tensor allocation."""

    def __init__(
        self,
        *args: Any,
        provider: TieredGlmTensorProvider,
        **kwargs: Any,
    ) -> None:
        self._tiered_provider = provider
        super().__init__(*args, **kwargs)

    def _get_quant_method(
        self,
        prefix: str,
        quant_config: Any,
        moe_config: FusedMoEConfig,
    ) -> FusedMoEMethodBase:
        return TieredGlm53MoEMethod(
            moe_config,
            provider=self._tiered_provider,
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[str]:
        for _ in weights:
            pass
        return iter(())


_TIERED_GLM_53_CONFIG_LOCK = threading.RLock()
_TIERED_GLM_53_CONFIG: TieredGlmTensorProvider | None = None


def configure_glm_53_tiered_experts(provider: TieredGlmTensorProvider) -> None:
    global _TIERED_GLM_53_CONFIG
    with _TIERED_GLM_53_CONFIG_LOCK:
        _TIERED_GLM_53_CONFIG = provider
    configure_tiered_routed_experts(
        lambda config, prefix: glm_53_tiered_experts(config, prefix)
    )


def clear_glm_53_tiered_experts() -> None:
    global _TIERED_GLM_53_CONFIG
    with _TIERED_GLM_53_CONFIG_LOCK:
        _TIERED_GLM_53_CONFIG = None
    clear_tiered_routed_experts()


def glm_53_tiered_experts(
    config: Any,
    prefix: str,
) -> tuple[type[RoutedExperts], dict[str, Any]] | None:
    with _TIERED_GLM_53_CONFIG_LOCK:
        tiered_config = _TIERED_GLM_53_CONFIG
    if tiered_config is None or not _is_glm_53(config, prefix):
        return None
    return TieredGlm53RoutedExperts, {"provider": tiered_config}


def glm_53_tiered_execution_callback(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts: dict[int, dict[str, TieredGlmProjection]],
) -> torch.Tensor:
    if hidden_states.ndim != 2:
        raise ValueError("Tiered GLM hidden states must be rank 2")
    num_tokens, hidden_size = hidden_states.shape
    if topk_weights.shape != topk_ids.shape or topk_weights.ndim != 2:
        raise ValueError("Tiered GLM routing tensors must have matching rank-2 shapes")
    if topk_weights.shape[0] != num_tokens:
        raise ValueError("Tiered GLM routing does not match the token count")
    if topk_weights.numel() == 0:
        raise ValueError("Tiered GLM routing selected no experts")

    selected_experts = {
        int(expert_id) for routing_row in topk_ids.tolist() for expert_id in routing_row
    }
    if any(expert_id < 0 for expert_id in selected_experts):
        raise ValueError("Tiered GLM routing produced an invalid expert id")
    if selected_experts != set(experts):
        raise RuntimeError("Tiered GLM resident experts do not match router selection")

    output = torch.zeros(
        (num_tokens, hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    for expert_id in sorted(selected_experts):
        expert_rows, routing_slots = torch.nonzero(topk_ids == expert_id, as_tuple=True)
        projections = experts[expert_id]
        expert_input = hidden_states[expert_rows].to(torch.float32)
        gate_weight = _dequant_glm_53_fp8_block(projections["gate_proj"])
        up_weight = _dequant_glm_53_fp8_block(projections["up_proj"])
        down_weight = _dequant_glm_53_fp8_block(projections["down_proj"])
        gate_output = expert_input @ gate_weight.T
        up_output = expert_input @ up_weight.T
        intermediate = F.silu(gate_output) * up_output
        expert_output = intermediate @ down_weight.T
        routing_weights = topk_weights[expert_rows, routing_slots].to(torch.float32)
        output.index_add_(
            0,
            expert_rows,
            expert_output * routing_weights.unsqueeze(1),
        )

    return output.to(hidden_states.dtype)


def _partition_rows_by_expert_union(
    topk_ids: torch.Tensor,
    experts_per_token: int,
) -> list[list[int]]:
    if topk_ids.ndim != 2:
        raise ValueError("Tiered GLM routing IDs must be rank 2")
    if topk_ids.shape[1] != experts_per_token:
        raise ValueError("Tiered GLM routing width does not match the expert budget")

    row_groups: list[list[int]] = []
    current_rows: list[int] = []
    current_expert_ids: set[int] = set()
    for row_index, routing_row in enumerate(topk_ids.tolist()):
        row_expert_ids = {int(expert_id) for expert_id in routing_row}
        if len(row_expert_ids) > experts_per_token:
            raise ValueError("Tiered GLM routing row contains duplicate experts")
        combined_expert_ids = current_expert_ids | row_expert_ids
        if current_expert_ids and len(combined_expert_ids) > experts_per_token:
            row_groups.append(current_rows)
            current_rows = []
            current_expert_ids = set()
        current_rows.append(row_index)
        current_expert_ids.update(row_expert_ids)
    if current_rows:
        row_groups.append(current_rows)
    return row_groups


def _glm_53_execution_scratch_bytes(
    hidden_states: torch.Tensor,
    experts: dict[int, dict[str, TieredGlmProjection]],
) -> int:
    if hidden_states.ndim != 2 or not experts:
        raise ValueError(
            "Tiered GLM execution scratch requires rank-2 input and experts"
        )
    num_tokens, hidden_size = hidden_states.shape
    gate_weight = next(iter(experts.values()))["gate_proj"].weight
    intermediate_size = gate_weight.shape[0]
    weight_elements = hidden_size * intermediate_size
    return (
        4 * num_tokens * (5 * hidden_size + 3 * intermediate_size)
        + 16 * weight_elements
    )


def _dequant_glm_53_fp8_block(projection: TieredGlmProjection) -> torch.Tensor:
    weight = projection.weight
    scale = projection.scale
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"Tiered GLM weight {projection.weight_unit_id!r} is not FP8 E4M3"
        )
    if scale.dtype != torch.float32:
        raise TypeError(f"Tiered GLM scale {projection.scale_unit_id!r} is not FP32")
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("Tiered GLM FP8 tensors must be rank 2")

    block_rows, block_columns = _GLM_53_WEIGHT_BLOCK_SIZE
    expected_scale_shape = (
        (weight.shape[0] + block_rows - 1) // block_rows,
        (weight.shape[1] + block_columns - 1) // block_columns,
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"Tiered GLM scale {projection.scale_unit_id!r} has shape "
            f"{tuple(scale.shape)}, expected {expected_scale_shape}"
        )

    expanded_scale = scale.repeat_interleave(block_rows, dim=0).repeat_interleave(
        block_columns, dim=1
    )[: weight.shape[0], : weight.shape[1]]
    return weight.to(torch.float32) * expanded_scale


def _is_glm_53(config: Any, prefix: str) -> bool:
    prefix_parts = prefix.rstrip(".").split(".")
    try:
        layer_index = prefix_parts[prefix_parts.index("layers") + 1]
    except (ValueError, IndexError):
        return False
    return (
        getattr(config, "model_type", None) == "glm_moe_dsa"
        and getattr(config, "n_routed_experts", None) == 256
        and getattr(config, "num_experts_per_tok", None) == 8
        and getattr(config, "hidden_size", None) == 6144
        and getattr(config, "moe_intermediate_size", None) == 2048
        and layer_index.isdigit()
        and _GLM_53_FIRST_SPARSE_LAYER <= int(layer_index) <= _GLM_53_LAST_SPARSE_LAYER
    )
