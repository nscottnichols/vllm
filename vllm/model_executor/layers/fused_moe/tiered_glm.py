# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused Tiered Weights execution path for GLM 5.3 routed experts."""

from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from tiered_weights.runtime.derived import DerivedTensorCache
from tiered_weights.runtime.device import (
    DeviceWeightCurrentLayerChunkPreparation,
    DeviceWeightCurrentLayerChunkScheduler,
)

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
from vllm.utils.torch_utils import direct_register_custom_op


@dataclass(frozen=True, slots=True)
class TieredGlmProjection:
    expert_id: int
    projection: str
    weight: torch.Tensor
    weight_unit_id: str
    scale: torch.Tensor
    scale_unit_id: str
    dequantized_weight: torch.Tensor | None = None


@dataclass(slots=True)
class _EagerRouteReplayPrefetch:
    layer_id: int
    tensor_specs: tuple[TieredGlmTensorSpec, ...]
    future: Future[list[torch.Tensor]]

    @property
    def byte_count(self) -> int:
        return sum(
            math.prod(spec[1]) * torch.empty((), dtype=spec[2]).element_size()
            for spec in self.tensor_specs
        )


TieredGlmExpertProjections = dict[int, dict[str, TieredGlmProjection]]
TieredGlmExecutionCallback = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, TieredGlmExpertProjections],
    torch.Tensor,
]
TieredGlmTensorSpec = tuple[str, tuple[int, ...], torch.dtype]
TieredGlmBatchedTensorFactory = Callable[
    [Sequence[TieredGlmTensorSpec]],
    "TieredGlmBatchedTensors | Sequence[torch.Tensor] | None",
]


class TieredGlmBatchedTensors(list[torch.Tensor]):
    def __init__(
        self,
        tensors: Iterable[torch.Tensor],
        *,
        batch_count: int,
    ) -> None:
        super().__init__(tensors)
        if type(batch_count) is not int or batch_count <= 0 or batch_count > len(self):
            raise ValueError("invalid Tiered GLM materialization batch count")
        self.batch_count = batch_count


@dataclass(slots=True)
class _CurrentLayerChunkCacheContext:
    layer_id: int
    cached_projections: dict[int, dict[str, TieredGlmProjection]]
    projection_names: dict[int, tuple[str, ...]]
    uncached_expert_ids: tuple[int, ...]
    cache_keys: list[tuple[int, int, str]]


class TieredGlmExecutionTimings:
    """Thread-safe cumulative timing buckets for the GLM execution callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seconds = {
            "routing_indexing_seconds": 0.0,
            "fp8_block_dequantization_seconds": 0.0,
            "matmul_activation_accumulation_seconds": 0.0,
        }
        self._counts = {
            "routing_indexing_count": 0,
            "fp8_block_dequantization_count": 0,
            "matmul_activation_accumulation_count": 0,
        }

    def record_routing_indexing(self, elapsed_seconds: float) -> None:
        self._record("routing_indexing", elapsed_seconds)

    def record_fp8_block_dequantization(self, elapsed_seconds: float) -> None:
        self._record("fp8_block_dequantization", elapsed_seconds)

    def record_matmul_activation_accumulation(self, elapsed_seconds: float) -> None:
        self._record("matmul_activation_accumulation", elapsed_seconds)

    def _record(self, stage_name: str, elapsed_seconds: float) -> None:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise RuntimeError(
                f"non-finite or negative GLM callback timing: {elapsed_seconds}"
            )
        with self._lock:
            updated_seconds = self._seconds[f"{stage_name}_seconds"] + elapsed_seconds
            if not math.isfinite(updated_seconds):
                raise RuntimeError(f"GLM callback timing overflow for {stage_name}")
            self._seconds[f"{stage_name}_seconds"] = updated_seconds
            self._counts[f"{stage_name}_count"] += 1

    def metrics(self) -> dict[str, float | int]:
        with self._lock:
            return {**self._seconds, **self._counts}


class TieredGlmApplyTimings:
    """Thread-safe cumulative timing buckets for the GLM MoE apply path."""

    _STAGE_METRIC_NAMES = {
        "apply_total": "apply_total",
        "route_controller": "route_controller",
        "demand_build": "demand_build",
        "current_layer_prefetch": "current_layer_prefetch",
        "row_partition": "row_partition",
        "provider_request": "apply_provider_request",
        "callback": "callback",
        "output_copy": "output_copy",
    }
    _STAGES = tuple(_STAGE_METRIC_NAMES)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        metric_names = self._STAGE_METRIC_NAMES.values()
        self._seconds = {f"{metric_name}_seconds": 0.0 for metric_name in metric_names}
        self._counts = {f"{metric_name}_count": 0 for metric_name in metric_names}

    def record(self, stage_name: str, elapsed_seconds: float) -> None:
        if stage_name not in self._STAGES:
            raise ValueError(f"unknown GLM apply timing stage: {stage_name!r}")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise RuntimeError(
                f"non-finite or negative GLM apply timing: {elapsed_seconds}"
            )
        metric_name = self._STAGE_METRIC_NAMES[stage_name]
        with self._lock:
            updated_seconds = self._seconds[f"{metric_name}_seconds"] + elapsed_seconds
            if not math.isfinite(updated_seconds):
                raise RuntimeError(f"GLM apply timing overflow for {stage_name}")
            self._seconds[f"{metric_name}_seconds"] = updated_seconds
            self._counts[f"{metric_name}_count"] += 1

    def metrics(self) -> dict[str, float | int]:
        with self._lock:
            return {**self._seconds, **self._counts}


def _synchronize_glm_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


_GLM_APPLY_NO_TIMING = nullcontext()


@contextmanager
def _record_glm_apply_timing(
    timings: TieredGlmApplyTimings,
    stage_name: str,
    device: torch.device,
):
    _synchronize_glm_timing(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        _synchronize_glm_timing(device)
        timings.record(stage_name, time.perf_counter() - started)


def _timed_glm_apply_stage(
    timings: TieredGlmApplyTimings | None,
    stage_name: str,
    device: torch.device,
):
    if timings is None:
        return _GLM_APPLY_NO_TIMING
    return _record_glm_apply_timing(timings, stage_name, device)


@contextmanager
def _timed_glm_execution_stage(
    timings: TieredGlmExecutionTimings | None,
    stage_name: str,
    device: torch.device,
):
    if timings is None:
        yield
        return

    if stage_name == "routing_indexing":
        record = timings.record_routing_indexing
    elif stage_name == "fp8_block_dequantization":
        record = timings.record_fp8_block_dequantization
    elif stage_name == "matmul_activation_accumulation":
        record = timings.record_matmul_activation_accumulation
    else:
        raise ValueError(f"unknown GLM callback timing stage: {stage_name!r}")

    _synchronize_glm_timing(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        _synchronize_glm_timing(device)
        record(time.perf_counter() - started)


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

    def drain_prefetches(self) -> None:
        """Wait for queued prefetch work without closing caches or storage."""
        ...


@dataclass(frozen=True, slots=True)
class _TieredGlm53Config:
    provider: TieredGlmTensorProvider
    adapter: Any | None


_GlmDeviceCacheKey = tuple[int, int, str]
_GlmDeviceCacheGroupKey = tuple[int, int]
_GlmDequantCacheKey = tuple[str, str]


@dataclass(slots=True)
class _GlmDeviceCacheEntry:
    projection: TieredGlmProjection
    byte_count: int
    active_leases: int
    scratch_reservation: ExitStack


class TieredGlmPreviousTokenPredictor:
    """Predict future-layer selections from the previous token."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._selections: dict[int, frozenset[int]] = {}

    def predict(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        *,
        depth: int = 1,
    ) -> tuple[tuple[int, frozenset[int]] | None, ...]:
        routing_rows = topk_ids.tolist()
        selected_experts = frozenset(
            int(expert_id) for row in routing_rows for expert_id in row
        )
        with self._lock:
            if layer_id != _GLM_53_LAST_SPARSE_LAYER:
                predictions: tuple[tuple[int, frozenset[int]] | None, ...] = (
                    None,
                ) * depth
            else:
                predictions = tuple(
                    (
                        _GLM_53_FIRST_SPARSE_LAYER + future_offset,
                        self._selections[_GLM_53_FIRST_SPARSE_LAYER + future_offset],
                    )
                    if _GLM_53_FIRST_SPARSE_LAYER + future_offset in self._selections
                    else None
                    for future_offset in range(depth)
                )
            self._selections[layer_id] = selected_experts
            return predictions


class TieredGlmCurrentRouterPredictor:
    """Predict future sparse layers from the current router selection.

    The prediction is advisory and is bounded to the router's per-token expert
    width so it remains a valid demand set for the GLM adapter.
    """

    def predict(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        *,
        depth: int = 1,
    ) -> tuple[tuple[int, frozenset[int]] | None, ...]:
        if topk_ids.ndim != 2:
            raise ValueError("Tiered GLM current-router routing IDs must be rank 2")

        expert_counts = Counter(
            int(expert_id) for row in topk_ids.tolist() for expert_id in row
        )
        if not expert_counts or topk_ids.shape[1] <= 0:
            return (None,) * depth
        selected_experts = frozenset(
            expert_id
            for expert_id, _ in sorted(
                expert_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[: topk_ids.shape[1]]
        )
        if not selected_experts:
            return (None,) * depth

        return tuple(
            (
                _glm_53_future_layer_id(layer_id, future_offset),
                selected_experts,
            )
            for future_offset in range(1, depth + 1)
        )


class TieredGlmPreviousTokenLayerPredictor:
    """Predict future layers from confidence-gated routing frequencies."""

    def __init__(self, *, min_confidence: float = 0.30) -> None:
        if not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0.0, 1.0]")
        self._lock = threading.Lock()
        self._history: dict[int, Counter[int]] = {}
        self._min_confidence = min_confidence

    def predict(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        *,
        depth: int = 1,
    ) -> tuple[tuple[int, frozenset[int]] | None, ...]:
        if topk_ids.ndim != 2 or topk_ids.shape[0] == 0:
            raise ValueError(
                "Tiered GLM previous-token routing IDs must be rank 2 and non-empty"
            )
        routing_rows = topk_ids.tolist()
        topk_width = topk_ids.shape[1]
        with self._lock:
            history = self._history.setdefault(layer_id, Counter())
            history.update(
                int(expert_id)
                for routing_row in routing_rows
                for expert_id in routing_row
            )
            predictions: list[tuple[int, frozenset[int]] | None] = []
            for future_offset in range(1, depth + 1):
                future_layer_id = _glm_53_future_layer_id(layer_id, future_offset)
                future_history = self._history.get(future_layer_id)
                if not future_history:
                    predictions.append(None)
                    continue
                total_selections = sum(future_history.values())
                top_experts = [
                    expert_id
                    for expert_id, _ in sorted(
                        future_history.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:topk_width]
                ]
                confidence = (
                    sum(future_history[expert_id] for expert_id in top_experts)
                    / total_selections
                )
                predictions.append(
                    (future_layer_id, frozenset(top_experts))
                    if confidence >= self._min_confidence
                    else None
                )
            return tuple(predictions)


_GLM_53_WEIGHT_BLOCK_SIZE = (128, 128)
_GLM_53_FIRST_SPARSE_LAYER = 3
_GLM_53_LAST_SPARSE_LAYER = 77
_GLM_53_MTP_LAYER = 78
_GLM_EXPERT_UNIT_ID_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer_id>\d+)\.mlp\.experts\."
    r"(?P<expert_id>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<suffix>weight|weight_scale_inv)$"
)


def _glm_53_future_layer_id(layer_id: int, future_offset: int) -> int:
    if not _GLM_53_FIRST_SPARSE_LAYER <= layer_id <= _GLM_53_LAST_SPARSE_LAYER:
        raise ValueError(
            f"Tiered GLM layer {layer_id} is outside the sparse-layer range"
        )

    future_layer_id = layer_id + future_offset
    if future_layer_id > _GLM_53_LAST_SPARSE_LAYER:
        future_layer_id = (
            _GLM_53_FIRST_SPARSE_LAYER + future_layer_id - _GLM_53_LAST_SPARSE_LAYER - 1
        )
    return future_layer_id


def _parse_expert_location(unit_id: str) -> tuple[int, int] | None:
    match = _GLM_EXPERT_UNIT_ID_PATTERN.fullmatch(unit_id)
    if match is None:
        return None
    return int(match.group("layer_id")), int(match.group("expert_id"))


def _parse_glm_layer_id(layer_name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if match is None:
        raise ValueError(f"invalid Tiered GLM layer name: {layer_name!r}")
    return int(match.group(1))


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


def _tiered_glm_replay_router_impl(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    replay_ids: torch.Tensor,
    replay_weights: torch.Tensor,
) -> None:
    topk_ids.copy_(replay_ids)
    topk_weights.copy_(replay_weights)


def _tiered_glm_replay_router_fake(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    replay_ids: torch.Tensor,
    replay_weights: torch.Tensor,
) -> None:
    return None


direct_register_custom_op(
    op_name="tiered_glm_replay_router",
    op_func=_tiered_glm_replay_router_impl,
    mutates_args=["topk_ids", "topk_weights"],
    fake_impl=_tiered_glm_replay_router_fake,
    dispatch_key="CompositeExplicitAutograd",
)


class TieredGlmRouteController:
    """Fixture-only controller for exact GLM router capture and replay."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._mode: str | None = None
        self._row_count = 0
        self._layer_count = 0
        self._top_k = 0
        self._capture_sample_active = False
        self._capture_ids: dict[int, torch.Tensor] = {}
        self._capture_weights: dict[int, torch.Tensor] = {}
        self._replay_ids: torch.Tensor | None = None
        self._replay_weights: torch.Tensor | None = None
        self._replay_layer_ids: set[int] = set()
        self._replay_complete = False
        self._stats = {
            "capture_count": 0,
            "capture_sample_count": 0,
            "capture_layer_count": 0,
            "replay_count": 0,
            "replay_layer_count": 0,
            "apply_count": 0,
        }

    @classmethod
    def capture(
        cls,
        *,
        row_count: int = 4,
        layer_count: int = 75,
        top_k: int = 8,
    ) -> TieredGlmRouteController:
        controller = cls()
        controller._configure_capture(
            row_count=row_count,
            layer_count=layer_count,
            top_k=top_k,
        )
        return controller

    def _configure_capture(
        self,
        *,
        row_count: int,
        layer_count: int,
        top_k: int,
    ) -> None:
        self._validate_dimensions(row_count, layer_count, top_k)
        with self._lock:
            self._fail_if_transition_is_incomplete()
            if self._capture_sample_active:
                raise RuntimeError("a Tiered GLM capture sample is already active")
            self._mode = "capture"
            self._row_count = row_count
            self._layer_count = layer_count
            self._top_k = top_k
            self._stats["capture_count"] += 1

    @classmethod
    def replay(
        cls,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        row_count: int = 4,
        layer_count: int = 75,
        top_k: int = 8,
        device: str | torch.device | None = None,
    ) -> TieredGlmRouteController:
        controller = cls()
        controller._configure_replay(
            topk_ids,
            topk_weights,
            row_count=row_count,
            layer_count=layer_count,
            top_k=top_k,
            device=device,
        )
        return controller

    def _configure_replay(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        row_count: int,
        layer_count: int,
        top_k: int,
        device: str | torch.device | None,
    ) -> None:
        self._validate_dimensions(row_count, layer_count, top_k)
        self._validate_replay_tensors(
            topk_ids,
            topk_weights,
            row_count=row_count,
            layer_count=layer_count,
            top_k=top_k,
        )
        replay_device = torch.device("cpu") if device is None else torch.device(device)
        persisted_ids = (
            topk_ids.detach().to(device=replay_device, copy=True).contiguous()
        )
        persisted_weights = (
            topk_weights.detach().to(device=replay_device, copy=True).contiguous()
        )
        with self._lock:
            self._fail_if_transition_is_incomplete()
            if self._capture_sample_active:
                raise RuntimeError("a Tiered GLM capture sample is still active")
            self._mode = "replay"
            self._row_count = row_count
            self._layer_count = layer_count
            self._top_k = top_k
            self._replay_ids = persisted_ids
            self._replay_weights = persisted_weights
            self._replay_layer_ids = set()
            self._replay_complete = False
            self._stats["replay_count"] += 1

    def start_capture_sample(self) -> None:
        with self._lock:
            if self._mode != "capture":
                raise RuntimeError("Tiered GLM route capture mode is not active")
            if self._capture_sample_active:
                raise RuntimeError("a Tiered GLM capture sample is already active")
            self._capture_sample_active = True
            self._capture_ids = {}
            self._capture_weights = {}

    def peek_layer_expert_ids(self, layer_id: int) -> tuple[int, ...] | None:
        with self._lock:
            if self._mode != "replay" or self._replay_ids is None:
                raise RuntimeError("Tiered GLM route replay mode is not active")
            layer_offset = layer_id - _GLM_53_FIRST_SPARSE_LAYER
            if not 0 <= layer_offset < self._layer_count:
                return None
            selected_experts = {
                int(expert_id)
                for row in self._replay_ids.tolist()
                for expert_id in row[layer_offset]
            }
            if not selected_experts:
                raise RuntimeError(f"Tiered GLM replay layer {layer_id} has no experts")
            return tuple(sorted(selected_experts))

    def finish_capture_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        with self._lock:
            if self._mode != "capture" or not self._capture_sample_active:
                raise RuntimeError("no Tiered GLM capture sample is active")
            expected_layer_ids = self._expected_layer_ids()
            missing_layer_ids = set(expected_layer_ids).difference(self._capture_ids)
            if missing_layer_ids:
                raise RuntimeError(
                    "Tiered GLM capture sample is missing layers: "
                    f"{sorted(missing_layer_ids)}"
                )

            ids = torch.empty(
                (self._row_count, self._layer_count, self._top_k),
                dtype=torch.int32,
                device="cpu",
            )
            weights = torch.empty(
                (self._row_count, self._layer_count, self._top_k),
                dtype=torch.float32,
                device="cpu",
            )
            for layer_id in expected_layer_ids:
                layer_index = layer_id - _GLM_53_FIRST_SPARSE_LAYER
                ids[:, layer_index].copy_(self._capture_ids[layer_id])
                weights[:, layer_index].copy_(self._capture_weights[layer_id])

            self._capture_sample_active = False
            self._capture_ids = {}
            self._capture_weights = {}
            self._stats["capture_sample_count"] += 1
            return ids, weights

    def apply(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if type(layer_id) is not int:
            raise TypeError("Tiered GLM route layer ID must be an integer")
        with self._lock:
            if self._mode is None:
                raise RuntimeError("Tiered GLM route controller has no active mode")
            if layer_id not in self._expected_layer_ids():
                raise ValueError(
                    f"Tiered GLM layer {layer_id} is outside the configured layer range"
                )
            self._validate_live_tensors(topk_ids, topk_weights)

            if self._mode == "capture":
                if not self._capture_sample_active:
                    raise RuntimeError("no Tiered GLM capture sample is active")
                if layer_id in self._capture_ids:
                    raise ValueError(
                        f"Tiered GLM layer {layer_id} was captured more than once"
                    )
                self._capture_ids[layer_id] = (
                    topk_ids.detach().to(device="cpu", copy=True).contiguous()
                )
                self._capture_weights[layer_id] = (
                    topk_weights.detach().to(device="cpu", copy=True).contiguous()
                )
                self._stats["capture_layer_count"] += 1
                self._stats["apply_count"] += 1
                return topk_ids, topk_weights

            if layer_id in self._replay_layer_ids:
                raise ValueError(
                    f"Tiered GLM layer {layer_id} was replayed more than once"
                )
            if not self._replay_layer_ids:
                self._replay_complete = False
            replay_ids = self._replay_ids
            replay_weights = self._replay_weights
            if replay_ids is None or replay_weights is None:
                raise RuntimeError("Tiered GLM replay tensors are unavailable")

            layer_index = layer_id - _GLM_53_FIRST_SPARSE_LAYER
            source_ids = replay_ids[:, layer_index].to(device=topk_ids.device)
            source_weights = replay_weights[:, layer_index].to(
                device=topk_weights.device
            )
            torch.ops.vllm.tiered_glm_replay_router(
                topk_ids,
                topk_weights,
                source_ids,
                source_weights,
            )
            self._replay_layer_ids.add(layer_id)
            if self._replay_layer_ids == set(self._expected_layer_ids()):
                self._replay_layer_ids = set()
                self._replay_complete = True
            self._stats["replay_layer_count"] += 1
            self._stats["apply_count"] += 1
        return topk_ids, topk_weights

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def _expected_layer_ids(self) -> range:
        return range(
            _GLM_53_FIRST_SPARSE_LAYER,
            _GLM_53_FIRST_SPARSE_LAYER + self._layer_count,
        )

    def _fail_if_transition_is_incomplete(self) -> None:
        if self._mode == "replay" and not self._replay_complete:
            missing_layer_ids = set(self._expected_layer_ids()).difference(
                self._replay_layer_ids
            )
            raise RuntimeError(
                f"Tiered GLM replay is missing layers: {sorted(missing_layer_ids)}"
            )

    @staticmethod
    def _validate_dimensions(row_count: int, layer_count: int, top_k: int) -> None:
        dimensions = {
            "row_count": row_count,
            "layer_count": layer_count,
            "top_k": top_k,
        }
        for name, value in dimensions.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"Tiered GLM route {name} must be a positive integer")
        if layer_count > _GLM_53_LAST_SPARSE_LAYER - _GLM_53_FIRST_SPARSE_LAYER + 1:
            raise ValueError("Tiered GLM route layer_count exceeds the sparse layers")

    def _validate_live_tensors(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        if not isinstance(topk_ids, torch.Tensor):
            raise TypeError("Tiered GLM routing IDs must be a tensor")
        if not isinstance(topk_weights, torch.Tensor):
            raise TypeError("Tiered GLM routing weights must be a tensor")
        if topk_ids.ndim != 2:
            raise ValueError("Tiered GLM route live IDs must be rank 2")
        if topk_ids.shape != (self._row_count, self._top_k):
            raise ValueError(
                "Tiered GLM route live IDs must have shape "
                f"{(self._row_count, self._top_k)}"
            )
        if topk_ids.dtype != torch.int32:
            raise ValueError("Tiered GLM route live IDs must be int32")
        if topk_weights.dtype != torch.float32:
            raise ValueError("Tiered GLM route live weights must be float32")
        if topk_weights.shape != topk_ids.shape:
            raise ValueError("Tiered GLM route live IDs and weights must match shape")
        if topk_ids.device != topk_weights.device:
            raise ValueError("Tiered GLM route live tensors must share a device")
        if not bool(torch.isfinite(topk_weights).all()):
            raise ValueError("Tiered GLM route live weights must be finite")
        if not bool((topk_weights >= 0).all()):
            raise ValueError("Tiered GLM route live weights must be nonnegative")
        top_k = topk_ids.shape[1]
        if any(len(set(row.tolist())) != top_k for row in topk_ids):
            raise ValueError("Tiered GLM route live IDs must be unique per row")

    @staticmethod
    def _validate_replay_tensors(
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        row_count: int,
        layer_count: int,
        top_k: int,
    ) -> None:
        expected_shape = (row_count, layer_count, top_k)
        if not isinstance(topk_ids, torch.Tensor):
            raise TypeError("Tiered GLM replay IDs must be a tensor")
        if not isinstance(topk_weights, torch.Tensor):
            raise TypeError("Tiered GLM replay weights must be a tensor")
        if topk_ids.shape != expected_shape:
            raise ValueError(f"Tiered GLM replay IDs must have shape {expected_shape}")
        if topk_weights.shape != expected_shape:
            raise ValueError(
                f"Tiered GLM replay weights must have shape {expected_shape}"
            )
        if topk_ids.dtype != torch.int32:
            raise ValueError("Tiered GLM replay IDs must be int32")
        if topk_weights.dtype != torch.float32:
            raise ValueError("Tiered GLM replay weights must be float32")
        if not bool(torch.isfinite(topk_weights).all()):
            raise ValueError("Tiered GLM replay weights must be finite")
        if not bool((topk_weights >= 0).all()):
            raise ValueError("Tiered GLM replay weights must be nonnegative")
        for row in topk_ids:
            if any(len(set(layer.tolist())) != top_k for layer in row):
                raise ValueError("Tiered GLM replay IDs must be unique per row")


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
        batched_tensor_factory: TieredGlmBatchedTensorFactory | None = None,
        execution_callback: TieredGlmExecutionCallback | None = None,
        enable_route_replay_next_layer_prefetch: bool = False,
    ) -> None:
        self._adapter = adapter
        self._tensor_factory = tensor_factory
        self._batched_tensor_factory = batched_tensor_factory
        self.execution_callback = execution_callback
        self.enable_route_replay_next_layer_prefetch = (
            enable_route_replay_next_layer_prefetch
        )
        self._lock = threading.RLock()
        self._prefetch_lock = threading.RLock()
        self._prefetch_executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="glm-route-replay-prefetch",
            )
            if enable_route_replay_next_layer_prefetch
            else None
        )
        self._route_replay_prefetch: _EagerRouteReplayPrefetch | None = None
        self._route_replay_prefetch_requests = 0
        self._route_replay_prefetch_hits = 0
        self._route_replay_prefetch_failures = 0
        self._route_replay_prefetch_fallbacks = 0
        self._active_unit_counts: Counter[str] = Counter()
        self._active_expert_counts: Counter[tuple[int, int]] = Counter()
        self._active_bytes = 0
        self._peak_bytes = 0
        self._requests = 0
        self._loads = 0
        self._read_bytes = 0
        self._materialization_batches = 0
        self._fallback_unit_loads = 0
        self._provider_materialization_seconds = 0.0

    def prefetch_route_replay_next_layer_experts(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
    ) -> None:
        if not self.enable_route_replay_next_layer_prefetch:
            return
        del topk_ids

        route_controller = getattr(self, "route_controller", None)
        peek_layer_expert_ids = getattr(route_controller, "peek_layer_expert_ids", None)
        if not callable(peek_layer_expert_ids):
            self._record_route_replay_prefetch_failure()
            raise RuntimeError(
                "Tiered GLM deterministic route replay prefetch requires a route "
                "controller"
            )

        next_layer_id = layer_id + 1
        next_expert_ids = peek_layer_expert_ids(next_layer_id)
        if next_expert_ids is None:
            return

        tensor_specs = tuple(
            self._ordered_tensor_specs(
                {(next_layer_id, expert_id) for expert_id in next_expert_ids},
                next_layer_id,
            )
        )
        with self._prefetch_lock:
            if self._route_replay_prefetch is not None:
                self._record_route_replay_prefetch_failure()
                raise RuntimeError(
                    "Tiered GLM deterministic route replay already has an active "
                    "next-layer prefetch"
                )
            assert self._prefetch_executor is not None
            self._route_replay_prefetch = _EagerRouteReplayPrefetch(
                layer_id=next_layer_id,
                tensor_specs=tensor_specs,
                future=self._prefetch_executor.submit(
                    self._materialize_tensors,
                    tensor_specs,
                ),
            )
            self._route_replay_prefetch_requests += 1

    def _record_route_replay_prefetch_failure(self) -> None:
        with self._prefetch_lock:
            self._route_replay_prefetch_failures += 1

    def _consume_route_replay_prefetch(
        self,
        layer_id: int,
        requested_units: set[str],
    ) -> dict[str, torch.Tensor]:
        if not self.enable_route_replay_next_layer_prefetch:
            return {}

        with self._prefetch_lock:
            prefetch = self._route_replay_prefetch
            if prefetch is None or prefetch.layer_id != layer_id:
                if layer_id == _GLM_53_FIRST_SPARSE_LAYER:
                    return {}
                self._route_replay_prefetch_fallbacks += 1
                raise RuntimeError(
                    "Tiered GLM deterministic route replay prefetch was missing for "
                    f"layer {layer_id}"
                )
            if {spec[0] for spec in prefetch.tensor_specs} != requested_units:
                self._route_replay_prefetch_fallbacks += 1
                raise RuntimeError(
                    "Tiered GLM deterministic route replay prefetch does not match "
                    f"the layer-{layer_id} expert demand"
                )

        try:
            tensors = prefetch.future.result()
        except Exception:
            self._record_route_replay_prefetch_failure()
            raise

        with self._prefetch_lock:
            if self._route_replay_prefetch is not prefetch:
                self._route_replay_prefetch_fallbacks += 1
                raise RuntimeError("Tiered GLM route replay prefetch state changed")
            self._route_replay_prefetch = None
            self._route_replay_prefetch_hits += 1
        return {
            spec[0]: tensor
            for spec, tensor in zip(prefetch.tensor_specs, tensors, strict=True)
        }

    def drain_prefetches(self) -> None:
        with self._prefetch_lock:
            prefetch = self._route_replay_prefetch
        if prefetch is not None:
            try:
                prefetch.future.result()
            except Exception:
                self._record_route_replay_prefetch_failure()
                raise

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

        tensor_specs = self._ordered_tensor_specs(selected_experts, layer_id)
        prefetched_tensors = self._consume_route_replay_prefetch(
            layer_id,
            requested_units,
        )
        tensors = (
            prefetched_tensors
            if prefetched_tensors
            else {
                spec[0]: tensor
                for spec, tensor in zip(
                    tensor_specs,
                    self._materialize_tensors(tensor_specs),
                    strict=True,
                )
            }
        )
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
        with self._prefetch_lock:
            route_replay_prefetch = self._route_replay_prefetch
            route_replay_stats = {
                "route_replay_next_layer_prefetch_enabled": (
                    self.enable_route_replay_next_layer_prefetch
                ),
                "route_replay_next_layer_prefetch_requests": (
                    self._route_replay_prefetch_requests
                ),
                "route_replay_next_layer_prefetch_hits": (
                    self._route_replay_prefetch_hits
                ),
                "route_replay_next_layer_prefetch_failures": (
                    self._route_replay_prefetch_failures
                ),
                "route_replay_next_layer_prefetch_fallbacks": (
                    self._route_replay_prefetch_fallbacks
                ),
                "route_replay_next_layer_prefetch_active": (
                    route_replay_prefetch is not None
                ),
                "route_replay_next_layer_prefetch_bytes": (
                    route_replay_prefetch.byte_count
                    if route_replay_prefetch is not None
                    else 0
                ),
            }
        with self._lock:
            return {
                **route_replay_stats,
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
                "materialization_batches": self._materialization_batches,
                "materialization_fallback_loads": self._fallback_unit_loads,
                "provider_materialization_seconds": (
                    self._provider_materialization_seconds
                ),
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
        return tensor

    def _materialize_tensors(
        self,
        tensor_specs: Sequence[TieredGlmTensorSpec],
    ) -> list[torch.Tensor]:
        started = time.perf_counter()
        tensors: list[torch.Tensor] | None = None
        materialization_batch = False
        materialization_batch_count = 0
        materialized = False
        try:
            if self._batched_tensor_factory is not None:
                returned_tensors = self._batched_tensor_factory(tensor_specs)
                if returned_tensors is not None:
                    tensors = self._validate_batched_tensors(
                        tensor_specs,
                        returned_tensors,
                    )
                    materialization_batch = True
                    materialization_batch_count = (
                        returned_tensors.batch_count
                        if isinstance(returned_tensors, TieredGlmBatchedTensors)
                        else 1
                    )
            if tensors is None:
                tensors = [
                    self._materialize(
                        unit_id,
                        expected_shape,
                        expected_dtype,
                    )
                    for unit_id, expected_shape, expected_dtype in tensor_specs
                ]
            materialized = True
        finally:
            elapsed_seconds = time.perf_counter() - started
            with self._lock:
                self._provider_materialization_seconds += elapsed_seconds
                if materialized:
                    tensor_count = len(tensors) if tensors is not None else 0
                    if materialization_batch:
                        self._materialization_batches += materialization_batch_count
                    else:
                        self._fallback_unit_loads += tensor_count
                    self._loads += tensor_count
                    self._read_bytes += sum(
                        self._tensor_byte_size(tensor) for tensor in tensors or []
                    )
        if tensors is None:
            raise RuntimeError("Tiered GLM materialization did not produce tensors")
        return tensors

    def _validate_batched_tensors(
        self,
        tensor_specs: Sequence[TieredGlmTensorSpec],
        returned_tensors: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not isinstance(returned_tensors, Sequence):
            raise TypeError(
                "Tiered GLM batched tensor factory must return a tensor sequence"
            )
        tensors = list(returned_tensors)
        if len(tensors) != len(tensor_specs):
            raise ValueError(
                "Tiered GLM batched tensor factory returned "
                f"{len(tensors)} tensors; expected {len(tensor_specs)}"
            )
        for tensor, (unit_id, expected_shape, expected_dtype) in zip(
            tensors,
            tensor_specs,
            strict=True,
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Tiered GLM unit {unit_id!r} is not a tensor: {type(tensor)!r}"
                )
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
        return tensors

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

    _DEVICE_CACHE_POLICIES = frozenset(
        ("lru", "causal-turn-frequency-slru", "causal-weighted-last-n")
    )
    _CAUSAL_DEVICE_CACHE_POLICIES = frozenset(
        ("causal-turn-frequency-slru", "causal-weighted-last-n")
    )
    _STABLE_WINDOW = 3
    _STABLE_MIN_PRIOR_DEMAND = 1
    _PROTECTED_FRACTION = 0.44
    _TURN_FREQUENCY_THRESHOLD = 3
    _TURN_FREQUENCY_WEIGHT = 2.0
    _STABLE_PRIORITY_BONUS = 8.0
    _WARM_TURN_CAUSAL_CACHE_BUDGET_BYTES = 1_073_741_824
    _WARM_TURN_CAUSAL_CACHE_SOURCE = "previous_completed_turn"
    _WEIGHTED_LAST_N_DECAY = 0.5
    _WEIGHTED_LAST_N_SOURCE = "completed_prior_turns"

    def __init__(
        self,
        *,
        runtime: Any,
        adapter: Any,
        execution_callback: TieredGlmExecutionCallback | None = None,
        enable_prefetch: bool = False,
        enable_next_layer_prefetch: bool = False,
        next_layer_prediction_provider: Callable[
            [int, torch.Tensor],
            Sequence[int] | None,
        ]
        | None = None,
        prefetch_depth: int = 1,
        device_cache_budget_bytes: int | None = None,
        device_cache_policy: str = "lru",
        layer_quota_groups: int | None = None,
        layer_quota_bytes: int | None = None,
        enable_warm_turn_causal_cache: bool = False,
        dequant_cache_budget_bytes: int | None = None,
        enable_current_layer_chunked_overlap: bool = False,
        current_layer_chunk_in_flight_bytes: int | None = None,
        apply_timings: TieredGlmApplyTimings | None = None,
    ) -> None:
        if type(prefetch_depth) is not int or not 1 <= prefetch_depth <= 4:
            raise ValueError("prefetch_depth must be an integer from 1 to 4")
        if device_cache_budget_bytes is not None and (
            type(device_cache_budget_bytes) is not int or device_cache_budget_bytes < 0
        ):
            raise ValueError(
                "device_cache_budget_bytes must be a non-negative integer or None"
            )
        if device_cache_policy not in self._DEVICE_CACHE_POLICIES:
            raise ValueError(
                "device_cache_policy must be 'lru', 'causal-turn-frequency-slru', "
                "or 'causal-weighted-last-n'"
            )
        if layer_quota_groups is not None and (
            type(layer_quota_groups) is not int or layer_quota_groups <= 0
        ):
            raise ValueError("layer_quota_groups must be a positive integer or None")
        if layer_quota_bytes is not None and (
            type(layer_quota_bytes) is not int or layer_quota_bytes <= 0
        ):
            raise ValueError("layer_quota_bytes must be a positive integer or None")
        if (layer_quota_groups is None) != (layer_quota_bytes is None):
            raise ValueError("layer byte and group quotas must be provided together")
        if device_cache_policy in self._CAUSAL_DEVICE_CACHE_POLICIES and (
            layer_quota_groups is None or layer_quota_bytes is None
        ):
            raise ValueError(
                f"{device_cache_policy} requires layer byte and group quotas"
            )
        if device_cache_policy == "lru" and layer_quota_groups is not None:
            raise ValueError("layer quotas require causal-turn-frequency-slru")
        if type(enable_warm_turn_causal_cache) is not bool:
            raise ValueError("enable_warm_turn_causal_cache must be a boolean")
        if (
            enable_warm_turn_causal_cache
            and device_cache_policy == "causal-weighted-last-n"
        ):
            raise ValueError(
                "enable_warm_turn_causal_cache is incompatible with "
                "causal-weighted-last-n"
            )
        if enable_warm_turn_causal_cache:
            if device_cache_budget_bytes != self._WARM_TURN_CAUSAL_CACHE_BUDGET_BYTES:
                raise ValueError(
                    "warm-turn causal cache requires the unchanged "
                    "1,073,741,824-byte device-cache budget"
                )
            if device_cache_policy != "causal-turn-frequency-slru":
                raise ValueError(
                    "warm-turn causal cache requires causal-turn-frequency-slru"
                )
            if layer_quota_groups is None or layer_quota_bytes is None:
                raise ValueError("warm-turn causal cache requires layer quotas")
        if dequant_cache_budget_bytes is not None and (
            type(dequant_cache_budget_bytes) is not int
            or dequant_cache_budget_bytes < 0
        ):
            raise ValueError(
                "dequant_cache_budget_bytes must be a non-negative integer or None"
            )
        if type(enable_current_layer_chunked_overlap) is not bool:
            raise ValueError("enable_current_layer_chunked_overlap must be a boolean")
        if current_layer_chunk_in_flight_bytes is not None and (
            type(current_layer_chunk_in_flight_bytes) is not int
            or current_layer_chunk_in_flight_bytes <= 0
        ):
            raise ValueError(
                "current_layer_chunk_in_flight_bytes must be a positive integer or None"
            )
        if enable_current_layer_chunked_overlap:
            if current_layer_chunk_in_flight_bytes is None:
                raise ValueError(
                    "current-layer chunked overlap requires a positive "
                    "in-flight byte bound"
                )
            if enable_next_layer_prefetch:
                raise ValueError(
                    "current-layer chunked overlap cannot use next-layer prefetch"
                )
            if dequant_cache_budget_bytes is not None:
                raise ValueError(
                    "current-layer chunked overlap cannot use the dequant cache"
                )

        self._runtime = runtime
        self._adapter = adapter
        self.execution_callback = execution_callback
        self.enable_prefetch = enable_prefetch
        self.enable_next_layer_prefetch = enable_next_layer_prefetch
        self.enable_current_layer_chunked_overlap = enable_current_layer_chunked_overlap
        self.current_layer_chunk_in_flight_bytes = current_layer_chunk_in_flight_bytes
        self.next_layer_prediction_provider = next_layer_prediction_provider
        self.prefetch_depth = prefetch_depth
        self._prefetch_stats_lock = threading.Lock()
        self._prefetch_successes = 0
        self._prefetch_failures = 0
        self._next_layer_successes = 0
        self._next_layer_failures = 0
        self._next_layer_predictor = TieredGlmPreviousTokenLayerPredictor()
        self._next_layer_cancellations = 0
        self._prefetch_queue: deque[tuple[int, tuple[int, ...]]] = deque()
        self._prefetch_queued_layer_ids: set[int] = set()
        self._prefetch_queue_lock = threading.RLock()
        self._prefetch_submission_active = False
        self._active_prefetches: dict[int, Any] = {}
        self._active_prefetch_layer_ids: dict[int, int] = {}
        self._device_cache_budget_bytes = device_cache_budget_bytes
        self._device_cache_policy = device_cache_policy
        self._layer_quota_groups = layer_quota_groups
        self._layer_quota_bytes = layer_quota_bytes
        self.enable_warm_turn_causal_cache = enable_warm_turn_causal_cache
        self.apply_timings = apply_timings
        self._device_cache: OrderedDict[
            _GlmDeviceCacheKey,
            _GlmDeviceCacheEntry,
        ] = OrderedDict()
        self._device_cache_groups: OrderedDict[
            _GlmDeviceCacheGroupKey,
            set[_GlmDeviceCacheKey],
        ] = OrderedDict()
        self._device_cache_group_sizes: dict[_GlmDeviceCacheGroupKey, int] = {}
        self._device_cache_group_bytes: dict[_GlmDeviceCacheGroupKey, int] = {}
        self._device_cache_layer_bytes: dict[int, int] = {}
        self._device_cache_layer_group_counts: dict[int, int] = {}
        self._device_cache_lock = threading.RLock()
        self._device_cache_bytes = 0
        self._device_cache_peak_bytes = 0
        self._device_cache_hits = 0
        self._device_cache_misses = 0
        self._device_cache_evictions = 0
        self._device_cache_active_leases = 0
        self._current_turn_groups: set[_GlmDeviceCacheGroupKey] = set()
        self._current_turn_demand_counts: dict[_GlmDeviceCacheGroupKey, int] = {}
        self._previous_turn_groups: deque[set[_GlmDeviceCacheGroupKey]] = deque(
            maxlen=self._STABLE_WINDOW
        )
        self._previous_turn_demand_counts: deque[dict[_GlmDeviceCacheGroupKey, int]] = (
            deque(maxlen=self._STABLE_WINDOW)
        )
        self._stable_pinned_groups: set[_GlmDeviceCacheGroupKey] = set()
        self._causal_pinned_groups: set[_GlmDeviceCacheGroupKey] = set()
        self._protected_groups: set[_GlmDeviceCacheGroupKey] = set()
        self._protected_group_target = (
            max(1, int(layer_quota_groups * self._PROTECTED_FRACTION))
            if layer_quota_groups is not None
            else 0
        )
        self._sequence_turn_index = 0
        self._dequant_cache_budget_bytes = dequant_cache_budget_bytes
        self._dequant_cache = DerivedTensorCache(
            dequant_cache_budget_bytes,
            self._reserve_dequant_scratch,
            metric_prefix="dequant_cache",
        )
        self._dequant_cache_closed = False
        self._current_layer_chunk_schedulers: dict[int, Any] = {}
        self._current_layer_chunk_scheduler_lock = threading.RLock()
        self._current_layer_projection_seconds = 0.0
        self._current_layer_timing_lock = threading.Lock()

    @property
    def uses_derived_cache(self) -> bool:
        return (
            self._device_cache_budget_bytes is not None
            or self._dequant_cache_budget_bytes is not None
        )

    @contextmanager
    def warmup_without_derived_caches(self):
        """Run one warm-up without allocating derived-cache entries."""
        clear_derived_payload_caches = getattr(
            self._runtime,
            "clear_derived_payload_caches",
            None,
        )
        if callable(clear_derived_payload_caches):
            clear_derived_payload_caches()
        with self._device_cache_lock:
            if self._device_cache_active_leases != 0 or self._device_cache:
                if self._device_cache_active_leases != 0:
                    raise RuntimeError(
                        "cannot start single-provider warm-up with an active "
                        "device-cache lease"
                    )
                while self._device_cache:
                    self._remove_cached_projection(next(iter(self._device_cache)))
        dequant_stats = self._dequant_cache.stats()
        if (
            dequant_stats.get("dequant_cache_active_leases", 0) != 0
            or dequant_stats.get("dequant_cache_bytes", 0) != 0
        ):
            if dequant_stats.get("dequant_cache_active_leases", 0) != 0:
                raise RuntimeError(
                    "cannot start single-provider warm-up with an active "
                    "dequant-cache lease"
                )
            self._dequant_cache.evict_all()

        device_cache_budget_bytes = self._device_cache_budget_bytes
        dequant_cache_budget_bytes = self._dequant_cache_budget_bytes
        dequant_cache_internal_budget_bytes = self._dequant_cache._budget_bytes
        self._device_cache_budget_bytes = None
        self._dequant_cache_budget_bytes = None
        self._dequant_cache._budget_bytes = None
        try:
            yield
        finally:
            self._device_cache_budget_bytes = device_cache_budget_bytes
            self._dequant_cache_budget_bytes = dequant_cache_budget_bytes
            self._dequant_cache._budget_bytes = dequant_cache_internal_budget_bytes
            if callable(clear_derived_payload_caches):
                clear_derived_payload_caches()

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
        for group_key in sorted(selected_experts):
            self._record_device_cache_policy_access(group_key)

        uncached_expert_ids: list[int] = []
        cached_experts: dict[
            int,
            tuple[tuple[str, ...], dict[str, TieredGlmProjection]],
        ] = {}
        cache_keys: list[_GlmDeviceCacheKey] = []
        dequant_cache_keys: list[_GlmDequantCacheKey] = []
        lease_stack = ExitStack()
        experts: dict[int, dict[str, TieredGlmProjection]] = {}

        try:
            for _, expert_id in sorted(selected_experts):
                projection_names = self._expert_projection_names(
                    expert_id,
                    layer_id=layer_id,
                )
                projections: dict[str, TieredGlmProjection] = {}
                for projection_name in projection_names:
                    cache_key = (layer_id, expert_id, projection_name)
                    cached_projection = self._acquire_cached_projection(cache_key)
                    if cached_projection is not None:
                        cache_keys.append(cache_key)
                        dequant_cache_key = (
                            cached_projection.weight_unit_id,
                            cached_projection.scale_unit_id,
                        )
                        cached_projection = self._acquire_dequantized_projection(
                            dequant_cache_key,
                            cached_projection,
                        )
                        if cached_projection.dequantized_weight is not None:
                            dequant_cache_keys.append(dequant_cache_key)
                        projections[projection_name] = cached_projection

                if len(projections) == len(projection_names):
                    cached_experts[expert_id] = (projection_names, projections)
                    experts[expert_id] = projections
                    continue

                uncached_expert_ids.append(expert_id)
                cached_experts[expert_id] = (projection_names, projections)

            if uncached_expert_ids:
                leases = lease_stack.enter_context(
                    self._runtime.acquire_experts(
                        uncached_expert_ids,
                        layer_id=layer_id,
                    )
                )
                leased_projections = self._build_experts(leases)

            all_projections_cached = True
            for expert_id in uncached_expert_ids:
                projection_names, partial_projections = cached_experts.get(
                    expert_id,
                    ((), {}),
                )
                if not projection_names:
                    projection_names = self._expert_projection_names(
                        expert_id,
                        layer_id=layer_id,
                    )
                projections = dict(partial_projections)
                for projection_name in projection_names:
                    if projection_name in projections:
                        continue

                    cache_key = (layer_id, expert_id, projection_name)
                    cached_projection = self._cache_projection(
                        cache_key,
                        leased_projections[expert_id][projection_name],
                    )
                    if cached_projection is not None:
                        cache_keys.append(cache_key)
                    else:
                        all_projections_cached = False
                        cached_projection = leased_projections[expert_id][
                            projection_name
                        ]

                    dequant_cache_key = (
                        cached_projection.weight_unit_id,
                        cached_projection.scale_unit_id,
                    )
                    cached_projection = self._acquire_dequantized_projection(
                        dequant_cache_key,
                        cached_projection,
                    )
                    if cached_projection.dequantized_weight is not None:
                        dequant_cache_keys.append(dequant_cache_key)
                    projections[projection_name] = cached_projection

                experts[expert_id] = projections
            if all_projections_cached:
                lease_stack.close()
        except BaseException:
            lease_stack.close()
            for cache_key in reversed(cache_keys):
                self._release_cached_projection(cache_key)
            for dequant_cache_key in reversed(dequant_cache_keys):
                self._release_dequantized_projection(dequant_cache_key)
            raise

        def release() -> None:
            lease_stack.close()
            for cache_key in reversed(cache_keys):
                self._release_cached_projection(cache_key)
            for dequant_cache_key in reversed(dequant_cache_keys):
                self._release_dequantized_projection(dequant_cache_key)

        return TieredGlmResidentExperts(experts, release)

    def run_current_layer_expert_chunks(
        self,
        layer_id: int,
        execution_chunks: Sequence[Sequence[int]],
        callback: Callable[[int, TieredGlmExpertProjections, tuple[int, ...]], Any],
    ) -> list[Any]:
        if not self.enable_current_layer_chunked_overlap:
            raise RuntimeError(
                "current-layer chunked overlap is not enabled for this provider"
            )
        if type(layer_id) is not int:
            raise TypeError("current-layer chunk layer ID must be an integer")
        normalized_chunks = tuple(tuple(chunk) for chunk in execution_chunks)
        if not normalized_chunks or any(not chunk for chunk in normalized_chunks):
            raise ValueError("current-layer execution chunks must be non-empty")
        if any(
            any(type(expert_id) is not int for expert_id in chunk)
            for chunk in normalized_chunks
        ):
            raise ValueError("current-layer execution chunks must contain integers")

        scheduler = self._current_layer_chunk_scheduler(layer_id)
        wait_seconds_before = scheduler.stats()["wait_seconds"]
        with self._current_layer_timing_lock:
            projection_seconds_before = self._current_layer_projection_seconds
        chunk_index = 0
        projection_ready_events = [threading.Event() for _ in normalized_chunks]
        projection_ready_events[0].set()
        projection_cancelled = threading.Event()
        prepare_index = 0
        prepare_index_lock = threading.Lock()

        if self._device_cache_budget_bytes is None:
            prepare = None
        else:

            def prepare(expert_ids: Sequence[int]):
                nonlocal prepare_index
                with prepare_index_lock:
                    current_prepare_index = prepare_index
                    prepare_index += 1
                if current_prepare_index != 0:
                    projection_ready_events[current_prepare_index].wait()
                if projection_cancelled.is_set():
                    raise RuntimeError("current-layer chunk preparation was cancelled")
                return self._prepare_current_layer_chunk(layer_id, expert_ids)

        def projection(
            lease_set: Any,
            context: _CurrentLayerChunkCacheContext | None = None,
        ) -> TieredGlmExpertProjections:
            started = time.perf_counter()
            try:
                if context is None:
                    return self._build_experts(tuple(lease_set))
                return self._combine_current_layer_chunk_projections(
                    tuple(lease_set),
                    context,
                )
            except BaseException:
                projection_cancelled.set()
                for projection_ready_event in projection_ready_events:
                    projection_ready_event.set()
                raise
            finally:
                elapsed_seconds = time.perf_counter() - started
                if chunk_index + 1 < len(projection_ready_events):
                    projection_ready_events[chunk_index + 1].set()
                with self._current_layer_timing_lock:
                    self._current_layer_projection_seconds += elapsed_seconds

        def scheduled_callback(experts: TieredGlmExpertProjections) -> Any:
            nonlocal chunk_index
            current_index = chunk_index
            chunk_index += 1
            return callback(current_index, experts, normalized_chunks[current_index])

        results = scheduler.run(
            normalized_chunks,
            scheduled_callback,
            projection=projection,
            prepare=prepare,
            release=self._release_current_layer_chunk_cache_context,
        )
        wait_seconds = scheduler.stats()["wait_seconds"] - wait_seconds_before
        if self.apply_timings is not None:
            with self._current_layer_timing_lock:
                projection_seconds = (
                    self._current_layer_projection_seconds - projection_seconds_before
                )
            self.apply_timings.record(
                "provider_request",
                wait_seconds + projection_seconds,
            )
        return results

    def _prepare_current_layer_chunk(
        self,
        layer_id: int,
        expert_ids: Sequence[int],
    ) -> DeviceWeightCurrentLayerChunkPreparation:
        cached_projections: dict[int, dict[str, TieredGlmProjection]] = {}
        projection_names: dict[int, tuple[str, ...]] = {}
        uncached_expert_ids: list[int] = []
        cache_keys: list[tuple[int, int, str]] = []
        try:
            for expert_id in sorted(expert_ids):
                self._record_device_cache_policy_access((layer_id, expert_id))
                names = self._expert_projection_names(
                    expert_id,
                    layer_id=layer_id,
                )
                projection_names[expert_id] = names
                projections: dict[str, TieredGlmProjection] = {}
                for projection_name in names:
                    cache_key = (layer_id, expert_id, projection_name)
                    cached_projection = self._acquire_cached_projection(cache_key)
                    if cached_projection is not None:
                        cache_keys.append(cache_key)
                        projections[projection_name] = cached_projection
                cached_projections[expert_id] = projections
                if len(projections) != len(names):
                    uncached_expert_ids.append(expert_id)
        except BaseException:
            for cache_key in reversed(cache_keys):
                self._release_cached_projection(cache_key)
            raise

        context = _CurrentLayerChunkCacheContext(
            layer_id=layer_id,
            cached_projections=cached_projections,
            projection_names=projection_names,
            uncached_expert_ids=tuple(uncached_expert_ids),
            cache_keys=cache_keys,
        )
        return DeviceWeightCurrentLayerChunkPreparation(
            uncached_expert_ids=context.uncached_expert_ids,
            context=context,
        )

    def _combine_current_layer_chunk_projections(
        self,
        lease_set: Any,
        context: _CurrentLayerChunkCacheContext,
    ) -> TieredGlmExpertProjections:
        leased_projections = self._build_experts(tuple(lease_set))
        experts: TieredGlmExpertProjections = {}
        for expert_id, projection_names in context.projection_names.items():
            projections = dict(context.cached_projections[expert_id])
            for projection_name in projection_names:
                if projection_name in projections:
                    continue
                projection = leased_projections[expert_id][projection_name]
                cache_key = (context.layer_id, expert_id, projection_name)
                cached_projection = self._cache_projection(cache_key, projection)
                if cached_projection is not None:
                    context.cache_keys.append(cache_key)
                    projection = cached_projection
                projections[projection_name] = projection
            experts[expert_id] = projections
        return experts

    def _release_current_layer_chunk_cache_context(
        self,
        context: _CurrentLayerChunkCacheContext,
    ) -> None:
        for cache_key in reversed(context.cache_keys):
            self._release_cached_projection(cache_key)

    def _current_layer_chunk_scheduler(self, layer_id: int) -> Any:
        with self._current_layer_chunk_scheduler_lock:
            scheduler = self._current_layer_chunk_schedulers.get(layer_id)
            if scheduler is None:
                scheduler = DeviceWeightCurrentLayerChunkScheduler(
                    self._runtime,
                    layer_id=layer_id,
                    max_in_flight_bytes=self.current_layer_chunk_in_flight_bytes,
                )
                self._current_layer_chunk_schedulers[layer_id] = scheduler
            return scheduler

    def _close_current_layer_chunk_schedulers(self) -> None:
        with self._current_layer_chunk_scheduler_lock:
            schedulers = tuple(self._current_layer_chunk_schedulers.items())
        for layer_id, scheduler in schedulers:
            scheduler.close()
            with self._current_layer_chunk_scheduler_lock:
                if self._current_layer_chunk_schedulers.get(layer_id) is scheduler:
                    del self._current_layer_chunk_schedulers[layer_id]

    def prefetch_experts(self, demands: Sequence[Any]) -> None:
        if not self.enable_prefetch:
            return

        try:
            if not demands:
                raise ValueError("Tiered GLM prefetch demands must not be empty")

            unit_ids = [demand.unit_id for demand in demands]
            if len(set(unit_ids)) != len(unit_ids):
                raise ValueError("Tiered GLM prefetch demands contain duplicate units")

            requested_units = set(unit_ids)
            selected_experts = self._selected_experts(requested_units)
            layer_id = _validate_exact_expert_demands(
                self._adapter,
                selected_experts,
                requested_units,
            )
            expert_ids = [expert_id for _, expert_id in sorted(selected_experts)]
            self._runtime.prefetch_experts(expert_ids, layer_id=layer_id)
        except Exception:
            with self._prefetch_stats_lock:
                self._prefetch_failures += 1
            return

        with self._prefetch_stats_lock:
            self._prefetch_successes += 1

    def begin_runtime_pass(self) -> None:
        begin_runtime_pass = getattr(
            self._runtime,
            "begin_runtime_pass",
            None,
        )
        if not callable(begin_runtime_pass):
            raise ValueError("Tiered GLM runtime does not support runtime passes")
        begin_runtime_pass()
        if self.enable_warm_turn_causal_cache:
            with self._device_cache_lock:
                self._refresh_warm_turn_causal_selection_locked()

    def begin_sequence_turn(self) -> None:
        begin_sequence_turn = getattr(
            self._runtime,
            "begin_sequence_turn",
            None,
        )
        if not callable(begin_sequence_turn):
            raise ValueError("Tiered GLM runtime does not support sequence turns")
        begin_sequence_turn()
        if self._device_cache_policy not in self._CAUSAL_DEVICE_CACHE_POLICIES:
            return

        with self._device_cache_lock:
            if self._current_turn_groups:
                self._previous_turn_groups.append(set(self._current_turn_groups))
                self._previous_turn_demand_counts.append(
                    dict(self._current_turn_demand_counts)
                )
                self._sequence_turn_index += 1
            self._current_turn_groups.clear()
            self._current_turn_demand_counts.clear()
            if self._device_cache_policy == "causal-weighted-last-n":
                self._refresh_weighted_last_n_selection_locked()
            elif self.enable_warm_turn_causal_cache:
                self._refresh_warm_turn_causal_selection_locked()
            else:
                self._stable_pinned_groups = self._select_stable_groups_locked()
                self._causal_pinned_groups = set(self._stable_pinned_groups)
                self._protected_groups = {
                    group_key
                    for group_key in self._stable_pinned_groups
                    if group_key in self._device_cache_groups
                }

    def prefetch_next_layer_experts(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
    ) -> None:
        if not self.enable_next_layer_prefetch:
            return

        try:
            if topk_ids.ndim != 2:
                raise ValueError("Tiered GLM next-layer routing IDs must be rank 2")

            prediction_provider = self.next_layer_prediction_provider
            if prediction_provider is None:
                predicted_expert_sets = self._next_layer_predictor.predict(
                    layer_id,
                    topk_ids,
                    depth=self.prefetch_depth,
                )
            else:
                predicted_expert_sets = []
                for future_offset in range(1, self.prefetch_depth + 1):
                    future_layer_id = _glm_53_future_layer_id(
                        layer_id,
                        future_offset,
                    )
                    prediction = prediction_provider(future_layer_id, topk_ids)
                    predicted_expert_sets.append(
                        None if prediction is None else list(prediction)
                    )

            with self._prefetch_queue_lock:
                self._next_layer_cancellations += len(self._prefetch_queue)
                self._prefetch_queue.clear()
                self._prefetch_queued_layer_ids.clear()

                remaining_capacity = self.prefetch_depth - len(self._active_prefetches)
                active_layer_ids = set(self._active_prefetch_layer_ids.values())
                predicted_layer_ids: set[int] = set()
                for future_offset, predicted_expert_ids in enumerate(
                    predicted_expert_sets,
                    start=1,
                ):
                    if remaining_capacity <= 0 or predicted_expert_ids is None:
                        continue

                    if prediction_provider is None:
                        future_layer_id, predicted_expert_ids = predicted_expert_ids
                    else:
                        future_layer_id = _glm_53_future_layer_id(
                            layer_id,
                            future_offset,
                        )
                    if (
                        future_layer_id in active_layer_ids
                        or future_layer_id in predicted_layer_ids
                    ):
                        continue

                    expert_ids = tuple(
                        sorted(
                            dict.fromkeys(
                                int(expert_id) for expert_id in predicted_expert_ids
                            )
                        )
                    )
                    if not expert_ids:
                        continue

                    self._prefetch_queue.append((future_layer_id, expert_ids))
                    self._prefetch_queued_layer_ids.add(future_layer_id)
                    predicted_layer_ids.add(future_layer_id)
                    remaining_capacity -= 1

            self._submit_next_prefetch()
        except Exception:
            with self._prefetch_stats_lock:
                self._next_layer_failures += 1
            return

    def _submit_next_prefetch(self) -> None:
        with self._prefetch_queue_lock:
            if self._prefetch_submission_active:
                return
            self._prefetch_submission_active = True
        submit_prefetch = self._select_prefetch_submission_method()
        try:
            while True:
                with self._prefetch_queue_lock:
                    if (
                        len(self._active_prefetches) >= self.prefetch_depth
                        or not self._prefetch_queue
                    ):
                        return
                    future_layer_id, expert_ids = self._prefetch_queue[0]
                    placeholder = object()
                    placeholder_id = id(placeholder)
                    self._active_prefetches[placeholder_id] = placeholder
                    self._active_prefetch_layer_ids[placeholder_id] = future_layer_id
                    self._prefetch_queued_layer_ids.discard(future_layer_id)

                try:
                    future = submit_prefetch(
                        expert_ids,
                        layer_id=future_layer_id,
                    )
                except Exception:
                    with self._prefetch_queue_lock:
                        self._active_prefetches.pop(placeholder_id, None)
                        self._active_prefetch_layer_ids.pop(placeholder_id, None)
                        self._prefetch_queue.popleft()
                        self._prefetch_queued_layer_ids.discard(future_layer_id)
                    with self._prefetch_stats_lock:
                        self._next_layer_failures += 1
                    continue

                with self._prefetch_queue_lock:
                    self._active_prefetches.pop(placeholder_id, None)
                    self._active_prefetch_layer_ids.pop(placeholder_id, None)
                    self._active_prefetches[id(future)] = future
                    self._active_prefetch_layer_ids[id(future)] = future_layer_id
                    self._prefetch_queue.popleft()
                with self._prefetch_stats_lock:
                    self._next_layer_successes += 1
                future.add_done_callback(self._on_prefetch_completion)
        finally:
            with self._prefetch_queue_lock:
                self._prefetch_submission_active = False

    def _select_prefetch_submission_method(self) -> Any:
        submit_expert_cache_prefetch = getattr(
            self._runtime,
            "submit_expert_cache_prefetch",
            None,
        )
        if callable(submit_expert_cache_prefetch):
            try:
                runtime_stats = self._runtime.stats()
            except Exception:
                return self._runtime.submit_prefetch_experts
            if runtime_stats.get("expert_cache_budget_bytes") is not None:
                return submit_expert_cache_prefetch
        return self._runtime.submit_prefetch_experts

    def _on_prefetch_completion(self, future: Any) -> None:
        with self._prefetch_queue_lock:
            self._active_prefetches.pop(id(future), None)
            self._active_prefetch_layer_ids.pop(id(future), None)
        self._submit_next_prefetch()

    def drain_prefetches(self) -> None:
        """Wait for active and queued prefetch work without closing runtime."""
        while True:
            self._submit_next_prefetch()
            self._runtime.drain_prefetches()
            with self._prefetch_queue_lock:
                if not self._prefetch_queue and not self._active_prefetches:
                    return

    def close_prefetches(self, *, wait: bool = True) -> None:
        self._close_current_layer_chunk_schedulers()
        close_prefetches = getattr(self._runtime, "close_prefetches", None)
        if callable(close_prefetches):
            close_prefetches(wait=wait)
        self._close_dequant_cache()

    def clear(self) -> None:
        runtime_stats = self._runtime.stats()
        if runtime_stats.get("active_leases", 0) != 0:
            raise RuntimeError(
                "cannot clear Tiered GLM provider with active runtime leases: "
                f"{runtime_stats['active_leases']}"
            )
        dequant_stats = self._dequant_cache.stats()
        if dequant_stats.get("dequant_cache_active_leases", 0) != 0:
            raise RuntimeError(
                "cannot clear Tiered GLM provider with active derived-cache leases: "
                f"{dequant_stats['dequant_cache_active_leases']}"
            )
        with self._device_cache_lock:
            if self._device_cache_active_leases != 0:
                raise RuntimeError(
                    "cannot clear Tiered GLM provider with active device-cache leases: "
                    f"{self._device_cache_active_leases}"
                )

        self._close_current_layer_chunk_schedulers()
        self.drain_prefetches()
        self.close_prefetches(wait=True)
        with self._device_cache_lock:
            if self._device_cache_active_leases != 0:
                raise RuntimeError(
                    "cannot clear Tiered GLM provider with active device-cache leases: "
                    f"{self._device_cache_active_leases}"
                )
            while self._device_cache:
                cache_key = next(iter(self._device_cache))
                self._remove_cached_projection(cache_key)
            if (
                self._device_cache_policy == "causal-weighted-last-n"
                or self.enable_warm_turn_causal_cache
            ):
                self._reset_turn_causal_history_locked()

        runtime_clear = getattr(self._runtime, "clear", None)
        if callable(runtime_clear):
            runtime_clear()

    def reserve_execution_scratch(self, scratch_bytes: int):
        reserve_execution_scratch = getattr(
            self._runtime,
            "reserve_execution_scratch",
            None,
        )
        if callable(reserve_execution_scratch):
            return reserve_execution_scratch(scratch_bytes)
        return self._runtime.reserve_scratch(scratch_bytes)

    def _reserve_dequant_scratch(self, scratch_bytes: int):
        return self._runtime.reserve_scratch(scratch_bytes)

    def stats(self) -> dict[str, int | float | str | None]:
        stats = dict(self._runtime.stats())
        callback_stats_method = getattr(self.execution_callback, "stats", None)
        if callable(callback_stats_method):
            callback_stats = callback_stats_method()
            if not isinstance(callback_stats, dict):
                raise RuntimeError("Tiered GLM execution callback stats must be a dict")
            overlapping_keys = [key for key in stats if key in callback_stats]
            if overlapping_keys:
                raise RuntimeError(
                    "Tiered GLM execution callback stats overlap runtime stats: "
                    f"{overlapping_keys}"
                )
            stats.update(callback_stats)
        if self.apply_timings is not None:
            stats.update(self.apply_timings.metrics())
        with self._prefetch_stats_lock:
            stats["prefetch_successes"] = self._prefetch_successes
            stats["prefetch_failures"] = self._prefetch_failures
            stats["next_layer_prefetch_successes"] = self._next_layer_successes
            stats["next_layer_prefetch_failures"] = self._next_layer_failures
            stats["next_layer_prefetch_cancellations"] = self._next_layer_cancellations
            stats["next_layer_prefetch_active"] = len(self._active_prefetches)
            stats["next_layer_prefetch_queued"] = len(self._prefetch_queue)
        with self._device_cache_lock:
            stats["device_cache_hits"] = self._device_cache_hits
            stats["device_cache_misses"] = self._device_cache_misses
            stats["device_cache_evictions"] = self._device_cache_evictions
            stats["device_cache_bytes"] = self._device_cache_bytes
            stats["device_cache_peak_bytes"] = self._device_cache_peak_bytes
            stats["device_cache_budget_bytes"] = self._device_cache_budget_bytes
            stats["device_cache_active_leases"] = self._device_cache_active_leases
            stats["device_cache_policy"] = self._device_cache_policy
            stats["device_cache_layer_quota_groups"] = self._layer_quota_groups
            stats["device_cache_layer_quota_bytes"] = self._layer_quota_bytes
            stats["device_cache_pinned_groups"] = len(self._causal_pinned_groups)
            stats["device_cache_protected_groups"] = len(self._protected_groups)
            stats["device_cache_sequence_turns"] = self._sequence_turn_index
            stats["device_cache_history_turns"] = (
                len(self._previous_turn_groups)
                if self._device_cache_policy == "causal-weighted-last-n"
                else 0
            )
            stats["device_cache_selected_groups"] = (
                len(self._causal_pinned_groups)
                if self._device_cache_policy == "causal-weighted-last-n"
                else 0
            )
            stats["device_cache_selection_source"] = (
                self._WEIGHTED_LAST_N_SOURCE
                if self._device_cache_policy == "causal-weighted-last-n"
                else None
            )
            stats["warm_turn_causal_cache_enabled"] = self.enable_warm_turn_causal_cache
            stats["warm_turn_causal_cache_prior_turn_pin_count"] = len(
                self._causal_pinned_groups
            )
            stats["warm_turn_causal_cache_protected_groups"] = len(
                self._protected_groups
            )
            stats["warm_turn_causal_cache_protected_bytes"] = sum(
                self._device_cache_group_bytes.get(group_key, 0)
                for group_key in self._protected_groups
            )
            stats["warm_turn_causal_cache_source"] = (
                self._WARM_TURN_CAUSAL_CACHE_SOURCE
                if self.enable_warm_turn_causal_cache
                else None
            )
        stats.update(self._dequant_cache.stats())
        with self._current_layer_chunk_scheduler_lock:
            schedulers = tuple(self._current_layer_chunk_schedulers.values())
        scheduler_stats = [scheduler.stats() for scheduler in schedulers]
        stats["current_layer_chunked_overlap_enabled"] = (
            self.enable_current_layer_chunked_overlap
        )
        stats["current_layer_chunk_in_flight_bytes"] = (
            self.current_layer_chunk_in_flight_bytes
        )
        stats["current_layer_chunk_scheduler_layers"] = len(schedulers)
        stats["current_layer_chunk_planned_groups"] = sum(
            scheduler_stats["planned_groups"] for scheduler_stats in scheduler_stats
        )
        stats["current_layer_chunk_execution_chunks"] = sum(
            scheduler_stats["execution_chunks"] for scheduler_stats in scheduler_stats
        )
        stats["current_layer_chunk_read_chunks"] = sum(
            scheduler_stats["read_chunks"] for scheduler_stats in scheduler_stats
        )
        stats["current_layer_chunk_useful_read_chunks"] = sum(
            scheduler_stats["useful_read_chunks"] for scheduler_stats in scheduler_stats
        )
        stats["current_layer_chunk_stalled_read_chunks"] = sum(
            scheduler_stats["stalled_read_chunks"]
            for scheduler_stats in scheduler_stats
        )
        stats["current_layer_chunk_overlap_seconds"] = sum(
            scheduler_stats["overlap_seconds"] for scheduler_stats in scheduler_stats
        )
        stats["current_layer_chunk_wait_seconds"] = sum(
            scheduler_stats["wait_seconds"] for scheduler_stats in scheduler_stats
        )
        stats["current_layer_chunk_in_flight_current_bytes"] = sum(
            scheduler_stats["in_flight_bytes"] for scheduler_stats in scheduler_stats
        )
        stats["current_layer_chunk_reserved_device_bytes"] = sum(
            scheduler_stats["reserved_device_bytes"]
            for scheduler_stats in scheduler_stats
        )
        with self._current_layer_timing_lock:
            stats["current_layer_projection_seconds"] = (
                self._current_layer_projection_seconds
            )
        return stats

    def _expert_projection_names(
        self,
        expert_id: int,
        *,
        layer_id: int,
    ) -> tuple[str, ...]:
        projection_names = tuple(
            unit_id.rsplit(".", maxsplit=2)[-2]
            for unit_id in self._adapter.expert_unit_ids(
                expert_id,
                layer_id=layer_id,
            )
            if unit_id.endswith(".weight")
        )
        if len(projection_names) != 3 or len(set(projection_names)) != 3:
            raise ValueError(
                f"Tiered GLM expert {expert_id} does not have three projections"
            )
        return projection_names

    def _record_device_cache_policy_access(
        self,
        group_key: _GlmDeviceCacheGroupKey,
    ) -> None:
        if self._device_cache_policy not in self._CAUSAL_DEVICE_CACHE_POLICIES:
            return

        with self._device_cache_lock:
            self._current_turn_groups.add(group_key)
            demand_count = self._current_turn_demand_counts.get(group_key, 0) + 1
            self._current_turn_demand_counts[group_key] = demand_count
            if (
                self.enable_warm_turn_causal_cache
                or self._device_cache_policy == "causal-weighted-last-n"
            ):
                return
            if demand_count < self._TURN_FREQUENCY_THRESHOLD:
                return

            self._causal_pinned_groups = self._select_causal_groups_locked()
            self._protected_groups.update(
                group_key
                for group_key in self._causal_pinned_groups
                if group_key in self._device_cache_groups
            )

    def _select_stable_groups_locked(
        self,
    ) -> set[_GlmDeviceCacheGroupKey]:
        if len(self._previous_turn_groups) < self._STABLE_WINDOW:
            return set()

        stable_groups = set(self._previous_turn_groups[-1])
        for previous_groups in self._previous_turn_groups:
            stable_groups &= previous_groups

        prior_counts: dict[_GlmDeviceCacheGroupKey, int] = {}
        for turn_counts in self._previous_turn_demand_counts:
            for group_key, demand_count in turn_counts.items():
                prior_counts[group_key] = prior_counts.get(group_key, 0) + demand_count

        stable_groups = {
            group_key
            for group_key in stable_groups
            if prior_counts.get(group_key, 0) >= self._STABLE_MIN_PRIOR_DEMAND
        }
        return self._select_quoted_groups_locked(
            stable_groups,
            lambda group_key: (-prior_counts.get(group_key, 0), group_key),
        )

    def _select_causal_groups_locked(
        self,
    ) -> set[_GlmDeviceCacheGroupKey]:
        scores: dict[_GlmDeviceCacheGroupKey, float] = {}
        for group_key in self._stable_pinned_groups:
            scores[group_key] = self._STABLE_PRIORITY_BONUS
        for group_key, demand_count in self._current_turn_demand_counts.items():
            if demand_count >= self._TURN_FREQUENCY_THRESHOLD:
                scores[group_key] = (
                    scores.get(group_key, 0.0)
                    + self._TURN_FREQUENCY_WEIGHT * demand_count
                )

        return self._select_quoted_groups_locked(
            set(scores),
            lambda group_key: (-scores[group_key], group_key),
        )

    def _select_quoted_groups_locked(
        self,
        groups: set[_GlmDeviceCacheGroupKey],
        sort_key: Callable[[_GlmDeviceCacheGroupKey], tuple[float, ...]],
    ) -> set[_GlmDeviceCacheGroupKey]:
        selected_groups: set[_GlmDeviceCacheGroupKey] = set()
        selected_layer_bytes: dict[int, int] = {}
        selected_layer_groups: dict[int, int] = {}
        for group_key in sorted(groups, key=sort_key):
            layer_id = group_key[0]
            group_bytes = self._device_cache_group_size_locked(group_key)
            if (
                selected_layer_bytes.get(layer_id, 0) + group_bytes
                > self._layer_quota_bytes
                or selected_layer_groups.get(layer_id, 0) + 1 > self._layer_quota_groups
            ):
                continue
            selected_groups.add(group_key)
            selected_layer_bytes[layer_id] = (
                selected_layer_bytes.get(layer_id, 0) + group_bytes
            )
            selected_layer_groups[layer_id] = selected_layer_groups.get(layer_id, 0) + 1
        return selected_groups

    def _refresh_warm_turn_causal_selection_locked(self) -> None:
        if not self._previous_turn_groups:
            self._stable_pinned_groups.clear()
            self._causal_pinned_groups.clear()
            self._protected_groups.clear()
            return

        prior_groups = self._previous_turn_groups[-1]
        prior_demand_counts = self._previous_turn_demand_counts[-1]
        selected_groups = self._select_quoted_groups_locked(
            set(prior_groups),
            lambda group_key: (
                -prior_demand_counts.get(group_key, 0),
                group_key,
            ),
        )
        self._stable_pinned_groups.clear()
        self._causal_pinned_groups = selected_groups
        self._protected_groups = {
            group_key
            for group_key in selected_groups
            if group_key in self._device_cache_groups
        }

    def _refresh_weighted_last_n_selection_locked(self) -> None:
        scores: dict[_GlmDeviceCacheGroupKey, float] = {}
        for turn_age, turn_counts in enumerate(
            reversed(self._previous_turn_demand_counts)
        ):
            turn_weight = self._WEIGHTED_LAST_N_DECAY**turn_age
            for group_key, demand_count in turn_counts.items():
                scores[group_key] = (
                    scores.get(group_key, 0.0) + turn_weight * demand_count
                )

        selected_groups = self._select_quoted_groups_locked(
            set(scores),
            lambda group_key: (-scores.get(group_key, 0.0), group_key),
        )
        self._stable_pinned_groups.clear()
        self._causal_pinned_groups = selected_groups
        self._protected_groups = {
            group_key
            for group_key in selected_groups
            if group_key in self._device_cache_groups
        }

    def _reset_turn_causal_history_locked(self) -> None:
        self._current_turn_groups.clear()
        self._current_turn_demand_counts.clear()
        self._previous_turn_groups.clear()
        self._previous_turn_demand_counts.clear()
        self._stable_pinned_groups.clear()
        self._causal_pinned_groups.clear()
        self._protected_groups.clear()
        self._sequence_turn_index = 0

    def _device_cache_group_size_locked(
        self,
        group_key: _GlmDeviceCacheGroupKey,
    ) -> int:
        cached_size = self._device_cache_group_sizes.get(group_key)
        if cached_size is not None:
            return cached_size

        layer_id, expert_id = group_key
        group_bytes = 0
        for unit_id in self._adapter.expert_unit_ids(
            expert_id,
            layer_id=layer_id,
        ):
            metadata = self._adapter.native_metadata(unit_id)
            unit_bytes = metadata.get("length_bytes")
            if type(unit_bytes) is not int or unit_bytes < 0:
                raise ValueError(
                    f"invalid native byte count for Tiered GLM unit {unit_id!r}"
                )
            group_bytes += unit_bytes

        self._device_cache_group_sizes[group_key] = group_bytes
        return group_bytes

    def _group_has_active_lease_locked(
        self,
        group_key: _GlmDeviceCacheGroupKey,
    ) -> bool:
        return any(
            self._device_cache[cache_key].active_leases != 0
            for cache_key in self._device_cache_groups[group_key]
        )

    def _select_group_victim_locked(
        self,
        candidates: Iterable[_GlmDeviceCacheGroupKey],
    ) -> _GlmDeviceCacheGroupKey | None:
        group_candidates = [
            group_key
            for group_key in candidates
            if not self._group_has_active_lease_locked(group_key)
        ]
        if not group_candidates:
            return None

        nonpinned_candidates = [
            group_key
            for group_key in group_candidates
            if group_key not in self._causal_pinned_groups
        ]
        probation_candidates = [
            group_key
            for group_key in nonpinned_candidates
            if group_key not in self._protected_groups
        ]
        if (
            self._device_cache_policy == "causal-weighted-last-n"
            and not probation_candidates
            and not nonpinned_candidates
        ):
            return None
        return (probation_candidates or nonpinned_candidates or group_candidates)[0]

    def _prepare_device_cache_group_locked(
        self,
        group_key: _GlmDeviceCacheGroupKey,
    ) -> bool:
        full_group_bytes = self._device_cache_group_size_locked(group_key)
        if full_group_bytes > self._device_cache_budget_bytes:
            return False

        layer_id = group_key[0]
        while True:
            current_group_bytes = self._device_cache_group_bytes.get(group_key, 0)
            projected_bytes = (
                self._device_cache_bytes - current_group_bytes + full_group_bytes
            )
            projected_layer_bytes = (
                self._device_cache_layer_bytes.get(layer_id, 0)
                - current_group_bytes
                + full_group_bytes
            )
            projected_layer_groups = self._device_cache_layer_group_counts.get(
                layer_id,
                0,
            ) + (0 if group_key in self._device_cache_groups else 1)

            exceeds_layer_quota = (
                projected_layer_bytes > self._layer_quota_bytes
                or projected_layer_groups > self._layer_quota_groups
            )
            exceeds_global_budget = projected_bytes > self._device_cache_budget_bytes
            if not exceeds_layer_quota and not exceeds_global_budget:
                return True

            victim_group = None
            if exceeds_layer_quota:
                victim_group = self._select_group_victim_locked(
                    current_group_key
                    for current_group_key in self._device_cache_groups
                    if current_group_key[0] == layer_id
                    and current_group_key != group_key
                )
            if victim_group is None and not exceeds_global_budget:
                if self._device_cache_policy == "causal-weighted-last-n":
                    return False
                return not (self.enable_warm_turn_causal_cache and exceeds_layer_quota)
            if victim_group is None:
                victim_group = self._select_group_victim_locked(
                    current_group_key
                    for current_group_key in self._device_cache_groups
                    if current_group_key != group_key
                )
                if victim_group is None:
                    return False

            self._remove_cached_group_locked(victim_group)

    def _remove_cached_group_locked(
        self,
        group_key: _GlmDeviceCacheGroupKey,
    ) -> None:
        for cache_key in tuple(self._device_cache_groups[group_key]):
            self._remove_cached_projection(cache_key)

    def _acquire_cached_projection(
        self,
        cache_key: _GlmDeviceCacheKey,
    ) -> TieredGlmProjection | None:
        if self._device_cache_budget_bytes is None:
            return None

        with self._device_cache_lock:
            entry = self._device_cache.get(cache_key)
            if entry is None:
                self._device_cache_misses += 1
                return None

            entry.active_leases += 1
            self._device_cache_active_leases += 1
            self._device_cache.move_to_end(cache_key)
            group_key = cache_key[:2]
            self._device_cache_groups.move_to_end(group_key)
            self._device_cache_hits += 1
            if (
                self._device_cache_policy == "causal-turn-frequency-slru"
                and not self.enable_warm_turn_causal_cache
                and group_key not in self._causal_pinned_groups
                and group_key not in self._protected_groups
                and len(self._protected_groups) < self._protected_group_target
            ):
                self._protected_groups.add(group_key)
            return entry.projection

    def _cache_projection(
        self,
        cache_key: _GlmDeviceCacheKey,
        projection: TieredGlmProjection,
    ) -> TieredGlmProjection | None:
        if self._device_cache_budget_bytes is None:
            return None
        with self._device_cache_lock:
            if cache_key in self._device_cache:
                return None

        byte_count = self._projection_byte_size(projection)
        if byte_count > self._device_cache_budget_bytes:
            return None

        scratch_reservation = ExitStack()
        try:
            scratch_reservation.enter_context(self._runtime.reserve_scratch(byte_count))
            cached_projection = TieredGlmProjection(
                expert_id=projection.expert_id,
                projection=projection.projection,
                weight=projection.weight.clone(),
                scale=projection.scale.clone(),
                weight_unit_id=projection.weight_unit_id,
                scale_unit_id=projection.scale_unit_id,
            )
        except Exception:
            scratch_reservation.close()
            return None
        except BaseException:
            scratch_reservation.close()
            raise

        with self._device_cache_lock:
            if cache_key in self._device_cache:
                del cached_projection
                scratch_reservation.close()
                return None

            group_key = cache_key[:2]
            if self._device_cache_policy in self._CAUSAL_DEVICE_CACHE_POLICIES:
                if not self._prepare_device_cache_group_locked(group_key):
                    del cached_projection
                    scratch_reservation.close()
                    return None
            else:
                budget_bytes = self._device_cache_budget_bytes
                while self._device_cache_bytes + byte_count > budget_bytes:
                    evictable_key = next(
                        (
                            key
                            for key, entry in self._device_cache.items()
                            if entry.active_leases == 0
                        ),
                        None,
                    )
                    if evictable_key is None:
                        del cached_projection
                        scratch_reservation.close()
                        return None
                    self._remove_cached_projection(evictable_key)

            self._device_cache[cache_key] = _GlmDeviceCacheEntry(
                projection=cached_projection,
                byte_count=byte_count,
                active_leases=1,
                scratch_reservation=scratch_reservation,
            )
            group_keys = self._device_cache_groups.get(group_key)
            if group_keys is None:
                group_keys = set()
                self._device_cache_groups[group_key] = group_keys
                self._device_cache_layer_group_counts[group_key[0]] = (
                    self._device_cache_layer_group_counts.get(group_key[0], 0) + 1
                )
                self._device_cache_group_bytes[group_key] = 0
            group_keys.add(cache_key)
            self._device_cache_groups.move_to_end(group_key)
            self._device_cache_group_bytes[group_key] += byte_count
            self._device_cache_layer_bytes[group_key[0]] = (
                self._device_cache_layer_bytes.get(group_key[0], 0) + byte_count
            )
            self._device_cache_bytes += byte_count
            self._device_cache_active_leases += 1
            self._device_cache_peak_bytes = max(
                self._device_cache_peak_bytes,
                self._device_cache_bytes,
            )
            if (
                self._device_cache_policy == "causal-turn-frequency-slru"
                and group_key in self._causal_pinned_groups
            ):
                self._protected_groups.add(group_key)
        return cached_projection

    @staticmethod
    def _projection_byte_size(projection: TieredGlmProjection) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (projection.weight, projection.scale)
        )

    def _remove_cached_projection(self, cache_key: _GlmDeviceCacheKey) -> None:
        entry = self._device_cache.pop(cache_key)
        if entry.active_leases != 0:
            raise RuntimeError("attempted to evict an active Tiered GLM cache lease")

        self._device_cache_bytes -= entry.byte_count
        group_key = cache_key[:2]
        self._device_cache_group_bytes[group_key] -= entry.byte_count
        self._device_cache_layer_bytes[group_key[0]] -= entry.byte_count
        group_keys = self._device_cache_groups[group_key]
        group_keys.remove(cache_key)
        if not group_keys:
            del self._device_cache_groups[group_key]
            del self._device_cache_group_bytes[group_key]
            self._device_cache_layer_group_counts[group_key[0]] -= 1
            self._protected_groups.discard(group_key)
        self._device_cache_evictions += 1
        scratch_reservation = entry.scratch_reservation
        del entry
        scratch_reservation.close()

    def _release_cached_projection(self, cache_key: _GlmDeviceCacheKey) -> None:
        with self._device_cache_lock:
            entry = self._device_cache.get(cache_key)
            if entry is None:
                return
            if entry.active_leases == 0:
                raise RuntimeError("released an unleased Tiered GLM cache tensor")

            entry.active_leases -= 1
            self._device_cache_active_leases -= 1

    @staticmethod
    def _without_dequantized_weight(
        projection: TieredGlmProjection,
    ) -> TieredGlmProjection:
        if projection.dequantized_weight is None:
            return projection
        return replace(projection, dequantized_weight=None)

    def _acquire_dequantized_projection(
        self,
        cache_key: _GlmDequantCacheKey,
        projection: TieredGlmProjection,
    ) -> TieredGlmProjection:
        if self._dequant_cache_budget_bytes is None or self._dequant_cache_closed:
            return self._without_dequantized_weight(projection)

        byte_count = projection.weight.numel() * 4
        try:
            result = self._dequant_cache.acquire(
                cache_key,
                lambda: _dequant_glm_53_fp8_block(projection),
                byte_count,
            )
        except Exception:
            return self._without_dequantized_weight(projection)
        if not result.cached:
            return self._without_dequantized_weight(projection)
        return replace(projection, dequantized_weight=result.value)

    def _release_dequantized_projection(
        self,
        cache_key: _GlmDequantCacheKey,
    ) -> None:
        self._dequant_cache.release(cache_key)

    def _close_dequant_cache(self) -> None:
        if self._dequant_cache_closed:
            return
        self._dequant_cache_closed = True
        self._dequant_cache.close()

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
        adapter: Any | None = None,
    ) -> None:
        super().__init__(moe)
        self._provider = provider
        self._route_controller = getattr(provider, "route_controller", None)
        self._apply_timings = getattr(provider, "apply_timings", None)
        self._residency_hook = TieredWeightsResidencyHook(adapter=adapter)
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
        apply_timings = self._apply_timings
        with _timed_glm_apply_stage(apply_timings, "apply_total", x.device):
            if topk_ids.ndim != 2:
                raise ValueError("Tiered GLM routing IDs must be rank 2")
            if self._route_controller is not None:
                with _timed_glm_apply_stage(
                    apply_timings, "route_controller", x.device
                ):
                    topk_ids, topk_weights = self._route_controller.apply(
                        _parse_glm_layer_id(layer.layer_name),
                        topk_ids,
                        topk_weights,
                    )
            experts_per_token = topk_ids.shape[1]
            if experts_per_token <= 0:
                raise ValueError("Tiered GLM routing selected no experts")
            max_expert_union = self._provider_max_expert_union(experts_per_token)
            run_current_layer_chunks = getattr(
                self._provider,
                "run_current_layer_expert_chunks",
                None,
            )
            current_layer_chunked_overlap = getattr(
                self._provider,
                "enable_current_layer_chunked_overlap",
                False,
            ) and callable(run_current_layer_chunks)

            next_layer_prefetch_fired = False
            prefetch_experts = getattr(self._provider, "prefetch_experts", None)
            if (
                getattr(self._provider, "enable_prefetch", False)
                and callable(prefetch_experts)
                and not current_layer_chunked_overlap
                and not getattr(
                    self._provider,
                    "uses_derived_cache",
                    False,
                )
            ):
                try:
                    with _timed_glm_apply_stage(
                        apply_timings, "demand_build", x.device
                    ):
                        prefetch_demands = (
                            self._residency_hook.build_demands_from_router(
                                router_topk_ids=topk_ids,
                                layer_prefix=layer.layer_name,
                                target_view_id="gpu-vram",
                            )
                        )
                    if prefetch_demands:
                        with _timed_glm_apply_stage(
                            apply_timings, "current_layer_prefetch", x.device
                        ):
                            prefetch_experts(prefetch_demands)
                except Exception:
                    pass

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
                with _timed_glm_apply_stage(apply_timings, "row_partition", x.device):
                    row_partitions = _partition_rows_by_expert_union(
                        topk_ids,
                        experts_per_token,
                        max_expert_union,
                    )
                if current_layer_chunked_overlap:
                    layer_id = _parse_glm_layer_id(layer.layer_name)
                    execution_chunks = _expert_unions_for_row_partitions(
                        topk_ids,
                        row_partitions,
                    )

                    def execute_chunk(
                        chunk_index: int,
                        experts: TieredGlmExpertProjections,
                        _expert_ids: tuple[int, ...],
                    ) -> None:
                        row_indices = row_partitions[chunk_index]
                        scratch_bytes = _glm_53_execution_scratch_bytes(
                            x[row_indices],
                            experts,
                            include_output=False,
                        )
                        if callable(reserve_execution_scratch):
                            scratch_reservation = reserve_execution_scratch(
                                scratch_bytes
                            )
                        else:
                            scratch_reservation = nullcontext()
                        with scratch_reservation:
                            with _timed_glm_apply_stage(
                                apply_timings, "callback", x.device
                            ):
                                chunk_output = self._execution_callback(
                                    x[row_indices],
                                    topk_weights[row_indices],
                                    topk_ids[row_indices],
                                    experts,
                                )
                            with _timed_glm_apply_stage(
                                apply_timings, "output_copy", x.device
                            ):
                                output.index_copy_(
                                    0,
                                    torch.as_tensor(
                                        row_indices,
                                        device=x.device,
                                        dtype=torch.long,
                                    ),
                                    chunk_output,
                                )

                    run_current_layer_chunks(
                        layer_id,
                        execution_chunks,
                        execute_chunk,
                    )
                    return output

                for row_indices in row_partitions:
                    with _timed_glm_apply_stage(
                        apply_timings, "demand_build", x.device
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
                    with ExitStack() as provider_context:
                        provider_context.enter_context(
                            _timed_glm_apply_stage(
                                apply_timings, "provider_request", x.device
                            )
                        )
                        resident_experts = provider_context.enter_context(
                            self._provider.request_experts(demands)
                        )
                        route_replay_prefetch = getattr(
                            self._provider,
                            "prefetch_route_replay_next_layer_experts",
                            None,
                        )
                        if callable(route_replay_prefetch):
                            route_replay_prefetch(
                                _parse_glm_layer_id(layer.layer_name),
                                topk_ids,
                            )
                        scratch_bytes = _glm_53_execution_scratch_bytes(
                            x[row_indices],
                            resident_experts.experts,
                            include_output=False,
                        )
                        if callable(reserve_execution_scratch):
                            scratch_reservation = reserve_execution_scratch(
                                scratch_bytes
                            )
                        else:
                            scratch_reservation = nullcontext()
                        with scratch_reservation:
                            if not next_layer_prefetch_fired:
                                next_layer_prefetch_fired = True
                                self._prefetch_next_layer_experts(layer, topk_ids)
                            with _timed_glm_apply_stage(
                                apply_timings, "callback", x.device
                            ):
                                chunk_output = self._execution_callback(
                                    x[row_indices],
                                    topk_weights[row_indices],
                                    topk_ids[row_indices],
                                    resident_experts.experts,
                                )
                            with _timed_glm_apply_stage(
                                apply_timings, "output_copy", x.device
                            ):
                                output.index_copy_(
                                    0,
                                    torch.as_tensor(
                                        row_indices,
                                        device=x.device,
                                        dtype=torch.long,
                                    ),
                                    chunk_output,
                                )
            return output

    @property
    def _execution_callback(self) -> TieredGlmExecutionCallback:
        callback = getattr(self._provider, "execution_callback", None)
        return glm_53_tiered_execution_callback if callback is None else callback

    def _prefetch_next_layer_experts(
        self,
        layer: RoutedExperts,
        topk_ids: torch.Tensor,
    ) -> None:
        prefetch_next_layer = getattr(
            self._provider,
            "prefetch_next_layer_experts",
            None,
        )
        if not getattr(
            self._provider, "enable_next_layer_prefetch", False
        ) or not callable(prefetch_next_layer):
            return

        try:
            layer_id = _parse_glm_layer_id(layer.layer_name)
            prefetch_next_layer(layer_id, topk_ids)
        except Exception:
            return

    def _provider_max_expert_union(self, experts_per_token: int) -> int:
        return experts_per_token

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
        adapter: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._tiered_provider = provider
        self._tiered_adapter = adapter
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
            adapter=self._tiered_adapter,
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[str]:
        for _ in weights:
            pass
        return iter(())


_TIERED_GLM_53_CONFIG_LOCK = threading.RLock()
_TIERED_GLM_53_CONFIG: _TieredGlm53Config | None = None


def configure_glm_53_tiered_experts(
    provider: TieredGlmTensorProvider,
    *,
    adapter: Any | None = None,
) -> None:
    global _TIERED_GLM_53_CONFIG
    with _TIERED_GLM_53_CONFIG_LOCK:
        _TIERED_GLM_53_CONFIG = _TieredGlm53Config(
            provider=provider,
            adapter=adapter,
        )
    configure_tiered_routed_experts(
        lambda config, prefix: glm_53_tiered_experts(config, prefix)
    )


def clear_glm_53_tiered_experts() -> None:
    global _TIERED_GLM_53_CONFIG
    with _TIERED_GLM_53_CONFIG_LOCK:
        tiered_config = _TIERED_GLM_53_CONFIG
        _TIERED_GLM_53_CONFIG = None
    if tiered_config is not None:
        close_prefetches = getattr(tiered_config.provider, "close_prefetches", None)
        if callable(close_prefetches):
            close_prefetches(wait=True)
    clear_tiered_routed_experts()


def glm_53_tiered_experts(
    config: Any,
    prefix: str,
) -> tuple[type[RoutedExperts], dict[str, Any]] | None:
    with _TIERED_GLM_53_CONFIG_LOCK:
        tiered_config = _TIERED_GLM_53_CONFIG
    if tiered_config is None or not _is_glm_53(config, prefix):
        return None
    factory_kwargs: dict[str, Any] = {"provider": tiered_config.provider}
    if tiered_config.adapter is not None:
        factory_kwargs["adapter"] = tiered_config.adapter
    return TieredGlm53RoutedExperts, factory_kwargs


def glm_53_tiered_execution_callback(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts: dict[int, dict[str, TieredGlmProjection]],
    timings: TieredGlmExecutionTimings | None = None,
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

    with _timed_glm_execution_stage(
        timings,
        "routing_indexing",
        hidden_states.device,
    ):
        selected_expert_tensor = torch.unique(topk_ids, sorted=True)
        if bool((selected_expert_tensor < 0).any()):
            raise ValueError("Tiered GLM routing produced an invalid expert id")
        resident_expert_tensor = torch.as_tensor(
            sorted(experts),
            device=topk_ids.device,
            dtype=topk_ids.dtype,
        )
        if not torch.equal(selected_expert_tensor, resident_expert_tensor):
            raise RuntimeError(
                "Tiered GLM resident experts do not match router selection"
            )

    output = torch.zeros(
        (num_tokens, hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    for expert_id in sorted(experts):
        with _timed_glm_execution_stage(
            timings,
            "routing_indexing",
            hidden_states.device,
        ):
            expert_rows, routing_slots = torch.nonzero(
                topk_ids == expert_id, as_tuple=True
            )
            projections = experts[expert_id]
            expert_input = hidden_states[expert_rows].to(torch.float32)

        with _timed_glm_execution_stage(
            timings,
            "fp8_block_dequantization",
            hidden_states.device,
        ):
            gate_weight = _dequant_glm_53_fp8_block(projections["gate_proj"])
            up_weight = _dequant_glm_53_fp8_block(projections["up_proj"])
            down_weight = _dequant_glm_53_fp8_block(projections["down_proj"])

        with _timed_glm_execution_stage(
            timings,
            "matmul_activation_accumulation",
            hidden_states.device,
        ):
            gate_output = expert_input.unsqueeze(1) @ gate_weight.T
            up_output = expert_input.unsqueeze(1) @ up_weight.T
            intermediate = F.silu(gate_output) * up_output
            expert_output = intermediate @ down_weight.T
            routing_weights = topk_weights[expert_rows, routing_slots].to(torch.float32)
            output.index_add_(
                0,
                expert_rows,
                expert_output.squeeze(1) * routing_weights.unsqueeze(1),
            )

    return output.to(hidden_states.dtype)


def _partition_rows_by_expert_union(
    topk_ids: torch.Tensor,
    experts_per_token: int,
    max_expert_union: int | None = None,
) -> list[list[int]]:
    if topk_ids.ndim != 2:
        raise ValueError("Tiered GLM routing IDs must be rank 2")
    if topk_ids.shape[1] != experts_per_token:
        raise ValueError("Tiered GLM routing width does not match the expert budget")
    if max_expert_union is None:
        max_expert_union = experts_per_token
    if type(max_expert_union) is not int or max_expert_union <= 0:
        raise ValueError("Tiered GLM expert union budget must be a positive integer")

    row_groups: list[list[int]] = []
    current_rows: list[int] = []
    current_expert_ids: set[int] = set()
    for row_index, routing_row in enumerate(topk_ids.tolist()):
        row_expert_ids = {int(expert_id) for expert_id in routing_row}
        if len(row_expert_ids) > experts_per_token:
            raise ValueError("Tiered GLM routing row contains duplicate experts")
        if len(row_expert_ids) > max_expert_union:
            raise ValueError("Tiered GLM routing row exceeds the expert union budget")
        combined_expert_ids = current_expert_ids | row_expert_ids
        if current_expert_ids and len(combined_expert_ids) > max_expert_union:
            row_groups.append(current_rows)
            current_rows = []
            current_expert_ids = set()
        current_rows.append(row_index)
        current_expert_ids.update(row_expert_ids)
    if current_rows:
        row_groups.append(current_rows)
    return row_groups


def _expert_unions_for_row_partitions(
    topk_ids: torch.Tensor,
    row_partitions: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if topk_ids.ndim != 2:
        raise ValueError("Tiered GLM routing IDs must be rank 2")
    execution_chunks: list[tuple[int, ...]] = []
    for row_indices in row_partitions:
        expert_ids = {
            int(expert_id)
            for row in topk_ids[row_indices].tolist()
            for expert_id in row
        }
        if not expert_ids:
            raise ValueError("Tiered GLM row partition selected no experts")
        execution_chunks.append(tuple(sorted(expert_ids)))
    return tuple(execution_chunks)


def _glm_53_execution_scratch_bytes(
    hidden_states: torch.Tensor,
    experts: dict[int, dict[str, TieredGlmProjection]],
    *,
    include_output: bool = True,
) -> int:
    if hidden_states.ndim != 2 or not experts:
        raise ValueError(
            "Tiered GLM execution scratch requires rank-2 input and experts"
        )
    num_tokens, hidden_size = hidden_states.shape
    gate_weight = next(iter(experts.values()))["gate_proj"].weight
    intermediate_size = gate_weight.shape[0]
    weight_elements = hidden_size * intermediate_size
    gemm_rows = num_tokens
    padded_activation_bytes = 4 * (
        (2 * num_tokens + 2 * gemm_rows) * hidden_size
        + 3 * gemm_rows * intermediate_size
    )
    callback_activation_bytes = max(
        4 * num_tokens * (5 * hidden_size + 3 * intermediate_size),
        padded_activation_bytes,
    )
    callback_reservation_bytes = callback_activation_bytes + 16 * weight_elements
    matmul_peak_bytes = (
        20 * num_tokens * hidden_size
        + 16 * num_tokens * intermediate_size
        + 12 * weight_elements
    )
    dequant_peak_bytes = 21 * weight_elements
    scratch_bytes = max(
        callback_reservation_bytes,
        matmul_peak_bytes,
        dequant_peak_bytes,
    )
    if include_output:
        scratch_bytes += 2 * num_tokens * hidden_size
    return scratch_bytes


def _dequant_glm_53_fp8_block(projection: TieredGlmProjection) -> torch.Tensor:
    if projection.dequantized_weight is not None:
        return projection.dequantized_weight

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

    scale_rows, scale_columns = scale.shape
    if (
        weight.shape[0] == scale_rows * block_rows
        and weight.shape[1] == scale_columns * block_columns
    ):
        blocked_weight = (
            weight.view(
                scale_rows,
                block_rows,
                scale_columns,
                block_columns,
            )
            if weight.is_contiguous()
            else weight.reshape(
                scale_rows,
                block_rows,
                scale_columns,
                block_columns,
            )
        )
        return (blocked_weight.to(torch.float32) * scale[:, None, :, None]).view(
            weight.shape
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
    if not layer_index.isdigit():
        return False
    layer_id = int(layer_index)
    model_type = getattr(config, "model_type", None)
    if model_type == "glm_moe_dsa":
        valid_layer = (
            _GLM_53_FIRST_SPARSE_LAYER <= layer_id <= _GLM_53_LAST_SPARSE_LAYER
        )
    elif model_type == "deepseek_mtp":
        valid_layer = layer_id == _GLM_53_MTP_LAYER
    else:
        valid_layer = False
    return (
        valid_layer
        and getattr(config, "n_routed_experts", None) == 256
        and getattr(config, "num_experts_per_tok", None) == 8
        and getattr(config, "hidden_size", None) == 6144
        and getattr(config, "moe_intermediate_size", None) == 2048
    )
