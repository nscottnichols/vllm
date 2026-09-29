# SPDX-License-Identifier: Apache-2.0
"""Prototype vLLM hook for building Tiered Weights routing demands."""
from __future__ import annotations

import inspect
import re
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig


_LAYER_PREFIX_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer_id>\d+)(?:\.mlp|\.block_sparse_moe)(?:\.experts)?$"
)
_TieredRoutedExpertsSelector = Callable[
    [Any, str],
    tuple[type[Any], dict[str, Any]] | None,
]
_TIERED_ROUTED_EXPERTS_LOCK = threading.RLock()
_TIERED_ROUTED_EXPERTS_SELECTOR: _TieredRoutedExpertsSelector | None = None

try:
    from tiered_weights.adapters import get_adapter
    from tiered_weights.adapters.base import ModelAdapter
    from tiered_weights.core.units import WeightDemand
    from tiered_weights.residency.manager import ResidencyManager
    HAS_TIERED_WEIGHTS = True
except Exception:
    HAS_TIERED_WEIGHTS = False


class TieredWeightsResidencyHook:
    """Build residency demands from vLLM router selections without loading weights."""

    def __init__(
        self,
        vllm_config: VllmConfig | None = None,
        adapter: ModelAdapter | None = None,
    ) -> None:
        self._vllm_config = vllm_config
        self._enabled = False
        self._manager: ResidencyManager | None = None
        self._adapter = adapter

    def enable_for_model(self, model_name_hint: str) -> bool:
        self._enabled = False
        if not HAS_TIERED_WEIGHTS:
            return False
        try:
            if self._adapter is None:
                self._adapter = get_adapter(model_name_hint)
            elif self._adapter.get_model_name() != model_name_hint:
                raise ValueError(
                    f"model hint {model_name_hint!r} does not match adapter "
                    f"{self._adapter.get_model_name()!r}"
                )
        except KeyError as error:
            raise ValueError(
                f"no adapter registered for model {model_name_hint!r}"
            ) from error
        self._enabled = True
        return True

    def set_residency_manager(self, manager: ResidencyManager) -> None:
        self._manager = manager

    def set_adapter(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter

    def build_demands_from_router(
        self,
        router_topk_ids: Any,
        layer_prefix: str,
        target_view_id: str,
    ) -> list[WeightDemand]:
        if not self._enabled or not HAS_TIERED_WEIGHTS:
            return []

        if not layer_prefix:
            raise ValueError("layer_prefix must not be empty")

        layer_id = self._layer_id_for_prefix(layer_prefix)
        routing_rows = self._router_routing_rows(router_topk_ids)

        if self._adapter is None:
            raise ValueError("no model adapter is configured")
        demands_by_unit_id: dict[str, WeightDemand] = {}
        for routing_row in routing_rows:
            selected_expert_ids = self._unique_expert_ids(routing_row)
            adapter_demands = self._build_adapter_demands(
                selected_expert_ids,
                layer_id,
            )
            for demand in adapter_demands:
                demands_by_unit_id[demand.unit_id] = demand
        return list(demands_by_unit_id.values())

    def _build_adapter_demands(
        self,
        selected_expert_ids: list[int],
        layer_id: int | None,
    ) -> list[WeightDemand]:
        if layer_id is not None and self._adapter_accepts_layer_id():
            return self._adapter.build_demands_for_top_k_routing(
                selected_expert_ids=selected_expert_ids,
                layer_id=layer_id,
            )
        return self._adapter.build_demands_for_top_k_routing(
            selected_expert_ids=selected_expert_ids
        )

    def _adapter_accepts_layer_id(self) -> bool:
        signature = inspect.signature(self._adapter.build_demands_for_top_k_routing)
        return "layer_id" in signature.parameters

    @staticmethod
    def _layer_id_for_prefix(layer_prefix: str) -> int | None:
        match = _LAYER_PREFIX_PATTERN.fullmatch(layer_prefix)
        if match is None:
            return None
        return int(match.group("layer_id"))

    @staticmethod
    def _router_routing_rows(router_topk_ids: Any) -> list[list[int]]:
        if hasattr(router_topk_ids, "detach"):
            flat_expert_ids = router_topk_ids.detach().reshape(-1).tolist()
            top_k = router_topk_ids.shape[-1] if router_topk_ids.ndim else 1
            if top_k <= 0:
                raise ValueError("router expert IDs must contain at least one expert")
            return [
                flat_expert_ids[start : start + top_k]
                for start in range(0, len(flat_expert_ids), top_k)
            ]
        if (
            isinstance(router_topk_ids, (list, tuple))
            and router_topk_ids
            and isinstance(router_topk_ids[0], (list, tuple))
        ):
            return [list(routing_row) for routing_row in router_topk_ids]
        return [list(router_topk_ids)]

    @staticmethod
    def _unique_expert_ids(raw_expert_ids: list[Any]) -> list[int]:
        selected_expert_ids: list[int] = []
        seen_expert_ids: set[int] = set()
        for raw_expert_id in raw_expert_ids:
            expert_id = int(raw_expert_id)
            if expert_id < 0:
                continue
            if expert_id not in seen_expert_ids:
                seen_expert_ids.add(expert_id)
                selected_expert_ids.append(expert_id)
        return selected_expert_ids

    def is_eager_pool_retained(self, resident_units: set[str] | None = None) -> bool:
        """Report eager retention; this prototype always disables it."""
        if self._manager is None or resident_units is None:
            return False
        return False


def configure_tiered_routed_experts(
    selector: _TieredRoutedExpertsSelector,
) -> None:
    global _TIERED_ROUTED_EXPERTS_SELECTOR
    with _TIERED_ROUTED_EXPERTS_LOCK:
        _TIERED_ROUTED_EXPERTS_SELECTOR = selector


def clear_tiered_routed_experts() -> None:
    global _TIERED_ROUTED_EXPERTS_SELECTOR
    with _TIERED_ROUTED_EXPERTS_LOCK:
        _TIERED_ROUTED_EXPERTS_SELECTOR = None


def tiered_routed_experts(
    config: Any,
    prefix: str,
) -> tuple[type[Any], dict[str, Any]] | None:
    with _TIERED_ROUTED_EXPERTS_LOCK:
        selector = _TIERED_ROUTED_EXPERTS_SELECTOR
    if selector is None:
        return None
    return selector(config, prefix)
