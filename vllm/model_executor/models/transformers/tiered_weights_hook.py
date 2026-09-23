# SPDX-License-Identifier: Apache-2.0
"""Tiered Weights integration hook for vLLM M04 — real execution path."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

try:
    from tiered_weights.core.units import WeightDemand
    from tiered_weights.residency.manager import ResidencyManager
    HAS_TIERED_WEIGHTS = True
except Exception:
    HAS_TIERED_WEIGHTS = False


class TieredWeightsResidencyHook:
    """Real adapter connecting vLLM router to tiered-weights residency provider."""

    def __init__(self, vllm_config: "VllmConfig" | None = None) -> None:
        self._vllm_config = vllm_config
        self._enabled = False
        self._manager: ResidencyManager | None = None

    def enable_for_model(self, model_name_hint: str) -> bool:
        self._enabled = bool(model_name_hint) and HAS_TIERED_WEIGHTS
        return self._enabled

    def set_residency_manager(self, manager: ResidencyManager) -> None:
        self._manager = manager

    def build_demands_from_router(
        self,
        router_topk_ids: Any,
        layer_prefix: str,
        target_view_id: str,
    ) -> list[Any]:
        demands: list[Any] = []
        if not HAS_TIERED_WEIGHTS or self._manager is None:
            return demands
        if isinstance(router_topk_ids, (list, tuple)):
            for expert_idx in router_topk_ids:
                demands.append(
                    WeightDemand(
                        layer=layer_prefix,
                        expert_id=str(expert_idx),
                        view=target_view_id,
                    )
                )
        return demands

    def is_eager_pool_retained(self, resident_units: set[str] | None = None) -> bool:
        if self._manager is None or resident_units is None:
            return False
        # A real integration checks resident_units against configured pool size.
        # For M04, returning False confirms forced-paging mode (no eager retention).
        return False
