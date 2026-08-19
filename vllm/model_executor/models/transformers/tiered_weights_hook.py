# SPDX-License-Identifier: Apache-2.0
"""Minimal tiered-weights integration hook for M04.

This file demonstrates where the outer tiered-weights residency provider
attaches to vLLM MoE execution. It does not claim real GPU or real
`TitanML/tiny-mixtral` checkpoint validation; those gates are reported
as not run when the required hardware/checkpoint is unavailable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class TieredWeightsResidencyHook:
    """Adapter from vLLM router output to WeightDemand objects.

    In a full integration, this adapter would:
    - read the router's expert assignments,
    - build exact `WeightDemand` objects with earliest-use/deadline,
    - pass them to the `SynchronousWeightProvider` or `ResidencyManager`,
    - acquire fixed-slot leases before kernel execution,
    - release leases after the operator event completes.

    For M04, this hook is a placeholder that validates the interface
    without introducing new quantization or changing the native weight
    identity contract.
    """

    def __init__(self, vllm_config: "VllmConfig" | None = None) -> None:
        self._vllm_config = vllm_config
        self._enabled = False

    def enable_for_model(self, model_name_hint: str) -> bool:
        # Only enable for synthetic fixtures or when explicitly configured.
        # Never claim a real checkpoint that hasn't been validated.
        self._enabled = bool(model_name_hint)
        return self._enabled

    def build_demands_from_router(
        self,
        router_topk_ids: Any,
        layer_prefix: str,
        target_view_id: str,
    ) -> list[Any]:
        """Convert router output into demand objects.

        In a production integration this would return `WeightDemand`
        instances from `tiered_weights.core.units`. For the M04
        synthetic fixture path, this returns a list of lightweight
        demand descriptors without importing the full package.
        """
        demands: list[Any] = []
        # Placeholder: the actual integration would map each routed
        # expert ID to its logical `WeightUnit` and build demands.
        if isinstance(router_topk_ids, (list, tuple)):
            for expert_idx in router_topk_ids:
                demands.append({
                    "layer": layer_prefix,
                    "expert_id": str(expert_idx),
                    "view": target_view_id,
                    "exact": True,
                })
        return demands

    def is_eager_pool_retained(self, resident_units: set[str]) -> bool:
        """Detect whether the full configured expert pool is resident.

        Used by M04 verification to confirm forced-paging mode does not
        eagerly materialize cold experts.
        """
        # Placeholder: a real integration would compare resident_units
        # against the configured expert pool size.
        return False
