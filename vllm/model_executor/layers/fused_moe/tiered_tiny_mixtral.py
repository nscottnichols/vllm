"""Tiered Weights bridge for tiny-Mixtral routed experts."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from tiered_weights.core.units import WeightDemand

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
class TinyMixtralProjection:
    expert_id: int
    projection: str
    weight: torch.Tensor
    weight_unit_id: str


TinyMixtralExpertProjections = dict[int, dict[str, TinyMixtralProjection]]
TinyMixtralExecutionCallback = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, TinyMixtralExpertProjections],
    torch.Tensor,
]


class TinyMixtralResidentExperts:
    def __init__(
        self,
        experts: TinyMixtralExpertProjections,
        release: Callable[[], None],
    ) -> None:
        self.experts = experts
        self._release = release
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._release()
            self._released = True

    def __enter__(self) -> TinyMixtralResidentExperts:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


class TinyMixtralTensorProvider(Protocol):
    def request_experts(
        self, demands: Sequence[Any]
    ) -> AbstractContextManager[TinyMixtralResidentExperts]:
        """Make only the requested tiny-Mixtral expert units resident."""
        ...


class _TinyMixtralDemandAdapter:
    _MODEL_NAME = "tiny-mixtral"
    _EXPERT_COUNT = 8

    def get_model_name(self) -> str:
        return self._MODEL_NAME

    def build_demands_for_top_k_routing(
        self,
        selected_expert_ids: list[int] | None = None,
        *,
        layer_id: int | None = None,
    ) -> list[WeightDemand]:
        resolved_layer_id = 0 if layer_id is None else layer_id
        expert_ids = (
            range(self._EXPERT_COUNT)
            if selected_expert_ids is None
            else selected_expert_ids
        )
        demands: list[WeightDemand] = []
        for expert_id in expert_ids:
            if not 0 <= expert_id < self._EXPERT_COUNT:
                raise ValueError(f"tiny-Mixtral expert id {expert_id} is out of range")
            demands.append(
                WeightDemand(
                    unit_id=f"tiny-mixtral-layer-{resolved_layer_id}-expert-{expert_id}",
                    target_view_id="gpu-vram",
                    earliest_use_step=0,
                    deadline_step=1,
                )
            )
        return demands


class TieredTinyMixtralMoEMethod(FusedMoEMethodBase):
    """MoE method that requests only router-selected tiny-Mixtral experts."""

    def __init__(
        self,
        moe: FusedMoEConfig,
        *,
        provider: TinyMixtralTensorProvider,
    ) -> None:
        super().__init__(moe)
        self._provider = provider
        self._residency_hook = TieredWeightsResidencyHook(
            adapter=_TinyMixtralDemandAdapter()
        )
        if not self._residency_hook.enable_for_model("tiny-mixtral"):
            raise RuntimeError("the tiny-Mixtral Tiered Weights adapter is unavailable")

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
        if x.ndim != 2:
            raise ValueError("tiny-Mixtral hidden states must be rank 2")
        if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
            raise ValueError(
                "tiny-Mixtral routing tensors must have matching rank-2 shapes"
            )
        if topk_weights.shape[0] != x.shape[0] or topk_ids.numel() == 0:
            raise ValueError("tiny-Mixtral routing does not match the token count")

        selected_expert_ids = {
            int(expert_id)
            for routing_row in topk_ids.tolist()
            for expert_id in routing_row
        }
        if any(expert_id < 0 for expert_id in selected_expert_ids):
            raise ValueError("tiny-Mixtral routing produced an invalid expert id")

        demands = self._residency_hook.build_demands_from_router(
            router_topk_ids=topk_ids,
            layer_prefix=layer.layer_name,
            target_view_id="gpu-vram",
        )
        if not demands:
            raise RuntimeError("router produced no tiny-Mixtral Tiered Weights demands")
        demand_expert_ids = {
            int(demand.unit_id.rsplit("-", 1)[1]) for demand in demands
        }
        if demand_expert_ids != selected_expert_ids:
            raise RuntimeError(
                "tiny-Mixtral demand set is not an exact router selection"
            )

        with self._provider.request_experts(demands) as resident_experts:
            return self._execution_callback(
                x,
                topk_weights,
                topk_ids,
                resident_experts.experts,
            )

    @property
    def _execution_callback(self) -> TinyMixtralExecutionCallback:
        callback = getattr(self._provider, "execution_callback", None)
        return tiny_mixtral_tiered_execution_callback if callback is None else callback

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("tiny-Mixtral requires modular router execution")


class TieredTinyMixtralRoutedExperts(RoutedExperts):
    """Routed-experts container without eager tiny-Mixtral tensor allocation."""

    def __init__(
        self,
        *args: Any,
        provider: TinyMixtralTensorProvider,
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
        return TieredTinyMixtralMoEMethod(
            moe_config,
            provider=self._tiered_provider,
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[str]:
        for _ in weights:
            pass
        return iter(())


_TIERED_TINY_MIXTRAL_CONFIG_LOCK = threading.RLock()
_TIERED_TINY_MIXTRAL_CONFIG: TinyMixtralTensorProvider | None = None


def configure_tiny_mixtral_tiered_experts(
    provider: TinyMixtralTensorProvider,
) -> None:
    global _TIERED_TINY_MIXTRAL_CONFIG
    with _TIERED_TINY_MIXTRAL_CONFIG_LOCK:
        _TIERED_TINY_MIXTRAL_CONFIG = provider
    configure_tiered_routed_experts(
        lambda config, prefix: tiny_mixtral_tiered_experts(config, prefix)
    )


def clear_tiny_mixtral_tiered_experts() -> None:
    global _TIERED_TINY_MIXTRAL_CONFIG
    with _TIERED_TINY_MIXTRAL_CONFIG_LOCK:
        _TIERED_TINY_MIXTRAL_CONFIG = None
    clear_tiered_routed_experts()


def tiny_mixtral_tiered_experts(
    config: Any,
    prefix: str,
) -> tuple[type[RoutedExperts], dict[str, Any]] | None:
    with _TIERED_TINY_MIXTRAL_CONFIG_LOCK:
        tiered_config = _TIERED_TINY_MIXTRAL_CONFIG
    if tiered_config is None or not _is_tiny_mixtral(config, prefix):
        return None
    return TieredTinyMixtralRoutedExperts, {"provider": tiered_config}


def tiny_mixtral_tiered_execution_callback(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts: TinyMixtralExpertProjections,
) -> torch.Tensor:
    if hidden_states.ndim != 2:
        raise ValueError("tiny-Mixtral hidden states must be rank 2")
    if topk_weights.shape != topk_ids.shape or topk_weights.ndim != 2:
        raise ValueError(
            "tiny-Mixtral routing tensors must have matching rank-2 shapes"
        )
    if topk_weights.shape[0] != hidden_states.shape[0]:
        raise ValueError("tiny-Mixtral routing does not match the token count")

    selected_experts = {
        int(expert_id) for routing_row in topk_ids.tolist() for expert_id in routing_row
    }
    if selected_experts != set(experts):
        raise RuntimeError("resident experts do not match the tiny-Mixtral router")

    num_tokens, hidden_size = hidden_states.shape
    output = torch.zeros(
        (num_tokens, hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    for expert_id in sorted(selected_experts):
        expert_rows, routing_slots = torch.nonzero(
            topk_ids == expert_id,
            as_tuple=True,
        )
        projections = experts[expert_id]
        expert_input = hidden_states[expert_rows].to(torch.float32)
        gate_output = expert_input @ projections["gate_proj"].weight.to(torch.float32).T
        up_output = expert_input @ projections["up_proj"].weight.to(torch.float32).T
        intermediate = F.silu(gate_output) * up_output
        expert_output = (
            intermediate @ projections["down_proj"].weight.to(torch.float32).T
        )
        routing_weights = topk_weights[expert_rows, routing_slots].to(torch.float32)
        output.index_add_(
            0,
            expert_rows,
            expert_output * routing_weights.unsqueeze(1),
        )
    return output.to(hidden_states.dtype)


def _is_tiny_mixtral(config: Any, prefix: str) -> bool:
    prefix_parts = prefix.rstrip(".").split(".")
    try:
        layer_index = prefix_parts[prefix_parts.index("layers") + 1]
    except (ValueError, IndexError):
        return False
    return (
        getattr(config, "model_type", None) == "mixtral"
        and getattr(config, "hidden_size", None) == 1024
        and getattr(config, "intermediate_size", None) == 3584
        and getattr(config, "num_local_experts", None) == 8
        and getattr(config, "num_experts_per_tok", None) == 2
        and layer_index.isdigit()
    )
