"""Tiered Weights execution path for DeepSeek-V2-Lite routed experts."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from tiered_weights.adapters.deepseek_v2_lite.moe import expert_output

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

_EXPERT_UNIT_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer_id>\d+)\.mlp\.experts\."
    r"(?P<expert_id>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_PROJECTIONS = frozenset(("gate_proj", "up_proj", "down_proj"))
_NATIVE_DTYPE = "BF16"
_NATIVE_LAYOUT = "safetensors:BF16:row-major"
_TENSOR_KIND = "unquantized_weight"


@dataclass(frozen=True, slots=True)
class DeepseekV2LiteProjection:
    expert_id: int
    projection: str
    weight: torch.Tensor
    weight_unit_id: str


DeepseekV2LiteExpertProjections = dict[int, dict[str, DeepseekV2LiteProjection]]
DeepseekV2LiteExecutionCallback = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        dict[str, DeepseekV2LiteProjection],
    ],
    torch.Tensor,
]


class DeepseekV2LiteResidentExperts:
    def __init__(
        self,
        layer_id: int,
        experts: DeepseekV2LiteExpertProjections,
        release: Callable[[], None],
    ) -> None:
        self.layer_id = layer_id
        self.experts = experts
        self._release = release
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._release()
            self._released = True

    def __enter__(self) -> DeepseekV2LiteResidentExperts:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self.release()


class TieredDeepseekV2LiteTensorProvider(Protocol):
    @property
    def adapter(self) -> Any:
        """Return the DeepSeek-V2-Lite runtime adapter."""
        ...

    def request_experts(
        self, demands: Sequence[Any]
    ) -> AbstractContextManager[DeepseekV2LiteResidentExperts]:
        """Make only the requested DeepSeek-V2-Lite experts resident."""
        ...

    def reserve_execution_scratch(self, scratch_bytes: int) -> Any:
        """Reserve execution scratch against the coherent device budget."""
        ...


class DeviceWeightRuntimeDeepseekV2LiteTensorProvider:
    """Adapt the model-agnostic runtime to DeepSeek-V2-Lite experts."""

    def __init__(
        self,
        *,
        runtime: Any,
        adapter: Any,
        execution_callback: DeepseekV2LiteExecutionCallback | None = None,
    ) -> None:
        self._runtime = runtime
        self._adapter = adapter
        self.execution_callback = execution_callback

    @property
    def adapter(self) -> Any:
        return self._adapter

    def request_experts(
        self, demands: Sequence[Any]
    ) -> DeepseekV2LiteResidentExperts:
        if not demands:
            raise ValueError("Tiered DeepSeek-V2-Lite demands must not be empty")

        unit_ids = [demand.unit_id for demand in demands]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("Tiered DeepSeek-V2-Lite demands contain duplicate units")

        requested_units = set(unit_ids)
        selected_experts = self._selected_experts(requested_units)
        layer_id = self._validate_exact_expert_demands(
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

        return DeepseekV2LiteResidentExperts(
            layer_id,
            experts,
            lease_stack.close,
        )

    def reserve_execution_scratch(self, scratch_bytes: int) -> Any:
        return self._runtime.reserve_scratch(scratch_bytes)

    def stats(self) -> dict[str, int | float | str]:
        return self._runtime.stats()

    def _selected_experts(self, unit_ids: set[str]) -> set[tuple[int, int]]:
        selected_experts: set[tuple[int, int]] = set()
        for unit_id in unit_ids:
            expert_location = _expert_location(unit_id)
            if expert_location is None:
                raise ValueError(
                    f"demand is not a DeepSeek-V2-Lite routed-expert unit: {unit_id!r}"
                )
            selected_experts.add(expert_location)
        return selected_experts

    def _validate_exact_expert_demands(
        self,
        selected_experts: set[tuple[int, int]],
        requested_units: set[str],
    ) -> int:
        layer_ids = {layer_id for layer_id, _ in selected_experts}
        if len(layer_ids) != 1:
            raise ValueError(
                "Tiered DeepSeek-V2-Lite demands must target exactly one layer"
            )
        layer_id = next(iter(layer_ids))

        expected_units: set[str] = set()
        for expert_layer_id, expert_id in selected_experts:
            expected_units.update(
                self._adapter.expert_unit_ids(
                    expert_id,
                    layer_id=expert_layer_id,
                )
            )
        if expected_units != requested_units:
            raise ValueError(
                "Tiered DeepSeek-V2-Lite demand set is not an exact expert union"
            )
        return layer_id

    def _build_experts(
        self,
        leases: Sequence[Any],
    ) -> DeepseekV2LiteExpertProjections:
        experts: DeepseekV2LiteExpertProjections = {}
        for lease in leases:
            projections: dict[str, DeepseekV2LiteProjection] = {}
            for unit_id in lease.unit_ids:
                projection_name = _projection_name(unit_id)
                if projection_name is None:
                    raise ValueError(f"lease contains invalid unit {unit_id!r}")
                metadata = self._adapter.native_metadata(unit_id)
                tensor = lease.tensor(unit_id)
                lease_spec = lease.metadata(unit_id)
                self._validate_native_tensor(unit_id, metadata, tensor, lease_spec)
                projections[projection_name] = DeepseekV2LiteProjection(
                    expert_id=lease.expert_id,
                    projection=projection_name,
                    weight=tensor,
                    weight_unit_id=unit_id,
                )
            if set(projections) != _PROJECTIONS:
                raise ValueError(
                    f"Tiered DeepSeek-V2-Lite expert {lease.expert_id} is incomplete"
                )
            experts[lease.expert_id] = projections
        return experts

    @staticmethod
    def _validate_native_tensor(
        unit_id: str,
        metadata: Any,
        tensor: torch.Tensor,
        lease_spec: Any,
    ) -> None:
        expected_shape = tuple(metadata["shape"])
        byte_length = tensor.numel() * tensor.element_size()
        checks = (
            metadata["unit_id"] == unit_id,
            metadata["native_dtype"] == _NATIVE_DTYPE,
            metadata["dtype"] == _NATIVE_DTYPE,
            metadata["target_dtype"] == _NATIVE_DTYPE,
            metadata["tensor_kind"] == _TENSOR_KIND,
            metadata["quantized"] is False,
            metadata["storage_layout"] == _NATIVE_LAYOUT,
            metadata["target_layout"] == _NATIVE_LAYOUT,
            metadata["checksum_sha256"] is not None,
            tuple(tensor.shape) == expected_shape,
            tensor.dtype == torch.bfloat16,
            tensor.is_contiguous(),
            byte_length == metadata["length_bytes"],
            byte_length == lease_spec.byte_length,
        )
        if not all(checks):
            raise ValueError(
                f"Tiered DeepSeek-V2-Lite unit {unit_id!r} changed native identity"
            )


class TieredDeepseekV2LiteMoEMethod(FusedMoEMethodBase):
    """Load and execute only router-selected DeepSeek-V2-Lite experts."""

    def __init__(
        self,
        moe: FusedMoEConfig,
        *,
        provider: TieredDeepseekV2LiteTensorProvider,
    ) -> None:
        super().__init__(moe)
        self._provider = provider
        self._adapter = provider.adapter
        self._residency_hook = TieredWeightsResidencyHook(
            adapter=self._adapter,
        )
        if not self._residency_hook.enable_for_model("deepseek-v2-lite"):
            raise RuntimeError(
                "the DeepSeek-V2-Lite Tiered Weights adapter is unavailable"
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
        if params_dtype != torch.bfloat16:
            raise ValueError("DeepSeek-V2-Lite Tiered Weights require native BF16")
        if (
            num_experts != 64
            or hidden_size != 2048
            or intermediate_size_per_partition != 1408
        ):
            raise ValueError(
                "DeepSeek-V2-Lite Tiered Weights require the unsharded expert layout"
            )
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
        if x.ndim != 2 or x.dtype != torch.bfloat16:
            raise ValueError(
                "Tiered DeepSeek-V2-Lite hidden states must be rank-2 BF16"
            )
        if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
            raise ValueError(
                "Tiered DeepSeek-V2-Lite routing tensors must have matching "
                "rank-2 shapes"
            )
        if topk_weights.shape[0] != x.shape[0] or topk_ids.shape[1] != 6:
            raise ValueError(
                "Tiered DeepSeek-V2-Lite routing does not match top-6 BF16 input"
            )
        if shared_experts is None or shared_experts_input is None:
            raise ValueError(
                "DeepSeek-V2-Lite requires vLLM's native shared-expert execution"
            )
        if shared_experts_input.shape != x.shape:
            raise ValueError(
                "Tiered DeepSeek-V2-Lite shared-expert input does not match"
            )
        if layer.global_num_experts != layer.local_num_experts:
            raise NotImplementedError(
                "DeepSeek-V2-Lite Tiered Weights require unsharded expert IDs"
            )

        selected_expert_ids = {
            int(expert_id)
            for routing_row in topk_ids.tolist()
            for expert_id in routing_row
            if int(expert_id) >= 0
        }
        if not selected_expert_ids:
            return torch.zeros_like(x)
        for routing_row in topk_ids.tolist():
            valid_expert_ids = [
                int(expert_id) for expert_id in routing_row if int(expert_id) >= 0
            ]
            if len(set(valid_expert_ids)) != len(valid_expert_ids):
                raise ValueError(
                    "Tiered DeepSeek-V2-Lite routing row repeats an expert"
                )

        demands = self._residency_hook.build_demands_from_router(
            router_topk_ids=topk_ids,
            layer_prefix=layer.layer_name,
            target_view_id="gpu-vram",
        )
        if not demands:
            raise RuntimeError(
                "router produced no DeepSeek-V2-Lite Tiered Weights demands"
            )
        demand_expert_ids = {
            _expert_location(demand.unit_id)[1] for demand in demands
        }
        if demand_expert_ids != selected_expert_ids:
            raise RuntimeError(
                "Tiered DeepSeek-V2-Lite demand set is not the exact router selection"
            )

        reserve_execution_scratch = self._provider.reserve_execution_scratch
        output = torch.zeros_like(x)
        with reserve_execution_scratch(x.numel() * x.element_size()):
            for expert_id in sorted(selected_expert_ids):
                expert_demands = [
                    demand
                    for demand in demands
                    if _expert_location(demand.unit_id)[1] == expert_id
                ]
                with self._provider.request_experts(
                    expert_demands
                ) as resident_experts:
                    projections = resident_experts.experts[expert_id]
                    expert_rows, routing_slots = torch.nonzero(
                        topk_ids == expert_id,
                        as_tuple=True,
                    )
                    scratch_bytes = _execution_scratch_bytes(
                        x[expert_rows],
                        projections,
                    )
                    with reserve_execution_scratch(scratch_bytes):
                        contribution = self._execution_callback(
                            x,
                            topk_weights,
                            topk_ids,
                            expert_id,
                            projections,
                        )
                    output.index_add_(
                        0,
                        expert_rows,
                        contribution,
                    )
        return output

    @property
    def _execution_callback(self) -> DeepseekV2LiteExecutionCallback:
        callback = getattr(self._provider, "execution_callback", None)
        return (
            deepseek_v2_lite_tiered_execution_callback
            if callback is None
            else callback
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "DeepSeek-V2-Lite Tiered Weights require modular router execution"
        )


class TieredDeepseekV2LiteRoutedExperts(RoutedExperts):
    """Routed-experts container without eager DeepSeek-V2-Lite allocation."""

    def __init__(
        self,
        *args: Any,
        provider: TieredDeepseekV2LiteTensorProvider,
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
        return TieredDeepseekV2LiteMoEMethod(
            moe_config,
            provider=self._tiered_provider,
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[str]:
        for _ in weights:
            pass
        return iter(())


_TIERED_DEEPSEEK_V2_LITE_LOCK = threading.RLock()
_TIERED_DEEPSEEK_V2_LITE_PROVIDER: (
    TieredDeepseekV2LiteTensorProvider | None
) = None


def configure_deepseek_v2_lite_tiered_experts(
    provider: TieredDeepseekV2LiteTensorProvider,
) -> None:
    global _TIERED_DEEPSEEK_V2_LITE_PROVIDER
    with _TIERED_DEEPSEEK_V2_LITE_LOCK:
        _TIERED_DEEPSEEK_V2_LITE_PROVIDER = provider
    configure_tiered_routed_experts(
        lambda config, prefix: deepseek_v2_lite_tiered_experts(
            config,
            prefix,
        )
    )


def clear_deepseek_v2_lite_tiered_experts() -> None:
    global _TIERED_DEEPSEEK_V2_LITE_PROVIDER
    with _TIERED_DEEPSEEK_V2_LITE_LOCK:
        _TIERED_DEEPSEEK_V2_LITE_PROVIDER = None
    clear_tiered_routed_experts()


def deepseek_v2_lite_tiered_experts(
    config: Any,
    prefix: str,
) -> tuple[type[RoutedExperts], dict[str, Any]] | None:
    with _TIERED_DEEPSEEK_V2_LITE_LOCK:
        provider = _TIERED_DEEPSEEK_V2_LITE_PROVIDER
    if provider is None or not _is_deepseek_v2_lite(config, prefix):
        return None
    return TieredDeepseekV2LiteRoutedExperts, {"provider": provider}


def deepseek_v2_lite_tiered_execution_callback(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_id: int,
    projections: dict[str, DeepseekV2LiteProjection],
) -> torch.Tensor:
    if hidden_states.ndim != 2:
        raise ValueError("Tiered DeepSeek-V2-Lite hidden states must be rank 2")
    if topk_weights.shape != topk_ids.shape or topk_ids.ndim != 2:
        raise ValueError(
            "Tiered DeepSeek-V2-Lite routing tensors must have matching rank-2 shapes"
        )
    if topk_weights.shape[0] != hidden_states.shape[0]:
        raise ValueError("Tiered DeepSeek-V2-Lite routing does not match tokens")

    expert_rows, routing_slots = torch.nonzero(
        topk_ids == expert_id,
        as_tuple=True,
    )
    expert_input = hidden_states[expert_rows]
    expert_result = expert_output(
        expert_input,
        projections["gate_proj"].weight,
        projections["up_proj"].weight,
        projections["down_proj"].weight,
    )
    routing_weights = topk_weights[expert_rows, routing_slots].to(
        hidden_states.dtype
    )
    return expert_result * routing_weights.unsqueeze(1)


def _expert_location(unit_id: str) -> tuple[int, int] | None:
    match = _EXPERT_UNIT_PATTERN.fullmatch(unit_id)
    if match is None:
        return None
    return int(match.group("layer_id")), int(match.group("expert_id"))


def _projection_name(unit_id: str) -> str | None:
    match = _EXPERT_UNIT_PATTERN.fullmatch(unit_id)
    return None if match is None else match.group("projection")


def _execution_scratch_bytes(
    hidden_states: torch.Tensor,
    projections: dict[str, DeepseekV2LiteProjection],
) -> int:
    if hidden_states.ndim != 2 or not projections:
        raise ValueError(
            "Tiered DeepSeek-V2-Lite scratch requires rank-2 input and experts"
        )
    num_tokens, hidden_size = hidden_states.shape
    intermediate_size = projections["gate_proj"].weight.shape[0]
    return num_tokens * (32 * intermediate_size + 16 * hidden_size)


def _is_deepseek_v2_lite(config: Any, prefix: str) -> bool:
    prefix_parts = prefix.rstrip(".").split(".")
    try:
        layer_index = prefix_parts[prefix_parts.index("layers") + 1]
    except (ValueError, IndexError):
        return False
    if not layer_index.isdigit():
        return False
    layer_id = int(layer_index)
    native_dtype = getattr(config, "torch_dtype", None)
    return (
        getattr(config, "model_type", None) == "deepseek_v2"
        and getattr(config, "hidden_size", None) == 2048
        and getattr(config, "intermediate_size", None) == 10944
        and getattr(config, "moe_intermediate_size", None) == 1408
        and getattr(config, "n_routed_experts", None) == 64
        and getattr(config, "n_shared_experts", None) == 2
        and getattr(config, "num_experts_per_tok", None) == 6
        and getattr(config, "num_hidden_layers", None) == 27
        and getattr(config, "first_k_dense_replace", None) == 1
        and getattr(config, "moe_layer_freq", None) == 1
        and native_dtype in (torch.bfloat16, "bfloat16")
        and 1 <= layer_id < 27
    )
