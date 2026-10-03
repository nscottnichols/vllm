# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm.config.compilation import CompilationMode
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import gpu_worker, startup_plan
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.

_MINIMAL_MODEL_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 2,
    "vocab_size": 32,
    "torch_dtype": "float16",
}


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _warmup_worker(vllm_config):
    """The minimal Worker surface needed by compile_or_warm_up_model."""
    worker = Worker.__new__(Worker)
    worker.vllm_config = vllm_config
    worker.model_config = vllm_config.model_config
    worker.cache_config = vllm_config.cache_config
    worker.parallel_config = vllm_config.parallel_config
    worker.compilation_config = vllm_config.compilation_config
    worker.observability_config = vllm_config.observability_config
    worker.model_runner = Mock(lora_config=None)
    worker.use_v2_model_runner = True
    worker.execute_model = Mock()
    worker.sample_tokens = Mock()
    worker.init_snapshot = SimpleNamespace(free_memory=8 * GiB_bytes)
    return worker


def _engine_config(tmp_path, skip_model_warmup, kv_cache_memory_bytes=None):
    return EngineArgs(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        enforce_eager=True,
        skip_model_warmup=skip_model_warmup,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
    ).create_engine_config()


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


def test_skip_model_warmup_defaults_false_and_gates_gpu_warmup(tmp_path, monkeypatch):
    """The opt-in flag skips warmup without changing the default path."""
    (tmp_path / "config.json").write_text(json.dumps(_MINIMAL_MODEL_CONFIG))
    default_args = EngineArgs(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        enforce_eager=True,
    )
    default_config = default_args.create_engine_config()
    default_config.compilation_config.mode = CompilationMode.NONE
    assert default_args.skip_model_warmup is False
    assert default_config.skip_model_warmup is False

    enabled_config = _engine_config(tmp_path, True)
    enabled_config.compilation_config.mode = CompilationMode.NONE
    assert enabled_config.skip_model_warmup is True

    kernel_warmup = Mock()
    warmup_kernels = Mock()
    monkeypatch.setattr(gpu_worker, "kernel_warmup", kernel_warmup)
    monkeypatch.setattr(gpu_worker, "warmup_kernels", warmup_kernels)
    for name in (
        "set_random_seed",
        "freeze_gc_heap",
        "maybe_attach_gc_debug_callback",
        "enable_gpu_sync_check",
        "set_torch_threads_for_runtime",
    ):
        monkeypatch.setattr(gpu_worker, name, Mock())
    monkeypatch.setattr("vllm.utils.jit_monitor.activate", Mock())

    worker = _warmup_worker(default_config)
    worker.compile_or_warm_up_model()
    assert kernel_warmup.call_count == 1
    assert warmup_kernels.call_count == 1

    worker.vllm_config = enabled_config
    worker.compile_or_warm_up_model()
    assert kernel_warmup.call_count == 1
    assert warmup_kernels.call_count == 1


def test_skip_model_warmup_skips_profile_with_explicit_kv_cache(tmp_path, monkeypatch):
    """Explicit KV sizing plus opt-in warmup skip avoids the profile run."""
    (tmp_path / "config.json").write_text(json.dumps(_MINIMAL_MODEL_CONFIG))
    default_config = _engine_config(tmp_path, False, kv_cache_memory_bytes=1024)
    skipped_config = _engine_config(tmp_path, True, kv_cache_memory_bytes=1024)
    reserve_mm_ipc_gpu_memory = Mock(return_value=1024)

    monkeypatch.setattr(gpu_worker, "maybe_apply_startup_plan", Mock())
    monkeypatch.setattr(
        gpu_worker,
        "reserve_mm_ipc_gpu_memory",
        reserve_mm_ipc_gpu_memory,
    )

    default_worker = _warmup_worker(default_config)
    assert default_worker.determine_available_memory() == 1024
    assert default_worker.model_runner.profile_run.call_count == 1

    skipped_worker = _warmup_worker(skipped_config)
    assert skipped_worker.determine_available_memory() == 1024
    assert skipped_worker.model_runner.profile_run.call_count == 0
    assert reserve_mm_ipc_gpu_memory.call_count == 2
