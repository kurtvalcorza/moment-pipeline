"""Structured provenance attached to every v1 export.

Shape follows the RFC "Output/provenance contract": a `model`, `runtime` and `inference`
block. Two things the RFC template leaves as placeholders are filled in concretely here
because they are the supply-chain claims that actually matter:

* `model.weight_file_loaded` — which weight file the runtime proof confirmed, not which
  one was requested;
* `runtime.momentfm_source` — momentfm is installed from an exact upstream commit, so the
  version string `0.1.5` alone is recorded together with the commit that produced it.
"""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .canonical import WindowSet
from .model import (
    MOMENTFM_SOURCE_COMMIT,
    MOMENTFM_SOURCE_URL,
    LoadedMoment,
    ModelIdentity,
)


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:  # pragma: no cover - all of these are hard dependencies
        return None


def measure_latency[T](fn: Callable[[], T], warmup: int = 1) -> tuple[T, float]:
    """Run `fn` after `warmup` discarded calls and return (result, seconds)."""
    for _ in range(warmup):
        fn()
    started = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - started


def model_block(identity: ModelIdentity) -> dict[str, Any]:
    return {
        "name": identity.name,
        "revision": identity.revision,
        "config_sha256": identity.config_sha256,
        "weights_sha256": identity.weights_sha256,
        "weights_bytes": identity.weights_bytes,
        "weight_file_loaded": identity.weight_file_loaded,
        "license": identity.license,
        "license_basis": identity.license_basis,
        "task": identity.task,
        "seq_len": identity.seq_len,
        "patch_len": identity.patch_len,
        "patch_stride": identity.patch_stride,
        "d_model_effective": identity.d_model_effective,
    }


def runtime_block(identity: ModelIdentity) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": sys.platform,
        "momentfm": _version("momentfm"),
        "momentfm_source": {"url": MOMENTFM_SOURCE_URL, "commit": MOMENTFM_SOURCE_COMMIT},
        "huggingface_hub": _version("huggingface-hub"),
        "transformers": _version("transformers"),
        "torch": _version("torch"),
        "numpy": _version("numpy"),
        "device": identity.device,
        "dtype": identity.dtype,
    }


def _inference_common(identity: ModelIdentity, windows: WindowSet) -> dict[str, Any]:
    return {
        "task": identity.task,
        "sequence_length": windows.sequence_length,
        "patch_length": windows.patch_length,
        "n_series": len(set(windows.series_ids)),
        "n_channels": windows.n_channels,
        "n_windows": windows.n_windows,
        "n_truncated_windows": int(sum(windows.truncated)),
        "n_padded_windows": int(sum(windows.padded)),
    }


def build_provenance(model: LoadedMoment, windows: WindowSet, result: Any) -> dict[str, Any]:
    """Assemble the export metadata for one inference call."""
    from .embedding import EmbeddingResult
    from .imputation import ReconstructionResult

    inference = _inference_common(model.identity, windows)
    inference["latency_seconds"] = round(float(getattr(result, "latency_seconds", 0.0)), 6)

    if isinstance(result, EmbeddingResult):
        inference["reduction"] = result.reduction
        inference["channel_policy"] = result.channel_policy
        inference["embedding_dim"] = result.d_model
    elif isinstance(result, ReconstructionResult):
        inference["masked_point_fraction"] = result.masked_point_fraction
        inference["masked_patch_fraction"] = result.masked_patch_fraction
        inference["masked_point_count"] = result.masked_point_count
        inference["masked_patch_count"] = result.masked_patch_count
        inference["mask_policy"] = (
            "explicit patch-quantized mask; mask=None is never passed to "
            "MOMENT.reconstruct"
        )
    else:  # pragma: no cover - defensive
        raise TypeError(f"unsupported result type {type(result).__name__}")

    return {
        "model": model_block(model.identity),
        "runtime": runtime_block(model.identity),
        "inference": inference,
        "load_proof": {
            key: value for key, value in model.proof.items() if key != "encoder_tensors_checked"
        }
        | {"encoder_tensors_checked": list(model.proof.get("encoder_tensors_checked", []))},
    }
