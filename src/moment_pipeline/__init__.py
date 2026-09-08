"""moment-pipeline — a DIMER pipeline for the pinned `AutonLab/MOMENT-1-base` checkpoint.

v1 exposes only what the pretrained base weights support: embeddings,
imputation/reconstruction, and (from Phase 4) reconstruction-based anomaly scoring.
Short-horizon forecasting and classification are deliberately absent from the public API.
"""

from __future__ import annotations

from .canonical import WindowSet, expand_patch_view, to_patch_view, to_windows
from .config import (
    N_PATCHES,
    PATCH_LENGTH,
    PATCH_STRIDE,
    SEQUENCE_LENGTH,
    ConfigError,
    MomentConfig,
    ResourceLimits,
)
from .embedding import EmbeddingResult, TaskMismatchError, embed
from .imputation import ReconstructionResult, reconstruct
from .model import (
    PINNED_CONFIG_SHA256,
    PINNED_MODEL_ID,
    PINNED_REVISION,
    PINNED_WEIGHTS_BYTES,
    PINNED_WEIGHTS_SHA256,
    IntegrityError,
    LoadedMoment,
    ModelIdentity,
    ModelSourceError,
    VerifiedSnapshot,
    fetch_verified_snapshot,
    load_moment,
)
from .provenance import build_provenance
from .validation import ValidationError, ValidationReport, validate_long_frame

__version__ = "0.1.0"

__all__ = [
    "N_PATCHES",
    "PATCH_LENGTH",
    "PATCH_STRIDE",
    "PINNED_CONFIG_SHA256",
    "PINNED_MODEL_ID",
    "PINNED_REVISION",
    "PINNED_WEIGHTS_BYTES",
    "PINNED_WEIGHTS_SHA256",
    "SEQUENCE_LENGTH",
    "ConfigError",
    "EmbeddingResult",
    "IntegrityError",
    "LoadedMoment",
    "ModelIdentity",
    "ModelSourceError",
    "MomentConfig",
    "ReconstructionResult",
    "ResourceLimits",
    "TaskMismatchError",
    "ValidationError",
    "ValidationReport",
    "VerifiedSnapshot",
    "WindowSet",
    "__version__",
    "build_provenance",
    "embed",
    "expand_patch_view",
    "fetch_verified_snapshot",
    "load_moment",
    "reconstruct",
    "to_patch_view",
    "to_windows",
    "validate_long_frame",
]
