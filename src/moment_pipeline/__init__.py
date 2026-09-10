"""moment-pipeline — a DIMER pipeline for the pinned `AutonLab/MOMENT-1-base` checkpoint.

v1 exposes only what the pretrained base weights support: embeddings,
imputation/reconstruction, and reconstruction-based anomaly scoring.
Short-horizon forecasting and classification are deliberately absent from the public API.
"""

from __future__ import annotations

from .anomaly import (
    ANOMALY_LOSSES,
    CHANNEL_AGGREGATIONS,
    AnomalyConfigError,
    AnomalyResult,
    aggregate_channels,
    residual,
    score_anomalies,
    score_from_reconstruction,
)
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
from .csvio import read_long_csv_bytes
from .embedding import EmbeddingResult, TaskMismatchError, embed
from .imputation import (
    DegenerateMaskError,
    ImputationMetrics,
    ReconstructionResult,
    impute,
    masked_point_metrics,
    reconstruct,
)
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
    "ANOMALY_LOSSES",
    "CHANNEL_AGGREGATIONS",
    "N_PATCHES",
    "PATCH_LENGTH",
    "PATCH_STRIDE",
    "PINNED_CONFIG_SHA256",
    "PINNED_MODEL_ID",
    "PINNED_REVISION",
    "PINNED_WEIGHTS_BYTES",
    "PINNED_WEIGHTS_SHA256",
    "SEQUENCE_LENGTH",
    "AnomalyConfigError",
    "AnomalyResult",
    "ConfigError",
    "DegenerateMaskError",
    "EmbeddingResult",
    "ImputationMetrics",
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
    "aggregate_channels",
    "build_provenance",
    "embed",
    "expand_patch_view",
    "fetch_verified_snapshot",
    "impute",
    "load_moment",
    "masked_point_metrics",
    "read_long_csv_bytes",
    "reconstruct",
    "residual",
    "score_anomalies",
    "score_from_reconstruction",
    "to_patch_view",
    "to_windows",
    "validate_long_frame",
]
