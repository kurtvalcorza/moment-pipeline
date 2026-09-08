"""Configuration for the pinned MOMENT-1-base pipeline.

Every value here is a DIMER-facing contract knob. The three structural constants
(`SEQUENCE_LENGTH`, `PATCH_LENGTH`, `PATCH_STRIDE`) mirror the pinned checkpoint's
`config.json` and are asserted against the downloaded file at load time
(`moment_pipeline.model.load_moment`) rather than trusted blindly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

#: Task names this pipeline exposes. v1 deliberately excludes ``forecasting`` and
#: ``classification``: those heads are freshly initialized, not pretrained (RFC M-1/M-4).
Task = Literal["embedding", "reconstruction"]

#: Structural constants of ``AutonLab/MOMENT-1-base`` at the pinned revision.
SEQUENCE_LENGTH = 512
PATCH_LENGTH = 8
PATCH_STRIDE = 8
N_PATCHES = SEQUENCE_LENGTH // PATCH_LENGTH  # 64

#: Finite sentinel substituted for missing payloads before the tensor reaches MOMENT.
#: Missingness itself is carried exclusively by the masks (RFC M-2, validation rules 11-12).
DEFAULT_PREFILL_VALUE = 0.0


class ConfigError(ValueError):
    """Raised when a `MomentConfig` is internally inconsistent or unsupported."""


@dataclass(frozen=True)
class ResourceLimits:
    """Explicit resource guards (RFC common-validation rule 10)."""

    max_rows: int = 5_000_000
    max_series: int = 1024
    max_channels: int = 32
    max_windows: int = 1024

    def __post_init__(self) -> None:
        for name in ("max_rows", "max_series", "max_channels", "max_windows"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1, got {getattr(self, name)}")


@dataclass(frozen=True)
class MomentConfig:
    """Runtime configuration for one MOMENT task instance.

    `task` is part of the identity of a loaded model: MOMENT replaces its head when
    the task changes (RFC M-8), so a config and a loaded pipeline travel together.
    """

    task: Task = "embedding"
    sequence_length: int = SEQUENCE_LENGTH
    patch_length: int = PATCH_LENGTH
    patch_stride: int = PATCH_STRIDE
    batch_size: int = 8
    device: str = "auto"
    dtype: str = "float32"
    prefill_value: float = DEFAULT_PREFILL_VALUE
    strict_frequency: bool = False
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self) -> None:
        if self.task not in ("embedding", "reconstruction"):
            raise ConfigError(
                f"task must be 'embedding' or 'reconstruction' (v1 scope), got {self.task!r}"
            )
        if self.sequence_length != SEQUENCE_LENGTH:
            raise ConfigError(
                f"the pinned checkpoint is fixed at sequence_length={SEQUENCE_LENGTH}, "
                f"got {self.sequence_length}"
            )
        if self.patch_length != PATCH_LENGTH or self.patch_stride != PATCH_STRIDE:
            raise ConfigError(
                f"the pinned checkpoint is fixed at patch_len={PATCH_LENGTH} / "
                f"stride={PATCH_STRIDE}, got {self.patch_length}/{self.patch_stride}"
            )
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ConfigError(f"device must be one of auto|cpu|cuda, got {self.device!r}")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ConfigError(f"unsupported dtype {self.dtype!r}")
        import math

        if not math.isfinite(self.prefill_value):
            raise ConfigError(
                "prefill_value must be finite (that is the whole point), got "
                f"{self.prefill_value!r}"
            )

    @property
    def n_patches(self) -> int:
        return self.sequence_length // self.patch_length

    def resolved_device(self) -> str:
        """Resolve ``auto`` against the actual runtime. Never a hard CUDA requirement."""
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a hard dependency in practice
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
