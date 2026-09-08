"""Pinned, integrity-verified loader for `AutonLab/MOMENT-1-base`.

The standard path is deliberately narrow (RFC "Supply-chain invariants"):

1. only `AutonLab/MOMENT-1-base` at revision `9fea447e…` is accepted — `main`, `latest`,
   any other sha, a local directory and any `s3://` / URL source are refused *before*
   any network call;
2. the resolved snapshot directory must be named after the pinned commit;
3. `pytorch_model.bin` must not be present in the snapshot — it exists upstream at this
   very revision (453,978,525 bytes), so a silent `.bin` fallback is a live hazard, and
   the download uses `allow_patterns` so the file is never fetched;
4. `config.json` and `model.safetensors` digests and the weight byte size are verified;
5. after construction, tensors in the live module are compared byte-for-byte against the
   entries read straight out of `model.safetensors` — that, not a filename, is the proof
   of which weight file was loaded (RFC invariant 7).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from .config import PATCH_LENGTH, PATCH_STRIDE, SEQUENCE_LENGTH, Task

PINNED_MODEL_ID = "AutonLab/MOMENT-1-base"
PINNED_REVISION = "9fea447e740eb968a9e8d80c7562ae122bdb5dde"
PINNED_CONFIG_SHA256 = "f1c66c2bb845229c0ed27a1600dbcc956b85ab21f9e5fd8a1663e6641bed7755"
PINNED_CONFIG_BYTES = 949
PINNED_WEIGHTS_FILENAME = "model.safetensors"
PINNED_WEIGHTS_SHA256 = "1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825"
PINNED_WEIGHTS_BYTES = 453_940_120

#: Present upstream at the pinned revision; must never be downloaded or loaded.
FORBIDDEN_WEIGHT_PATTERNS = ("*.bin",)
KNOWN_FORBIDDEN_WEIGHT_FILE = "pytorch_model.bin"
KNOWN_FORBIDDEN_WEIGHT_SHA256 = (
    "23c3d65bbb6dcd323352029e9fbe4ee3a3da0fff55b45ee4e00f38fff4e9bfb9"
)

ALLOW_PATTERNS = ["config.json", "model.safetensors", "README.md"]

#: The weights licence. There is NO LICENSE file in the HF repo at this revision; MIT is
#: declared in the model-card metadata only. Code licence is tracked separately (LICENSE).
MODEL_LICENSE = "MIT"
MODEL_LICENSE_BASIS = (
    "declared as `license: mit` in the model card metadata at revision "
    "9fea447e740eb968a9e8d80c7562ae122bdb5dde; no LICENSE file exists in the "
    "Hugging Face repository at that revision"
)

#: momentfm is installed from source at an exact upstream commit (RFC M-7).
MOMENTFM_SOURCE_URL = "https://github.com/moment-timeseries-foundation-model/moment"
MOMENTFM_SOURCE_COMMIT = "38f7310ad594100747ca2a8357e9c7ca7d323e0e"

_TASK_TO_UPSTREAM = {"embedding": "embedding", "reconstruction": "reconstruction"}


class ModelSourceError(ValueError):
    """The requested model source is not the single approved pinned source."""


class IntegrityError(RuntimeError):
    """A supply-chain assertion about the downloaded snapshot failed."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        self.code = code
        self.details = details or {}
        super().__init__(f"[{code}] {message}")


@dataclass(frozen=True)
class VerifiedSnapshot:
    """A snapshot directory that has passed every supply-chain assertion."""

    path: Path
    model_id: str
    revision: str
    config_sha256: str
    config_bytes: int
    weights_path: Path
    weights_sha256: str
    weights_bytes: int
    seq_len: int
    patch_len: int
    patch_stride: int
    task_name_in_config: str


@dataclass(frozen=True)
class ModelIdentity:
    """Everything an export needs to say exactly what produced it."""

    name: str
    revision: str
    config_sha256: str
    weights_sha256: str
    weights_bytes: int
    license: str
    license_basis: str
    weight_file_loaded: str
    seq_len: int
    patch_len: int
    patch_stride: int
    d_model_effective: int
    task: str
    device: str
    dtype: str


@dataclass(eq=False)
class LoadedMoment:
    """A task-specific MOMENT instance plus its identity and load proof."""

    pipeline: Any = field(repr=False)
    identity: ModelIdentity
    snapshot: VerifiedSnapshot = field(repr=False)
    proof: dict[str, Any] = field(repr=False, default_factory=dict)


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_pinned_source(model_id: str, revision: str) -> None:
    """Refuse anything but the one approved immutable source. No network is touched."""
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise ModelSourceError("model_id and revision must be strings")
    lowered = model_id.strip().lower()
    if "://" in lowered:
        raise ModelSourceError(
            f"URL-style model sources are refused in the standard path: {model_id!r}"
        )
    if lowered.startswith((".", "/", "~")) or "\\" in model_id or os.path.isabs(model_id):
        raise ModelSourceError(
            f"local-path model sources are refused in the standard path: {model_id!r}"
        )
    if model_id != PINNED_MODEL_ID:
        raise ModelSourceError(
            f"only {PINNED_MODEL_ID!r} is approved for the standard path, got {model_id!r}"
        )
    if revision != PINNED_REVISION:
        raise ModelSourceError(
            "the standard path resolves only the pinned immutable revision "
            f"{PINNED_REVISION!r}; refused {revision!r} "
            "(mutable refs such as 'main'/'latest' and other commits are not approved)"
        )


def find_forbidden_weight_files(directory: str | os.PathLike[str]) -> list[str]:
    root = Path(directory)
    found: list[str] = []
    for pattern in FORBIDDEN_WEIGHT_PATTERNS:
        found.extend(str(p.relative_to(root)) for p in root.rglob(pattern) if p.is_file())
    return sorted(set(found))


def verify_snapshot_dir(
    directory: str | os.PathLike[str],
    *,
    expected_revision: str = PINNED_REVISION,
    expected_config_sha256: str = PINNED_CONFIG_SHA256,
    expected_weights_sha256: str = PINNED_WEIGHTS_SHA256,
    expected_weights_bytes: int = PINNED_WEIGHTS_BYTES,
    check_directory_name: bool = True,
) -> tuple[str, int, str, int, dict[str, Any]]:
    """Assert the snapshot is the pinned artifact. Returns digests plus the parsed config.

    Ordering matters: the forbidden-`.bin` check runs first, so a snapshot polluted with
    `pytorch_model.bin` is refused before anything else is considered.
    """
    import json

    root = Path(directory)
    if not root.is_dir():
        raise IntegrityError("SNAPSHOT_MISSING", f"snapshot directory not found: {root}")

    forbidden = find_forbidden_weight_files(root)
    if forbidden:
        raise IntegrityError(
            "FORBIDDEN_WEIGHT_FILE",
            f"snapshot contains pickle weight file(s) {forbidden}; the standard path "
            "must load model.safetensors and must never risk a silent .bin fallback",
            {"files": forbidden},
        )

    if check_directory_name and root.name != expected_revision:
        raise IntegrityError(
            "REVISION_MISMATCH",
            f"snapshot resolved to {root.name!r}, expected commit {expected_revision!r}",
            {"resolved": root.name, "expected": expected_revision},
        )

    config_path = root / "config.json"
    weights_path = root / PINNED_WEIGHTS_FILENAME
    for path in (config_path, weights_path):
        if not path.is_file():
            raise IntegrityError(
                "SNAPSHOT_INCOMPLETE", f"required file missing from snapshot: {path.name}"
            )

    config_sha = sha256_file(config_path)
    config_bytes = config_path.stat().st_size
    if config_sha != expected_config_sha256:
        raise IntegrityError(
            "CONFIG_DIGEST_MISMATCH",
            f"config.json sha256 {config_sha} != expected {expected_config_sha256}",
            {"actual": config_sha, "expected": expected_config_sha256},
        )

    weights_bytes = weights_path.stat().st_size
    if weights_bytes != expected_weights_bytes:
        raise IntegrityError(
            "WEIGHTS_SIZE_MISMATCH",
            f"{PINNED_WEIGHTS_FILENAME} is {weights_bytes} bytes, "
            f"expected {expected_weights_bytes}",
            {"actual": weights_bytes, "expected": expected_weights_bytes},
        )
    weights_sha = sha256_file(weights_path)
    if weights_sha != expected_weights_sha256:
        raise IntegrityError(
            "WEIGHTS_DIGEST_MISMATCH",
            f"{PINNED_WEIGHTS_FILENAME} sha256 {weights_sha} != expected "
            f"{expected_weights_sha256}",
            {"actual": weights_sha, "expected": expected_weights_sha256},
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config_sha, config_bytes, weights_sha, weights_bytes, config


def fetch_verified_snapshot(
    model_id: str = PINNED_MODEL_ID,
    revision: str = PINNED_REVISION,
    cache_dir: str | os.PathLike[str] | None = None,
) -> VerifiedSnapshot:
    """Download (or reuse) and fully verify the pinned snapshot."""
    assert_pinned_source(model_id, revision)
    local = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=ALLOW_PATTERNS,
        cache_dir=cache_dir,
    )
    config_sha, config_bytes, weights_sha, weights_bytes, config = verify_snapshot_dir(local)

    seq_len = int(config["seq_len"])
    patch_len = int(config["patch_len"])
    patch_stride = int(config["patch_stride_len"])
    if (seq_len, patch_len, patch_stride) != (SEQUENCE_LENGTH, PATCH_LENGTH, PATCH_STRIDE):
        raise IntegrityError(
            "CONFIG_SHAPE_MISMATCH",
            f"config.json declares seq_len/patch_len/stride {seq_len}/{patch_len}/"
            f"{patch_stride}, expected {SEQUENCE_LENGTH}/{PATCH_LENGTH}/{PATCH_STRIDE}",
            {"seq_len": seq_len, "patch_len": patch_len, "patch_stride_len": patch_stride},
        )

    return VerifiedSnapshot(
        path=Path(local),
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha,
        config_bytes=config_bytes,
        weights_path=Path(local) / PINNED_WEIGHTS_FILENAME,
        weights_sha256=weights_sha,
        weights_bytes=weights_bytes,
        seq_len=seq_len,
        patch_len=patch_len,
        patch_stride=patch_stride,
        task_name_in_config=str(config.get("task_name")),
    )


def _selected_encoder_keys(file_keys: list[str], n: int = 5) -> list[str]:
    encoder = sorted(k for k in file_keys if k.startswith("encoder."))
    if len(encoder) < n:
        return encoder
    step = (len(encoder) - 1) / (n - 1)
    return [encoder[round(i * step)] for i in range(n)]


def prove_safetensors_load(pipeline: Any, weights_path: str | os.PathLike[str], task: str) -> dict:
    """Prove the live module holds the tensors from `model.safetensors`.

    Compares a deterministic spread of encoder tensors — and, for the reconstruction
    task, every `head.*` tensor — against `safetensors.safe_open` entries with
    `torch.equal`. A `.bin` load, a re-initialized head or a truncated download all
    fail here; a filename check would not.
    """
    import torch
    from safetensors import safe_open

    state = pipeline.state_dict()
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        file_keys = sorted(handle.keys())
        checked_encoder = _selected_encoder_keys(file_keys)
        if len(checked_encoder) < 3:
            raise IntegrityError(
                "PROOF_UNAVAILABLE",
                f"{weights_path} exposes {len(checked_encoder)} encoder tensors; "
                "cannot prove the load",
            )
        for key in checked_encoder:
            if key not in state:
                raise IntegrityError(
                    "PROOF_KEY_MISSING", f"loaded module has no tensor {key!r}", {"key": key}
                )
            if not torch.equal(state[key].detach().cpu(), handle.get_tensor(key)):
                raise IntegrityError(
                    "PROOF_TENSOR_MISMATCH",
                    f"tensor {key!r} differs from the model.safetensors entry",
                    {"key": key},
                )

        head_keys = sorted(k for k in file_keys if k.startswith("head."))
        checked_head: list[str] = []
        if task == "reconstruction":
            for key in head_keys:
                if key not in state:
                    raise IntegrityError(
                        "PROOF_HEAD_MISSING",
                        f"reconstruction head tensor {key!r} is absent from the loaded "
                        "module; the pretrained head was not restored",
                        {"key": key},
                    )
                if not torch.equal(state[key].detach().cpu(), handle.get_tensor(key)):
                    raise IntegrityError(
                        "PROOF_TENSOR_MISMATCH",
                        f"head tensor {key!r} differs from the model.safetensors entry",
                        {"key": key},
                    )
                checked_head.append(key)
        else:
            live_head_keys = [k for k in state if k.startswith("head.")]
            if live_head_keys:
                raise IntegrityError(
                    "UNEXPECTED_HEAD_PARAMETERS",
                    f"task {task!r} must expose no head parameters, found {live_head_keys}",
                    {"keys": live_head_keys},
                )

    return {
        "weight_file": Path(weights_path).name,
        "n_tensors_in_file": len(file_keys),
        "encoder_tensors_checked": checked_encoder,
        "head_tensors_checked": checked_head,
        "method": "torch.equal against safetensors.safe_open entries",
    }


def load_moment(
    task: Task = "embedding",
    device: str = "auto",
    dtype: str = "float32",
    snapshot: VerifiedSnapshot | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> LoadedMoment:
    """Load a task-specific MOMENT instance from the verified pinned snapshot.

    Task instances are explicit because `MOMENTPipeline.init()` replaces the head when
    the task changes (RFC M-8): `reconstruction` keeps the pretrained `PretrainHead`;
    `embedding` swaps in `nn.Identity` and emits a harmless upstream warning that only
    concerns *heads*, never the encoder that produces the embeddings.
    """
    import torch
    from momentfm import MOMENTPipeline

    if task not in _TASK_TO_UPSTREAM:
        raise ModelSourceError(
            f"v1 exposes only {sorted(_TASK_TO_UPSTREAM)}; refused task {task!r}. "
            "Forecasting and classification heads are NOT pretrained (RFC M-1/M-4)."
        )
    snapshot = snapshot or fetch_verified_snapshot(cache_dir=cache_dir)

    pipeline = MOMENTPipeline.from_pretrained(
        str(snapshot.path), model_kwargs={"task_name": _TASK_TO_UPSTREAM[task]}
    )
    pipeline.init()
    pipeline.eval()

    proof = prove_safetensors_load(pipeline, snapshot.weights_path, task)
    if task == "embedding":
        head_type = type(pipeline.head).__name__
        if head_type != "Identity":
            raise IntegrityError(
                "UNEXPECTED_HEAD",
                f"embedding task must use nn.Identity, found {head_type}",
                {"head": head_type},
            )
        proof["head_type"] = head_type
    else:
        proof["head_type"] = type(pipeline.head).__name__

    d_model = int(pipeline.config.d_model)
    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype
    ]
    pipeline = pipeline.to(device=resolved_device, dtype=torch_dtype)
    pipeline.eval()

    identity = ModelIdentity(
        name=snapshot.model_id,
        revision=snapshot.revision,
        config_sha256=snapshot.config_sha256,
        weights_sha256=snapshot.weights_sha256,
        weights_bytes=snapshot.weights_bytes,
        license=MODEL_LICENSE,
        license_basis=MODEL_LICENSE_BASIS,
        weight_file_loaded=PINNED_WEIGHTS_FILENAME,
        seq_len=snapshot.seq_len,
        patch_len=snapshot.patch_len,
        patch_stride=snapshot.patch_stride,
        d_model_effective=d_model,
        task=task,
        device=resolved_device,
        dtype=dtype,
    )
    return LoadedMoment(pipeline=pipeline, identity=identity, snapshot=snapshot, proof=proof)
