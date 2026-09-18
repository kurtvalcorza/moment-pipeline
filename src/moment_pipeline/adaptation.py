"""Bounded supervised adaptation of the MOMENT encoder to a labelled-window classification task — the
first task-head contract of this repository — under an explicit frozen-vs-unfrozen policy.

MOMENT-1-base ships no classification head (its upstream `classification` task head is *not* pretrained,
RFC M-1/M-4), so the head trained here is the caller's own, on top of the pooled 768-d window embeddings
of the `embedding` task instance. The **frozen policy** trains only that linear head on frozen embeddings
(a linear probe); the **unfrozen policy** continues by training the last `trainable_blocks` T5 encoder
blocks with the head end to end; the epoch with the lowest validation log-loss — which may be the probe
itself — is kept and its tensors restored. Metrics are accuracy and macro-F1 with per-class recall and a
confusion matrix, framed by a majority floor and a cosine k-NN vote over the frozen embeddings.

Everything goes through the repository's own path: records → `records_to_long_frame` → `validate_long_frame`
→ `to_windows` → `embed`; nothing here calls momentfm outside `LoadedMoment.pipeline.embed`.
"""

# ruff: noqa: E501  -- contract prose and error messages are kept on one line for grep-ability
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import WindowSet, to_windows
from .config import MomentConfig
from .embedding import embed
from .model import PINNED_WEIGHTS_FILENAME, LoadedMoment
from .samples import MAX_RECORDS, records_to_long_frame, validate_dataset
from .validation import validate_long_frame

ENCODER_BLOCKS = 12
BLOCK_PREFIX = "encoder.block."
DEFAULT_TRAINABLE_BLOCKS = (
    2  # the unfrozen policy trains the last two encoder blocks (14,158,848 parameters)
)
MIN_SCORED_RECORDS = 50  # below this a scored dataset is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.moment-1-base.classifier-adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
POLICY_FROZEN = "frozen encoder + linear probe"

METRIC_DEFINITIONS = {
    "accuracy": "fraction of windows whose predicted label equals the gold label",
    "macro_f1": "unweighted mean over classes of the per-class F1 (precision-recall harmonic mean)",
    "per_class": "precision, recall, F1 and support for every class of the gold label set",
    "confusion": "rows are gold classes, columns predicted classes, in the order of `classes`",
    "log_loss": "mean negative log-probability the head assigns to the gold label (the selection signal)",
}


# ---- metrics -------------------------------------------------------------------------------------


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class precision / recall / F1 / support and the confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} gold labels for {len(y_pred)} predictions")
    if not y_true:
        raise ValueError("at least one labelled window is required")
    index = {c: i for i, c in enumerate(classes)}
    unknown = sorted({*y_true, *y_pred} - set(index))
    if unknown:
        raise ValueError(f"labels outside the class list: {unknown}")
    confusion = [[0] * len(classes) for _ in classes]
    for gold, pred in zip(y_true, y_pred, strict=True):
        confusion[index[gold]][index[pred]] += 1
    per_class = {}
    f1s = []
    for i, name in enumerate(classes):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(len(classes))) - tp
        fn = sum(confusion[i]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        f1s.append(f1)
    correct = sum(confusion[i][i] for i in range(len(classes)))
    return {
        "n": len(y_true),
        "accuracy": correct / len(y_true),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion": confusion,
        "classes": list(classes),
        "definitions": dict(METRIC_DEFINITIONS),
    }


def majority_baseline(
    train_labels: Sequence[str], test_labels: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    """The most frequent training label predicted for every window (the floor)."""
    counts: dict[str, int] = {}
    for label in train_labels:
        counts[label] = counts.get(label, 0) + 1
    majority = max(sorted(counts), key=lambda c: counts[c])
    out = classification_metrics(list(test_labels), [majority] * len(test_labels), classes)
    out["baseline"] = f"majority training label ({majority!r}) predicted for every window"
    return out


def knn_predict(
    train_features: np.ndarray, train_labels: Sequence[str], test_features: np.ndarray, k: int = 5
) -> list[str]:
    """Cosine k-NN vote over L2-normalised feature rows (ties broken by the nearer neighbour)."""
    if k < 1 or k > len(train_labels):
        raise ValueError(f"k must be in 1..{len(train_labels)}")
    a = np.asarray(train_features, dtype=np.float32)
    b = np.asarray(test_features, dtype=np.float32)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    sims = b @ a.T
    out = []
    for row in sims:
        order = np.argsort(-row, kind="stable")[:k]
        votes: dict[str, float] = {}
        for rank, j in enumerate(order):
            label = str(train_labels[j])
            votes[label] = votes.get(label, 0.0) + 1.0 + 1e-6 * (k - rank)
        out.append(max(votes, key=lambda c: votes[c]))
    return out


# ---- the canonical bridge and the frozen features -------------------------------------------------


def windows_for(
    records: Sequence[Mapping[str, Any]], config: MomentConfig | None = None
) -> tuple[WindowSet, list[dict[str, Any]]]:
    """Validated records → the canonical `WindowSet` through the repository's own validation and
    windowing path, with the records re-ordered to the window order (series ids sort lexically)."""
    checked = validate_dataset(records, min_records=1, max_records=MAX_RECORDS)["records"]
    config = config or MomentConfig(task="embedding")
    report, frame = validate_long_frame(records_to_long_frame(checked), config)
    windows = to_windows(frame, config, report=report, frame=frame)
    by_id = {r["id"]: r for r in checked}
    ordered = [by_id[sid] for sid in windows.series_ids]
    if len(ordered) != len(checked):
        raise ValueError("windowing dropped records; ids must be unique")
    return windows, ordered


def features(
    records: Sequence[Mapping[str, Any]], model: LoadedMoment, *, batch_size: int = 8
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Frozen-policy features: one L2-normalised 768-d pooled embedding per validated window, through
    `embed` (eval mode, no gradient), plus the records in window order."""
    windows, ordered = windows_for(records)
    result = embed(windows, model, batch_size=batch_size, warmup=False)
    vectors = np.asarray(result.embeddings, dtype=np.float32)
    vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    return vectors, ordered


def knn_baseline(
    model: LoadedMoment,
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Cosine k-NN vote over the frozen embeddings — what the representation gives with no training."""
    train_vectors, train_ordered = features(train, model)
    test_vectors, test_ordered = features(test, model)
    classes = sorted({r["label"] for r in train_ordered})
    predictions = knn_predict(train_vectors, [r["label"] for r in train_ordered], test_vectors, k=k)
    out = classification_metrics([r["label"] for r in test_ordered], predictions, classes)
    out["baseline"] = f"cosine {k}-NN vote over the frozen embeddings of the training windows"
    out["k"] = k
    return out


# ---- the adapter -----------------------------------------------------------------------------------


@dataclass(eq=False)
class ClassifierAdapter:
    """A trained head over the (possibly adapted) encoder plus the record of how it was chosen."""

    head: Any = field(repr=False)
    classes: list[str]
    policy: str
    config: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    trainable_names: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"policy": self.policy, "classes": list(self.classes), **self.config}


def _trainable_names(model: LoadedMoment, trainable_blocks: int) -> list[str]:
    if (
        isinstance(trainable_blocks, bool)
        or not isinstance(trainable_blocks, int)
        or not 0 <= trainable_blocks <= ENCODER_BLOCKS
    ):
        raise ValueError(f"trainable_blocks must be an int in 0..{ENCODER_BLOCKS}")
    if trainable_blocks == 0:
        return []
    first = ENCODER_BLOCKS - trainable_blocks
    prefixes = tuple(f"{BLOCK_PREFIX}{k}." for k in range(first, ENCODER_BLOCKS))
    return [name for name, _p in model.pipeline.named_parameters() if name.startswith(prefixes)]


def _embed_batch(
    model: LoadedMoment, windows: WindowSet, index: Sequence[int], *, grad: bool
) -> Any:
    import torch

    x = torch.from_numpy(windows.x_enc[list(index)]).to(model.identity.device)
    mask = torch.from_numpy(windows.input_mask[list(index)]).to(model.identity.device)
    if grad:
        out = model.pipeline.embed(x_enc=x, input_mask=mask, reduction="mean")
    else:
        with torch.no_grad():
            out = model.pipeline.embed(x_enc=x, input_mask=mask, reduction="mean")
    return torch.nn.functional.normalize(out.embeddings.float(), dim=-1)


def _probabilities(
    model: LoadedMoment, adapter: ClassifierAdapter, windows: WindowSet, *, batch_size: int = 8
) -> Any:
    import torch

    model.pipeline.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, windows.n_windows, batch_size):
            feats = _embed_batch(
                model, windows, range(start, min(start + batch_size, windows.n_windows)), grad=False
            )
            rows.append(torch.softmax(adapter.head(feats), dim=-1).cpu())
    return torch.cat(rows, dim=0)


def classify(
    model: LoadedMoment, adapter: ClassifierAdapter, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Label validated windows with the trained head over the (possibly adapted) encoder."""
    windows, ordered = windows_for(records)
    probabilities = _probabilities(model, adapter, windows)
    return {
        "ids": [r["id"] for r in ordered],
        "labels": [adapter.classes[int(i)] for i in probabilities.argmax(dim=-1)],
        "probabilities": [[float(v) for v in row] for row in probabilities.tolist()],
        "classes": list(adapter.classes),
        "policy": adapter.policy,
        "decision_rule": "argmax of the head's softmax; not calibrated, no threshold",
        "model_id": model.identity.name,
        "model_revision": model.identity.revision,
    }


def evaluate(
    model: LoadedMoment, adapter: ClassifierAdapter, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score the trained head on validated labelled windows: accuracy, macro-F1, per-class, confusion, log-loss."""
    import torch

    windows, ordered = windows_for(records)
    started = time.perf_counter()
    probabilities = _probabilities(model, adapter, windows)
    gold = [r["label"] for r in ordered]
    unknown = sorted(set(gold) - set(adapter.classes))
    if unknown:
        raise ValueError(f"labels outside the head's classes: {unknown}")
    targets = torch.tensor([adapter.classes.index(g) for g in gold])
    predictions = [adapter.classes[int(i)] for i in probabilities.argmax(dim=-1)]
    metrics = classification_metrics(gold, predictions, adapter.classes)
    metrics["log_loss"] = float(
        -torch.log(probabilities[torch.arange(len(gold)), targets].clamp_min(1e-12)).mean()
    )
    metrics["policy"] = adapter.policy
    metrics["adapted"] = True
    metrics["verdict"] = "measured" if len(gold) >= MIN_SCORED_RECORDS else "measured-small-sample"
    metrics["seconds"] = round(time.perf_counter() - started, 3)
    metrics["model_id"] = model.identity.name
    metrics["model_revision"] = model.identity.revision
    return metrics


def adapt(
    model: LoadedMoment,
    train: Sequence[Mapping[str, Any]],
    val: Sequence[Mapping[str, Any]] | None = None,
    *,
    probe_steps: int = 300,
    probe_lr: float = 1e-2,
    trainable_blocks: int = DEFAULT_TRAINABLE_BLOCKS,
    epochs: int = 3,
    lr: float = 3e-4,
    batch_size: int = 8,
    seed: int = 0,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> ClassifierAdapter:
    """Two-stage bounded adaptation, selected on validation.

    Stage A (the **frozen policy**): a `Linear(768, classes)` head trained full-batch on the frozen
    L2-normalised embeddings with AdamW (`probe_lr`, weight decay 1e-4) for `probe_steps` steps — the linear
    probe — recorded as epoch 0. Stage B (the **unfrozen policy**, when `trainable_blocks` > 0 and `epochs` > 0):
    the last `trainable_blocks` T5 encoder blocks and the head trained end to end on the windows for `epochs`
    epochs (AdamW at `lr`, weight decay 0.01, gradient clipping 1.0, seeded shuffling, no augmentation); the
    patch embedding, the earlier blocks and the final norm stay frozen. Every epoch is scored on `val` and the
    epoch with the **lowest validation log-loss** is kept (epoch 0 competes) and its tensors restored; without
    `val` the final epoch is kept. The encoder inside `model.pipeline` is modified in place when the unfrozen
    policy wins — `embed` then returns different vectors for every window.
    """
    import torch

    if (
        isinstance(probe_steps, bool)
        or not isinstance(probe_steps, int)
        or not 1 <= probe_steps <= 5_000
    ):
        raise ValueError("probe_steps must be an int in 1..5000")
    if not isinstance(probe_lr, int | float) or not 0.0 < float(probe_lr) <= 1.0:
        raise ValueError("probe_lr must be in (0, 1]")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 0 <= epochs <= 20:
        raise ValueError("epochs must be an int in 0..20")
    if not isinstance(lr, int | float) or not 0.0 < float(lr) <= 1e-2:
        raise ValueError("lr must be in (0, 1e-2]")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
        raise ValueError("batch_size must be an int in 1..32")
    names = _trainable_names(model, trainable_blocks)
    if model.identity.task != "embedding":
        raise ValueError("adapt() needs a pipeline loaded with task='embedding'")
    started = time.perf_counter()
    train_windows, train_ordered = windows_for(train)
    val_windows, val_ordered = windows_for(val) if val is not None else (None, None)
    classes = sorted({r["label"] for r in train_ordered})
    if len(classes) < 2:
        raise ValueError("training records need at least two classes")
    targets = torch.tensor([classes.index(r["label"]) for r in train_ordered])
    device = model.identity.device
    d_model = int(model.identity.d_model_effective)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    # Stage A: the probe on frozen features.
    model.pipeline.eval()
    with torch.no_grad():
        frozen = torch.cat(
            [
                _embed_batch(
                    model,
                    train_windows,
                    range(s, min(s + batch_size, train_windows.n_windows)),
                    grad=False,
                )
                for s in range(0, train_windows.n_windows, batch_size)
            ]
        )
    head = torch.nn.Linear(d_model, len(classes)).to(device)
    probe_opt = torch.optim.AdamW(head.parameters(), lr=float(probe_lr), weight_decay=1e-4)
    probe_loss = math.nan
    for _ in range(probe_steps):
        probe_opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(head(frozen.to(device)), targets.to(device))
        loss.backward()
        probe_opt.step()
        probe_loss = float(loss.detach())
    adapter = ClassifierAdapter(
        head=head, classes=classes, policy=POLICY_FROZEN, config={}, trainable_names=[]
    )

    def score() -> dict[str, float] | None:
        if val_windows is None:
            return None
        metrics = evaluate(model, adapter, val_ordered)
        return {
            "n": metrics["n"],
            "accuracy": round(metrics["accuracy"], 6),
            "macro_f1": round(metrics["macro_f1"], 6),
            "log_loss": round(metrics["log_loss"], 6),
        }

    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "stage": "linear probe (frozen encoder)",
            "train_loss": probe_loss,
            "val": score(),
        }
    ]
    params = dict(model.pipeline.named_parameters())
    best_epoch = 0
    best_score = history[0]["val"]["log_loss"] if history[0]["val"] else math.inf
    best_state = {
        "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
        "blocks": {n: params[n].detach().clone() for n in names},
    }
    policy = POLICY_FROZEN
    if names and epochs > 0:
        policy_b = f"unfrozen last {trainable_blocks} blocks + linear head"
        for p in model.pipeline.parameters():
            p.requires_grad_(False)
        for n in names:
            params[n].requires_grad_(True)
        optimiser = torch.optim.AdamW(
            [*head.parameters(), *(params[n] for n in names)], lr=float(lr), weight_decay=0.01
        )
        for epoch in range(1, epochs + 1):
            model.pipeline.train()
            order = list(range(train_windows.n_windows))
            rng.shuffle(order)
            losses = []
            for start in range(0, len(order), batch_size):
                index = order[start : start + batch_size]
                feats = _embed_batch(model, train_windows, index, grad=True)
                loss = torch.nn.functional.cross_entropy(head(feats), targets[index].to(device))
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*head.parameters(), *(params[n] for n in names)], 1.0
                )
                optimiser.step()
                losses.append(float(loss.detach()))
            model.pipeline.eval()
            adapter.policy = policy_b
            val_metrics = score()
            entry = {
                "epoch": epoch,
                "stage": policy_b,
                "train_loss": float(np.mean(losses)),
                "val": val_metrics,
            }
            history.append(entry)
            if progress is not None:
                progress(entry)
            current = (
                val_metrics["log_loss"] if val_metrics else -epoch
            )  # no val: the last epoch wins
            if current < best_score:
                best_epoch, best_score, policy = epoch, current, policy_b
                best_state = {
                    "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
                    "blocks": {n: params[n].detach().clone() for n in names},
                }
        with torch.no_grad():
            head.load_state_dict(best_state["head"])
            for n, value in best_state["blocks"].items():
                params[n].copy_(value)
    for p in model.pipeline.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(False)
    model.pipeline.eval()
    head.eval()
    adapter.policy = policy
    adapter.history = history
    adapter.trainable_names = names if policy != POLICY_FROZEN else []
    adapter.config = {
        "probe_steps": probe_steps,
        "probe_lr": float(probe_lr),
        "probe_final_loss": probe_loss,
        "trainable_blocks": trainable_blocks,
        "n_trainable_head": sum(p.numel() for p in head.parameters()),
        "n_trainable_blocks": sum(params[n].numel() for n in names),
        "n_total": sum(p.numel() for p in model.pipeline.parameters()),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "selection": "lowest validation log-loss (epoch 0 = linear probe)"
        if val is not None
        else "final epoch (no validation split)",
        "lr": float(lr),
        "batch_size": batch_size,
        "n_train": train_windows.n_windows,
        "n_val": val_windows.n_windows if val_windows is not None else 0,
        "seed": seed,
        "seconds": round(time.perf_counter() - started, 3),
    }
    return adapter


# ---- artifacts ------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_artifact(
    model: LoadedMoment,
    adapter: ClassifierAdapter,
    output_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write the head (and any trained encoder-block tensors) as safetensors with a manifest naming the base."""
    from safetensors.torch import save_file

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {
        f"head.{k}": v.detach().cpu().contiguous() for k, v in adapter.head.state_dict().items()
    }
    state = dict(model.pipeline.named_parameters())
    tensors.update({k: state[k].detach().cpu().contiguous() for k in adapter.trainable_names})
    weights_path = out / ARTIFACT_WEIGHTS_NAME
    save_file(tensors, str(weights_path), metadata={"format": "pt"})
    manifest = {
        "format": ARTIFACT_FORMAT,
        "format_version": ARTIFACT_FORMAT_VERSION,
        "base_model": {
            "id": model.identity.name,
            "revision": model.identity.revision,
            "weight_file": PINNED_WEIGHTS_FILENAME,
            "weight_sha256": model.identity.weights_sha256,
            "task": model.identity.task,
        },
        "adapter": adapter.summary(),
        "history": list(adapter.history),
        "tensors": sorted(tensors),
        "files": [
            {
                "path": ARTIFACT_WEIGHTS_NAME,
                "bytes": weights_path.stat().st_size,
                "sha256": _sha256_file(weights_path),
            }
        ],
        "metadata": dict(metadata or {}),
    }
    (out / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def load_artifact(model: LoadedMoment, artifact_dir: str | Path) -> ClassifierAdapter:
    """Verify an adapter's manifest and digest **before** deserialising, rebuild the head from the
    manifest's classes and overlay any encoder-block tensors onto `model.pipeline`."""
    root = Path(artifact_dir)
    manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    base = manifest.get("base_model", {})
    if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
        model.identity.name,
        model.identity.revision,
        model.identity.weights_sha256,
    ):
        raise ValueError(
            "artifact was adapted from a different base model, revision or weight file"
        )
    if model.identity.task != "embedding":
        raise ValueError("load_artifact() needs a pipeline loaded with task='embedding'")
    entry = manifest["files"][0]
    weights_path = root / entry["path"]
    if not weights_path.is_file():
        raise FileNotFoundError(f"artifact weights missing: {weights_path}")
    if (
        _sha256_file(weights_path) != entry["sha256"]
        or weights_path.stat().st_size != entry["bytes"]
    ):
        raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
    classes = list(manifest.get("adapter", {}).get("classes") or [])
    if len(classes) < 2:
        raise ValueError("artifact manifest does not name at least two classes")
    import torch
    from safetensors.torch import load_file

    tensors = load_file(str(weights_path))
    if sorted(tensors) != manifest["tensors"]:
        raise ValueError("artifact tensor names differ from its manifest")
    d_model = int(model.identity.d_model_effective)
    if (
        tuple(tensors.get("head.weight", torch.empty(0)).shape) != (len(classes), d_model)
        or "head.bias" not in tensors
    ):
        raise ValueError("artifact head does not match d_model and the manifest's classes")
    params = dict(model.pipeline.named_parameters())
    block_tensors = {k: v for k, v in tensors.items() if not k.startswith("head.")}
    for key, value in block_tensors.items():
        if key not in params or not key.startswith(BLOCK_PREFIX):
            raise ValueError(
                f"artifact tensor {key} is not an adaptable encoder-block tensor of the base"
            )
        if tuple(value.shape) != tuple(params[key].shape):
            raise ValueError(
                f"artifact tensor {key}: shape {tuple(value.shape)} != {tuple(params[key].shape)}"
            )
    head = torch.nn.Linear(d_model, len(classes))
    head.load_state_dict(
        {"weight": tensors["head.weight"].float(), "bias": tensors["head.bias"].float()}
    )
    head = head.to(model.identity.device).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        for key, value in block_tensors.items():
            params[key].copy_(value.to(params[key].dtype))
    model.pipeline.eval()
    summary = dict(manifest["adapter"])
    policy = summary.pop("policy")
    summary.pop("classes", None)
    return ClassifierAdapter(
        head=head,
        classes=classes,
        policy=policy,
        config=summary,
        history=list(manifest.get("history", [])),
        trainable_names=sorted(block_tensors),
    )
