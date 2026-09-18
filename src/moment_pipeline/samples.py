"""Labelled-window dataset contract for adapting the encoder: the pinned UCI HAPT sample, validation,
seeded user-level splitting, the long-frame bridge to the canonical windowing path, BYOD loaders and
CSV export.

The default dataset is **real**: 179 ten-second windows of smartphone motion data from the UCI
*Smartphone-Based Recognition of Human Activities and Postural Transitions* dataset (HAPT, CC BY 4.0) —
for each of its 30 volunteers and each of the six basic activities (walking, walking upstairs, walking
downstairs, sitting, standing, laying) the centre 512 samples (10.24 s at 50 Hz) of the first labelled
segment long enough to hold them (one volunteer has no downstairs segment that long, hence 179), over six
channels (total accelerometer x/y/z in g, gyroscope x/y/z in rad/s). The
single 79.6 MB archive is pinned by byte size and SHA-256, fetched from the UCI repository at run time and
refused on any mismatch; the raw text files are read from the archive, never extracted; the repository
redistributes none of it. Windows of one volunteer share a body and a phone, so the sample is split
**by user**, never by window.

A record is ``{id, x, label}``: a float32 ``(n_channels, 512)`` array (or a path to a ``.npy``) and its
activity key. ``user`` (or ``group``) is the split unit when present. ``records_to_long_frame`` turns
records into the ``series_id, timestamp, channel, value`` frame that `validation.validate_long_frame`
and `canonical.to_windows` — the same two calls every inference tutorial makes — validate and window.
"""

# ruff: noqa: E501  -- contract prose and error messages are kept on one line for grep-ability
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SEQUENCE_LENGTH

CORPUS_NAME = "UCI HAPT smartphone motion windows (six activities)"
CORPUS_RELEASE = (
    "UCI Machine Learning Repository dataset 341 (Reyes-Ortiz, Anguita, Oneto, Parra 2015), "
    "static archive pinned 2026-09-19"
)
CORPUS_URL = (
    "https://archive.ics.uci.edu/static/public/341/"
    "smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"
)
CORPUS_BYTES = 79_596_192
CORPUS_SHA256 = "4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09"
CORPUS_LICENSE = "CC BY 4.0 (UCI Machine Learning Repository licence notice for dataset 341)"
CORPUS_ARCHIVE_NAME = "hapt.zip"
SAMPLE_RATE_HZ = 50
WINDOW_LENGTH = SEQUENCE_LENGTH  # 512 samples = 10.24 s
ACTIVITIES: dict[int, str] = {
    1: "walking",
    2: "walking_upstairs",
    3: "walking_downstairs",
    4: "sitting",
    5: "standing",
    6: "laying",
}
CHANNELS: tuple[str, ...] = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
N_USERS = 30
DEFAULT_CACHE_DIR = Path("weights") / "hapt"  # working-directory-relative, like the notebook
SAMPLE_SEED = 42
SAMPLE_SPLIT = {
    "train": 18,
    "validation": 6,
    "test": 6,
}  # users; 6 activities each -> 108 / 36 / 36 windows
MIN_RECORDS = 8
MAX_RECORDS = 1_024  # the canonical path's max_windows
MIN_CLASSES = 2
MAX_CLASSES = 100
MAX_CHANNELS = 32
MAX_LABEL_CHARS = 64
MAX_ABS_VALUE = 1_000.0
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9 _.,'()/&+-]{1,64}$")
_EPOCH = np.datetime64("2000-01-01T00:00:00", "ns")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> bytes:
    """Return the pinned HAPT archive bytes from the cache or the UCI repository, digest-verified."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / CORPUS_ARCHIVE_NAME
    data = local.read_bytes() if local.is_file() else b""
    if len(data) != CORPUS_BYTES or _sha256_bytes(data) != CORPUS_SHA256:
        if fetcher is not None:
            data = fetcher(CORPUS_URL)
        else:
            request = urllib.request.Request(
                CORPUS_URL, headers={"User-Agent": "dimer-moment-tutorial/1.0"}
            )
            with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310 (pinned https URL)
                data = response.read()
        if len(data) != CORPUS_BYTES or _sha256_bytes(data) != CORPUS_SHA256:
            raise ValueError(
                f"{CORPUS_ARCHIVE_NAME}: fetched {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, "
                f"pinned {CORPUS_BYTES} / {CORPUS_SHA256[:16]}…"
            )
        local.write_bytes(data)
    return data


def _member(archive: zipfile.ZipFile, name: str) -> str:
    matches = [n for n in archive.namelist() if n.endswith(name)]
    if len(matches) != 1:
        raise ValueError(f"archive must hold exactly one {name!r}, found {len(matches)}")
    return matches[0]


def _read_signal(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    text = archive.read(member).decode("utf-8")
    return np.loadtxt(io.StringIO(text), dtype=np.float32)


def read_corpus(payload: bytes) -> list[dict[str, Any]]:
    """Cut the 179 pinned windows out of the verified archive: per user and activity, the centre 512
    samples of the first labelled segment (by experiment, then start) at least 512 samples long."""
    archive = zipfile.ZipFile(io.BytesIO(payload))
    labels = np.loadtxt(
        io.StringIO(archive.read(_member(archive, "RawData/labels.txt")).decode("utf-8")), dtype=int
    )
    if labels.ndim != 2 or labels.shape[1] != 5:
        raise ValueError(
            "labels.txt must have five integer columns (experiment, user, activity, start, end)"
        )
    chosen: dict[tuple[int, int], tuple[int, int, int]] = {}
    for experiment, user, activity, start, end in labels[np.lexsort((labels[:, 3], labels[:, 0]))]:
        key = (int(user), int(activity))
        if activity not in ACTIVITIES or key in chosen or end - start + 1 < WINDOW_LENGTH:
            continue
        chosen[key] = (int(experiment), int(start), int(end))
    signals: dict[tuple[str, int, int], np.ndarray] = {}
    out = []
    for (user, activity), (experiment, start, end) in sorted(chosen.items()):
        centre = (start + end) // 2
        first = centre - WINDOW_LENGTH // 2
        rows = []
        for sensor in ("acc", "gyro"):
            key = (sensor, experiment, user)
            if key not in signals:
                signals[key] = _read_signal(
                    archive,
                    _member(archive, f"RawData/{sensor}_exp{experiment:02d}_user{user:02d}.txt"),
                )
            rows.append(signals[key][first : first + WINDOW_LENGTH].T)
        x = np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)
        if x.shape != (len(CHANNELS), WINDOW_LENGTH):
            raise ValueError(f"user {user} activity {activity}: window shape {x.shape}")
        out.append(
            {
                "id": f"u{user:02d}-{ACTIVITIES[activity]}",
                "x": x,
                "label": ACTIVITIES[activity],
                "user": f"user{user:02d}",
                "experiment": experiment,
                "segment": [start, end],
                "first_sample": first,
                "source": f"RawData/{{acc,gyro}}_exp{experiment:02d}_user{user:02d}.txt",
            }
        )
    return out


def records_to_long_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The `series_id, timestamp, channel, value` frame of validated records (50 Hz stamps from a fixed
    epoch), one 512-sample series per record, in the shape `validate_long_frame` and `to_windows` take."""
    frames = []
    stamps = _EPOCH + (np.arange(WINDOW_LENGTH) * (1_000_000_000 // SAMPLE_RATE_HZ)).astype(
        "timedelta64[ns]"
    )
    for record in records:
        x = np.asarray(record["x"], dtype=np.float32)
        names = list(record.get("channels") or CHANNELS[: x.shape[0]])
        for c, name in enumerate(names):
            frames.append(
                pd.DataFrame(
                    {
                        "series_id": record["id"],
                        "timestamp": stamps,
                        "channel": name,
                        "value": x[c].astype(np.float64),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded draw of whole users: `sizes` counts users for train / validation / test."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    by_user: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_user.setdefault(str(record["user"]), []).append(dict(record))
    users = sorted(by_user)
    rng.shuffle(users)
    needed = sum(sizes.values())
    if len(users) < needed:
        raise ValueError(f"only {len(users)} users available, need {needed}")
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in sizes}
    cursor = 0
    for name, per_split in sizes.items():
        for user in users[cursor : cursor + per_split]:
            out[name].extend(by_user[user])
        cursor += per_split
    for name in out:
        rng.shuffle(out[name])
        out[name] = [
            {**r, "id": f"{name}-{i:03d}", "source_id": r["id"]} for i, r in enumerate(out[name])
        ]
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    return build_sample_dataset(
        read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes
    )


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label_name = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label_name} must be a mapping with id/x/label")
    for key in ("id", "x", "label"):
        if key not in record:
            raise ValueError(f"{label_name} is missing {key!r}")
    rid, x, label = record["id"], record["x"], record["label"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label_name}: id must match {_ID_RE.pattern}")
    if isinstance(x, str | Path):
        path = Path(x)
        if not path.is_file():
            raise ValueError(f"{label_name}: x file not found: {path}")
        x = np.load(path, allow_pickle=False)
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"{label_name}: x must be a numeric numpy array or a path to a .npy file")
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != WINDOW_LENGTH or not 1 <= x.shape[0] <= MAX_CHANNELS:
        raise ValueError(
            f"{label_name}: x must be (1..{MAX_CHANNELS} channels, {WINDOW_LENGTH}) samples, got shape {x.shape}"
        )
    if not np.isfinite(x).all() or float(np.abs(x).max()) > MAX_ABS_VALUE:
        raise ValueError(f"{label_name}: x must be finite with |value| <= {MAX_ABS_VALUE}")
    if not isinstance(label, str) or not _LABEL_RE.match(label.strip()):
        raise ValueError(
            f"{label_name}: label must be a non-empty string of at most {MAX_LABEL_CHARS} plain characters"
        )
    item = {"id": rid, "x": np.ascontiguousarray(x, dtype=np.float32), "label": label.strip()}
    for key in (
        "source_id",
        "user",
        "group",
        "experiment",
        "segment",
        "first_sample",
        "source",
        "channels",
    ):
        if key in record:
            item[key] = record[key]
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
) -> dict[str, Any]:
    """Structural validation of a labelled-window dataset; raises ValueError before any model import."""
    if (
        isinstance(records, Mapping)
        or not isinstance(records, Sequence)
        or isinstance(records, str | bytes)
    ):
        raise ValueError("records must be a list of {id, x, label} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = [_check_record(record, index) for index, record in enumerate(records)]
    ids = [r["id"] for r in checked]
    if len(set(ids)) != len(ids):
        duplicate = next(i for i in ids if ids.count(i) > 1)
        raise ValueError(f"duplicate id {duplicate!r}")
    n_channels = {int(r["x"].shape[0]) for r in checked}
    if len(n_channels) != 1:
        raise ValueError(
            f"every record must have the same channel count, found {sorted(n_channels)}"
        )
    labels = sorted({r["label"] for r in checked})
    if not MIN_CLASSES <= len(labels) <= MAX_CLASSES:
        raise ValueError(
            f"{len(labels)} distinct labels; {MIN_CLASSES}..{MAX_CLASSES} are required"
        )
    counts = {label: sum(1 for r in checked if r["label"] == label) for label in labels}
    return {
        "records": checked,
        "n_records": len(checked),
        "n_channels": n_channels.pop(),
        "window_length": WINDOW_LENGTH,
        "classes": labels,
        "label_counts": counts,
        "users": len({str(r.get("user", r.get("group", r["id"]))) for r in checked}),
        "value_range": {
            "min": float(min(r["x"].min() for r in checked)),
            "max": float(max(r["x"].max() for r in checked)),
        },
        "digest": dataset_digest(checked),
    }


def class_names(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The sorted label vocabulary of a dataset (the `classes` a head is trained for)."""
    names = sorted({str(r["label"]) for r in records})
    if len(names) < MIN_CLASSES:
        raise ValueError(f"a dataset needs at least {MIN_CLASSES} distinct labels")
    return names


def window_digest(x: np.ndarray) -> str:
    """SHA-256 of the float32 samples (shape + bytes), so a re-saved copy of the same window matches."""
    arr = np.ascontiguousarray(x, dtype=np.float32)
    return _sha256_bytes(f"{arr.shape}:".encode() + arr.tobytes())


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], window_digest(r["x"]), r["label"]] for r in records]
    return _sha256_bytes(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _split_unit(record: Mapping[str, Any]) -> str:
    return str(record.get("user") or record.get("group") or record.get("source_id") or record["id"])


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no window (by sample digest) and no user appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    units: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = window_digest(record["x"])
            if key in seen and seen[key] != name:
                raise ValueError(f"window {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
            unit = _split_unit(record)
            if unit in units and units[unit] != name:
                raise ValueError(f"user {unit!r} has windows in both {units[unit]} and {name}")
            units[unit] = name
    return {name: len(records) for name, records in splits.items()}


def user_summary(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Users and label counts per split (an observation of what the split unit was)."""
    return {
        name: {
            "windows": len(records),
            "users": len({_split_unit(r) for r in records}),
            "label_counts": {
                label: sum(1 for r in records if r["label"] == label)
                for label in sorted({r["label"] for r in records})
            },
        }
        for name, records in splits.items()
    }


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test by split unit (`user` / `group`, else the
    window itself) after de-duplicating windows, so one person's data never straddles splits."""
    if not (
        0.0 <= val_fraction < 1.0
        and 0.0 < test_fraction < 1.0
        and val_fraction + test_fraction < 1.0
    ):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        key = window_digest(record["x"])
        if key not in seen:
            seen.add(key)
            by_unit.setdefault(_split_unit(record), []).append(record)
    rng = random.Random(seed)
    units = sorted(by_unit)
    rng.shuffle(units)
    n_test = max(1, round(len(units) * test_fraction))
    n_val = round(len(units) * val_fraction)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for name, chosen in (
        ("test", units[:n_test]),
        ("validation", units[n_test : n_test + n_val]),
        ("train", units[n_test + n_val :]),
    ):
        for unit in chosen:
            splits[name].extend(by_unit[unit])
    for part in splits.values():
        rng.shuffle(part)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, x, label}` records from a directory or a zip holding `records.csv` (columns `id`, `file`,
    `label`, optional `group`) beside `.npy` windows of shape (channels, 512); arrays are decoded from the
    archive, never extracted to disk."""
    source = Path(path)
    if source.is_dir():
        table = (source / "records.csv").read_text(encoding="utf-8")
        loader = lambda name: (source / name).read_bytes()  # noqa: E731
    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        members = {Path(n).name: n for n in archive.namelist()}
        if "records.csv" not in members:
            raise ValueError("BYOD zip must contain records.csv")
        table = archive.read(members["records.csv"]).decode("utf-8")
        loader = lambda name: archive.read(members[name])  # noqa: E731
    else:
        raise ValueError(
            "BYOD datasets must be a directory or a .zip holding records.csv and the .npy windows"
        )
    rows = list(csv.DictReader(io.StringIO(table)))
    missing = {"id", "file", "label"} - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"records.csv is missing columns {sorted(missing)}")
    out = []
    for row in rows:
        item: dict[str, Any] = {
            "id": row["id"],
            "x": np.load(io.BytesIO(loader(row["file"])), allow_pickle=False),
            "label": row["label"],
        }
        if row.get("group"):
            item["group"] = row["group"]
        out.append(item)
    return out


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the records table of a split (id, file, label, group, provenance) in the shape BYOD expects."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "file", "label", "group", "experiment", "first_sample", "source"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "file": f"{record.get('source_id') or record['id']}.npy",
                    "label": record["label"],
                    "group": _split_unit(record),
                    "experiment": record.get("experiment", ""),
                    "first_sample": record.get("first_sample", ""),
                    "source": record.get("source", ""),
                }
            )
    return out
