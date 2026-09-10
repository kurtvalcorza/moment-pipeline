from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
NOTEBOOKS = {
    "moment_embeddings_colab.ipynb": "embeddings",
    "moment_imputation_colab.ipynb": "imputation",
    "moment_anomaly_detection_colab.ipynb": "anomaly-scoring",
}
PLACEHOLDER = re.compile(r"\b(?:TODO|TBD|FIXME)\b")
NUMBERED_STAGE = re.compile(r"^## \d+\.")


def _load(name: str) -> dict:
    return json.loads((TUTORIALS / name).read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _texts(payload: dict) -> tuple[str, str]:
    markdown = "\n".join(
        _source(cell) for cell in payload["cells"] if cell.get("cell_type") == "markdown"
    )
    code = "\n".join(
        _source(cell) for cell in payload["cells"] if cell.get("cell_type") == "code"
    )
    return markdown, code


def test_all_release_notebooks_declare_task_inference_profile_and_spec() -> None:
    for name, capability in NOTEBOOKS.items():
        payload = _load(name)
        metadata = payload.get("metadata", {}).get("dimer", {})
        assert metadata == {
            "notebook_profile": "TASK-INFERENCE",
            "notebook_spec": "1.0",
            "capability": capability,
        }
        markdown, _ = _texts(payload)
        assert "**Profile:** `TASK-INFERENCE`" in markdown
        assert "**Notebook spec:** `1.0`" in markdown


def test_common_release_grade_learning_contract_is_durable() -> None:
    for name in NOTEBOOKS:
        payload = _load(name)
        markdown, code = _texts(payload)
        lower = markdown.lower()

        assert "no gradient training" in lower
        assert "upstream vs. this repository" in lower
        assert "by the end of this notebook you will be able to" in lower
        assert "prerequisites" in lower
        assert all(column in markdown for column in ("series_id", "timestamp", "channel", "value"))
        assert "byod privacy" in lower
        assert "5,000,000" in markdown and "1,024" in markdown and "512-step" in markdown
        assert "does not prove" in lower
        assert not PLACEHOLDER.search(markdown)

        assert 'UV_VERSION = "0.12.9"' in code
        assert "requirements.lock.txt" in code
        assert "PINNED_REVISION" in code
        assert 'version("momentfm")' in code
        assert "validate_long_frame" in code
        assert "build_provenance" in code
        assert "read_long_csv_bytes(payload)" in code
        assert "pd.read_csv(io.BytesIO(payload))" not in code

        for cell in payload["cells"]:
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_numbered_major_stages_explain_success_semantics() -> None:
    for name in NOTEBOOKS:
        payload = _load(name)
        for cell in payload["cells"]:
            if cell.get("cell_type") != "markdown":
                continue
            source = _source(cell).strip()
            if not NUMBERED_STAGE.match(source):
                continue
            lines = source.splitlines()
            prose = "\n".join(lines[1:]).strip().lower()
            assert prose, f"{name}: numbered stage is heading-only: {lines[0]}"
            assert "success" in prose, f"{name}: stage lacks successful-output semantics: {lines[0]}"


def test_notebook_json_remains_reviewable_and_stably_formatted() -> None:
    for name in NOTEBOOKS:
        raw = (TUTORIALS / name).read_text(encoding="utf-8")
        assert raw.endswith("\n")
        assert raw.startswith("{\n")
        assert '\n  "cells": [' in raw
        assert raw.count("\n") > 100, f"{name} appears minified or unstable for line review"


def test_embeddings_tutorial_carries_representation_semantics() -> None:
    payload = _load("moment_embeddings_colab.ipynb")
    markdown, code = _texts(payload)
    lower = markdown.lower()
    assert "no intrinsic accuracy metric" in lower
    assert "embeddings are representations, not predictions" in lower
    assert "missing" in lower and "visible to the encoder" in lower
    assert "embedding shape" in code
    assert "moment_embeddings.csv" in code


def test_imputation_tutorial_carries_masked_evaluation_and_baseline_semantics() -> None:
    payload = _load("moment_imputation_colab.ipynb")
    markdown, code = _texts(payload)
    lower = markdown.lower()
    assert "sample metrics" in lower or "sample/tutorial evidence" in lower
    assert "linear interpolation baseline" in lower
    assert "no uncertainty interval" in lower
    assert "masked_point_metrics" in code
    assert "model_masked_point_fraction" in code
    assert "masked_patch_fraction" in code
    assert "linear_interpolation_baseline" in code


def test_anomaly_tutorial_carries_score_direction_threshold_and_byod_semantics() -> None:
    payload = _load("moment_anomaly_detection_colab.ipynb")
    markdown, code = _texts(payload)
    lower = markdown.lower()
    assert "higher residual scores mean stronger anomaly evidence" in lower
    assert "no universal/default threshold" in lower
    assert "not a calibrated detector" in lower
    assert "labels = None" in code
    assert "top_k_recall" in code
    assert "moment_anomaly_scores.csv" in code


def test_tutorial_registry_maps_every_notebook_to_normative_profile() -> None:
    registry = (TUTORIALS / "README.md").read_text(encoding="utf-8")
    assert "DIMER Notebook Specification:** `1.0`" in registry
    for name in NOTEBOOKS:
        assert name in registry
    assert registry.count("`TASK-INFERENCE`") >= len(NOTEBOOKS)
    assert "current revision must pass live-notebook CI" in registry
