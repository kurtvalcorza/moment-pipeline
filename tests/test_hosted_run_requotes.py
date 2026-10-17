"""Worked answers quote the hosted Colab T4 runs of commit 032c262 (2026-10-17), REL13-labelled."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = "docs/execution-evidence/2026-10-17/"
BLOBS = {
    "moment_anomaly_detection_colab": "d68ae2035924",
    "moment_classification_colab": "0bc12ed4ab0d",
    "moment_embeddings_colab": "c2322ba8cda1",
    "moment_imputation_colab": "f76d3be12ca7",
}


def _markdown(stem: str) -> str:
    nb = json.loads((ROOT / "tutorials" / f"{stem}.ipynb").read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


@pytest.mark.parametrize("stem", sorted(BLOBS))
def test_worked_answers_cite_the_032c262_runs(stem: str) -> None:
    md = _markdown(stem)
    assert "the `7f442e3` blob of this notebook" not in md
    label = (
        "the hosted Colab T4 run of 17 October 2026 of commit `032c262` "
        f"(MOMENT-1-large @ `ca58581b`, blob `{BLOBS[stem]}`, Colab account 2; "
        "REL13: an earlier blob of this notebook carrying the same modules; "
        f"executed notebook in `{EVIDENCE}`)"
    )
    assert label in md
    stem_evidence = ROOT / EVIDENCE / f"{stem}_032c262_colab-cli-t4.ipynb"
    assert stem_evidence.is_file()


@pytest.mark.parametrize("stem", ["moment_anomaly_detection_colab", "moment_embeddings_colab"])
def test_md02_no_longer_denies_recorded_outputs(stem: str) -> None:
    md = _markdown(stem)
    assert "No per-cell outputs of the hosted runs are recorded" not in md
    recorded = "The executed notebooks of the hosted runs are recorded byte-for-byte under "
    assert recorded + "`docs/execution-evidence/`" in md
