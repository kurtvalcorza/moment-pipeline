"""Regression tests for the 2026-10-02 notebook-review fixes of tutorials/moment_anomaly_detection_colab.ipynb
(MAD-M2 per-channel residual scale, channel choice and pooled ranking, MAD-m2 the aggregation experiment, MAD-m3 the
|z-score| baseline, MAD-m4 Section 3 identity, MAD-m5 external access, MAD-m6 per-series plot, MAD-m7 hygiene).

They need only CI's dependencies: Sections 6–8 are executed on a stand-in score table built from the sample's own
formulas (the "model" residual is synthetic: large and smooth on the clean channel, spiky on the labelled one, as the
review measured), so the accounting, the ranking, the baseline and the plot are exercised — not pretrained inference."""
# ruff: noqa: E501  -- assertion messages and notebook source fragments are kept on single lines

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "moment_anomaly_detection_colab.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code_cells(nb: dict) -> list[str]:
    return [_src(c) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell_with(nb: dict, needle: str) -> str:
    hits = [c for c in _code_cells(nb) if needle in c]
    assert len(hits) == 1, f"exactly one code cell contains {needle!r}, got {len(hits)}"
    return hits[0]


def _defs(source: str, names: set[str]) -> str:
    tree = ast.parse(source)
    parts = [ast.get_source_segment(source, node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(parts) == len(names), f"missing helper(s) among {names}"
    return "\n\n".join(parts)


def _top_k_recall(scores: pd.DataFrame, labels: pd.DataFrame, *, channel: str | None = None) -> dict:
    """The package's ranking metric, re-stated here so the test needs no torch import (same semantics as anomaly.py)."""
    ranked = scores[(scores["channel"] == channel) & scores["scored"]].copy().sort_values("anomaly_score", ascending=False, kind="mergesort").reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    marks = labels.copy()
    marks["timestamp"] = pd.to_datetime(marks["timestamp"])
    ranked = ranked.merge(marks, on=["series_id", "timestamp"], how="left")
    ranked["is_injected_anomaly"] = ranked["is_injected_anomaly"].fillna(False).astype(bool)
    k = int(ranked["is_injected_anomaly"].sum())
    hits = int(ranked.head(k)["is_injected_anomaly"].sum())
    return {"k": k, "channel": channel, "value": hits / k, "injected_ranks": ranked.loc[ranked["is_injected_anomaly"], ["timestamp", "rank", "anomaly_score"]].to_dict("records"), "ranked": ranked}


def _stand_in(series: tuple[str, ...] = ("A",), labelled: bool = True):
    """A score table shaped like `AnomalyResult.to_frame()` for the sample formulas: the clean `temperature` channel
    gets a large smooth residual, `vibration` a small one with the three injected spikes on top (what the review saw)."""
    n = 256
    stamps = pd.date_range("2026-01-01", periods=n, freq="15min")
    step = np.arange(n, dtype=float)
    rows, raw = [], []
    for series_id in series:
        for channel, base, amplitude, period in (("vibration", 1.5, 0.35, 32.0), ("temperature", 28.0, 2.5, 96.0)):
            values = base + amplitude * np.sin(2 * np.pi * step / period)
            residual = 1.2 + 0.8 * np.abs(np.cos(2 * np.pi * step / 48.0)) if channel == "temperature" else 0.2 + 0.1 * np.abs(np.sin(2 * np.pi * step / 24.0))
            if channel == "vibration":
                for index, delta in {184: 2.8, 201: -3.2, 233: 3.6}.items():
                    values[index] += delta
                    residual[index] += abs(delta)
            for i in range(n):
                rows.append({"series_id": series_id, "window_id": f"{series_id}::w0", "timestamp": stamps[i], "channel": channel, "reconstruction": float(values[i] - residual[i]), "reconstruction_error": float(residual[i]), "anomaly_score": float(residual[i]), "scored": True})
                raw.append((series_id, stamps[i], channel, float(values[i])))
    scores = pd.DataFrame(rows)
    normalized = pd.DataFrame(raw, columns=["series_id", "timestamp", "channel", "value"])
    labels = None
    if labelled:
        labels = pd.DataFrame({"series_id": "A", "timestamp": stamps, "is_injected_anomaly": False})
        labels.loc[[184, 201, 233], "is_injected_anomaly"] = True
    return scores, normalized, labels


def _run_sections(nb: dict, *, series=("A",), labelled=True, score_channel_field="") -> tuple[dict, str]:
    scores, normalized, labels = _stand_in(series, labelled)
    reports: list = []

    def evaluation_report(result, labels, model=None, sample_kind="synthetic", channel=None, baseline=None, baseline_id="x"):
        reports.append((channel, baseline, baseline_id))
        return {"verdict": "sample-sanity" if labels is not None else "not-measurable", "metrics": [{"id": "top_k_recall", "value": 1.0}] if labels is not None else [], "baselines": []}

    ns: dict = {"np": np, "pd": pd, "json": json, "Path": Path, "scores": scores, "normalized": normalized, "labels": labels, "result": types.SimpleNamespace(score_policy="p", threshold_policy="t", scored_point_fraction=1.0), "pipe": None, "sample_kind": "synthetic" if labelled else "BYOD", "top_k_recall": _top_k_recall, "evaluation_report": evaluation_report, "windows": types.SimpleNamespace(series_ids=series), "stem": "moment_anomaly_detection"}
    Path("outputs").mkdir(exist_ok=True)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        # Section 6 after the model call: the per-channel summary
        cell6 = _cell_with(nb, "def residual_summary(")
        exec(cell6[cell6.index("def residual_summary("):], ns)
        source7 = _cell_with(nb, "def choose_score_channel(").replace('SCORE_CHANNEL = ""  # @param', f'SCORE_CHANNEL = {score_channel_field!r}  # @param')
        exec(source7, ns)
        exec(_cell_with(nb, "def write_score_svg("), ns)
    ns["_reports"] = reports
    return ns, out.getvalue()


# --- MAD-M2 -----------------------------------------------------------------------------------------------------------


def test_mad_m2_per_channel_residual_statistics_are_printed_and_explained(nb: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns, out = _run_sections(nb)
    summary = ns["channel_summary"]
    assert list(summary.columns) == ["scored_points", "median_residual", "max_residual"]
    assert summary.loc["temperature", "median_residual"] > summary.loc["vibration", "median_residual"]
    assert "per-channel residual scale" in out
    md = _markdown(nb)
    assert "**raw residuals are comparable only within one channel of one series**" in md
    assert "the clean `temperature` channel (no spike in it) has a far larger typical residual than `vibration`" in md


def test_mad_m2_channel_choice_is_stated_before_the_recall_and_pooled_ranking_is_shown(nb: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns, out = _run_sections(nb)
    assert ns["score_channel"] == "vibration" and "the channel the label table names" in ns["channel_reason"]
    assert out.index("ranked channel: vibration") < out.index("tutorial top-3 recall")
    assert "pooled all-channel ranking: injected points rank" in out
    assert ns["pooled_ranks"]["rank"].min() > 0 and len(ns["pooled_ranks"]) == 3
    assert "highest residuals of every channel" in out and "temperature" in out.split("highest residuals of every channel")[1]
    assert ns["report"]["ranked_channel"]["channels_not_ranked"] == ["temperature"]


def test_mad_m2_score_channel_field_overrides_and_rejects_unknown(nb: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns, _ = _run_sections(nb, score_channel_field="temperature")
    assert ns["score_channel"] == "temperature" and ns["channel_reason"] == "chosen in the SCORE_CHANNEL field"
    with pytest.raises(ValueError, match="SCORE_CHANNEL='flow' is not one of the scored channels"):
        _run_sections(nb, score_channel_field="flow")


def test_mad_m2_byod_without_labels_lists_every_channel_and_names_the_omitted_ones(nb: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns, out = _run_sections(nb, series=("A", "B"), labelled=False)
    assert ns["score_channel"] == "vibration" and "the first channel" in ns["channel_reason"]
    assert "channels not ranked for the metric" in out and "No labels supplied" in out
    assert ns["baseline_metrics"] is None and ns["_reports"][0][1] is None


# --- MAD-m3: naive baseline in the report -------------------------------------------------------------------------


def test_mad_m3_abs_zscore_baseline_is_computed_on_the_same_channel_and_k(nb: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns, out = _run_sections(nb)
    assert ns["baseline_metrics"] == {"top_k_recall": 1.0}  # three isolated spikes: a |z-score| finds them too
    assert ns["_reports"] == [("vibration", {"top_k_recall": 1.0}, "abs_zscore")]
    assert "VERDICT: model top-k recall 1.0 vs |z-score| baseline 1.0" in out
    assert "does not separate from a one-line statistic" in out
    z = ns["abs_zscore_scores"](ns["normalized"], "A", "vibration")
    assert list(z.columns) == ["series_id", "timestamp", "channel", "anomaly_score", "scored"] and z["scored"].all()


# --- MAD-m6: plot --------------------------------------------------------------------------------------------------------


def test_mad_m6_plot_is_per_series_with_axis_labels_and_spike_markers(nb: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns, _ = _run_sections(nb)
    svg = ns["score_plots"][0].read_text(encoding="utf-8")
    assert svg.count("<circle") == 6, "three labelled spikes marked on the score panel and on the raw-value panel"
    assert "raw residual score" in svg and "raw value" in svg and "o = labelled spike" in svg
    assert "2026-01-01 00:00:00" in svg and "2026-01-03 15:45:00" in svg
    assert svg.count("<line") == 2 and len(re.findall(r'font-size="10">[-0-9.e]+</text>', svg)) == 6, "two labelled axes with three ticks each"
    ns2, _ = _run_sections(nb, series=("A", "B"), labelled=False)
    assert [p.name for p in ns2["score_plots"]] == ["moment_anomaly_scores_A.svg", "moment_anomaly_scores_B.svg"]


# --- MAD-m2: the suggested experiment can change the outcome -----------------------------------------------------


def test_mad_m2_experiment_is_the_aggregation_with_rerun_instructions(nb: dict) -> None:
    md = _markdown(nb)
    assert 'channel_aggregation="mean"' in md and 'then `"max"`' in md
    assert "then Sections 7–8" in md
    assert "cannot change the ranking while channels are *not* aggregated" in md
    assert 'switch `loss="mse"` and compare how the injected spikes rank' not in md


# --- MAD-m4 / MAD-m5 / MAD-m7 ------------------------------------------------------------------------------------


def test_mad_m4_section_3_prints_the_real_device_and_weight_file(nb: dict) -> None:
    cell = _cell_with(nb, "pipe = load_moment(")
    assert "'device': pipe.identity.device" in cell and "'weight_file': pipe.identity.weight_file_loaded" in cell
    assert "getattr(pipe, 'device', None)" not in cell


def test_mad_m5_external_access_names_github_and_a_measured_download(nb: dict) -> None:
    bullet = re.search(r"\*\*External access:\*\*[^\n]*", _markdown(nb)).group(0)
    assert "GitHub (`github.com`)" in bullet and "`momentfm`" in bullet and "No GitHub access" not in bullet
    assert re.search(r"Measured size of the locked wheels: ~\d+\.\d GB \(sum of the locked manylinux x86_64 wheel sizes from PyPI release metadata, 2026-10-07", bullet)


def test_mad_m7_opening_documents_executor_variables_and_byod_field(nb: dict) -> None:
    opening = _src(nb["cells"][0])
    assert "`DIMER_NOTEBOOK_CI_PREINSTALLED=1`" in opening and "`DIMER_BYOD_PATH`" in opening and "`BYOD_PATH` form field" in opening
    byod = _cell_with(nb, "USE_BYOD = False")
    assert byod.index("if BYOD_PATH:") < byod.index("from google.colab import files")


def test_every_code_cell_parses_and_no_placeholders(nb: dict) -> None:
    for i, source in enumerate(_code_cells(nb)):
        ast.parse(source, f"cell{i}")
    md = _markdown(nb)
    assert "{{" not in md and "{MODEL_ID}" not in md and "@P:" not in md
