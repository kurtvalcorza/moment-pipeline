"""Regression tests for the 2026-10-02 notebook-review fixes of tutorials/moment_imputation_colab.ipynb
(MIM-B1 verdict and fidelity, MIM-M2 interior holdout and baseline naming, MIM-m1 per-channel errors, MIM-m2 holdout
field and activity, MIM-m3 Section 3 identity print, MIM-m4 external access, MIM-m5 plot, MIM-m6 hygiene).

They need only CI's dependencies: the notebook's own cell source is executed with stand-ins for the model result
(no torch, no weights), so what is exercised is the holdout selection, the per-channel accounting, the baseline naming,
the verdict, the fidelity figure and the plot — not pretrained inference."""
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
NOTEBOOK = ROOT / "tutorials" / "moment_imputation_colab.ipynb"


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
    """The source of the named top-level function definitions of a cell (so they run without the model)."""
    tree = ast.parse(source)
    parts = [ast.get_source_segment(source, node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(parts) == len(names), f"missing helper(s) among {names}"
    return "\n\n".join(parts)


# --- MIM-M2 / MIM-m2: the holdout is interior by default and controllable ------------------------------------------


def _choose_holdout(nb: dict):
    ns: dict = {"np": np}
    exec(_defs(_cell_with(nb, "def choose_holdout("), {"choose_holdout"}), ns)
    return ns["choose_holdout"]


def test_mim_m2_default_holdout_is_the_middle_interior_patch(nb: dict) -> None:
    """MIM-M2: on the 256-step left-padded sample the automatic choice has observed points on both sides."""
    choose = _choose_holdout(nb)
    observed = np.zeros(512, dtype=bool)
    observed[256:] = True  # the sample: 256 valid positions, left-padded
    start, complete, interior = choose(observed, 8, -1)
    assert complete == list(range(256, 512, 8))
    assert interior == list(range(264, 504, 8))  # 256 has padding before it, 504 has nothing after it
    assert start == 384 and observed[start - 1] and observed[start + 8]
    assert start + 8 < 512, "the default holdout is not the trailing patch"


def test_mim_m2_holdout_field_is_validated_and_names_the_candidates(nb: dict) -> None:
    """MIM-M2/m2: a HOLDOUT_START that is not a complete patch start is refused with the candidate list."""
    choose = _choose_holdout(nb)
    observed = np.ones(512, dtype=bool)
    observed[100:104] = False
    start, complete, interior = choose(observed, 8, 256)
    assert start == 256
    assert 96 not in complete and 0 not in interior and 504 not in interior
    with pytest.raises(ValueError, match=r"HOLDOUT_START=100 .* one of \["):
        choose(observed, 8, 100)
    with pytest.raises(ValueError, match="no complete 8-step"):
        choose(np.zeros(512, dtype=bool), 8, -1)
    # a gap-free series with no interior candidate falls back to the last complete patch
    short = np.zeros(512, dtype=bool)
    short[504:] = True
    assert choose(short, 8, -1)[0] == 504


def test_mim_m2_section_5_exposes_the_field_and_names_the_placement(nb: dict) -> None:
    cell = _cell_with(nb, "def choose_holdout(")
    assert 'HOLDOUT_START = -1  # @param {type:"integer"}' in cell
    assert "holdout_placement" in cell and "interior (source-observed points on both sides" in cell
    assert "source-observed points after the holdout" in cell
    assert "only the first canonical window is held out and scored" in cell
    md = _markdown(nb)
    assert "`HOLDOUT_START = -1` picks the **middle interior patch**" in md
    assert "Only that first window is held out and scored" in md


# --- MIM-B1 / MIM-m1: verdict, per-channel errors and visible-point fidelity ---------------------------------------


def _section7_helpers(nb: dict) -> dict:
    ns: dict = {"np": np, "pd": pd, "json": json}
    exec(_defs(_cell_with(nb, "def compare_to_baseline("), {"error_summary", "compare_to_baseline", "visible_point_fidelity"}), ns)
    return ns


def test_mim_b1_verdict_names_the_direction_and_the_ratio(nb: dict) -> None:
    """MIM-B1: a model that loses 6.6x to the baseline is called WORSE, with the ratio; wins and ties are named too."""
    h = _section7_helpers(nb)
    ratio, text = h["compare_to_baseline"](1.121, 0.171)
    assert round(ratio, 1) == 6.6 and text.startswith("model WORSE than the baseline by 6.6x")
    assert h["compare_to_baseline"](0.05, 0.10)[1].startswith("model BETTER than the baseline by 2.0x")
    assert "about equal" in h["compare_to_baseline"](0.100, 0.101)[1]
    assert h["compare_to_baseline"](1.0, 0.0)[0] == float("inf")


def test_mim_b1_visible_point_fidelity_distinguishes_tracking_from_noise(nb: dict) -> None:
    """MIM-B1: a reconstruction that tracks the input has Pearson near 1; a flat output near the mean does not."""
    h = _section7_helpers(nb)
    t = np.linspace(0, 6.28, 64)
    truth = np.sin(t)
    good = h["visible_point_fidelity"](truth, truth + 0.01)
    flat = h["visible_point_fidelity"](truth, np.full_like(truth, truth.mean()) + 1e-3 * np.cos(7 * t))
    assert good["pearson"] > 0.99 and good["mae_reconstruction"] < good["mae_mean_predictor"]
    assert abs(flat["pearson"]) < 0.3 and flat["n"] == 64
    assert np.isnan(h["visible_point_fidelity"](truth, np.zeros_like(truth))["pearson"])


def _stand_in_run(nb: dict, *, interior: bool, trailing_gap: bool = False) -> tuple[dict, str]:
    """Execute Section 7 (and 8) of the notebook with a stand-in reconstruction result on a two-channel series.

    The stand-in model returns the channel mean plus small noise (what the review measured), so the test checks the
    accounting, not the model. Returns the namespace and the captured stdout."""
    rng = np.random.default_rng(0)
    n = 256
    stamps = pd.date_range("2026-01-01", periods=n, freq="15min")
    step = np.arange(n, dtype=float)
    channels = ("vibration", "temperature")
    values = {
        "vibration": 1.5 + 0.0008 * step + 0.35 * np.sin(2 * np.pi * step / 32.0),
        "temperature": 28.0 + 0.0015 * step + 2.5 * np.sin(2 * np.pi * (step + 9) / 96.0),
    }
    rows = [("A", s, ch, float(v)) for ch in channels for s, v in zip(stamps, values[ch], strict=True)]
    normalized = pd.DataFrame(rows, columns=["series_id", "timestamp", "channel", "value"])
    seq, patch = 512, 8
    observed = np.zeros(seq, dtype=bool)
    observed[256:] = True
    start = 504 if not interior else 384
    stop = start + patch
    ts = np.full(seq, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ts[256:] = stamps.to_numpy()
    windows = types.SimpleNamespace(channels=channels, series_ids=("A",), window_ids=("A::w0",), timestamps=[ts], patch_length=patch, sequence_length=seq, n_channels=2)
    frame_rows = []
    for ch in channels:
        for pos in range(256, seq):
            truth = float(values[ch][pos - 256])
            recon = float(values[ch].mean() + 0.05 * rng.standard_normal())
            hidden = start <= pos < stop
            frame_rows.append({"series_id": "A", "window_id": "A::w0", "timestamp": pd.Timestamp(stamps[pos - 256]), "channel": ch, "original_value": truth, "imputed_value": recon if hidden else truth, "reconstruction": recon, "source_observed": True, "source_missing": False, "requested_hidden": hidden, "model_hidden": hidden})
    frame = pd.DataFrame(frame_rows)
    held = frame[frame["requested_hidden"]]
    err = held["imputed_value"] - held["original_value"]
    metrics = types.SimpleNamespace(n=len(held), mae=float(err.abs().mean()), rmse=float(np.sqrt((err**2).mean())))
    result = types.SimpleNamespace(to_frame=lambda: frame)
    reports: list = []

    def evaluation_report(result, model=None, sample_kind="synthetic", baseline=None, baseline_id="linear_interpolation"):
        reports.append(baseline_id)
        return {"verdict": "sample-sanity", "baselines": [] if baseline is None else [{"id": baseline_id, "metrics": [{"metric": k, "value": v} for k, v in baseline.items()]}]}

    ns: dict = {
        "np": np, "pd": pd, "json": json, "Path": Path,
        "windows": windows, "normalized": normalized, "result": result, "metrics_stub": metrics,
        "masked_point_metrics": lambda r: metrics, "evaluation_report": evaluation_report, "pipe": None, "sample_kind": "synthetic",
        "start": start, "stop": stop, "holdout_interior": interior, "stem": "moment_imputation",
    }
    Path("outputs").mkdir(exist_ok=True)  # Section 5 creates it on the real path
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(_cell_with(nb, "def compare_to_baseline("), ns)
        exec(_cell_with(nb, "def write_line_svg("), ns)
    ns["_reports"] = reports
    return ns, out.getvalue()


def test_mim_b1_default_path_prints_verdict_per_channel_and_fidelity(nb: dict, tmp_path: Path, monkeypatch) -> None:
    """MIM-B1/m1 (acceptance b): the printed VERDICT, the per-channel model/baseline MAE and the fidelity line exist."""
    monkeypatch.chdir(tmp_path)
    ns, out = _stand_in_run(nb, interior=True)
    assert "VERDICT: model WORSE than the baseline by" in out
    assert "per-channel masked-point errors (model vs baseline" in out
    assert "visible-point fidelity" in out
    for channel in ("vibration", "temperature"):
        row = ns["per_channel"][channel]
        assert {"model", "baseline", "channel_std"} <= set(row)
        assert row["baseline"]["mae"] < row["model"]["mae"]
        assert abs(ns["fidelity"][channel]["pearson"]) < 0.5  # the stand-in returns noise around the mean
    assert ns["mae_ratio"] > 1 and ns["report"]["verdict_line"].startswith("model WORSE")
    assert ns["_reports"] == ["linear_interpolation"]
    assert json.loads((tmp_path / "outputs" / "moment_imputation_evaluation_report.json").read_text())["per_channel"]


def test_mim_m2_interior_baseline_interpolates_and_trailing_is_named_a_hold(nb: dict, tmp_path: Path, monkeypatch) -> None:
    """MIM-M2 (acceptance): with an interior holdout the baseline predictions vary within a channel and it is called
    linear interpolation; with a trailing holdout it is called a last-value hold and reported under that id."""
    monkeypatch.chdir(tmp_path)
    ns, _ = _stand_in_run(nb, interior=True)
    assert ns["baseline_name"].startswith("linear interpolation")
    for channel in ("vibration", "temperature"):
        assert ns["per_channel"][channel]["baseline_predictions_all_equal"] is False
    ns2, out2 = _stand_in_run(nb, interior=False)
    assert ns2["baseline_name"].startswith("last-value hold")
    assert "tutorial baseline (last-value hold" in out2
    for channel in ("vibration", "temperature"):
        assert ns2["per_channel"][channel]["baseline_predictions_all_equal"] is True
    assert ns2["_reports"] == ["last_value_hold"]


# --- MIM-m5: the plot shows the held-out span of the scored window with truth, model and baseline -----------------


def test_mim_m5_plot_centres_on_the_holdout_with_three_layers(nb: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns, _ = _stand_in_run(nb, interior=True)
    paths = ns["plot_paths"]
    assert [p.name for p in paths] == ["moment_imputation_vibration.svg", "moment_imputation_temperature.svg"]
    svg = paths[0].read_text(encoding="utf-8")
    assert "held out" in svg and 'fill="#fde68a"' in svg
    assert "truth (source)" in svg and "model reconstruction / imputed" in svg and "baseline: linear interpolation" in svg
    assert "positions 384-391 held out" in svg and "series A, channel vibration" in svg
    assert svg.count("<polyline") == 3, "truth and model are continuous; the baseline is one segment over the holdout"


def test_mim_m5_plot_breaks_lines_at_missing_values(nb: dict, tmp_path: Path) -> None:
    ns: dict = {"np": np, "Path": Path}
    exec(_defs(_cell_with(nb, "def write_line_svg("), {"write_line_svg"}), ns)
    path = ns["write_line_svg"](tmp_path / "x.svg", [("a", [1.0, 2.0, None, 3.0, 4.0]), ("b", [None, None, 2.5, None, None])], title="t", shade=(2, 2))
    svg = path.read_text(encoding="utf-8")
    assert svg.count("<polyline") == 2  # 'a' is split into two segments; a single point draws nothing


# --- MIM-m3 / MIM-m4 / MIM-m6: Section 3 identity, external access, opening documents the executor variables ------


def test_mim_m3_section_3_prints_the_real_device_and_weight_file(nb: dict) -> None:
    cell = _cell_with(nb, "pipe = load_moment(")
    assert "'device': pipe.identity.device" in cell and "'dtype': pipe.identity.dtype" in cell
    assert "'weight_file': pipe.identity.weight_file_loaded" in cell and "'weights_dir': str(WEIGHTS_DIR)" in cell
    assert "getattr(pipe, 'device', None)" not in cell and "'local-snapshot'" not in cell


def test_mim_m4_external_access_names_github_and_a_measured_download(nb: dict) -> None:
    md = _markdown(nb)
    bullet = re.search(r"\*\*External access:\*\*[^\n]*", md).group(0)
    assert "GitHub (`github.com`)" in bullet and "`momentfm`" in bullet
    assert re.search(r"Measured size of the locked wheels: ~\d+\.\d GB \(sum of the locked manylinux x86_64 wheel sizes from PyPI release metadata, 2026-10-07", bullet)
    assert "No GitHub access" not in bullet
    assert "CUDA build" in md  # the Compute bullet no longer calls torch merely "the largest download"


def test_mim_m6_opening_documents_both_executor_variables_and_byod_field(nb: dict) -> None:
    opening = _src(nb["cells"][0])
    assert "`DIMER_NOTEBOOK_CI_PREINSTALLED=1`" in opening and "`DIMER_BYOD_PATH`" in opening
    assert "`BYOD_PATH` form field" in opening and "exists only there" in opening
    cell = _cell_with(nb, "USE_BYOD = False")
    assert 'BYOD_PATH = ""  # @param {type:"string"}' in cell
    assert cell.index("if BYOD_PATH:") < cell.index("from google.colab import files"), "the path field is read before any Colab import"


def test_mim_b1_prose_reads_the_loss_and_the_registry_no_longer_claims_working_imputation(nb: dict) -> None:
    md = _markdown(nb)
    assert "did not beat the naive baseline on any recorded run" in md
    assert "The recorded result is a negative one" in md
    assert "the model lost at every one (interior median 0.99 vs 0.062)" in md
    assert "that MOMENT beats the interpolation baseline reliably" not in md
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "Recorded result: a documented negative one" in registry
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "### Recorded result on the synthetic sample (a documented negative result)" in card


def test_mim_m2_activity_asks_to_change_the_holdout(nb: dict) -> None:
    md = _markdown(nb)
    assert "**Change one thing (optional):** pick another interior candidate" in md
    assert "re-run Sections 5–9" in md


def test_every_code_cell_parses_and_no_placeholders(nb: dict) -> None:
    for i, source in enumerate(_code_cells(nb)):
        ast.parse(source, f"cell{i}")
    md = _markdown(nb)
    assert "{{" not in md and "{MODEL_ID}" not in md and "@P:" not in md
