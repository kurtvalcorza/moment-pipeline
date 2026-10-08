"""Regression tests for the 2026-10-02 notebook-review fixes of tutorials/moment_embeddings_colab.ipynb
(MEM-M2 missingness semantics and experiment, MEM-m2 level/scale invariance prose, MEM-m3 Section 3 identity and head
warning, MEM-m4 external access, MEM-m5 per-window norms and executor variables).

They need only CI's dependencies: the experiment cell is executed with a stand-in `embed` (a deterministic function of
the window values, no torch, no weights), so what is exercised is the gap construction, the interpolation, the
comparison and the printed reading — not pretrained inference."""
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
NOTEBOOK = ROOT / "tutorials" / "moment_embeddings_colab.ipynb"


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


# --- MEM-M2: the printed label says what it means, and the experiment runs on the default path -------------------


def test_mem_m2_no_printed_text_implies_missing_values_are_ignored(nb: dict) -> None:
    """MEM-M2 (acceptance a): the old 'missingness visible to model: False' line is gone; the label names the mask."""
    code = "\n".join(_code_cells(nb))
    assert 'print("missingness visible to model:"' not in code
    cell = _cell_with(nb, "per-point missingness mask passed to model")
    assert "(pre-filled values ARE seen as data)" in cell
    assert 'print("missingness policy:", result.missingness_policy)' in cell
    md = _markdown(nb)
    assert "**that filled value is seen by the encoder as data**" in md
    assert "never reaches the model" not in md


def _sample_frame() -> pd.DataFrame:
    n = 256
    stamps = pd.date_range("2026-01-01", periods=n, freq="15min")
    step = np.arange(n, dtype=float)
    rows = []
    for channel, base, amplitude, period in (("vibration", 1.5, 0.35, 32.0), ("temperature", 28.0, 2.5, 96.0)):
        values = base + amplitude * np.sin(2 * np.pi * step / period)
        rows.extend(("A", s, channel, float(v)) for s, v in zip(stamps, values, strict=True))
    return pd.DataFrame(rows, columns=["series_id", "timestamp", "channel", "value"])


def _run_experiment(nb: dict, stride: int = 8) -> tuple[dict, str]:
    """Execute Section 6b with stand-ins: validation/windowing turn the frame into a (channels, 256) array with NaN
    pre-filled by `prefill_value`, and `embed` returns a fixed random projection of the pre-filled values."""
    rng = np.random.default_rng(1)
    projection = rng.standard_normal((2 * 256, 16))
    config = types.SimpleNamespace(prefill_value=0.0)

    def validate_long_frame(frame, config):
        return None, frame

    def to_windows(source, config, report=None, frame=None):
        grid = []
        for channel in ("vibration", "temperature"):
            values = source[source["channel"] == channel].sort_values("timestamp")["value"].to_numpy(dtype=float)
            grid.append(values)
        grid = np.stack(grid)
        missing = ~np.isfinite(grid)
        filled = np.where(missing, config.prefill_value, grid)
        return types.SimpleNamespace(values=filled, masked_point_fraction=float(missing.mean()))

    def embed(windows, pipe, warmup=False):
        vector = windows.values.reshape(-1) @ projection
        return types.SimpleNamespace(embeddings=vector[None, :], missingness_visible_to_model=False)

    frame = _sample_frame()
    clean = embed(to_windows(frame, config), None)
    ns: dict = {"np": np, "pd": pd, "frame": frame, "config": config, "pipe": None, "result": clean, "validate_long_frame": validate_long_frame, "to_windows": to_windows, "embed": embed}
    source = _cell_with(nb, "def with_gaps(").replace("MISSING_STRIDE = 8  # @param", f"MISSING_STRIDE = {stride}  # @param")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(source, ns)
    return ns, out.getvalue()


def test_mem_m2_experiment_embeds_a_gappy_copy_and_prints_its_similarity(nb: dict) -> None:
    """MEM-M2 (acceptance b): the default-path cell embeds a NaN-bearing copy and reports cosine/L2 to the clean vector;
    zeros written into the gaps give the same vector as leaving them missing; interpolation lands closer."""
    ns, out = _run_experiment(nb)
    e = ns["experiment"]
    assert e["missing_fraction_of_window"] == pytest.approx(0.125)
    assert e["per_point_mask_passed_to_model"] is False
    assert e["max_abs_diff_missing_vs_zeros_written"] < 1e-9
    assert e["l2_clean_vs_interpolated"] < e["l2_clean_vs_missing"]
    assert e["cosine_clean_vs_interpolated"] > e["cosine_clean_vs_missing"]
    assert "closer to the clean vector: interpolated copy" in out
    assert "zeros written into the gaps give the same vector as leaving them missing" in out
    gappy = ns["gappy_frame"]
    per_channel = gappy.groupby("channel")["value"].apply(lambda v: v.isna().sum())
    assert set(per_channel) == {32}, "one point in every 8 per channel is missing"
    assert not ns["interpolate_gaps"](gappy)["value"].isna().any()


def test_mem_m2_experiment_refuses_a_degenerate_stride(nb: dict) -> None:
    with pytest.raises(ValueError, match="MISSING_STRIDE must be at least 2"):
        _run_experiment(nb, stride=1)


def test_mem_m2_experiment_has_a_prediction_and_a_what_to_notice_note(nb: dict) -> None:
    md = _markdown(nb)
    assert "## 6b. Missingness experiment" in md
    assert "**Predict before running:** which copy will land closer to the clean vector" in md
    assert "<details><summary>What to notice</summary>" in md
    assert "gappy windows are not embedded safely by default" in md


# --- MEM-m2: level/scale invariance stated --------------------------------------------------------------------------


def test_mem_m2_section_6_states_the_revin_invariance(nb: dict) -> None:
    md = _markdown(nb)
    assert "invariant to its level and scale" in md and "(RevIN)" in md
    assert "carry it as a separate feature" in md


# --- MEM-m3 / MEM-m4 / MEM-m5 -------------------------------------------------------------------------------------


def test_mem_m3_section_3_prints_identity_and_explains_the_head_warning(nb: dict) -> None:
    cell = _cell_with(nb, "pipe = load_moment(")
    assert "'device': pipe.identity.device" in cell and "'weight_file': pipe.identity.weight_file_loaded" in cell
    assert "getattr(pipe, 'device', None)" not in cell and "'local-snapshot'" not in cell
    md = _markdown(nb)
    assert "Only reconstruction head is pre-trained" in md and "nothing untrained is involved" in md


def test_mem_m4_external_access_names_github_and_a_measured_download(nb: dict) -> None:
    bullet = re.search(r"\*\*External access:\*\*[^\n]*", _markdown(nb)).group(0)
    assert "GitHub (`github.com`)" in bullet and "`momentfm`" in bullet and "No GitHub access" not in bullet
    assert re.search(r"Measured size of the locked wheels: ~\d+\.\d GB \(sum of the locked manylinux x86_64 wheel sizes from PyPI release metadata, 2026-10-07", bullet)


def test_mem_m5_norm_line_covers_every_window_and_opening_documents_executor_variables(nb: dict) -> None:
    cell = _cell_with(nb, "per-point missingness mask passed to model")
    assert "norms = np.linalg.norm(np.asarray(result.embeddings, dtype=np.float64), axis=1)" in cell
    assert "result.embeddings[0] ** 2" not in cell
    opening = _src(nb["cells"][0])
    assert "`DIMER_NOTEBOOK_CI_PREINSTALLED=1`" in opening and "`DIMER_BYOD_PATH`" in opening and "`BYOD_PATH` form field" in opening
    byod = _cell_with(nb, "USE_BYOD = False")
    assert byod.index("if BYOD_PATH:") < byod.index("from google.colab import files")


def test_every_code_cell_parses_and_no_placeholders(nb: dict) -> None:
    for i, source in enumerate(_code_cells(nb)):
        ast.parse(source, f"cell{i}")
    md = _markdown(nb)
    assert "{{" not in md and "{MODEL_ID}" not in md and "@P:" not in md
