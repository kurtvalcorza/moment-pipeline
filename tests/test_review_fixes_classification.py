"""Regression tests for the 2026-10-02 notebook-review fixes of tutorials/moment_classification_colab.ipynb
(MCL-M2 rerun instructions over the frozen snapshot, MCL-M3 no build-record result presented as the reader's and the
delta read in windows, MCL-m3 timing figures with their environment, MCL-m4 braces, MCL-m5 Section 3 identity,
MCL-m6 GitHub, MCL-m7 within/between-class cosine table).

They need only CI's dependencies: the class-cosine helper and the delta-in-windows reading are executed on synthetic
vectors and numbers (no torch, no weights)."""
# ruff: noqa: E501  -- assertion messages and notebook source fragments are kept on single lines

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "moment_classification_colab.ipynb"


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


# --- MCL-M3: no build-record number presented as the reader's result; the delta is read in windows ---------------


def test_mcl_m3_no_build_record_result_in_the_prose(nb: dict) -> None:
    md = _markdown(nb)
    assert "build record" not in md
    assert "77.8" not in md and "0.776" not in md, "the pre-flight accuracy/macro-F1 of another blob is gone"
    assert "partly recovering the static postures" not in md and "static postures partly recovered" not in md
    assert "epoch 2 selected, 75.0 %" not in md
    for cell in (c for c in nb["cells"] if c["cell_type"] == "markdown"):
        text = _src(cell)
        if "75.0 %" in text or "72.2 %" in text:
            assert "Kaggle T4" in text or "Kaggle Tesla T4" in text, f"a quoted number must name its run: {text[:120]!r}"


def test_mcl_m3_delta_is_converted_to_windows_in_code(nb: dict) -> None:
    cell = _cell_with(nb, "delta_windows = int(round(")
    assert "comparison['delta_in_windows']" in cell and "'one_window_in_points': round(100.0 / n_test, 1)" in cell
    ns: dict = {}
    body = "\n".join(line for line in cell.splitlines() if line.startswith(("n_test = ", "delta_windows = ", "comparison['delta_in_windows']")))
    for policy, delta, expected in (("frozen", 0.0, "the probe was kept"), ("unfrozen last 2 blocks + linear head", 0.0278, "within chance"), ("unfrozen last 2 blocks + linear head", 0.111, "more than two windows")):
        ns = {"adapted_test": {"n": 36}, "comparison": {"delta_vs_frozen": {"accuracy": delta}}, "adapter": type("A", (), {"policy": policy})(), "POLICY_FROZEN": "frozen"}
        exec(body, ns)
        assert ns["comparison"]["delta_in_windows"]["reading"].startswith(expected)
    assert ns["comparison"]["delta_in_windows"]["delta_windows"] == 4 and ns["comparison"]["delta_in_windows"]["one_window_in_points"] == 2.8


def test_mcl_m3_interpretation_reads_the_delta_in_windows_and_names_device_variation(nb: dict) -> None:
    md = _markdown(nb)
    assert "`delta_windows = round(delta × n_test)`" in md
    assert "Read the delta from *your* output" in md
    assert "CPU and CUDA runs of the same split can differ by a window" in md
    assert "neither the head nor the unfrozen blocks can recover what it removed" in md
    assert "## Conclusion (your notes)" in md


# --- MCL-M2: rerun instructions, frozen snapshot before each policy -------------------------------------------------


def test_mcl_m2_rerun_instructions_and_frozen_restore_before_both_policies(nb: dict) -> None:
    md = _markdown(nb)
    assert "then re-run Section 6 and then Sections 7–9 in order" in md
    assert "re-run Section 6, then Sections 7–9" in md
    assert "re-run from Section 4" in md
    code = _code_cells(nb)
    s6 = _cell_with(nb, "_FROZEN_ENCODER = copy.deepcopy(")
    s7 = _cell_with(nb, "TRAINABLE_BLOCKS = 2  # @param")
    assert code.index(s6) < code.index(s7)
    for cell in (s6, s7):
        assert "pipe.pipeline.load_state_dict(_FROZEN_ENCODER, strict=True)" in cell
        assert cell.index("load_state_dict(_FROZEN_ENCODER") < cell.index("adapt(pipe,")


# --- MCL-m7: within/between-class cosine table ------------------------------------------------------------------


def test_mcl_m7_class_cosine_table_reads_the_gap(nb: dict) -> None:
    ns: dict = {"np": np}
    exec(_defs(_cell_with(nb, "def class_cosine_table("), {"class_cosine_table"}), ns)
    rng = np.random.default_rng(0)
    centres = {"walking": rng.standard_normal(16), "sitting": rng.standard_normal(16)}
    vectors, labels = [], []
    for name, centre in centres.items():
        for _ in range(6):
            v = centre + 0.3 * rng.standard_normal(16)
            vectors.append(v / np.linalg.norm(v))
            labels.append(name)
    table = ns["class_cosine_table"](np.array(vectors), labels)
    assert set(table) == {"walking", "sitting"}
    for row in table.values():
        assert row["n"] == 6 and row["within"] > row["between"] and row["gap"] == round(row["within"] - row["between"], 4)
    md = _markdown(nb)
    assert "a single pair is an anecdote" in md and "(0.984 vs 0.997)" in md


# --- MCL-m3: timing figures name their environment ----------------------------------------------------------------


def test_mcl_m3_timing_figures_name_their_environment(nb: dict) -> None:
    md = _markdown(nb)
    assert "about 150 s" not in md and "about three minutes on CPU" not in md and "about half a minute on CPU" not in md
    assert "307.5 s on the local CPU pre-flight of 2026-09-19" in md
    assert "294 s on the recorded Kaggle Tesla T4 run of 2026-09-19" in md
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "seven min of model time" not in registry and "430 s of cell time on the local CPU pre-flight of 2026-09-19" in registry


# --- MCL-m4 / MCL-m5 / MCL-m6 ---------------------------------------------------------------------------------------


def test_mcl_m4_prerequisites_render_single_braces(nb: dict) -> None:
    md = _markdown(nb)
    assert "`{id, x, label}`" in md and "[A-Za-z0-9_.:-]{1,64}" in md
    assert "{{" not in md


def test_mcl_m5_section_3_prints_identity_and_explains_the_head_warning(nb: dict) -> None:
    cell = _cell_with(nb, "pipe = load_moment(task=\"embedding\", weights_dir=WEIGHTS_DIR)\nprint(")
    assert "'device': pipe.identity.device" in cell and "'weight_file': pipe.identity.weight_file_loaded" in cell
    assert "getattr(pipe, 'device', None)" not in cell
    assert "Only reconstruction head is pre-trained" in _markdown(nb)


def test_mcl_m6_external_access_names_github_and_a_measured_download(nb: dict) -> None:
    bullet = re.search(r"\*\*External access:\*\*[^\n]*", _markdown(nb)).group(0)
    assert "GitHub (`github.com`)" in bullet and "`momentfm`" in bullet and "No GitHub access" not in bullet
    assert re.search(r"Measured size of the locked wheels: ~\d+\.\d GB", bullet)


def test_every_code_cell_parses_and_no_placeholders(nb: dict) -> None:
    for i, source in enumerate(_code_cells(nb)):
        ast.parse(source, f"cell{i}")
    md = _markdown(nb)
    assert "{{" not in md and "{MODEL_ID}" not in md and "@P:" not in md
