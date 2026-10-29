"""DAT20: in all four MOMENT notebooks a refused BYOD input shows exactly one line, ``Refused: <file>: <reason>``, with
no traceback, and the cell raises so Run all stops there; the router relays it as the same line. The in-cell
invalidation is kept, and a genuine failure keeps its traceback. Each BYOD cell is executed from its committed notebook."""
# ruff: noqa: E501

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

import moment_pipeline as mp

ROOT = Path(__file__).resolve().parents[1]
CSV_KINDS = ("anomaly_detection", "embeddings", "imputation")
REFUSAL_CELLS = {
    "anomaly_detection": "moment_anomaly_detection-31",
    "classification": "moment_classification-35",
    "embeddings": "moment_embeddings-31",
    "imputation": "moment_imputation-31",
}


def _notebook(kind: str) -> dict:
    return json.loads((ROOT / "tutorials" / f"moment_{kind}_colab.ipynb").read_text(encoding="utf-8"))


def _line(exc: BaseException) -> str:
    assert type(exc).__name__ == "BYODRefusal" and exc.dimer_refusal is True
    rendered = exc._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0] and "Traceback" not in rendered[0]
    assert exc.__suppress_context__ or exc.__cause__ is None
    return rendered[0]


def _csv_run(kind: str):
    return importlib.import_module(f"test_fleet_sweep_fixes_{kind}")._run_section4


def _run_classification(path: str, ns: dict | None = None) -> dict:
    cell = next(c for c in _notebook("classification")["cells"] if c.get("id") == REFUSAL_CELLS["classification"])
    source = "".join(cell["source"]).replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    source = source.replace('BYOD_PATH = ""  # @param', f"BYOD_PATH = {path!r}  # @param", 1)
    if ns is None:
        ns = {name: getattr(mp, name) for name in dir(mp) if not name.startswith("_")}
        ns.update(os=os, Path=Path)
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


@pytest.mark.parametrize("kind", CSV_KINDS)
def test_the_scripted_refusal_input_is_one_line_naming_the_file(kind, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime-host\n", encoding="utf-8")
    with pytest.raises(Exception) as caught:
        _csv_run(kind)(_notebook(kind), True, str(hostname))
    line = _line(caught.value)
    assert line.startswith("Refused: hostname: ") and "BYOD input refused" not in line
    assert "NOT results for this input" in (tmp_path / "outputs" / "RUN_INVALIDATED.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("kind", CSV_KINDS)
def test_a_directory_and_no_upload_dialog_are_refused_by_name(kind, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "etc").mkdir()
    with pytest.raises(Exception) as caught:
        _csv_run(kind)(_notebook(kind), True, str(tmp_path / "etc"))
    assert _line(caught.value).startswith("Refused: etc: the path does not exist or is not a file")
    monkeypatch.setitem(__import__("sys").modules, "google.colab", None)
    with pytest.raises(Exception) as caught:
        _csv_run(kind)(_notebook(kind), True, "")
    assert _line(caught.value).startswith("Refused: (no file): ")


def test_classification_refuses_the_scripted_input_a_directory_and_no_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime-host\n", encoding="utf-8")
    (tmp_path / "etc").mkdir()
    monkeypatch.setitem(__import__("sys").modules, "google.colab", None)
    for path, prefix in ((str(hostname), "Refused: hostname: BYOD datasets must be"), (str(tmp_path / "etc"), "Refused: etc: BYOD path "), ("", "Refused: (no file): BYOD is on but its path field is empty")):
        with pytest.raises(Exception) as caught:
            _run_classification(path)
        assert _line(caught.value).startswith(prefix)


def test_classification_drops_the_earlier_results_and_keeps_genuine_failures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime-host\n", encoding="utf-8")
    ns = {name: getattr(mp, name) for name in dir(mp) if not name.startswith("_")}
    ns.update(os=os, Path=Path, splits={"stale": []}, train_records=["stale"], result={"stale": True})
    with pytest.raises(Exception) as caught:
        _run_classification(str(hostname), ns)
    _line(caught.value)
    assert not {"splits", "train_records", "result"} & set(ns)

    def broken(path):
        raise RuntimeError("internal bug")

    ns = {name: getattr(mp, name) for name in dir(mp) if not name.startswith("_")}
    ns.update(os=os, Path=Path)
    exec_ns = dict(ns)
    cell = next(c for c in _notebook("classification")["cells"] if c.get("id") == REFUSAL_CELLS["classification"])
    source = "".join(cell["source"]).replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    source = source.replace('BYOD_PATH = ""  # @param', f"BYOD_PATH = {str(hostname)!r}  # @param", 1)
    source = source.replace("        records = load_byod_dataset(byod_path)\n", "        records = _broken_loader(byod_path)\n", 1)
    exec_ns["_broken_loader"] = broken
    with pytest.raises(RuntimeError, match="internal bug"):
        exec(compile(source, "<section 4>", "exec"), exec_ns)


@pytest.mark.parametrize("kind", sorted(REFUSAL_CELLS))
def test_the_refusal_cell_ids_and_the_router_line(kind):
    nb = _notebook(kind)
    cell = next(c for c in nb["cells"] if c.get("id") == REFUSAL_CELLS[kind])
    source = "".join(cell["source"])
    assert cell["cell_type"] == "code" and 'BYOD_PATH = ""  # @param' in source and "raise refuse_byod(" in source
    assert 'return [f"Refused: {self}"]' in source and "BYOD input refused" not in source
    router = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "class IsolatedCellRefusal(Exception):" in "".join(c["source"]))
    assert 'return [f"Refused: {self}"]' in router and 'getattr(exc, "dimer_refusal", False)' in router
