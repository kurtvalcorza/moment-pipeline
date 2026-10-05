"""Regression tests for the 2026-10-05 fleet-sweep fixes of the tutorial notebook (SWP-R restart guard, SWP-G guided
layer, SWP-B BYOD path; anomaly_detection notebook). They need only CI's dependencies: the notebook's own kernel cell is executed with a stand-in
IPython shell, and the isolated worker it starts runs on this interpreter (a stand-in for the managed CPython), so
the routing protocol, environment reuse and Section 1 idempotence are exercised for real without any download."""
# ruff: noqa: E501  -- assertion messages and notebook source fragments are kept on single lines

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook_sweep", TOOLS / "build_notebook.py")
TEMPLATE = _load("notebook_template_sweep", TOOLS / "notebook_template_anomaly_detection.py").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


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


def _kernel_cell(nb: dict) -> dict:
    cells = [c for c in nb["cells"] if c["cell_type"] == "code" and "# dimer: kernel cell" in _src(c)]
    assert len(cells) == 1, "exactly one kernel (bootstrap) cell"
    return cells[0]


def _cell_with(nb: dict, marker: str) -> str:
    found = [s for s in _code_cells(nb) if marker in s]
    assert len(found) == 1, f"exactly one code cell must contain {marker!r}"
    return found[0]


def _function(source: str, name: str, namespace: dict) -> object:
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"<{name}>", "exec"), namespace)
    return namespace[name]


# --- SWP-R: restart guard replaced by the isolated runtime -------------------------------------------------------


def test_swp_r_no_pip_install_or_restart_in_any_cell(nb: dict) -> None:
    """SWP-R: nothing is pip-installed into the kernel and no cell asks for a restart."""
    code = "\n".join(_code_cells(nb))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in code
    assert "packages_distributions" not in code


def test_swp_r_lock_is_carried_hash_locked_and_matches_pins(nb: dict) -> None:
    """SWP-R: the carried lock is the committed lock, every pin is in it at the same version, every entry hashed."""
    kernel = _src(_kernel_cell(nb))
    lock_text = (ROOT / TEMPLATE["lock"]).read_text(encoding="utf-8")
    digest = hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_SHA256 = {digest!r}" in kernel
    build.check_lock(build.load_context(ROOT, TEMPLATE)["pins"], lock_text)
    assert "'--require-hashes', '--only-binary', ':all:'" in kernel


def test_swp_r_environment_keyed_on_lock_and_child_env_cleaned(nb: dict) -> None:
    """SWP-R: the environment folder is keyed on the lock digest and reused; the worker gets MPLBACKEND=Agg and no
    PYTHONPATH/PYTHONHOME/PYTHONSTARTUP."""
    kernel = _src(_kernel_cell(nb))
    assert "'dimer_isolated_env_' + LOCK_SHA256[:12]" in kernel
    assert "elif _isolated_environment_ready():" in kernel
    assert 'MPLBACKEND="Agg"' in kernel
    assert '("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"' in kernel
    meta = _kernel_cell(nb)["metadata"]
    assert meta.get("cellView") == "form"


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def _run_kernel_cell(source: str, namespace: dict, shell: _Shell) -> str:
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    display_mod = types.ModuleType("IPython.display")
    display_mod.display = lambda *a, **k: None
    saved = {k: sys.modules.get(k) for k in ("IPython", "IPython.display")}
    sys.modules["IPython"], sys.modules["IPython.display"] = ipython, display_mod
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(source, "<kernel cell>", "exec"), namespace)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return out.getvalue()


@pytest.mark.skipif(sys.platform != "linux", reason="the worker protocol uses Linux pass_fds")
def test_swp_r_section1_reuses_environment_and_worker_when_rerun(nb: dict, tmp_path: Path, monkeypatch) -> None:
    """SWP-R: with a complete environment for this lock present, Section 1 builds nothing (no download), starts one
    worker, routes cells to it, and a second run of Section 1 keeps the same worker and its variables."""
    kernel = _src(_kernel_cell(nb))
    digest = re.search(r"LOCK_SHA256 = '([0-9a-f]{64})'", kernel).group(1)
    env = tmp_path / ("dimer_isolated_env_" + digest[:12])
    (env / "bin").mkdir(parents=True)
    os.symlink(sys.executable, env / "bin" / "python")  # stand-in for the managed CPython
    (env / ".dimer-lock-sha256").write_text(digest + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("a ready environment must not download"))
    shell = _Shell()
    namespace: dict = {"__name__": "__main__"}
    first = _run_kernel_cell(kernel, namespace, shell)
    assert "'reused': True" in first and "'worker_reused': False" in first
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert len(shell.input_transformers_cleanup) == 1
        routed = shell.input_transformers_cleanup[0](["value = 41\n"])
        assert routed == ["_DIMER_ISOLATED_RUNTIME.run('value = 41\\n')\n"]
        assert shell.input_transformers_cleanup[0](["# dimer: kernel cell\nx = 1\n"]) == ["# dimer: kernel cell\nx = 1\n"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            runtime.run("value = 41\nimport os\nprint(value + 1, os.environ.get('MPLBACKEND'), 'PYTHONSTARTUP' in os.environ)")
        assert out.getvalue().split() == ["42", "Agg", "False"]
        second = _run_kernel_cell(kernel, namespace, shell)
        assert "'worker_reused': True" in second
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and len(shell.input_transformers_cleanup) == 1
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            runtime.run("print(value)")
        assert out.getvalue().strip() == "41", "re-running Section 1 must not strand later cells"
        with pytest.raises(namespace["IsolatedCellError"]), contextlib.redirect_stderr(io.StringIO()):
            runtime.run("raise ValueError('boom')")
    finally:
        runtime.close()


# --- SWP-G: guided layer ------------------------------------------------------------------------------------------

GUIDED_MARKERS = (
    "**Who this notebook is for.**",
    "**How to use this notebook.**",
    "**Roadmap:**",
    "## Troubleshooting",
    "## Glossary",
    "## Conclusion",
)


def test_swp_g_guided_layer_present(nb: dict) -> None:
    """SWP-G: audience, how-to-use, roadmap, predictions, checkpoints, troubleshooting, glossary, conclusion."""
    md = _markdown(nb)
    missing = [m for m in GUIDED_MARKERS if m not in md]
    assert not missing, missing
    assert md.count("**Predict before running:**") >= 2
    assert md.count("<summary>Check your reasoning</summary>") >= 2


def test_swp_g_infrastructure_cells_labelled_and_collapsed(nb: dict) -> None:
    """SWP-G (GDL11): setup, carried-module and snapshot cells are labelled Infrastructure and collapsed."""
    md = _markdown(nb)
    assert md.count("> **Infrastructure.**") >= 3
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = _src(cell)
        if "# dimer: kernel cell" in s or cell["metadata"].get("dimer", {}).get("embedded_module") or "MANIFEST = {" in s:
            assert cell["metadata"].get("cellView") == "form", s[:80]


def test_swp_g_no_leftover_placeholders(nb: dict) -> None:
    """SWP-G: no literal template placeholders reach the learner."""
    learner_code = [_src(c) for c in nb["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_module")]
    text = _markdown(nb) + "\n".join(learner_code)
    for token in ("{{MODEL_ID}}", "{MODEL_ID}", "{MODEL_REVISION}", "{stem}", "@P:"):
        assert token not in text, token
    assert "{{" not in _markdown(nb)


def test_swp_r_every_code_cell_parses(nb: dict) -> None:
    """SWP-R/SWP-G: every generated code cell is valid Python (the router hands cells to the worker verbatim)."""
    for source in _code_cells(nb):
        ast.parse(source)


def test_swp_r_preinstalled_executor_runs_cells_in_its_own_kernel(nb: dict, monkeypatch) -> None:
    """SWP-R: with DIMER_NOTEBOOK_CI_PREINSTALLED=1 (an executor that already installed exactly these pins) the
    bootstrap installs nothing, downloads nothing and does not route, so every cell runs in that kernel."""
    monkeypatch.setenv("DIMER_NOTEBOOK_CI_PREINSTALLED", "1")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("no download when preinstalled"))
    shell = _Shell()
    namespace: dict = {"__name__": "__main__"}
    out = _run_kernel_cell(_src(_kernel_cell(nb)), namespace, shell)
    assert "Routing disabled" in out and not shell.input_transformers_cleanup
    assert "_DIMER_ISOLATED_RUNTIME" not in namespace


# --- SWP-B: BYOD_PATH form field (works on Kaggle/Jupyter); guarded Colab upload ----------------------------------


def _section4(nb: dict, use_byod: bool, path: str) -> str:
    source = _cell_with(nb, "def build_samples()")
    source = source.replace("USE_BYOD = False  # @param", f"USE_BYOD = {use_byod}  # @param", 1)
    return source.replace('BYOD_PATH = ""  # @param', f"BYOD_PATH = {path!r}  # @param", 1)


def _run_section4(nb: dict, use_byod: bool, path: str) -> dict:
    from moment_pipeline.csvio import read_long_csv_bytes

    ns = {"os": os, "read_long_csv_bytes": read_long_csv_bytes}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_section4(nb, use_byod, path), "<section 4>", "exec"), ns)
    return ns


def test_swp_b_byod_path_field_default_off(nb: dict) -> None:
    """SWP-B: a BYOD_PATH form field exists and BYOD stays off by default (the sample path runs)."""
    source = _cell_with(nb, "def build_samples()")
    assert 'BYOD_PATH = ""  # @param {type:"string"}' in source and "USE_BYOD = False  # @param" in source


def test_swp_b_byod_path_reads_a_csv_without_colab(nb: dict, tmp_path: Path, monkeypatch) -> None:
    """SWP-B: with USE_BYOD and BYOD_PATH the CSV is read on any runtime; a missing path is named."""
    monkeypatch.delenv("DIMER_BYOD_PATH", raising=False)
    csv = tmp_path / "mine.csv"
    csv.write_text("series_id,timestamp,channel,value\nS,2026-01-01T00:00:00,c,1.0\nS,2026-01-01T00:15:00,c,2.0\n", encoding="utf-8")
    ns = _run_section4(nb, True, str(csv))
    assert ns["sample_kind"] == "BYOD" and ns["sample_identity"]["name"] == "mine.csv"
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        _run_section4(nb, True, str(tmp_path / "missing.csv"))


def test_swp_b_empty_path_off_colab_and_cancelled_upload_are_named(nb: dict, monkeypatch) -> None:
    """SWP-B: USE_BYOD with an empty path off Colab, or a cancelled Colab upload, stops with a clear message."""
    monkeypatch.delenv("DIMER_BYOD_PATH", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    with pytest.raises(RuntimeError, match="only in Google Colab"):
        _run_section4(nb, True, "")
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: {}
    colab.files = files
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError, match="Upload exactly one CSV"):
        _run_section4(nb, True, "")
