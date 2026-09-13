from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
EXPECTED = {
    "moment_embeddings_colab.ipynb",
    "moment_imputation_colab.ipynb",
    "moment_anomaly_detection_colab.ipynb",
}


def test_exact_v1_tutorial_set_exists_and_parses() -> None:
    actual = {path.name for path in TUTORIALS.glob("*.ipynb")}
    assert EXPECTED <= actual
    for name in EXPECTED:
        payload = json.loads((TUTORIALS / name).read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert any(cell.get("cell_type") == "code" for cell in payload["cells"])


def test_every_notebook_code_cell_compiles_as_plain_python() -> None:
    for name in sorted(EXPECTED):
        payload = json.loads((TUTORIALS / name).read_text(encoding="utf-8"))
        for index, cell in enumerate(payload["cells"], start=1):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            assert not source.lstrip().startswith(("%", "!")), (
                f"{name} cell {index} uses notebook-only magic; executable CI uses plain Python"
            )
            compile(source, f"{name}:cell-{index}", "exec")


def test_tutorial_bootstraps_reference_the_lock_and_pinned_public_api() -> None:
    combined = "\n".join(
        (TUTORIALS / name).read_text(encoding="utf-8") for name in sorted(EXPECTED)
    )
    # Standalone carrier (NOTEBOOK_SPEC 2.0 §4): inline PINS replace the lock bootstrap and the
    # loader reads the manifest-described weights directory; the device follows the runtime.
    assert "PINS = [" in combined
    assert "momentfm @ git+https://github.com/moment-timeseries-foundation-model/" in combined
    assert "load_moment" in combined
    assert "weights_dir=WEIGHTS_DIR" in combined
    assert "threshold" in (TUTORIALS / "moment_anomaly_detection_colab.ipynb").read_text(
        encoding="utf-8"
    ).lower()
