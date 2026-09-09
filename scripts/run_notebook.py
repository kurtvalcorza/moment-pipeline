"""Execute notebook code cells in order without requiring Jupyter.

This is intentionally small: the live tutorials contain ordinary Python cells and no
IPython magics. CI uses this runner so the exact code a Colab user sees is executed
against the repository checkout and pinned model runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def execute_notebook(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    for index, cell in enumerate(payload.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        try:
            exec(compile(source, f"{path}:cell-{index}", "exec"), namespace, namespace)
        except Exception as exc:
            raise RuntimeError(f"notebook cell {index} failed in {path}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+", type=Path)
    args = parser.parse_args()
    for notebook in args.notebooks:
        print(f"==> executing {notebook}")
        execute_notebook(notebook)
        print(f"==> passed {notebook}")


if __name__ == "__main__":
    main()
