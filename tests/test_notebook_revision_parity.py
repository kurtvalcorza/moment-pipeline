"""Review M3: each notebook's `repository_revision` names a commit whose modules ARE the carried
modules.

The notebook records `metadata.dimer.generated_from.revision` and the per-module SHA-256 it carries.
Where that commit
is available locally (a full clone), the module text at that commit must equal the repository's
module text; a
shallow CI checkout that lacks the object skips the git half, and the digest half always runs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "tutorials").glob("moment_*_colab.ipynb"))


def _git_show(revision: str, rel: str) -> bytes | None:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{revision}:{rel}"], check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_repository_revision_carries_the_same_modules(path: Path) -> None:
    meta = json.loads(path.read_text(encoding="utf-8"))["metadata"]["dimer"]["generated_from"]
    revision = meta["revision"]
    assert len(revision) == 40, revision
    rels = meta["modules"]
    live = {rel: (ROOT / rel).read_bytes() for rel in rels}
    digest = hashlib.sha256(b"".join(live[r] for r in rels)).hexdigest()
    assert digest == meta["module_sha256"], (
        "carried modules differ from the repository's modules: regenerate"
    )
    shown = {rel: _git_show(revision, rel) for rel in rels}
    if any(v is None for v in shown.values()):
        pytest.skip(f"revision {revision[:12]} not available in this checkout (shallow clone)")
    for rel in rels:
        assert shown[rel].replace(b"\r\n", b"\n") == live[rel].replace(b"\r\n", b"\n"), (
            f"{path.name}: {rel} at repository_revision {revision[:12]} is not the carried module"
        )
