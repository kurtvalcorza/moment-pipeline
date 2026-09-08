"""Contracts about the repository's own artifacts: the CI gate and the lock files.

These are unit tests on purpose. The lock-parity gate is the first step of the always-on
CI job, so a defect there is invisible to every other test in the suite until a runner
executes it -- which, for a branch that has never been pushed, is never.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
LOCK_TXT = REPO_ROOT / "requirements.lock.txt"
README = REPO_ROOT / "README.md"

EXPORT_FLAGS = ("--format requirements-txt", "--locked", "--no-dev", "--no-emit-project")


def _lock_parity_step() -> str:
    """The `run:` body of the CI lock-parity step."""
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"- name: Lock parity.*?(?=\n      - name: )", text, re.S)
    assert match, "the CI workflow no longer has a 'Lock parity' step"
    return match.group(0)


# --- R-1: the always-on CI job must not fail on the export header -------------------


def test_committed_lock_carries_no_uv_export_header():
    """`uv export` embeds its own `-o <path>` in the header comment.

    CI regenerates into `/tmp/requirements.check.txt`, so a committed file that carries a
    header can never `diff` clean against the CI copy -- the always-on unit job dies at
    step 2, before ruff or pytest run. The committed artifact must therefore be
    header-free (`uv export --no-header`).
    """
    first_lines = LOCK_TXT.read_text(encoding="utf-8").splitlines()[:5]
    assert first_lines, "requirements.lock.txt is empty"
    assert not first_lines[0].startswith("#"), (
        f"requirements.lock.txt starts with a generated header: {first_lines[0]!r}"
    )
    for line in first_lines:
        assert "uv export" not in line, f"lock file embeds its generating command: {line!r}"
        assert " -o " not in line, f"lock file embeds an output path: {line!r}"


def test_ci_lock_parity_step_exports_without_a_header():
    step = _lock_parity_step()
    for flag in EXPORT_FLAGS:
        assert flag in step, f"CI export step lost {flag!r}"
    assert "--no-header" in step, (
        "the CI lock-parity export must pass --no-header, otherwise the regenerated file "
        "carries `-o /tmp/requirements.check.txt` in its header and the diff always fails"
    )
    assert "diff -u requirements.lock.txt" in step


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")
def test_ci_lock_parity_command_reproduces_the_committed_file(tmp_path):
    """Run the literal CI export and diff, offline, and require exit 0."""
    target = tmp_path / "requirements.check.txt"
    export = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            "uv",
            "export",
            "--format",
            "requirements-txt",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--no-header",
            "-o",
            str(target),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert export.returncode == 0, export.stderr
    committed = LOCK_TXT.read_text(encoding="utf-8")
    regenerated = target.read_text(encoding="utf-8")
    assert regenerated == committed, "regenerated export differs from requirements.lock.txt"


# --- R-7: the lock is a uv artifact, not a pip install path -------------------------


def _unhashed_direct_requirements() -> list[str]:
    lines = LOCK_TXT.read_text(encoding="utf-8").splitlines()
    found = []
    for index, line in enumerate(lines):
        if " @ " not in line or line.startswith((" ", "#")):
            continue
        continuation = lines[index + 1] if index + 1 < len(lines) else ""
        if "--hash=" not in line and "--hash=" not in continuation:
            found.append(line.strip())
    return found


def test_readme_documents_that_the_lock_is_not_pip_installable():
    """pip refuses an unhashed requirement once any requirement carries a hash.

    `requirements.lock.txt` holds a hashed graph *plus* a bare `momentfm @ git+...` and a
    `torch==...+cpu` that lives only on the PyTorch CPU index, so `pip install -r` cannot
    work. If that is still true, the README must say so and name `uv sync --locked` as
    the install path -- the RFC claims CI and Colab install from the same lock.
    """
    unhashed = _unhashed_direct_requirements()
    assert unhashed, (
        "no unhashed direct requirement remains -- if the lock is now pip-installable, "
        "prove it with an install test and drop this contract"
    )
    readme = README.read_text(encoding="utf-8")
    assert "uv sync --locked" in readme
    assert "not pip-installable" in readme, (
        "README must state that requirements.lock.txt is a uv artifact and not a pip "
        f"install path; unhashed direct requirements present: {unhashed}"
    )
