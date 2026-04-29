"""End-to-end harness for `xclang`.

For each `.c` file under ``tests/c/{fragments,solvers,advection,swm}``
(excluding ``swm_openmp.c``) try driving it through xclang to an executable
and run it. The current pass coverage doesn't yet handle the full corpus
(see Phase 5 in ``cir-to-core-plan.md``); each test is therefore marked
``xfail(strict=False)`` until Phase 5 hardens the missing pieces.

The point of the harness even at this stage is:
  1. The fixture surfaces every C test as a separate, parametrised case.
  2. As Phase 5 lands features, tests flip to ``XPASS`` automatically and
     the strict=False allows them to be promoted to ``passed`` without
     failing the suite.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

C_ROOT = Path(__file__).parent
REPO_ROOT = C_ROOT.parent.parent
ASSERTION_O = C_ROOT / "build" / "assertion.o"
XCLANG = shutil.which("xclang")


CORPUS_DIRS = ["fragments", "solvers", "advection", "swm"]
SKIP_FILES = {"swm_openmp.c"}

# Per-test runtime budgets. The default 60s is fine for fragments and the
# small solvers; the heavy CPU-bound benchmarks (full-size iterative kernels)
# need longer. Times empirically measured on a desktop x86_64 box; CI may
# need to adjust if its hardware is materially slower.
DEFAULT_TIMEOUT = 60
RUN_TIMEOUTS = {
    "solvers/jacobi.c": 300,
    "swm/swm.c": 120,
    "swm/swm_orig.c": 120,
}

# Phase 5 hardening turns these into xpass — list of test ids that we
# expect to fail (strict=False so xpass doesn't error).
EXPECTED_FAIL = {
    # `tra_adv.c` builds and runs but the 1024×512×512 NEMO advection
    # kernel allocates ~25 GB of memref descriptors and exceeds any
    # reasonable per-test runtime budget. The companion `tra_adv_small.c`
    # exercises the same code path with a reduced grid and runs in the
    # default 60s budget.
    "advection/tra_adv.c",
}


def _gather_corpus() -> list[Path]:
    out: list[Path] = []
    for sub in CORPUS_DIRS:
        for f in sorted((C_ROOT / sub).glob("*.c")):
            if f.name in SKIP_FILES:
                continue
            out.append(f)
    return out


@pytest.mark.parametrize(
    "src", _gather_corpus(), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_xclang_e2e(tmp_path: Path, src: Path):
    if XCLANG is None:
        pytest.skip("xclang not on PATH")
    if not ASSERTION_O.exists():
        pytest.skip("assertion.o missing — run `make -C tests/c/build baseline` first")

    rel = f"{src.parent.name}/{src.name}"
    if rel in EXPECTED_FAIL:
        pytest.xfail(f"{rel}: Phase 5 hardening pending")

    out = tmp_path / src.stem
    workdir = tmp_path / "tmp"
    workdir.mkdir()

    rc = subprocess.run(
        [
            XCLANG,
            str(src),
            "-o",
            str(out),
            "--tempdir",
            str(workdir),
            "--linkobj",
            str(ASSERTION_O),
            "-v",
            "0",
        ],
        check=False,
    ).returncode
    assert rc == 0, f"xclang failed (rc={rc}) on {src}"
    assert out.exists(), f"executable not produced: {out}"

    # Quick tests use the [PASS] harness; numerical kernels just exit 0.
    runres = subprocess.run(
        [str(out)],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUTS.get(rel, DEFAULT_TIMEOUT),
    )
    assert runres.returncode == 0, f"{src} exited {runres.returncode}\n{runres.stdout}"
    if src.parent.name == "fragments":
        assert runres.stdout.startswith("[PASS]"), (
            f"{src} did not print [PASS]:\n{runres.stdout}"
        )
