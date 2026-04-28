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

# Phase 5 hardening turns these into xpass — list of test ids that we
# expect to fail (strict=False so xpass doesn't error).
EXPECTED_FAIL = {
    # Function-local !cir.const #cir.const_array<...> + #cir.zero of array type
    "fragments/arrays.c",
    "fragments/array_ops.c",  # malloc/free void* bitcast
    "fragments/intrinsics.c",
    "fragments/allocatables.c",
    "fragments/pointers.c",
    # break/continue
    "fragments/do_loops.c",
    "fragments/while_loops.c",
    # cir.br (block-graph control flow from C)
    "fragments/procedures.c",
    "fragments/conditionals.c",  # cir.return inside cir.if
    # heavy benchmarks — Phase 5
    "solvers/jacobi.c",
    "solvers/gauss_seidel_stack.c",
    "solvers/gauss_seidel_heap.c",
    "advection/tra_adv.c",
    "advection/pwadvection.c",
    "swm/swm.c",
    "swm/swm_orig.c",
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
        pytest.skip(
            f"assertion.o missing — run `make -C tests/c/build baseline` first"
        )

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
            "-o", str(out),
            "--tempdir", str(workdir),
            "--linkobj", str(ASSERTION_O),
            "-v", "0",
        ],
        check=False,
    ).returncode
    assert rc == 0, f"xclang failed (rc={rc}) on {src}"
    assert out.exists(), f"executable not produced: {out}"

    # Quick tests use the [PASS] harness; numerical kernels just exit 0.
    runres = subprocess.run([str(out)], capture_output=True, text=True, timeout=60)
    assert runres.returncode == 0, f"{src} exited {runres.returncode}\n{runres.stdout}"
    if src.parent.name == "fragments":
        assert runres.stdout.startswith("[PASS]"), (
            f"{src} did not print [PASS]:\n{runres.stdout}"
        )
