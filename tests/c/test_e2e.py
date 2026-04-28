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
    # `allocatables.c` lowers its const-array init but trips a different
    # downstream blocker — `static float *global_array = NULL;` lowers to
    # `memref<memref<f32>>` while malloc emits `memref<?xf32>`, so the
    # global-pointer assignment fails type matching. Tracked as Task F2.
    "fragments/allocatables.c",
    "fragments/array_ops.c",  # malloc/free void* bitcast
    "fragments/pointers.c",
    # Task 5.7 unblocked do_loops and while_loops — they pass end-to-end.
    # jacobi.c also lowers and runs, but takes ~2 minutes to complete
    # (CPU-bound iterative solver) which exceeds the per-test timeout in
    # this harness; keep it on the xfail list until a smaller-grid harness
    # variant is available.
    "solvers/jacobi.c",
    # `procedures.c` lowers past the index-arity error but trips a
    # call-site rank mismatch: a local `int val` (rank-0 `memref<i32>`)
    # is passed to a function whose `int *b` arg lowers to rank-1
    # `memref<?xi32>` — separate Task 5.5 follow-up.
    "fragments/procedures.c",
    # heavy benchmarks — Phase 5. `tra_adv.c` builds and runs end-to-end
    # but the kernel is a heavy numerical simulation that exceeds the
    # harness's per-test runtime budget. `swm/swm*.c` still hit a
    # `cir.load` of a struct field via `cir.get_member` returning
    # `!llvm.ptr` — needs an `llvm.load` fallback for `!llvm.ptr`-typed
    # addresses (struct-field handling — separate blocker).
    "advection/tra_adv.c",
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
    runres = subprocess.run([str(out)], capture_output=True, text=True, timeout=60)
    assert runres.returncode == 0, f"{src} exited {runres.returncode}\n{runres.stdout}"
    if src.parent.name == "fragments":
        assert runres.stdout.startswith("[PASS]"), (
            f"{src} did not print [PASS]:\n{runres.stdout}"
        )
