"""``xclang`` — drive C source through clang/CIR → cir-to-core → exe.

Mirror of ``ftn/tools/xftn.py``, simpler. Stages:

  clang  : `.c` → `.cir`         via `clang -fclangir -emit-cir`
  cir    : `.cir` → `_res.mlir`  via `xdsl-opt -p cir-to-core,canonicalize,cse`
  post   : `_res.mlir` → `_post.mlir`  (currently a no-op pass-through)
  mlir   : `_post.mlir` → `_res.bc`    via `mlir-opt | mlir-translate`
  exe    : `_res.bc` → `out`           via `clang` link

Output type is inferred from the suffix of `--out`:
  no suffix → executable, `.o` → object, `.bc` → bitcode, `.mlir` → MLIR text.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from enum import Enum

CLANG_DEFAULT = os.environ.get("XCLANG_CLANG", "/opt/llvm/22.1.2/bin/clang")
MLIR_OPT_DEFAULT = os.environ.get("XCLANG_MLIR_OPT", "/opt/llvm/22.1.2/bin/mlir-opt")
MLIR_TRANSLATE_DEFAULT = os.environ.get(
    "XCLANG_MLIR_TRANSLATE", "/opt/llvm/22.1.2/bin/mlir-translate"
)
XDSL_OPT_DEFAULT = os.environ.get("XCLANG_XDSL_OPT", "xdsl-opt")


class OutputType(Enum):
    EXECUTABLE = 1
    OBJECT = 2
    BITCODE = 3
    MLIR = 4


# Extern-function ABI is now handled at the cir-to-core level (Task 5.5):
# extern decls take `!llvm.ptr` directly, and call sites bridge from
# memref descriptors via `memref.extract_aligned_pointer_as_index` +
# `arith.index_cast` + `llvm.inttoptr`. Internal functions keep the full
# memref descriptor at boundaries.
MLIR_PIPELINE = (
    "builtin.module(canonicalize,cse,loop-invariant-code-motion,"
    "convert-scf-to-cf,fold-memref-alias-ops,lower-affine,"
    "convert-arith-to-llvm{index-bitwidth=64},convert-math-to-llvm,"
    "convert-func-to-llvm,"
    "finalize-memref-to-llvm,"
    "convert-cf-to-llvm{index-bitwidth=64},"
    "reconcile-unrealized-casts)"
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xclang", description="xDSL/CIR-based C compiler flow"
    )
    parser.add_argument("source", help="Filename of source C code (.c)")
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output filename (executable | .o | .bc | .mlir)",
    )
    parser.add_argument(
        "-D",
        "--define-macro",
        action="append",
        dest="defines",
        default=[],
        help="Preprocessor define (passed to clang)",
    )
    parser.add_argument(
        "-I",
        "--include-directory",
        action="append",
        dest="includes",
        default=[],
        help="Include directory (passed to clang)",
    )
    parser.add_argument(
        "--linkobj",
        action="append",
        dest="linkobjs",
        default=[],
        help="Additional object/bitcode files to link against",
    )
    parser.add_argument(
        "-e",
        "--extra-pass",
        action="append",
        dest="extra_passes",
        default=[],
        help="Additional cir-to-core stage passes after `cir-to-core`",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove temporary compilation files on successful build",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Echo final stage output to stdout",
    )
    parser.add_argument(
        "-t",
        "--tempdir",
        default="tmp",
        help="Temporary compilation directory (default: 'tmp')",
    )
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated list of stages to run (clang,cir,post,mlir,obj,exe)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity (0=quiet, 1=stage names, 2=full commands)",
    )
    parser.add_argument(
        "--clang",
        default=CLANG_DEFAULT,
        help=f"clang binary (default {CLANG_DEFAULT})",
    )
    parser.add_argument(
        "--mlir-opt",
        default=MLIR_OPT_DEFAULT,
        help=f"mlir-opt binary (default {MLIR_OPT_DEFAULT})",
    )
    parser.add_argument(
        "--mlir-translate",
        default=MLIR_TRANSLATE_DEFAULT,
        help=f"mlir-translate binary (default {MLIR_TRANSLATE_DEFAULT})",
    )
    parser.add_argument(
        "--xdsl-opt",
        default=XDSL_OPT_DEFAULT,
        help=f"xdsl-opt binary (default {XDSL_OPT_DEFAULT})",
    )
    return parser


def _output_type_from_name(name: str) -> OutputType:
    if "." not in os.path.basename(name):
        return OutputType.EXECUTABLE
    if name.endswith(".o"):
        return OutputType.OBJECT
    if name.endswith(".bc"):
        return OutputType.BITCODE
    if name.endswith(".mlir"):
        return OutputType.MLIR
    return OutputType.EXECUTABLE


def _enabled_stages(out_type: OutputType, override: str | None) -> dict[str, bool]:
    """Decide which stages to run based on the output type, then apply
    `--stages` override if provided."""
    stages = {
        "clang": True,
        "cir": True,
        "post": out_type
        in (OutputType.EXECUTABLE, OutputType.OBJECT, OutputType.BITCODE),
        "mlir": out_type
        in (OutputType.EXECUTABLE, OutputType.OBJECT, OutputType.BITCODE),
        "obj": out_type == OutputType.OBJECT,
        "exe": out_type == OutputType.EXECUTABLE,
    }
    if override is not None:
        wanted = set(override.split(","))
        unknown = wanted - stages.keys()
        if unknown:
            print(f"xclang: unknown stage(s) {sorted(unknown)}", file=sys.stderr)
            sys.exit(2)
        stages = {k: (k in wanted) for k in stages}
    return stages


def _run(cmd: list[str], *, verbose: int, stage: str) -> None:
    if verbose >= 1:
        print(f"  [{stage}] {shlex.join(cmd)}", file=sys.stderr)
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        print(f"xclang: stage '{stage}' failed (exit {rc})", file=sys.stderr)
        sys.exit(rc)


def _shell(cmd: str, *, verbose: int, stage: str) -> None:
    if verbose >= 1:
        print(f"  [{stage}] {cmd}", file=sys.stderr)
    rc = subprocess.run(cmd, shell=True, check=False).returncode
    if rc != 0:
        print(f"xclang: stage '{stage}' failed (exit {rc})", file=sys.stderr)
        sys.exit(rc)


def main() -> int:
    args = _build_arg_parser().parse_args()
    src = args.source
    if not src.endswith(".c"):
        print(f"xclang: source file must end in '.c', got {src!r}", file=sys.stderr)
        return 2

    stem = os.path.splitext(os.path.basename(src))[0]
    out = args.out if args.out is not None else stem
    out_type = _output_type_from_name(out)
    stages = _enabled_stages(out_type, args.stages)

    os.makedirs(args.tempdir, exist_ok=True)

    cir_path = os.path.join(args.tempdir, f"{stem}.cir")
    res_path = os.path.join(args.tempdir, f"{stem}_res.mlir")
    post_path = os.path.join(args.tempdir, f"{stem}_post.mlir")
    bc_path = os.path.join(args.tempdir, f"{stem}_res.bc")

    define_args = [f"-D{m}" for m in args.defines]
    include_args = [f"-I{i}" for i in args.includes]

    if stages["clang"]:
        _run(
            [
                args.clang,
                "-fclangir",
                "-emit-cir",
                "-O0",
                "-Xclang",
                "-no-enable-noundef-analysis",
                *define_args,
                *include_args,
                src,
                "-o",
                cir_path,
            ],
            verbose=args.verbose,
            stage="clang",
        )

    if stages["cir"]:
        passes = "cir-to-core,canonicalize,cse"
        for ep in args.extra_passes:
            passes += "," + ep
        cir_target = out if out_type == OutputType.MLIR else res_path
        _run(
            [args.xdsl_opt, "-p", passes, cir_path, "-o", cir_target],
            verbose=args.verbose,
            stage="cir",
        )

    if stages["post"]:
        # Currently a no-op fixup; kept as a seam.
        _shell(
            f"cp {shlex.quote(res_path)} {shlex.quote(post_path)}",
            verbose=args.verbose,
            stage="post",
        )

    if stages["mlir"]:
        bc_target = out if out_type == OutputType.BITCODE else bc_path
        mlir_opt_q = shlex.quote(args.mlir_opt)
        pipeline_q = shlex.quote(MLIR_PIPELINE)
        post_q = shlex.quote(post_path)
        mlir_translate_q = shlex.quote(args.mlir_translate)
        bc_target_q = shlex.quote(bc_target)
        _shell(
            f"{mlir_opt_q} --pass-pipeline={pipeline_q} {post_q} | "
            f"{mlir_translate_q} --mlir-to-llvmir -o {bc_target_q}",
            verbose=args.verbose,
            stage="mlir",
        )

    if stages["obj"]:
        _run(
            [args.clang, "-O3", "-c", bc_path, "-o", out],
            verbose=args.verbose,
            stage="obj",
        )

    if stages["exe"]:
        _run(
            [args.clang, "-O3", bc_path, *args.linkobjs, "-lm", "-o", out],
            verbose=args.verbose,
            stage="exe",
        )

    if args.stdout and out_type == OutputType.MLIR:
        with open(out) as f:
            sys.stdout.write(f.read())

    if args.cleanup:
        for f in (cir_path, res_path, post_path, bc_path):
            try:
                os.remove(f)
            except OSError:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
