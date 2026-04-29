# xdsl-clang: C Compilation via xDSL and ClangIR

`xdsl-clang` provides support for compiling C through [xDSL](https://github.com/xdslproject/xdsl)
using [ClangIR (CIR)](https://llvm.github.io/clangir/) as the front-end IR. It
ships the `xclang` driver, an xDSL `cir` dialect, and a `cir-to-core` lowering
pipeline that bridges CIR to the standard MLIR dialects (`func`, `scf`, `cf`,
`memref`, `arith`, `math`) so the result can be fed through the LLVM backend
to produce executables.

In short, `xdsl-clang` makes it possible to:

- Take a `.c` source file, lower it to xDSL's `cir` dialect via clang
- Apply Python-native xDSL passes to transform the IR
- Lower to standard MLIR dialects and on to LLVM IR / a native binary
- Inspect every intermediate stage as plain text for debugging

## Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [The `xclang` driver](#the-xclang-driver)
- [Examples](#examples)
- [Discussion](#discussion)

## Installation

To set up a development environment:

```bash
make install
```

This runs `uv sync --group dev --group docs` and installs pre-commit hooks.

`xdsl-clang` is validated against a specific LLVM/MLIR version (22.1.2). The
`xclang` driver expects the following binaries from that toolchain to be
available; by default it looks under `/opt/llvm/22.1.2/bin/`:

- `clang` — built with ClangIR support (`-fclangir`)
- `mlir-opt`
- `mlir-translate`

These can be overridden via environment variables (`XCLANG_CLANG`,
`XCLANG_MLIR_OPT`, `XCLANG_MLIR_TRANSLATE`, `XCLANG_XDSL_OPT`) or per-invocation
flags (`--clang`, `--mlir-opt`, `--mlir-translate`, `--xdsl-opt`).

## Getting Started

Once installed, the `xclang` script is on your `$PATH`:

```bash
uv run xclang path/to/source.c
```

The Makefile exposes the common test targets:

```bash
make tests          # pytest + filecheck
make tests-c        # full C end-to-end suite
make filecheck      # lit/filecheck only
```

## The `xclang` driver

`xclang` walks a `.c` source file through five stages, writing each
intermediate artefact into a temporary directory (`tmp/` by default):

| Stage   | Tool                                | Input        | Output           |
| ------- | ----------------------------------- | ------------ | ---------------- |
| `clang` | `clang -fclangir -emit-cir`         | `foo.c`      | `foo.cir`        |
| `cir`   | `xdsl-opt -p cir-to-core,...`       | `foo.cir`    | `foo_res.mlir`   |
| `post`  | (no-op fixup seam)                  | `foo_res.mlir` | `foo_post.mlir` |
| `mlir`  | `mlir-opt | mlir-translate`         | `foo_post.mlir` | `foo_res.bc`  |
| `exe`   | `clang -O3 ... -lm`                 | `foo_res.bc` | `foo` (binary)   |

The output type is inferred from the suffix passed to `-o`:

- *no suffix* → executable
- `.o` → relocatable object
- `.bc` → LLVM bitcode
- `.mlir` → MLIR text (stops after the `cir` stage)

## Examples

All commands assume you are at the repository root.

### 1. Compile a C file to an executable

```bash
uv run xclang tests/c/fragments/conditionals.c -o conditionals
./conditionals
```

This runs the full pipeline (`clang → cir → post → mlir → exe`) and leaves
the binary in the working directory.

### 2. Stop at MLIR text and inspect the lowered IR

```bash
uv run xclang tests/c/fragments/do_loops.c -o do_loops.mlir --stdout
```

`-o do_loops.mlir` makes `xclang` stop after the `cir` stage and emit MLIR
text. `--stdout` echoes the result to stdout so you can pipe it into
`less`, `diff`, etc.

### 3. Emit an LLVM bitcode file

```bash
uv run xclang tests/c/fragments/arrays.c -o arrays.bc
/opt/llvm/22.1.2/bin/llvm-dis arrays.bc -o -
```

### 4. Run only a subset of the pipeline

The `--stages` flag takes a comma-separated list of `clang,cir,post,mlir,obj,exe`:

```bash
# only run the front-end + cir-to-core, write tmp/conditionals_res.mlir
uv run xclang tests/c/fragments/conditionals.c --stages clang,cir
```

### 5. Add an extra xDSL pass after `cir-to-core`

```bash
uv run xclang tests/c/fragments/while_loops.c \
    -e canonicalize -e cse -o while_loops.mlir
```

`-e/--extra-pass` can be given multiple times; passes run in order *after*
the default `cir-to-core,canonicalize,cse` chain.

### 6. Compile with preprocessor defines and include paths

```bash
uv run xclang tests/c/solvers/jacobi.c \
    -D N=128 -I tests/c/util -o jacobi
```

### 7. Link against external object/bitcode files

```bash
uv run xclang main.c --linkobj kernels.o --linkobj rt.bc -o app
```

### 8. Verbose diagnostics and cleanup

```bash
uv run xclang tests/c/fragments/switch.c -o switch -v 2 --cleanup
```

`-v 2` prints every shell command that gets executed; `--cleanup` removes
the intermediate files in `tmp/` after a successful build.

## Project structure

```text
.
├── docs/
├── src/
│   └── xdsl_clang/
│       ├── dialects/cir.py         # xDSL ClangIR dialect
│       ├── tools/xclang.py         # `xclang` driver entry point
│       └── transforms/cir_to_core/ # CIR → standard-dialect lowering
└── tests/
    ├── c/                          # C end-to-end fixtures + driver tests
    └── filecheck/                  # MLIR filecheck tests
```

## Discussion

This project follows the wider [xDSL](https://github.com/xdslproject/xdsl)
community. Discussion happens on the
[xDSL Zulip chat room](https://xdsl.zulipchat.com).
