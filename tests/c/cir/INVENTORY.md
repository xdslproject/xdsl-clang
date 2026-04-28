# CIR Surface Inventory

Auto-generated from the `.cir` baselines under `tests/c/cir/`.
This is the **finite** set of CIR constructs the `cir-to-core` pass must
handle to lower the existing C corpus (excluding `swm_openmp.c`).

## Ops in corpus

| Op                | Phase | Notes                                          |
| ----------------- | ----- | ---------------------------------------------- |
| `cir.alloca`      | 2a    | Static + dynamic                               |
| `cir.binop`       | 2c    | Sign-aware op selection                        |
| `cir.br`          | 2d    |                                                |
| `cir.break`       | 2d    | Triggers unstructured loop lowering            |
| `cir.call`        | 2e    |                                                |
| `cir.cast`        | 2b    |                                                |
| `cir.cmp`         | 2c    | Sign-aware predicate                           |
| `cir.condition`   | 2d    | `scf.while` before-region terminator           |
| `cir.const`       | 2c    | Including `#cir.const_array`, `#cir.zero`      |
| `cir.continue`    | 2d    | Triggers unstructured loop lowering            |
| `cir.for`         | 2d    |                                                |
| `cir.func`        | 2e    |                                                |
| `cir.get_element` | 2b    | Array element access                           |
| `cir.get_global`  | 2a    |                                                |
| `cir.get_member`  | 2b    | Record field access                            |
| `cir.global`      | 2a    |                                                |
| `cir.if`          | 2d    |                                                |
| `cir.load`        | 2b    |                                                |
| `cir.module_asm`  | —     | Out of scope, skipped silently                 |
| `cir.ptr_stride`  | 2b    |                                                |
| `cir.return`      | 2d/2e |                                                |
| `cir.scope`       | 2d    | Inline; introduces alloca scope                |
| `cir.select`      | 2c    |                                                |
| `cir.store`       | 2b    |                                                |
| `cir.ternary`     | 2c    |                                                |
| `cir.unary`       | 2c    |                                                |
| `cir.while`       | 2d    |                                                |
| `cir.yield`       | 2d    | Region terminator                              |

## Ops *not* present in corpus (handled by best-effort fallback)

`cir.do`, `cir.brcond`, `cir.switch`, `cir.switch.flat`, `cir.unreachable`,
`cir.trap`, `cir.copy`, `cir.complex.*`, `cir.vec.*`, `cir.va_*`. These are
deferred until a corpus test actually needs them; the dispatcher should
raise on unknown ops so we notice.

## Types

`!cir.array`, `!cir.bool`, `!cir.double`, `!cir.float`, `!cir.int`,
`!cir.ptr`, `!cir.record`, `!cir.void`. No `f16` / `bf16` / `f80` /
`f128` / `long_double` / `complex` / `vector` types appear.

## Attribute markers

`#cir.bool`, `#cir.const_array`, `#cir.fp`, `#cir.int`, `#cir.lang`,
`#cir.ptr`, `#cir.zero`.
