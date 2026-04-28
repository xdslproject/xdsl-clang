// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

// Phase 1 deliverable: the cir-to-core pass loads and runs against a
// CIR module. With no functions, no globals, and no body ops, the pass
// emits an empty `builtin.module` and strips the CIR-specific module
// attributes (`cir.lang`, `cir.triple`).

!s32i = !cir.int<s, 32>

module attributes {cir.triple = "x86_64-unknown-linux-gnu", cir.lang = #cir.lang<c>} {
}

// CHECK:      builtin.module {
// CHECK-NEXT: }
