// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

// Phase 5 Task 5.5: extern-function ABI without bare-ptr cc.
//
// Pointer args on extern declarations lower directly to `!llvm.ptr` so they
// match the plain-C ABI. Internal functions keep the memref-descriptor
// convention; call sites bridge from `memref<…>` to `!llvm.ptr` via
// `memref.extract_aligned_pointer_as_index` + `arith.index_cast` +
// `llvm.inttoptr`. Internal-function call sites are unaffected.

!s8i = !cir.int<s, 8>
!s32i = !cir.int<s, 32>
!s64i = !cir.int<s, 64>

module {
  // Extern decl with a `!cir.ptr<!s8i>` arg: lowers to `!llvm.ptr` — not
  // `memref<?xi8>` — so it matches `int assert_check(const char*)`.
  cir.func @assert_check(!cir.ptr<!s8i>) -> !s32i

  // CHECK:      func.func private @assert_check(!llvm.ptr) -> i32

  // Internal function with the usual decayed-pointer arg convention:
  // `memref<?xi8>` at the boundary, bridged to `!llvm.ptr` at the extern
  // call site.
  cir.func @run(%p: !cir.ptr<!s8i>) -> !s32i {
    %r = cir.call @assert_check(%p) : (!cir.ptr<!s8i>) -> !s32i
    cir.return %r : !s32i
  }
  // CHECK:      func.func @run(%[[P:.*]]: memref<?xi8>) -> i32 {
  // CHECK-NEXT:   %[[I:.*]] = "memref.extract_aligned_pointer_as_index"(%[[P]]) : (memref<?xi8>) -> index
  // CHECK-NEXT:   %[[I64:.*]] = arith.index_cast %[[I]] : index to i64
  // CHECK-NEXT:   %[[PTR:.*]] = llvm.inttoptr %[[I64]] : i64 to !llvm.ptr
  // CHECK-NEXT:   %[[R:.*]] = func.call @assert_check(%[[PTR]]) : (!llvm.ptr) -> i32

  // Stack array decayed to `!cir.ptr<!s8i>` and passed to the extern: the
  // `memref.cast` to dynamic shape happens, then the same bridge.
  cir.func @run_local() -> !s32i {
    %a = cir.alloca !cir.array<!s8i x 16>, !cir.ptr<!cir.array<!s8i x 16>>, ["buf"] {alignment = 1 : i64}
    %d = cir.cast array_to_ptrdecay %a : !cir.ptr<!cir.array<!s8i x 16>> -> !cir.ptr<!s8i>
    %r = cir.call @assert_check(%d) : (!cir.ptr<!s8i>) -> !s32i
    cir.return %r : !s32i
  }
  // CHECK:      func.func @run_local() -> i32 {
  // CHECK-NEXT:   %[[A:.*]] = memref.alloca() : memref<16xi8>
  // CHECK-NEXT:   %[[C:.*]] = "memref.cast"(%[[A]]) : (memref<16xi8>) -> memref<?xi8>
  // CHECK-NEXT:   %[[I2:.*]] = "memref.extract_aligned_pointer_as_index"(%[[C]]) : (memref<?xi8>) -> index
  // CHECK-NEXT:   %[[I642:.*]] = arith.index_cast %[[I2]] : index to i64
  // CHECK-NEXT:   %[[PTR2:.*]] = llvm.inttoptr %[[I642]] : i64 to !llvm.ptr
  // CHECK-NEXT:   %{{.*}} = func.call @assert_check(%[[PTR2]]) : (!llvm.ptr) -> i32

  // Internal-function call site: no bridging; memref descriptor flows
  // through unchanged.
  cir.func @internal_callee(%p: !cir.ptr<!s8i>) -> !s32i {
    %z = cir.const #cir.int<0> : !s32i
    cir.return %z : !s32i
  }

  cir.func @caller(%p: !cir.ptr<!s8i>) -> !s32i {
    %r = cir.call @internal_callee(%p) : (!cir.ptr<!s8i>) -> !s32i
    cir.return %r : !s32i
  }
  // CHECK:      func.func @caller(%[[P3:.*]]: memref<?xi8>) -> i32 {
  // CHECK-NEXT:   %[[R3:.*]] = func.call @internal_callee(%[[P3]]) : (memref<?xi8>) -> i32

  // Task F3: address-of-local-scalar passed to an internal callee whose
  // arg lowers to rank-1 `memref<?xi32>`. The call site needs a
  // `memref.cast` to lift the static rank to dynamic.
  cir.func @helper(%q: !cir.ptr<!s32i>) -> !s32i {
    %z = cir.const #cir.int<0> : !s32i
    cir.return %z : !s32i
  }

  cir.func @ranklift_caller() -> !s32i {
    %x = cir.alloca !s32i, !cir.ptr<!s32i>, ["x"] {alignment = 4 : i64}
    %r = cir.call @helper(%x) : (!cir.ptr<!s32i>) -> !s32i
    cir.return %r : !s32i
  }
  // CHECK:      func.func @ranklift_caller() -> i32 {
  // CHECK-NEXT:   %[[X:.*]] = memref.alloca() : memref<i32>
  // CHECK-NEXT:   %[[XS:.*]] = memref.reinterpret_cast %[[X]] to offset: [0], sizes: [1], strides: [1] : memref<i32> to memref<1xi32>
  // CHECK-NEXT:   %[[XD:.*]] = "memref.cast"(%[[XS]]) : (memref<1xi32>) -> memref<?xi32>
  // CHECK-NEXT:   %{{.*}} = func.call @helper(%[[XD]]) : (memref<?xi32>) -> i32

  // ---------------------------------------------------------------------
  // Variadic externs (e.g. `printf`). The `func` dialect has no variadic
  // concept, so variadic externs lower to `llvm.func` declarations and
  // call sites use `llvm.call` (with `var_callee_type` carrying the
  // variadic signature so mlir-opt can reify the call).
  // ---------------------------------------------------------------------

  cir.func private @printf(!cir.ptr<!s8i>, ...) -> !s32i

  // CHECK:      llvm.func @printf(!llvm.ptr, ...) -> i32

  // No variadic args (just the format string): the call still needs the
  // `var_callee_type` so mlir-opt accepts it.
  cir.func @run_printf_noargs(%fmt: !cir.ptr<!s8i>) -> !s32i {
    %r = cir.call @printf(%fmt) : (!cir.ptr<!s8i>) -> !s32i
    cir.return %r : !s32i
  }
  // CHECK:      func.func @run_printf_noargs(%[[F:.*]]: memref<?xi8>) -> i32 {
  // CHECK:        %[[FP:.*]] = llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
  // CHECK-NEXT:   %{{.*}} = "llvm.call"(%[[FP]]){{.*}}callee = @printf{{.*}}var_callee_type = !llvm.func<i32 (!llvm.ptr, ...)>{{.*}}: (!llvm.ptr) -> i32

  // Variadic args of mixed types: the format pointer goes through the
  // memref → `!llvm.ptr` bridge; the trailing scalar passes through.
  cir.func @run_printf_mixed(%fmt: !cir.ptr<!s8i>, %n: !s64i) -> !s32i {
    %r = cir.call @printf(%fmt, %n) : (!cir.ptr<!s8i>, !s64i) -> !s32i
    cir.return %r : !s32i
  }
  // CHECK:      func.func @run_printf_mixed(%[[F2:.*]]: memref<?xi8>, %[[N:.*]]: i64) -> i32 {
  // CHECK:        %[[FP2:.*]] = llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
  // CHECK-NEXT:   %{{.*}} = "llvm.call"(%[[FP2]], %{{.*}}){{.*}}callee = @printf{{.*}}var_callee_type = !llvm.func<i32 (!llvm.ptr, ...)>{{.*}}: (!llvm.ptr, i64) -> i32
}
