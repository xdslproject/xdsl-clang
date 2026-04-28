// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

!s32i = !cir.int<s, 32>
!rec_S = !cir.record<struct "S" {!s32i}>

module {
  // `int *p = NULL;` — pointer-to-scalar lowers to a memref under
  // Decision 1. The null is materialised as `llvm.mlir.zero : !llvm.ptr`
  // and bridged to the memref slot via `unrealized_conversion_cast`.
  // Dereferencing the resulting memref is UB (matches C NULL-deref
  // semantics); we just need the SSA value to typecheck against the
  // slot.
  cir.func @null_int_ptr() {
    %0 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p", init] {alignment = 8 : i64}
    %1 = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
    cir.store %1, %0 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
    cir.return
  }
  // CHECK:      func.func @null_int_ptr() {
  // CHECK-NEXT:   %{{.*}} = memref.alloca() : memref<memref<?xi32>>
  // CHECK-NEXT:   %[[Z:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %[[Z]] : !llvm.ptr to memref<?xi32>
  // CHECK-NEXT:   memref.store {{.*}} : memref<memref<?xi32>>
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

  // `struct S *q = NULL;` — pointer-to-record lowers directly to
  // `!llvm.ptr` per Decision 1, so the null becomes `llvm.mlir.zero`
  // with no cast required.
  cir.func @null_record_ptr() {
    %0 = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["q", init] {alignment = 8 : i64}
    %1 = cir.const #cir.ptr<null> : !cir.ptr<!rec_S>
    cir.store %1, %0 : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
    cir.return
  }
  // CHECK:      func.func @null_record_ptr() {
  // CHECK-NEXT:   %{{.*}} = memref.alloca() : memref<!llvm.ptr>
  // CHECK-NEXT:   %{{.*}} = llvm.mlir.zero : !llvm.ptr
  // CHECK-NEXT:   memref.store {{.*}} : memref<!llvm.ptr>
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }
}
