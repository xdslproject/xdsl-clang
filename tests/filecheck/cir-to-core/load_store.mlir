// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

!s32i = !cir.int<s, 32>
!s64i = !cir.int<s, 64>
!f64  = !cir.double

module {
  cir.func @scalar_rw() -> !s32i {
    %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["x"] {alignment = 4 : i64}
    %c = cir.const #cir.int<7> : !s32i
    cir.store %c, %0 : !s32i, !cir.ptr<!s32i>
    %1 = cir.load %0 : !cir.ptr<!s32i>, !s32i
    cir.return %1 : !s32i
  }
  // CHECK:      func.func @scalar_rw() -> i32 {
  // CHECK-NEXT:   %[[A:.*]] = memref.alloca() : memref<i32>
  // CHECK-NEXT:   %[[C:.*]] = arith.constant 7 : i32
  // CHECK-NEXT:   memref.store %[[C]], %[[A]][] : memref<i32>
  // CHECK-NEXT:   %{{.*}} = memref.load %[[A]][] : memref<i32>

  cir.func @array_index(%idx: !s32i) -> !f64 {
    %a = cir.alloca !cir.array<!f64 x 100>, !cir.ptr<!cir.array<!f64 x 100>>, ["a"] {alignment = 8 : i64}
    %d = cir.cast array_to_ptrdecay %a : !cir.ptr<!cir.array<!f64 x 100>> -> !cir.ptr<!f64>
    %p = cir.ptr_stride %d, %idx : (!cir.ptr<!f64>, !s32i) -> !cir.ptr<!f64>
    %v = cir.load %p : !cir.ptr<!f64>, !f64
    cir.return %v : !f64
  }
  // CHECK:      func.func @array_index(%[[I:.*]]: i32) -> f64 {
  // CHECK-NEXT:   %[[A:.*]] = memref.alloca() : memref<100xf64>
  // CHECK-NEXT:   %[[D:.*]] = "memref.cast"(%[[A]]) : (memref<100xf64>) -> memref<?xf64>
  // CHECK-NEXT:   %[[II:.*]] = arith.index_cast %[[I]] : i32 to index
  // CHECK-NEXT:   %{{.*}} = memref.load %[[D]][%[[II]]] : memref<?xf64>

  cir.func @cast_widen(%a: !s32i) -> !s64i {
    %0 = cir.cast integral %a : !s32i -> !s64i
    cir.return %0 : !s64i
  }
  // CHECK:      func.func @cast_widen(%[[X:.*]]: i32) -> i64 {
  // CHECK-NEXT:   %{{.*}} = arith.extsi %[[X]] : i32 to i64

  cir.func @cast_int2float(%a: !s32i) -> !f64 {
    %0 = cir.cast int_to_float %a : !s32i -> !f64
    cir.return %0 : !f64
  }
  // CHECK:      func.func @cast_int2float(%[[X:.*]]: i32) -> f64 {
  // CHECK-NEXT:   %{{.*}} = arith.sitofp %[[X]] : i32 to f64
}
