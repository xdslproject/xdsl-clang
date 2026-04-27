// RUN: XDSL_ROUNDTRIP

!s32i = !cir.int<s, 32>
!s8i = !cir.int<s, 8>
#true = #cir.bool<true> : !cir.bool

module {
  cir.func @binary_assign() {
    %0 = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b"] {alignment = 1 : i64}
    %1 = cir.alloca !s8i, !cir.ptr<!s8i>, ["c"] {alignment = 1 : i64}
    %2 = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f"] {alignment = 4 : i64}
    %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["i"] {alignment = 4 : i64}
    %4 = cir.const #true
    cir.store %4, %0 : !cir.bool, !cir.ptr<!cir.bool>
    %5 = cir.const #cir.int<65> : !s32i
    %6 = cir.cast integral %5 : !s32i -> !s8i
    cir.store %6, %1 : !s8i, !cir.ptr<!s8i>
    %7 = cir.const #cir.fp<3.140000e+00> : !cir.float
    cir.store %7, %2 : !cir.float, !cir.ptr<!cir.float>
    %8 = cir.const #cir.int<42> : !s32i
    cir.store %8, %3 : !s32i, !cir.ptr<!s32i>
    cir.return
  }
}

// CHECK:      cir.func @binary_assign() {
// CHECK-NEXT:   %0 = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b"] {alignment = 1 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.int<s, 8>, !cir.ptr<!cir.int<s, 8>>, ["c"] {alignment = 1 : i64}
// CHECK-NEXT:   %2 = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f"] {alignment = 4 : i64}
// CHECK-NEXT:   %3 = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["i"] {alignment = 4 : i64}
// CHECK-NEXT:   %4 = cir.const #cir.bool<true> : !cir.bool
// CHECK-NEXT:   cir.store %4, %0 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   %5 = cir.const #cir.int<65> : !cir.int<s, 32>
// CHECK-NEXT:   %6 = cir.cast integral %5 : !cir.int<s, 32> -> !cir.int<s, 8>
// CHECK-NEXT:   cir.store %6, %1 : !cir.int<s, 8>, !cir.ptr<!cir.int<s, 8>>
// CHECK-NEXT:   %7 = cir.const #cir.fp<3.140000e+00> : !cir.float
// CHECK-NEXT:   cir.store %7, %2 : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT:   %8 = cir.const #cir.int<42> : !cir.int<s, 32>
// CHECK-NEXT:   cir.store %8, %3 : !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
