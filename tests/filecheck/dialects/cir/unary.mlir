// RUN: XDSL_ROUNDTRIP

!u32i = !cir.int<u, 32>

module {
  cir.func @test_unary_unsigned() {
    %0 = cir.alloca !u32i, !cir.ptr<!u32i>, ["a"] {alignment = 4 : i64}
    %1 = cir.load %0 : !cir.ptr<!u32i>, !u32i
    %2 = cir.unary(plus, %1) : !u32i, !u32i
    %3 = cir.unary(minus, %1) : !u32i, !u32i
    %4 = cir.unary(not, %1) : !u32i, !u32i
    %5 = cir.unary(inc, %1) : !u32i, !u32i
    %6 = cir.unary(dec, %1) : !u32i, !u32i
    cir.return
  }
}

// CHECK:      cir.func @test_unary_unsigned() {
// CHECK-NEXT:   %0 = cir.alloca !cir.int<u, 32>, !cir.ptr<!cir.int<u, 32>>, ["a"] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.load %0 : !cir.ptr<!cir.int<u, 32>>, !cir.int<u, 32>
// CHECK-NEXT:   %2 = cir.unary(plus, %1) : !cir.int<u, 32>, !cir.int<u, 32>
// CHECK-NEXT:   %3 = cir.unary(minus, %1) : !cir.int<u, 32>, !cir.int<u, 32>
// CHECK-NEXT:   %4 = cir.unary(not, %1) : !cir.int<u, 32>, !cir.int<u, 32>
// CHECK-NEXT:   %5 = cir.unary(inc, %1) : !cir.int<u, 32>, !cir.int<u, 32>
// CHECK-NEXT:   %6 = cir.unary(dec, %1) : !cir.int<u, 32>, !cir.int<u, 32>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

// -----

!s32i = !cir.int<s, 32>

module {
  cir.func @test_unary_signed() {
    %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"] {alignment = 4 : i64}
    %1 = cir.load %0 : !cir.ptr<!s32i>, !s32i
    %2 = cir.unary(plus, %1) : !s32i, !s32i
    %3 = cir.unary(minus, %1) nsw : !s32i, !s32i
    %4 = cir.unary(not, %1) : !s32i, !s32i
    %5 = cir.unary(inc, %1) nsw : !s32i, !s32i
    %6 = cir.unary(dec, %1) nsw : !s32i, !s32i
    cir.return
  }
}

// CHECK:      cir.func @test_unary_signed() {
// CHECK-NEXT:   %0 = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["a"] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.load %0 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %2 = cir.unary(plus, %1) : !cir.int<s, 32>, !cir.int<s, 32>
// CHECK-NEXT:   %3 = cir.unary(minus, %1) nsw : !cir.int<s, 32>, !cir.int<s, 32>
// CHECK-NEXT:   %4 = cir.unary(not, %1) : !cir.int<s, 32>, !cir.int<s, 32>
// CHECK-NEXT:   %5 = cir.unary(inc, %1) nsw : !cir.int<s, 32>, !cir.int<s, 32>
// CHECK-NEXT:   %6 = cir.unary(dec, %1) nsw : !cir.int<s, 32>, !cir.int<s, 32>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
