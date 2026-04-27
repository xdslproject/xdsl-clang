// RUN: XDSL_ROUNDTRIP

!s32i = !cir.int<s, 32>

module {
  cir.func private @f1()

  cir.func @f2() {
    cir.call @f1() : () -> ()
    cir.call @f1() side_effect(pure) : () -> ()
    cir.call @f1() side_effect(const) : () -> ()
    cir.return
  }
}

// CHECK:      cir.func private @f1()
// CHECK:      cir.func @f2() {
// CHECK-NEXT:   cir.call @f1() : () -> ()
// CHECK-NEXT:   cir.call @f1() side_effect(pure) : () -> ()
// CHECK-NEXT:   cir.call @f1() side_effect(const) : () -> ()
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

// -----

!s32i = !cir.int<s, 32>

module {
  cir.func private @f3() -> !s32i

  cir.func @f4() -> !s32i {
    %0 = cir.call @f3() : () -> !s32i
    cir.return %0 : !s32i
  }
}

// CHECK:      cir.func private @f3() -> !cir.int<s, 32>
// CHECK:      cir.func @f4() -> !cir.int<s, 32> {
// CHECK-NEXT:   %0 = cir.call @f3() : () -> !cir.int<s, 32>
// CHECK-NEXT:   cir.return %0 : !cir.int<s, 32>
// CHECK-NEXT: }

// -----

!s32i = !cir.int<s, 32>

module {
  cir.func private @f5(!s32i, !s32i) -> !s32i

  cir.func @f6() -> !s32i {
    %0 = cir.const #cir.int<1> : !s32i
    %1 = cir.const #cir.int<2> : !s32i
    %2 = cir.call @f5(%0, %1) : (!s32i, !s32i) -> !s32i
    cir.return %2 : !s32i
  }
}

// CHECK:      cir.func private @f5(!cir.int<s, 32>, !cir.int<s, 32>) -> !cir.int<s, 32>
// CHECK:      cir.func @f6() -> !cir.int<s, 32> {
// CHECK-NEXT:   %0 = cir.const #cir.int<1> : !cir.int<s, 32>
// CHECK-NEXT:   %1 = cir.const #cir.int<2> : !cir.int<s, 32>
// CHECK-NEXT:   %2 = cir.call @f5(%0, %1) : (!cir.int<s, 32>, !cir.int<s, 32>) -> !cir.int<s, 32>
// CHECK-NEXT:   cir.return %2 : !cir.int<s, 32>
// CHECK-NEXT: }

// -----

!s32i = !cir.int<s, 32>

module {
  cir.func @f7(%arg0: !cir.ptr<!cir.func<(!s32i, !s32i) -> !s32i>>) -> !s32i {
    %0 = cir.const #cir.int<1> : !s32i
    %1 = cir.const #cir.int<2> : !s32i
    %2 = cir.call %arg0(%0, %1) : (!cir.ptr<!cir.func<(!s32i, !s32i) -> !s32i>>, !s32i, !s32i) -> !s32i
    cir.return %2 : !s32i
  }
}

// CHECK:      cir.func @f7(%arg0: !cir.ptr<!cir.func<(!cir.int<s, 32>, !cir.int<s, 32>) -> !cir.int<s, 32>>>) -> !cir.int<s, 32> {
// CHECK-NEXT:   %0 = cir.const #cir.int<1> : !cir.int<s, 32>
// CHECK-NEXT:   %1 = cir.const #cir.int<2> : !cir.int<s, 32>
// CHECK-NEXT:   %2 = cir.call %arg0(%0, %1) : (!cir.ptr<!cir.func<(!cir.int<s, 32>, !cir.int<s, 32>) -> !cir.int<s, 32>>>, !cir.int<s, 32>, !cir.int<s, 32>) -> !cir.int<s, 32>
// CHECK-NEXT:   cir.return %2 : !cir.int<s, 32>
// CHECK-NEXT: }
