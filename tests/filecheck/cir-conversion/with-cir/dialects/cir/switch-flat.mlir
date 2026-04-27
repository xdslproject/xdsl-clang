// RUN: CIR_ROUNDTRIP

!s32i = !cir.int<s, 32>

cir.func @FlatSwitchWithoutDefault(%arg0: !s32i) {
  cir.switch.flat %arg0 : !s32i, ^bb2 [
    1: ^bb1
  ]
  ^bb1:
    cir.br ^bb2
  ^bb2:
    cir.return
}

// CHECK:      cir.switch.flat %{{.*}} : !cir.int<s, 32>, ^{{.*}} [
// CHECK-NEXT:   1: ^{{.*}}
// CHECK-NEXT: ]
// CHECK:      cir.br ^{{.*}}
// CHECK:      cir.return

// -----

!s32i = !cir.int<s, 32>

cir.func @FlatSwitchWithDefault(%arg0: !s32i) {
  cir.switch.flat %arg0 : !s32i, ^bb2 [
    1: ^bb1
  ]
  ^bb1:
    cir.br ^bb3
  ^bb2:
    cir.br ^bb3
  ^bb3:
    cir.return
}

// CHECK:      cir.switch.flat %{{.*}} : !cir.int<s, 32>, ^{{.*}} [
// CHECK-NEXT:   1: ^{{.*}}
// CHECK-NEXT: ]

// -----

!s32i = !cir.int<s, 32>

cir.func @switchWithOperands(%arg0: !s32i, %arg1: !s32i, %arg2: !s32i) {
  cir.switch.flat %arg0 : !s32i, ^bb3 [
    0: ^bb1(%arg1, %arg2 : !s32i, !s32i),
    1: ^bb2(%arg2, %arg1 : !s32i, !s32i)
  ]
^bb1:
  cir.br ^bb3

^bb2:
  cir.br ^bb3

^bb3:
  cir.return
}

// CHECK:      cir.switch.flat %{{.*}} : !cir.int<s, 32>, ^{{.*}} [
// CHECK-NEXT:   0: ^{{.*}}(%{{.*}}, %{{.*}} : !cir.int<s, 32>, !cir.int<s, 32>),
// CHECK-NEXT:   1: ^{{.*}}(%{{.*}}, %{{.*}} : !cir.int<s, 32>, !cir.int<s, 32>)
// CHECK-NEXT: ]
