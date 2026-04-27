// RUN: CIR_ROUNDTRIP

!s32i = !cir.int<s, 32>

module {
  cir.func @shouldParseCopyOp(%arg0 : !cir.ptr<!s32i>, %arg1 : !cir.ptr<!s32i>) {
    cir.copy %arg0 to %arg1 : !cir.ptr<!s32i>
    cir.return
  }
}

// CHECK:      cir.func @shouldParseCopyOp(%arg0: !cir.ptr<!cir.int<s, 32>>, %arg1: !cir.ptr<!cir.int<s, 32>>) {
// CHECK-NEXT:   cir.copy %arg0 to %arg1 : !cir.ptr<!cir.int<s, 32>>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
