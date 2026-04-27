// RUN: CIR_ROUNDTRIP

!s32i = !cir.int<s, 32>

module {
  cir.func @yolo(%arg0 : !s32i) {
    %a = cir.cast int_to_bool %arg0 : !s32i -> !cir.bool
    %0 = cir.const #cir.int<0> : !s32i
    cir.return
  }
}

// CHECK:      cir.func @yolo(%arg0: !cir.int<s, 32>) {
// CHECK-NEXT:   %{{.*}} = cir.cast int_to_bool %arg0 : !cir.int<s, 32> -> !cir.bool
// CHECK-NEXT:   %{{.*}} = cir.const #cir.int<0> : !cir.int<s, 32>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

// -----

!s32i = !cir.int<s, 32>

module {
  cir.func @bitcast(%p: !cir.ptr<!s32i>) {
    %0 = cir.cast bitcast %p : !cir.ptr<!s32i> -> !cir.ptr<f32>
    cir.return
  }
}

// CHECK:      cir.func @bitcast(%[[P:.*]]: !cir.ptr<!cir.int<s, 32>>) {
// CHECK-NEXT:   %{{.*}} = cir.cast bitcast %[[P]] : !cir.ptr<!cir.int<s, 32>> -> !cir.ptr<f32>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
