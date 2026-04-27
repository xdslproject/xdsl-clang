// RUN: XDSL_ROUNDTRIP

!s32i = !cir.int<s, 32>

module {
  cir.func @select_int_valid(%arg0 : !cir.bool, %arg1 : !cir.vector<8 x !s32i>, %arg2 : !cir.vector<8 x !s32i>) -> !cir.vector<8 x !s32i> {
    %0 = cir.select if %arg0 then %arg1 else %arg2 : (!cir.bool, !cir.vector<8 x !s32i>, !cir.vector<8 x !s32i>) -> !cir.vector<8 x !s32i>
    cir.return %0 : !cir.vector<8 x !s32i>
  }
}

// CHECK:      cir.func @select_int_valid(%arg0: !cir.bool, %arg1: !cir.vector<8 x !cir.int<s, 32>>, %arg2: !cir.vector<8 x !cir.int<s, 32>>) -> !cir.vector<8 x !cir.int<s, 32>> {
// CHECK-NEXT:   %0 = cir.select if %arg0 then %arg1 else %arg2 : (!cir.bool, !cir.vector<8 x !cir.int<s, 32>>, !cir.vector<8 x !cir.int<s, 32>>) -> !cir.vector<8 x !cir.int<s, 32>>
// CHECK-NEXT:   cir.return %0 : !cir.vector<8 x !cir.int<s, 32>>
// CHECK-NEXT: }
