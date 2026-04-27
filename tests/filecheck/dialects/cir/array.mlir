// RUN: XDSL_ROUNDTRIP

!s32i = !cir.int<s, 32>

module {
  cir.global external @a = #cir.zero : !cir.array<!s32i x 10>
  cir.global external @aa = #cir.zero : !cir.array<!cir.array<!s32i x 10> x 10>
  cir.global external @b = #cir.zero : !cir.array<!s32i x 10>
  cir.global external @bb = #cir.zero : !cir.array<!cir.array<!s32i x 10> x 10>
  cir.global external @c = #cir.zero : !cir.array<!s32i x 10>
  cir.global external @d = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3>
  cir.global external @dd = #cir.const_array<[#cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<5> : !s32i, #cir.int<6> : !s32i]> : !cir.array<!s32i x 2>]> : !cir.array<!cir.array<!s32i x 2> x 3>
  cir.global external @e = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i], trailing_zeros> : !cir.array<!s32i x 10>
}

// CHECK:      cir.global external @a = #cir.zero : !cir.array<!cir.int<s, 32> x 10>
// CHECK-NEXT: cir.global external @aa = #cir.zero : !cir.array<!cir.array<!cir.int<s, 32> x 10> x 10>
// CHECK-NEXT: cir.global external @b = #cir.zero : !cir.array<!cir.int<s, 32> x 10>
// CHECK-NEXT: cir.global external @bb = #cir.zero : !cir.array<!cir.array<!cir.int<s, 32> x 10> x 10>
// CHECK-NEXT: cir.global external @c = #cir.zero : !cir.array<!cir.int<s, 32> x 10>
// CHECK-NEXT: cir.global external @d = #cir.const_array<[#cir.int<1> : !cir.int<s, 32>, #cir.int<2> : !cir.int<s, 32>, #cir.int<3> : !cir.int<s, 32>]> : !cir.array<!cir.int<s, 32> x 3>
// CHECK-NEXT: cir.global external @dd = #cir.const_array<[#cir.const_array<[#cir.int<1> : !cir.int<s, 32>, #cir.int<2> : !cir.int<s, 32>]> : !cir.array<!cir.int<s, 32> x 2>, #cir.const_array<[#cir.int<3> : !cir.int<s, 32>, #cir.int<4> : !cir.int<s, 32>]> : !cir.array<!cir.int<s, 32> x 2>, #cir.const_array<[#cir.int<5> : !cir.int<s, 32>, #cir.int<6> : !cir.int<s, 32>]> : !cir.array<!cir.int<s, 32> x 2>]> : !cir.array<!cir.array<!cir.int<s, 32> x 2> x 3>
// CHECK-NEXT: cir.global external @e = #cir.const_array<[#cir.int<1> : !cir.int<s, 32>, #cir.int<2> : !cir.int<s, 32>], trailing_zeros> : !cir.array<!cir.int<s, 32> x 10>

// -----

!s32i = !cir.int<s, 32>

module {
  cir.func @func() {
    %0 = cir.alloca !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>, ["l"] {alignment = 4 : i64}
    cir.return
  }
}

// CHECK:      cir.func @func() {
// CHECK-NEXT:   %0 = cir.alloca !cir.array<!cir.int<s, 32> x 10>, !cir.ptr<!cir.array<!cir.int<s, 32> x 10>>, ["l"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

// -----

!s32i = !cir.int<s, 32>

module {
  cir.func @func2(%arg0: !cir.ptr<!s32i>) {
    %0 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
    cir.return
  }
}

// CHECK:      cir.func @func2(%arg0: !cir.ptr<!cir.int<s, 32>>) {
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!cir.int<s, 32>>, !cir.ptr<!cir.ptr<!cir.int<s, 32>>>, ["p", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!cir.int<s, 32>>, !cir.ptr<!cir.ptr<!cir.int<s, 32>>>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

// -----

!s32i = !cir.int<s, 32>

module {
  cir.func @func3(%arg0: !cir.ptr<!cir.array<!s32i x 10>>) {
    %0 = cir.alloca !cir.ptr<!cir.array<!s32i x 10>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 10>>>, ["pp", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!cir.array<!s32i x 10>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 10>>>
    cir.return
  }
}

// CHECK:      cir.func @func3(%arg0: !cir.ptr<!cir.array<!cir.int<s, 32> x 10>>) {
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!cir.array<!cir.int<s, 32> x 10>>, !cir.ptr<!cir.ptr<!cir.array<!cir.int<s, 32> x 10>>>, ["pp", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!cir.array<!cir.int<s, 32> x 10>>, !cir.ptr<!cir.ptr<!cir.array<!cir.int<s, 32> x 10>>>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
