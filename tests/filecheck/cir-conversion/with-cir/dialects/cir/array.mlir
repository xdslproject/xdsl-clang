// RUN: CIR_ROUNDTRIP

!s32i = !cir.int<s, 32>

module {
  cir.func @func() {
    %0 = cir.alloca !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>, ["l"] {alignment = 4 : i64}
    cir.return
  }
}

// CHECK:      cir.func @func() {
// CHECK-NEXT:   %{{.*}} = cir.alloca !cir.array<!cir.int<s, 32> x 10>, !cir.ptr<!cir.array<!cir.int<s, 32> x 10>>, ["l"] {alignment = 4 : i64}
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

// CHECK:      cir.func @func2(%[[ARG0:.*]]: !cir.ptr<!cir.int<s, 32>>) {
// CHECK-NEXT:   %[[P:.*]] = cir.alloca !cir.ptr<!cir.int<s, 32>>, !cir.ptr<!cir.ptr<!cir.int<s, 32>>>, ["p", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %[[ARG0]], %[[P]] : !cir.ptr<!cir.int<s, 32>>, !cir.ptr<!cir.ptr<!cir.int<s, 32>>>
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

// CHECK:      cir.func @func3(%[[ARG0:.*]]: !cir.ptr<!cir.array<!cir.int<s, 32> x 10>>) {
// CHECK-NEXT:   %[[PP:.*]] = cir.alloca !cir.ptr<!cir.array<!cir.int<s, 32> x 10>>, !cir.ptr<!cir.ptr<!cir.array<!cir.int<s, 32> x 10>>>, ["pp", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %[[ARG0]], %[[PP]] : !cir.ptr<!cir.array<!cir.int<s, 32> x 10>>, !cir.ptr<!cir.ptr<!cir.array<!cir.int<s, 32> x 10>>>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

// -----

// Constant array attributes (including the `trailing_zeros` modifier) used
// inline as `cir.const` initializers, since `cir.global` requires explicit
// linkage when round-tripping through cir-opt.

!s32i = !cir.int<s, 32>

module {
  cir.func @use_const_array() {
    %0 = cir.const #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3>
    %1 = cir.const #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i], trailing_zeros> : !cir.array<!s32i x 10>
    cir.return
  }
}

// CHECK:      cir.func @use_const_array() {
// CHECK-NEXT:   %{{.*}} = cir.const #cir.const_array<[#cir.int<1> : !cir.int<s, 32>, #cir.int<2> : !cir.int<s, 32>, #cir.int<3> : !cir.int<s, 32>]> : !cir.array<!cir.int<s, 32> x 3>
// CHECK-NEXT:   %{{.*}} = cir.const #cir.const_array<[#cir.int<1> : !cir.int<s, 32>, #cir.int<2> : !cir.int<s, 32>], trailing_zeros> : !cir.array<!cir.int<s, 32> x 10>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
