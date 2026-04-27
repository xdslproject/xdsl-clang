// RUN: XDSL_ROUNDTRIP

!s32i = !cir.int<s, 32>

module {
  cir.func @c0(%arg0: !s32i, %arg1: !s32i) {
    %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
    %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init] {alignment = 4 : i64}
    %2 = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init] {alignment = 1 : i64}
    cir.store %arg0, %0 : !s32i, !cir.ptr<!s32i>
    cir.store %arg1, %1 : !s32i, !cir.ptr<!s32i>
    %3 = cir.load %0 : !cir.ptr<!s32i>, !s32i
    %4 = cir.load %1 : !cir.ptr<!s32i>, !s32i
    %5 = cir.cmp(gt, %3, %4) : !s32i, !cir.bool
    cir.store %5, %2 : !cir.bool, !cir.ptr<!cir.bool>
    %6 = cir.load %0 : !cir.ptr<!s32i>, !s32i
    %7 = cir.load %1 : !cir.ptr<!s32i>, !s32i
    %8 = cir.cmp(lt, %6, %7) : !s32i, !cir.bool
    cir.store %8, %2 : !cir.bool, !cir.ptr<!cir.bool>
    %9 = cir.load %0 : !cir.ptr<!s32i>, !s32i
    %10 = cir.load %1 : !cir.ptr<!s32i>, !s32i
    %11 = cir.cmp(le, %9, %10) : !s32i, !cir.bool
    cir.store %11, %2 : !cir.bool, !cir.ptr<!cir.bool>
    %12 = cir.load %0 : !cir.ptr<!s32i>, !s32i
    %13 = cir.load %1 : !cir.ptr<!s32i>, !s32i
    %14 = cir.cmp(ge, %12, %13) : !s32i, !cir.bool
    cir.store %14, %2 : !cir.bool, !cir.ptr<!cir.bool>
    %15 = cir.load %0 : !cir.ptr<!s32i>, !s32i
    %16 = cir.load %1 : !cir.ptr<!s32i>, !s32i
    %17 = cir.cmp(ne, %15, %16) : !s32i, !cir.bool
    cir.store %17, %2 : !cir.bool, !cir.ptr<!cir.bool>
    %18 = cir.load %0 : !cir.ptr<!s32i>, !s32i
    %19 = cir.load %1 : !cir.ptr<!s32i>, !s32i
    %20 = cir.cmp(eq, %18, %19) : !s32i, !cir.bool
    cir.store %20, %2 : !cir.bool, !cir.ptr<!cir.bool>
    cir.return
  }
}

// CHECK:      cir.func @c0(%arg0: !cir.int<s, 32>, %arg1: !cir.int<s, 32>) {
// CHECK-NEXT:   %0 = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["b", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init] {alignment = 1 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>
// CHECK-NEXT:   cir.store %arg1, %1 : !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>
// CHECK-NEXT:   %3 = cir.load %0 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %4 = cir.load %1 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %5 = cir.cmp(gt, %3, %4) : !cir.int<s, 32>, !cir.bool
// CHECK-NEXT:   cir.store %5, %2 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   %6 = cir.load %0 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %7 = cir.load %1 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %8 = cir.cmp(lt, %6, %7) : !cir.int<s, 32>, !cir.bool
// CHECK-NEXT:   cir.store %8, %2 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   %9 = cir.load %0 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %10 = cir.load %1 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %11 = cir.cmp(le, %9, %10) : !cir.int<s, 32>, !cir.bool
// CHECK-NEXT:   cir.store %11, %2 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   %12 = cir.load %0 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %13 = cir.load %1 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %14 = cir.cmp(ge, %12, %13) : !cir.int<s, 32>, !cir.bool
// CHECK-NEXT:   cir.store %14, %2 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   %15 = cir.load %0 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %16 = cir.load %1 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %17 = cir.cmp(ne, %15, %16) : !cir.int<s, 32>, !cir.bool
// CHECK-NEXT:   cir.store %17, %2 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   %18 = cir.load %0 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %19 = cir.load %1 : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK-NEXT:   %20 = cir.cmp(eq, %18, %19) : !cir.int<s, 32>, !cir.bool
// CHECK-NEXT:   cir.store %20, %2 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

// -----

!u32i = !cir.int<u, 32>

module {
  cir.func @c0_unsigned(%arg0: !u32i, %arg1: !u32i) {
    %0 = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init] {alignment = 4 : i64}
    %1 = cir.alloca !u32i, !cir.ptr<!u32i>, ["b", init] {alignment = 4 : i64}
    %2 = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init] {alignment = 1 : i64}
    cir.store %arg0, %0 : !u32i, !cir.ptr<!u32i>
    cir.store %arg1, %1 : !u32i, !cir.ptr<!u32i>
    %3 = cir.load %0 : !cir.ptr<!u32i>, !u32i
    %4 = cir.load %1 : !cir.ptr<!u32i>, !u32i
    %5 = cir.cmp(gt, %3, %4) : !u32i, !cir.bool
    cir.store %5, %2 : !cir.bool, !cir.ptr<!cir.bool>
    %6 = cir.load %0 : !cir.ptr<!u32i>, !u32i
    %7 = cir.load %1 : !cir.ptr<!u32i>, !u32i
    %8 = cir.cmp(lt, %6, %7) : !u32i, !cir.bool
    cir.store %8, %2 : !cir.bool, !cir.ptr<!cir.bool>
    %9 = cir.load %0 : !cir.ptr<!u32i>, !u32i
    %10 = cir.load %1 : !cir.ptr<!u32i>, !u32i
    %11 = cir.cmp(eq, %9, %10) : !u32i, !cir.bool
    cir.store %11, %2 : !cir.bool, !cir.ptr<!cir.bool>
    cir.return
  }
}

// CHECK:      cir.func @c0_unsigned(%arg0: !cir.int<u, 32>, %arg1: !cir.int<u, 32>) {
// CHECK-NEXT:   %0 = cir.alloca !cir.int<u, 32>, !cir.ptr<!cir.int<u, 32>>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.int<u, 32>, !cir.ptr<!cir.int<u, 32>>, ["b", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init] {alignment = 1 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.int<u, 32>, !cir.ptr<!cir.int<u, 32>>
// CHECK-NEXT:   cir.store %arg1, %1 : !cir.int<u, 32>, !cir.ptr<!cir.int<u, 32>>
// CHECK-NEXT:   %3 = cir.load %0 : !cir.ptr<!cir.int<u, 32>>, !cir.int<u, 32>
// CHECK-NEXT:   %4 = cir.load %1 : !cir.ptr<!cir.int<u, 32>>, !cir.int<u, 32>
// CHECK-NEXT:   %5 = cir.cmp(gt, %3, %4) : !cir.int<u, 32>, !cir.bool
// CHECK-NEXT:   cir.store %5, %2 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   %6 = cir.load %0 : !cir.ptr<!cir.int<u, 32>>, !cir.int<u, 32>
// CHECK-NEXT:   %7 = cir.load %1 : !cir.ptr<!cir.int<u, 32>>, !cir.int<u, 32>
// CHECK-NEXT:   %8 = cir.cmp(lt, %6, %7) : !cir.int<u, 32>, !cir.bool
// CHECK-NEXT:   cir.store %8, %2 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   %9 = cir.load %0 : !cir.ptr<!cir.int<u, 32>>, !cir.int<u, 32>
// CHECK-NEXT:   %10 = cir.load %1 : !cir.ptr<!cir.int<u, 32>>, !cir.int<u, 32>
// CHECK-NEXT:   %11 = cir.cmp(eq, %9, %10) : !cir.int<u, 32>, !cir.bool
// CHECK-NEXT:   cir.store %11, %2 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

// -----

module {
  cir.func @c0_float(%arg0: !cir.float, %arg1: !cir.float) {
    %0 = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init] {alignment = 4 : i64}
    %1 = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init] {alignment = 4 : i64}
    %2 = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init] {alignment = 1 : i64}
    cir.store %arg0, %0 : !cir.float, !cir.ptr<!cir.float>
    cir.store %arg1, %1 : !cir.float, !cir.ptr<!cir.float>
    %3 = cir.load %0 : !cir.ptr<!cir.float>, !cir.float
    %4 = cir.load %1 : !cir.ptr<!cir.float>, !cir.float
    %5 = cir.cmp(gt, %3, %4) : !cir.float, !cir.bool
    cir.store %5, %2 : !cir.bool, !cir.ptr<!cir.bool>
    %6 = cir.load %0 : !cir.ptr<!cir.float>, !cir.float
    %7 = cir.load %1 : !cir.ptr<!cir.float>, !cir.float
    %8 = cir.cmp(eq, %6, %7) : !cir.float, !cir.bool
    cir.store %8, %2 : !cir.bool, !cir.ptr<!cir.bool>
    cir.return
  }
}

// CHECK:      cir.func @c0_float(%arg0: !cir.float, %arg1: !cir.float) {
// CHECK-NEXT:   %0 = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init] {alignment = 4 : i64}
// CHECK-NEXT:   %2 = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init] {alignment = 1 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT:   cir.store %arg1, %1 : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT:   %3 = cir.load %0 : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT:   %4 = cir.load %1 : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT:   %5 = cir.cmp(gt, %3, %4) : !cir.float, !cir.bool
// CHECK-NEXT:   cir.store %5, %2 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   %6 = cir.load %0 : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT:   %7 = cir.load %1 : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT:   %8 = cir.cmp(eq, %6, %7) : !cir.float, !cir.bool
// CHECK-NEXT:   cir.store %8, %2 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

// -----

!s32i = !cir.int<s, 32>

module {
  cir.func @pointer_cmp(%arg0: !cir.ptr<!s32i>, %arg1: !cir.ptr<!s32i>) {
    %0 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["a", init] {alignment = 8 : i64}
    %1 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["b", init] {alignment = 8 : i64}
    %2 = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init] {alignment = 1 : i64}
    cir.store %arg0, %0 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
    cir.store %arg1, %1 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
    %3 = cir.load %0 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
    %4 = cir.load %1 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
    %5 = cir.cmp(eq, %3, %4) : !cir.ptr<!s32i>, !cir.bool
    cir.store %5, %2 : !cir.bool, !cir.ptr<!cir.bool>
    cir.return
  }
}

// CHECK:      cir.func @pointer_cmp(%arg0: !cir.ptr<!cir.int<s, 32>>, %arg1: !cir.ptr<!cir.int<s, 32>>) {
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!cir.int<s, 32>>, !cir.ptr<!cir.ptr<!cir.int<s, 32>>>, ["a", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.ptr<!cir.int<s, 32>>, !cir.ptr<!cir.ptr<!cir.int<s, 32>>>, ["b", init] {alignment = 8 : i64}
// CHECK-NEXT:   %2 = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init] {alignment = 1 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!cir.int<s, 32>>, !cir.ptr<!cir.ptr<!cir.int<s, 32>>>
// CHECK-NEXT:   cir.store %arg1, %1 : !cir.ptr<!cir.int<s, 32>>, !cir.ptr<!cir.ptr<!cir.int<s, 32>>>
// CHECK-NEXT:   %3 = cir.load %0 : !cir.ptr<!cir.ptr<!cir.int<s, 32>>>, !cir.ptr<!cir.int<s, 32>>
// CHECK-NEXT:   %4 = cir.load %1 : !cir.ptr<!cir.ptr<!cir.int<s, 32>>>, !cir.ptr<!cir.int<s, 32>>
// CHECK-NEXT:   %5 = cir.cmp(eq, %3, %4) : !cir.ptr<!cir.int<s, 32>>, !cir.bool
// CHECK-NEXT:   cir.store %5, %2 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
