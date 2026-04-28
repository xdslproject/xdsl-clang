// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

!s32i = !cir.int<s, 32>
!f32  = !cir.float

module {
  // `float z[10] = {1, 2, …, 10};` — clang emits a `cir.const
  // #cir.const_array<…>` followed by `cir.store %c, %slot`. We hoist the
  // initialiser to a private `memref.global` and lower the store to a
  // `memref.copy` (Task 5.2).
  cir.func @const_array_init() {
    %0 = cir.alloca !cir.array<!f32 x 10>, !cir.ptr<!cir.array<!f32 x 10>>, ["z", init] {alignment = 16 : i64}
    %1 = cir.const #cir.const_array<[#cir.fp<1.000000e+00> : !f32, #cir.fp<2.000000e+00> : !f32, #cir.fp<3.000000e+00> : !f32, #cir.fp<4.000000e+00> : !f32, #cir.fp<5.000000e+00> : !f32, #cir.fp<6.000000e+00> : !f32, #cir.fp<7.000000e+00> : !f32, #cir.fp<8.000000e+00> : !f32, #cir.fp<9.000000e+00> : !f32, #cir.fp<1.000000e+01> : !f32]> : !cir.array<!f32 x 10>
    cir.store align(16) %1, %0 : !cir.array<!f32 x 10>, !cir.ptr<!cir.array<!f32 x 10>>
    cir.return
  }

  // `int v[4] = {0};` — clang emits `cir.const #cir.zero` of array type.
  // Same hoisting strategy, but the global gets a zero-splat initialiser.
  cir.func @zero_array_init() {
    %0 = cir.alloca !cir.array<!s32i x 4>, !cir.ptr<!cir.array<!s32i x 4>>, ["v", init] {alignment = 16 : i64}
    %1 = cir.const #cir.zero : !cir.array<!s32i x 4>
    cir.store align(16) %1, %0 : !cir.array<!s32i x 4>, !cir.ptr<!cir.array<!s32i x 4>>
    cir.return
  }

  // The two hoisted private globals appear in module prelude order:
  // CHECK:      "memref.global"() <{sym_name = "_xclang_lit{{[0-9]+}}", type = memref<10xf32>, initial_value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]> : tensor<10xf32>, sym_visibility = "private", constant}> : () -> ()
  // CHECK-NEXT: "memref.global"() <{sym_name = "_xclang_lit{{[0-9]+}}", type = memref<4xi32>, initial_value = dense<0> : tensor<4xi32>, sym_visibility = "private", constant}> : () -> ()

  // CHECK:      func.func @const_array_init() {
  // CHECK-NEXT:   %[[SLOT:.*]] = memref.alloca() : memref<10xf32>
  // CHECK-NEXT:   %[[G:.*]] = memref.get_global @_xclang_lit{{[0-9]+}} : memref<10xf32>
  // CHECK-NEXT:   "memref.copy"(%[[G]], %[[SLOT]]) : (memref<10xf32>, memref<10xf32>) -> ()
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

  // CHECK:      func.func @zero_array_init() {
  // CHECK-NEXT:   %[[SLOT2:.*]] = memref.alloca() : memref<4xi32>
  // CHECK-NEXT:   %[[G2:.*]] = memref.get_global @_xclang_lit{{[0-9]+}} : memref<4xi32>
  // CHECK-NEXT:   "memref.copy"(%[[G2]], %[[SLOT2]]) : (memref<4xi32>, memref<4xi32>) -> ()
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }
}
