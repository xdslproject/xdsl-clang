// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

!s32i = !cir.int<s, 32>
!f32  = !cir.float

module {
  cir.global external @gi = #cir.int<42> : !s32i
  // CHECK:      "memref.global"() <{sym_name = "gi", type = memref<i32>, initial_value = dense<42> : tensor<i32>, sym_visibility = "public"}>

  cir.global external @gf = #cir.fp<1.5> : !f32
  // CHECK:      "memref.global"() <{sym_name = "gf", type = memref<f32>, initial_value = dense<1.500000e+00> : tensor<f32>, sym_visibility = "public"}>

  cir.global external @ga = #cir.zero : !cir.array<!s32i x 4>
  // CHECK:      "memref.global"() <{sym_name = "ga", type = memref<4xi32>, initial_value, sym_visibility = "public"}>

  cir.func @stack_scalar() -> !s32i {
    %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["x"] {alignment = 4 : i64}
    %1 = cir.const #cir.int<7> : !s32i
    cir.return %1 : !s32i
  }
  // CHECK:      func.func @stack_scalar() -> i32 {
  // CHECK-NEXT:   %{{.*}} = memref.alloca() : memref<i32>

  cir.func @stack_array() -> !s32i {
    %0 = cir.alloca !cir.array<!s32i x 16>, !cir.ptr<!cir.array<!s32i x 16>>, ["a"] {alignment = 16 : i64}
    %1 = cir.const #cir.int<0> : !s32i
    cir.return %1 : !s32i
  }
  // CHECK:      func.func @stack_array() -> i32 {
  // CHECK-NEXT:   %{{.*}} = memref.alloca() : memref<16xi32>

  cir.func @load_global() -> !s32i {
    %g = cir.get_global @gi : !cir.ptr<!s32i>
    %1 = cir.const #cir.int<3> : !s32i
    cir.return %1 : !s32i
  }
  // CHECK:      func.func @load_global() -> i32 {
  // CHECK-NEXT:   %{{.*}} = memref.get_global @gi : memref<i32>
}
