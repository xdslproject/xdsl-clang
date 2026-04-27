// RUN: XDSL_ROUNDTRIP

!u64i = !cir.int<u, 64>
!u8i = !cir.int<u, 8>
!void = !cir.void

module {
  cir.func dso_local @_Z11test_allocam(%arg0: !u64i) -> !cir.ptr<!void> {
    %0 = cir.alloca !u64i, !cir.ptr<!u64i>, ["n", init] {alignment = 8 : i64}
    %1 = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["__retval"] {alignment = 8 : i64}
    cir.store %arg0, %0 : !u64i, !cir.ptr<!u64i>
    %2 = cir.load align(8) %0 : !cir.ptr<!u64i>, !u64i
    // Dynamically sized alloca
    %3 = cir.alloca !u8i, !cir.ptr<!u8i>, %2 : !u64i, ["bi_alloca"] {alignment = 16 : i64}
    %4 = cir.cast bitcast %3 : !cir.ptr<!u8i> -> !cir.ptr<!void>
    cir.store %4, %1 : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
    %5 = cir.load %1 : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
    cir.return %5 : !cir.ptr<!void>
  }
}

// CHECK:      cir.func dso_local @_Z11test_allocam(%arg0: !cir.int<u, 64>) -> !cir.ptr<!cir.void> {
// CHECK-NEXT:   %0 = cir.alloca !cir.int<u, 64>, !cir.ptr<!cir.int<u, 64>>, ["n", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.ptr<!cir.void>, !cir.ptr<!cir.ptr<!cir.void>>, ["__retval"] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.int<u, 64>, !cir.ptr<!cir.int<u, 64>>
// CHECK-NEXT:   %2 = cir.load align(8) %0 : !cir.ptr<!cir.int<u, 64>>, !cir.int<u, 64>
// CHECK-NEXT:   %3 = cir.alloca !cir.int<u, 8>, !cir.ptr<!cir.int<u, 8>>, %2 : !cir.int<u, 64>, ["bi_alloca"] {alignment = 16 : i64}
// CHECK-NEXT:   %4 = cir.cast bitcast %3 : !cir.ptr<!cir.int<u, 8>> -> !cir.ptr<!cir.void>
// CHECK-NEXT:   cir.store %4, %1 : !cir.ptr<!cir.void>, !cir.ptr<!cir.ptr<!cir.void>>
// CHECK-NEXT:   %5 = cir.load %1 : !cir.ptr<!cir.ptr<!cir.void>>, !cir.ptr<!cir.void>
// CHECK-NEXT:   cir.return %5 : !cir.ptr<!cir.void>
// CHECK-NEXT: }
