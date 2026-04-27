// RUN: XDSL_ROUNDTRIP

!u64i = !cir.int<u, 64>
!void = !cir.void

module {
  cir.func @test_max(%arg0: !cir.ptr<!void>) -> !u64i {
    %0 = cir.objsize max %arg0 : !cir.ptr<!void> -> !u64i
    cir.return %0 : !u64i
  }
}

// CHECK:      cir.func @test_max(%arg0: !cir.ptr<!cir.void>) -> !cir.int<u, 64> {
// CHECK-NEXT:   %0 = cir.objsize max %arg0 : !cir.ptr<!cir.void> -> !cir.int<u, 64>
// CHECK-NEXT:   cir.return %0 : !cir.int<u, 64>
// CHECK-NEXT: }

// -----

!u64i = !cir.int<u, 64>
!void = !cir.void

module {
  cir.func @test_max_nullunknown(%arg0: !cir.ptr<!void>) -> !u64i {
    %0 = cir.objsize max nullunknown %arg0 : !cir.ptr<!void> -> !u64i
    cir.return %0 : !u64i
  }
}

// CHECK:      cir.func @test_max_nullunknown(%arg0: !cir.ptr<!cir.void>) -> !cir.int<u, 64> {
// CHECK-NEXT:   %0 = cir.objsize max nullunknown %arg0 : !cir.ptr<!cir.void> -> !cir.int<u, 64>
// CHECK-NEXT:   cir.return %0 : !cir.int<u, 64>
// CHECK-NEXT: }

// -----

!u64i = !cir.int<u, 64>
!void = !cir.void

module {
  cir.func @test_max_dynamic(%arg0: !cir.ptr<!void>) -> !u64i {
    %0 = cir.objsize max dynamic %arg0 : !cir.ptr<!void> -> !u64i
    cir.return %0 : !u64i
  }
}

// CHECK:      cir.func @test_max_dynamic(%arg0: !cir.ptr<!cir.void>) -> !cir.int<u, 64> {
// CHECK-NEXT:   %0 = cir.objsize max dynamic %arg0 : !cir.ptr<!cir.void> -> !cir.int<u, 64>
// CHECK-NEXT:   cir.return %0 : !cir.int<u, 64>
// CHECK-NEXT: }

// -----

!u64i = !cir.int<u, 64>
!void = !cir.void

module {
  cir.func @test_max_nullunknown_dynamic(%arg0: !cir.ptr<!void>) -> !u64i {
    %0 = cir.objsize max nullunknown dynamic %arg0 : !cir.ptr<!void> -> !u64i
    cir.return %0 : !u64i
  }
}

// CHECK:      cir.func @test_max_nullunknown_dynamic(%arg0: !cir.ptr<!cir.void>) -> !cir.int<u, 64> {
// CHECK-NEXT:   %0 = cir.objsize max nullunknown dynamic %arg0 : !cir.ptr<!cir.void> -> !cir.int<u, 64>
// CHECK-NEXT:   cir.return %0 : !cir.int<u, 64>
// CHECK-NEXT: }

// -----

!u64i = !cir.int<u, 64>
!void = !cir.void

module {
  cir.func @test_min(%arg0: !cir.ptr<!void>) -> !u64i {
    %0 = cir.objsize min %arg0 : !cir.ptr<!void> -> !u64i
    cir.return %0 : !u64i
  }
}

// CHECK:      cir.func @test_min(%arg0: !cir.ptr<!cir.void>) -> !cir.int<u, 64> {
// CHECK-NEXT:   %0 = cir.objsize min %arg0 : !cir.ptr<!cir.void> -> !cir.int<u, 64>
// CHECK-NEXT:   cir.return %0 : !cir.int<u, 64>
// CHECK-NEXT: }

// -----

!u64i = !cir.int<u, 64>
!void = !cir.void

module {
  cir.func @test_min_nullunknown(%arg0: !cir.ptr<!void>) -> !u64i {
    %0 = cir.objsize min nullunknown %arg0 : !cir.ptr<!void> -> !u64i
    cir.return %0 : !u64i
  }
}

// CHECK:      cir.func @test_min_nullunknown(%arg0: !cir.ptr<!cir.void>) -> !cir.int<u, 64> {
// CHECK-NEXT:   %0 = cir.objsize min nullunknown %arg0 : !cir.ptr<!cir.void> -> !cir.int<u, 64>
// CHECK-NEXT:   cir.return %0 : !cir.int<u, 64>
// CHECK-NEXT: }

// -----

!u64i = !cir.int<u, 64>
!void = !cir.void

module {
  cir.func @test_min_dynamic(%arg0: !cir.ptr<!void>) -> !u64i {
    %0 = cir.objsize min dynamic %arg0 : !cir.ptr<!void> -> !u64i
    cir.return %0 : !u64i
  }
}

// CHECK:      cir.func @test_min_dynamic(%arg0: !cir.ptr<!cir.void>) -> !cir.int<u, 64> {
// CHECK-NEXT:   %0 = cir.objsize min dynamic %arg0 : !cir.ptr<!cir.void> -> !cir.int<u, 64>
// CHECK-NEXT:   cir.return %0 : !cir.int<u, 64>
// CHECK-NEXT: }

// -----

!u64i = !cir.int<u, 64>
!void = !cir.void

module {
  cir.func @test_min_nullunknown_dynamic(%arg0: !cir.ptr<!void>) -> !u64i {
    %0 = cir.objsize min nullunknown dynamic %arg0 : !cir.ptr<!void> -> !u64i
    cir.return %0 : !u64i
  }
}

// CHECK:      cir.func @test_min_nullunknown_dynamic(%arg0: !cir.ptr<!cir.void>) -> !cir.int<u, 64> {
// CHECK-NEXT:   %0 = cir.objsize min nullunknown dynamic %arg0 : !cir.ptr<!cir.void> -> !cir.int<u, 64>
// CHECK-NEXT:   cir.return %0 : !cir.int<u, 64>
// CHECK-NEXT: }
