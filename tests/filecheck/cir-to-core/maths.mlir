// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

!s32i = !cir.int<s, 32>
!u32i = !cir.int<u, 32>
!f64  = !cir.double

module {
  cir.func @addi(%a: !s32i, %b: !s32i) -> !s32i {
    %0 = cir.binop(add, %a, %b) nsw : !s32i
    cir.return %0 : !s32i
  }
  // CHECK:      func.func @addi(%[[A:.*]]: i32, %[[B:.*]]: i32) -> i32 {
  // CHECK-NEXT:   %[[R:.*]] = arith.addi %[[A]], %[[B]] : i32
  // CHECK-NEXT:   func.return %[[R]] : i32
  // CHECK-NEXT: }

  cir.func @sdiv(%a: !s32i, %b: !s32i) -> !s32i {
    %0 = cir.binop(div, %a, %b) : !s32i
    cir.return %0 : !s32i
  }
  // CHECK:      func.func @sdiv(%{{.*}}: i32, %{{.*}}: i32) -> i32 {
  // CHECK-NEXT:   {{.*}} = arith.divsi {{.*}} : i32

  cir.func @udiv(%a: !u32i, %b: !u32i) -> !u32i {
    %0 = cir.binop(div, %a, %b) : !u32i
    cir.return %0 : !u32i
  }
  // CHECK:      func.func @udiv(%{{.*}}: i32, %{{.*}}: i32) -> i32 {
  // CHECK-NEXT:   {{.*}} = arith.divui {{.*}} : i32

  cir.func @fdiv(%a: !f64, %b: !f64) -> !f64 {
    %0 = cir.binop(div, %a, %b) : !f64
    cir.return %0 : !f64
  }
  // CHECK:      func.func @fdiv(%{{.*}}: f64, %{{.*}}: f64) -> f64 {
  // CHECK-NEXT:   {{.*}} = arith.divf {{.*}} : f64

  cir.func @cmps(%a: !s32i, %b: !s32i) -> !cir.bool {
    %0 = cir.cmp(lt, %a, %b) : !s32i, !cir.bool
    cir.return %0 : !cir.bool
  }
  // CHECK:      func.func @cmps(%{{.*}}: i32, %{{.*}}: i32) -> i1 {
  // CHECK-NEXT:   {{.*}} = arith.cmpi slt, {{.*}} : i32

  cir.func @cmpu(%a: !u32i, %b: !u32i) -> !cir.bool {
    %0 = cir.cmp(lt, %a, %b) : !u32i, !cir.bool
    cir.return %0 : !cir.bool
  }
  // CHECK:      func.func @cmpu(%{{.*}}: i32, %{{.*}}: i32) -> i1 {
  // CHECK-NEXT:   {{.*}} = arith.cmpi ult, {{.*}} : i32

  cir.func @cmpf(%a: !f64, %b: !f64) -> !cir.bool {
    %0 = cir.cmp(le, %a, %b) : !f64, !cir.bool
    cir.return %0 : !cir.bool
  }
  // CHECK:      func.func @cmpf(%{{.*}}: f64, %{{.*}}: f64) -> i1 {
  // CHECK-NEXT:   {{.*}} = arith.cmpf ole, {{.*}} : f64

  cir.func @neg(%a: !s32i) -> !s32i {
    %0 = cir.unary(minus, %a) : !s32i, !s32i
    cir.return %0 : !s32i
  }
  // CHECK:      func.func @neg(%[[X:.*]]: i32) -> i32 {
  // CHECK-NEXT:   %[[Z:.*]] = arith.constant 0 : i32
  // CHECK-NEXT:   %{{.*}} = arith.subi %[[Z]], %[[X]] : i32

  cir.func @fneg(%a: !f64) -> !f64 {
    %0 = cir.unary(minus, %a) : !f64, !f64
    cir.return %0 : !f64
  }
  // CHECK:      func.func @fneg(%[[X:.*]]: f64) -> f64 {
  // CHECK-NEXT:   {{.*}} = arith.negf %[[X]] : f64

  cir.func @intk() -> !s32i {
    %0 = cir.const #cir.int<42> : !s32i
    cir.return %0 : !s32i
  }
  // CHECK:      func.func @intk() -> i32 {
  // CHECK-NEXT:   %{{.*}} = arith.constant 42 : i32

  cir.func @fpk() -> !f64 {
    %0 = cir.const #cir.fp<3.5> : !f64
    cir.return %0 : !f64
  }
  // CHECK:      func.func @fpk() -> f64 {
  // CHECK-NEXT:   %{{.*}} = arith.constant 3.5{{.*}} : f64
}
