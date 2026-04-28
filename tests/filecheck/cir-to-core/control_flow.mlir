// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

!s32i = !cir.int<s, 32>

module {
  cir.func @ifelse(%a: !s32i, %b: !s32i) -> !s32i {
    %c = cir.cmp(gt, %a, %b) : !s32i, !cir.bool
    %r = cir.alloca !s32i, !cir.ptr<!s32i>, ["r"] {alignment = 4 : i64}
    cir.if %c {
      cir.store %a, %r : !s32i, !cir.ptr<!s32i>
    } else {
      cir.store %b, %r : !s32i, !cir.ptr<!s32i>
    }
    %lr = cir.load %r : !cir.ptr<!s32i>, !s32i
    cir.return %lr : !s32i
  }
  // CHECK:      func.func @ifelse(
  // CHECK:        %{{.*}} = arith.cmpi sgt
  // CHECK:        scf.if %{{.*}} {
  // CHECK:          memref.store
  // CHECK:        } else {
  // CHECK:          memref.store

  cir.func @forloop(%n: !s32i) -> !s32i {
    %i = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
    %s = cir.alloca !s32i, !cir.ptr<!s32i>, ["s", init] {alignment = 4 : i64}
    %z = cir.const #cir.int<0> : !s32i
    cir.store %z, %i : !s32i, !cir.ptr<!s32i>
    cir.store %z, %s : !s32i, !cir.ptr<!s32i>
    cir.scope {
      cir.for : cond {
        %li = cir.load %i : !cir.ptr<!s32i>, !s32i
        %c = cir.cmp(lt, %li, %n) : !s32i, !cir.bool
        cir.condition(%c)
      } body {
        %li = cir.load %i : !cir.ptr<!s32i>, !s32i
        %ls = cir.load %s : !cir.ptr<!s32i>, !s32i
        %a = cir.binop(add, %ls, %li) : !s32i
        cir.store %a, %s : !s32i, !cir.ptr<!s32i>
        cir.yield
      } step {
        %li = cir.load %i : !cir.ptr<!s32i>, !s32i
        %one = cir.const #cir.int<1> : !s32i
        %ip = cir.binop(add, %li, %one) : !s32i
        cir.store %ip, %i : !s32i, !cir.ptr<!s32i>
        cir.yield
      }
    }
    %fs = cir.load %s : !cir.ptr<!s32i>, !s32i
    cir.return %fs : !s32i
  }
  // CHECK:      func.func @forloop(
  // CHECK:        scf.while
  // CHECK:          arith.cmpi slt
  // CHECK:          scf.condition(
  // CHECK:        do
  // CHECK:          scf.yield
}
