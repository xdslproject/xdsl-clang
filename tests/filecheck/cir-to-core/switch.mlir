// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

// Phase 5 Task 5.9: lowering `cir.switch` (and the LLVM-style flattened
// `cir.switch.flat`) to a `cf.cond_br` dispatch chain.
//
// `scf.index_switch` was considered but it can't represent C-style
// fall-through across cases (each case region must terminate with
// `scf.yield`) and disallows `break`-style early exits, so the block
// graph is the cleaner target. The whole surrounding function switches to
// the unstructured emitter — see the `is_unstructured` flag and the
// `_has_switch_anywhere` scan in `components/functions.py`.

!s32i = !cir.int<s, 32>

module {
  // ----- structured `cir.switch` -------------------------------------------
  // C source equivalent:
  //   int classify(int v) {
  //     int r = 0;
  //     switch (v) {
  //       case 1: r = 10; break;
  //       case 2: r = 20; /* fall-through */
  //       case 3: r += 100; break;
  //       default: r = -1;
  //     }
  //     return r;
  //   }
  cir.func @classify(%v: !s32i) -> !s32i {
    %r = cir.alloca !s32i, !cir.ptr<!s32i>, ["r", init] {alignment = 4 : i64}
    %z = cir.const #cir.int<0> : !s32i
    cir.store %z, %r : !s32i, !cir.ptr<!s32i>
    cir.scope {
      cir.switch (%v : !s32i) {
        cir.case(equal, [#cir.int<1> : !s32i]) {
          %c1 = cir.const #cir.int<10> : !s32i
          cir.store %c1, %r : !s32i, !cir.ptr<!s32i>
          cir.break
        }
        cir.case(equal, [#cir.int<2> : !s32i]) {
          %c2 = cir.const #cir.int<20> : !s32i
          cir.store %c2, %r : !s32i, !cir.ptr<!s32i>
          cir.yield
        }
        cir.case(equal, [#cir.int<3> : !s32i]) {
          %c3 = cir.const #cir.int<100> : !s32i
          %lr = cir.load %r : !cir.ptr<!s32i>, !s32i
          %sum = cir.binop(add, %lr, %c3) : !s32i
          cir.store %sum, %r : !s32i, !cir.ptr<!s32i>
          cir.break
        }
        cir.case(default, []) {
          %cm1 = cir.const #cir.int<-1> : !s32i
          cir.store %cm1, %r : !s32i, !cir.ptr<!s32i>
          cir.yield
        }
        cir.yield
      }
    }
    %ret = cir.load %r : !cir.ptr<!s32i>, !s32i
    cir.return %ret : !s32i
  }
  // CHECK:      func.func @classify(%[[V:.+]]: i32) -> i32
  // The dispatch chain compares the condition against each case value in
  // source order; the false target of the last value-bearing test is the
  // default block (or the exit block if no default).
  // CHECK:        %[[K1:.+]] = arith.constant 1 : i32
  // CHECK:        %[[E1:.+]] = arith.cmpi eq, %[[V]], %[[K1]]
  // CHECK:        cf.cond_br %[[E1]], ^[[CASE1:.+]], ^[[D1:.+]]
  // case 1: store 10 then break — branches straight to the exit block.
  // CHECK:      ^[[CASE1]]:
  // CHECK:        memref.store
  // CHECK:        cf.br ^[[EXIT:.+]]
  // case 2: store 20 then fall-through — branches to case 3's body block.
  // CHECK:      ^[[CASE2:.+]]:
  // CHECK:        memref.store
  // CHECK:        cf.br ^[[CASE3:.+]]
  // case 3: store r += 100 then break — branches to exit.
  // CHECK:      ^[[CASE3]]:
  // CHECK:        memref.store
  // CHECK:        cf.br ^[[EXIT]]
  // default body: store -1 then implicit fall-through (last case).
  // CHECK:      ^[[DEFAULT:.+]]:
  // CHECK:        memref.store
  // CHECK:        cf.br ^[[EXIT]]
  // Exit block hosts the post-switch continuation (the return).
  // CHECK:      ^[[EXIT]]:
  // CHECK:        func.return
  // Dispatch chain: each successive equality test lives in its own block.
  // CHECK:      ^[[D1]]:
  // CHECK:        %[[K2:.+]] = arith.constant 2 : i32
  // CHECK:        %[[E2:.+]] = arith.cmpi eq, %[[V]], %[[K2]]
  // CHECK:        cf.cond_br %[[E2]], ^[[CASE2]], ^[[D2:.+]]
  // CHECK:      ^[[D2]]:
  // CHECK:        %[[K3:.+]] = arith.constant 3 : i32
  // CHECK:        %[[E3:.+]] = arith.cmpi eq, %[[V]], %[[K3]]
  // The false target of the last value-bearing case is the default block.
  // CHECK:        cf.cond_br %[[E3]], ^[[CASE3]], ^[[DEFAULT]]

  // ----- `anyof` and `range` cases -----------------------------------------
  // `case 1: case 3: case 5:` and `case 10 ... 20:` exercise the OR-chain
  // and the GE/LE pair lowerings respectively.
  cir.func @anyof_range(%v: !s32i) -> !s32i {
    %r = cir.alloca !s32i, !cir.ptr<!s32i>, ["r", init] {alignment = 4 : i64}
    %z = cir.const #cir.int<0> : !s32i
    cir.store %z, %r : !s32i, !cir.ptr<!s32i>
    cir.scope {
      cir.switch (%v : !s32i) {
        cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i]) {
          %c1 = cir.const #cir.int<7> : !s32i
          cir.store %c1, %r : !s32i, !cir.ptr<!s32i>
          cir.break
        }
        cir.case(range, [#cir.int<10> : !s32i, #cir.int<20> : !s32i]) {
          %c2 = cir.const #cir.int<8> : !s32i
          cir.store %c2, %r : !s32i, !cir.ptr<!s32i>
          cir.break
        }
        cir.yield
      }
    }
    %ret = cir.load %r : !cir.ptr<!s32i>, !s32i
    cir.return %ret : !s32i
  }
  // CHECK:      func.func @anyof_range(%[[A:.+]]: i32) -> i32
  // anyof — disjunction of equality tests OR'd together.
  // CHECK:        arith.cmpi eq, %[[A]], %{{.*}}
  // CHECK:        arith.cmpi eq, %[[A]], %{{.*}}
  // CHECK:        arith.ori
  // CHECK:        arith.cmpi eq, %[[A]], %{{.*}}
  // CHECK:        %[[ANYOF:.+]] = arith.ori
  // CHECK:        cf.cond_br %[[ANYOF]], ^{{.+}}, ^[[RANGE_HEAD:.+]]
  // range — `[lo, hi]` lowers to sge AND sle.
  // CHECK:      ^[[RANGE_HEAD]]:
  // CHECK:        arith.cmpi sge, %[[A]], %{{.*}}
  // CHECK:        arith.cmpi sle, %[[A]], %{{.*}}
  // CHECK:        %[[INRANGE:.+]] = arith.andi
  // CHECK:        cf.cond_br %[[INRANGE]], ^{{.+}}, ^{{.+}}

  // ----- `cir.switch.flat` (LLVM-style flattened terminator) ---------------
  // Not produced by clang's default lowering pipeline (clang's internal
  // `cir-flatten-cfg` pass produces it), but we lower it for completeness.
  cir.func @flat(%v: !s32i) {
    cir.switch.flat %v : !s32i, ^default [
      1 : ^case1,
      2 : ^case2
    ]
  ^case1:
    cir.return
  ^case2:
    cir.return
  ^default:
    cir.return
  }
  // CHECK:      func.func @flat(%[[F:.+]]: i32) {
  // CHECK:        %[[FK1:.+]] = arith.constant 1 : i32
  // CHECK:        %[[FE1:.+]] = arith.cmpi eq, %[[F]], %[[FK1]]
  // CHECK:        cf.cond_br %[[FE1]], ^{{.+}}, ^[[FCHAIN:.+]]
  // CHECK:      ^[[FCHAIN]]:
  // CHECK:        %[[FK2:.+]] = arith.constant 2 : i32
  // CHECK:        %[[FE2:.+]] = arith.cmpi eq, %[[F]], %[[FK2]]
  // The false branch of the final test is the default successor.
  // CHECK:        cf.cond_br %[[FE2]], ^{{.+}}, ^{{.+}}
}
