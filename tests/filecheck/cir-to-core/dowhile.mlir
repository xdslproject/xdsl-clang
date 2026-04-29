// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

// Phase 5 Task 5.8: lowering `cir.do` (do-while loops).
//
// Decision 2: do-while has body-then-cond shape, which doesn't map cleanly
// to `scf.while` (cond-then-body). We always lower it through the
// unstructured `cf.br` / `cf.cond_br` block-graph emitter — the
// surrounding function is flagged `is_unstructured` regardless of whether
// the body contains break/continue.
//
// Layout:
//     cur:     cf.br ^body          (body always runs once)
//     ^body:   <body>; cf.br ^header
//     ^header: <cond>; cf.cond_br %c, ^body, ^exit
//     ^exit:   <continuation>

!s32i = !cir.int<s, 32>

module {
  // ----- plain do-while ------------------------------------------------------
  // do { c += n; n++; } while (n < k);
  cir.func @sum_until(%k: !s32i) -> !s32i {
    %n = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init] {alignment = 4 : i64}
    %s = cir.alloca !s32i, !cir.ptr<!s32i>, ["s", init] {alignment = 4 : i64}
    %z = cir.const #cir.int<0> : !s32i
    cir.store %z, %n : !s32i, !cir.ptr<!s32i>
    cir.store %z, %s : !s32i, !cir.ptr<!s32i>
    cir.scope {
      cir.do {
        %ln = cir.load %n : !cir.ptr<!s32i>, !s32i
        %ls = cir.load %s : !cir.ptr<!s32i>, !s32i
        %a = cir.binop(add, %ls, %ln) : !s32i
        cir.store %a, %s : !s32i, !cir.ptr<!s32i>
        %one = cir.const #cir.int<1> : !s32i
        %np = cir.binop(add, %ln, %one) : !s32i
        cir.store %np, %n : !s32i, !cir.ptr<!s32i>
        cir.yield
      } while {
        %ln = cir.load %n : !cir.ptr<!s32i>, !s32i
        %c = cir.cmp(lt, %ln, %k) : !s32i, !cir.bool
        cir.condition(%c)
      }
    }
    %fs = cir.load %s : !cir.ptr<!s32i>, !s32i
    cir.return %fs : !s32i
  }
  // CHECK:      func.func @sum_until(
  // Loop entry: jump straight to the body block (do-while runs body first).
  // CHECK:        cf.br ^[[BODY:.+]]
  // Body block: accumulates and increments, then branches to the cond header.
  // CHECK:      ^[[BODY]]:
  // CHECK:        memref.store
  // CHECK:        cf.br ^[[HEADER:.+]]
  // Header block: cond eval, then conditional branch back to body or exit.
  // CHECK:      ^[[HEADER]]:
  // CHECK:        %{{.*}} = arith.cmpi slt
  // CHECK:        cf.cond_br %{{.*}}, ^[[BODY]], ^[[EXIT:.+]]
  // Exit block hosts the post-loop continuation (return).
  // CHECK:      ^[[EXIT]]:
  // CHECK:        func.return

  // ----- do-while with `break` ----------------------------------------------
  // do { if (n == k) break; s += n; n++; } while (n < lim);
  cir.func @sum_until_break(%k: !s32i, %lim: !s32i) -> !s32i {
    %n = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init] {alignment = 4 : i64}
    %s = cir.alloca !s32i, !cir.ptr<!s32i>, ["s", init] {alignment = 4 : i64}
    %z = cir.const #cir.int<0> : !s32i
    cir.store %z, %n : !s32i, !cir.ptr<!s32i>
    cir.store %z, %s : !s32i, !cir.ptr<!s32i>
    cir.scope {
      cir.do {
        %ln = cir.load %n : !cir.ptr<!s32i>, !s32i
        %eq = cir.cmp(eq, %ln, %k) : !s32i, !cir.bool
        cir.if %eq {
          cir.break
        }
        %ls = cir.load %s : !cir.ptr<!s32i>, !s32i
        %a = cir.binop(add, %ls, %ln) : !s32i
        cir.store %a, %s : !s32i, !cir.ptr<!s32i>
        %one = cir.const #cir.int<1> : !s32i
        %np = cir.binop(add, %ln, %one) : !s32i
        cir.store %np, %n : !s32i, !cir.ptr<!s32i>
        cir.yield
      } while {
        %ln = cir.load %n : !cir.ptr<!s32i>, !s32i
        %c = cir.cmp(lt, %ln, %lim) : !s32i, !cir.bool
        cir.condition(%c)
      }
    }
    %fs = cir.load %s : !cir.ptr<!s32i>, !s32i
    cir.return %fs : !s32i
  }
  // CHECK:      func.func @sum_until_break(
  // CHECK:        cf.br ^[[BBODY:.+]]
  // Body diamond: equality test branches to the `break` arm or the body
  // continuation.
  // CHECK:      ^[[BBODY]]:
  // CHECK:        %{{.*}} = arith.cmpi eq
  // CHECK:        cf.cond_br %{{.*}}, ^[[BTHEN:.+]], ^[[BMERGE:.+]]
  // Header block: cond eval, branches to the body or to the exit block.
  // CHECK:      ^[[BHEADER:.+]]:
  // CHECK:        %{{.*}} = arith.cmpi slt
  // CHECK:        cf.cond_br %{{.*}}, ^[[BBODY]], ^[[BEXIT:.+]]
  // Exit block hosts the return continuation.
  // CHECK:      ^[[BEXIT]]:
  // CHECK:        func.return
  // The `break` arm jumps to the loop's exit block, NOT to the cond header.
  // CHECK:      ^[[BTHEN]]:
  // CHECK-NEXT:   cf.br ^[[BEXIT]]
  // The merge of the body's inner cir.if continues with the body's
  // remaining statements, then branches to the cond header.
  // CHECK:      ^[[BMERGE]]:
  // CHECK:        memref.store
  // CHECK:        cf.br ^[[BHEADER]]

  // ----- do-while with `continue` -------------------------------------------
  // do { n++; if (n % 2 == 0) continue; s += n; } while (n < lim);
  cir.func @sum_odd(%lim: !s32i) -> !s32i {
    %n = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init] {alignment = 4 : i64}
    %s = cir.alloca !s32i, !cir.ptr<!s32i>, ["s", init] {alignment = 4 : i64}
    %z = cir.const #cir.int<0> : !s32i
    cir.store %z, %n : !s32i, !cir.ptr<!s32i>
    cir.store %z, %s : !s32i, !cir.ptr<!s32i>
    cir.scope {
      cir.do {
        %ln = cir.load %n : !cir.ptr<!s32i>, !s32i
        %one = cir.const #cir.int<1> : !s32i
        %np = cir.binop(add, %ln, %one) : !s32i
        cir.store %np, %n : !s32i, !cir.ptr<!s32i>
        %two = cir.const #cir.int<2> : !s32i
        %m = cir.binop(rem, %np, %two) : !s32i
        %iszero = cir.cmp(eq, %m, %z) : !s32i, !cir.bool
        cir.if %iszero {
          cir.continue
        }
        %ls = cir.load %s : !cir.ptr<!s32i>, !s32i
        %a = cir.binop(add, %ls, %np) : !s32i
        cir.store %a, %s : !s32i, !cir.ptr<!s32i>
        cir.yield
      } while {
        %ln = cir.load %n : !cir.ptr<!s32i>, !s32i
        %c = cir.cmp(lt, %ln, %lim) : !s32i, !cir.bool
        cir.condition(%c)
      }
    }
    %fs = cir.load %s : !cir.ptr<!s32i>, !s32i
    cir.return %fs : !s32i
  }
  // CHECK:      func.func @sum_odd(
  // CHECK:        cf.br ^[[CBODY:.+]]
  // CHECK:      ^[[CBODY]]:
  // CHECK:        %{{.*}} = arith.cmpi eq
  // CHECK:        cf.cond_br %{{.*}}, ^[[CTHEN:.+]], ^[[CMERGE:.+]]
  // Header block: branches to body or to exit.
  // CHECK:      ^[[CHEADER:.+]]:
  // CHECK:        %{{.*}} = arith.cmpi slt
  // CHECK:        cf.cond_br %{{.*}}, ^[[CBODY]], ^[[CEXIT:.+]]
  // CHECK:      ^[[CEXIT]]:
  // CHECK:        func.return
  // `continue` in a do-while jumps to the cond header (the test runs).
  // CHECK:      ^[[CTHEN]]:
  // CHECK-NEXT:   cf.br ^[[CHEADER]]
  // The merge of the body's cir.if accumulates and branches to the header.
  // CHECK:      ^[[CMERGE]]:
  // CHECK:        memref.store
  // CHECK:        cf.br ^[[CHEADER]]
}
