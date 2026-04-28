// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

// Phase 5 Task 5.7: lowering loops that contain `cir.break` / `cir.continue`.
//
// Decision 2: when a loop body contains an early-exit (break / continue),
// the structured `scf.while` form can't represent the control flow; we
// emit a block graph using `cf.br` / `cf.cond_br` instead. The whole
// surrounding function switches to the unstructured emitter — see the
// `is_unstructured` flag on `ComponentState`.

!s32i = !cir.int<s, 32>

module {
  // ----- for-loop with `break` ---------------------------------------------
  // for (i = 0; i < n; i++) { if (i == k) break; s += i; }
  cir.func @sum_until(%n: !s32i, %k: !s32i) -> !s32i {
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
        %eq = cir.cmp(eq, %li, %k) : !s32i, !cir.bool
        cir.if %eq {
          cir.break
        }
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
  // CHECK:      func.func @sum_until(
  // Loop entry: branch to the cond header block.
  // CHECK:        cf.br ^[[HEADER:.+]]
  // Header: evaluate cond, branch into body or exit.
  // CHECK:      ^[[HEADER]]:
  // CHECK:        %{{.*}} = arith.cmpi slt
  // CHECK:        cf.cond_br %{{.*}}, ^[[BODY:.+]], ^[[EXIT:.+]]
  // Body: nested cir.if — diamond on the equality test, with `break`
  // emitted as a `cf.br` to the loop's exit block.
  // CHECK:      ^[[BODY]]:
  // CHECK:        %{{.*}} = arith.cmpi eq
  // CHECK:        cf.cond_br %{{.*}}, ^[[THEN:.+]], ^[[MERGE:.+]]
  // Step block branches back to the cond header.
  // CHECK:      ^{{.+}}:
  // CHECK:        memref.store {{.*}} : memref<i32>
  // CHECK:        cf.br ^[[HEADER]]
  // Exit block hosts the post-loop continuation (return).
  // CHECK:      ^[[EXIT]]:
  // CHECK:        func.return
  // `cir.break` becomes a direct cf.br to the loop's exit block.
  // CHECK:      ^[[THEN]]:
  // CHECK-NEXT:   cf.br ^[[EXIT]]
  // The merge block of the inner cir.if continues with the body's
  // remaining statements, then falls through to the step block.
  // CHECK:      ^[[MERGE]]:
  // CHECK:        memref.store

  // ----- while-loop with `continue` ----------------------------------------
  // while (i < n) { i++; if (i % 2 == 0) continue; s += i; }
  cir.func @sum_odd(%n: !s32i) -> !s32i {
    %i = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
    %s = cir.alloca !s32i, !cir.ptr<!s32i>, ["s", init] {alignment = 4 : i64}
    %z = cir.const #cir.int<0> : !s32i
    cir.store %z, %i : !s32i, !cir.ptr<!s32i>
    cir.store %z, %s : !s32i, !cir.ptr<!s32i>
    cir.scope {
      cir.while {
        %li = cir.load %i : !cir.ptr<!s32i>, !s32i
        %c = cir.cmp(lt, %li, %n) : !s32i, !cir.bool
        cir.condition(%c)
      } do {
        %li = cir.load %i : !cir.ptr<!s32i>, !s32i
        %one = cir.const #cir.int<1> : !s32i
        %ip = cir.binop(add, %li, %one) : !s32i
        cir.store %ip, %i : !s32i, !cir.ptr<!s32i>
        %two = cir.const #cir.int<2> : !s32i
        %m = cir.binop(rem, %ip, %two) : !s32i
        %iszero = cir.cmp(eq, %m, %z) : !s32i, !cir.bool
        cir.if %iszero {
          cir.continue
        }
        %ls = cir.load %s : !cir.ptr<!s32i>, !s32i
        %a = cir.binop(add, %ls, %ip) : !s32i
        cir.store %a, %s : !s32i, !cir.ptr<!s32i>
        cir.yield
      }
    }
    %fs = cir.load %s : !cir.ptr<!s32i>, !s32i
    cir.return %fs : !s32i
  }
  // CHECK:      func.func @sum_odd(
  // Loop entry.
  // CHECK:        cf.br ^[[WHEADER:.+]]
  // Header: cond eval → branch body / exit.
  // CHECK:      ^[[WHEADER]]:
  // CHECK:        %{{.*}} = arith.cmpi slt
  // CHECK:        cf.cond_br %{{.*}}, ^[[WBODY:.+]], ^[[WEXIT:.+]]
  // Body: increment then test even-ness; the cir.if guards a `cir.continue`.
  // CHECK:      ^[[WBODY]]:
  // CHECK:        %{{.*}} = arith.cmpi eq
  // CHECK:        cf.cond_br %{{.*}}, ^[[WTHEN:.+]], ^[[WMERGE:.+]]
  // No step block for `cir.while` — both `continue` and the implicit
  // fall-through at body end branch back to the header.
  // CHECK:      ^[[WEXIT]]:
  // CHECK:        func.return
  // `continue` jumps back to the cond header.
  // CHECK:      ^[[WTHEN]]:
  // CHECK-NEXT:   cf.br ^[[WHEADER]]
  // Merge block does the accumulation and also branches back to header.
  // CHECK:      ^[[WMERGE]]:
  // CHECK:        memref.store
  // CHECK-NEXT:   cf.br ^[[WHEADER]]
}
