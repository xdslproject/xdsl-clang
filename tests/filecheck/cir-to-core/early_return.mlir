// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

!s32i = !cir.int<s, 32>

module {
  // Early return inside `cir.if` — must lower to a `cf.cond_br` diamond
  // with `func.return` inside the then-arm.
  cir.func @early_return_if(%a: !s32i) -> !s32i {
    %z = cir.const #cir.int<0> : !s32i
    %c = cir.cmp(eq, %a, %z) : !s32i, !cir.bool
    cir.if %c {
      cir.return %z : !s32i
    }
    %one = cir.const #cir.int<1> : !s32i
    cir.return %one : !s32i
  }
  // CHECK:      func.func @early_return_if(%{{.*}}: i32) -> i32 {
  // CHECK:        cf.cond_br %{{.*}}, ^[[THEN:.*]], ^[[MERGE:.*]]
  // CHECK:      ^[[THEN]]
  // CHECK:        func.return %{{.*}} : i32
  // CHECK:      ^[[MERGE]]
  // CHECK:        func.return %{{.*}} : i32

  // Early return inside `cir.scope { cir.if { cir.return } }` — the scope
  // is inlined and the `cir.if` becomes a diamond.
  cir.func @early_return_scope(%a: !s32i) -> !s32i {
    %z = cir.const #cir.int<0> : !s32i
    cir.scope {
      %c = cir.cmp(eq, %a, %z) : !s32i, !cir.bool
      cir.if %c {
        cir.return %z : !s32i
      }
    }
    %one = cir.const #cir.int<1> : !s32i
    cir.return %one : !s32i
  }
  // CHECK:      func.func @early_return_scope(%{{.*}}: i32) -> i32 {
  // CHECK:        cf.cond_br %{{.*}}, ^[[T2:.*]], ^[[M2:.*]]
  // CHECK:      ^[[T2]]
  // CHECK:        func.return %{{.*}} : i32
  // CHECK:      ^[[M2]]
  // CHECK:        func.return %{{.*}} : i32

  // Early return inside `cir.if`/else — both arms terminate with their own
  // `func.return`. CIR still requires a trailing return at function scope
  // (the `cir.if` op itself isn't a terminator); we leave one in.
  cir.func @early_return_both_arms(%a: !s32i) -> !s32i {
    %z = cir.const #cir.int<0> : !s32i
    %one = cir.const #cir.int<1> : !s32i
    %c = cir.cmp(eq, %a, %z) : !s32i, !cir.bool
    cir.if %c {
      cir.return %z : !s32i
    } else {
      cir.return %one : !s32i
    }
    cir.return %one : !s32i
  }
  // CHECK:      func.func @early_return_both_arms(%{{.*}}: i32) -> i32 {
  // CHECK:        cf.cond_br %{{.*}}, ^[[BTHEN:.*]], ^[[BELSE:.*]]
  // CHECK:      ^[[BTHEN]]
  // CHECK:        func.return %{{.*}} : i32
  // CHECK:      ^[[BELSE]]
  // CHECK:        func.return %{{.*}} : i32

  // Block-graph function (multiple CIR blocks via `cir.br`) that ALSO
  // contains a nested return inside a `cir.if`. The unstructured emitter
  // must lower each block 1:1 and still expand the `cir.if` to a diamond.
  cir.func @block_graph_with_nested_return(%a: !s32i) -> !s32i {
    %z = cir.const #cir.int<0> : !s32i
    cir.br ^bb1
  ^bb1:
    %c = cir.cmp(eq, %a, %z) : !s32i, !cir.bool
    cir.if %c {
      cir.return %z : !s32i
    }
    cir.return %a : !s32i
  }
  // CHECK:      func.func @block_graph_with_nested_return(%{{.*}}: i32) -> i32 {
  // CHECK:        cf.br ^[[BB1:.*]]
  // CHECK:      ^[[BB1]]
  // CHECK:        cf.cond_br
  // CHECK:        func.return

  // Sanity: a function with NO nested return must still go through the
  // structured emitter (`scf.if`).
  cir.func @structured_if(%a: !s32i) -> !s32i {
    %r = cir.alloca !s32i, !cir.ptr<!s32i>, ["r"] {alignment = 4 : i64}
    %z = cir.const #cir.int<0> : !s32i
    %c = cir.cmp(eq, %a, %z) : !s32i, !cir.bool
    cir.if %c {
      cir.store %z, %r : !s32i, !cir.ptr<!s32i>
    } else {
      cir.store %a, %r : !s32i, !cir.ptr<!s32i>
    }
    %lr = cir.load %r : !cir.ptr<!s32i>, !s32i
    cir.return %lr : !s32i
  }
  // CHECK:      func.func @structured_if(
  // CHECK:        scf.if
  // CHECK:        func.return %{{.*}} : i32
}
