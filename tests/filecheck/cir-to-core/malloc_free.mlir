// RUN: xdsl-opt -p cir-to-core %s 2>&1 | filecheck %s

!s32i = !cir.int<s, 32>
!u64i = !cir.int<u, 64>
!void = !cir.void

module {
  cir.func private @malloc(!u64i) -> !cir.ptr<!void>
  cir.func private @free(!cir.ptr<!void>)

  // `float *p = malloc(N * sizeof(float)); free(p);` — the malloc/cast
  // pair lowers to `memref.alloc(<size_in_bytes>/sizeof(T))`, the
  // matching cast/free pair to `memref.dealloc`. Neither `malloc` nor
  // `free` produces a `func.func` declaration in the lowered module.
  cir.func @malloc_free_float() {
    %0 = cir.alloca !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>, ["p", init] {alignment = 8 : i64}
    %1 = cir.const #cir.int<60> : !u64i
    %2 = cir.const #cir.int<4> : !u64i
    %3 = cir.binop(mul, %1, %2) : !u64i
    %4 = cir.call @malloc(%3) : (!u64i) -> !cir.ptr<!void>
    %5 = cir.cast bitcast %4 : !cir.ptr<!void> -> !cir.ptr<!cir.float>
    cir.store %5, %0 : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>
    %6 = cir.load %0 : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
    %7 = cir.cast bitcast %6 : !cir.ptr<!cir.float> -> !cir.ptr<!void>
    cir.call @free(%7) : (!cir.ptr<!void>) -> ()
    cir.return
  }
  // CHECK-LABEL: func.func @malloc_free_float()
  // CHECK:        %[[BYTES:.*]] = arith.muli
  // CHECK:        %[[BIDX:.*]] = arith.index_cast %[[BYTES]] : i64 to index
  // CHECK:        %[[SZ:.*]] = arith.constant 4 : index
  // CHECK:        %[[N:.*]] = arith.divui %[[BIDX]], %[[SZ]] : index
  // CHECK:        %[[P:.*]] = memref.alloc(%[[N]]) : memref<?xf32>
  // CHECK:        memref.dealloc %{{.*}} : memref<?xf32>
  // CHECK:        func.return

  // Same shape, but element type is `int` (i32, sizeof=4) — verifies the
  // size table picks the right divisor for a different scalar.
  cir.func @malloc_free_int() {
    %0 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["q", init] {alignment = 8 : i64}
    %1 = cir.const #cir.int<10> : !u64i
    %2 = cir.const #cir.int<4> : !u64i
    %3 = cir.binop(mul, %1, %2) : !u64i
    %4 = cir.call @malloc(%3) : (!u64i) -> !cir.ptr<!void>
    %5 = cir.cast bitcast %4 : !cir.ptr<!void> -> !cir.ptr<!s32i>
    cir.store %5, %0 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
    %6 = cir.load %0 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
    %7 = cir.cast bitcast %6 : !cir.ptr<!s32i> -> !cir.ptr<!void>
    cir.call @free(%7) : (!cir.ptr<!void>) -> ()
    cir.return
  }
  // CHECK-LABEL: func.func @malloc_free_int()
  // CHECK:        %[[N2:.*]] = arith.divui %{{.*}}, %{{.*}} : index
  // CHECK:        %[[Q:.*]] = memref.alloc(%[[N2]]) : memref<?xi32>
  // CHECK:        memref.dealloc %{{.*}} : memref<?xi32>

  // Phase F1: `calloc(n, size)` is recognised the same way as `malloc`,
  // with the byte count materialised as an explicit `arith.muli`. The
  // call to `@calloc` is elided in the same way as `@malloc`.
  cir.func private @calloc(!u64i, !u64i) -> !cir.ptr<!void>
  cir.func @calloc_int() {
    %0 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["r", init] {alignment = 8 : i64}
    %n = cir.const #cir.int<8> : !u64i
    %sz = cir.const #cir.int<4> : !u64i
    %p = cir.call @calloc(%n, %sz) : (!u64i, !u64i) -> !cir.ptr<!void>
    %t = cir.cast bitcast %p : !cir.ptr<!void> -> !cir.ptr<!s32i>
    cir.store %t, %0 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
    cir.return
  }
  // CHECK-LABEL: func.func @calloc_int()
  // CHECK:        %[[NB:.*]] = arith.muli %{{.*}}, %{{.*}} : i64
  // CHECK:        %[[NIDX:.*]] = arith.index_cast %[[NB]] : i64 to index
  // CHECK:        %[[ESZ:.*]] = arith.constant 4 : index
  // CHECK:        %[[NE:.*]] = arith.divui %[[NIDX]], %[[ESZ]] : index
  // CHECK:        %[[CR:.*]] = memref.alloc(%[[NE]]) : memref<?xi32>

  // No `func.func @malloc`, `@calloc`, or `@free` declarations should
  // appear in the lowered module — they're elided externs.
  // CHECK-NOT: func.func{{.*}}@malloc
  // CHECK-NOT: func.func{{.*}}@calloc
  // CHECK-NOT: func.func{{.*}}@free
}
