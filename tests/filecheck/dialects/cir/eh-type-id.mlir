// RUN: XDSL_ROUNDTRIP

!u8i = !cir.int<u, 8>

module {
  cir.global "private" constant external @_ZTIi : !cir.ptr<!u8i>
}

// CHECK:      builtin.module {
// CHECK-NEXT:   cir.global "private" constant external @_ZTIi : !cir.ptr<!cir.int<u, 8>>
// CHECK-NEXT: }
