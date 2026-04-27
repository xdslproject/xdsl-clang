// RUN: CIR_ROUNDTRIP

!u8i = !cir.int<u, 8>

module {
  cir.func @stack_save_restore() {
    %0 = cir.stacksave : !cir.ptr<!u8i>
    cir.stackrestore %0 : !cir.ptr<!u8i>
    cir.return
  }
}

// CHECK:      cir.func @stack_save_restore() {
// CHECK-NEXT:   %{{.*}} = cir.stacksave : !cir.ptr<!cir.int<u, 8>>
// CHECK-NEXT:   cir.stackrestore %{{.*}} : !cir.ptr<!cir.int<u, 8>>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
