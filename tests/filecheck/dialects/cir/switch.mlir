// RUN: XDSL_ROUNDTRIP

!s32i = !cir.int<s, 32>

cir.func @s0() {
  %1 = cir.const #cir.int<2> : !s32i
  cir.switch (%1 : !s32i) {
    cir.case (default, []) {
      cir.return
    }
    cir.case (equal, [#cir.int<3> : !s32i]) {
      cir.yield
    }
    cir.case (anyof, [#cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i]) {
      cir.break
    }
    cir.case (equal, [#cir.int<5> : !s32i]) {
      cir.yield
    }
    cir.yield
  }
  cir.return
}

// CHECK:      cir.switch (%0 : !cir.int<s, 32>) {
// CHECK-NEXT:   cir.case (default, []) {
// CHECK-NEXT:     cir.return
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.case (equal, [#cir.int<3> : !cir.int<s, 32>]) {
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.case (anyof, [#cir.int<6> : !cir.int<s, 32>, #cir.int<7> : !cir.int<s, 32>, #cir.int<8> : !cir.int<s, 32>]) {
// CHECK-NEXT:     cir.break
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.case (equal, [#cir.int<5> : !cir.int<s, 32>]) {
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: }

// -----

!s32i = !cir.int<s, 32>

cir.func @s1(%1 : !s32i) {
  cir.switch (%1 : !s32i) all_enum_cases_covered {
    cir.case (default, []) {
      cir.return
    }
    cir.case (equal, [#cir.int<1> : !s32i]) {
      cir.yield
    }
    cir.case (equal, [#cir.int<2> : !s32i]) {
      cir.yield
    }
    cir.yield
  }
  cir.return
}

// CHECK:      cir.switch (%0 : !cir.int<s, 32>) all_enum_cases_covered {
// CHECK-NEXT:   cir.case (default, []) {
// CHECK-NEXT:     cir.return
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.case (equal, [#cir.int<1> : !cir.int<s, 32>]) {
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.case (equal, [#cir.int<2> : !cir.int<s, 32>]) {
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: }
