// RUN: CIR_ROUNDTRIP

// Should parse and print C source language attribute.
module attributes {cir.lang = #cir.lang<c>} { }
// CHECK: builtin.module attributes {cir.lang = #cir.lang<c>}

// -----

// Should parse and print C++ source language attribute.
module attributes {cir.lang = #cir.lang<cxx>} { }
// CHECK: builtin.module attributes {cir.lang = #cir.lang<cxx>}
