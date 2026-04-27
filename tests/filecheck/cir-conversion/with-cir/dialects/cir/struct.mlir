// RUN: CIR_ROUNDTRIP

!u8i = !cir.int<u, 8>
!u16i = !cir.int<u, 16>
!s32i = !cir.int<s, 32>
!u32i = !cir.int<u, 32>

!rec_C = !cir.record<class "C" incomplete>
!rec_S = !cir.record<struct "S" incomplete>
!rec_U = !cir.record<union "U" incomplete>

!rec_anon_struct = !cir.record<struct packed {!s32i, !s32i, !cir.array<!s32i x 8>}>
!rec_anon_struct1 = !cir.record<struct {!cir.array<!cir.ptr<!u8i> x 5>}>
!rec_anon_struct2 = !cir.record<struct {!cir.ptr<!u8i>, !cir.ptr<!u8i>, !cir.ptr<!u8i>}>
!rec_S1 = !cir.record<struct "S1" {!s32i, !s32i}>
!rec_Sc = !cir.record<struct "Sc" {!u8i, !u16i, !u32i}>

!rec_P1 = !cir.record<struct "P1" packed {!s32i, !s32i}>
!rec_P2 = !cir.record<struct "P2" padded {!u8i, !u16i, !u32i}>
!rec_P3 = !cir.record<struct "P3" packed padded {!u8i, !u16i, !u32i}>

!rec_Ac = !cir.record<class "A" {!u8i, !s32i}>

module {
  cir.func @useTypes(%arg0: !rec_anon_struct,
                     %arg1: !rec_anon_struct1,
                     %arg2: !rec_anon_struct2,
                     %arg3: !rec_S1,
                     %arg4: !rec_Sc,
                     %arg5: !rec_Ac,
                     %arg6: !rec_C,
                     %arg7: !rec_S,
                     %arg8: !rec_U,
                     %arg9: !rec_P1,
                     %arg10: !rec_P2,
                     %arg11: !rec_P3) {
    cir.return
  }
}

// CHECK:      cir.func @useTypes(
// CHECK-SAME:    %{{.*}}: !cir.record<struct packed {!cir.int<s, 32>, !cir.int<s, 32>, !cir.array<!cir.int<s, 32> x 8>}>,
// CHECK-SAME:    !cir.record<struct {!cir.array<!cir.ptr<!cir.int<u, 8>> x 5>}>,
// CHECK-SAME:    !cir.record<struct {!cir.ptr<!cir.int<u, 8>>, !cir.ptr<!cir.int<u, 8>>, !cir.ptr<!cir.int<u, 8>>}>,
// CHECK-SAME:    !cir.record<struct "S1" {!cir.int<s, 32>, !cir.int<s, 32>}>,
// CHECK-SAME:    !cir.record<struct "Sc" {!cir.int<u, 8>, !cir.int<u, 16>, !cir.int<u, 32>}>,
// CHECK-SAME:    !cir.record<class "A" {!cir.int<u, 8>, !cir.int<s, 32>}>,
// CHECK-SAME:    !cir.record<class "C" incomplete>,
// CHECK-SAME:    !cir.record<struct "S" incomplete>,
// CHECK-SAME:    !cir.record<union "U" incomplete>,
// CHECK-SAME:    !cir.record<struct "P1" packed {!cir.int<s, 32>, !cir.int<s, 32>}>,
// CHECK-SAME:    !cir.record<struct "P2" padded {!cir.int<u, 8>, !cir.int<u, 16>, !cir.int<u, 32>}>,
// CHECK-SAME:    !cir.record<struct "P3" packed padded {!cir.int<u, 8>, !cir.int<u, 16>, !cir.int<u, 32>}>) {
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

// -----

!u8i = !cir.int<u, 8>
!u16i = !cir.int<u, 16>
!u32i = !cir.int<u, 32>
!rec_Sc = !cir.record<struct "Sc" {!u8i, !u16i, !u32i}>
!rec_U = !cir.record<union "U" incomplete>

module {
  cir.func @structs() {
    %0 = cir.alloca !cir.ptr<!rec_Sc>, !cir.ptr<!cir.ptr<!rec_Sc>>, ["sc", init]
    %1 = cir.alloca !cir.ptr<!rec_U>, !cir.ptr<!cir.ptr<!rec_U>>, ["u", init]
    cir.return
  }
}

// CHECK:      cir.func @structs() {
// CHECK-NEXT:   %{{.*}} = cir.alloca !cir.ptr<!cir.record<struct "Sc" {!cir.int<u, 8>, !cir.int<u, 16>, !cir.int<u, 32>}>>, !cir.ptr<!cir.ptr<!cir.record<struct "Sc" {!cir.int<u, 8>, !cir.int<u, 16>, !cir.int<u, 32>}>>>, ["sc", init]
// CHECK-NEXT:   %{{.*}} = cir.alloca !cir.ptr<!cir.record<union "U" incomplete>>, !cir.ptr<!cir.ptr<!cir.record<union "U" incomplete>>>, ["u", init]
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
