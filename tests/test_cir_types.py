"""Unit coverage for ``components/cir_types.convert_cir_type_to_standard``.

Exercises the four design decisions that pin the surface of the
type-conversion module:

  1. Pointer representation (memref<T> / memref<?xT> / !llvm.ptr).
  3. Records → !llvm.struct<…>.
  4. signedness_of() helper for the sign-tracking pipeline.

Decisions 2 (loop policy) and the runtime sign-aware op selection in (4)
are exercised by the Phase 2c/2d component tests.
"""

from __future__ import annotations

from xdsl.dialects import builtin, llvm
from xdsl.dialects.builtin import DYNAMIC_INDEX

import xdsl_clang  # noqa: F401  (registers cir-compat patches)
from xdsl_clang.dialects import cir
from xdsl_clang.transforms.cir_to_core.components.cir_types import (
    DECAYED_PTR,
    SCALAR_PTR,
    convert_cir_type_to_standard,
    signedness_of,
)
from xdsl_clang.transforms.cir_to_core.misc.c_code_description import ProgramState


def _ps() -> ProgramState:
    return ProgramState()


# --- Decision 1: pointer mapping --------------------------------------------


def test_int_widths_map_to_iN():
    ps = _ps()
    for w in (1, 8, 16, 32, 64, 128):
        assert convert_cir_type_to_standard(
            cir.IntType(w, True), ps
        ) == builtin.IntegerType(w)


def test_bool_maps_to_i1():
    assert convert_cir_type_to_standard(cir.BoolType(), _ps()) == builtin.IntegerType(1)


def test_floats_map():
    ps = _ps()
    assert convert_cir_type_to_standard(cir.SingleType(), ps) == builtin.Float32Type()
    assert convert_cir_type_to_standard(cir.DoubleType(), ps) == builtin.Float64Type()


def test_array_maps_to_static_memref():
    i32 = cir.IntType(32, True)
    out = convert_cir_type_to_standard(cir.ArrayType(i32, 10), _ps())
    assert out == builtin.MemRefType(builtin.IntegerType(32), [10])


def test_scalar_pointer_maps_to_memref_T():
    i32 = cir.IntType(32, True)
    out = convert_cir_type_to_standard(cir.PointerType(i32), _ps())
    assert out == builtin.MemRefType(builtin.IntegerType(32), [])


def test_decayed_pointer_maps_to_memref_dyn_T():
    i32 = cir.IntType(32, True)
    out = convert_cir_type_to_standard(
        cir.PointerType(i32), _ps(), ptr_mode=DECAYED_PTR
    )
    assert out == builtin.MemRefType(builtin.IntegerType(32), [DYNAMIC_INDEX])


def test_pointer_to_array_unwraps_to_static_memref():
    # ptr<array<i8 x 25>> always lowers to memref<25xi8>; the array carries
    # the length so we don't need the dynamic memref form.
    u8 = cir.IntType(8, False)
    out = convert_cir_type_to_standard(
        cir.PointerType(cir.ArrayType(u8, 25)), _ps()
    )
    assert out == builtin.MemRefType(builtin.IntegerType(8), [25])


def test_void_pointer_is_opaque():
    out = convert_cir_type_to_standard(cir.PointerType(cir.VoidType()), _ps())
    assert isinstance(out, llvm.LLVMPointerType)


def test_record_pointer_is_opaque():
    rec = cir.RecordType(
        [cir.IntType(32, True), cir.DoubleType()], record_name="P"
    )
    out = convert_cir_type_to_standard(cir.PointerType(rec), _ps(), ptr_mode=SCALAR_PTR)
    assert isinstance(out, llvm.LLVMPointerType)


def test_function_pointer_is_opaque():
    fnt = cir.FuncType([cir.IntType(32, True)], cir.IntType(32, True))
    out = convert_cir_type_to_standard(cir.PointerType(fnt), _ps())
    assert isinstance(out, llvm.LLVMPointerType)


# --- Decision 3: records ----------------------------------------------------


def test_record_maps_to_llvm_struct():
    rec = cir.RecordType(
        [cir.IntType(32, True), cir.DoubleType()], record_name="Pair"
    )
    out = convert_cir_type_to_standard(rec, _ps())
    expected = llvm.LLVMStructType.from_type_list(
        [builtin.IntegerType(32), builtin.Float64Type()]
    )
    assert out == expected


# --- Decision 4: signedness helper -----------------------------------------


def test_signedness_of_signed_int():
    # Build an SSA value carrying `!cir.int<s, 32>` via a simple ConstantOp.
    op = cir.ConstantOp.create(
        result_types=[cir.IntType(32, True)],
        properties={"value": cir.CIRIntAttr(0, cir.IntType(32, True))},
    )
    assert signedness_of(op.results[0]) is True


def test_signedness_of_unsigned_int():
    op = cir.ConstantOp.create(
        result_types=[cir.IntType(32, False)],
        properties={"value": cir.CIRIntAttr(0, cir.IntType(32, False))},
    )
    assert signedness_of(op.results[0]) is False


def test_signedness_of_bool_is_unsigned():
    op = cir.ConstantOp.create(
        result_types=[cir.BoolType()],
        properties={"value": cir.CIRBoolAttr(False)},
    )
    assert signedness_of(op.results[0]) is False


def test_signedness_of_float_is_none():
    op = cir.ConstantOp.create(
        result_types=[cir.SingleType()],
        properties={"value": cir.CIRFPAttr(0.0, cir.SingleType())},
    )
    assert signedness_of(op.results[0]) is None
