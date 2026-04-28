"""Phase 2b — `cir.cast` lowering.

Each cast kind maps to one of:
  * a no-op (the value passes through unchanged at the lowered level)
  * an `arith.{ext,trunc,bitcast,sitofp,uitofp,fptosi,fptoui,extf,truncf}`
  * an `arith.cmpi ne <x>, 0` for `int_to_bool`
  * an `arith.extui` for `bool_to_int`

`array_to_ptrdecay` is the most common cast in the corpus (every array
indexing in C goes through it). At the lowered level the source type is
already `memref<NxT>` and the destination is `memref<?xT>`; we model this
by carrying the source SSA value through unchanged and relying on
`memref.load`/`store` accepting the static shape.
"""

from __future__ import annotations

from xdsl.dialects import arith, memref
from xdsl.dialects.builtin import (
    AnyFloat,
    DYNAMIC_INDEX,
    Float32Type,
    Float64Type,
    IntegerType,
    MemRefType,
)
from xdsl.ir import Operation, SSAValue
from xdsl.utils.hints import isa

from xdsl_clang.dialects import cir
from xdsl_clang.transforms.cir_to_core.components.cir_types import (
    convert_cir_type_to_standard,
    signedness_of,
)
from xdsl_clang.transforms.cir_to_core.misc.c_code_description import ProgramState
from xdsl_clang.transforms.cir_to_core.misc.ssa_context import SSAValueCtx


def translate_cast(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.CastOp
) -> list[Operation]:
    kind = cir._CAST_KIND_INV[op.kind.value.data]  # type: ignore[attr-defined]
    src = ctx[op.src]
    if src is None:
        src = op.src
    src_cir_ty = op.src.type
    dst_cir_ty = op.res.type
    dst_ty = convert_cir_type_to_standard(dst_cir_ty, program_state)

    if kind == "array_to_ptrdecay":
        # `!cir.ptr<!cir.array<T x N>>` (memref<NxT>) →
        # `!cir.ptr<T>` decayed (memref<?xT>). The default conversion of
        # `!cir.ptr<T>` is `memref<T>`; we override here because decayed
        # array pointers are the canonical lowering for this cast kind.
        from xdsl_clang.transforms.cir_to_core.components.cir_types import (
            DECAYED_PTR,
        )

        decay_ty = convert_cir_type_to_standard(
            dst_cir_ty, program_state, ptr_mode=DECAYED_PTR
        )
        src_ty = src.type
        if isinstance(src_ty, MemRefType) and isinstance(decay_ty, MemRefType):
            if list(src_ty.get_shape()) != list(decay_ty.get_shape()):
                cst = memref.CastOp.get(src, decay_ty)
                ctx[op.results[0]] = cst.results[0]
                return [cst]
        ctx[op.results[0]] = src
        return []
    if kind in ("bitcast", "ptr_to_int", "int_to_ptr"):
        # Memref/llvm.ptr representation absorbs these.
        ctx[op.results[0]] = src
        return []

    if kind == "int_to_bool":
        zero = arith.ConstantOp(_int_zero_attr(src.type), src.type)
        cmp = arith.CmpiOp(src, zero.results[0], 1)  # ne
        ctx[op.results[0]] = cmp.results[0]
        return [zero, cmp]

    if kind == "float_to_bool":
        from xdsl.dialects.builtin import FloatAttr

        assert isinstance(src.type, AnyFloat)
        zero = arith.ConstantOp(FloatAttr(0.0, src.type), src.type)
        # cmpf "une" (unordered or not equal) ≈ "non-zero".
        cmp = arith.CmpfOp(src, zero.results[0], 12)
        ctx[op.results[0]] = cmp.results[0]
        return [zero, cmp]

    if kind == "bool_to_int":
        new = arith.ExtUIOp(src, dst_ty)
        ctx[op.results[0]] = new.results[0]
        return [new]

    if kind in ("integral", "boolean"):
        return _integral_cast(src, op, dst_ty, ctx, src_cir_ty)

    if kind in ("floating",):
        # f64 ↔ f32 (or wider). Choose extf vs truncf by bitwidth.
        return _float_resize(src, dst_ty, ctx, op)

    if kind in ("int_to_float", "int_to_float_legacy"):
        # signed → sitofp, unsigned → uitofp
        signed = signedness_of(op.src)
        cls = arith.SIToFPOp if (signed is None or signed) else arith.UIToFPOp
        new = cls(src, dst_ty)
        ctx[op.results[0]] = new.results[0]
        return [new]

    if kind in ("float_to_int", "floating_to_int"):
        signed = signedness_of(op.results[0])
        cls = arith.FPToSIOp if (signed is None or signed) else arith.FPToUIOp
        new = cls(src, dst_ty)
        ctx[op.results[0]] = new.results[0]
        return [new]

    raise NotImplementedError(f"cir-to-core: unsupported cast kind {kind!r}")


def _int_zero_attr(ty):
    from xdsl.dialects.builtin import IntegerAttr

    return IntegerAttr(0, ty)


def _integral_cast(
    src: SSAValue,
    op: cir.CastOp,
    dst_ty,
    ctx: SSAValueCtx,
    src_cir_ty,
) -> list[Operation]:
    src_w = src.type.width.data  # type: ignore[attr-defined]
    dst_w = dst_ty.width.data  # type: ignore[attr-defined]
    if dst_w == src_w:
        ctx[op.results[0]] = src
        return []
    if dst_w < src_w:
        new = arith.TrunciOp(src, dst_ty)
    else:
        # Widening: pick signed vs unsigned by source signedness.
        signed = signedness_of(op.src)
        cls = arith.ExtSIOp if (signed is None or signed) else arith.ExtUIOp
        new = cls(src, dst_ty)
    ctx[op.results[0]] = new.results[0]
    return [new]


def _float_resize(
    src: SSAValue, dst_ty, ctx: SSAValueCtx, op: cir.CastOp
) -> list[Operation]:
    src_ty = src.type
    assert isinstance(src_ty, AnyFloat) and isinstance(dst_ty, AnyFloat)
    src_bits = _float_bits(src_ty)
    dst_bits = _float_bits(dst_ty)
    if dst_bits == src_bits:
        ctx[op.results[0]] = src
        return []
    if dst_bits < src_bits:
        new = arith.TruncFOp(src, dst_ty)
    else:
        new = arith.ExtFOp(src, dst_ty)
    ctx[op.results[0]] = new.results[0]
    return [new]


def _float_bits(ty: AnyFloat) -> int:
    if isinstance(ty, Float32Type):
        return 32
    if isinstance(ty, Float64Type):
        return 64
    raise NotImplementedError(f"cir-to-core: unsupported float type {ty}")
