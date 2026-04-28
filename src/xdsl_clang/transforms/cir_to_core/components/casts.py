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
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
)
from xdsl.ir import Operation, SSAValue
from xdsl.utils.hints import isa

from xdsl_clang.dialects import cir
from xdsl_clang.transforms.cir_to_core.components.cir_types import (
    cir_type_size_in_bytes,
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
    if kind == "bitcast":
        # Phase 5 Task 5.4: pattern-match `void* ↔ T*` bitcasts around the
        # `malloc` / `free` C idiom and lower them to `memref.alloc` /
        # `memref.dealloc` directly. This avoids ever producing
        # `unrealized_conversion_cast`s between `!llvm.ptr` and
        # `memref<?xT>`, which can't be reconciled at the LLVM stage.
        malloc_alloc = _try_lower_malloc_bitcast(program_state, ctx, op)
        if malloc_alloc is not None:
            return malloc_alloc
        if _is_free_void_ptr_bitcast(op):
            # Result is consumed by a subsequent `cir.call @free`; emit
            # nothing here and let the call handler dealloc the typed
            # memref directly. Map the result to the underlying value so
            # any defensive lookup still works.
            ctx[op.results[0]] = src
            return []
        # Default: at the lowered level the memref/llvm.ptr representation
        # absorbs the bitcast.
        ctx[op.results[0]] = src
        return []
    if kind in ("ptr_to_int", "int_to_ptr"):
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


def _is_void_ptr(ty) -> bool:
    return isa(ty, cir.PointerType) and isa(ty.pointee, cir.VoidType)


def _is_free_void_ptr_bitcast(op: cir.CastOp) -> bool:
    """Return True if `op` is the `T* -> void*` bitcast feeding a
    subsequent `cir.call @free` in the same block.
    """
    if not _is_void_ptr(op.res.type):
        return False
    if not isa(op.src.type, cir.PointerType):
        return False
    # Walk forward in the block looking for a `cir.call @free` that uses
    # `op.res` as its only argument.
    for use in op.results[0].uses:
        owner = use.operation
        if isa(owner, cir.CallOp):
            callee = owner.callee
            if callee is not None and callee.root_reference.data == "free":
                return True
    return False


def _try_lower_malloc_bitcast(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.CastOp
) -> list[Operation] | None:
    """Detect `cir.cast bitcast %m : !cir.ptr<!cir.void> -> !cir.ptr<T>`
    where `%m` is the result of `cir.call @malloc(...)`. If so, emit
    `memref.alloc(<size>) : memref<?xT>` and return the ops; otherwise
    return None so the caller falls back to the default bitcast handling.
    """
    if not _is_void_ptr(op.src.type):
        return None
    dst_ty = op.res.type
    if not isa(dst_ty, cir.PointerType):
        return None
    pointee = dst_ty.pointee
    # Records / functions / nested void* aren't sensible malloc targets
    # in the corpus — bail to the default and let the existing cast logic
    # handle them.
    if isa(pointee, cir.RecordType) or isa(pointee, cir.FuncType) or isa(
        pointee, cir.VoidType
    ):
        return None

    # `op.src` must be the result of `cir.call @malloc`.
    src_owner = op.src.owner
    if not isa(src_owner, cir.CallOp):
        return None
    callee = src_owner.callee
    if callee is None or callee.root_reference.data != "malloc":
        return None

    # Compute the dynamic element count = byte_count / sizeof(T). The
    # byte count is the single argument to `malloc`. We always emit a
    # divui at lowering time — element-size folding into a constant is
    # cheap for the verifier and survives canonicalisation.
    byte_arg = src_owner.arg_ops[0]
    mapped_byte = ctx[byte_arg]
    if mapped_byte is None:
        mapped_byte = byte_arg

    elem_size = cir_type_size_in_bytes(pointee)
    elem_ty = convert_cir_type_to_standard(pointee, program_state)
    memref_ty = MemRefType(elem_ty, [DYNAMIC_INDEX])

    # Convert byte count (i64 / iN) → index, divide by element size.
    pre_ops: list[Operation] = []
    byte_ty = mapped_byte.type
    # If the byte count is already an index, skip the cast.
    if isinstance(byte_ty, IndexType):
        byte_idx = mapped_byte
    else:
        cast = arith.IndexCastOp(mapped_byte, IndexType())
        pre_ops.append(cast)
        byte_idx = cast.results[0]

    if elem_size == 1:
        nelems = byte_idx
    else:
        size_const = arith.ConstantOp(
            IntegerAttr(elem_size, IndexType()), IndexType()
        )
        div = arith.DivUIOp(byte_idx, size_const.results[0])
        pre_ops.append(size_const)
        pre_ops.append(div)
        nelems = div.results[0]

    alloc = memref.AllocOp.get(
        elem_ty,
        shape=[DYNAMIC_INDEX],
        dynamic_sizes=[nelems],
    )
    ctx[op.results[0]] = alloc.memref
    # Also map the malloc call's result so any other defensive lookups
    # don't end up with a None — chase-back through `_try_lower_malloc`
    # consumers normally goes through the bitcast result, but this is
    # cheap insurance.
    ctx[src_owner.results[0]] = alloc.memref
    return [*pre_ops, alloc]


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
