"""Phase 2c — arithmetic, compare, select, ternary, constants.

Sign-aware op selection per Decision 4 of the plan: every op that admits
both signed and unsigned variants reads `IntType.is_signed` from the
operand types and dispatches accordingly.
"""

from __future__ import annotations

from xdsl.dialects import arith, llvm, memref
from xdsl.dialects.builtin import (
    AnyFloat,
    Float32Type,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    StringAttr,
    UnitAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Operation, SSAValue
from xdsl.utils.hints import isa

from xdsl_clang.dialects import cir
from xdsl_clang.transforms.cir_to_core.components.cir_types import (
    convert_cir_type_to_standard,
    signedness_of,
)
from xdsl_clang.transforms.cir_to_core.components.memory import (
    const_array_to_dense,
    zero_dense_for_type,
)
from xdsl_clang.transforms.cir_to_core.misc.c_code_description import ProgramState
from xdsl_clang.transforms.cir_to_core.misc.ssa_context import SSAValueCtx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opd(ctx: SSAValueCtx, value: SSAValue) -> SSAValue:
    """Look up a CIR-side SSA value's lowered counterpart. Falls back to the
    original value if no mapping is registered (which means the op was kept
    type-compatible across the lowering — true for arith/cmp ops)."""
    mapped = ctx[value]
    return mapped if mapped is not None else value


def _is_signed(value: SSAValue) -> bool:
    """Best-effort signedness, defaulting to signed (matches MLIR's default
    when emitting ambiguous arith ops)."""
    sgn = signedness_of(value)
    return True if sgn is None else sgn


def _is_int(value: SSAValue) -> bool:
    return isa(value.type, cir.IntType) or isa(value.type, cir.BoolType)


def _is_float(value: SSAValue) -> bool:
    return isa(value.type, cir.SingleType) or isa(value.type, cir.DoubleType)


# ---------------------------------------------------------------------------
# cir.const
# ---------------------------------------------------------------------------


def translate_constant(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.ConstantOp
) -> list[Operation]:
    attr = op.value
    if isa(attr, cir.CIRIntAttr):
        out_ty = convert_cir_type_to_standard(attr.int_type, program_state)
        # CIRIntAttr stores the raw bit-pattern as an i64; re-wrap it at the
        # target width.
        v = attr.value.value.data
        # For signed types we may need to re-encode negatives.
        const = arith.ConstantOp(IntegerAttr(v, out_ty), out_ty)
        ctx[op.results[0]] = const.results[0]
        return [const]
    if isa(attr, cir.CIRBoolAttr):
        const = arith.ConstantOp(IntegerAttr(1 if attr.value else 0, 1), IntegerType(1))
        ctx[op.results[0]] = const.results[0]
        return [const]
    if isa(attr, cir.CIRFPAttr):
        out_ty = convert_cir_type_to_standard(attr.fp_type, program_state)
        assert isinstance(out_ty, AnyFloat)
        v = attr.value.value.data
        const = arith.ConstantOp(FloatAttr(v, out_ty), out_ty)
        ctx[op.results[0]] = const.results[0]
        return [const]
    if isa(attr, cir.ConstPtrAttr):
        # Null-pointer constant (`int *p = NULL;`). The result type
        # depends on Decision 1 — record/function/void* pointers lower to
        # `!llvm.ptr`; scalar/decayed pointers lower to a memref.
        #
        # Loads/stores *through* a NULL pointer are UB in C and we don't
        # model them. The SSA value still has to *exist*, however, so the
        # surrounding `cir.store %null, %slot` typechecks against the
        # parent slot's element type. For the memref case we therefore
        # materialise a zero `!llvm.ptr` and `unrealized_conversion_cast`
        # it to the target memref type — the cast leaves no runtime
        # artefact (it's resolved away by `reconcile-unrealized-casts`)
        # and the resulting memref is poison: dereferencing it is UB,
        # which matches C's NULL-deref semantics exactly.
        #
        # We use the decayed (`memref<?xT>`) pointer convention so the
        # result type matches the slot type that pointer-of-pointer
        # `cir.alloca`s produce (see `translate_alloca`).
        from xdsl_clang.transforms.cir_to_core.components.cir_types import (
            DECAYED_PTR,
        )

        target_ty = convert_cir_type_to_standard(
            attr.ptr_type, program_state, ptr_mode=DECAYED_PTR
        )
        zero = llvm.ZeroOp.build(result_types=[llvm.LLVMPointerType()])
        if isinstance(target_ty, MemRefType):
            cast = UnrealizedConversionCastOp.get(
                [zero.results[0]], [target_ty]
            )
            ctx[op.results[0]] = cast.results[0]
            return [zero, cast]
        # `!llvm.ptr` directly — record / function / void* pointer.
        ctx[op.results[0]] = zero.results[0]
        return [zero]
    if isa(attr, cir.ZeroAttr):
        # Scalar zero — emit `arith.constant 0`. Array zeros at the
        # function-local scope are hoisted to a private `memref.global`
        # with a `dense<0>` initial value (Task 5.2).
        ty = op.results[0].type
        if isa(ty, cir.IntType) or isa(ty, cir.BoolType):
            out_ty = convert_cir_type_to_standard(ty, program_state)
            const = arith.ConstantOp(IntegerAttr(0, out_ty), out_ty)
            ctx[op.results[0]] = const.results[0]
            return [const]
        if isa(ty, cir.SingleType) or isa(ty, cir.DoubleType):
            out_ty = convert_cir_type_to_standard(ty, program_state)
            assert isinstance(out_ty, AnyFloat)
            const = arith.ConstantOp(FloatAttr(0.0, out_ty), out_ty)
            ctx[op.results[0]] = const.results[0]
            return [const]
        if isa(ty, cir.ArrayType):
            return _hoist_zero_array_literal(program_state, ctx, op, ty)
        raise NotImplementedError(
            f"cir-to-core: function-local cir.const #cir.zero of type {ty} "
            f"(Phase 5 — needs constant hoisting)"
        )
    if isa(attr, cir.ConstArrayAttr):
        # Constant array literal — `float a[10] = {1, 2, …};`. Hoist the
        # initialiser to a private `memref.global` and replace the
        # `cir.const` with a `memref.get_global` (Task 5.2). The
        # surrounding `cir.store %c, %slot` is fused into a `memref.copy`
        # by `translate_store`.
        ty = op.results[0].type
        if isa(ty, cir.ArrayType):
            return _hoist_const_array_literal(program_state, ctx, op, attr, ty)
        raise NotImplementedError(
            f"cir-to-core: cir.const #cir.const_array of non-array type {ty}"
        )
    raise NotImplementedError(
        f"cir-to-core: unsupported constant attribute {type(attr).__name__}"
    )


def _hoist_const_array_literal(
    program_state: ProgramState,
    ctx: SSAValueCtx,
    op: cir.ConstantOp,
    attr: cir.ConstArrayAttr,
    array_ty: "cir.ArrayType",
) -> list[Operation]:
    """Hoist a `cir.const #cir.const_array` to a private `memref.global`.

    The result of the original `cir.const` is replaced with a
    `memref.get_global` returning a memref of the same shape. Subsequent
    `cir.store` of the SSA value is recognised by `translate_store` and
    fused into a `memref.copy`.
    """
    target_ty = convert_cir_type_to_standard(array_ty, program_state)
    assert isinstance(target_ty, MemRefType)
    dense = const_array_to_dense(program_state, attr, target_ty)
    sym = program_state.fresh_literal_symbol()
    global_op = memref.GlobalOp.get(
        StringAttr(sym),
        target_ty,
        initial_value=dense,
        sym_visibility=StringAttr("private"),
        constant=UnitAttr(),
    )
    program_state.append_module_prelude_op(global_op)
    get = memref.GetGlobalOp(sym, target_ty)
    ctx[op.results[0]] = get.results[0]
    return [get]


def _hoist_zero_array_literal(
    program_state: ProgramState,
    ctx: SSAValueCtx,
    op: cir.ConstantOp,
    array_ty: "cir.ArrayType",
) -> list[Operation]:
    """Hoist a `cir.const #cir.zero : !cir.array<…>` to a zero-init
    private `memref.global`.
    """
    target_ty = convert_cir_type_to_standard(array_ty, program_state)
    assert isinstance(target_ty, MemRefType)
    dense = zero_dense_for_type(target_ty)
    sym = program_state.fresh_literal_symbol()
    global_op = memref.GlobalOp.get(
        StringAttr(sym),
        target_ty,
        initial_value=dense,
        sym_visibility=StringAttr("private"),
        constant=UnitAttr(),
    )
    program_state.append_module_prelude_op(global_op)
    get = memref.GetGlobalOp(sym, target_ty)
    ctx[op.results[0]] = get.results[0]
    return [get]


# ---------------------------------------------------------------------------
# cir.binop  — Decision 4: signed/unsigned variants chosen from operand sign
# ---------------------------------------------------------------------------


_BIN_KIND_INT_SIGNED = {
    "mul": arith.MuliOp,
    "add": arith.AddiOp,
    "sub": arith.SubiOp,
    "div": arith.DivSIOp,
    "rem": arith.RemSIOp,
    "shl": arith.ShLIOp,
    "shr": arith.ShRSIOp,
    "and": arith.AndIOp,
    "or": arith.OrIOp,
    "xor": arith.XOrIOp,
}

_BIN_KIND_INT_UNSIGNED = {
    "mul": arith.MuliOp,
    "add": arith.AddiOp,
    "sub": arith.SubiOp,
    "div": arith.DivUIOp,
    "rem": arith.RemUIOp,
    "shl": arith.ShLIOp,
    "shr": arith.ShRUIOp,
    "and": arith.AndIOp,
    "or": arith.OrIOp,
    "xor": arith.XOrIOp,
}

_BIN_KIND_FLOAT = {
    "mul": arith.MulfOp,
    "add": arith.AddfOp,
    "sub": arith.SubfOp,
    "div": arith.DivfOp,
}


def _binop_kind_name(op: cir.BinOp) -> str:
    return cir._BIN_OP_KIND_INV[op.kind.value.data]  # type: ignore[attr-defined]


def translate_binop(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.BinOp
) -> list[Operation]:
    kind = _binop_kind_name(op)
    lhs = _opd(ctx, op.lhs)
    rhs = _opd(ctx, op.rhs)
    if _is_int(op.lhs):
        table = _BIN_KIND_INT_SIGNED if _is_signed(op.lhs) else _BIN_KIND_INT_UNSIGNED
        cls = table.get(kind)
        if cls is None:
            raise NotImplementedError(
                f"cir-to-core: unsupported integer binop kind {kind!r}"
            )
        new_op = cls(lhs, rhs)
    elif _is_float(op.lhs):
        cls_f = _BIN_KIND_FLOAT.get(kind)
        if cls_f is None:
            raise NotImplementedError(
                f"cir-to-core: unsupported float binop kind {kind!r}"
            )
        new_op = cls_f(lhs, rhs)
    else:
        raise NotImplementedError(
            f"cir-to-core: binop on type {op.lhs.type}"
        )
    ctx[op.results[0]] = new_op.results[0]
    return [new_op]


# ---------------------------------------------------------------------------
# cir.cmp  — Decision 4 again
# ---------------------------------------------------------------------------


# arith.cmpi predicates
# 0=eq 1=ne 2=slt 3=sle 4=sgt 5=sge 6=ult 7=ule 8=ugt 9=uge
_CMPI_SIGNED = {"eq": 0, "ne": 1, "lt": 2, "le": 3, "gt": 4, "ge": 5}
_CMPI_UNSIGNED = {"eq": 0, "ne": 1, "lt": 6, "le": 7, "gt": 8, "ge": 9}
# arith.cmpf predicates (ordered)
# 0=false 1=oeq 2=ogt 3=oge 4=olt 5=ole 6=one 7=ord ...
_CMPF_ORDERED = {"eq": 1, "gt": 2, "ge": 3, "lt": 4, "le": 5, "ne": 6}


def translate_cmp(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.CmpOp
) -> list[Operation]:
    kind = cir._CMP_OP_KIND_INV[op.kind.value.data]  # type: ignore[attr-defined]
    lhs = _opd(ctx, op.lhs)
    rhs = _opd(ctx, op.rhs)
    if _is_int(op.lhs):
        table = _CMPI_SIGNED if _is_signed(op.lhs) else _CMPI_UNSIGNED
        pred = table[kind]
        new_op = arith.CmpiOp(lhs, rhs, pred)
    elif _is_float(op.lhs):
        pred = _CMPF_ORDERED[kind]
        new_op = arith.CmpfOp(lhs, rhs, pred)
    else:
        raise NotImplementedError(f"cir-to-core: cmp on type {op.lhs.type}")
    # `cir.cmp` results in `!cir.bool` (= i1 post-lowering) — arith.cmp{i,f}
    # produces i1, types match.
    ctx[op.results[0]] = new_op.results[0]
    return [new_op]


# ---------------------------------------------------------------------------
# cir.unary
# ---------------------------------------------------------------------------


def translate_unary(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.UnaryOp
) -> list[Operation]:
    kind = cir._UNARY_OP_KIND_INV[op.kind.value.data]  # type: ignore[attr-defined]
    src = _opd(ctx, op.input)
    src_ty = op.input.type

    if kind == "plus":
        # Unary plus is a no-op in C semantics; just rewire.
        ctx[op.results[0]] = src
        return []

    if kind == "minus":
        if _is_int(op.input):
            zero = arith.ConstantOp(IntegerAttr(0, src.type), src.type)
            sub = arith.SubiOp(zero.results[0], src)
            ctx[op.results[0]] = sub.results[0]
            return [zero, sub]
        if _is_float(op.input):
            neg = arith.NegfOp(src)
            ctx[op.results[0]] = neg.results[0]
            return [neg]

    if kind == "not":
        # Bitwise complement on integers, logical-not on bool — both the
        # same i1/iN xor-with-all-ones in MLIR.
        if _is_int(op.input):
            ones = arith.ConstantOp(IntegerAttr(-1, src.type), src.type)
            xorop = arith.XOrIOp(src, ones.results[0])
            ctx[op.results[0]] = xorop.results[0]
            return [ones, xorop]

    if kind in ("inc", "dec"):
        # Pre-/post-increment of a value (the address-update happens via a
        # surrounding load+store sequence in CIR).
        if _is_int(op.input):
            one = arith.ConstantOp(IntegerAttr(1, src.type), src.type)
            cls = arith.AddiOp if kind == "inc" else arith.SubiOp
            new = cls(src, one.results[0])
            ctx[op.results[0]] = new.results[0]
            return [one, new]
        if _is_float(op.input):
            assert isinstance(src.type, AnyFloat)
            one = arith.ConstantOp(FloatAttr(1.0, src.type), src.type)
            cls_f = arith.AddfOp if kind == "inc" else arith.SubfOp
            new = cls_f(src, one.results[0])
            ctx[op.results[0]] = new.results[0]
            return [one, new]

    raise NotImplementedError(
        f"cir-to-core: unsupported unary kind {kind!r} on {src_ty}"
    )


# ---------------------------------------------------------------------------
# cir.select / cir.ternary
# ---------------------------------------------------------------------------


def translate_select(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.SelectOp
) -> list[Operation]:
    cond = _opd(ctx, op.cond)
    t = _opd(ctx, op.true_value)
    f = _opd(ctx, op.false_value)
    new_op = arith.SelectOp(cond, t, f)
    ctx[op.results[0]] = new_op.results[0]
    return [new_op]


def translate_ternary(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.TernaryOp
) -> list[Operation]:
    # `cir.ternary` is the C `?:` operator with two yielded regions. For
    # constant-foldable cases CIR canonicalises to `cir.select`. The
    # general region form is handled by the control-flow component; this
    # entry point only fires when the value is consumed as an expression.
    from xdsl_clang.transforms.cir_to_core.components import control_flow as _ctrl

    return _ctrl.translate_ternary(program_state, ctx, op)
