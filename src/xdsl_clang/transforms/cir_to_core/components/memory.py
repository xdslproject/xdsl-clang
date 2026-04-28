"""Phase 2a — globals + memory.

Lowers:

* `cir.global`     →  `memref.global` (numeric scalars / static arrays) or
                       `llvm.mlir.global` (struct globals — Decision 3).
* `cir.get_global` →  `memref.get_global` / `llvm.mlir.addressof`.
* `cir.alloca`     →  `memref.alloca` (or `llvm.alloca` for record types,
                       per Decision 3 in `structs.md`).
"""

from __future__ import annotations

from xdsl.dialects import arith, llvm, memref
from xdsl.dialects.builtin import (
    AnyFloat,
    DYNAMIC_INDEX,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    StringAttr,
    UnitAttr,
    i32,
    i64,
)
from xdsl.ir import Attribute, Block, Operation, Region
from xdsl.utils.hints import isa

from xdsl_clang.dialects import cir
from xdsl_clang.transforms.cir_to_core.components.cir_types import (
    SCALAR_PTR,
    convert_cir_type_to_standard,
)
from xdsl_clang.transforms.cir_to_core.misc.c_code_description import ProgramState
from xdsl_clang.transforms.cir_to_core.misc.ssa_context import SSAValueCtx


# ---------------------------------------------------------------------------
# cir.alloca
# ---------------------------------------------------------------------------


def translate_alloca(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.AllocaOp
) -> list[Operation]:
    alloca_ty = op.alloca_type
    # Records → llvm.alloca returning !llvm.ptr (Decision 3).
    if isa(alloca_ty, cir.RecordType):
        struct_ty = convert_cir_type_to_standard(alloca_ty, program_state)
        one = arith.ConstantOp(IntegerAttr(1, 32), IntegerType(32))
        new = llvm.AllocaOp(one.results[0], elem_type=struct_ty)
        ctx[op.results[0]] = new.results[0]
        return [one, new]

    # Numeric/pointer alloca → memref.alloca/alloc. When the alloca is a
    # *slot for a pointer* (`int **`-style), use the same decayed memref
    # convention as function arguments so subsequent `cir.store %decayed,
    # %slot` types line up. See Decision 1 + Phase 2b plan.
    from xdsl_clang.transforms.cir_to_core.components.cir_types import DECAYED_PTR

    if isa(alloca_ty, cir.PointerType):
        elem_ty = convert_cir_type_to_standard(
            alloca_ty, program_state, ptr_mode=DECAYED_PTR
        )
    else:
        elem_ty = convert_cir_type_to_standard(
            alloca_ty, program_state, ptr_mode=SCALAR_PTR
        )
    if op.dyn_alloc_size:
        # Dynamic-sized stack allocation: `T alloc[N]` where N is a runtime
        # integer. CIR shape is element-count of `alloca_type`. Lower the
        # result to `memref<?x<elem>>`.
        from xdsl.dialects.builtin import IndexType

        dyn_in = ctx[op.dyn_alloc_size[0]]
        if dyn_in is None:
            dyn_in = op.dyn_alloc_size[0]
        cast = arith.IndexCastOp(dyn_in, IndexType())
        alloca_op = memref.AllocaOp.get(
            elem_ty, dynamic_sizes=[cast.results[0]], shape=[DYNAMIC_INDEX]
        )
        ctx[op.results[0]] = alloca_op.memref
        return [cast, alloca_op]

    # Static alloca: scalar or fixed-size array. The lowered type matches
    # what the C code expects to dereference — for scalars `memref<T>`,
    # for arrays `memref<NxT>`.
    if isa(alloca_ty, cir.ArrayType):
        result_ty = convert_cir_type_to_standard(alloca_ty, program_state)
        # `result_ty` is already `memref<NxT>`.
        assert isinstance(result_ty, MemRefType)
        alloca_op = memref.AllocaOp.get(
            result_ty.get_element_type(), shape=list(result_ty.get_shape())
        )
    else:
        alloca_op = memref.AllocaOp.get(elem_ty, shape=[])
    ctx[op.results[0]] = alloca_op.memref
    return [alloca_op]


# ---------------------------------------------------------------------------
# cir.get_global
# ---------------------------------------------------------------------------


def translate_get_global(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.GetGlobalOp
) -> list[Operation]:
    sym = op.glob_name.root_reference.data
    info = program_state.cir_globals.get(sym)
    if info is None:
        raise NotImplementedError(
            f"cir-to-core: cir.get_global to undeclared symbol {sym!r}"
        )
    cir_ty = info.cir_type
    if isa(cir_ty, cir.RecordType):
        # Decision 3: record globals live in `llvm.mlir.global`.
        addr = llvm.AddressOfOp(global_name=sym, result_type=llvm.LLVMPointerType())
        ctx[op.results[0]] = addr.results[0]
        return [addr]
    # memref-typed globals (numeric/array) — load via memref.get_global.
    if isa(cir_ty, cir.ArrayType):
        target_ty = convert_cir_type_to_standard(cir_ty, program_state)
    else:
        # scalar global — must be a memref<T>
        target_ty = MemRefType(
            convert_cir_type_to_standard(cir_ty, program_state), []
        )
    assert isinstance(target_ty, MemRefType)
    new = memref.GetGlobalOp(sym, target_ty)
    ctx[op.results[0]] = new.results[0]
    return [new]


# ---------------------------------------------------------------------------
# cir.global
# ---------------------------------------------------------------------------


def _const_ops_for_init(
    program_state: ProgramState, attr: Attribute, target_ty: Attribute
) -> tuple[list[Operation], Attribute | None]:
    """Build init-region ops for a global initialiser.

    Returns `(ops, dense_value)`. If `dense_value` is non-None the caller
    can shortcut to `memref.global` with that initial value; otherwise the
    `ops` list ends in the value to wrap in `llvm.return`.
    """
    if isa(attr, cir.CIRIntAttr):
        # Either a memref.global with an integer initial value (legal only
        # for memref<...xiN>, not memref<iN>), or an arith.constant for the
        # llvm.global region form.
        const = arith.ConstantOp(
            IntegerAttr(attr.value.value.data, target_ty), target_ty
        )
        return [const], None
    if isa(attr, cir.CIRFPAttr):
        assert isinstance(target_ty, AnyFloat)
        const = arith.ConstantOp(
            FloatAttr(attr.value.value.data, target_ty), target_ty
        )
        return [const], None
    if isa(attr, cir.ZeroAttr):
        # `#cir.zero` for an array type — emit a zero memref initial value.
        # Caller will detect this via the second return value.
        return [], _zero_dense_for_type(target_ty)
    if isa(attr, cir.ConstArrayAttr):
        # Materialise as a DenseElementsAttr if possible — only legal for
        # element-wise scalar constants of uniform type.
        dense = _const_array_to_dense(program_state, attr, target_ty)
        return [], dense
    raise NotImplementedError(
        f"cir-to-core: unsupported global initialiser {type(attr).__name__}"
    )


def _zero_dense_for_type(target_ty: Attribute) -> Attribute:
    if isinstance(target_ty, MemRefType):
        elem = target_ty.get_element_type()
        if isinstance(elem, IntegerType):
            zero = IntegerAttr(0, elem)
        elif isinstance(elem, AnyFloat):
            zero = FloatAttr(0.0, elem)
        else:
            raise NotImplementedError(
                f"cir-to-core: can't zero-init memref of {elem}"
            )
        from xdsl.dialects.builtin import RankedTensorType, TensorType

        # memref.global accepts a tensor splat as its initial_value.
        tensor_ty = TensorType(elem, list(target_ty.get_shape()))
        return DenseIntOrFPElementsAttr.create_dense_int(tensor_ty, [zero.value.data]) if isinstance(elem, IntegerType) else DenseIntOrFPElementsAttr.create_dense_float(tensor_ty, [zero.value.data])
    raise NotImplementedError("cir-to-core: zero-init unsupported here")


def _const_array_to_dense(
    program_state: ProgramState, attr: cir.ConstArrayAttr, target_ty: Attribute
) -> Attribute:
    if not isinstance(target_ty, MemRefType):
        raise NotImplementedError("cir-to-core: const array global must target memref")
    elem = target_ty.get_element_type()
    from xdsl.dialects.builtin import TensorType

    tensor_ty = TensorType(elem, list(target_ty.get_shape()))
    values: list = []
    elements = attr.elts
    if isinstance(elements, StringAttr):
        # String literals: each byte → i8 element.
        for ch in elements.data:
            values.append(ord(ch))
        # Trailing null is implicit at the C level, but the array length
        # already carries it through the type.
        # Pad with zeros if the array type is longer than the string.
        n = target_ty.get_shape()[0]
        while len(values) < n:
            values.append(0)
        return DenseIntOrFPElementsAttr.create_dense_int(tensor_ty, values)
    # ArrayAttr of typed scalar attrs.
    for entry in elements.data:
        if isa(entry, cir.CIRIntAttr):
            values.append(entry.value.value.data)
        elif isa(entry, cir.CIRFPAttr):
            values.append(entry.value.value.data)
        else:
            raise NotImplementedError(
                f"cir-to-core: const-array element {type(entry).__name__}"
            )
    if isinstance(elem, IntegerType):
        return DenseIntOrFPElementsAttr.create_dense_int(tensor_ty, values)
    return DenseIntOrFPElementsAttr.create_dense_float(tensor_ty, values)


def translate_global(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.GlobalOp
) -> list[Operation]:
    sym = op.sym_name
    cir_ty = op.sym_type

    # Records → llvm.mlir.global (Decision 3). For now we only handle the
    # zero-initialised case — `static struct` with no initialiser, or with
    # `#cir.zero`.
    if isa(cir_ty, cir.RecordType):
        struct_ty = convert_cir_type_to_standard(cir_ty, program_state)
        body_block = Block()
        if op.initial_value is not None:
            # Limited support: zero-init only.
            if not isa(op.initial_value, cir.ZeroAttr):
                raise NotImplementedError(
                    "cir-to-core: non-zero record global initialisers"
                )
        zero = llvm.ZeroOp(result_type=struct_ty)
        ret = llvm.ReturnOp(operands=[zero.results[0]])
        body_block.add_op(zero)
        body_block.add_op(ret)
        return [
            llvm.GlobalOp(
                global_type=struct_ty,
                sym_name=sym,
                linkage="external" if op.sym_visibility is None else "internal",
                addr_space=0,
                body=Region([body_block]),
            )
        ]

    # Scalar/array → memref.global where possible.
    if isa(cir_ty, cir.ArrayType):
        target_ty = convert_cir_type_to_standard(cir_ty, program_state)
    else:
        target_ty = MemRefType(
            convert_cir_type_to_standard(cir_ty, program_state), []
        )
    assert isinstance(target_ty, MemRefType)

    init_attr: Attribute | None = None
    if op.initial_value is not None:
        attr = op.initial_value
        if isa(attr, cir.ZeroAttr):
            # Use UnitAttr to mark "zero-initialised" for memref.global.
            init_attr = UnitAttr()
        elif isa(attr, cir.CIRIntAttr):
            # Scalar memref<T>: wrap in a 1-element dense.
            from xdsl.dialects.builtin import TensorType

            elem = target_ty.get_element_type()
            assert isinstance(elem, IntegerType)
            tensor_ty = TensorType(elem, [])
            init_attr = DenseIntOrFPElementsAttr.create_dense_int(
                tensor_ty, [attr.value.value.data]
            )
        elif isa(attr, cir.CIRFPAttr):
            from xdsl.dialects.builtin import TensorType

            elem = target_ty.get_element_type()
            assert isinstance(elem, AnyFloat)
            tensor_ty = TensorType(elem, [])
            init_attr = DenseIntOrFPElementsAttr.create_dense_float(
                tensor_ty, [attr.value.value.data]
            )
        elif isa(attr, cir.ConstArrayAttr):
            init_attr = _const_array_to_dense(program_state, attr, target_ty)
        elif isa(attr, cir.ConstPtrAttr):
            # NULL-initialised pointer global. Treat as zero-init at the
            # memref level — this means the pointer slot starts as a
            # dangling memref descriptor, but `cir.load` on it is UB in C
            # and we don't lower it.
            init_attr = UnitAttr()
        else:
            raise NotImplementedError(
                f"cir-to-core: global initialiser {type(attr).__name__}"
            )

    sym_visibility = "private" if op.sym_visibility is not None and op.sym_visibility.data == "private" else "public"
    constant = op.constant is not None

    if init_attr is None:
        # `memref.global` requires an initial_value. For uninitialised globals
        # (extern decl) we still need something; use UnitAttr to mark "no
        # initialiser, just declaration" — equivalent to MLIR's `external` form.
        init_attr = UnitAttr()

    new = memref.GlobalOp.get(
        sym,
        target_ty,
        initial_value=init_attr,
        sym_visibility=StringAttr(sym_visibility),
        constant=UnitAttr() if constant else None,
    )
    return [new]


# Silence unused-import warnings for placeholder helpers.
_ = (DenseArrayBase, i32, i64)
