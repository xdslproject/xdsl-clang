"""Phase 2b — load/store, pointer arithmetic, record member access.

Per Decision 1 of the plan, pointers lower to:

* scalar pointer        →  `memref<T>`
* array-decayed pointer →  `memref<?xT>`
* record / function ptr →  `!llvm.ptr` (opaque)

Pointer arithmetic (`cir.ptr_stride`) and array indexing (`cir.get_element`)
*don't* emit IR ops on their own — they push an `index`-typed SSA value
onto a per-function index chain. The eventual `cir.load` / `cir.store`
consumes the chain to build the right `memref.load[%i]` / `memref.store`.

Record member access (`cir.get_member`) goes through the LLVM dialect
per Decision 3 / `structs.md`.
"""

from __future__ import annotations

from xdsl.dialects import arith, llvm, memref
from xdsl.dialects.builtin import IndexType, IntegerAttr, MemRefType
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.utils.hints import isa

from xdsl_clang.dialects import cir
from xdsl_clang.transforms.cir_to_core.components.cir_types import (
    convert_cir_type_to_standard,
)
from xdsl_clang.transforms.cir_to_core.misc.c_code_description import ProgramState
from xdsl_clang.transforms.cir_to_core.misc.ssa_context import SSAValueCtx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opd(ctx: SSAValueCtx, value: SSAValue) -> SSAValue:
    mapped = ctx[value]
    return mapped if mapped is not None else value


def _index_chain(program_state: ProgramState) -> dict[object, list[SSAValue]]:
    return program_state.getCurrentFnState().index_chain  # type: ignore[return-value]


def _to_index(value: SSAValue) -> tuple[list[Operation], SSAValue]:
    if isinstance(value.type, IndexType):
        return [], value
    cast = arith.IndexCastOp(value, IndexType())
    return [cast], cast.results[0]


def _index_zero() -> tuple[list[Operation], SSAValue]:
    """Materialise an `index`-typed constant 0.

    Used for `*p` / `cir.load %p` and `cir.store v, %p` where `%p` is a
    pointer that lowers to a rank-1 `memref<?xT>` (e.g. a function argument
    or a loaded pointer slot) with no preceding `cir.ptr_stride` — i.e.
    semantically `p[0]`.
    """
    c = arith.ConstantOp(IntegerAttr(0, IndexType()))
    return [c], c.results[0]


# ---------------------------------------------------------------------------
# cir.load
# ---------------------------------------------------------------------------


def translate_load(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.LoadOp
) -> list[Operation]:
    addr_cir = op.addr
    addr = _opd(ctx, addr_cir)
    chain_table = _index_chain(program_state)
    chain = chain_table.get(addr_cir, [])

    # Address is opaque `!llvm.ptr` — produced by `cir.alloca` of a record
    # (whole-record load), `cir.get_member` (scalar/pointer field GEP), or
    # `llvm.mlir.zero` for a null record/function/void* pointer. memref.load
    # cannot index into !llvm.ptr, so fall back to llvm.load with the result
    # type derived from the CIR result type.
    pointee = addr_cir.type.pointee  # type: ignore[attr-defined]
    if isa(addr.type, llvm.LLVMPointerType):
        result_ty = convert_cir_type_to_standard(op.results[0].type, program_state)
        new = llvm.LoadOp(addr, result_ty)
        ctx[op.results[0]] = new.results[0]
        chain_table[op.results[0]] = []
        return [new]
    if isa(pointee, cir.PointerType):
        # Loading a pointer through a pointer. Two sub-cases for the address:
        #   * rank-0 `memref<memref<...>>` — a spilled-pointer slot; load
        #     with no indices yields the inner descriptor.
        #   * rank-1 `memref<?xmemref<...>>` — a decayed pointer-of-pointer
        #     used directly as `*pp` (i.e. `pp[0]`); materialise index 0
        #     just like the scalar case.
        # The chain branch (e.g. `pp[i]`) goes through `_scalar_load_store_indices`
        # implicitly via the chain path below.
        if not chain:
            prelude, indices = _scalar_load_store_indices(addr)
            new_ld = memref.LoadOp.get(addr, indices)
            ctx[op.results[0]] = new_ld.results[0]
            # IMPORTANT: the loaded pointer is a fresh base for any subsequent
            # `cir.ptr_stride` — see Phase 5 Task 5.6.
            chain_table[op.results[0]] = []
            return [*prelude, new_ld]
        new_ld = memref.LoadOp.get(addr, list(chain))
        ctx[op.results[0]] = new_ld.results[0]
        chain_table[op.results[0]] = []
        return [new_ld]
    # Scalar / array element load.
    if not chain:
        # No accumulated indices. Two sub-cases:
        #   * `addr` lowers to `memref<T>` (rank-0) — a true scalar slot.
        #     Emit `memref.load %addr[]`.
        #   * `addr` lowers to `memref<?xT>` (rank-1) — a decayed pointer
        #     used directly as `*p` (i.e. `p[0]`). Materialise an index 0.
        prelude, indices = _scalar_load_store_indices(addr)
        new = memref.LoadOp.get(addr, indices)
        ctx[op.results[0]] = new.results[0]
        return [*prelude, new]
    # Indexed load: chain holds the index SSA values in order.
    new_l = memref.LoadOp.get(addr, list(chain))
    ctx[op.results[0]] = new_l.results[0]
    return [new_l]


def _scalar_load_store_indices(
    addr: SSAValue,
) -> tuple[list[Operation], list[SSAValue]]:
    """Return `(prelude_ops, indices)` for a scalar-style load/store.

    When `addr` is a rank-1 `memref<?xT>` but no `cir.ptr_stride` chain
    has been pushed, the source-level construct is `*p` — i.e. `p[0]`.
    We materialise an `index 0` to match the memref rank. For rank-0
    `memref<T>`, the index list is empty.
    """
    addr_ty = addr.type
    if isinstance(addr_ty, MemRefType) and addr_ty.get_num_dims() == 1:
        ops, idx = _index_zero()
        return ops, [idx]
    return [], []


# ---------------------------------------------------------------------------
# cir.store
# ---------------------------------------------------------------------------


def translate_store(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.StoreOp
) -> list[Operation]:
    addr_cir = op.addr
    addr = _opd(ctx, addr_cir)
    val = _opd(ctx, op.value)
    chain = _index_chain(program_state).get(addr_cir, [])

    pointee = addr_cir.type.pointee  # type: ignore[attr-defined]
    # Whole-array store: the source SSA value is itself a memref of the
    # same shape as the destination slot. Happens when a `cir.const` of
    # array type was hoisted to a `memref.global` (Task 5.2). MLIR's
    # `memref.store` cannot store a whole memref, so emit `memref.copy`
    # from the source memref into the destination memref.
    if (
        isa(val.type, MemRefType[Attribute])
        and isa(addr.type, MemRefType[Attribute])
        and val.type.get_shape() == addr.type.get_shape()
        and val.type.get_element_type() == addr.type.get_element_type()
    ):
        return [memref.CopyOp(val, addr)]
    # Address is opaque `!llvm.ptr` — record store, GEP-into-struct field,
    # or null-pointer slot. Mirror the load path and emit llvm.store.
    if isa(addr.type, llvm.LLVMPointerType):
        new = llvm.StoreOp(val, addr)
        return [new]
    if isa(pointee, cir.PointerType):
        # Storing a pointer through a pointer slot (e.g., `*pp = q`).
        # Same rank-0 vs rank-1 distinction as the load path: a rank-1
        # decayed `memref<?xmemref<...>>` destination needs an index 0.
        if not chain:
            prelude, indices = _scalar_load_store_indices(addr)
            new_s = memref.StoreOp.get(val, addr, indices)
            return [*prelude, new_s]
        new_s = memref.StoreOp.get(val, addr, list(chain))
        return [new_s]
    if not chain:
        # See `_scalar_load_store_indices` — a rank-1 destination with no
        # chain means `*p = v` (i.e. `p[0] = v`).
        prelude, indices = _scalar_load_store_indices(addr)
        new_s = memref.StoreOp.get(val, addr, indices)
        return [*prelude, new_s]
    new_s = memref.StoreOp.get(val, addr, list(chain))
    return [new_s]


# ---------------------------------------------------------------------------
# cir.ptr_stride — push an index onto the chain, no IR emitted (besides
# the index cast).
# ---------------------------------------------------------------------------


def translate_ptr_stride(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.PtrStrideOp
) -> list[Operation]:
    base_cir = op.base
    base = _opd(ctx, base_cir)
    stride = _opd(ctx, op.stride)
    cast_ops, idx = _to_index(stride)

    chain_table = _index_chain(program_state)
    parent_chain = chain_table.get(base_cir, [])
    chain_table[op.results[0]] = list(parent_chain) + [idx]
    ctx[op.results[0]] = base
    return cast_ops


# ---------------------------------------------------------------------------
# cir.get_element — same idea as ptr_stride
# ---------------------------------------------------------------------------


def translate_get_element(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.GetElementOp
) -> list[Operation]:
    base_cir = op.base
    base = _opd(ctx, base_cir)
    idx_v = _opd(ctx, op.index)
    cast_ops, idx = _to_index(idx_v)

    chain_table = _index_chain(program_state)
    parent_chain = chain_table.get(base_cir, [])
    chain_table[op.results[0]] = list(parent_chain) + [idx]
    ctx[op.results[0]] = base
    return cast_ops


# ---------------------------------------------------------------------------
# cir.get_member — record field access via llvm.getelementptr. Per
# Decision 3, records live in `!llvm.struct` and are addressed with GEP
# `[0, fieldIdx]` returning `!llvm.ptr`.
# ---------------------------------------------------------------------------


def translate_get_member(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.GetMemberOp
) -> list[Operation]:
    base = _opd(ctx, op.addr)
    # Determine the element type of the surrounding pointer for GEP.
    pointee = op.addr.type.pointee  # type: ignore[attr-defined]
    if not isa(pointee, cir.RecordType):
        raise NotImplementedError("cir-to-core: cir.get_member on non-record pointer")
    elem = convert_cir_type_to_standard(pointee, program_state)
    field_idx = op.index_attr.value.data
    gep = llvm.GEPOp(
        ptr=base,
        indices=[0, field_idx],
        pointee_type=elem,
        result_type=llvm.LLVMPointerType(),
    )
    ctx[op.results[0]] = gep.results[0]
    return [gep]
