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
from xdsl.dialects.builtin import IndexType, IntegerType, MemRefType
from xdsl.ir import Operation, SSAValue
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


# ---------------------------------------------------------------------------
# cir.load
# ---------------------------------------------------------------------------


def translate_load(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.LoadOp
) -> list[Operation]:
    addr_cir = op.addr
    addr = _opd(ctx, addr_cir)
    chain = _index_chain(program_state).get(addr_cir, [])

    # Pointer-to-record: emit llvm.load.
    pointee = addr_cir.type.pointee  # type: ignore[attr-defined]
    if isa(pointee, cir.RecordType):
        struct_ty = convert_cir_type_to_standard(pointee, program_state)
        new = llvm.LoadOp(addr, struct_ty)
        ctx[op.results[0]] = new.results[0]
        return [new]
    if isa(pointee, cir.PointerType):
        # Loading a pointer through a pointer (e.g., `int *p` where p was
        # spilled to the stack). The lowered addr is `memref<memref<...>>`
        # in principle, but our convention flattens all stack ptr slots to
        # a single memref of the pointee's element type. We treat it as a
        # straight memref.load.
        inner_ptr_ty = convert_cir_type_to_standard(
            pointee, program_state
        )
        new_ld = memref.LoadOp.get(addr, [])
        ctx[op.results[0]] = new_ld.results[0]
        return [new_ld]
    # Scalar / array element load.
    if not chain:
        # No accumulated indices — scalar pointer (`memref<T>` is rank-0).
        new = memref.LoadOp.get(addr, [])
        ctx[op.results[0]] = new.results[0]
        return [new]
    # Indexed load: chain holds the index SSA values in order.
    new_l = memref.LoadOp.get(addr, list(chain))
    ctx[op.results[0]] = new_l.results[0]
    return [new_l]


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
    if isa(pointee, cir.RecordType):
        new = llvm.StoreOp(val, addr)
        return [new]
    if isa(pointee, cir.PointerType):
        new_s = memref.StoreOp.get(val, addr, [])
        return [new_s]
    if not chain:
        new_s = memref.StoreOp.get(val, addr, [])
        return [new_s]
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
        raise NotImplementedError(
            "cir-to-core: cir.get_member on non-record pointer"
        )
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
