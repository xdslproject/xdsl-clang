"""Phase 2e — `cir.func`, `cir.call`, `cir.return` → `func` dialect.

Function bodies are translated by walking each block and dispatching every
op through `statements.translate_stmt`. Block arguments are pre-mapped:
`SSAValueCtx[<cir-block-arg>] = <core-block-arg>`, so the dispatcher can
resolve them when handlers look up `ctx[op.lhs]`.
"""

from __future__ import annotations

from xdsl.dialects import arith, func, llvm, memref
from xdsl.dialects.builtin import FunctionType, IndexType, IntegerType, MemRefType
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.utils.hints import isa

from xdsl_clang.dialects import cir
from xdsl_clang.transforms.cir_to_core.components.cir_types import (
    DECAYED_PTR,
    SCALAR_PTR,
    convert_cir_type_to_standard,
)
from xdsl_clang.transforms.cir_to_core.misc.c_code_description import ProgramState
from xdsl_clang.transforms.cir_to_core.misc.ssa_context import SSAValueCtx


def _convert_func_signature(
    func_type: cir.FuncType,
    program_state: ProgramState,
    *,
    is_extern: bool = False,
) -> tuple[list, list]:
    inputs: list = []
    for arg_ty in func_type.inputs.data:
        if is_extern and isa(arg_ty, cir.PointerType):
            # Extern declarations follow the plain C ABI: every pointer
            # argument lowers to opaque `!llvm.ptr`, regardless of pointee.
            # Call sites materialise a memref→`!llvm.ptr` conversion
            # (memref.extract_aligned_pointer_as_index + arith.index_cast +
            # llvm.inttoptr) so the descriptor convention used internally
            # collapses at the boundary.
            inputs.append(llvm.LLVMPointerType())
            continue
        # Pointer args coming in from the C ABI are array-decayed by
        # convention — Clang emits `T*` for what was an array on the C
        # side, so use the dynamic memref form.
        mode = DECAYED_PTR if isa(arg_ty, cir.PointerType) else SCALAR_PTR
        inputs.append(
            convert_cir_type_to_standard(arg_ty, program_state, ptr_mode=mode)
        )
    results: list = []
    if not func_type.has_void_return:
        results.append(
            convert_cir_type_to_standard(
                func_type.return_type, program_state, ptr_mode=SCALAR_PTR
            )
        )
    return inputs, results


def translate_function(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.FuncOp
) -> Operation | None:
    sym = op.sym_name.data
    # `malloc` / `free` are folded into `memref.alloc` / `memref.dealloc`
    # at the cast/call sites; never emit `func.func` declarations for them.
    if sym in ("malloc", "free"):
        return None

    is_extern = len(op.body.blocks) == 0
    inputs, results = _convert_func_signature(
        op.function_type, program_state, is_extern=is_extern
    )
    fn_type = FunctionType.from_lists(inputs, results)

    if is_extern:
        # External declaration → `func.func` with empty region and `private`
        # visibility (xdsl's `FuncOp.external` builder does this). Pointer
        # args on extern decls are `!llvm.ptr` to match the plain C ABI;
        # call sites bridge from memref descriptors at the boundary.
        return func.FuncOp.external(sym, inputs, results)

    program_state.enterFunction(sym)
    fn_ctx = SSAValueCtx(parent_scope=ctx)

    new_region = Region()
    # Translate each block. CIR uses block-local SSA, so we mirror the
    # structure 1:1 — one new block per old block, with arg types
    # converted via the signature.
    block_map: dict[Block, Block] = {}
    for old_block in op.body.blocks:
        arg_types = [
            convert_cir_type_to_standard(
                a.type,
                program_state,
                ptr_mode=DECAYED_PTR if isa(a.type, cir.PointerType) else SCALAR_PTR,
            )
            for a in old_block.args
        ]
        new_block = Block(arg_types=arg_types)
        block_map[old_block] = new_block
        new_region.add_block(new_block)
        for old_arg, new_arg in zip(old_block.args, new_block.args):
            fn_ctx[old_arg] = new_arg

    # Now translate ops; we need block_map visible to control-flow
    # handlers, so park it on `program_state` for the duration.
    program_state.getCurrentFnState()
    setattr(program_state, "_block_map", block_map)
    try:
        from xdsl_clang.transforms.cir_to_core import statements

        for old_block in op.body.blocks:
            new_block = block_map[old_block]
            for body_op in list(old_block.ops):
                for new_op in statements.translate_stmt(
                    program_state, fn_ctx, body_op
                ):
                    new_block.add_op(new_op)
    finally:
        delattr(program_state, "_block_map")
        program_state.leaveFunction()

    visibility = "private" if op.sym_visibility is not None and op.sym_visibility.data == "private" else None
    new_func = func.FuncOp(
        sym, fn_type, region=new_region, visibility=visibility
    )
    return new_func


def translate_call(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.CallOp
) -> list[Operation]:
    if op.callee is None:
        raise NotImplementedError(
            "cir-to-core: indirect cir.call lowering not yet implemented"
        )

    callee_sym = op.callee.root_reference.data

    # Phase 5 Task 5.4: pattern-match the malloc/free idiom.
    if callee_sym == "malloc":
        # The result of `cir.call @malloc` is consumed by a subsequent
        # `cir.cast bitcast` to `!cir.ptr<T>`; that cast handler emits the
        # `memref.alloc` directly. Emit nothing here — the cast will walk
        # back through the call. Mark the result as un-mapped so that an
        # accidental use of the void* result outside of the bitcast
        # surfaces as a clear error rather than a silent miscompile.
        return []

    if callee_sym == "free":
        # `cir.call @free(%void_ptr)` where `%void_ptr` came from a
        # `cir.cast bitcast %T_ptr -> !cir.ptr<!cir.void>`. Walk back
        # through the cast to find the typed memref, drop the cast, and
        # emit `memref.dealloc` on the typed value.
        from xdsl.dialects import memref as memref_dialect

        void_arg = op.arg_ops[0]
        owner = void_arg.owner
        # Resolve the underlying typed pointer: skip the bitcast that
        # produced %void_ptr if there is one.
        target = void_arg
        if isa(owner, cir.CastOp):
            kind = cir._CAST_KIND_INV[owner.kind.value.data]  # type: ignore[attr-defined]
            if kind == "bitcast":
                target = owner.src
        mapped = ctx[target]
        if mapped is None:
            mapped = target
        return [memref_dialect.DeallocOp.get(mapped)]

    fn_def = program_state.function_definitions.get(callee_sym)
    if fn_def is None:
        raise NotImplementedError(
            f"cir-to-core: call to unknown function {callee_sym!r}"
        )
    is_extern_callee = fn_def.is_definition_only

    pre_ops: list[Operation] = []
    args: list[SSAValue] = []
    for a in op.arg_ops:
        mapped = ctx[a]
        val = mapped if mapped is not None else a
        if is_extern_callee and isa(val.type, MemRefType):
            # Extern decl signature has `!llvm.ptr` here — bridge the memref
            # descriptor down to its aligned base pointer. The combination
            # `memref.extract_aligned_pointer_as_index` + `arith.index_cast`
            # + `llvm.inttoptr` collapses cleanly under
            # `finalize-memref-to-llvm` and `reconcile-unrealized-casts`.
            extract = memref.ExtractAlignedPointerAsIndexOp.get(val)
            cast = arith.IndexCastOp(extract.results[0], IntegerType(64))
            to_ptr = llvm.IntToPtrOp(cast.results[0])
            pre_ops.extend([extract, cast, to_ptr])
            args.append(to_ptr.results[0])
        else:
            args.append(val)

    result_types = []
    if fn_def.return_type is not None:
        result_types.append(
            convert_cir_type_to_standard(fn_def.return_type, program_state)
        )
    new_call = func.CallOp(callee_sym, args, result_types)
    if result_types:
        ctx[op.results[0]] = new_call.results[0]
    return [*pre_ops, new_call]
