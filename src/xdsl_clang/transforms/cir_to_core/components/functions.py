"""Phase 2e — `cir.func`, `cir.call`, `cir.return` → `func` dialect.

Function bodies are translated by walking each block and dispatching every
op through `statements.translate_stmt`. Block arguments are pre-mapped:
`SSAValueCtx[<cir-block-arg>] = <core-block-arg>`, so the dispatcher can
resolve them when handlers look up `ctx[op.lhs]`.
"""

from __future__ import annotations

from xdsl.dialects import arith, func, llvm, memref
from xdsl.dialects.builtin import FunctionType, IntegerType, MemRefType
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
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
) -> tuple[list[Attribute], list[Attribute]]:
    inputs: list[Attribute] = []
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
    results: list[Attribute] = []
    if not func_type.has_void_return:
        results.append(
            convert_cir_type_to_standard(
                func_type.return_type, program_state, ptr_mode=SCALAR_PTR
            )
        )
    return inputs, results


def _has_break_or_continue_anywhere(op: cir.FuncOp) -> bool:
    """True iff the function body contains a `cir.break` or `cir.continue`
    at any nesting depth — the trigger for unstructured emission (Task 5.7)."""
    for inner in op.walk():
        if isa(inner, cir.BreakOp) or isa(inner, cir.ContinueOp):
            return True
    return False


def _has_nested_return(op: cir.FuncOp) -> bool:
    """True iff a `cir.return` appears inside any *nested* region of the
    function body (e.g. inside a `cir.if` or `cir.scope`). Returns at the
    function body's block scope are fine — the structured emitter handles
    those. Nested returns force the unstructured emitter (Task 5.1)."""
    for block in op.body.blocks:
        for top_op in block.ops:
            for region in top_op.regions:
                for nested_block in region.blocks:
                    for nested_op in nested_block.walk():
                        if isa(nested_op, cir.ReturnOp):
                            return True
    return False


def translate_function(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.FuncOp
) -> Operation | None:
    sym = op.sym_name.data
    # `malloc` / `calloc` / `free` are folded into `memref.alloc` /
    # `memref.dealloc` at the cast/call sites; never emit `func.func`
    # declarations for them.
    if sym in ("malloc", "calloc", "free"):
        return None

    is_extern = len(op.body.blocks) == 0
    is_var_arg = bool(op.function_type.varargs)
    inputs, results = _convert_func_signature(
        op.function_type, program_state, is_extern=is_extern
    )
    fn_type = FunctionType.from_lists(inputs, results)

    if is_extern:
        # External declaration. Pointer args on extern decls are `!llvm.ptr`
        # to match the plain C ABI; call sites bridge from memref descriptors
        # at the boundary.
        if is_var_arg:
            # Variadic externs (e.g. `printf(const char*, ...)`) can't be
            # modelled with `func.func` because the `func` dialect has no
            # variadic concept. Emit an `llvm.func` declaration so call
            # sites can use `llvm.call` with a per-call var-callee-type.
            ret_for_llvm = results[0] if results else llvm.LLVMVoidType()
            llvm_fty = llvm.LLVMFunctionType(inputs, ret_for_llvm, True)
            return llvm.FuncOp(
                sym,
                llvm_fty,
                linkage=llvm.LinkageAttr("external"),
            )
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
    fn_state = program_state.getCurrentFnState()
    setattr(program_state, "_block_map", block_map)
    fn_state.function_region = new_region
    fn_state.entry_block = block_map[op.body.blocks[0]] if op.body.blocks else None
    fn_state.is_unstructured = _has_break_or_continue_anywhere(
        op
    ) or _has_nested_return(op)
    try:
        from xdsl_clang.transforms.cir_to_core import statements

        for old_block in op.body.blocks:
            fn_state.current_block = block_map[old_block]
            fn_state.block_terminated = False
            for body_op in list(old_block.ops):
                # In unstructured mode a prior handler may have emitted a
                # terminator (cf.br / cf.cond_br / func.return) into the
                # current block. Any remaining CIR ops in this source block
                # are dead — typically a trailing `cir.yield`. Drop them.
                if fn_state.is_unstructured and fn_state.block_terminated:
                    continue
                for new_op in statements.translate_stmt(program_state, fn_ctx, body_op):
                    if fn_state.current_block is None:  # type: ignore[reportUnnecessaryComparison]
                        raise RuntimeError(
                            "cir-to-core: current_block became None during "
                            "function translation"
                        )
                    fn_state.current_block.add_op(new_op)
    finally:
        delattr(program_state, "_block_map")
        fn_state.function_region = None
        fn_state.current_block = None
        fn_state.entry_block = None
        program_state.leaveFunction()

    visibility = (
        "private"
        if op.sym_visibility is not None and op.sym_visibility.data == "private"
        else None
    )
    new_func = func.FuncOp(sym, fn_type, region=new_region, visibility=visibility)
    return new_func


def translate_call(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.CallOp
) -> list[Operation]:
    if op.callee is None:
        raise NotImplementedError(
            "cir-to-core: indirect cir.call lowering not yet implemented"
        )

    callee_sym = op.callee.root_reference.data

    # Phase 5 Task 5.4 + F1: pattern-match the malloc/calloc/free idiom.
    if callee_sym in ("malloc", "calloc"):
        # The result of `cir.call @malloc` / `@calloc` is consumed by a
        # subsequent `cir.cast bitcast` to `!cir.ptr<T>`; that cast
        # handler emits the `memref.alloc` directly. Emit nothing here —
        # the cast walks back through the call.
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
    # Variadic externs are declared as `llvm.func`; their call sites must
    # use `llvm.call` so the trailing variadic args don't trip the func
    # verifier's strict arity check.
    is_variadic_callee = is_extern_callee and fn_def.is_var_arg

    pre_ops: list[Operation] = []
    args: list[SSAValue] = []
    for arg_idx, a in enumerate(op.arg_ops):
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
            continue
        # Task F3: internal-call shape lift. `helper(&local)` passes a
        # rank-0 `memref<T>` (address-of-scalar) to a callee whose `T*`
        # parameter lowers to rank-1 `memref<?xT>` under the decayed-pointer
        # convention. Insert a `memref.cast` so the static rank lifts to
        # dynamic. Same fix applies to passing `memref<memref<?xT>>`
        # (rank-0 pointer slot) for a `memref<?xmemref<?xT>>` parameter
        # (e.g. `dswap(&u, &unew)` in swm_orig.c).
        if not is_extern_callee and arg_idx < len(fn_def.args):
            formal_cir = fn_def.args[arg_idx].cir_type
            mode = DECAYED_PTR if isa(formal_cir, cir.PointerType) else SCALAR_PTR
            formal_ty = convert_cir_type_to_standard(
                formal_cir, program_state, ptr_mode=mode
            )
            if (
                isa(val.type, MemRefType)
                and isa(formal_ty, MemRefType)
                and val.type != formal_ty
                and val.type.get_element_type() == formal_ty.get_element_type()
            ):
                src_rank = val.type.get_num_dims()
                dst_rank = formal_ty.get_num_dims()
                if src_rank == dst_rank:
                    # Same rank, only static-vs-dynamic shape differs —
                    # `memref.cast` is the right shape lift.
                    shape_lift = memref.CastOp.get(val, formal_ty)
                    pre_ops.append(shape_lift)
                    args.append(shape_lift.results[0])
                    continue
                if src_rank == 0 and dst_rank == 1:
                    # Address-of-scalar passed as a decayed pointer arg:
                    # lift rank-0 `memref<T>` → rank-1 `memref<?xT>` of
                    # length 1. `memref.cast` rejects rank changes, so we
                    # go through `memref.reinterpret_cast` (rank-0 →
                    # `memref<1xT>`) and then `memref.cast` (static →
                    # dynamic shape).
                    static_ty = MemRefType(formal_ty.get_element_type(), [1])
                    re_cast = memref.ReinterpretCastOp.from_dynamic(
                        source=val,
                        offsets=[0],
                        sizes=[1],
                        strides=[1],
                        result_type=static_ty,
                    )
                    shape_lift = memref.CastOp.get(re_cast.results[0], formal_ty)
                    pre_ops.extend([re_cast, shape_lift])
                    args.append(shape_lift.results[0])
                    continue
        args.append(val)

    result_types: list[Attribute] = []
    if fn_def.return_type is not None:
        result_types.append(
            convert_cir_type_to_standard(fn_def.return_type, program_state)
        )

    if is_variadic_callee:
        # For variadic externs, count the variadic-tail by subtracting the
        # number of fixed parameters declared on the callee. The fixed-arg
        # count is the length of the declared formal-args list.
        n_fixed = len(fn_def.args)
        n_variadic = max(0, len(args) - n_fixed)
        ret_for_call: Attribute | None = result_types[0] if result_types else None
        new_call = llvm.CallOp(
            callee_sym,
            *args,
            return_type=ret_for_call,
            variadic_args=n_variadic,
        )
        # Always populate `var_callee_type` for variadic callees — xdsl's
        # builder skips it when `variadic_args==0`, but mlir-opt rejects
        # the call without it (e.g. `printf("hi")`).
        if new_call.var_callee_type is None:
            fixed_input_types = [SSAValue.get(args[i]).type for i in range(n_fixed)]
            void_or_ret: Attribute = (
                ret_for_call if ret_for_call is not None else llvm.LLVMVoidType()
            )
            new_call.properties["var_callee_type"] = llvm.LLVMFunctionType(
                fixed_input_types, void_or_ret, True
            )
        if result_types:
            ctx[op.results[0]] = new_call.results[0]
        return [*pre_ops, new_call]

    new_call = func.CallOp(callee_sym, args, result_types)
    if result_types:
        ctx[op.results[0]] = new_call.results[0]
    return [*pre_ops, new_call]
