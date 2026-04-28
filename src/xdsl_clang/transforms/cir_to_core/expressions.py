"""Per-expression dispatcher: SSA-value-defining CIR ops → core ops.

Mirror of ``ftn/transforms/to_core/expressions.py``. Each component module
registers its handlers here via the ``elif isa(op, …)`` chain. Phase 1
ships only the dispatch shell; Phase 2 fills in the components.
"""

from __future__ import annotations

from xdsl.ir import BlockArgument, Operation, SSAValue
from xdsl.utils.hints import isa

from xdsl_clang.transforms.cir_to_core.misc.c_code_description import ProgramState
from xdsl_clang.transforms.cir_to_core.misc.ssa_context import SSAValueCtx


def translate_expr(
    program_state: ProgramState, ctx: SSAValueCtx, ssa_value: SSAValue
) -> list[Operation]:
    """Translate the op that defines `ssa_value` and record the mapping in
    `ctx`. Returns the list of new ops (in IR order). Block arguments
    (which have no defining op) translate to an empty list."""
    if isinstance(ssa_value, BlockArgument):
        return []
    owner = ssa_value.owner
    if not isinstance(owner, Operation):
        return []
    ops = try_translate_expr(program_state, ctx, owner)
    if ops is None:
        raise NotImplementedError(
            f"cir-to-core: no expression handler for {owner.name}"
        )
    return ops


def try_translate_expr(
    program_state: ProgramState, ctx: SSAValueCtx, op: Operation
) -> list[Operation] | None:
    """Component handlers are appended to this chain. Returns None when no
    handler matches (the caller decides whether that's an error)."""
    # Late imports keep the dispatcher a thin shell — components are free
    # to import each other transitively.
    from xdsl_clang.dialects import cir
    from xdsl_clang.transforms.cir_to_core.components import (
        casts as cir_casts,
        control_flow as cir_ctrl,
        functions as cir_functions,
        load_store as cir_load_store,
        maths as cir_maths,
        memory as cir_memory,
    )

    # --- Phase 2c: maths / cmp / select / const -------------------------
    if isa(op, cir.ConstantOp):
        return cir_maths.translate_constant(program_state, ctx, op)
    if isa(op, cir.BinOp):
        return cir_maths.translate_binop(program_state, ctx, op)
    if isa(op, cir.UnaryOp):
        return cir_maths.translate_unary(program_state, ctx, op)
    if isa(op, cir.CmpOp):
        return cir_maths.translate_cmp(program_state, ctx, op)
    if isa(op, cir.SelectOp):
        return cir_maths.translate_select(program_state, ctx, op)
    if isa(op, cir.TernaryOp):
        return cir_maths.translate_ternary(program_state, ctx, op)
    # --- Phase 2a: globals + memory ------------------------------------
    if isa(op, cir.AllocaOp):
        return cir_memory.translate_alloca(program_state, ctx, op)
    if isa(op, cir.GetGlobalOp):
        return cir_memory.translate_get_global(program_state, ctx, op)
    # --- Phase 2b: load/store/casts/records ----------------------------
    if isa(op, cir.LoadOp):
        return cir_load_store.translate_load(program_state, ctx, op)
    if isa(op, cir.CastOp):
        return cir_casts.translate_cast(program_state, ctx, op)
    if isa(op, cir.PtrStrideOp):
        return cir_load_store.translate_ptr_stride(program_state, ctx, op)
    if isa(op, cir.GetElementOp):
        return cir_load_store.translate_get_element(program_state, ctx, op)
    if isa(op, cir.GetMemberOp):
        return cir_load_store.translate_get_member(program_state, ctx, op)
    # --- Phase 2e: calls (when used as expressions, i.e. non-void) -----
    if isa(op, cir.CallOp):
        return cir_functions.translate_call(program_state, ctx, op)
    # --- Phase 2d: structured-region results ---------------------------
    if isa(op, cir.ScopeOp):
        return cir_ctrl.translate_scope(program_state, ctx, op)
    return None
