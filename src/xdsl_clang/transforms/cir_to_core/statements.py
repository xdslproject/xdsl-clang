"""Per-statement dispatcher: void-result CIR ops → core ops.

Mirror of ``ftn/transforms/to_core/statements.py``. Each component
registers its handlers via the ``elif isa(op, …)`` chain. Phase 1 ships
only the shell; Phase 2 fills handlers in.
"""

from __future__ import annotations

from xdsl.ir import Operation
from xdsl.utils.hints import isa

from xdsl_clang.transforms.cir_to_core.misc.c_code_description import ProgramState
from xdsl_clang.transforms.cir_to_core.misc.ssa_context import SSAValueCtx


def translate_stmt(
    program_state: ProgramState, ctx: SSAValueCtx, op: Operation
) -> list[Operation]:
    ops = try_translate_stmt(program_state, ctx, op)
    if ops is None:
        raise NotImplementedError(f"cir-to-core: no statement handler for {op.name}")
    return ops


def try_translate_stmt(
    program_state: ProgramState, ctx: SSAValueCtx, op: Operation
) -> list[Operation] | None:
    # Late imports — see expressions.py for the rationale.
    from xdsl_clang.dialects import cir
    from xdsl_clang.transforms.cir_to_core import expressions
    from xdsl_clang.transforms.cir_to_core.components import (
        control_flow as cir_ctrl,
    )
    from xdsl_clang.transforms.cir_to_core.components import (
        functions as cir_functions,
    )
    from xdsl_clang.transforms.cir_to_core.components import (
        load_store as cir_load_store,
    )
    from xdsl_clang.transforms.cir_to_core.components import (
        memory as cir_memory,
    )

    # --- pure-expression CIR ops also appear as block-level statements
    # in CIR's textual form (alloca, const, binop, …) — defer to
    # translate_expr, which records the SSA mapping. The op produces
    # no new statement on its own; the new ops are returned and the
    # outer block-walk inserts them in order.
    EXPR_AS_STMT = (
        cir.AllocaOp,
        cir.ConstantOp,
        cir.BinOp,
        cir.UnaryOp,
        cir.CmpOp,
        cir.SelectOp,
        cir.TernaryOp,
        cir.LoadOp,
        cir.CastOp,
        cir.PtrStrideOp,
        cir.GetElementOp,
        cir.GetMemberOp,
        cir.GetGlobalOp,
    )
    if isinstance(op, EXPR_AS_STMT) and len(op.results) > 0:
        return expressions.translate_expr(program_state, ctx, op.results[0])

    # --- Phase 2a: stores into memory ----------------------------------
    if isa(op, cir.StoreOp):
        return cir_load_store.translate_store(program_state, ctx, op)
    # --- Phase 2a: alloca-only memory bookkeeping ----------------------
    if isa(op, cir.GlobalOp):
        return cir_memory.translate_global(program_state, ctx, op)
    # --- Phase 2d: control flow ----------------------------------------
    if isa(op, cir.IfOp):
        return cir_ctrl.translate_if(program_state, ctx, op)
    if isa(op, cir.ScopeOp):
        return cir_ctrl.translate_scope(program_state, ctx, op)
    if isa(op, cir.ForOp):
        return cir_ctrl.translate_for(program_state, ctx, op)
    if isa(op, cir.WhileOp):
        return cir_ctrl.translate_while(program_state, ctx, op)
    if isa(op, cir.DoWhileOp):
        return cir_ctrl.translate_dowhile(program_state, ctx, op)
    if isa(op, cir.SwitchOp):
        return cir_ctrl.translate_switch(program_state, ctx, op)
    if isa(op, cir.SwitchFlatOp):
        return cir_ctrl.translate_switch_flat(program_state, ctx, op)
    if isa(op, cir.YieldOp):
        return cir_ctrl.translate_yield(program_state, ctx, op)
    if isa(op, cir.ConditionOp):
        return cir_ctrl.translate_condition(program_state, ctx, op)
    if isa(op, cir.BreakOp):
        return cir_ctrl.translate_break(program_state, ctx, op)
    if isa(op, cir.ContinueOp):
        return cir_ctrl.translate_continue(program_state, ctx, op)
    if isa(op, cir.BrOp):
        return cir_ctrl.translate_br(program_state, ctx, op)
    if isa(op, cir.BrCondOp):
        return cir_ctrl.translate_brcond(program_state, ctx, op)
    if isa(op, cir.ReturnOp):
        return cir_ctrl.translate_return(program_state, ctx, op)
    # --- Phase 2e: void calls + function decls -------------------------
    if isa(op, cir.CallOp):
        return cir_functions.translate_call(program_state, ctx, op)
    return None
