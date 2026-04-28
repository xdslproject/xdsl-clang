"""Phase 2d — control flow.

Decision 2: structured lowering when no `cir.break` / `cir.continue`
appears inside the loop body, otherwise unstructured `cf.br` / `cf.cond_br`
block graph. The unstructured path is implemented in Phase 5; for now the
structured handlers raise on break/continue and we document the gap.
"""

from __future__ import annotations

from collections.abc import Iterable

from xdsl.dialects import arith, cf, func, scf
from xdsl.dialects.builtin import IndexType
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.utils.hints import isa

from xdsl_clang.dialects import cir
from xdsl_clang.transforms.cir_to_core.misc.c_code_description import ProgramState
from xdsl_clang.transforms.cir_to_core.misc.ssa_context import SSAValueCtx


def _todo(name: str) -> list[Operation]:
    raise NotImplementedError(
        f"cir-to-core: Phase 2d handler {name!r} not yet implemented"
    )


# ---------------------------------------------------------------------------
# Helpers — region translation
# ---------------------------------------------------------------------------


def _translate_region_into_block(
    program_state: ProgramState,
    ctx: SSAValueCtx,
    region: Region,
    target_block: Block,
    *,
    skip_terminator: bool = True,
) -> None:
    """Translate every op in `region.block` into `target_block`, in order,
    using `ctx` directly so the caller can resolve SSA values afterward.

    `skip_terminator` drops `cir.yield` / `cir.condition` (they're handled
    by the surrounding structured-cf op).
    """
    from xdsl_clang.transforms.cir_to_core import statements

    for old_op in list(region.block.ops):
        if skip_terminator and (
            isa(old_op, cir.YieldOp) or isa(old_op, cir.ConditionOp)
        ):
            continue
        for new_op in statements.translate_stmt(program_state, ctx, old_op):
            target_block.add_op(new_op)


def _has_break_or_continue(region: Region) -> bool:
    for block in region.blocks:
        for op in block.walk():
            if isa(op, cir.BreakOp) or isa(op, cir.ContinueOp):
                return True
    return False


def _yielded_values(region: Region, ctx: SSAValueCtx) -> list[SSAValue]:
    """Find the trailing `cir.yield` and resolve its operands via `ctx`."""
    last = list(region.block.ops)[-1] if region.block.ops else None
    if last is None or not isa(last, cir.YieldOp):
        return []
    out: list[SSAValue] = []
    for v in last.arguments:
        m = ctx[v]
        out.append(m if m is not None else v)
    return out


# ---------------------------------------------------------------------------
# cir.if — `scf.if`
# ---------------------------------------------------------------------------


def translate_if(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.IfOp
) -> list[Operation]:
    if _has_break_or_continue(op.then_region) or _has_break_or_continue(op.else_region):
        raise NotImplementedError(
            "cir-to-core: cir.if containing break/continue (Phase 5)"
        )
    cond = ctx[op.cond]
    if cond is None:
        cond = op.cond

    then_block = Block()
    _translate_region_into_block(program_state, ctx, op.then_region, then_block)
    then_block.add_op(scf.YieldOp())

    else_blocks: list[Block]
    if op.else_region.blocks:
        else_block = Block()
        _translate_region_into_block(program_state, ctx, op.else_region, else_block)
        else_block.add_op(scf.YieldOp())
        else_region = Region([else_block])
    else:
        else_region = Region([Block([scf.YieldOp()])])

    if_op = scf.IfOp(cond, [], Region([then_block]), else_region)
    return [if_op]


# ---------------------------------------------------------------------------
# cir.scope — inline the body into the parent block
# ---------------------------------------------------------------------------


def translate_scope(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.ScopeOp
) -> list[Operation]:
    # `cir.scope` introduces a lexical scope but no SSA boundary; we inline
    # its body into the surrounding block. If it produces results, the
    # trailing `cir.yield` carries them — wire those to `op.results_`.
    out: list[Operation] = []
    inner_ctx = SSAValueCtx(parent_scope=ctx)
    from xdsl_clang.transforms.cir_to_core import statements

    for body_op in list(op.body.block.ops):
        if isa(body_op, cir.YieldOp):
            yielded = list(body_op.arguments)
            if yielded:
                if len(yielded) != len(op.results_):
                    raise NotImplementedError(
                        "cir-to-core: cir.scope yield arity mismatch"
                    )
                for old_res, ssa in zip(op.results_, yielded):
                    mapped = inner_ctx[ssa]
                    ctx[old_res] = mapped if mapped is not None else ssa
            continue
        for new_op in statements.translate_stmt(program_state, inner_ctx, body_op):
            out.append(new_op)
    return out


# ---------------------------------------------------------------------------
# cir.for / cir.while — `scf.while`
# ---------------------------------------------------------------------------


def _build_while_loop(
    program_state: ProgramState,
    ctx: SSAValueCtx,
    cond_region: Region,
    body_region: Region,
    *,
    step_region: Region | None = None,
    body_first: bool = False,
) -> list[Operation]:
    """Common builder for `cir.while`, `cir.for`, and `cir.do`.

    - body_first=False, step_region=None  → `cir.while` (top-tested).
    - body_first=False, step_region!=None → `cir.for` (cond, body, step).
    - body_first=True,  step_region=None  → `cir.do`   (bottom-tested).
    """
    if (
        _has_break_or_continue(cond_region)
        or _has_break_or_continue(body_region)
        or (step_region is not None and _has_break_or_continue(step_region))
    ):
        raise NotImplementedError(
            "cir-to-core: loop with break/continue (Phase 5 unstructured form)"
        )

    # `scf.while` has signature: `before(args) -> i1, before-yield(args)` and
    # `after(args) -> after-yield(args)`. With no loop-carried values our
    # before block is just the cond eval, and after is body + step.
    before_block = Block()
    after_block = Block()

    # ---- before region ----
    before_ctx = SSAValueCtx(parent_scope=ctx)
    if body_first:
        # `cir.do` has no separate cond region; the cond is the last op of
        # the body. Building it up correctly requires more state — defer.
        raise NotImplementedError("cir-to-core: cir.do (Phase 5)")
    _translate_region_into_block(
        program_state,
        before_ctx,
        cond_region,
        before_block,
        skip_terminator=True,
    )
    # Resolve the condition value from the trailing `cir.condition`.
    cond_term = list(cond_region.block.ops)[-1] if cond_region.block.ops else None
    if cond_term is None or not isa(cond_term, cir.ConditionOp):
        raise NotImplementedError(
            "cir-to-core: malformed loop cond region (no cir.condition terminator)"
        )
    cond_val = before_ctx[cond_term.cond]
    if cond_val is None:
        raise RuntimeError(
            f"cir-to-core: loop cond op {cond_term.cond} unmapped after "
            f"translating cond region (keys={list(before_ctx.dictionary.keys())})"
        )
    before_block.add_op(scf.ConditionOp(cond_val))

    # ---- after region ----
    after_ctx = SSAValueCtx(parent_scope=ctx)
    _translate_region_into_block(
        program_state, after_ctx, body_region, after_block, skip_terminator=True
    )
    if step_region is not None:
        _translate_region_into_block(
            program_state, after_ctx, step_region, after_block, skip_terminator=True
        )
    after_block.add_op(scf.YieldOp())

    return [scf.WhileOp([], [], Region([before_block]), Region([after_block]))]


def translate_for(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.ForOp
) -> list[Operation]:
    return _build_while_loop(
        program_state,
        ctx,
        op.cond_region,
        op.body_region,
        step_region=op.step_region,
    )


def translate_while(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.WhileOp
) -> list[Operation]:
    return _build_while_loop(
        program_state, ctx, op.cond_region, op.body_region
    )


def translate_dowhile(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.DoWhileOp
) -> list[Operation]:
    return _todo("do-while")


# ---------------------------------------------------------------------------
# Region terminators handled by their parents — reaching these directly
# means a malformed CIR module.
# ---------------------------------------------------------------------------


def translate_yield(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.YieldOp
) -> list[Operation]:
    # Should normally be consumed by the parent (cir.if/scope/while/for/scope).
    # If we land here, it's at function scope — drop it.
    return []


def translate_condition(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.ConditionOp
) -> list[Operation]:
    return _todo("condition (top-level)")


def translate_break(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.BreakOp
) -> list[Operation]:
    return _todo("break (Phase 5)")


def translate_continue(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.ContinueOp
) -> list[Operation]:
    return _todo("continue (Phase 5)")


def _block_map(program_state: ProgramState) -> dict[Block, Block]:
    bm = getattr(program_state, "_block_map", None)
    if bm is None:
        raise RuntimeError(
            "cir-to-core: cir.br/brcond outside a function translator context"
        )
    return bm


def _lookup_block(program_state: ProgramState, old: Block) -> Block:
    bm = _block_map(program_state)
    if old not in bm:
        raise RuntimeError(
            f"cir-to-core: branch target {old} has no lowered counterpart"
        )
    return bm[old]


def _resolve_args(ctx: SSAValueCtx, ops: list[SSAValue]) -> list[SSAValue]:
    out: list[SSAValue] = []
    for v in ops:
        m = ctx[v]
        out.append(m if m is not None else v)
    return out


def translate_br(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.BrOp
) -> list[Operation]:
    target = _lookup_block(program_state, op.successor[0])
    args = _resolve_args(ctx, list(op.arguments))
    return [cf.BranchOp(target, *args)]


def translate_brcond(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.BrCondOp
) -> list[Operation]:
    cond = ctx[op.cond]
    if cond is None:
        cond = op.cond
    if len(op.successor) != 2:
        raise RuntimeError("cir.brcond must have exactly two successors")
    then_block = _lookup_block(program_state, op.successor[0])
    else_block = _lookup_block(program_state, op.successor[1])
    then_args = _resolve_args(ctx, list(op.dest_operands_true))
    else_args = _resolve_args(ctx, list(op.dest_operands_false))
    return [
        cf.ConditionalBranchOp(cond, then_block, then_args, else_block, else_args)
    ]


def translate_ternary(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.TernaryOp
) -> list[Operation]:
    # CIR's ternary carries two regions; lower as a `scf.if` that yields a
    # value, then thread the result back through `ctx`.
    if _has_break_or_continue(op.true_region) or _has_break_or_continue(
        op.false_region
    ):
        raise NotImplementedError(
            "cir-to-core: cir.ternary with break/continue (Phase 5)"
        )

    res_ty = op.results[0].type
    from xdsl_clang.transforms.cir_to_core.components.cir_types import (
        convert_cir_type_to_standard,
    )

    lowered_ty = convert_cir_type_to_standard(res_ty, program_state)

    cond = ctx[op.cond] if ctx[op.cond] is not None else op.cond

    def _build_branch(region: Region) -> Block:
        block = Block()
        sub_ctx = SSAValueCtx(parent_scope=ctx)
        _translate_region_into_block(
            program_state, sub_ctx, region, block, skip_terminator=True
        )
        yielded = _yielded_values(region, sub_ctx)
        block.add_op(scf.YieldOp(*yielded))
        return block

    then_b = _build_branch(op.true_region)
    else_b = _build_branch(op.false_region)
    if_op = scf.IfOp(cond, [lowered_ty], Region([then_b]), Region([else_b]))
    ctx[op.results[0]] = if_op.results[0]
    return [if_op]


def translate_return(
    program_state: ProgramState, ctx: SSAValueCtx, op: cir.ReturnOp
) -> list[Operation]:
    args: list[SSAValue] = []
    for a in op.arguments:
        mapped = ctx[a]
        args.append(mapped if mapped is not None else a)
    return [func.ReturnOp(*args)]


# Silence unused-import noise.
_ = (Iterable, IndexType, arith)
