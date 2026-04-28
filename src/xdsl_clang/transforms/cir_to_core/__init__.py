"""CIR → core MLIR lowering pass — top-level entry point.

Mirror of ``ftn/transforms/rewrite_fir_to_core.py``. The pass:

  1. Runs a pair of read-only visitors over the input module to populate a
     `ProgramState` (record layouts, function signatures, globals).
  2. Emits a fresh `builtin.ModuleOp` whose body is the lowered IR by
     dispatching every CIR op through `expressions.py` / `statements.py`.
  3. Detaches the original module body and replaces it with the lowered
     blocks.

Phase 1 ships the dispatcher skeleton + type system + ModulePass shell.
Phase 2 fills the component handlers in.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.utils.hints import isa

from xdsl_clang.dialects import cir
from xdsl_clang.transforms.cir_to_core.misc.c_code_description import (
    ArgumentDefinition,
    FieldDef,
    FunctionDefinition,
    GlobalCIRComponent,
    ProgramState,
    RecordLayout,
)
from xdsl_clang.transforms.cir_to_core.misc.ssa_context import SSAValueCtx
from xdsl_clang.transforms.cir_to_core.misc.visitor import Visitor


# ---------------------------------------------------------------------------
# Gather visitors
# ---------------------------------------------------------------------------


class GatherCIRGlobals(Visitor):
    """Collect every `cir.global` so handlers can resolve `cir.get_global`."""

    def __init__(self, program_state: ProgramState) -> None:
        self.program_state = program_state

    def traverse_global_op(self, op: cir.GlobalOp) -> None:
        sym = op.sym_name.data
        self.program_state.cir_globals[sym] = GlobalCIRComponent(
            sym_name=sym, cir_type=op.sym_type, cir_op=op
        )


class GatherFunctionInformation(Visitor):
    """Collect each function signature + scan record types referenced by it
    to populate `record_layouts`."""

    def __init__(self, program_state: ProgramState) -> None:
        self.program_state = program_state

    def _ensure_record_layout(self, record: cir.RecordType) -> None:
        name = record.record_name.data
        if not name or name in self.program_state.record_layouts:
            return
        layout = RecordLayout(name=name)
        for idx, member in enumerate(record.members.data):
            # CIR records use positional member types; field names are not
            # carried in the dialect. The frontend addresses fields by index
            # (`cir.get_member %p[<idx>]`) so we mint synthetic field names
            # `f0`, `f1`, … that match the GEP indices used downstream.
            layout.fields.append(
                FieldDef(name=f"f{idx}", index=idx, cir_type=member)
            )
            if isa(member, cir.RecordType):
                self._ensure_record_layout(member)
        self.program_state.record_layouts[name] = layout

    def traverse_func_op(self, op: cir.FuncOp) -> None:
        fn_name = op.sym_name.data
        # `malloc` / `free` are pattern-matched by `translate_cast` /
        # `translate_call` and lowered to `memref.alloc` / `memref.dealloc`
        # — we never want to emit `func.func` declarations for them, so
        # skip them entirely from the function table. See Phase 5 Task 5.4.
        if fn_name in ("malloc", "calloc", "free"):
            return
        return_type = op.function_type.return_type
        if isa(return_type, cir.VoidType):
            return_type_for_def = None
        else:
            return_type_for_def = return_type
        is_definition_only = len(op.body.blocks) == 0
        # Phase 5 Task 5.7 follow-up: track variadics so extern decls /
        # call sites can switch to `llvm.func` / `llvm.call` (the func
        # dialect has no variadic support — needed for `printf` & friends
        # in `jacobi.c` once break/continue is lowered).
        is_var_arg = bool(op.function_type.varargs)
        fn_def = FunctionDefinition(
            name=fn_name,
            return_type=return_type_for_def,
            is_definition_only=is_definition_only,
            is_var_arg=is_var_arg,
        )
        for input_type in op.function_type.inputs.data:
            is_scalar = not (
                isa(input_type, cir.PointerType) or isa(input_type, cir.ArrayType)
            )
            fn_def.add_arg_def(
                ArgumentDefinition(name="", cir_type=input_type, is_scalar=is_scalar)
            )
            self._scan_for_records(input_type)
        self._scan_for_records(return_type)
        self.program_state.addFunctionDefinition(fn_name, fn_def)

    def _scan_for_records(self, t: object) -> None:
        if isa(t, cir.RecordType):
            self._ensure_record_layout(t)
        elif isa(t, cir.PointerType):
            self._scan_for_records(t.pointee)
        elif isa(t, cir.ArrayType):
            self._scan_for_records(t.element_type)


# ---------------------------------------------------------------------------
# Top-level translator
# ---------------------------------------------------------------------------


def translate_program(
    program_state: ProgramState, input_module: builtin.ModuleOp
) -> builtin.ModuleOp:
    """Build a fresh module by dispatching each top-level CIR op."""
    # Late imports: the dispatcher pulls in components, components import
    # types — keeping these here avoids cycles during package init.
    from xdsl_clang.transforms.cir_to_core import statements
    from xdsl_clang.transforms.cir_to_core.components import (
        functions as cir_functions,
    )

    global_ctx = SSAValueCtx()
    body_block = Block()
    for op in list(input_module.body.block.ops):
        if isa(op, cir.FuncOp):
            new_op = cir_functions.translate_function(program_state, global_ctx, op)
            if new_op is not None:
                body_block.add_op(new_op)
            continue
        if isa(op, cir.GlobalOp):
            for new in statements.translate_stmt(program_state, global_ctx, op):
                body_block.add_op(new)
            continue
        # Fallback: any other op gets dispatched (some C-level decls are
        # represented by CIR ops we haven't catalogued yet).
        for new in statements.translate_stmt(program_state, global_ctx, op):
            body_block.add_op(new)

    # Splice any prelude ops (e.g. hoisted `memref.global`s for
    # function-local constant arrays — Task 5.2) into the front of the
    # lowered module body so they appear at module scope before any
    # function that references them.
    if program_state.module_prelude_ops:
        existing_ops = list(body_block.ops)
        for op in existing_ops:
            op.detach()
        for prelude_op in program_state.module_prelude_ops:
            body_block.add_op(prelude_op)
        for op in existing_ops:
            body_block.add_op(op)
    return builtin.ModuleOp(Region([body_block]))


# ---------------------------------------------------------------------------
# ModulePass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CIRToCore(ModulePass):
    """Lower CIR ops to core MLIR dialects (memref, arith, scf, cf, func, llvm)."""

    name = "cir-to-core"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:  # noqa: ARG002
        program_state = ProgramState()
        GatherCIRGlobals(program_state).traverse(op)
        GatherFunctionInformation(program_state).traverse(op)

        new_module = translate_program(program_state, op)

        # Detach the original CIR body and replace it with the lowered one,
        # mirroring `RewriteFIRToCore.apply` from ftn.
        op.body.detach_block(op.body.block)
        new_module.regions[0].move_blocks(op.regions[0])

        # Strip CIR-specific module attributes (triple, lang) that core
        # dialects don't understand.
        for attr_name in list(op.attributes):
            if attr_name.startswith("cir."):
                del op.attributes[attr_name]


__all__ = [
    "CIRToCore",
    "GatherCIRGlobals",
    "GatherFunctionInformation",
    "translate_program",
]
