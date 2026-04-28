"""Program-state record carried through CIR → core lowering.

C-specific analogue of ``ftn/transforms/to_core/misc/fortran_code_description.py``.
Much smaller because C has no INTENT, no allocatable descriptors, no
character-box ABI — but the same shape (per-function ``ComponentState``
nested inside a global ``ProgramState``) so handlers from the two projects
read identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xdsl.ir import Attribute, Block, Operation, Region


@dataclass
class FieldDef:
    """One field of a ``!cir.record`` after layout resolution.

    ``index`` is the field's position in the lowered ``!llvm.struct``.
    ``cir_type`` is kept around so handlers can re-derive sign/element info.
    """

    name: str
    index: int
    cir_type: Attribute


@dataclass
class RecordLayout:
    """Lowered layout for a single ``!cir.record`` symbol."""

    name: str
    fields: list[FieldDef] = field(default_factory=list[FieldDef])

    def index_of(self, field_name: str) -> int:
        for f in self.fields:
            if f.name == field_name:
                return f.index
        raise KeyError(f"unknown field {field_name!r} in record {self.name!r}")


@dataclass
class ArgumentDefinition:
    """One formal parameter of a CIR function."""

    name: str
    cir_type: Attribute
    # `True` for plain scalars, `False` for pointer/array decayed args.
    is_scalar: bool


@dataclass
class FunctionDefinition:
    """Resolved metadata for a single ``cir.func``."""

    name: str
    return_type: Attribute | None
    is_definition_only: bool
    is_var_arg: bool = False
    args: list[ArgumentDefinition] = field(default_factory=list[ArgumentDefinition])

    def add_arg_def(self, arg_def: ArgumentDefinition) -> None:
        self.args.append(arg_def)


@dataclass
class GlobalCIRComponent:
    """One ``cir.global`` (constant or otherwise) gathered up front."""

    sym_name: str
    cir_type: Attribute
    cir_op: Operation


@dataclass
class ComponentState:
    """Per-function bookkeeping. Currently a thin record; grows with the
    pass (allocas-by-name caches, loop nesting state, etc.)."""

    fn_name: str | None = None
    # `index_chain[<cir-ssa-result-of-ptr_stride/get_element>]` holds the
    # accumulated list of `index`-typed SSA values to use when the eventual
    # `memref.load` / `memref.store` fires. Cleared on function entry.
    index_chain: dict[object, list[object]] = field(
        default_factory=dict[object, list[object]]
    )
    # --- Unstructured emission state (Task 5.7) --------------------------
    # When a function contains a `cir.break` / `cir.continue`, control-flow
    # handlers switch to a block-graph emission mode. The following fields
    # describe that mode:
    #   * `is_unstructured` is True iff the active function uses the
    #     unstructured emitter for its top-level body (and any ifs / loops
    #     containing break/continue).
    #   * `current_block` is the cursor: every op produced by an unstructured
    #     handler is appended here. Control-flow handlers (loops, ifs with
    #     break/continue) may swap it for a fresh block.
    #   * `function_region` is the region those new blocks should be added
    #     to.
    #   * `break_targets` / `continue_targets` are stacks of MLIR blocks
    #     that `cir.break` / `cir.continue` should branch to. Pushed by
    #     each unstructured loop emitter, popped on exit.
    #   * `block_terminated` flags that the current block has already
    #     received a terminator (cf.br / cf.cond_br / func.return) so the
    #     driver should drop any remaining (dead) ops in the source block.
    is_unstructured: bool = False
    current_block: Block | None = None
    function_region: Region | None = None
    break_targets: list[Block] = field(default_factory=list["Block"])
    continue_targets: list[Block] = field(default_factory=list["Block"])
    block_terminated: bool = False


class ProgramState:
    """Module-level bookkeeping shared by every component handler."""

    function_definitions: dict[str, FunctionDefinition]
    record_layouts: dict[str, RecordLayout]
    cir_globals: dict[str, GlobalCIRComponent]
    string_literals: dict[str, str]  # text → emitted symbol name
    is_in_global: bool
    function_state: ComponentState | None
    # Module-level prelude ops to splice in front of the lowered module body
    # (used by component handlers to hoist e.g. `memref.global` for
    # function-local constant arrays — Task 5.2).
    module_prelude_ops: list[Operation]
    next_literal_id: int

    def __init__(self) -> None:
        self.function_definitions = {}
        self.record_layouts = {}
        self.cir_globals = {}
        self.string_literals = {}
        self.is_in_global = False
        self.function_state = None
        self.module_prelude_ops = []
        self.next_literal_id = 0

    # --- module prelude (hoisted globals) --------------------------------

    def fresh_literal_symbol(self) -> str:
        """Mint a unique private-global symbol name for hoisted constants."""
        sym = f"_xclang_lit{self.next_literal_id}"
        self.next_literal_id += 1
        return sym

    def append_module_prelude_op(self, op: Operation) -> None:
        """Queue an op to be spliced into the front of the lowered module."""
        self.module_prelude_ops.append(op)

    # --- function scoping -------------------------------------------------

    def addFunctionDefinition(self, name: str, fn_def: FunctionDefinition) -> None:
        if name in self.function_definitions:
            raise KeyError(f"function {name!r} declared twice")
        self.function_definitions[name] = fn_def

    def enterFunction(self, fn_name: str) -> None:
        if self.function_state is not None:
            raise RuntimeError("nested function entry not supported")
        self.function_state = ComponentState(fn_name=fn_name)

    def getCurrentFnState(self) -> ComponentState:
        if self.function_state is None:
            raise RuntimeError("not inside a function")
        return self.function_state

    def leaveFunction(self) -> None:
        self.function_state = None

    # --- global-scope toggling -------------------------------------------

    def enterGlobal(self) -> None:
        if self.function_state is not None:
            raise RuntimeError("can't enter global from inside a function")
        self.is_in_global = True

    def leaveGlobal(self) -> None:
        self.is_in_global = False

    def isInGlobal(self) -> bool:
        return self.is_in_global
