"""SSA-value context — maps source-level identifiers and CIR SSA values to
the corresponding SSA values in the lowered (core-dialect) IR.

Direct port of ``ftn/transforms/to_core/misc/ssa_context.py`` with the same
parent-scope chain, kept structurally identical so component handlers can
move between the two projects without rewriting bookkeeping.
"""

from __future__ import annotations

from xdsl.ir import SSAValue


class SSAValueCtx:
    """Identifier (or CIR SSA value) → lowered SSA value.

    Keys can be either a ``str`` (source-level name, when a CIR op carried a
    name attribute) or an ``SSAValue`` (the CIR-side value being replaced).
    """

    dictionary: dict[object, SSAValue]
    parent_scope: SSAValueCtx | None

    def __init__(self, parent_scope: SSAValueCtx | None = None) -> None:
        self.parent_scope = parent_scope
        self.dictionary = {}

    def __getitem__(self, identifier: object) -> SSAValue | None:
        ssa_value = self.dictionary.get(identifier, None)
        if ssa_value is not None:
            return ssa_value
        if self.parent_scope is not None:
            return self.parent_scope[identifier]
        return None

    def __delitem__(self, identifier: object) -> None:
        if identifier in self.dictionary:
            del self.dictionary[identifier]

    def __setitem__(self, identifier: object, ssa_value: SSAValue) -> None:
        if identifier in self.dictionary:
            raise KeyError(f"identifier {identifier!r} already in scope")
        self.dictionary[identifier] = ssa_value

    def contains(self, identifier: object) -> bool:
        if identifier in self.dictionary:
            return True
        if self.parent_scope is not None:
            return self.parent_scope.contains(identifier)
        return False
