"""Compatibility shims for ingesting MLIR CIR `.cir` files via `xdsl-opt`.

CIR files emit attribute aliases (`#locN = ...`) *after* the operations that
reference them. xdsl's MLIR parser walks the input top-to-bottom and rejects
unresolved aliases on first sight, so the file must be re-ordered before it
reaches the parser.

This module monkey-patches `xdsl.parser.Parser.__init__` to hoist any
`#name = …` / `!name = …` alias definitions to the top of the input. The
relative order among aliases is preserved (so chained `loc(fused[#a, #b])`
references still resolve in order). Patching is idempotent.

Aliases that already appear before their first use are unaffected (they hoist
to the same relative position they were already in). Patching does shift line
numbers in error messages — acceptable for the read-only ingestion path.
"""

from __future__ import annotations

import re
from typing import Any

from xdsl.parser import Parser

_ALIAS_DEF_RE = re.compile(r"^[#!][A-Za-z_][A-Za-z0-9_]*\s*=", re.ASCII)
_PATCHED_FLAG = "_xdsl_clang_cir_alias_hoist_patched"


def _hoist_aliases(text: str) -> str:
    """Move every `#name = ...` / `!name = ...` line to the top of `text`,
    preserving their relative order."""
    aliases: list[str] = []
    rest: list[str] = []
    for line in text.splitlines(keepends=True):
        if _ALIAS_DEF_RE.match(line):
            aliases.append(line)
        else:
            rest.append(line)
    if not aliases:
        return text
    return "".join(aliases) + "".join(rest)


def install() -> None:
    """Install the alias-hoist shim and register the `.cir` file extension.

    Safe to call more than once."""
    if getattr(Parser, _PATCHED_FLAG, False):
        return

    original_init = Parser.__init__

    def patched_init(
        self: Parser, ctx: Any, input: str, name: str = "<unknown>"
    ) -> None:
        original_init(self, ctx, _hoist_aliases(input), name)

    Parser.__init__ = patched_init
    setattr(Parser, _PATCHED_FLAG, True)

    # Make `xdsl-opt foo.cir` route through the same MLIR frontend as `.mlir`.
    from xdsl.tools.command_line_tool import CommandLineTool

    original_register_frontends = CommandLineTool.register_all_frontends

    def patched_register_frontends(self: CommandLineTool) -> None:
        original_register_frontends(self)
        if "mlir" in self.available_frontends and "cir" not in self.available_frontends:
            self.available_frontends["cir"] = self.available_frontends["mlir"]

    CommandLineTool.register_all_frontends = patched_register_frontends
