"""Tiny op-class-name-driven visitor.

Direct port of ``ftn/ftn/util/visitor.py`` with the same semantics: a
subclass defines ``traverse_<snake_case_class_name>`` to intercept a
specific op type; otherwise the visitor recurses into nested regions.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from xdsl.ir import Operation


_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _camel_to_snake(name: str) -> str:
    return _CAMEL_RE.sub("_", name).lower()


def _get_method(instance: object, method: str) -> Callable[..., Any] | None:
    if not hasattr(instance, method):
        return None
    f = getattr(instance, method)
    if callable(f):
        return f
    return None


class Visitor:
    """Subclasses define ``traverse_<snake_case_op_class_name>`` hooks."""

    def traverse(self, operation: Operation) -> None:
        class_name = _camel_to_snake(type(operation).__name__)
        hook = _get_method(self, f"traverse_{class_name}")
        if hook is not None:
            hook(operation)
            return
        for region in operation.regions:
            for block in region.blocks:
                for op in block.ops:
                    self.traverse(op)
