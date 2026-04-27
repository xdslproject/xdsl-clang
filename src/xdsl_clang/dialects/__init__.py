from collections.abc import Callable

from xdsl.ir import Dialect


def get_all_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available dialects."""

    def get_cir():
        from xdsl_clang.dialects.cir import CIR

        return CIR

    return {
        "cir": get_cir,
    }
