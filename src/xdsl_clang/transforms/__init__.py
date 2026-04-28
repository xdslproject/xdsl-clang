from collections.abc import Callable

from xdsl.passes import ModulePass


def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Return the list of all available passes."""

    def get_cir_to_core() -> type[ModulePass]:
        from xdsl_clang.transforms.cir_to_core import CIRToCore

        return CIRToCore

    return {
        "cir-to-core": get_cir_to_core,
    }
