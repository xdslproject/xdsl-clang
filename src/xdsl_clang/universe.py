from xdsl.universe import Universe

from xdsl_clang.dialects import get_all_dialects
from xdsl_clang.transforms import get_all_passes

XDSL_CLANG_UNIVERSE = Universe(
    all_dialects=get_all_dialects(),
    all_passes=get_all_passes(),
)
