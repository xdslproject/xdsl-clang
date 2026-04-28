"""CIR → core MLIR type conversion.

Mirrors ``ftn/transforms/to_core/components/ftn_types.py`` for the C side.
The four locked-in design decisions for the pass are encoded here:

    1. `!cir.ptr<T>` lowers to `memref<T>` for scalars, `memref<?xT>` for
       array-decayed pointers, and `!llvm.ptr` for pointers-to-record and
       function pointers.
    2. n/a here — loop-lowering policy lives in `components/control_flow`.
    3. `!cir.record<…>` lowers to `!llvm.struct<(…)>`, interned per record
       name on `ProgramState.record_layouts`.
    4. n/a here — sign tracking lives in `components/maths` and uses the
       `signedness_of` helper exported from this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects import builtin, llvm
from xdsl.dialects.builtin import DYNAMIC_INDEX
from xdsl.ir import Attribute, SSAValue
from xdsl.utils.hints import isa

from xdsl_clang.dialects import cir
from xdsl_clang.transforms.cir_to_core.misc.c_code_description import ProgramState

# ---------------------------------------------------------------------------
# Pointer-mode tracking — Decision 1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointerMode:
    """Tag attached to a `!cir.ptr<T>` lowering decision.

    `decayed` is True when the pointer was produced by an `array_to_ptrdecay`
    cast (or otherwise points to the first element of an array) and should
    therefore lower to `memref<?xT>`. False when the pointer addresses a
    single scalar value and lowers to `memref<T>`. Pointers to records and
    pointers to functions ignore this and always lower to `!llvm.ptr`.
    """

    decayed: bool = False


SCALAR_PTR = PointerMode(decayed=False)
DECAYED_PTR = PointerMode(decayed=True)


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------


def convert_cir_type_to_standard(
    cir_type: Attribute,
    program_state: ProgramState,
    *,
    ptr_mode: PointerMode = SCALAR_PTR,
) -> Attribute:
    """Lower a CIR type to its core-dialect equivalent.

    `ptr_mode` only matters when `cir_type` is `!cir.ptr<T>` and `T` is not
    a record or function — see Decision 1 in the plan.
    """
    if isa(cir_type, cir.IntType):
        return builtin.IntegerType(cir_type.bitwidth)
    if isa(cir_type, cir.BoolType):
        return builtin.IntegerType(1)
    if isa(cir_type, cir.SingleType):
        return builtin.Float32Type()
    if isa(cir_type, cir.DoubleType):
        return builtin.Float64Type()
    if isa(cir_type, cir.VoidType):
        # Void only legitimately appears as a function return type or as the
        # pointee of `!cir.ptr<!cir.void>` (i.e. `void *`). Callers handling
        # function returns special-case `VoidType` and never ask for its
        # standard form. For `void *` we return an opaque pointer — see
        # PointerType branch below.
        return llvm.LLVMVoidType()
    if isa(cir_type, cir.ArrayType):
        elem = convert_cir_type_to_standard(
            cir_type.element_type, program_state, ptr_mode=SCALAR_PTR
        )
        return builtin.MemRefType(elem, [cir_type.size.value.data])
    if isa(cir_type, cir.PointerType):
        return _convert_pointer(cir_type, program_state, ptr_mode)
    if isa(cir_type, cir.RecordType):
        return _convert_record(cir_type, program_state)
    if isa(cir_type, cir.FuncType):
        # A bare function type only appears as the pointee of a function
        # pointer. The pointer itself becomes `!llvm.ptr`; callers who need
        # the function signature go through the original CIR op.
        return llvm.LLVMPointerType()
    raise NotImplementedError(f"unsupported CIR type {cir_type}")


def _convert_pointer(
    ptr: cir.PointerType,
    program_state: ProgramState,
    ptr_mode: PointerMode,
) -> Attribute:
    pointee = ptr.pointee
    # Record / function pointers → opaque !llvm.ptr (Decision 1).
    if isa(pointee, cir.RecordType) or isa(pointee, cir.FuncType):
        return llvm.LLVMPointerType()
    # `void *` — opaque `!llvm.ptr` is the only sensible target.
    if isa(pointee, cir.VoidType):
        return llvm.LLVMPointerType()
    # `!cir.ptr<!cir.array<T x N>>` — array decay always meant; we expose
    # the underlying memref<NxT> rather than memref<memref<NxT>>.
    if isa(pointee, cir.ArrayType):
        return convert_cir_type_to_standard(pointee, program_state)
    # Scalar pointee
    elem = convert_cir_type_to_standard(pointee, program_state, ptr_mode=SCALAR_PTR)
    if ptr_mode.decayed:
        return builtin.MemRefType(elem, [DYNAMIC_INDEX])  # memref<?xT>
    return builtin.MemRefType(elem, [])  # memref<T>


def _convert_record(rec: cir.RecordType, program_state: ProgramState) -> Attribute:
    """`!cir.record<…>` → `!llvm.struct<(…)>`."""
    field_types: list[Attribute] = []
    for member in rec.members.data:
        field_types.append(
            convert_cir_type_to_standard(member, program_state, ptr_mode=SCALAR_PTR)
        )
    return llvm.LLVMStructType.from_type_list(field_types)


# ---------------------------------------------------------------------------
# Sign tracking — Decision 4
# ---------------------------------------------------------------------------


def cir_type_size_in_bytes(cir_type: Attribute) -> int:
    """Return the byte size of a CIR scalar/record type.

    Used by the `malloc`/`free` cir.cast pattern in `casts.py` to convert
    a byte-count operand back into an element count for `memref.alloc`.
    Sizes mirror the data-layout used elsewhere in the corpus:

    * `!cir.int<_, N>` → ceil(N/8)
    * `!cir.float`     → 4
    * `!cir.double`    → 8
    * `!cir.bool`      → 1
    * `!cir.array<T x N>` → N * sizeof(T)
    * `!cir.record<…>` → sum of field sizes (no alignment padding modelled)
    """
    if isa(cir_type, cir.IntType):
        return (cir_type.bitwidth + 7) // 8
    if isa(cir_type, cir.BoolType):
        return 1
    if isa(cir_type, cir.SingleType):
        return 4
    if isa(cir_type, cir.DoubleType):
        return 8
    if isa(cir_type, cir.ArrayType):
        return cir_type.size.value.data * cir_type_size_in_bytes(cir_type.element_type)
    if isa(cir_type, cir.RecordType):
        total = 0
        for member in cir_type.members.data:
            total += cir_type_size_in_bytes(member)
        return total
    if isa(cir_type, cir.PointerType):
        # 64-bit target.
        return 8
    raise NotImplementedError(f"cir_type_size_in_bytes: unsupported type {cir_type}")


def signedness_of(value: SSAValue) -> bool | None:
    """Best-effort extraction of source-level signedness for an SSA value.

    Walks back through `cir.cast` ops looking for the originating
    `!cir.int<s|u, N>` type. Returns `True` for signed, `False` for
    unsigned, `None` if the value's type carries no signedness signal
    (e.g. floats, pointers, or already-lowered `iN`).
    """
    cir_type = value.type
    if isa(cir_type, cir.IntType):
        return cir_type.signed
    if isa(cir_type, cir.BoolType):
        # `bool` is unsigned at the C level — important for cmp/cast lowering.
        return False
    # Try the producing op; cast chains are common.
    owner = value.owner
    if isa(owner, cir.CastOp):
        # Recurse on the source.
        return signedness_of(owner.src)
    return None
