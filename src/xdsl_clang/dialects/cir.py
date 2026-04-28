"""
The CIR (ClangIR) dialect.

A direct port of the upstream MLIR CIR dialect from LLVM 22.1.2
(`clang/include/clang/CIR/Dialect/IR/`). This is a fixture-driven port: it
covers the subset of the upstream dialect that the round-trip fixtures in
`c_tests/cir_generic/` actually exercise. See `docs/cir-port/inventory.md` for
the contract and `docs/cir-port/blocked.md` for any fixtures skipped by the
round-trip gate.

The fixtures use MLIR's generic operation form, which means most enum-valued
attributes appear as plain `i32` integer attributes (e.g.
`linkage = 8 : i32`, `kind = 11 : i32`); they are modelled here as
`IntegerAttr` properties rather than typed enum classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from xdsl.dialects.builtin import (
    AnyFloat,
    ArrayAttr,
    DenseArrayBase,
    FlatSymbolRefAttr,
    Float32Type,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
    i32,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
    var_successor_def,
)
from xdsl.parser import AttrParser, Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.traits import IsTerminator, NoTerminator

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@irdl_attr_definition
class IntType(ParametrizedAttribute, TypeAttribute):
    """`!cir.int<s|u, N>` — CIR arbitrary-precision integer type."""

    name = "cir.int"

    width: IntegerAttr
    is_signed: IntegerAttr  # 0/1 stored as i1 IntegerAttr

    def __init__(self, width: int, is_signed: bool):
        super().__init__(
            IntegerAttr(width, 64),
            IntegerAttr(1 if is_signed else 0, 1),
        )

    @property
    def signed(self) -> bool:
        return bool(self.is_signed.value.data)

    @property
    def bitwidth(self) -> int:
        return self.width.value.data

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("s" if self.signed else "u")
            printer.print_string(", ")
            printer.print_string(str(self.bitwidth))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            sign_kw = parser.parse_identifier()
            if sign_kw not in ("s", "u"):
                parser.raise_error("expected 's' or 'u'")
            parser.parse_punctuation(",")
            width = parser.parse_integer()
            if not (1 <= width <= 128):
                parser.raise_error("integer width must be between 1 and 128")
        return [IntegerAttr(width, 64), IntegerAttr(1 if sign_kw == "s" else 0, 1)]


@irdl_attr_definition
class BoolType(ParametrizedAttribute, TypeAttribute):
    """`!cir.bool` — CIR bool type."""

    name = "cir.bool"


@irdl_attr_definition
class SingleType(ParametrizedAttribute, TypeAttribute):
    """`!cir.float` — CIR single-precision (32-bit) float."""

    name = "cir.float"

    @property
    def format(self) -> str:
        return "<f"


@irdl_attr_definition
class DoubleType(ParametrizedAttribute, TypeAttribute):
    """`!cir.double` — CIR double-precision (64-bit) float."""

    name = "cir.double"

    @property
    def format(self) -> str:
        return "<d"


@irdl_attr_definition
class FP16Type(ParametrizedAttribute, TypeAttribute):
    """`!cir.f16` — CIR half-precision (16-bit) float."""

    name = "cir.f16"

    @property
    def format(self) -> str:
        return "<e"


@irdl_attr_definition
class BF16Type(ParametrizedAttribute, TypeAttribute):
    """`!cir.bf16` — CIR bfloat16 (16-bit) float."""

    name = "cir.bf16"

    @property
    def format(self) -> str:
        # Python `struct` has no native bfloat16 packing; defer to the
        # CIRFPAttr printer which handles bf16 directly.
        raise NotImplementedError()


@irdl_attr_definition
class FP80Type(ParametrizedAttribute, TypeAttribute):
    """`!cir.f80` — CIR x87 80-bit extended-precision float."""

    name = "cir.f80"

    @property
    def format(self) -> str:
        # Python `struct` has no 80-bit float packing.
        raise NotImplementedError()


@irdl_attr_definition
class FP128Type(ParametrizedAttribute, TypeAttribute):
    """`!cir.f128` — CIR quad-precision (128-bit) float."""

    name = "cir.f128"

    @property
    def format(self) -> str:
        # Python `struct` has no 128-bit float packing.
        raise NotImplementedError()


@irdl_attr_definition
class LongDoubleType(ParametrizedAttribute, TypeAttribute):
    """`!cir.long_double<T>` — CIR `long double`, parametric on its underlying
    floating-point format (one of `!cir.double`, `!cir.f80`, `!cir.f128`)."""

    name = "cir.long_double"

    underlying: Attribute

    def __init__(self, underlying: Attribute):
        super().__init__(underlying)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.underlying)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            underlying = parser.parse_type()
            if not isinstance(underlying, (DoubleType, FP80Type, FP128Type)):
                parser.raise_error(
                    "expected !cir.double, !cir.f80 or !cir.f128 underlying type"
                )
        return [underlying]


@irdl_attr_definition
class ComplexType(ParametrizedAttribute, TypeAttribute):
    """`!cir.complex<T>` — CIR `_Complex` type. The element type must be a CIR
    integer or floating-point type."""

    name = "cir.complex"

    element_type: Attribute

    def __init__(self, element_type: Attribute):
        super().__init__(element_type)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.element_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            element_type = parser.parse_type()
        return [element_type]


@irdl_attr_definition
class VectorType(ParametrizedAttribute, TypeAttribute):
    """`!cir.vector<N x T>` (fixed) or `!cir.vector<[N] x T>` (scalable) — CIR
    one-dimensional vector type."""

    name = "cir.vector"

    element_type: Attribute
    size: IntegerAttr
    is_scalable: IntegerAttr

    def __init__(self, element_type: Attribute, size: int, is_scalable: bool = False):
        super().__init__(
            element_type,
            IntegerAttr(size, 64),
            IntegerAttr(1 if is_scalable else 0, 1),
        )

    @property
    def length(self) -> int:
        return self.size.value.data

    @property
    def scalable(self) -> bool:
        return bool(self.is_scalable.value.data)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            if self.scalable:
                printer.print_string("[")
            printer.print_string(str(self.length))
            if self.scalable:
                printer.print_string("]")
            printer.print_string(" x ")
            printer.print_attribute(self.element_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            is_scalable = parser.parse_optional_punctuation("[") is not None
            size = parser.parse_integer()
            if is_scalable:
                parser.parse_punctuation("]")
            parser.parse_keyword("x")
            element_type = parser.parse_type()
        return [
            element_type,
            IntegerAttr(size, 64),
            IntegerAttr(1 if is_scalable else 0, 1),
        ]


@irdl_attr_definition
class VoidType(ParametrizedAttribute, TypeAttribute):
    """`!cir.void` — CIR void type."""

    name = "cir.void"


@irdl_attr_definition
class PointerType(ParametrizedAttribute, TypeAttribute):
    """`!cir.ptr<T>` — CIR pointer type. Address space is not modelled."""

    name = "cir.ptr"

    pointee: Attribute

    def __init__(self, pointee: Attribute):
        super().__init__(pointee)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.pointee)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            pointee = parser.parse_type()
            # addrspace clause: `, target_address_space(N)` — not present in fixtures
        return [pointee]


@irdl_attr_definition
class ArrayType(ParametrizedAttribute, TypeAttribute):
    """`!cir.array<T x N>` — CIR fixed-size array type."""

    name = "cir.array"

    element_type: Attribute
    size: IntegerAttr

    def __init__(self, element_type: Attribute, size: int):
        super().__init__(element_type, IntegerAttr(size, 64))

    @property
    def length(self) -> int:
        return self.size.value.data

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.element_type)
            printer.print_string(" x ")
            printer.print_string(str(self.length))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            element_type = parser.parse_type()
            parser.parse_shape_delimiter()
            size = parser.parse_integer()
        return [element_type, IntegerAttr(size, 64)]


@irdl_attr_definition
class FuncType(ParametrizedAttribute, TypeAttribute):
    """`!cir.func<(T1, T2, ...) -> R>` — CIR function type.

    The return type is optional; absence means a `void`-like return. A trailing
    `...` in the parameter list marks the function as variadic.
    """

    name = "cir.func"

    inputs: ArrayAttr[Attribute]
    return_type: Attribute  # VoidType used as sentinel for "no return"
    is_var_arg: IntegerAttr

    def __init__(
        self,
        inputs: Sequence[Attribute] | ArrayAttr[Attribute],
        return_type: Attribute | None = None,
        is_var_arg: bool = False,
    ):
        if not isinstance(inputs, ArrayAttr):
            inputs = ArrayAttr(inputs)
        if return_type is None:
            return_type = VoidType()
        super().__init__(inputs, return_type, IntegerAttr(1 if is_var_arg else 0, 1))

    @property
    def has_void_return(self) -> bool:
        return isinstance(self.return_type, VoidType)

    @property
    def varargs(self) -> bool:
        return bool(self.is_var_arg.value.data)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            with printer.in_parens():
                printer.print_list(self.inputs.data, printer.print_attribute)
                if self.varargs:
                    if self.inputs.data:
                        printer.print_string(", ")
                    printer.print_string("...")
            if not self.has_void_return:
                printer.print_string(" -> ")
                printer.print_attribute(self.return_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            inputs: list[Attribute] = []
            is_var_arg = False
            parser.parse_punctuation("(")
            if parser.parse_optional_punctuation(")") is None:
                while True:
                    if parser.parse_optional_punctuation("...") is not None:
                        is_var_arg = True
                        parser.parse_punctuation(")")
                        break
                    inputs.append(parser.parse_type())
                    if parser.parse_optional_punctuation(",") is None:
                        parser.parse_punctuation(")")
                        break
            return_type: Attribute = VoidType()
            if parser.parse_optional_punctuation("->") is not None:
                return_type = parser.parse_type()
        return [
            ArrayAttr(inputs),
            return_type,
            IntegerAttr(1 if is_var_arg else 0, 1),
        ]


@irdl_attr_definition
class RecordType(ParametrizedAttribute, TypeAttribute):
    """`!cir.record<struct|union|class "name"? (packed)?
                                    (padded)? (incomplete | { members })>`.

    The fixtures only exercise complete identified records, so mutability /
    self-reference is not modelled here.
    """

    name = "cir.record"

    members: ArrayAttr[Attribute]
    record_name: StringAttr  # empty string means anonymous
    kind: StringAttr  # "struct" | "union" | "class"
    is_incomplete: IntegerAttr
    is_packed: IntegerAttr
    is_padded: IntegerAttr

    def __init__(
        self,
        members: Sequence[Attribute] | ArrayAttr[Attribute],
        record_name: str | StringAttr = "",
        kind: str | StringAttr = "struct",
        incomplete: bool = False,
        packed: bool = False,
        padded: bool = False,
    ):
        if not isinstance(members, ArrayAttr):
            members = ArrayAttr(members)
        if isinstance(record_name, str):
            record_name = StringAttr(record_name)
        if isinstance(kind, str):
            kind = StringAttr(kind)
        super().__init__(
            members,
            record_name,
            kind,
            IntegerAttr(1 if incomplete else 0, 1),
            IntegerAttr(1 if packed else 0, 1),
            IntegerAttr(1 if padded else 0, 1),
        )

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.kind.data)
            if self.record_name.data:
                printer.print_string(" ")
                printer.print_string_literal(self.record_name.data)
            printer.print_string(" ")
            if self.is_packed.value.data:
                printer.print_string("packed ")
            if self.is_padded.value.data:
                printer.print_string("padded ")
            if self.is_incomplete.value.data:
                printer.print_string("incomplete")
            else:
                with printer.in_braces():
                    printer.print_list(self.members.data, printer.print_attribute)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            kind = parser.parse_identifier()
            if kind not in ("struct", "union", "class"):
                parser.raise_error("expected 'struct', 'union' or 'class'")
            record_name = parser.parse_optional_str_literal() or ""
            packed = parser.parse_optional_keyword("packed") is not None
            padded = parser.parse_optional_keyword("padded") is not None
            members: list[Attribute] = []
            incomplete = parser.parse_optional_keyword("incomplete") is not None
            if not incomplete:
                # An identified record may appear as a forward reference of
                # the form `<struct "Name">` (no body, no `incomplete`); treat
                # this as the incomplete form.
                if parser.parse_optional_punctuation("{") is None:
                    if not record_name:
                        parser.raise_error(
                            "expected 'incomplete' or '{' for record body"
                        )
                    incomplete = True
                else:
                    if parser.parse_optional_punctuation("}") is None:
                        members.append(parser.parse_type())
                        while parser.parse_optional_punctuation(",") is not None:
                            members.append(parser.parse_type())
                        parser.parse_punctuation("}")
        return [
            ArrayAttr(members),
            StringAttr(record_name),
            StringAttr(kind),
            IntegerAttr(1 if incomplete else 0, 1),
            IntegerAttr(1 if packed else 0, 1),
            IntegerAttr(1 if padded else 0, 1),
        ]


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


@irdl_attr_definition
class CIRBoolAttr(ParametrizedAttribute):
    """`#cir.bool<true|false>`."""

    name = "cir.bool"

    bool_type: BoolType
    value: IntegerAttr

    def __init__(self, value: bool):
        super().__init__(BoolType(), IntegerAttr(1 if value else 0, 1))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("true" if self.value.value.data else "false")
        printer.print_string(" : ")
        printer.print_attribute(self.bool_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            kw = parser.parse_identifier()
            if kw not in ("true", "false"):
                parser.raise_error("expected 'true' or 'false'")
        parser.parse_punctuation(":")
        bool_type = parser.parse_type()
        if not isinstance(bool_type, BoolType):
            parser.raise_error("expected !cir.bool type")
        return [bool_type, IntegerAttr(1 if kw == "true" else 0, 1)]

    def get_type(self) -> Attribute:
        return self.bool_type


@irdl_attr_definition
class CIRIntAttr(ParametrizedAttribute):
    """`#cir.int<N> : !cir.int<...>` — typed integer constant."""

    name = "cir.int"

    int_type: IntType
    value: IntegerAttr

    def __init__(self, value: int, int_type: IntType):
        super().__init__(int_type, IntegerAttr(value, 64))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            v = self.value.value.data
            if self.int_type.signed:
                width = self.int_type.bitwidth
                if v >= (1 << (width - 1)):
                    v -= 1 << width
            printer.print_string(str(v))
        printer.print_string(" : ")
        printer.print_attribute(self.int_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            value = parser.parse_integer(allow_negative=True)
        parser.parse_punctuation(":")
        int_type = parser.parse_type()
        if not isinstance(int_type, IntType):
            parser.raise_error("expected !cir.int type for #cir.int attribute")
        return [int_type, IntegerAttr(value, 64)]

    def get_type(self) -> Attribute:
        return self.int_type


@irdl_attr_definition
class CIRFPAttr(ParametrizedAttribute):
    """`#cir.fp<F> : !cir.<float|double>` — typed float constant."""

    name = "cir.fp"

    fp_type: Attribute  # any CIR FP type
    value: FloatAttr

    def __init__(self, value: float, fp_type: Attribute):
        builtin_fp = _cir_fp_to_builtin(fp_type)
        super().__init__(fp_type, FloatAttr(value, builtin_fp))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            value = self.value.value.data
            # For builtin float types whose `format` is not implemented
            # (BFloat16Type, Float80Type, Float128Type), Python's `struct`
            # cannot pack/unpack their representation, so xdsl's
            # `printer.print_float` round-trip check crashes. Format such
            # values directly with mlir-opt-style scientific notation.
            try:
                _ = self.value.type.format
                use_native = True
            except NotImplementedError:
                use_native = False
            if use_native:
                printer.print_float(value, self.value.type)
            else:
                printer.print_string(_format_fp_scientific(value))
        printer.print_string(" : ")
        printer.print_attribute(self.fp_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            f = parser.parse_float()
        parser.parse_punctuation(":")
        fp_type = parser.parse_type()
        if not isinstance(
            fp_type,
            (
                SingleType,
                DoubleType,
                FP16Type,
                BF16Type,
                FP80Type,
                FP128Type,
                LongDoubleType,
            ),
        ):
            parser.raise_error("expected a CIR floating-point type for #cir.fp")
        builtin_fp = _cir_fp_to_builtin(fp_type)
        return [fp_type, FloatAttr(f, builtin_fp)]

    def get_type(self) -> Attribute:
        return self.fp_type


def _format_fp_scientific(value: float) -> str:
    """Format a float in mlir-opt scientific notation (e.g. `1.250000e+00`).

    Used for CIR FP types whose builtin equivalent does not implement the
    `format` property (BF16, FP80, FP128) — for these, xdsl's
    `printer.print_float` cannot pack/unpack the value through Python's
    `struct` module to verify a lossless round-trip.
    """
    import math

    if math.isnan(value) or math.isinf(value):
        return repr(value)
    s = f"{value:.6e}"
    # mlir uses two-digit exponents; Python already does on Linux but force it.
    idx = s.find("e")
    sign = s[idx + 1]
    digits = s[idx + 2 :]
    if len(digits) < 2:
        digits = digits.zfill(2)
    return s[:idx] + "e" + sign + digits


def _cir_fp_to_builtin(t: Attribute) -> AnyFloat:
    from xdsl.dialects.builtin import (
        BFloat16Type,
        Float16Type,
        Float80Type,
        Float128Type,
    )

    if isinstance(t, SingleType):
        return Float32Type()
    if isinstance(t, DoubleType):
        return Float64Type()
    if isinstance(t, FP16Type):
        return Float16Type()
    if isinstance(t, BF16Type):
        return BFloat16Type()
    if isinstance(t, FP80Type):
        return Float80Type()
    if isinstance(t, FP128Type):
        return Float128Type()
    if isinstance(t, LongDoubleType):
        return _cir_fp_to_builtin(t.underlying)
    raise ValueError(f"unsupported CIR FP type {t}")


@irdl_attr_definition
class ZeroAttr(ParametrizedAttribute):
    """`#cir.zero : T` — zero-initialiser."""

    name = "cir.zero"

    zero_type: Attribute

    def __init__(self, zero_type: Attribute):
        super().__init__(zero_type)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(" : ")
        printer.print_attribute(self.zero_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation(":")
        zero_type = parser.parse_type()
        return [zero_type]

    def get_type(self) -> Attribute:
        return self.zero_type


@irdl_attr_definition
class ConstPtrAttr(ParametrizedAttribute):
    """`#cir.ptr<null>` or `#cir.ptr<N>` — typed pointer constant."""

    name = "cir.ptr"

    ptr_type: PointerType
    value: IntegerAttr

    def __init__(self, value: int, ptr_type: PointerType):
        super().__init__(ptr_type, IntegerAttr(value, 64))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            if self.value.value.data == 0:
                printer.print_string("null")
            else:
                printer.print_int(self.value.value.data, IntegerType(64))
        printer.print_string(" : ")
        printer.print_attribute(self.ptr_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            if parser.parse_optional_keyword("null") is not None:
                value = 0
            else:
                value = parser.parse_integer(allow_negative=True)
        parser.parse_punctuation(":")
        ptr_type = parser.parse_type()
        if not isinstance(ptr_type, PointerType):
            parser.raise_error("expected !cir.ptr type for #cir.ptr attribute")
        return [ptr_type, IntegerAttr(value, 64)]

    def get_type(self) -> Attribute:
        return self.ptr_type


@irdl_attr_definition
class ConstArrayAttr(ParametrizedAttribute):
    """`#cir.const_array<elts : !cir.array<T x N>>` — array constant.

    `elts` is either a string literal (for char arrays) or an `[...]` array of
    typed attributes.
    """

    name = "cir.const_array"

    arr_type: Attribute
    elts: Attribute  # StringAttr or ArrayAttr
    trailing_zeros: IntegerAttr

    def __init__(
        self,
        arr_type: Attribute,
        elts: Attribute,
        trailing_zeros: int = 0,
    ):
        super().__init__(arr_type, elts, IntegerAttr(trailing_zeros, 64))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            if isinstance(self.elts, StringAttr):
                # MLIR-style hex escapes for string payloads.
                printer.print_bytes_literal(
                    self.elts.data.encode("utf-8", "surrogateescape")
                )
                # String literals carry their type via the inner `: T`.
                printer.print_string(" : ")
                printer.print_attribute(self.arr_type)
            else:
                printer.print_attribute(self.elts)
            if self.trailing_zeros.value.data != 0:
                printer.print_string(", trailing_zeros")
        # Outer typed-attribute `: T` suffix (TypedAttrInterface).
        printer.print_string(" : ")
        printer.print_attribute(self.arr_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        explicit_trailing_zeros = False
        with parser.in_angle_brackets():
            elts = parser.parse_attribute()
            inner_type = None
            # Inner `: T` only present when elts is a string literal.
            if parser.parse_optional_punctuation(":") is not None:
                inner_type = parser.parse_type()
            # Optional `, trailing_zeros` modifier.
            if parser.parse_optional_punctuation(",") is not None:
                parser.parse_keyword("trailing_zeros")
                explicit_trailing_zeros = True
        # Outer `: T` typed-attribute suffix.
        parser.parse_punctuation(":")
        outer_type = parser.parse_type()
        arr_type = inner_type if inner_type is not None else outer_type
        zeros = 0
        if explicit_trailing_zeros and isinstance(arr_type, ArrayType):
            if isinstance(elts, StringAttr):
                zeros = max(0, arr_type.length - len(elts.data))
            elif isinstance(elts, ArrayAttr):
                arr_elts = cast(ArrayAttr[Attribute], elts)
                zeros = max(0, arr_type.length - len(arr_elts.data))
            # If lengths match, still mark as explicitly trailing_zeros via 0.
            # When the keyword is absent, we keep zeros = 0 regardless.
        return [arr_type, elts, IntegerAttr(zeros, 64)]

    def get_type(self) -> Attribute:
        return self.arr_type


@irdl_attr_definition
class SourceLanguageAttr(ParametrizedAttribute):
    """`#cir.lang<c>` or `#cir<lang c>` — source language enum."""

    name = "cir.lang"

    value: StringAttr

    def __init__(self, value: str | StringAttr):
        if isinstance(value, str):
            value = StringAttr(value)
        super().__init__(value)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.value.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        # Both opaque (`#cir<lang c>`) and pretty (`#cir.lang<c>`) forms reach
        # this code with the cursor either on `<` (pretty) or just past the
        # bare-ident `lang` (opaque).
        if parser.parse_optional_punctuation("<") is not None:
            kw = parser.parse_identifier()
            parser.parse_punctuation(">")
        else:
            kw = parser.parse_identifier()
        return [StringAttr(kw)]


@irdl_attr_definition
class VisibilityAttr(ParametrizedAttribute):
    """`#cir.visibility<default>` or `#cir<visibility default>` — visibility kind."""

    name = "cir.visibility"

    value: StringAttr

    def __init__(self, value: str | StringAttr):
        if isinstance(value, str):
            value = StringAttr(value)
        super().__init__(value)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.value.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        if parser.parse_optional_punctuation("<") is not None:
            kw = parser.parse_identifier()
            parser.parse_punctuation(">")
        else:
            kw = parser.parse_identifier()
        return [StringAttr(kw)]


@irdl_attr_definition
class UndefAttr(ParametrizedAttribute):
    """`#cir.undef : T` — typed `undef` constant."""

    name = "cir.undef"

    undef_type: Attribute

    def __init__(self, undef_type: Attribute):
        super().__init__(undef_type)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(" : ")
        printer.print_attribute(self.undef_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation(":")
        return [parser.parse_type()]

    def get_type(self) -> Attribute:
        return self.undef_type


@irdl_attr_definition
class PoisonAttr(ParametrizedAttribute):
    """`#cir.poison : T` — typed `poison` constant."""

    name = "cir.poison"

    poison_type: Attribute

    def __init__(self, poison_type: Attribute):
        super().__init__(poison_type)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(" : ")
        printer.print_attribute(self.poison_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation(":")
        return [parser.parse_type()]

    def get_type(self) -> Attribute:
        return self.poison_type


@irdl_attr_definition
class ConstRecordAttr(ParametrizedAttribute):
    """`#cir.const_record<{m1, m2, ...}> : !cir.record<...>` — typed
    record/struct initialiser. `members` is an `mlir::ArrayAttr` of typed
    member values, printed inside braces."""

    name = "cir.const_record"

    record_type: Attribute
    members: ArrayAttr[Attribute]

    def __init__(
        self,
        record_type: Attribute,
        members: Sequence[Attribute] | ArrayAttr[Attribute],
    ):
        if not isinstance(members, ArrayAttr):
            members = ArrayAttr(members)
        super().__init__(record_type, members)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            with printer.in_braces():
                printer.print_list(self.members.data, printer.print_attribute)
        printer.print_string(" : ")
        printer.print_attribute(self.record_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            members: list[Attribute] = []
            parser.parse_punctuation("{")
            if parser.parse_optional_punctuation("}") is None:
                members.append(parser.parse_attribute())
                while parser.parse_optional_punctuation(",") is not None:
                    members.append(parser.parse_attribute())
                parser.parse_punctuation("}")
        parser.parse_punctuation(":")
        record_type = parser.parse_type()
        return [record_type, ArrayAttr(members)]

    def get_type(self) -> Attribute:
        return self.record_type


@irdl_attr_definition
class GlobalViewAttr(ParametrizedAttribute):
    """`#cir.global_view<@sym (, [i, j, ...])?> : T` — typed pointer to a
    global symbol, optionally with sub-element indices."""

    name = "cir.global_view"

    view_type: Attribute
    symbol: SymbolRefAttr
    indices: ArrayAttr[Attribute]

    def __init__(
        self,
        view_type: Attribute,
        symbol: str | SymbolRefAttr,
        indices: Sequence[Attribute] | ArrayAttr[Attribute] = (),
    ):
        if isinstance(symbol, str):
            symbol = SymbolRefAttr(symbol)
        if not isinstance(indices, ArrayAttr):
            indices = ArrayAttr(indices)
        super().__init__(view_type, symbol, indices)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.symbol)
            if self.indices.data:
                printer.print_string(", [")
                printer.print_list(self.indices.data, printer.print_attribute)
                printer.print_string("]")
        printer.print_string(" : ")
        printer.print_attribute(self.view_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            symbol = parser.parse_attribute()
            indices: list[Attribute] = []
            if parser.parse_optional_punctuation(",") is not None:
                parser.parse_punctuation("[")
                if parser.parse_optional_punctuation("]") is None:
                    indices.append(parser.parse_attribute())
                    while parser.parse_optional_punctuation(",") is not None:
                        indices.append(parser.parse_attribute())
                    parser.parse_punctuation("]")
        parser.parse_punctuation(":")
        view_type = parser.parse_type()
        if not isinstance(symbol, SymbolRefAttr):
            parser.raise_error("expected a symbol reference")
        return [view_type, symbol, ArrayAttr(indices)]

    def get_type(self) -> Attribute:
        return self.view_type


@irdl_attr_definition
class OptInfoAttr(ParametrizedAttribute):
    """`#cir.opt_info<level = N, size = M>` — module-level optimisation flags."""

    name = "cir.opt_info"

    level: IntegerAttr
    size: IntegerAttr

    def __init__(self, level: int, size: int):
        super().__init__(IntegerAttr(level, 32), IntegerAttr(size, 32))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("level = ")
            printer.print_string(str(self.level.value.data))
            printer.print_string(", size = ")
            printer.print_string(str(self.size.value.data))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_keyword("level")
            parser.parse_punctuation("=")
            level = parser.parse_integer()
            parser.parse_punctuation(",")
            parser.parse_keyword("size")
            parser.parse_punctuation("=")
            size = parser.parse_integer()
        return [IntegerAttr(level, 32), IntegerAttr(size, 32)]


# ---------------------------------------------------------------------------
# Enum kind tables
#
# CIR encodes a number of enums as plain `i32` IntegerAttr values in generic
# form. The pretty form spells them as keywords. The numeric ↔ keyword mapping
# below is recovered from the upstream MLIR CIR dialect (LLVM 22) and validated
# against the round-trip fixtures in `c_tests/`.
# ---------------------------------------------------------------------------


_BIN_OP_KIND = {
    "mul": 0,
    "div": 1,
    "rem": 2,
    "add": 3,
    "sub": 4,
    "shl": 5,
    "shr": 6,
    "and": 7,
    "xor": 8,
    "or": 9,
    "max": 10,
}
_BIN_OP_KIND_INV = {v: k for k, v in _BIN_OP_KIND.items()}

_UNARY_OP_KIND = {
    "inc": 0,
    "dec": 1,
    "plus": 2,
    "minus": 3,
    "not": 4,
}
_UNARY_OP_KIND_INV = {v: k for k, v in _UNARY_OP_KIND.items()}

_CMP_OP_KIND = {
    "lt": 0,
    "le": 1,
    "gt": 2,
    "ge": 3,
    "eq": 4,
    "ne": 5,
}
_CMP_OP_KIND_INV = {v: k for k, v in _CMP_OP_KIND.items()}

_CAST_KIND = {
    "int_to_bool": 0,
    "bitcast": 1,
    "ptr_to_bool": 2,
    "ptr_to_int": 3,
    "int_to_ptr": 4,
    "address_space": 5,
    "float_to_bool": 6,
    "bool_to_int": 7,
    "float_to_int": 8,
    "float_to_float": 9,
    "int_to_float": 10,
    "array_to_ptrdecay": 11,
    "complex_to_real": 12,
    "complex_to_imag": 13,
    "complex_to_bool": 14,
    "complex_create": 15,
    "real_to_complex": 16,
    "complex_cast_real": 17,
    "complex_cast_imag": 18,
    "member_ptr_to_bool": 19,
    "complex_real_to_complex_int": 20,
    "complex_real_to_complex_float": 21,
    "complex_int_to_complex_float": 22,
    "complex_float_to_complex_int": 23,
    "ptr_to_member_function": 24,
    "ptr_to_member_data": 25,
    "boolean": 26,
    "integral": 27,
    "floating_to_int": 28,
    "int_to_float_legacy": 29,
    "floating": 39,
}
# (Duplicates for the keywords actually emitted by upstream's pretty printer in
# the test corpus: the codegen we ingest uses int_to_float = 29 for s32→float
# rather than 10; we keep the table loose so both spellings round-trip.)
_CAST_KIND["int_to_float"] = 29
_CAST_KIND_INV: dict[int, str] = {}
for _k, _v in _CAST_KIND.items():
    _CAST_KIND_INV.setdefault(_v, _k)


_LINKAGE_KIND = {
    "external": 0,
    "available_externally": 1,
    "linkonce": 2,
    "linkonce_odr": 3,
    "weak": 4,
    "weak_odr": 5,
    "extern_weak": 6,
    "internal": 7,
    "cir_private": 8,
    "common": 9,
}
_LINKAGE_KIND_INV = {v: k for k, v in _LINKAGE_KIND.items()}


# inline_kind: 0 = no, 1 = no_inline, 2 = always, 3 = inline_hint (best guess
# from upstream — only `no_inline` is actually exercised in the fixtures).
_INLINE_KIND = {"no": 0, "no_inline": 1, "always_inline": 2, "inline_hint": 3}
_INLINE_KIND_INV = {v: k for k, v in _INLINE_KIND.items()}


def _parse_keyword_kind(parser: Parser, table: dict[str, int]) -> IntegerAttr:
    """Parse one of the keywords in `table` and return the corresponding
    32-bit integer attribute."""
    kw = parser.parse_identifier()
    if kw not in table:
        parser.raise_error(f"expected one of {sorted(table)} but got '{kw}'")
    return IntegerAttr(table[kw], 32)


def _print_kind(printer: Printer, attr: IntegerAttr, table: dict[int, str]) -> None:
    printer.print_string(table.get(attr.value.data, str(attr.value.data)))


def _props(
    items: dict[str, Attribute | None],
) -> dict[str, Attribute]:
    """Drop None values from a property dict so it is acceptable to
    `IRDLOperation.create`."""
    return {k: v for k, v in items.items() if v is not None}


def _typed_attr_type(attr: Attribute) -> Attribute | None:
    """Return the type carried by a typed CIR constant attribute."""
    get_type = getattr(attr, "get_type", None)
    if get_type is not None:
        try:
            ty = get_type()
        except Exception:
            return None
        if isinstance(ty, Attribute):
            return ty
    return None


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@irdl_op_definition
class AllocaOp(IRDLOperation):
    """`cir.alloca` — scope-local stack allocation.

    Pretty form:
        `cir.alloca` alloca-type `,` ptr-type `,`
            (`(` dyn-size `:` size-type `)`)?
            `[` name (`,` `init`)? (`,` `const`)? `]`
            (`{` attrs `}`)?
    """

    name = "cir.alloca"

    dyn_alloc_size = var_operand_def()
    addr = result_def(PointerType)

    alloca_type = prop_def(Attribute, prop_name="allocaType")
    alloc_name = opt_prop_def(StringAttr, prop_name="name")
    init = opt_prop_def(UnitAttr)
    constant = opt_prop_def(UnitAttr)
    alignment = opt_prop_def(IntegerAttr)
    annotations = opt_prop_def(ArrayAttr, prop_name="annotations")

    @classmethod
    def parse(cls, parser: Parser) -> AllocaOp:
        alloca_type = parser.parse_type()
        parser.parse_punctuation(",")
        ptr_type = parser.parse_type()
        parser.parse_punctuation(",")
        dyn_operands: list[SSAValue] = []
        # Two legacy forms for the dynamic-size operand:
        #   1. parenthesised  `(%size : !sizeT), [name...]`
        #   2. comma-form     `%size : !sizeT, [name...]`
        # The comma-form matches upstream cir-opt; the parenthesised form is kept
        # for backwards compatibility with existing IR.
        if parser.parse_optional_punctuation("(") is not None:
            dyn = parser.parse_unresolved_operand()
            parser.parse_punctuation(":")
            dyn_type = parser.parse_type()
            parser.parse_punctuation(")")
            dyn_operands.append(parser.resolve_operand(dyn, dyn_type))
            parser.parse_punctuation(",")
        else:
            dyn = parser.parse_optional_unresolved_operand()
            if dyn is not None:
                parser.parse_punctuation(":")
                dyn_type = parser.parse_type()
                dyn_operands.append(parser.resolve_operand(dyn, dyn_type))
                parser.parse_punctuation(",")
        parser.parse_punctuation("[")
        alloc_name_t = parser.parse_optional_str_literal()
        alloc_name = "" if alloc_name_t is None else alloc_name_t
        init = None
        constant = None
        while parser.parse_optional_punctuation(",") is not None:
            kw = parser.parse_identifier()
            if kw == "init":
                init = UnitAttr()
            elif kw == "const":
                constant = UnitAttr()
            else:
                parser.raise_error(f"unexpected alloca flag '{kw}'")
        parser.parse_punctuation("]")
        attrs = parser.parse_optional_dictionary_attr_dict()
        return cls.create(
            operands=dyn_operands,
            result_types=[ptr_type],
            properties=_props(
                {
                    "allocaType": alloca_type,
                    "name": StringAttr(alloc_name),
                    "init": init,
                    "constant": constant,
                    "alignment": attrs.pop("alignment", None),
                    "annotations": attrs.pop("annotations", None),
                }
            ),
            attributes=attrs,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_attribute(self.alloca_type)
        printer.print_string(", ")
        printer.print_attribute(self.addr.type)
        printer.print_string(", ")
        if self.dyn_alloc_size:
            printer.print_operand(self.dyn_alloc_size[0])
            printer.print_string(" : ")
            printer.print_attribute(self.dyn_alloc_size[0].type)
            printer.print_string(", ")
        printer.print_string("[")
        printer.print_string_literal(
            self.alloc_name.data if self.alloc_name is not None else ""
        )
        if self.init is not None:
            printer.print_string(", init")
        if self.constant is not None:
            printer.print_string(", const")
        printer.print_string("]")
        rest: dict[str, Attribute] = dict(self.attributes)
        if self.alignment is not None:
            rest["alignment"] = self.alignment
        if self.annotations is not None:
            rest["annotations"] = self.annotations
        if rest:
            printer.print_string(" ")
            printer.print_attr_dict(rest)


@irdl_op_definition
class GlobalOp(IRDLOperation):
    """`cir.global` — module-level global variable.

    Pretty form (subset used by the fixtures):

        `cir.global`
            ("private" | "public" | "nested")?
            (`constant`)? (`comdat`)?
            linkage-keyword?
            (`dso_local`)?
            @sym
            (`=` initial-value (`:` type)?)?
            (`{` attrs `}`)?
    """

    name = "cir.global"

    sym_name = prop_def(StringAttr)
    sym_type = prop_def(Attribute)
    sym_visibility = opt_prop_def(StringAttr)
    linkage = prop_def(IntegerAttr)
    global_visibility = prop_def(VisibilityAttr)
    alignment = opt_prop_def(IntegerAttr)
    constant = opt_prop_def(UnitAttr)
    dso_local = opt_prop_def(UnitAttr)
    initial_value = opt_prop_def(Attribute)
    tls_model = opt_prop_def(IntegerAttr, prop_name="tls_model")
    section = opt_prop_def(StringAttr)
    comdat = opt_prop_def(UnitAttr)
    annotations = opt_prop_def(ArrayAttr)
    addr_space = opt_prop_def(Attribute, prop_name="addr_space")
    has_ctor = opt_prop_def(UnitAttr)
    has_dtor = opt_prop_def(UnitAttr)

    ctor_region = region_def()
    dtor_region = region_def()

    traits = traits_def(NoTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> GlobalOp:
        sym_visibility: StringAttr | None = None
        # Visibility may be a quoted string ("private") or a keyword.
        vis_str = parser.parse_optional_str_literal()
        if vis_str is not None:
            sym_visibility = StringAttr(vis_str)
        else:
            for kw in ("private", "public", "nested"):
                if parser.parse_optional_keyword(kw) is not None:
                    sym_visibility = StringAttr(kw)
                    break

        # Optional linkage spec as a quoted string (e.g. `"private"`,
        # `"internal"`). This is distinct from the visibility quoted string
        # above — visibility is one of "private"/"public"/"nested"; anything
        # else is treated as a linkage specifier.
        linkage_str = parser.parse_optional_str_literal()
        if linkage_str is not None:
            # If the visibility wasn't set yet and this string actually names
            # a visibility, treat it as visibility.
            if sym_visibility is None and linkage_str in (
                "private",
                "public",
                "nested",
            ):
                sym_visibility = StringAttr(linkage_str)
                linkage_str = None

        constant = (
            UnitAttr()
            if parser.parse_optional_keyword("constant") is not None
            else None
        )
        comdat = (
            UnitAttr() if parser.parse_optional_keyword("comdat") is not None else None
        )

        linkage_val: int | None = None
        if linkage_str is not None:
            if linkage_str not in _LINKAGE_KIND:
                parser.raise_error(
                    f"unknown linkage specifier {linkage_str!r}; expected one of "
                    f"{sorted(_LINKAGE_KIND)}"
                )
            linkage_val = _LINKAGE_KIND[linkage_str]
        else:
            for kw, val in _LINKAGE_KIND.items():
                if parser.parse_optional_keyword(kw) is not None:
                    linkage_val = val
                    break
        if linkage_val is None:
            linkage_val = 0  # default to `external`

        dso_local = (
            UnitAttr()
            if parser.parse_optional_keyword("dso_local") is not None
            else None
        )

        sym_name = parser.parse_symbol_name()

        initial_value: Attribute | None = None
        sym_type: Attribute | None = None
        has_ctor: UnitAttr | None = None
        has_dtor: UnitAttr | None = None
        ctor_region = Region()
        dtor_region = Region()
        if parser.parse_optional_punctuation("=") is not None:
            # `= ctor : <type> { region }` form.
            if parser.parse_optional_keyword("ctor") is not None:
                has_ctor = UnitAttr()
                parser.parse_punctuation(":")
                sym_type = parser.parse_type()
                ctor_region = parser.parse_region()
                # Optional trailing `dtor { region }`.
                if parser.parse_optional_keyword("dtor") is not None:
                    has_dtor = UnitAttr()
                    dtor_region = parser.parse_region()
            else:
                initial_value = parser.parse_attribute()
                # Some attribute forms carry their own type, others use a `:` suffix.
                inner_type = _typed_attr_type(initial_value)
                if parser.parse_optional_punctuation(":") is not None:
                    sym_type = parser.parse_type()
                elif inner_type is not None:
                    sym_type = inner_type
                else:
                    parser.raise_error(
                        "cir.global initial value must carry a type or "
                        "be followed by `:`"
                    )
                # Optional trailing `dtor { region }`.
                if parser.parse_optional_keyword("dtor") is not None:
                    has_dtor = UnitAttr()
                    dtor_region = parser.parse_region()
        elif parser.parse_optional_punctuation(":") is not None:
            # No-init form: `@sym : <type>`.
            sym_type = parser.parse_type()

        attrs = parser.parse_optional_dictionary_attr_dict()
        alignment = attrs.pop("alignment", None)
        if alignment is not None and not isinstance(alignment, IntegerAttr):
            parser.raise_error("alignment must be an integer attribute")

        if sym_type is None:
            # External declaration with no initial value: type must be in attrs.
            sym_type = attrs.pop("sym_type", None)
            if sym_type is None:
                parser.raise_error(
                    "cir.global without initial value needs a `sym_type` attribute"
                )

        properties: dict[str, Attribute | None] = {
            "sym_name": sym_name,
            "sym_type": sym_type,
            "sym_visibility": sym_visibility,
            "linkage": IntegerAttr(linkage_val, 32),
            "global_visibility": VisibilityAttr("default"),
            "constant": constant,
            "comdat": comdat,
            "dso_local": dso_local,
            "initial_value": initial_value,
            "alignment": alignment,
            "has_ctor": has_ctor,
            "has_dtor": has_dtor,
        }
        return cls.create(
            properties=_props(properties),
            attributes=attrs,
            regions=[ctor_region, dtor_region],
        )

    def print(self, printer: Printer) -> None:
        if self.sym_visibility is not None:
            printer.print_string(" ")
            printer.print_string_literal(self.sym_visibility.data)
        if self.constant is not None:
            printer.print_string(" constant")
        if self.comdat is not None:
            printer.print_string(" comdat")
        linkage_val = self.linkage.value.data
        kw = _LINKAGE_KIND_INV.get(linkage_val)
        if kw:
            printer.print_string(" ")
            printer.print_string(kw)
        if self.dso_local is not None:
            printer.print_string(" dso_local")
        printer.print_string(" ")
        printer.print_symbol_name(self.sym_name.data)
        if self.has_ctor is not None:
            printer.print_string(" = ctor : ")
            printer.print_attribute(self.sym_type)
            printer.print_string(" ")
            printer.print_region(self.ctor_region)
            if self.has_dtor is not None:
                printer.print_string(" dtor ")
                printer.print_region(self.dtor_region)
        elif self.initial_value is not None:
            printer.print_string(" = ")
            printer.print_attribute(self.initial_value)
            if _typed_attr_type(self.initial_value) is None:
                printer.print_string(" : ")
                printer.print_attribute(self.sym_type)
            if self.has_dtor is not None:
                printer.print_string(" dtor ")
                printer.print_region(self.dtor_region)
        else:
            printer.print_string(" : ")
            printer.print_attribute(self.sym_type)
        rest: dict[str, Attribute] = dict(self.attributes)
        if self.alignment is not None:
            rest["alignment"] = self.alignment
        if rest:
            printer.print_string(" ")
            printer.print_attr_dict(rest)


@irdl_op_definition
class FuncOp(IRDLOperation):
    """`cir.func` — function definition or declaration.

    Pretty form (greatly simplified — covers what the fixtures actually use):

        `cir.func`
            (`no_inline` | `always_inline` | `inline_hint`)?
            (`builtin`)? (`coroutine`)? (`lambda`)? (`no_proto`)? (`comdat`)?
            (linkage-keyword)?
            (`private` | `public` | `nested`)?
            (`dso_local`)?
            @sym
            `(` (named-args | type-list) `)`
            (`->` ret-type)?
            (`{` body `}`)?
    """

    name = "cir.func"

    sym_name = prop_def(StringAttr)
    function_type = prop_def(FuncType)
    sym_visibility = opt_prop_def(StringAttr)
    linkage = prop_def(IntegerAttr)
    global_visibility = prop_def(VisibilityAttr)
    inline_kind = opt_prop_def(IntegerAttr)
    dso_local = opt_prop_def(UnitAttr)
    builtin = opt_prop_def(UnitAttr)
    coroutine = opt_prop_def(UnitAttr)
    lambda_ = opt_prop_def(UnitAttr, prop_name="lambda")
    no_proto = opt_prop_def(UnitAttr)
    extra_attrs = opt_prop_def(Attribute)
    arg_attrs = opt_prop_def(ArrayAttr)
    res_attrs = opt_prop_def(ArrayAttr)
    aliasee = opt_prop_def(Attribute)
    global_ctor_priority = opt_prop_def(IntegerAttr)
    global_dtor_priority = opt_prop_def(IntegerAttr)
    cxx_special_member = opt_prop_def(Attribute)
    annotations = opt_prop_def(ArrayAttr)
    comdat = opt_prop_def(UnitAttr)

    body = region_def()

    @classmethod
    def parse(cls, parser: Parser) -> FuncOp:
        inline_kind: IntegerAttr | None = None
        for kw, code in (
            ("no_inline", 1),
            ("always_inline", 2),
            ("inline_hint", 3),
        ):
            if parser.parse_optional_keyword(kw) is not None:
                inline_kind = IntegerAttr(code, 32)
                break

        unit_flags: dict[str, Attribute | None] = {
            "builtin": None,
            "coroutine": None,
            "lambda": None,
            "no_proto": None,
            "comdat": None,
        }
        for flag in ("builtin", "coroutine", "lambda", "no_proto", "comdat"):
            if parser.parse_optional_keyword(flag) is not None:
                unit_flags[flag] = UnitAttr()

        linkage_val = 0
        for kw, val in _LINKAGE_KIND.items():
            if parser.parse_optional_keyword(kw) is not None:
                linkage_val = val
                break

        sym_visibility: StringAttr | None = None
        for kw in ("private", "public", "nested"):
            if parser.parse_optional_keyword(kw) is not None:
                sym_visibility = StringAttr(kw)
                break

        dso_local = (
            UnitAttr()
            if parser.parse_optional_keyword("dso_local") is not None
            else None
        )

        sym_name = parser.parse_symbol_name()

        # Argument list — either typed-only (declaration) or named (definition).
        parser.parse_punctuation("(")
        input_types: list[Attribute] = []
        entry_args: list[Parser.Argument] = []
        is_var_arg = False
        if parser.parse_optional_punctuation(")") is None:
            while True:
                if parser.parse_optional_punctuation("...") is not None:
                    is_var_arg = True
                    parser.parse_punctuation(")")
                    break
                arg = parser.parse_optional_argument()
                if arg is not None:
                    arg.location = parser.parse_optional_location()
                    entry_args.append(arg)
                    input_types.append(arg.type)
                else:
                    input_types.append(parser.parse_type())
                if parser.parse_optional_punctuation(",") is None:
                    parser.parse_punctuation(")")
                    break

        return_type: Attribute = VoidType()
        if parser.parse_optional_punctuation("->") is not None:
            return_type = parser.parse_type()

        function_type = FuncType(input_types, return_type, is_var_arg=is_var_arg)

        # Optional body region.
        body = parser.parse_optional_region(entry_args if entry_args else None)
        if body is None:
            body = Region()

        properties: dict[str, Attribute | None] = {
            "sym_name": sym_name,
            "function_type": function_type,
            "linkage": IntegerAttr(linkage_val, 32),
            "global_visibility": VisibilityAttr("default"),
            "sym_visibility": sym_visibility,
            "dso_local": dso_local,
            "inline_kind": inline_kind,
            **unit_flags,
        }
        return cls.create(properties=_props(properties), regions=[body])

    def print(self, printer: Printer) -> None:
        if self.inline_kind is not None:
            kw = _INLINE_KIND_INV.get(self.inline_kind.value.data)
            if kw and kw != "no":
                printer.print_string(" ")
                printer.print_string(kw)
        for flag_attr, kw in (
            (self.builtin, "builtin"),
            (self.coroutine, "coroutine"),
            (self.lambda_, "lambda"),
            (self.no_proto, "no_proto"),
            (self.comdat, "comdat"),
        ):
            if flag_attr is not None:
                printer.print_string(" ")
                printer.print_string(kw)
        linkage_val = self.linkage.value.data
        if linkage_val != 0:
            kw2 = _LINKAGE_KIND_INV.get(linkage_val)
            if kw2:
                printer.print_string(" ")
                printer.print_string(kw2)
        if self.sym_visibility is not None:
            printer.print_string(" ")
            printer.print_string(self.sym_visibility.data)
        if self.dso_local is not None:
            printer.print_string(" dso_local")
        printer.print_string(" ")
        printer.print_symbol_name(self.sym_name.data)

        printer.print_string("(")
        if self.body.blocks:
            block_args = list(self.body.blocks[0].args)
            printer.print_list(
                block_args,
                lambda a: (
                    (
                        printer.print_operand(a),
                        printer.print_string(": "),
                        printer.print_attribute(a.type),
                    )
                    and None
                ),
            )
        else:
            printer.print_list(self.function_type.inputs.data, printer.print_attribute)
        if self.function_type.varargs:
            if self.function_type.inputs.data:
                printer.print_string(", ")
            printer.print_string("...")
        printer.print_string(")")
        if not self.function_type.has_void_return:
            printer.print_string(" -> ")
            printer.print_attribute(self.function_type.return_type)
        if self.body.blocks:
            printer.print_string(" ")
            printer.print_region(self.body, print_entry_block_args=False)


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """`cir.return` — return from a function.

    Pretty: `cir.return` ($val `:` type)?
    """

    name = "cir.return"

    arguments = var_operand_def()
    traits = traits_def(IsTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> ReturnOp:
        unresolved = parser.parse_optional_unresolved_operand()
        operands: list[SSAValue] = []
        if unresolved is not None:
            parser.parse_punctuation(":")
            ty = parser.parse_type()
            operands.append(parser.resolve_operand(unresolved, ty))
        return cls.create(operands=operands)

    def print(self, printer: Printer) -> None:
        if self.arguments:
            printer.print_string(" ")
            printer.print_operand(self.arguments[0])
            printer.print_string(" : ")
            printer.print_attribute(self.arguments[0].type)


@irdl_op_definition
class CallOp(IRDLOperation):
    """`cir.call` — call a function.

    Pretty: `cir.call` @callee `(` $args `)` (`nothrow`)? (`exception`)?
                       `:` `(` arg-types `)` `->` (ret-type | `()`)
    """

    name = "cir.call"

    arg_ops = var_operand_def()
    results_ = var_result_def()

    callee = opt_prop_def(FlatSymbolRefAttr)
    side_effect = opt_prop_def(IntegerAttr)
    nothrow = opt_prop_def(UnitAttr)
    exception = opt_prop_def(UnitAttr)
    extra_attrs = opt_prop_def(Attribute)
    calling_conv = opt_prop_def(Attribute)
    arg_attrs = opt_prop_def(ArrayAttr)
    res_attrs = opt_prop_def(ArrayAttr)

    @classmethod
    def parse(cls, parser: Parser) -> CallOp:
        # Two callee forms:
        #   - direct:   `@symbol`
        #   - indirect: `%ssa-value`  (the callee is the first operand)
        sym: SymbolRefAttr | None = None
        indirect_callee: UnresolvedOperand | None = None
        indirect_callee = parser.parse_optional_unresolved_operand()
        if indirect_callee is None:
            attr = parser.parse_attribute()
            if not isinstance(attr, SymbolRefAttr) or attr.nested_references.data:
                parser.raise_error("expected a flat callee symbol reference")
            sym = attr
        parser.parse_punctuation("(")
        arg_uns: list[UnresolvedOperand] = []
        if parser.parse_optional_punctuation(")") is None:
            arg_uns.append(parser.parse_unresolved_operand())
            while parser.parse_optional_punctuation(",") is not None:
                arg_uns.append(parser.parse_unresolved_operand())
            parser.parse_punctuation(")")
        nothrow = (
            UnitAttr() if parser.parse_optional_keyword("nothrow") is not None else None
        )
        exception = (
            UnitAttr()
            if parser.parse_optional_keyword("exception") is not None
            else None
        )
        side_effect: IntegerAttr | None = None
        if parser.parse_optional_keyword("side_effect") is not None:
            parser.parse_punctuation("(")
            kw = parser.parse_identifier()
            parser.parse_punctuation(")")
            # 0 = all (default), 1 = const, 2 = pure (best-effort mapping).
            side_effect = IntegerAttr({"all": 0, "const": 1, "pure": 2}.get(kw, 0), 32)
        parser.parse_punctuation(":")
        parser.parse_punctuation("(")
        all_types: list[Attribute] = []
        if parser.parse_optional_punctuation(")") is None:
            all_types.append(parser.parse_type())
            while parser.parse_optional_punctuation(",") is not None:
                all_types.append(parser.parse_type())
            parser.parse_punctuation(")")
        parser.parse_punctuation("->")
        result_types: list[Attribute] = []
        if parser.parse_optional_punctuation("(") is not None:
            if parser.parse_optional_punctuation(")") is None:
                result_types.append(parser.parse_type())
                while parser.parse_optional_punctuation(",") is not None:
                    result_types.append(parser.parse_type())
                parser.parse_punctuation(")")
        else:
            result_types.append(parser.parse_type())
        # For indirect calls, the type list begins with the callee pointer type.
        if indirect_callee is not None:
            if not all_types:
                parser.raise_error(
                    "indirect cir.call expects callee type as first operand-list type"
                )
            callee_type = all_types[0]
            arg_types = all_types[1:]
            if len(arg_types) != len(arg_uns):
                parser.raise_error("operand and type-list lengths differ for cir.call")
            operands: list[SSAValue] = [
                parser.resolve_operand(indirect_callee, callee_type)
            ]
            operands.extend(
                parser.resolve_operand(u, t) for u, t in zip(arg_uns, arg_types)
            )
        else:
            arg_types = all_types
            if len(arg_types) != len(arg_uns):
                parser.raise_error("operand and type-list lengths differ for cir.call")
            operands = [
                parser.resolve_operand(u, t) for u, t in zip(arg_uns, arg_types)
            ]
        return cls.create(
            operands=operands,
            result_types=result_types,
            properties=_props(
                {
                    "callee": sym,
                    "nothrow": nothrow,
                    "exception": exception,
                    "side_effect": side_effect,
                }
            ),
        )

    def print(self, printer: Printer) -> None:
        # Indirect calls: callee is the first operand and `callee` symbol is unset.
        # Direct calls:   callee is a `@sym` attribute, all `arg_ops` are real args.
        printer.print_string(" ")
        if self.callee is not None:
            printer.print_attribute(self.callee)
            real_args = list(self.arg_ops)
        else:
            assert len(self.arg_ops) >= 1, "indirect cir.call requires callee operand"
            printer.print_operand(self.arg_ops[0])
            real_args = list(self.arg_ops[1:])
        printer.print_string("(")
        printer.print_list(real_args, printer.print_operand)
        printer.print_string(")")
        if self.nothrow is not None:
            printer.print_string(" nothrow")
        if self.exception is not None:
            printer.print_string(" exception")
        if self.side_effect is not None:
            kw = {0: "all", 1: "const", 2: "pure"}.get(
                self.side_effect.value.data, "all"
            )
            printer.print_string(f" side_effect({kw})")
        printer.print_string(" : (")
        printer.print_list(self.arg_ops, lambda v: printer.print_attribute(v.type))
        printer.print_string(") -> ")
        if not self.results_:
            printer.print_string("()")
        elif len(self.results_) == 1:
            printer.print_attribute(self.results_[0].type)
        else:
            printer.print_string("(")
            printer.print_list(self.results_, lambda v: printer.print_attribute(v.type))
            printer.print_string(")")


@irdl_op_definition
class GetGlobalOp(IRDLOperation):
    """`cir.get_global` — take the address of a global symbol.

    Pretty: `cir.get_global` (`thread_local`)? @sym `:` ptr-type
    """

    name = "cir.get_global"

    addr = result_def(PointerType)
    glob_name = prop_def(FlatSymbolRefAttr, prop_name="name")
    tls = opt_prop_def(UnitAttr)

    @classmethod
    def parse(cls, parser: Parser) -> GetGlobalOp:
        tls = (
            UnitAttr()
            if parser.parse_optional_keyword("thread_local") is not None
            else None
        )
        sym = parser.parse_attribute()
        if not isinstance(sym, SymbolRefAttr) or sym.nested_references.data:
            parser.raise_error("expected a flat symbol reference")
        parser.parse_punctuation(":")
        ptr_type = parser.parse_type()
        return cls.create(
            result_types=[ptr_type],
            properties=_props({"name": sym, "tls": tls}),
        )

    def print(self, printer: Printer) -> None:
        if self.tls is not None:
            printer.print_string(" thread_local")
        printer.print_string(" ")
        printer.print_attribute(self.glob_name)
        printer.print_string(" : ")
        printer.print_attribute(self.addr.type)


@irdl_op_definition
class ConstantOp(IRDLOperation):
    """`cir.const` — materialise a constant attribute as an SSA value.

    Pretty: `cir.const` $value
    The attribute carries its own type.
    """

    name = "cir.const"

    res = result_def()
    value = prop_def(Attribute)

    @classmethod
    def parse(cls, parser: Parser) -> ConstantOp:
        attr = parser.parse_attribute()
        result_type = _typed_attr_type(attr)
        if result_type is None:
            parser.raise_error("cir.const requires a typed attribute")
        return cls.create(
            result_types=[result_type],
            properties={"value": attr},
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_attribute(self.value)


@irdl_op_definition
class LoadOp(IRDLOperation):
    """`cir.load` — load through a pointer.

    Pretty: `cir.load` (`deref`)? (`volatile`)? (`nontemporal`)?
                       (`align` `(` N `)`)? $addr `:` ptr-type `,` val-type
    """

    name = "cir.load"

    addr = operand_def(PointerType)
    res = result_def()

    alignment = opt_prop_def(IntegerAttr)
    is_volatile = opt_prop_def(UnitAttr)
    is_nontemporal = opt_prop_def(UnitAttr)
    is_deref = opt_prop_def(UnitAttr, prop_name="isDeref")
    mem_order = opt_prop_def(IntegerAttr)
    sync_scope = opt_prop_def(IntegerAttr)
    tbaa = opt_prop_def(Attribute)

    @classmethod
    def parse(cls, parser: Parser) -> LoadOp:
        is_deref = (
            UnitAttr() if parser.parse_optional_keyword("deref") is not None else None
        )
        is_volatile = (
            UnitAttr()
            if parser.parse_optional_keyword("volatile") is not None
            else None
        )
        is_nontemporal = (
            UnitAttr()
            if parser.parse_optional_keyword("nontemporal") is not None
            else None
        )
        alignment: IntegerAttr | None = None
        if parser.parse_optional_keyword("align") is not None:
            parser.parse_punctuation("(")
            alignment = IntegerAttr(parser.parse_integer(), 64)
            parser.parse_punctuation(")")
        addr_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        ptr_type = parser.parse_type()
        parser.parse_punctuation(",")
        val_type = parser.parse_type()
        addr = parser.resolve_operand(addr_un, ptr_type)
        return cls.create(
            operands=[addr],
            result_types=[val_type],
            properties=_props(
                {
                    "isDeref": is_deref,
                    "is_volatile": is_volatile,
                    "is_nontemporal": is_nontemporal,
                    "alignment": alignment,
                }
            ),
        )

    def print(self, printer: Printer) -> None:
        if self.is_deref is not None:
            printer.print_string(" deref")
        if self.is_volatile is not None:
            printer.print_string(" volatile")
        if self.is_nontemporal is not None:
            printer.print_string(" nontemporal")
        if self.alignment is not None:
            printer.print_string(" align(")
            printer.print_string(str(self.alignment.value.data))
            printer.print_string(")")
        printer.print_string(" ")
        printer.print_operand(self.addr)
        printer.print_string(" : ")
        printer.print_attribute(self.addr.type)
        printer.print_string(", ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class StoreOp(IRDLOperation):
    """`cir.store` — store a value through a pointer.

    Pretty: `cir.store` (`volatile`)? (`nontemporal`)? (`align` `(` N `)`)?
                        $value `,` $addr `:` val-type `,` ptr-type
    """

    name = "cir.store"

    value = operand_def()
    addr = operand_def(PointerType)

    alignment = opt_prop_def(IntegerAttr)
    is_volatile = opt_prop_def(UnitAttr)
    is_nontemporal = opt_prop_def(UnitAttr)
    mem_order = opt_prop_def(IntegerAttr)
    sync_scope = opt_prop_def(IntegerAttr)
    tbaa = opt_prop_def(Attribute)

    @classmethod
    def parse(cls, parser: Parser) -> StoreOp:
        is_volatile = (
            UnitAttr()
            if parser.parse_optional_keyword("volatile") is not None
            else None
        )
        is_nontemporal = (
            UnitAttr()
            if parser.parse_optional_keyword("nontemporal") is not None
            else None
        )
        alignment: IntegerAttr | None = None
        if parser.parse_optional_keyword("align") is not None:
            parser.parse_punctuation("(")
            alignment = IntegerAttr(parser.parse_integer(), 64)
            parser.parse_punctuation(")")
        val_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        addr_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        val_type = parser.parse_type()
        parser.parse_punctuation(",")
        ptr_type = parser.parse_type()
        return cls.create(
            operands=[
                parser.resolve_operand(val_un, val_type),
                parser.resolve_operand(addr_un, ptr_type),
            ],
            properties=_props(
                {
                    "is_volatile": is_volatile,
                    "is_nontemporal": is_nontemporal,
                    "alignment": alignment,
                }
            ),
        )

    def print(self, printer: Printer) -> None:
        if self.is_volatile is not None:
            printer.print_string(" volatile")
        if self.is_nontemporal is not None:
            printer.print_string(" nontemporal")
        if self.alignment is not None:
            printer.print_string(" align(")
            printer.print_string(str(self.alignment.value.data))
            printer.print_string(")")
        printer.print_string(" ")
        printer.print_operand(self.value)
        printer.print_string(", ")
        printer.print_operand(self.addr)
        printer.print_string(" : ")
        printer.print_attribute(self.value.type)
        printer.print_string(", ")
        printer.print_attribute(self.addr.type)


@irdl_op_definition
class CastOp(IRDLOperation):
    """`cir.cast` — type conversion.

    Pretty: `cir.cast` kind $src `:` src-type `->` dst-type
    """

    name = "cir.cast"

    src = operand_def()
    res = result_def()
    kind = prop_def(IntegerAttr)

    @classmethod
    def parse(cls, parser: Parser) -> CastOp:
        kind = _parse_keyword_kind(parser, _CAST_KIND)
        src_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        src_type = parser.parse_type()
        parser.parse_punctuation("->")
        dst_type = parser.parse_type()
        return cls.create(
            operands=[parser.resolve_operand(src_un, src_type)],
            result_types=[dst_type],
            properties={"kind": kind},
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        _print_kind(printer, self.kind, _CAST_KIND_INV)
        printer.print_string(" ")
        printer.print_operand(self.src)
        printer.print_string(" : ")
        printer.print_attribute(self.src.type)
        printer.print_string(" -> ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class UnaryOp(IRDLOperation):
    """`cir.unary` — unary arithmetic.

    Pretty: `cir.unary` `(` kind `,` $src `)` (`nsw`)? (`nuw`)? `:` in-type `,` out-type
    """

    name = "cir.unary"

    input = operand_def()
    res = result_def()

    kind = prop_def(IntegerAttr)
    no_signed_wrap = opt_prop_def(UnitAttr)
    no_unsigned_wrap = opt_prop_def(UnitAttr)

    @classmethod
    def parse(cls, parser: Parser) -> UnaryOp:
        parser.parse_punctuation("(")
        kind = _parse_keyword_kind(parser, _UNARY_OP_KIND)
        parser.parse_punctuation(",")
        src_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(")")
        nsw = UnitAttr() if parser.parse_optional_keyword("nsw") is not None else None
        nuw = UnitAttr() if parser.parse_optional_keyword("nuw") is not None else None
        parser.parse_punctuation(":")
        in_type = parser.parse_type()
        parser.parse_punctuation(",")
        out_type = parser.parse_type()
        return cls.create(
            operands=[parser.resolve_operand(src_un, in_type)],
            result_types=[out_type],
            properties=_props(
                {"kind": kind, "no_signed_wrap": nsw, "no_unsigned_wrap": nuw}
            ),
        )

    def print(self, printer: Printer) -> None:
        printer.print_string("(")
        _print_kind(printer, self.kind, _UNARY_OP_KIND_INV)
        printer.print_string(", ")
        printer.print_operand(self.input)
        printer.print_string(")")
        if self.no_signed_wrap is not None:
            printer.print_string(" nsw")
        if self.no_unsigned_wrap is not None:
            printer.print_string(" nuw")
        printer.print_string(" : ")
        printer.print_attribute(self.input.type)
        printer.print_string(", ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class BinOp(IRDLOperation):
    """`cir.binop` — binary arithmetic.

    Pretty: `cir.binop` `(` kind `,` $lhs `,` $rhs `)` (`nsw`)? (`nuw`)? `:` type
    """

    name = "cir.binop"

    lhs = operand_def()
    rhs = operand_def()
    res = result_def()

    kind = prop_def(IntegerAttr)
    no_signed_wrap = opt_prop_def(UnitAttr)
    no_unsigned_wrap = opt_prop_def(UnitAttr)

    @classmethod
    def parse(cls, parser: Parser) -> BinOp:
        parser.parse_punctuation("(")
        kind = _parse_keyword_kind(parser, _BIN_OP_KIND)
        parser.parse_punctuation(",")
        lhs_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        rhs_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(")")
        nsw = UnitAttr() if parser.parse_optional_keyword("nsw") is not None else None
        nuw = UnitAttr() if parser.parse_optional_keyword("nuw") is not None else None
        parser.parse_punctuation(":")
        ty = parser.parse_type()
        return cls.create(
            operands=[
                parser.resolve_operand(lhs_un, ty),
                parser.resolve_operand(rhs_un, ty),
            ],
            result_types=[ty],
            properties=_props(
                {"kind": kind, "no_signed_wrap": nsw, "no_unsigned_wrap": nuw}
            ),
        )

    def print(self, printer: Printer) -> None:
        printer.print_string("(")
        _print_kind(printer, self.kind, _BIN_OP_KIND_INV)
        printer.print_string(", ")
        printer.print_operand(self.lhs)
        printer.print_string(", ")
        printer.print_operand(self.rhs)
        printer.print_string(")")
        if self.no_signed_wrap is not None:
            printer.print_string(" nsw")
        if self.no_unsigned_wrap is not None:
            printer.print_string(" nuw")
        printer.print_string(" : ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class CmpOp(IRDLOperation):
    """`cir.cmp` — comparison.

    Pretty: `cir.cmp` `(` kind `,` $lhs `,` $rhs `)` `:` operand-type `,` `!cir.bool`
    """

    name = "cir.cmp"

    lhs = operand_def()
    rhs = operand_def()
    res = result_def(BoolType)
    kind = prop_def(IntegerAttr)

    @classmethod
    def parse(cls, parser: Parser) -> CmpOp:
        parser.parse_punctuation("(")
        kind = _parse_keyword_kind(parser, _CMP_OP_KIND)
        parser.parse_punctuation(",")
        lhs_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        rhs_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(")")
        parser.parse_punctuation(":")
        operand_type = parser.parse_type()
        parser.parse_punctuation(",")
        res_type = parser.parse_type()
        if not isinstance(res_type, BoolType):
            parser.raise_error("expected result type !cir.bool")
        return cls.create(
            operands=[
                parser.resolve_operand(lhs_un, operand_type),
                parser.resolve_operand(rhs_un, operand_type),
            ],
            result_types=[res_type],
            properties={"kind": kind},
        )

    def print(self, printer: Printer) -> None:
        printer.print_string("(")
        _print_kind(printer, self.kind, _CMP_OP_KIND_INV)
        printer.print_string(", ")
        printer.print_operand(self.lhs)
        printer.print_string(", ")
        printer.print_operand(self.rhs)
        printer.print_string(") : ")
        printer.print_attribute(self.lhs.type)
        printer.print_string(", ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class IfOp(IRDLOperation):
    """`cir.if` — conditional branch with two regions.

    Pretty: `cir.if` $cond `{` then-region `}` (`else` `{` else-region `}`)?
    """

    name = "cir.if"

    cond = operand_def(BoolType)

    then_region = region_def()
    else_region = region_def()
    traits = traits_def(NoTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> IfOp:
        cond_un = parser.parse_unresolved_operand()
        cond = parser.resolve_operand(cond_un, BoolType())
        then_region = parser.parse_region()
        else_region = Region()
        if parser.parse_optional_keyword("else") is not None:
            else_region = parser.parse_region()
        return cls.create(operands=[cond], regions=[then_region, else_region])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.cond)
        printer.print_string(" ")
        printer.print_region(self.then_region)
        if self.else_region.blocks:
            printer.print_string(" else ")
            printer.print_region(self.else_region)


@irdl_op_definition
class ScopeOp(IRDLOperation):
    """`cir.scope` — lexical scope with optional yielded results.

    Pretty: `cir.scope` `{` body `}` (`:` (`(`type-list`)`|type))?
    """

    name = "cir.scope"

    results_ = var_result_def()
    body = region_def()
    traits = traits_def(NoTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> ScopeOp:
        body = parser.parse_region()
        result_types: list[Attribute] = []
        if parser.parse_optional_punctuation(":") is not None:
            if parser.parse_optional_punctuation("(") is not None:
                if parser.parse_optional_punctuation(")") is None:
                    result_types.append(parser.parse_type())
                    while parser.parse_optional_punctuation(",") is not None:
                        result_types.append(parser.parse_type())
                    parser.parse_punctuation(")")
            else:
                result_types.append(parser.parse_type())
        return cls.create(result_types=result_types, regions=[body])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_region(self.body)
        if self.results_:
            printer.print_string(" : ")
            if len(self.results_) == 1:
                printer.print_attribute(self.results_[0].type)
            else:
                printer.print_string("(")
                printer.print_list(
                    self.results_, lambda v: printer.print_attribute(v.type)
                )
                printer.print_string(")")


@irdl_op_definition
class WhileOp(IRDLOperation):
    """`cir.while` — top-tested loop. Two regions: condition and body.

    Pretty: `cir.while` `{` cond-region `}` `do` `{` body-region `}`
    """

    name = "cir.while"

    cond_region = region_def()
    body_region = region_def()
    traits = traits_def(NoTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> WhileOp:
        cond_region = parser.parse_region()
        parser.parse_keyword("do")
        body_region = parser.parse_region()
        return cls.create(regions=[cond_region, body_region])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_region(self.cond_region)
        printer.print_string(" do ")
        printer.print_region(self.body_region)


@irdl_op_definition
class ForOp(IRDLOperation):
    """`cir.for` — three-region C-style for-loop (cond, body, step).

    Pretty: `cir.for` `:` `cond` `{` cond `}` `body` `{` body `}` `step` `{` step `}`
    """

    name = "cir.for"

    cond_region = region_def()
    body_region = region_def()
    step_region = region_def()
    traits = traits_def(NoTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> ForOp:
        parser.parse_punctuation(":")
        parser.parse_keyword("cond")
        cond_region = parser.parse_region()
        parser.parse_keyword("body")
        body_region = parser.parse_region()
        parser.parse_keyword("step")
        step_region = parser.parse_region()
        return cls.create(regions=[cond_region, body_region, step_region])

    def print(self, printer: Printer) -> None:
        printer.print_string(" : cond ")
        printer.print_region(self.cond_region)
        printer.print_string(" body ")
        printer.print_region(self.body_region)
        printer.print_string(" step ")
        printer.print_region(self.step_region)


@irdl_op_definition
class ConditionOp(IRDLOperation):
    """`cir.condition` — terminator for the cond-region of `cir.while`/`cir.for`.

    Pretty: `cir.condition` `(` $cond `)`
    """

    name = "cir.condition"

    cond = operand_def(BoolType)
    traits = traits_def(IsTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> ConditionOp:
        parser.parse_punctuation("(")
        cond_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(")")
        cond = parser.resolve_operand(cond_un, BoolType())
        return cls.create(operands=[cond])

    def print(self, printer: Printer) -> None:
        printer.print_string("(")
        printer.print_operand(self.cond)
        printer.print_string(")")


@irdl_op_definition
class YieldOp(IRDLOperation):
    """`cir.yield` — terminator for structured control-flow regions.

    Pretty: `cir.yield` ($args `:` types)?
    """

    name = "cir.yield"

    arguments = var_operand_def()
    traits = traits_def(IsTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> YieldOp:
        unresolved: list[UnresolvedOperand] = []
        first = parser.parse_optional_unresolved_operand()
        if first is None:
            return cls.create()
        unresolved.append(first)
        while parser.parse_optional_punctuation(",") is not None:
            unresolved.append(parser.parse_unresolved_operand())
        parser.parse_punctuation(":")
        types = [parser.parse_type()]
        while parser.parse_optional_punctuation(",") is not None:
            types.append(parser.parse_type())
        if len(types) != len(unresolved):
            parser.raise_error("number of operands and types must match")
        operands = [parser.resolve_operand(u, t) for u, t in zip(unresolved, types)]
        return cls.create(operands=operands)

    def print(self, printer: Printer) -> None:
        if not self.arguments:
            return
        printer.print_string(" ")
        printer.print_list(self.arguments, printer.print_operand)
        printer.print_string(" : ")
        printer.print_list(self.arguments, lambda v: printer.print_attribute(v.type))


@irdl_op_definition
class BreakOp(IRDLOperation):
    """`cir.break` — terminator transferring control out of a loop body."""

    name = "cir.break"

    traits = traits_def(IsTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> BreakOp:
        return cls.create()

    def print(self, printer: Printer) -> None:
        pass


@irdl_op_definition
class ContinueOp(IRDLOperation):
    """`cir.continue` — terminator transferring control to the loop step."""

    name = "cir.continue"

    traits = traits_def(IsTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> ContinueOp:
        return cls.create()

    def print(self, printer: Printer) -> None:
        pass


@irdl_op_definition
class BrOp(IRDLOperation):
    """`cir.br` — unconditional branch with optional block-arg payload.

    Pretty: `cir.br` ^bbN (`(` $args `:` types `)`)?
    """

    name = "cir.br"

    arguments = var_operand_def()
    successor = var_successor_def()
    traits = traits_def(IsTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> BrOp:
        block = parser.parse_successor()
        operands: list[SSAValue] = []
        if parser.parse_optional_punctuation("(") is not None:
            uns = [parser.parse_unresolved_operand()]
            while parser.parse_optional_punctuation(",") is not None:
                uns.append(parser.parse_unresolved_operand())
            parser.parse_punctuation(":")
            types = [parser.parse_type()]
            while parser.parse_optional_punctuation(",") is not None:
                types.append(parser.parse_type())
            parser.parse_punctuation(")")
            if len(uns) != len(types):
                parser.raise_error("number of branch operands and types must match")
            operands = [parser.resolve_operand(u, t) for u, t in zip(uns, types)]
        return cls.create(operands=operands, successors=[block])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_block_name(self.successor[0])
        if self.arguments:
            printer.print_string("(")
            printer.print_list(self.arguments, printer.print_operand)
            printer.print_string(" : ")
            printer.print_list(
                self.arguments, lambda v: printer.print_attribute(v.type)
            )
            printer.print_string(")")


@irdl_op_definition
class PtrStrideOp(IRDLOperation):
    """`cir.ptr_stride` — pointer + element-stride offset.

    Pretty: `cir.ptr_stride` $base `,` $stride
            `:` `(` ptr-type `,` stride-type `)` `->` ptr-type
    """

    name = "cir.ptr_stride"

    base = operand_def(PointerType)
    stride = operand_def()
    res = result_def(PointerType)

    @classmethod
    def parse(cls, parser: Parser) -> PtrStrideOp:
        base_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        stride_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        parser.parse_punctuation("(")
        base_type = parser.parse_type()
        parser.parse_punctuation(",")
        stride_type = parser.parse_type()
        parser.parse_punctuation(")")
        parser.parse_punctuation("->")
        res_type = parser.parse_type()
        return cls.create(
            operands=[
                parser.resolve_operand(base_un, base_type),
                parser.resolve_operand(stride_un, stride_type),
            ],
            result_types=[res_type],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.base)
        printer.print_string(", ")
        printer.print_operand(self.stride)
        printer.print_string(" : (")
        printer.print_attribute(self.base.type)
        printer.print_string(", ")
        printer.print_attribute(self.stride.type)
        printer.print_string(") -> ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class GetElementOp(IRDLOperation):
    """`cir.get_element` — array element address.

    Pretty: `cir.get_element` $base `[` $index `:` index-type `]`
            `:` base-type `->` res-type
    """

    name = "cir.get_element"

    base = operand_def(PointerType)
    index = operand_def()
    res = result_def(PointerType)

    @classmethod
    def parse(cls, parser: Parser) -> GetElementOp:
        base_un = parser.parse_unresolved_operand()
        parser.parse_punctuation("[")
        index_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        index_type = parser.parse_type()
        parser.parse_punctuation("]")
        parser.parse_punctuation(":")
        base_type = parser.parse_type()
        parser.parse_punctuation("->")
        res_type = parser.parse_type()
        return cls.create(
            operands=[
                parser.resolve_operand(base_un, base_type),
                parser.resolve_operand(index_un, index_type),
            ],
            result_types=[res_type],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.base)
        printer.print_string("[")
        printer.print_operand(self.index)
        printer.print_string(" : ")
        printer.print_attribute(self.index.type)
        printer.print_string("] : ")
        printer.print_attribute(self.base.type)
        printer.print_string(" -> ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class GetMemberOp(IRDLOperation):
    """`cir.get_member` — record field address.

    Pretty: `cir.get_member` $addr `[` index `]` `{` `name` `=` "field" `}`
                              `:` base-type `->` res-type
    """

    name = "cir.get_member"

    addr = operand_def(PointerType)
    res = result_def(PointerType)

    index_attr = prop_def(IntegerAttr)
    member_name = prop_def(StringAttr, prop_name="name")

    @classmethod
    def parse(cls, parser: Parser) -> GetMemberOp:
        addr_un = parser.parse_unresolved_operand()
        parser.parse_punctuation("[")
        idx = parser.parse_integer()
        parser.parse_punctuation("]")
        attrs = parser.parse_optional_dictionary_attr_dict()
        member_name = attrs.pop("name", None)
        if not isinstance(member_name, StringAttr):
            parser.raise_error("cir.get_member requires `name` attribute")
        parser.parse_punctuation(":")
        base_type = parser.parse_type()
        parser.parse_punctuation("->")
        res_type = parser.parse_type()
        return cls.create(
            operands=[parser.resolve_operand(addr_un, base_type)],
            result_types=[res_type],
            properties={
                "index_attr": IntegerAttr(idx, 32),
                "name": member_name,
            },
            attributes=attrs,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.addr)
        printer.print_string("[")
        printer.print_string(str(self.index_attr.value.data))
        printer.print_string("] {name = ")
        printer.print_attribute(self.member_name)
        printer.print_string("} : ")
        printer.print_attribute(self.addr.type)
        printer.print_string(" -> ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class TernaryOp(IRDLOperation):
    """`cir.ternary` — `?:`-style structured choice with two regions.

    Pretty: `cir.ternary` `(` $cond `,` `true` `{` true-region `}` `,`
                              `false` `{` false-region `}` `)` (`:` type)?
    """

    name = "cir.ternary"

    cond = operand_def(BoolType)
    results_ = var_result_def()

    true_region = region_def()
    false_region = region_def()
    traits = traits_def(NoTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> TernaryOp:
        parser.parse_punctuation("(")
        cond_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        parser.parse_keyword("true")
        true_region = parser.parse_region()
        parser.parse_punctuation(",")
        parser.parse_keyword("false")
        false_region = parser.parse_region()
        parser.parse_punctuation(")")
        result_types: list[Attribute] = []
        if parser.parse_optional_punctuation(":") is not None:
            # Either `: type` or `: (cond-type) -> result-type`.
            if parser.parse_optional_punctuation("(") is not None:
                parser.parse_type()  # operand type, redundant — same as cond.type
                parser.parse_punctuation(")")
                parser.parse_punctuation("->")
                result_types.append(parser.parse_type())
            else:
                result_types.append(parser.parse_type())
        cond = parser.resolve_operand(cond_un, BoolType())
        return cls.create(
            operands=[cond],
            result_types=result_types,
            regions=[true_region, false_region],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string("(")
        printer.print_operand(self.cond)
        printer.print_string(", true ")
        printer.print_region(self.true_region)
        printer.print_string(", false ")
        printer.print_region(self.false_region)
        printer.print_string(")")
        if self.results_:
            printer.print_string(" : ")
            printer.print_attribute(self.results_[0].type)


@irdl_op_definition
class SelectOp(IRDLOperation):
    """`cir.select` — value-level select.

    Pretty: `cir.select` `if` $cond `then` $t `else` $f
            `:` `(` cond-type `,` val-type `,` val-type `)` `->` val-type
    """

    name = "cir.select"

    cond = operand_def(BoolType)
    true_value = operand_def()
    false_value = operand_def()
    res = result_def()

    @classmethod
    def parse(cls, parser: Parser) -> SelectOp:
        parser.parse_keyword("if")
        cond_un = parser.parse_unresolved_operand()
        parser.parse_keyword("then")
        t_un = parser.parse_unresolved_operand()
        parser.parse_keyword("else")
        f_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        parser.parse_punctuation("(")
        cond_type = parser.parse_type()
        parser.parse_punctuation(",")
        t_type = parser.parse_type()
        parser.parse_punctuation(",")
        f_type = parser.parse_type()
        parser.parse_punctuation(")")
        parser.parse_punctuation("->")
        res_type = parser.parse_type()
        return cls.create(
            operands=[
                parser.resolve_operand(cond_un, cond_type),
                parser.resolve_operand(t_un, t_type),
                parser.resolve_operand(f_un, f_type),
            ],
            result_types=[res_type],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" if ")
        printer.print_operand(self.cond)
        printer.print_string(" then ")
        printer.print_operand(self.true_value)
        printer.print_string(" else ")
        printer.print_operand(self.false_value)
        printer.print_string(" : (")
        printer.print_attribute(self.cond.type)
        printer.print_string(", ")
        printer.print_attribute(self.true_value.type)
        printer.print_string(", ")
        printer.print_attribute(self.false_value.type)
        printer.print_string(") -> ")
        printer.print_attribute(self.res.type)


# ---------------------------------------------------------------------------
# Tier-1 ops: extra control flow + lifecycle helpers
# ---------------------------------------------------------------------------


@irdl_op_definition
class BrCondOp(IRDLOperation):
    """`cir.brcond` — conditional branch with two successors."""

    name = "cir.brcond"

    cond = operand_def(BoolType)
    dest_operands_true = var_operand_def()
    dest_operands_false = var_operand_def()
    successor = var_successor_def()
    traits = traits_def(IsTerminator())

    irdl_options = (AttrSizedOperandSegments(),)


@irdl_op_definition
class DoWhileOp(IRDLOperation):
    """`cir.do` — bottom-tested do-while loop.

    Pretty: `cir.do` `{` body `}` `while` `{` cond `}`
    """

    name = "cir.do"

    body_region = region_def()
    cond_region = region_def()
    traits = traits_def(NoTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> DoWhileOp:
        body_region = parser.parse_region()
        parser.parse_keyword("while")
        cond_region = parser.parse_region()
        return cls.create(regions=[body_region, cond_region])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_region(self.body_region)
        printer.print_string(" while ")
        printer.print_region(self.cond_region)


_CASE_KIND_NAME = {0: "default", 1: "equal", 2: "anyof", 3: "range"}
_CASE_KIND_VALUE = {v: k for k, v in _CASE_KIND_NAME.items()}


@irdl_op_definition
class SwitchOp(IRDLOperation):
    """`cir.switch` — structured C/C++ switch.

    Pretty: `cir.switch` `(` $cond `:` cond-type `)`
                         (`all_enum_cases_covered`)? `{` body-region `}`
    """

    name = "cir.switch"

    condition = operand_def(IntType)
    body = region_def()
    all_enum_cases_covered = opt_prop_def(UnitAttr)

    @classmethod
    def parse(cls, parser: Parser) -> SwitchOp:
        parser.parse_punctuation("(")
        cond_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        cond_type = parser.parse_type()
        parser.parse_punctuation(")")
        all_enum = (
            UnitAttr()
            if parser.parse_optional_keyword("all_enum_cases_covered") is not None
            else None
        )
        body = parser.parse_region()
        return cls.create(
            operands=[parser.resolve_operand(cond_un, cond_type)],
            regions=[body],
            properties=_props({"all_enum_cases_covered": all_enum}),
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" (")
        printer.print_operand(self.condition)
        printer.print_string(" : ")
        printer.print_attribute(self.condition.type)
        printer.print_string(")")
        if self.all_enum_cases_covered is not None:
            printer.print_string(" all_enum_cases_covered")
        printer.print_string(" ")
        printer.print_region(self.body)


@irdl_op_definition
class CaseOp(IRDLOperation):
    """`cir.case` — case clause within a `cir.switch`.

    Pretty: `cir.case` `(` kind `,` `[` (typed-int (`,` typed-int)*)? `]` `)`
                       `{` region `}`
    """

    name = "cir.case"

    case_region = region_def()
    value = prop_def(ArrayAttr)
    kind = prop_def(IntegerAttr)

    @classmethod
    def parse(cls, parser: Parser) -> CaseOp:
        parser.parse_punctuation("(")
        kind_kw = parser.parse_identifier()
        if kind_kw not in _CASE_KIND_VALUE:
            parser.raise_error(
                f"expected one of {sorted(_CASE_KIND_VALUE)}, got '{kind_kw}'"
            )
        parser.parse_punctuation(",")
        parser.parse_punctuation("[")
        values: list[Attribute] = []
        if parser.parse_optional_punctuation("]") is None:
            values.append(parser.parse_attribute())
            while parser.parse_optional_punctuation(",") is not None:
                values.append(parser.parse_attribute())
            parser.parse_punctuation("]")
        parser.parse_punctuation(")")
        region = parser.parse_region()
        return cls.create(
            regions=[region],
            properties={
                "value": ArrayAttr(values),
                "kind": IntegerAttr(_CASE_KIND_VALUE[kind_kw], 32),
            },
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" (")
        printer.print_string(
            _CASE_KIND_NAME.get(self.kind.value.data, str(self.kind.value.data))
        )
        printer.print_string(", [")
        printer.print_list(self.value.data, printer.print_attribute)
        printer.print_string("]) ")
        printer.print_region(self.case_region)


@irdl_op_definition
class SwitchFlatOp(IRDLOperation):
    """`cir.switch.flat` — region-less, LLVM-style switch terminator.

    Pretty: `cir.switch.flat` $cond `:` cond-type `,` ^default
                              (`(` $defaultOperands `:` types `)`)?
                              `[` (case (`,` case)*)? `]`
    where case ::= integer `:` ^bb (`(` operands `:` types `)`)?
    """

    name = "cir.switch.flat"

    condition = operand_def(IntType)
    default_operands = var_operand_def()
    case_operands = var_operand_def()
    case_values = prop_def(ArrayAttr)
    case_operand_segments = prop_def(DenseArrayBase)
    successor = var_successor_def()
    traits = traits_def(IsTerminator())

    irdl_options = (AttrSizedOperandSegments(),)

    @classmethod
    def parse(cls, parser: Parser) -> SwitchFlatOp:
        cond_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        cond_type = parser.parse_type()
        if not isinstance(cond_type, IntType):
            parser.raise_error("cir.switch.flat condition must be a !cir.int")
        parser.parse_punctuation(",")
        default_block = parser.parse_successor()
        default_operands: list[SSAValue] = []
        if parser.parse_optional_punctuation("(") is not None:
            d_uns = [parser.parse_unresolved_operand()]
            while parser.parse_optional_punctuation(",") is not None:
                d_uns.append(parser.parse_unresolved_operand())
            parser.parse_punctuation(":")
            d_types = [parser.parse_type()]
            while parser.parse_optional_punctuation(",") is not None:
                d_types.append(parser.parse_type())
            parser.parse_punctuation(")")
            if len(d_uns) != len(d_types):
                parser.raise_error("default operand count and type count must match")
            default_operands = [
                parser.resolve_operand(u, t) for u, t in zip(d_uns, d_types)
            ]

        parser.parse_punctuation("[")
        case_values: list[Attribute] = []
        case_blocks: list[Block] = []
        case_operands: list[SSAValue] = []
        case_segments: list[int] = []

        if parser.parse_optional_punctuation("]") is None:

            def parse_one_case() -> None:
                int_val = parser.parse_integer(allow_negative=True)
                parser.parse_punctuation(":")
                blk = parser.parse_successor()
                seg = 0
                if parser.parse_optional_punctuation("(") is not None:
                    uns = [parser.parse_unresolved_operand()]
                    while parser.parse_optional_punctuation(",") is not None:
                        uns.append(parser.parse_unresolved_operand())
                    parser.parse_punctuation(":")
                    types = [parser.parse_type()]
                    while parser.parse_optional_punctuation(",") is not None:
                        types.append(parser.parse_type())
                    parser.parse_punctuation(")")
                    if len(uns) != len(types):
                        parser.raise_error(
                            "case operand count and type count must match"
                        )
                    for u, t in zip(uns, types):
                        case_operands.append(parser.resolve_operand(u, t))
                    seg = len(uns)
                case_values.append(CIRIntAttr(int_val, cond_type))
                case_blocks.append(blk)
                case_segments.append(seg)

            parse_one_case()
            while parser.parse_optional_punctuation(",") is not None:
                parse_one_case()
            parser.parse_punctuation("]")

        return cls.build(
            operands=[
                parser.resolve_operand(cond_un, cond_type),
                default_operands,
                case_operands,
            ],
            successors=[[default_block, *case_blocks]],
            properties={
                "case_values": ArrayAttr(case_values),
                "case_operand_segments": DenseArrayBase.from_list(i32, case_segments),
            },
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.condition)
        printer.print_string(" : ")
        printer.print_attribute(self.condition.type)
        printer.print_string(", ")
        printer.print_block_name(self.successor[0])
        if self.default_operands:
            printer.print_string("(")
            printer.print_list(self.default_operands, printer.print_operand)
            printer.print_string(" : ")
            printer.print_list(
                self.default_operands,
                lambda v: printer.print_attribute(v.type),
            )
            printer.print_string(")")
        printer.print_string(" [")
        case_blocks = list(self.successor)[1:]
        case_values = list(self.case_values.data)
        seg_attr = cast(DenseArrayBase[IntegerType], self.case_operand_segments)
        segments = [int(x) for x in seg_attr.iter_values()]
        operands = list(self.case_operands)
        op_idx = 0
        if case_blocks:
            printer.print_string("\n")
        for i, (val, blk, seg) in enumerate(zip(case_values, case_blocks, segments)):
            if i > 0:
                printer.print_string(",\n")
            printer.print_string("  ")
            if isinstance(val, CIRIntAttr):
                v = val.value.value.data
                if val.int_type.signed:
                    width = val.int_type.bitwidth
                    if v >= (1 << (width - 1)):
                        v -= 1 << width
                printer.print_string(str(v))
            else:
                printer.print_attribute(val)
            printer.print_string(": ")
            printer.print_block_name(blk)
            if seg:
                printer.print_string("(")
                segment_ops = operands[op_idx : op_idx + seg]
                printer.print_list(segment_ops, printer.print_operand)
                printer.print_string(" : ")
                printer.print_list(
                    segment_ops, lambda v: printer.print_attribute(v.type)
                )
                printer.print_string(")")
            op_idx += seg
        if case_blocks:
            printer.print_string("\n")
        printer.print_string("]")


@irdl_op_definition
class CopyOp(IRDLOperation):
    """`cir.copy` — typed pointer-to-pointer memcpy.

    Pretty: `cir.copy` $src `to` $dst (`volatile`)? attr-dict `:` ptr-type
    """

    name = "cir.copy"

    dst = operand_def(PointerType)
    src = operand_def(PointerType)
    is_volatile = opt_prop_def(UnitAttr)
    tbaa = opt_prop_def(Attribute)

    @classmethod
    def parse(cls, parser: Parser) -> CopyOp:
        src_un = parser.parse_unresolved_operand()
        parser.parse_keyword("to")
        dst_un = parser.parse_unresolved_operand()
        is_volatile = (
            UnitAttr()
            if parser.parse_optional_keyword("volatile") is not None
            else None
        )
        attrs = parser.parse_optional_dictionary_attr_dict()
        tbaa = attrs.pop("tbaa", None)
        parser.parse_punctuation(":")
        ptr_type = parser.parse_type()
        return cls.create(
            operands=[
                parser.resolve_operand(dst_un, ptr_type),
                parser.resolve_operand(src_un, ptr_type),
            ],
            properties=_props(
                {
                    "is_volatile": is_volatile,
                    "tbaa": tbaa,
                }
            ),
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.src)
        printer.print_string(" to ")
        printer.print_operand(self.dst)
        if self.is_volatile is not None:
            printer.print_string(" volatile")
        if self.tbaa is not None:
            printer.print_string(" ")
            printer.print_attr_dict({"tbaa": self.tbaa})
        printer.print_string(" : ")
        printer.print_attribute(self.dst.type)


@irdl_op_definition
class UnreachableOp(IRDLOperation):
    """`cir.unreachable` — `__builtin_unreachable` / immediate UB."""

    name = "cir.unreachable"

    traits = traits_def(IsTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> UnreachableOp:
        return cls.create()

    def print(self, printer: Printer) -> None:
        pass


@irdl_op_definition
class TrapOp(IRDLOperation):
    """`cir.trap` — `__builtin_trap`."""

    name = "cir.trap"

    traits = traits_def(IsTerminator())

    @classmethod
    def parse(cls, parser: Parser) -> TrapOp:
        return cls.create()

    def print(self, printer: Printer) -> None:
        pass


@irdl_op_definition
class ExpectOp(IRDLOperation):
    """`cir.expect` — `__builtin_expect[_with_probability]`."""

    name = "cir.expect"

    val = operand_def(IntType)
    expected = operand_def(IntType)
    res = result_def(IntType)
    prob = opt_prop_def(FloatAttr)


@irdl_op_definition
class AssumeOp(IRDLOperation):
    """`cir.assume` — `__builtin_assume`."""

    name = "cir.assume"

    predicate = operand_def(BoolType)


@irdl_op_definition
class AssumeAlignedOp(IRDLOperation):
    """`cir.assume_aligned` — `__builtin_assume_aligned`."""

    name = "cir.assume_aligned"

    pointer = operand_def(PointerType)
    offset = var_operand_def()
    res = result_def(PointerType)
    alignment = prop_def(IntegerAttr)


@irdl_op_definition
class AssumeSeparateStorageOp(IRDLOperation):
    """`cir.assume_separate_storage` — `__builtin_assume_separate_storage`."""

    name = "cir.assume_separate_storage"

    ptr1 = operand_def(PointerType)
    ptr2 = operand_def(PointerType)


# ---------------------------------------------------------------------------
# Tier-2 ops: builtins, math, vectors, complex, varargs, misc
# ---------------------------------------------------------------------------


@irdl_op_definition
class IsConstantOp(IRDLOperation):
    """`cir.is_constant` — `__builtin_constant_p`."""

    name = "cir.is_constant"

    val = operand_def()
    res = result_def(BoolType)


@irdl_op_definition
class ObjSizeOp(IRDLOperation):
    """`cir.objsize` — `__builtin_object_size`.

    Pretty: `cir.objsize` (`min` | `max`) (`nullunknown`)? (`dynamic`)?
                          $ptr `:` ptr-type `->` int-type
    """

    name = "cir.objsize"

    ptr = operand_def(PointerType)
    res = result_def(IntType)
    min = opt_prop_def(UnitAttr)
    nullunknown = opt_prop_def(UnitAttr)
    dynamic = opt_prop_def(UnitAttr)

    @classmethod
    def parse(cls, parser: Parser) -> ObjSizeOp:
        kw = parser.parse_identifier()
        if kw == "min":
            min_attr: UnitAttr | None = UnitAttr()
        elif kw == "max":
            min_attr = None
        else:
            parser.raise_error(f"expected 'min' or 'max' but got '{kw}'")
        nullunknown = (
            UnitAttr()
            if parser.parse_optional_keyword("nullunknown") is not None
            else None
        )
        dynamic = (
            UnitAttr() if parser.parse_optional_keyword("dynamic") is not None else None
        )
        ptr_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        ptr_type = parser.parse_type()
        parser.parse_punctuation("->")
        res_type = parser.parse_type()
        return cls.create(
            operands=[parser.resolve_operand(ptr_un, ptr_type)],
            result_types=[res_type],
            properties=_props(
                {
                    "min": min_attr,
                    "nullunknown": nullunknown,
                    "dynamic": dynamic,
                }
            ),
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_string("min" if self.min is not None else "max")
        if self.nullunknown is not None:
            printer.print_string(" nullunknown")
        if self.dynamic is not None:
            printer.print_string(" dynamic")
        printer.print_string(" ")
        printer.print_operand(self.ptr)
        printer.print_string(" : ")
        printer.print_attribute(self.ptr.type)
        printer.print_string(" -> ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class PtrDiffOp(IRDLOperation):
    """`cir.ptr_diff` — typed pointer subtraction."""

    name = "cir.ptr_diff"

    lhs = operand_def(PointerType)
    rhs = operand_def(PointerType)
    res = result_def(IntType)


@irdl_op_definition
class IsFPClassOp(IRDLOperation):
    """`cir.is_fp_class` — `__builtin_fpclassify` / isnan / isinf."""

    name = "cir.is_fp_class"

    src = operand_def()
    res = result_def(BoolType)
    flags = prop_def(IntegerAttr)


@irdl_op_definition
class PrefetchOp(IRDLOperation):
    """`cir.prefetch` — `__builtin_prefetch`."""

    name = "cir.prefetch"

    addr = operand_def(PointerType)
    locality = opt_prop_def(IntegerAttr)
    is_write = opt_prop_def(UnitAttr, prop_name="isWrite")


@irdl_op_definition
class StackSaveOp(IRDLOperation):
    """`cir.stacksave` — VLA stack snapshot.

    Pretty: `cir.stacksave` `:` ptr-type
    """

    name = "cir.stacksave"

    res = result_def(PointerType)

    @classmethod
    def parse(cls, parser: Parser) -> StackSaveOp:
        parser.parse_punctuation(":")
        res_type = parser.parse_type()
        return cls.create(result_types=[res_type])

    def print(self, printer: Printer) -> None:
        printer.print_string(" : ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class StackRestoreOp(IRDLOperation):
    """`cir.stackrestore` — restore stack to a previously-saved snapshot.

    Pretty: `cir.stackrestore` $ptr `:` ptr-type
    """

    name = "cir.stackrestore"

    ptr = operand_def(PointerType)

    @classmethod
    def parse(cls, parser: Parser) -> StackRestoreOp:
        ptr_un = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        ptr_type = parser.parse_type()
        return cls.create(operands=[parser.resolve_operand(ptr_un, ptr_type)])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.ptr)
        printer.print_string(" : ")
        printer.print_attribute(self.ptr.type)


@irdl_op_definition
class ReturnAddrOp(IRDLOperation):
    """`cir.return_address` — `__builtin_return_address`."""

    name = "cir.return_address"

    level = operand_def(IntType)
    res = result_def(PointerType)


@irdl_op_definition
class FrameAddrOp(IRDLOperation):
    """`cir.frame_address` — `__builtin_frame_address`."""

    name = "cir.frame_address"

    level = operand_def(IntType)
    res = result_def(PointerType)


@irdl_op_definition
class AddrOfReturnAddrOp(IRDLOperation):
    """`cir.address_of_return_address` — MSVC `_AddressOfReturnAddress`."""

    name = "cir.address_of_return_address"

    res = result_def(PointerType)


@irdl_op_definition
class DynamicCastOp(IRDLOperation):
    """`cir.dyn_cast` — C++ `dynamic_cast`."""

    name = "cir.dyn_cast"

    src = operand_def(PointerType)
    res = result_def(PointerType)
    kind = prop_def(IntegerAttr)
    info = opt_prop_def(Attribute)
    relative_layout = opt_prop_def(UnitAttr)


# Math FP→FP builtins (sqrt, sin, cos, …).


def _make_unary_fp_op(mnemonic: str):
    cls = type(
        f"FP{mnemonic.capitalize()}Op",
        (IRDLOperation,),
        {
            "name": f"cir.{mnemonic}",
            "src": operand_def(),
            "res": result_def(),
            "__doc__": f"`cir.{mnemonic}` — unary FP→FP math builtin.",
        },
    )
    return irdl_op_definition(cls)


SqrtOp = _make_unary_fp_op("sqrt")
ACosOp = _make_unary_fp_op("acos")
ASinOp = _make_unary_fp_op("asin")
ATanOp = _make_unary_fp_op("atan")
CeilOp = _make_unary_fp_op("ceil")
CosOp = _make_unary_fp_op("cos")
ExpOp = _make_unary_fp_op("exp")
Exp2Op = _make_unary_fp_op("exp2")
FAbsOp = _make_unary_fp_op("fabs")
FloorOp = _make_unary_fp_op("floor")
SinOp = _make_unary_fp_op("sin")


# Bit manipulation builtins (`cir.<mnemonic>`).


@irdl_op_definition
class BitClrsbOp(IRDLOperation):
    """`cir.clrsb` — count leading redundant sign bits."""

    name = "cir.clrsb"
    input = operand_def(IntType)
    result = result_def(IntType)


@irdl_op_definition
class BitClzOp(IRDLOperation):
    """`cir.clz` — count leading zero bits."""

    name = "cir.clz"
    input = operand_def(IntType)
    result = result_def(IntType)
    poison_zero = opt_prop_def(UnitAttr)


@irdl_op_definition
class BitCtzOp(IRDLOperation):
    """`cir.ctz` — count trailing zero bits."""

    name = "cir.ctz"
    input = operand_def(IntType)
    result = result_def(IntType)
    poison_zero = opt_prop_def(UnitAttr)


@irdl_op_definition
class BitFfsOp(IRDLOperation):
    """`cir.ffs` — find first set bit (1-based)."""

    name = "cir.ffs"
    input = operand_def(IntType)
    result = result_def(IntType)


@irdl_op_definition
class BitParityOp(IRDLOperation):
    """`cir.parity` — parity of input bits."""

    name = "cir.parity"
    input = operand_def(IntType)
    result = result_def(IntType)


@irdl_op_definition
class BitPopcountOp(IRDLOperation):
    """`cir.popcount` — population count."""

    name = "cir.popcount"
    input = operand_def(IntType)
    result = result_def(IntType)


@irdl_op_definition
class BitReverseOp(IRDLOperation):
    """`cir.bitreverse` — reverse the bit pattern."""

    name = "cir.bitreverse"
    input = operand_def(IntType)
    result = result_def(IntType)


@irdl_op_definition
class ByteSwapOp(IRDLOperation):
    """`cir.byte_swap` — byte-order reverse."""

    name = "cir.byte_swap"
    input = operand_def(IntType)
    result = result_def(IntType)


@irdl_op_definition
class RotateOp(IRDLOperation):
    """`cir.rotate` — bit rotation; `rotateLeft` selects direction."""

    name = "cir.rotate"
    input = operand_def(IntType)
    amount = operand_def(IntType)
    result = result_def(IntType)
    rotate_left = opt_prop_def(UnitAttr, prop_name="rotateLeft")


# Complex number ops.


@irdl_op_definition
class ComplexCreateOp(IRDLOperation):
    """`cir.complex.create` — build a complex value from real/imag parts."""

    name = "cir.complex.create"
    real = operand_def()
    imag = operand_def()
    result = result_def(ComplexType)


@irdl_op_definition
class ComplexRealOp(IRDLOperation):
    """`cir.complex.real` — real part of a complex (or scalar pass-through)."""

    name = "cir.complex.real"
    operand = operand_def()
    result = result_def()


@irdl_op_definition
class ComplexImagOp(IRDLOperation):
    """`cir.complex.imag` — imaginary part of a complex."""

    name = "cir.complex.imag"
    operand = operand_def()
    result = result_def()


@irdl_op_definition
class ComplexRealPtrOp(IRDLOperation):
    """`cir.complex.real_ptr` — pointer to the real part of a complex object."""

    name = "cir.complex.real_ptr"
    operand = operand_def(PointerType)
    result = result_def(PointerType)


@irdl_op_definition
class ComplexImagPtrOp(IRDLOperation):
    """`cir.complex.imag_ptr` — pointer to the imaginary part."""

    name = "cir.complex.imag_ptr"
    operand = operand_def(PointerType)
    result = result_def(PointerType)


@irdl_op_definition
class ComplexAddOp(IRDLOperation):
    """`cir.complex.add` — complex addition."""

    name = "cir.complex.add"
    lhs = operand_def(ComplexType)
    rhs = operand_def(ComplexType)
    result = result_def(ComplexType)


@irdl_op_definition
class ComplexSubOp(IRDLOperation):
    """`cir.complex.sub` — complex subtraction."""

    name = "cir.complex.sub"
    lhs = operand_def(ComplexType)
    rhs = operand_def(ComplexType)
    result = result_def(ComplexType)


@irdl_op_definition
class ComplexMulOp(IRDLOperation):
    """`cir.complex.mul` — complex multiplication; `range` enum encoded as i32."""

    name = "cir.complex.mul"
    lhs = operand_def(ComplexType)
    rhs = operand_def(ComplexType)
    result = result_def(ComplexType)
    range = prop_def(IntegerAttr)


@irdl_op_definition
class ComplexDivOp(IRDLOperation):
    """`cir.complex.div` — complex division; `range` enum encoded as i32."""

    name = "cir.complex.div"
    lhs = operand_def(ComplexType)
    rhs = operand_def(ComplexType)
    result = result_def(ComplexType)
    range = prop_def(IntegerAttr)


# Vector ops.


@irdl_op_definition
class VecCreateOp(IRDLOperation):
    """`cir.vec.create` — build a vector value from element operands."""

    name = "cir.vec.create"
    elements = var_operand_def()
    result = result_def(VectorType)


@irdl_op_definition
class VecInsertOp(IRDLOperation):
    """`cir.vec.insert` — replace one element of a vector."""

    name = "cir.vec.insert"
    vec = operand_def(VectorType)
    value = operand_def()
    index = operand_def(IntType)
    result = result_def(VectorType)


@irdl_op_definition
class VecExtractOp(IRDLOperation):
    """`cir.vec.extract` — extract one element from a vector."""

    name = "cir.vec.extract"
    vec = operand_def(VectorType)
    index = operand_def(IntType)
    result = result_def()


@irdl_op_definition
class VecCmpOp(IRDLOperation):
    """`cir.vec.cmp` — element-wise comparison; `kind` encoded as i32."""

    name = "cir.vec.cmp"
    lhs = operand_def(VectorType)
    rhs = operand_def(VectorType)
    result = result_def(VectorType)
    kind = prop_def(IntegerAttr)


@irdl_op_definition
class VecShuffleOp(IRDLOperation):
    """`cir.vec.shuffle` — `__builtin_shufflevector` (compile-time indices)."""

    name = "cir.vec.shuffle"
    vec1 = operand_def(VectorType)
    vec2 = operand_def(VectorType)
    result = result_def(VectorType)
    indices = prop_def(ArrayAttr)


@irdl_op_definition
class VecShuffleDynamicOp(IRDLOperation):
    """`cir.vec.shuffle.dynamic` — `__builtin_shufflevector` (runtime indices)."""

    name = "cir.vec.shuffle.dynamic"
    vec = operand_def(VectorType)
    indices = operand_def(VectorType)
    result = result_def(VectorType)


@irdl_op_definition
class VecTernaryOp(IRDLOperation):
    """`cir.vec.ternary` — element-wise `cond ? a : b` for vectors."""

    name = "cir.vec.ternary"
    cond = operand_def(VectorType)
    lhs = operand_def(VectorType)
    rhs = operand_def(VectorType)
    result = result_def(VectorType)


@irdl_op_definition
class VecSplatOp(IRDLOperation):
    """`cir.vec.splat` — replicate a scalar across a vector."""

    name = "cir.vec.splat"
    value = operand_def()
    result = result_def(VectorType)


# Variadic-arg ops.


@irdl_op_definition
class VAStartOp(IRDLOperation):
    """`cir.va_start` — initialise a `va_list`."""

    name = "cir.va_start"
    arg_list = operand_def(PointerType)
    count = operand_def(IntType)


@irdl_op_definition
class VAEndOp(IRDLOperation):
    """`cir.va_end` — finalise a `va_list`."""

    name = "cir.va_end"
    arg_list = operand_def(PointerType)


@irdl_op_definition
class VACopyOp(IRDLOperation):
    """`cir.va_copy` — copy one `va_list` into another."""

    name = "cir.va_copy"
    src_list = operand_def(PointerType)
    dst_list = operand_def(PointerType)


@irdl_op_definition
class VAArgOp(IRDLOperation):
    """`cir.va_arg` — fetch next variadic argument as a typed value."""

    name = "cir.va_arg"
    arg_list = operand_def(PointerType)
    result = result_def()


# ---------------------------------------------------------------------------
# Dialect
# ---------------------------------------------------------------------------


CIR = Dialect(
    "cir",
    [
        ACosOp,
        ASinOp,
        ATanOp,
        AddrOfReturnAddrOp,
        AllocaOp,
        AssumeAlignedOp,
        AssumeOp,
        AssumeSeparateStorageOp,
        BinOp,
        BitClrsbOp,
        BitClzOp,
        BitCtzOp,
        BitFfsOp,
        BitParityOp,
        BitPopcountOp,
        BitReverseOp,
        BrCondOp,
        BrOp,
        BreakOp,
        ByteSwapOp,
        CallOp,
        CaseOp,
        CastOp,
        CeilOp,
        CmpOp,
        ComplexAddOp,
        ComplexCreateOp,
        ComplexDivOp,
        ComplexImagOp,
        ComplexImagPtrOp,
        ComplexMulOp,
        ComplexRealOp,
        ComplexRealPtrOp,
        ComplexSubOp,
        ConditionOp,
        ConstantOp,
        ContinueOp,
        CopyOp,
        CosOp,
        DoWhileOp,
        DynamicCastOp,
        Exp2Op,
        ExpOp,
        ExpectOp,
        FAbsOp,
        FloorOp,
        ForOp,
        FrameAddrOp,
        FuncOp,
        GetElementOp,
        GetGlobalOp,
        GetMemberOp,
        GlobalOp,
        IfOp,
        IsConstantOp,
        IsFPClassOp,
        LoadOp,
        ObjSizeOp,
        PrefetchOp,
        PtrDiffOp,
        PtrStrideOp,
        ReturnAddrOp,
        ReturnOp,
        RotateOp,
        ScopeOp,
        SelectOp,
        SinOp,
        SqrtOp,
        StackRestoreOp,
        StackSaveOp,
        StoreOp,
        SwitchFlatOp,
        SwitchOp,
        TernaryOp,
        TrapOp,
        UnaryOp,
        UnreachableOp,
        VAArgOp,
        VACopyOp,
        VAEndOp,
        VAStartOp,
        VecCmpOp,
        VecCreateOp,
        VecExtractOp,
        VecInsertOp,
        VecShuffleDynamicOp,
        VecShuffleOp,
        VecSplatOp,
        VecTernaryOp,
        WhileOp,
        YieldOp,
    ],
    [
        ArrayType,
        BF16Type,
        BoolType,
        CIRBoolAttr,
        CIRFPAttr,
        CIRIntAttr,
        ComplexType,
        ConstArrayAttr,
        ConstPtrAttr,
        ConstRecordAttr,
        DoubleType,
        FP16Type,
        FP80Type,
        FP128Type,
        FuncType,
        GlobalViewAttr,
        IntType,
        LongDoubleType,
        OptInfoAttr,
        PoisonAttr,
        PointerType,
        RecordType,
        SingleType,
        SourceLanguageAttr,
        UndefAttr,
        VectorType,
        VisibilityAttr,
        VoidType,
        ZeroAttr,
    ],
)
