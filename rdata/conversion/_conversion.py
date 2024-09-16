from __future__ import annotations

import abc
import warnings
from collections import ChainMap
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType, SimpleNamespace
from typing import Any, Final, NamedTuple, Union, cast

import numpy as np
import pandas as pd
import xarray
from typing_extensions import override

from .. import parser

ConversionFunction = Callable[[Union[parser.RData, parser.RObject]], Any]


class RLanguage(NamedTuple):
    """R language construct."""

    elements: list[Any]
    attributes: Mapping[str, Any]


class RExpression(NamedTuple):
    """R expression."""

    elements: list[RLanguage]


@dataclass
class RBuiltin:
    """R builtin."""

    name: str


@dataclass
class RFunction:
    """R function."""

    environment: Mapping[str, Any]
    formals: Mapping[str, Any] | None
    body: RLanguage
    attributes: Mapping[str, Any]

    @property
    def source(self) -> str:
        return "\n".join(self.attributes["srcref"].srcfile.lines)


@dataclass
class RExternalPointer:
    """R bytecode."""

    protected: Any
    tag: Any


@dataclass
class RBytecode:
    """R bytecode."""

    code: xarray.DataArray
    constants: Sequence[Any]
    attributes: Mapping[str, Any]


class REnvironment(ChainMap[str, Any]):
    """R environment."""

    def __init__(
        self,
        *maps: MutableMapping[str, Any],
        frame: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(*maps)
        self.frame = frame


def convert_list(
    r_list: parser.RObject,
    conversion_function: ConversionFunction,
) -> Mapping[str, Any] | list[Any]:
    """
    Expand a tagged R pairlist to a Python dictionary.

    Args:
        r_list: Pairlist R object, with tags.
        conversion_function: Conversion function to apply to the elements of
            the list. By default is the identity function.

    Returns:
        A dictionary with the tags of the pairwise list as keys and their
        corresponding values as values.

    See Also:
        convert_vector

    """
    if r_list.info.type is parser.RObjectType.NILVALUE:
        return {}

    if r_list.info.type not in {
        parser.RObjectType.LIST,
        parser.RObjectType.LANG,
    }:
        msg = "Must receive a LIST, LANG or NILVALUE object"
        raise TypeError(msg)

    tag = None if r_list.tag is None else conversion_function(r_list.tag)

    cdr = conversion_function(r_list.value[1])

    if tag is not None:
        if cdr is None:
            cdr = {}

        return {tag: conversion_function(r_list.value[0]), **cdr}

    if cdr is None:
        cdr = []

    return [conversion_function(r_list.value[0]), *cdr]


def convert_env(
    r_env: parser.RObject,
    conversion_function: ConversionFunction,
) -> REnvironment:
    """Convert environment objects."""
    if r_env.info.type is not parser.RObjectType.ENV:
        msg = "Must receive a ENV object"
        raise TypeError(msg)

    frame = conversion_function(r_env.value.frame)
    enclosure = conversion_function(r_env.value.enclosure)
    hash_table = conversion_function(r_env.value.hash_table)

    dictionary = {}
    if hash_table is not None:
        for d in hash_table:
            if d is not None:
                dictionary.update(d)

    return REnvironment(dictionary, enclosure, frame=frame)


def convert_attrs(
    r_obj: parser.RObject,
        conversion_function: ConversionFunction,
) -> Mapping[str, Any]:
    """
    Return the attributes of an object as a Python dictionary.

    Args:
        r_obj: R object.
        conversion_function: Conversion function to apply to the elements of
            the attribute list. By default is the identity function.

    Returns:
        A dictionary with the names of the attributes as keys and their
        corresponding values as values.

    See Also:
        convert_list

    """
    if r_obj.attributes:
        attrs = cast(
            Mapping[str, Any],
            conversion_function(r_obj.attributes),
        )
    else:
        attrs = {}
    return attrs


def convert_vector(
    r_vec: parser.RObject,
    conversion_function: ConversionFunction,
    attrs: Mapping[str, Any] | None = None,
) -> list[Any] | Mapping[str, Any]:
    """
    Convert a R vector to a Python list or dictionary.

    If the vector has a ``names`` attribute, the result is a dictionary with
    the names as keys. Otherwise, the result is a Python list.

    Args:
        r_vec: R vector.
        conversion_function: Conversion function to apply to the elements of
            the vector. By default is the identity function.
        attrs: Attributes of the vector.

    Returns:
        A dictionary with the ``names`` of the vector as keys and their
        corresponding values as values. If the vector does not have an
        argument ``names``, then a normal Python list is returned.

    See Also:
        convert_list

    """
    if attrs is None:
        attrs = {}

    if r_vec.info.type not in {
        parser.RObjectType.VEC,
        parser.RObjectType.EXPR,
    }:
        msg = "Must receive a VEC or EXPR object"
        raise TypeError(msg)

    value: list[Any] | Mapping[str, Any] = [
        conversion_function(o) for o in r_vec.value
    ]

    # If it has the name attribute, use a dict instead
    field_names = attrs.get("names")
    if field_names is not None:
        value = dict(zip(field_names, value))

    return value


def safe_decode(byte_str: bytes, encoding: str) -> str | bytes:
    """Decode a (possibly malformed) string."""
    try:
        return byte_str.decode(encoding)
    except UnicodeDecodeError as e:
        warnings.warn(  # noqa: B028
            f"Exception while decoding {byte_str!r}: {e}",
        )
        return byte_str


def convert_char(
    r_char: parser.RObject,
    *,
    default_encoding: str | None = None,
    force_default_encoding: bool = False,
) -> str | bytes | None:
    """
    Decode a R character array to a Python string or bytes.

    The bits that signal the encoding are in the general pointer. The
    string can be encoded in UTF8, LATIN1 or ASCII, or can be a sequence
    of bytes.

    Args:
        r_char: R character array.
        default_encoding: Default encoding to apply when encoding info
            is not available.
        force_default_encoding: Always use the default encoding.

    Returns:
        Decoded string.

    See Also:
        convert_symbol

    """
    if r_char.info.type is not parser.RObjectType.CHAR:
        msg = "Must receive a CHAR object"
        raise TypeError(msg)

    if r_char.value is None:
        return None

    assert isinstance(r_char.value, bytes)

    encoding = None

    if not force_default_encoding:
        if r_char.info.gp & parser.CharFlags.UTF8:
            encoding = "utf_8"
        elif r_char.info.gp & parser.CharFlags.LATIN1:
            encoding = "latin_1"
        elif r_char.info.gp & parser.CharFlags.ASCII:
            encoding = "ascii"
        elif r_char.info.gp & parser.CharFlags.BYTES:
            encoding = "bytes"

    if encoding is None:
        if default_encoding:
            encoding = default_encoding
        else:
            # Assume ASCII if no encoding is marked
            warnings.warn("Unknown encoding. Assumed ASCII.")  # noqa: B028
            encoding = "ascii"

    return (
        r_char.value
        if encoding == "bytes"
        else safe_decode(r_char.value, encoding)
    )


def convert_symbol(
    r_symbol: parser.RObject,
    conversion_function: ConversionFunction,
) -> str | bytes:
    """
    Decode a R symbol to a Python string or bytes.

    Args:
        r_symbol: R symbol.
        conversion_function: Conversion function to apply to the char element
            of the symbol. By default is the identity function.

    Returns:
        Decoded string.

    See Also:
        convert_char

    """
    if r_symbol.info.type is parser.RObjectType.SYM:
        symbol = conversion_function(r_symbol.value)
        assert isinstance(symbol, str)
        return symbol

    msg = "Must receive a SYM object"
    raise TypeError(msg)


def convert_array(
    r_array: parser.RObject,
    attrs: Mapping[str, Any] | None = None,
) -> np.ndarray[Any, Any] | xarray.DataArray:
    """
    Convert a R array to a Numpy ndarray or a Xarray DataArray.

    If the array has attribute ``dimnames`` the output will be a
    Xarray DataArray, preserving the dimension names.

    Args:
        r_array: R array.
        attrs: Attributes of the array.

    Returns:
        Array.

    See Also:
        convert_vector

    """
    if attrs is None:
        attrs = {}

    if r_array.info.type not in {
        parser.RObjectType.LGL,
        parser.RObjectType.INT,
        parser.RObjectType.REAL,
        parser.RObjectType.CPLX,
    }:
        msg = "Must receive an array object"
        raise TypeError(msg)

    value = r_array.value

    shape = attrs.get("dim")
    if shape is not None:
        # R matrix order is like FORTRAN
        value = np.reshape(value, shape, order="F")

    dimension_names = None
    coords = None

    dimnames = attrs.get("dimnames")
    if dimnames:
        if isinstance(dimnames, Mapping):
            dimension_names = list(dimnames.keys())
            coords = dimnames
        else:
            dimension_names = [f"dim_{i}" for i, _ in enumerate(dimnames)]
            coords = {
                dimension_names[i]: d
                for i, d in enumerate(dimnames)
                if d is not None
            }

        value = xarray.DataArray(
            value,
            dims=dimension_names,
            coords=coords,
        )

    return value  # type: ignore [no-any-return]


def convert_altrep_to_range(
    r_altrep: parser.RObject,
) -> range:
    """
    Convert a R altrep to range object.

    Args:
        r_altrep: R altrep object

    Returns:
        Array.

    See Also:
        convert_array
    """
    if r_altrep.info.type != parser.RObjectType.ALTREP:
        msg = "Must receive an altrep object"
        raise TypeError(msg)

    info, state, attr = r_altrep.value
    assert attr.info.type == parser.RObjectType.NILVALUE

    assert info.info.type == parser.RObjectType.LIST

    class_sym = info.value[0]
    while class_sym.info.type == parser.RObjectType.REF:
        class_sym = class_sym.referenced_object

    assert class_sym.info.type == parser.RObjectType.SYM
    assert class_sym.value.info.type == parser.RObjectType.CHAR

    altrep_name = class_sym.value.value
    assert isinstance(altrep_name, bytes)

    if altrep_name != b"compact_intseq":
        msg = "Only compact integer sequences can be converted to range"
        raise NotImplementedError(msg)

    n = int(state.value[0])
    start = int(state.value[1])
    step = int(state.value[2])
    stop = start + (n - 1) * step
    return range(start, stop + 1, step)


def _dataframe_column_transform(source: Any) -> Any:  # noqa: ANN401

    if isinstance(source, np.ndarray):
        dtype: Any
        if np.issubdtype(source.dtype, np.integer):
            dtype = pd.Int32Dtype()
        elif np.issubdtype(source.dtype, np.floating):
            # We return the numpy array here, which keeps
            # R_FLOAT_NA, np.nan, and other NaNs as they were originally in the file.
            # Users can then decide if they prefer to interpret
            # only R_FLOAT_NA or all NaNs as "missing".
            return source
            # This would create an array with all NaNs as "missing":
            # dtype = pd.Float64Dtype()  # noqa: ERA001
        elif np.issubdtype(source.dtype, np.complexfloating):
            # There seems to be no pandas type for complex array
            return source
        elif np.issubdtype(source.dtype, np.bool_):
            dtype = pd.BooleanDtype()
        elif np.issubdtype(source.dtype, np.str_):
            dtype = pd.StringDtype()
        elif np.issubdtype(source.dtype, np.object_):
            for value in source:
                assert isinstance(value, str) or value is None
            dtype = pd.StringDtype()
        else:
            return source

        return pd.Series(source, dtype=dtype).array

    return source


def dataframe_constructor(
    obj: Mapping[str, Any],
    attrs: Mapping[str, Any],
) -> pd.DataFrame:

    row_names = attrs["row.names"]

    obj = {key: _dataframe_column_transform(val) for key, val in obj.items()}

    # Default row names are stored as [R_INT_NA, -len]
    default_row_names_len = 2
    index: pd.RangeIndex | tuple[str, ...] = (
        pd.RangeIndex(1, abs(row_names[1]) + 1)
        if (
            len(row_names) == default_row_names_len
            and isinstance(row_names, np.ma.MaskedArray)
            and row_names.mask[0]
        )
        else row_names
    )

    return pd.DataFrame(obj, columns=obj, index=index)


def _factor_constructor_internal(
    obj: np.ndarray[Any, np.dtype[np.integer[Any]]],
    attrs: Mapping[str, Any],
    *,
    ordered: bool,
) -> pd.Categorical:
    values = [attrs["levels"][i - 1] if i >= 0 else None for i in obj]

    return pd.Categorical(values, attrs["levels"], ordered=ordered)


def factor_constructor(
    obj: np.ndarray[Any, np.dtype[np.integer[Any]]],
    attrs: Mapping[str, Any],
) -> pd.Categorical:
    """Construct a factor objects."""
    return _factor_constructor_internal(obj, attrs, ordered=False)


def ordered_constructor(
    obj: np.ndarray[Any, np.dtype[np.integer[Any]]],
    attrs: Mapping[str, Any],
) -> pd.Categorical:
    """Contruct an ordered factor."""
    return _factor_constructor_internal(obj, attrs, ordered=True)


def ts_constructor(
    obj: np.ndarray[Any, Any],
    attrs: Mapping[str, Any],
) -> pd.Series[Any]:
    """Construct a time series object."""
    start, end, frequency = attrs["tsp"]

    frequency = int(frequency)

    real_start = Fraction(int(round(start * frequency)), frequency)
    real_end = Fraction(int(round(end * frequency)), frequency)

    index: np.ndarray[Any, Any] = np.arange(
        real_start,
        real_end + Fraction(1, frequency),
        Fraction(1, frequency),
    )

    if frequency == 1:
        index = index.astype(int)

    return pd.Series(obj, index=index)


@dataclass
class SrcRef:
    """Reference to a source file location."""
    first_line: int
    first_byte: int
    last_line: int
    last_byte: int
    first_column: int
    last_column: int
    first_parsed: int
    last_parsed: int
    srcfile: SrcFile


def srcref_constructor(
    obj: tuple[int, int, int, int, int, int, int, int],
    attrs: Mapping[str, Any],
) -> SrcRef:
    return SrcRef(*obj, srcfile=attrs["srcfile"])


@dataclass
class SrcFile:
    """Source file."""
    filename: str
    file_encoding: str | None
    string_encoding: str | None


def srcfile_constructor(
    obj: REnvironment,
    attrs: Mapping[str, Any],  # noqa: ARG001
) -> SrcFile:

    frame = obj.frame
    assert frame is not None
    filename = frame["filename"][0]
    file_encoding = frame.get("encoding")
    string_encoding = frame.get("Enc")

    return SrcFile(
        filename=filename,
        file_encoding=file_encoding,
        string_encoding=string_encoding,
    )


@dataclass
class SrcFileCopy(SrcFile):
    """Source file with a copy of its lines."""
    lines: Sequence[str]


def srcfilecopy_constructor(
    obj: REnvironment,
    attrs: Mapping[str, Any],  # noqa: ARG001
) -> SrcFileCopy:

    frame = obj.frame
    assert frame is not None
    filename = frame["filename"][0]
    file_encoding = frame.get("encoding", (None,))[0]
    string_encoding = frame.get("Enc", (None,))[0]
    lines = frame["lines"]

    return SrcFileCopy(
        filename=filename,
        file_encoding=file_encoding,
        string_encoding=string_encoding,
        lines=lines,
    )


Constructor = Callable[[Any, Mapping[str, Any]], Any]
ConstructorDict = Mapping[
    Union[str, bytes],
    Constructor,
]

default_class_map_dict: Final[ConstructorDict] = {
    "data.frame": dataframe_constructor,
    "factor": factor_constructor,
    "ordered": ordered_constructor,
    "ts": ts_constructor,
    "srcref": srcref_constructor,
    "srcfile": srcfile_constructor,
    "srcfilecopy": srcfilecopy_constructor,
}

#: Default mapping of constructor functions.
DEFAULT_CLASS_MAP: Final = MappingProxyType(default_class_map_dict)


class Converter(abc.ABC):
    """Interface of a class converting R objects in Python objects."""

    @abc.abstractmethod
    def convert(self, data: parser.RData | parser.RObject) -> Any:  # noqa: ANN401
        """Convert a R object to a Python one."""


@dataclass
class UnresolvedReference:
    references: MutableMapping[int, Any]
    index: int


class SimpleConverter(Converter):
    """
    Class converting R objects to Python objects.

    Args:
        constructor_dict:
            Dictionary mapping names of R classes to constructor functions with
            the following prototype:

            .. code-block :: python

                def constructor(obj, attrs):
                    ...

            This dictionary can be used to support custom R classes. By
            default, the dictionary used is
            :data:`~rdata.conversion._conversion.DEFAULT_CLASS_MAP`
            which has support for several common classes.
        default_encoding:
            Default encoding used for strings with unknown encoding. If `None`,
            the one stored in the file will be used, or ASCII as a fallback.
        force_default_encoding:
            Use the default encoding even if the strings specify other
            encoding.
        global_environment: Global environment to use. By default is an empty
            environment.
        base_environment: Base environment to use. By default is an empty
            environment.

    """

    def __init__(
        self,
        constructor_dict: ConstructorDict = DEFAULT_CLASS_MAP,
        *,
        default_encoding: str | None = None,
        force_default_encoding: bool = False,
        global_environment: MutableMapping[str, Any] | None = None,
        base_environment: MutableMapping[str, Any] | None = None,
    ) -> None:

        self.constructor_dict = constructor_dict
        self.default_encoding = default_encoding
        self.force_default_encoding = force_default_encoding
        self.global_environment = REnvironment(
            {} if global_environment is None
            else global_environment,
        )
        self.base_environment = REnvironment(
            {} if base_environment is None
            else base_environment,
        )
        self.empty_environment: Mapping[str, Any] = REnvironment({})

        self._reset()

    def _reset(self) -> None:
        self.references: MutableMapping[int, Any] = {}
        self.default_encoding_used = self.default_encoding

    @override
    def convert(
        self,
        data: parser.RData | parser.RObject,
    ) -> Any:
        self._reset()
        return self._convert_next(data)

    def _convert_next(  # noqa: C901, PLR0912, PLR0915
        self,
        data: parser.RData | parser.RObject,
    ) -> Any:  # noqa: ANN401
        """Convert a R object to a Python one."""
        obj: parser.RObject
        if isinstance(data, parser.RData):
            obj = data.object
            if self.default_encoding is None:
                self.default_encoding_used = data.extra.encoding
        else:
            obj = data

        attrs = convert_attrs(obj, self._convert_next)

        reference_id = id(obj)

        # Return the value if previously referenced
        value: Any = self.references.get(id(obj))
        if value is not None:
            pass

        if obj.info.type == parser.RObjectType.SYM:

            # Return the internal string
            value = convert_symbol(obj, self._convert_next)

        elif obj.info.type == parser.RObjectType.LIST:

            # Expand the list and process the elements
            value = convert_list(obj, self._convert_next)

        elif obj.info.type == parser.RObjectType.CLO:
            assert obj.tag is not None
            assert obj.attributes is not None
            environment = self._convert_next(obj.tag)
            formals = self._convert_next(obj.value[0])
            body = self._convert_next(obj.value[1])
            attributes = self._convert_next(obj.attributes)

            value = RFunction(
                environment=environment,
                formals=formals,
                body=body,
                attributes=attributes,
            )

        elif obj.info.type == parser.RObjectType.ENV:

            # Return a ChainMap of the environments
            value = convert_env(obj, self._convert_next)

        elif obj.info.type == parser.RObjectType.LANG:

            # Expand the list and process the elements, returning a
            # special object
            rlanguage_list = convert_list(obj, self._convert_next)
            assert isinstance(rlanguage_list, list)
            attributes = self._convert_next(
                obj.attributes,
            ) if obj.attributes else {}

            value = RLanguage(rlanguage_list, attributes)

        elif obj.info.type in {
            parser.RObjectType.SPECIAL,
            parser.RObjectType.BUILTIN,
        }:

            value = RBuiltin(name=obj.value.decode("ascii"))

        elif obj.info.type == parser.RObjectType.CHAR:

            # Return the internal string
            value = convert_char(
                obj,
                default_encoding=self.default_encoding_used,
                force_default_encoding=self.force_default_encoding,
            )

        elif obj.info.type in {
            parser.RObjectType.LGL,
            parser.RObjectType.INT,
            parser.RObjectType.REAL,
            parser.RObjectType.CPLX,
        }:

            # Return the internal array
            value = convert_array(obj, attrs=attrs)

        elif obj.info.type == parser.RObjectType.STR:

            # Convert the internal strings
            value = np.array([self._convert_next(o) for o in obj.value])

        elif obj.info.type == parser.RObjectType.VEC:

            # Convert the internal objects
            value = convert_vector(obj, self._convert_next, attrs=attrs)

        elif obj.info.type == parser.RObjectType.EXPR:
            rexpression_list = convert_vector(
                obj,
                self._convert_next,
                attrs=attrs,
            )
            assert isinstance(rexpression_list, list)

            # Convert the internal objects returning a special object
            value = RExpression(rexpression_list)

        elif obj.info.type == parser.RObjectType.BCODE:

            value = RBytecode(
                code=self._convert_next(obj.value[0]),
                constants=[self._convert_next(c) for c in obj.value[1]],
                attributes=attrs,
            )

        elif obj.info.type == parser.RObjectType.EXTPTR:

            value = RExternalPointer(
                protected=self._convert_next(obj.value[0]),
                tag=self._convert_next(obj.value[1]),
            )

        elif obj.info.type == parser.RObjectType.S4:
            value = SimpleNamespace(**attrs)

        elif obj.info.type == parser.RObjectType.BASEENV:
            value = self.base_environment

        elif obj.info.type == parser.RObjectType.EMPTYENV:
            value = self.empty_environment

        elif obj.info.type == parser.RObjectType.MISSINGARG:
            value = NotImplemented

        elif obj.info.type == parser.RObjectType.GLOBALENV:
            value = self.global_environment

        elif obj.info.type == parser.RObjectType.REF:

            # Return the referenced value
            value = self.references.get(id(obj.referenced_object))
            if value is None:
                reference_id = id(obj.referenced_object)
                assert obj.referenced_object is not None
                self.references[reference_id] = UnresolvedReference(
                    self.references,
                    reference_id,
                )
                value = self._convert_next(obj.referenced_object)

        elif obj.info.type == parser.RObjectType.NILVALUE:

            value = None

        elif obj.info.type == parser.RObjectType.ALTREP:
            value = convert_altrep_to_range(obj)

        else:
            msg = f"Type {obj.info.type} not implemented"
            raise NotImplementedError(msg)

        if obj.info.object and attrs is not None:
            classname = attrs.get("class", ())
            for i, c in enumerate(classname):

                constructor = self.constructor_dict.get(c, None)

                new_value = (
                    constructor(value, attrs)
                    if constructor
                    else NotImplemented
                )

                if new_value is NotImplemented:
                    missing_msg = (
                        f"Missing constructor for R class \"{c}\". "
                    )

                    if len(classname) > (i + 1):
                        solution_msg = (
                            f"The constructor for class "
                            f"\"{classname[i+1]}\" will be "
                            f"used instead."
                        )
                    else:
                        solution_msg = (
                            "The underlying R object is "
                            "returned instead."
                        )

                    warnings.warn(
                        missing_msg + solution_msg,
                        stacklevel=1,
                    )
                else:
                    value = new_value
                    break

        self.references[reference_id] = value

        return value


def convert(
    data: parser.RData | parser.RObject,
    constructor_dict: ConstructorDict = DEFAULT_CLASS_MAP,
    *,
    default_encoding: str | None = None,
    force_default_encoding: bool = False,
    global_environment: MutableMapping[str, Any] | None = None,
    base_environment: MutableMapping[str, Any] | None = None,
) -> Any:  # noqa: ANN401
    """
    Use the default converter (:func:`SimpleConverter`) to convert the data.

    Args:
        data: Parsed data.
        constructor_dict: Dictionary mapping names of R classes to constructor
            functions with the following prototype:

            .. code-block :: python

                def constructor(obj, attrs):
                    ...

            This dictionary can be used to support custom R classes. By
            default, the dictionary used is
            :data:`~rdata.conversion._conversion.DEFAULT_CLASS_MAP`
            which has support for several common classes.
        default_encoding: Default encoding used for strings with unknown
            encoding. If `None`, the one stored in the file will be used, or
            ASCII as a fallback.
        force_default_encoding:
            Use the default encoding even if the strings specify other
            encoding.
        global_environment: Global environment to use. By default is an empty
            environment.
        base_environment: Base environment to use. By default is an empty
            environment.

    Examples:
        Parse one of the included examples, containing a vector

        >>> import rdata
        >>>
        >>> parsed = rdata.parser.parse_file(
        ...              rdata.TESTDATA_PATH / "test_vector.rda")
        >>> converted = rdata.conversion.convert(parsed)
        >>> converted
        {'test_vector': array([1., 2., 3.])}

        Parse another example, containing a dataframe

        >>> import rdata
        >>>
        >>> parsed = rdata.parser.parse_file(
        ...              rdata.TESTDATA_PATH / "test_dataframe.rda")
        >>> converted = rdata.conversion.convert(parsed)
        >>> converted
        {'test_dataframe':   class  value
        1     a      1
        2     b      2
        3     b      3}

    """
    return SimpleConverter(
        constructor_dict=constructor_dict,
        default_encoding=default_encoding,
        force_default_encoding=force_default_encoding,
        global_environment=global_environment,
        base_environment=base_environment,
    ).convert(data)
