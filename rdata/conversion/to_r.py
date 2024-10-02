"""Conversion functions from Python object to RData object."""

from __future__ import annotations

import string
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rdata.missing import R_FLOAT_NA, R_INT_NA
from rdata.parser import (
    CharFlags,
    RData,
    RExtraInfo,
    RObject,
    RObjectInfo,
    RObjectType,
    RVersions,
)

from . import (
    RExpression,
    RLanguage,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any, Final, Literal, Protocol

    import numpy.typing as npt

    from rdata.unparser import FileType

    Encoding = Literal["utf-8", "cp1252"]

    class ConversionFunction(Protocol):
        """Protocol for Py-to-R conversion function."""

        def __call__(self, data: Any) -> RObject:  # noqa: ANN401
            """Convert Python object to R object."""


    class ConstructorFunction(Protocol):
        """Protocol for Py-to-R constructor function."""

        def __call__(self,
            data: Any,  # noqa: ANN401
            convert_to_r_object: ConversionFunction,
        ) -> tuple[RObjectType, Any, dict[str, Any]]:
            """Convert Python object to R object components."""

    ConstructorDict = Mapping[type, ConstructorFunction]


# Default values for RVersions object
DEFAULT_FORMAT_VERSION: Final[int] = 3
DEFAULT_R_VERSION_SERIALIZED: Final[int] = 0x40201

# Mapping from format version to minimum R version
R_MINIMUM_VERSIONS: Final[Mapping[int, int]] = MappingProxyType({
    2: 0x20300,
    3: 0x30500,
})
R_MINIMUM_VERSION_WITH_ENCODING: Final[int] = 3
R_MINIMUM_VERSION_WITH_ALTREP: Final[int] = 3


def categorical_constructor(
    data: pd.Categorical,
    convert_to_r_object: ConversionFunction,  # noqa: ARG001
) -> tuple[RObjectType, Any, dict[str, Any]]:
    """
    Construct R object components from pandas categorical.

    Args:
        data: Pandas categorical.
        convert_to_r_object: Conversion function.

    Returns:
        Components of the R object.
    """
    assert isinstance(data, pd.Categorical)
    r_type = RObjectType.INT
    r_value = data.codes + 1
    attributes = {
        "levels": data.categories.to_numpy(),
        "class": "factor",
    }
    return r_type, r_value, attributes


def dataframe_constructor(
    data: pd.DataFrame,
    convert_to_r_object: ConversionFunction,
) -> tuple[RObjectType, Any, dict[str, Any]]:
    """
    Construct R object components from pandas dataframe.

    Args:
        data: Pandas dataframe.
        convert_to_r_object: Conversion function.

    Returns:
        Components of the R object.
    """
    assert isinstance(data, pd.DataFrame)
    r_type = RObjectType.VEC
    column_names = []
    r_value = []
    for column, series in data.items():
        assert isinstance(column, str)
        column_names.append(column)

        pd_array = series.array
        array: pd.Categorical | npt.NDArray[Any]
        if isinstance(pd_array, pd.Categorical):
            array = pd_array
        else:
            array = convert_pd_array_to_np_array(pd_array)
        r_series = convert_to_r_object(array)
        r_value.append(r_series)

    index = data.index
    if isinstance(index, pd.RangeIndex):
        assert isinstance(index.start, int)
        if (index.start == 1
            and index.stop == data.shape[0] + 1
            and index.step == 1
        ):
            row_names = np.ma.array(
                data=[R_INT_NA, -data.shape[0]],
                mask=[True, False],
                fill_value=R_INT_NA,
            )
        else:
            row_names = range(index.start, index.stop, index.step)
    elif isinstance(index, pd.Index):
        if (index.dtype == "object"
            or np.issubdtype(str(index.dtype), np.integer)):
            row_names = index.to_numpy()
        else:
            msg = f"pd.DataFrame pd.Index {index.dtype} not implemented"
            raise NotImplementedError(msg)
    else:
        msg = f"pd.DataFrame index {type(index)} not implemented"
        raise NotImplementedError(msg)

    attributes = {
        "names": np.array(column_names, dtype=np.dtype("U")),
        "class": "data.frame",
        "row.names": row_names,
    }
    return r_type, r_value, attributes


DEFAULT_CONSTRUCTOR_DICT: Final[ConstructorDict] = MappingProxyType({
    pd.Categorical: categorical_constructor,
    pd.DataFrame: dataframe_constructor,
})


def convert_pd_array_to_np_array(
    pd_array: pd.api.extensions.ExtensionArray,
) -> npt.NDArray[Any]:
    """
    Convert pandas array object to numpy array.

    Args:
        pd_array: Pandas array.

    Returns:
        Numpy array.
    """
    if isinstance(pd_array, pd.arrays.StringArray):
        return pd_array.to_numpy()

    if isinstance(pd_array, (
        pd.arrays.BooleanArray,
        pd.arrays.IntegerArray,
    )):
        dtype: type[Any]
        fill_value: bool | int
        if isinstance(pd_array, pd.arrays.BooleanArray):
            dtype = np.bool_
            fill_value = True
        elif isinstance(pd_array, pd.arrays.IntegerArray):
            dtype = np.int32
            fill_value = R_INT_NA

        mask = pd_array.isna()  # type: ignore [no-untyped-call]
        if np.any(mask):
            data = pd_array.to_numpy(dtype=dtype, na_value=fill_value)
            array = np.ma.array(  # type: ignore [no-untyped-call]
                data=data,
                mask=mask,
                fill_value=fill_value,
            )
        else:
            array = pd_array.to_numpy()
        assert array.dtype == dtype
        assert isinstance(array, np.ndarray)  # for mypy
        return array

    if isinstance(pd_array, (
        pd.arrays.FloatingArray,  # type: ignore [attr-defined]
    )):
        # Note that this possibly maps all NaNs (not only R_FLOAT_NA)
        # to the same `na_value` depending on how the array was built:
        array = pd_array.to_numpy(dtype=np.float64, na_value=R_FLOAT_NA)
        assert isinstance(array, np.ndarray)  # for mypy
        return array

    if isinstance(pd_array, (
        pd.arrays.NumpyExtensionArray,  # type: ignore [attr-defined]
    )):
        array = pd_array.to_numpy()
        assert isinstance(array, np.ndarray)  # for mypy
        return array

    msg = f"pandas array {type(array)} not implemented"
    raise NotImplementedError(msg)


def build_r_object(
    r_type: RObjectType,
    *,
    value: Any = None,  # noqa: ANN401
    is_object: bool = False,
    attributes: RObject | None = None,
    tag: RObject | None = None,
    gp: int = 0,
    reference: tuple[int, RObject | None] = (0, None),
) -> RObject:
    """
    Build R object.

    Args:
        r_type: Type indentifier.
        value: Value for RObject.
        is_object: True if RObject represents object.
        attributes: Same as in RObject.
        tag: Same as in RObject.
        gp: Same as in RObjectInfo.
        reference: Tuple of integer and object.

    Returns:
        R object.

    See Also:
        RObject
        RObjectInfo
    """
    assert r_type is not None
    reference_id, referenced_object = reference
    assert ((reference_id == 0)
            == (referenced_object is None)
            == (r_type != RObjectType.REF)
            )
    return RObject(
        RObjectInfo(
            r_type,
            object=is_object,
            attributes=attributes is not None,
            tag=tag is not None,
            gp=gp,
            reference=reference_id,
         ),
         value,
         attributes,
         tag,
         referenced_object,
     )


def build_r_list(
    data: list[RObject] | list[tuple[RObject, RObject]],
) -> RObject:
    """
    Build R object representing (named) linked list.

    Args:
        data: Non-empty list of values or (key, value) pairs.

    Returns:
        R object.
    """
    if len(data) == 0:
        msg = "data must not be empty"
        raise ValueError(msg)

    head = data[0]
    tail = data[1:]
    if isinstance(head, tuple):
        tag, car = head
    else:
        tag = None
        car = head

    cdr = build_r_object(RObjectType.NILVALUE) if len(tail) == 0 else build_r_list(tail)

    return build_r_object(RObjectType.LIST, value=(car, cdr), tag=tag)


def build_r_str(
    data: str,
    *,
    encoding: Encoding,
) -> RObject:
    """
    Build R object representing string.

    Args:
        data: String.
        encoding: Encoding used for strings.

    Returns:
        R object.
    """
    value = [build_r_char(data, encoding=encoding)]
    return build_r_object(RObjectType.STR, value=value)


def build_r_char(
    data: str | bytes | None,
    *,
    encoding: Encoding,
) -> RObject:
    """
    Build R object representing characters.

    Args:
        data: String or bytestring.
        encoding: Encoding used for strings.

    Returns:
        R object.
    """
    if data is None:
        return build_r_object(RObjectType.CHAR)

    if isinstance(data, str):
        data = data.encode(encoding)

    if all(chr(byte) in string.printable for byte in data):
        gp = CharFlags.ASCII
    elif encoding == "utf-8":
        gp = CharFlags.UTF8
    elif encoding == "cp1252":
        # Note!
        # CP1252 and Latin1 are not the same.
        # Does CharFlags.LATIN1 mean actually CP1252
        # as R on Windows mentions CP1252 as encoding?
        # Or does CP1252 change to e.g. CP1250 depending on localization?
        gp = CharFlags.LATIN1
    else:
        msg = f"unsupported encoding: {encoding}"
        raise ValueError(msg)
    return build_r_object(RObjectType.CHAR, value=data, gp=gp)


class ConverterFromPythonToR:
    """
    Class converting Python objects to R objects.

    Attributes:
        constructor_dict: Dictionary mapping Python types to R classes.
        encoding: Encoding to be used for strings within data.
        format_version: File format version.
        r_version_serialized: R version written as the creator of the object.
    """
    def __init__(self,
        constructor_dict: ConstructorDict = DEFAULT_CONSTRUCTOR_DICT,
        *,
        encoding: Encoding = "utf-8",
        format_version: int = DEFAULT_FORMAT_VERSION,
        r_version_serialized: int = DEFAULT_R_VERSION_SERIALIZED,
    ) -> None:
        """
        Init class.

        Args:
            constructor_dict: Dictionary mapping Python types to R classes.
            encoding: Encoding to be used for strings within data.
            format_version: File format version.
            r_version_serialized: R version written as the creator of the object.
        """
        self.constructor_dict = constructor_dict
        self.encoding = encoding
        self.format_version = format_version
        self.r_version_serialized = r_version_serialized
        self._references: dict[str | None, tuple[int, RObject | None]] \
            = {None: (0, None)}

        # In test files the order in which dataframe attributes are written varies.
        # R can read files with attributes in any order, but this variable
        # is used in tests to change the attribute order to match with the test file.
        self.df_attr_order: list[str] | None = None


    def convert_to_r_data(self,
        data: Any,  # noqa: ANN401
        *,
        file_type: FileType = "rds",
    ) -> RData:
        """
        Convert Python data to R data.

        Args:
            data: Any Python object.
            file_type: File type.

        Returns:
            Corresponding RData object.

        See Also:
            convert_to_r_object
        """
        if file_type == "rda":
            if not isinstance(data, dict):
                msg = f"for RDA file, data must be a dictionary, not type {type(data)}"
                raise TypeError(msg)
            r_object = self.convert_to_r_attributes(data)
        else:
            r_object = self.convert_to_r_object(data)

        versions = RVersions(
            self.format_version,
            self.r_version_serialized,
            R_MINIMUM_VERSIONS[self.format_version],
        )

        extra = (RExtraInfo(self.encoding.upper())
                 if versions.format >= R_MINIMUM_VERSION_WITH_ENCODING
                 else RExtraInfo(None))

        return RData(versions, extra, r_object)


    def convert_to_r_attributes(self,
        data: dict[str, Any],
    ) -> RObject:
        """
        Convert dictionary to R attributes list.

        Args:
            data: Non-empty dictionary.

        Returns:
            R object.
        """
        converted = []
        for key, value in data.items():
            converted.append((
                self.convert_to_r_sym(key),
                self.convert_to_r_object(value),
            ))

        return build_r_list(converted)


    def convert_to_r_sym(self,
        name: str,
    ) -> RObject:
        """
        Convert string to R symbol.

        Args:
            name: String.

        Returns:
            R object.
        """
        # Reference to existing symbol if exists
        if name in self._references:
            reference = self._references[name]
            return build_r_object(RObjectType.REF, reference=reference)

        # Create a new symbol
        r_value = self.convert_to_r_object(name.encode(self.encoding))
        r_object = build_r_object(RObjectType.SYM, value=r_value)

        # Add to reference list
        self._references[name] = (len(self._references), r_object)
        return r_object


    def convert_to_r_object(self,  # noqa: C901, PLR0912, PLR0915
        data: Any,  # noqa: ANN401
    ) -> RObject:
        """
        Convert Python data to R object.

        Args:
            data: Any Python object.

        Returns:
            Corresponding R object.
        """
        # Default args for most types (None/False/0)
        r_type = None
        values: list[Any] | tuple[Any, ...]
        r_value: Any = None
        is_object = False
        attributes: dict[str, Any] | None = None

        if data is None:
            r_type = RObjectType.NILVALUE

        elif isinstance(data, RExpression):
            r_type = RObjectType.EXPR
            r_value = [self.convert_to_r_object(el) for el in data.elements]

        elif isinstance(data, RLanguage):
            r_type = RObjectType.LANG
            symbols = [self.convert_to_r_sym(el) for el in data.elements]
            r_value = (symbols[0], build_r_list(symbols[1:]))

            if len(data.attributes) > 0:
                # The following might work here (untested)
                # attributes = data.attributes  # noqa: ERA001
                msg = f"type {r_type} with attributes not implemented"
                raise NotImplementedError(msg)

        elif isinstance(data, (list, tuple, dict)):
            r_type = RObjectType.VEC
            values = list(data.values()) if isinstance(data, dict) else data
            r_value = [self.convert_to_r_object(el) for el in values]

            if isinstance(data, dict):
                names = np.array(list(data.keys()), dtype=np.dtype("U"))
                attributes = {"names": names}

        elif isinstance(data, np.ndarray):
            if data.dtype.kind in ["O"]:
                assert data.ndim == 1
                r_type = RObjectType.STR
                r_value = []
                for el in data:
                    if el is None or pd.isna(el):
                        r_el = build_r_char(None, encoding=self.encoding)
                    elif isinstance(el, str):
                        r_el = build_r_char(el, encoding=self.encoding)
                    else:
                        msg = "general object array not implemented"
                        raise NotImplementedError(msg)
                    r_value.append(r_el)

            elif data.dtype.kind in ["S", "U"]:
                assert data.ndim == 1
                r_type = RObjectType.STR
                r_value = [build_r_char(el, encoding=self.encoding)
                           for el in data]

            else:
                r_type = {
                    "b": RObjectType.LGL,
                    "i": RObjectType.INT,
                    "f": RObjectType.REAL,
                    "c": RObjectType.CPLX,
                }[data.dtype.kind]

                if data.ndim == 0:
                    r_value = data[np.newaxis]
                elif data.ndim == 1:
                    r_value = data
                else:
                    # R uses column-major order like Fortran
                    r_value = np.ravel(data, order="F")
                    attributes = {"dim": np.array(data.shape)}

        elif isinstance(data, (bool, int, float, complex)):
            return self.convert_to_r_object(np.array(data))

        elif isinstance(data, str):
            return build_r_str(data, encoding=self.encoding)

        elif isinstance(data, bytes):
            return build_r_char(data, encoding=self.encoding)

        elif isinstance(data, range):
            if self.format_version < R_MINIMUM_VERSION_WITH_ALTREP:
                # ALTREP support is from R version 3.5.0
                # (minimum version for format version 3)
                return self.convert_to_r_object(np.array(data))

            if data.step != 1:
                # R supports compact sequences only with step 1;
                # convert the range to an array of values
                return self.convert_to_r_object(np.array(data))

            r_type = RObjectType.ALTREP
            r_value = (
                build_r_list([
                    self.convert_to_r_sym("compact_intseq"),
                    self.convert_to_r_sym("base"),
                    self.convert_to_r_object(RObjectType.INT.value),
                ]),
                self.convert_to_r_object(np.array([
                    len(data),
                    data.start,
                    data.step,
                ], dtype=float)),
                self.convert_to_r_object(None),
            )

        else:
            # Check available constructors
            for t, constructor in self.constructor_dict.items():
                if isinstance(data, t):
                    r_type, r_value, attributes \
                        = constructor(data, self.convert_to_r_object)
                    break

            if r_type is None:
                msg = f"type {type(data)} not implemented"
                raise NotImplementedError(msg)

            # Fix for test files where dataframe attribute order varies
            assert isinstance(attributes, dict)
            if isinstance(data, pd.DataFrame) and self.df_attr_order is not None:
                attributes = {k: attributes[k] for k in self.df_attr_order}

        if attributes is not None:
            is_object = "class" in attributes
            r_attributes = self.convert_to_r_attributes(attributes)
        else:
            r_attributes = None

        return build_r_object(
            r_type,
            value=r_value,
            is_object=is_object,
            attributes=r_attributes,
        )


def convert_python_to_r_data(
    data: Any,  # noqa: ANN401
    *,
    encoding: Encoding = "utf-8",
    format_version: int = DEFAULT_FORMAT_VERSION,
    r_version_serialized: int = DEFAULT_R_VERSION_SERIALIZED,
    file_type: FileType = "rds",
) -> RData:
    """
    Convert Python data to R data.

    Args:
        data: Any Python object.
        encoding: Encoding to be used for strings within data.
        format_version: File format version.
        r_version_serialized: R version written as the creator of the object.
        file_type: File type.

    Returns:
        Corresponding RData object.

    See Also:
        convert_python_to_r_object
    """
    return ConverterFromPythonToR(
        encoding=encoding,
        format_version=format_version,
        r_version_serialized=r_version_serialized,
    ).convert_to_r_data(data, file_type=file_type)


def convert_python_to_r_object(
    data: Any,  # noqa: ANN401
    *,
    encoding: Encoding = "utf-8",
    format_version: int = DEFAULT_FORMAT_VERSION,
    r_version_serialized: int = DEFAULT_R_VERSION_SERIALIZED,
) -> RObject:
    """
    Convert Python data to R object.

    Args:
        data: Any Python object.
        encoding: Encoding to be used for strings within data.
        format_version: File format version.
        r_version_serialized: R version written as the creator of the object.

    Returns:
        Corresponding RObject object.

    See Also:
        convert_python_to_r_data
    """
    return ConverterFromPythonToR(
        encoding=encoding,
        format_version=format_version,
        r_version_serialized=r_version_serialized,
    ).convert_to_r_object(data)
