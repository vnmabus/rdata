"""Conversion functions from Python object to RData object."""

from __future__ import annotations

import string
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rdata.parser import (
    R_INT_NA,
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

    Encoding = Literal["utf-8", "cp1252"]


    class Converter(Protocol):
        """Protocol for Py-to-R conversion."""

        def __call__(self, data: Any, *, encoding: Encoding) -> RObject: # noqa: ANN401
            """Convert Python object to R object."""



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


def create_unicode_array(
        names: Any,  # noqa: ANN401
) -> npt.NDArray[Any]:
    """
    Create unicode array from sequence/iterator of strings.

    Args:
        names: Strings.

    Returns:
        Array.
    """
    name_list = []
    for name in names:
        assert isinstance(name, str)
        name_list.append(name)
    return np.array(name_list, dtype=np.dtype("U"))


def build_r_object(
        r_type: RObjectType,
        *,
        value: Any = None,  # noqa: ANN401
        is_object: bool = False,
        attributes: RObject | None = None,
        tag: RObject | None = None,
        gp: int = 0,
        reference: tuple(int, RObject | None) = (0, None),
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



class ConverterFromPythonToR:
    """
    Class converting Python objects to R objects.

    Attributes:
        encoding: Encoding to be used for strings within data.
        format_version: File format version.
        r_version_serialized: R version written as the creator of the object.
    """
    def __init__(self, *,
            encoding: Encoding = "utf-8",
            format_version: int = DEFAULT_FORMAT_VERSION,
            r_version_serialized: int = DEFAULT_R_VERSION_SERIALIZED,
    ) -> None:
        """
        Init class.

        Args:
            encoding: Encoding to be used for strings within data.
            format_version: File format version.
            r_version_serialized: R version written as the creator of the object.
        """
        self.encoding = encoding
        self.format_version = format_version
        self.r_version_serialized = r_version_serialized
        self.reference_name_list = [None]
        self.reference_obj_list = [None]


    def build_r_data(self,
            r_object: RObject,
    ) -> RData:
        """
        Build RData object from R object.

        Args:
            r_object: R object.

        Returns:
            Corresponding RData object.

        See Also:
            convert_to_r_object
        """
        versions = RVersions(
            self.format_version,
            self.r_version_serialized,
            R_MINIMUM_VERSIONS[self.format_version],
        )

        extra = (RExtraInfo(self.encoding.upper())
                 if versions.format >= R_MINIMUM_VERSION_WITH_ENCODING
                 else RExtraInfo(None))

        return RData(versions, extra, r_object)


    def build_r_list(self,
            data: Mapping[str, Any] | list[Any],
            *,
            convert_value: Converter | None = None,
    ) -> RObject:
        """
        Build R object representing named linked list.

        Args:
            data: Non-empty dictionary or list.
            convert_value: Function used for converting value to R object
                (for example, convert_to_r_object).

        Returns:
            R object.
        """
        if convert_value is None:
            convert_value = self.convert_to_r_object

        if len(data) == 0:
            msg = "data must not be empty"
            raise ValueError(msg)

        if isinstance(data, dict):
            data = data.copy()
            key = next(iter(data))
            tag = self.build_r_sym(key)
            car = data.pop(key)
        elif isinstance(data, list):
            car = data[0]
            data = data[1:]
            tag = None

        if not isinstance(car, RObject):
            car = convert_value(car)

        if len(data) == 0:
            cdr = build_r_object(RObjectType.NILVALUE)
        else:
            cdr = self.build_r_list(data, convert_value=convert_value)

        return build_r_object(
            RObjectType.LIST,
            value=(car, cdr),
            tag=tag,
            )


    def build_r_sym(self,
            name: str,
    ) -> RObject:
        """
        Build R object representing symbol.

        Args:
            name: String.

        Returns:
            R object.
        """
        # Reference to existing symbol if exists
        if name in self.reference_name_list:
            idx = self.reference_name_list.index(name)
            obj = self.reference_obj_list[idx]
            return build_r_object(RObjectType.REF, reference=(idx, obj))

        # Create a new symbol
        r_value = self.convert_to_r_object(name.encode(self.encoding))
        r_object = build_r_object(RObjectType.SYM, value=r_value)

        # Add to reference list
        self.reference_name_list.append(name)
        self.reference_obj_list.append(r_object)
        return r_object


    def convert_to_r_object_for_rda(self,
            data: Mapping[str, Any],
    ) -> RObject:
        """
        Convert Python dictionary to R object for RDA file.

        Args:
            data: Python dictionary with data and variable names.

        Returns:
            Corresponding R object.

        See Also:
            convert_to_r_object
        """
        if not isinstance(data, dict):
            msg = f"for RDA file, data must be a dictionary, not type {type(data)}"
            raise TypeError(msg)
        return self.build_r_list(data)


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
        attributes = None
        tag = None
        gp = 0

        if data is None:
            r_type = RObjectType.NILVALUE

        elif isinstance(data, RExpression):
            r_type = RObjectType.EXPR
            r_value = [self.convert_to_r_object(el) for el in data.elements]

        elif isinstance(data, RLanguage):
            r_type = RObjectType.LANG
            values = data.elements
            r_value = (self.build_r_sym(str(values[0])),
                       self.build_r_list(values[1:],
                                         convert_value=self.build_r_sym))

            if len(data.attributes) > 0:
                # The following might work here (untested)
                # attributes = build_r_list(data.attributes)  # noqa: ERA001
                msg = f"type {r_type} with attributes not implemented"
                raise NotImplementedError(msg)

        elif isinstance(data, (list, tuple, dict)):
            r_type = RObjectType.VEC
            values = list(data.values()) if isinstance(data, dict) else data
            r_value = [self.convert_to_r_object(el) for el in values]

            if isinstance(data, dict):
                names = create_unicode_array(data.keys())
                attributes = self.build_r_list({"names": names})

        elif isinstance(data, np.ndarray):
            if data.dtype.kind in ["O"]:
                # This is a special case handling only np.array([None])
                if data.size != 1 or data[0] is not None:
                    msg = "general object array not implemented"
                    raise NotImplementedError(msg)
                r_type = RObjectType.STR
                r_value = [build_r_object(RObjectType.CHAR)]

            elif data.dtype.kind in ["S"]:
                assert data.ndim == 1
                r_type = RObjectType.STR
                r_value = [self.convert_to_r_object(el) for el in data]

            elif data.dtype.kind in ["U"]:
                assert data.ndim == 1
                data = np.array([s.encode(self.encoding) for s in data],
                                dtype=np.dtype("S"))
                return self.convert_to_r_object(data)

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
                    attributes = self.build_r_list({"dim": np.array(data.shape)})

        elif isinstance(data, (bool, int, float, complex)):
            return self.convert_to_r_object(np.array(data))

        elif isinstance(data, str):
            r_type = RObjectType.STR
            r_value = [self.convert_to_r_object(data.encode(self.encoding))]

        elif isinstance(data, bytes):
            r_type = RObjectType.CHAR
            if all(chr(byte) in string.printable for byte in data):
                gp = CharFlags.ASCII
            elif self.encoding == "utf-8":
                gp = CharFlags.UTF8
            elif self.encoding == "cp1252":
                # Note!
                # CP1252 and Latin1 are not the same.
                # Does CharFlags.LATIN1 mean actually CP1252
                # as R on Windows mentions CP1252 as encoding?
                # Or does CP1252 change to e.g. CP1250 depending on localization?
                gp = CharFlags.LATIN1
            else:
                msg = f"unsupported encoding: {self.encoding}"
                raise ValueError(msg)
            r_value = data

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
                self.build_r_list([
                    self.build_r_sym("compact_intseq"),
                    self.build_r_sym("base"),
                    RObjectType.INT.value,
                ]),
                self.convert_to_r_object(np.array([
                    len(data),
                    data.start,
                    data.step,
                ], dtype=float)),
                self.convert_to_r_object(None),
            )

        elif isinstance(data, pd.Series):
            msg = "pd.Series not implemented"
            raise NotImplementedError(msg)

        elif isinstance(data, pd.Categorical):
            is_object = True
            r_type = RObjectType.INT
            r_value = data.codes + 1
            attributes = self.build_r_list({
                "levels": create_unicode_array(data.categories),
                "class": "factor",
                })

        elif isinstance(data, pd.DataFrame):
            is_object = True
            r_type = RObjectType.VEC
            names = []
            r_value = []
            for column, series in data.items():
                assert isinstance(column, str)
                names.append(column)

                array = series.array
                if isinstance(array, pd.Categorical):
                    r_series = self.convert_to_r_object(array)
                elif isinstance(array, pd.arrays.StringArray):
                    r_series = self.convert_to_r_object(create_unicode_array(array))
                elif isinstance(array, (
                         pd.arrays.IntegerArray,
                         pd.arrays.NumpyExtensionArray,
                )):
                    r_series = self.convert_to_r_object(array.to_numpy())
                else:
                    msg = f"pd.DataFrame with pd.Series {type(array)} not implemented"
                    raise NotImplementedError(msg)

                r_value.append(r_series)

            index = data.index
            attr_order = ["names", "row.names", "class"]
            if isinstance(index, pd.RangeIndex):
                if (index.start == 1
                    and index.stop == data.shape[0] + 1
                    and index.step == 1
                ):
                    row_names = np.ma.array(  # type: ignore [no-untyped-call]
                            data=[R_INT_NA, -data.shape[0]],
                            mask=[True, False],
                            fill_value=R_INT_NA,
                        )
                else:
                    row_names = range(index.start, index.stop, index.step)
            elif isinstance(index, pd.Index):
                attr_order = ["names", "class", "row.names"]
                if index.dtype == "object":
                    row_names = create_unicode_array(index)
                elif np.issubdtype(index.dtype, np.integer):
                    row_names = index.to_numpy()
                else:
                    msg = f"pd.DataFrame pd.Index {index.dtype} not implemented"
                    raise NotImplementedError(msg)
            else:
                msg = f"pd.DataFrame index {type(index)} not implemented"
                raise NotImplementedError(msg)

            attr_dict = {
                "names": create_unicode_array(names),
                "row.names": row_names,
                "class": "data.frame",
            }

            attributes = self.build_r_list({k: attr_dict[k] for k in attr_order})

        else:
            msg = f"type {type(data)} not implemented"
            raise NotImplementedError(msg)

        return build_r_object(r_type, value=r_value,
                              is_object=is_object,
                              attributes=attributes,
                              tag=tag, gp=gp)
