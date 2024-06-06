"""Conversion functions from Python object to RData object."""

from __future__ import annotations

import string
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

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
    from typing import Any, Final, Protocol

    class Converter(Protocol):
        """Protocol for Py-to-R conversion."""

        def __call__(self, data: Any, *, encoding: str) -> RObject: # noqa: ANN401
            """Convert Python object to R object."""



# Default values for RVersions object
DEFAULT_FORMAT_VERSION: Final[int] = 3
DEFAULT_R_VERSION_SERIALIZED: Final[int] = 0x40201

# Mapping from format version to minimum R version
R_MINIMUM_VERSIONS: Final[Mapping[int, int]] = MappingProxyType({
    2: 0x20300,
    3: 0x30500,
})


def build_r_object(
        r_type: RObjectType,
        *,
        value: Any = None,  # noqa: ANN401
        attributes: RObject | None = None,
        tag: RObject | None = None,
        gp: int = 0,
) -> RObject:
    """
    Build R object.

    Parameters
    ----------
    r_type:
        Type indentifier
    value:
        Value
    attributes:
        Same as in RObject
    tag:
        Same as in RObject
    gp:
        Same as in RObjectInfo

    Returns:
    -------
    r_object:
        RObject object.

    See Also:
    --------
    RObject
    RObjectInfo
    """
    assert r_type is not None
    return RObject(
        RObjectInfo(
            r_type,
            object=False,
            attributes=attributes is not None,
            tag=tag is not None,
            gp=int(gp),
            reference=0,
         ),
         value,
         attributes,
         tag,
         None,
     )


def build_r_list(
        data: Mapping[str, Any] | list[Any],
        *,
        encoding: str,
        convert_value: Converter | None = None,
) -> RObject:
    """
    Build R object representing named linked list.

    Parameters
    ----------
    data:
        Dictionary or list.
    encoding:
        Encoding to be used for strings within data.
    convert_value:
        Function used for converting value to R object
        (for example, convert_to_r_object).

    Returns:
    -------
    r_object:
        RObject object.
    """
    if convert_value is None:
        convert_value = convert_to_r_object

    if isinstance(data, dict):
        data = data.copy()
        key = next(iter(data))
        value1 = convert_value(data.pop(key), encoding=encoding)
        tag = build_r_sym(key, encoding=encoding)
    elif isinstance(data, list):
        value1 = convert_value(data[0], encoding=encoding)
        data = data[1:]
        tag = None

    if len(data) == 0:
        value2 = build_r_object(RObjectType.NILVALUE)
    else:
        value2 = build_r_list(data, encoding=encoding, convert_value=convert_value)

    return build_r_object(
        RObjectType.LIST,
        value=(value1, value2),
        tag=tag,
        )


def build_r_sym(
        data: str,
        *,
        encoding: str,
) -> RObject:
    """
    Build R object representing symbol.

    Parameters
    ----------
    data:
        String.
    encoding:
        Encoding to be used for strings within data.

    Returns:
    -------
    r_object:
        RObject object.
    """
    r_type = RObjectType.SYM
    r_value = convert_to_r_object(data.encode(encoding), encoding=encoding)
    return build_r_object(r_type, value=r_value)


def build_r_data(
        r_object: RObject,
        *,
        encoding: str = "UTF-8",
        format_version: int = DEFAULT_FORMAT_VERSION,
        r_version_serialized: int = DEFAULT_R_VERSION_SERIALIZED,
) -> RData:
    """
    Build RData object from R object.

    Parameters
    ----------
    r_object:
        R object.
    encoding:
        Encoding to be used for strings within data.
    format_version:
        File format version.
    r_version_serialized:
        R version written as the creator of the object.

    Returns:
    -------
    r_data:
        Corresponding RData object.

    See Also:
    --------
    convert_to_r_object
    """
    versions = RVersions(
        format_version,
        r_version_serialized,
        R_MINIMUM_VERSIONS[format_version],
    )

    minimum_version_with_encoding = 3
    extra = RExtraInfo(encoding) if versions.format >= minimum_version_with_encoding \
            else RExtraInfo(None)

    return RData(versions, extra, r_object)


def convert_to_r_object_for_rda(
        data: Mapping[str, Any],
        *,
        encoding: str = "UTF-8",
) -> RObject:
    """
    Convert Python dictionary to R object for RDA file.

    Parameters
    ----------
    data:
        Python dictionary with data and variable names.
    encoding:
        Encoding to be used for strings within data.

    Returns:
    -------
    r_object:
        Corresponding R object.

    See Also:
    --------
    convert_to_r_object
    """
    if not isinstance(data, dict):
        msg = "For RDA file, data must be a dictionary."
        raise TypeError(msg)
    return build_r_list(data, encoding=encoding)


def convert_to_r_object(  # noqa: C901, PLR0912, PLR0915
        data: Any,  # noqa: ANN401
        *,
        encoding: str = "UTF-8",
) -> RObject:
    """
    Convert Python data to R object.

    Parameters
    ----------
    data:
        Any Python object.
    encoding:
        Encoding to be used for strings within data.

    Returns:
    -------
    r_object:
        Corresponding R object.

    See Also:
    --------
    convert_to_r_data
    """
    # Default args for most types (None/False/0)
    r_type = None
    values: list[Any] | tuple[Any, ...]
    r_value: Any = None
    gp = 0
    attributes = None
    tag = None

    if data is None:
        r_type = RObjectType.NILVALUE

    elif isinstance(data, RExpression):
        r_type = RObjectType.EXPR
        r_value = [convert_to_r_object(el, encoding=encoding) for el in data.elements]

    elif isinstance(data, RLanguage):
        r_type = RObjectType.LANG
        values = data.elements
        r_value = (build_r_sym(str(values[0]), encoding=encoding),
                   build_r_list(values[1:], encoding=encoding,
                                convert_value=build_r_sym))

        if len(data.attributes) > 0:
            # The following might work here (untested)
            # attributes = build_r_list(data.attributes, encoding=encoding)  # noqa: ERA001,E501
            msg = f"type {r_type} with attributes not implemented"
            raise NotImplementedError(msg)

    elif isinstance(data, (list, tuple, dict)):
        r_type = RObjectType.VEC
        values = list(data.values()) if isinstance(data, dict) else data
        r_value = [convert_to_r_object(el, encoding=encoding) for el in values]

        if isinstance(data, dict):
            attributes = build_r_list({"names": np.array(list(data.keys()))},
                                      encoding=encoding)

    elif isinstance(data, np.ndarray):
        if data.dtype.kind in ["O"]:
            assert data.size == 1
            assert data[0] is None
            r_type = RObjectType.STR
            r_value = [build_r_object(RObjectType.CHAR)]

        elif data.dtype.kind in ["S"]:
            assert data.ndim == 1
            r_type = RObjectType.STR
            r_value = [convert_to_r_object(el, encoding=encoding) for el in data]

        elif data.dtype.kind in ["U"]:
            assert data.ndim == 1
            data = np.array([s.encode(encoding) for s in data])
            return convert_to_r_object(data, encoding=encoding)

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
                attributes = build_r_list({"dim": np.array(data.shape)},
                                          encoding=encoding)

    elif isinstance(data, (bool, int, float, complex)):
        return convert_to_r_object(np.array(data), encoding=encoding)

    elif isinstance(data, str):
        r_type = RObjectType.STR
        r_value = [convert_to_r_object(data.encode(encoding), encoding=encoding)]

    elif isinstance(data, bytes):
        r_type = RObjectType.CHAR
        if all(chr(byte) in string.printable for byte in data):
            gp = CharFlags.ASCII
        elif encoding == "UTF-8":
            gp = CharFlags.UTF8
        elif encoding == "CP1252":
            # Note!
            # CP1252 and Latin1 are not the same.
            # Does CharFlags.LATIN1 mean actually CP1252
            # as R on Windows mentions CP1252 as encoding?
            # Or does CP1252 change to e.g. CP1250 depending on localization?
            gp = CharFlags.LATIN1
        else:
            msg = "unsupported encoding: {encoding}"
            raise ValueError(msg)
        r_value = data

    else:
        msg = f"type {type(data)} not implemented"
        raise NotImplementedError(msg)

    return build_r_object(r_type, value=r_value, attributes=attributes, tag=tag, gp=gp)
