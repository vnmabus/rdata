import string
from typing import (
    Any,
    Optional,
)

import numpy as np

from rdata.parser._parser import (
    CharFlags,
    RData,
    RExtraInfo,
    RObject,
    RObjectInfo,
    RObjectType,
    RVersions,
)


def build_r_object(
        r_type: RObjectType,
        *,
        value: Any = None,
        attributes: Optional[RObject] = None,
        tag: Optional[RObject] = None,
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
    r_object = RObject(RObjectInfo(r_type,
                                   object=False,
                                   attributes=attributes is not None,
                                   tag=tag is not None,
                                   gp=int(gp),
                                   reference=0),
                       value,
                       attributes,
                       tag,
                       None)
    return r_object


def build_r_list(
        dct: dict[str, Any],
        *,
        encoding: str,
) -> RObject:
    """
    Build R object representing named linked list.

    Parameters
    ----------
    dct:
        Dictionary.
    encoding:
        Encoding to be used for strings within data.

    Returns:
    -------
    r_object:
        RObject object.
    """
    dct = dct.copy()
    key = next(iter(dct))
    value = convert_to_r_object(dct.pop(key), encoding=encoding)

    if len(dct) == 0:
        value2 = build_r_object(RObjectType.NILVALUE)
    else:
        value2 = build_r_list(dct, encoding=encoding)

    r_list = build_r_object(
        RObjectType.LIST,
        value=(
            value,
            value2,
            ),
        tag=build_r_object(
            RObjectType.SYM,
            value=convert_to_r_object(key.encode("ascii"), encoding=encoding),
            ),
        )
    return r_list


def convert_to_r_data(
        data: Any,
        *,
        encoding: str = "UTF-8",
        rds: bool = True,
        versions: RVersions = None,
) -> RData:
    """
    Convert Python data to RData object.

    Parameters
    ----------
    data:
        Any Python object.
    encoding:
        Encoding to be used for strings within data.

    Returns:
    -------
    r_data:
        Corresponding RData object.

    See Also:
    --------
    convert_to_r_object
    """
    if versions is None:
        versions = RVersions(3, 262657, 197888)
    if versions.format == 2:
        extra = RExtraInfo(None)
    else:
        extra = RExtraInfo(encoding)

    if rds:
        obj = convert_to_r_object(data, encoding=encoding)
    else:
        if not isinstance(data, dict):
            raise ValueError("For RDA file, data must be a dictionary.")
        obj = build_r_list(data, encoding=encoding)

    return RData(versions, extra, obj)


def convert_to_r_object(
        data: Any,
        *,
        encoding: str,
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
    if encoding not in ["UTF-8", "CP1252"]:
        raise ValueError(f"Unknown encoding: {encoding}")

    # Default args for most types (None/False/0)
    r_type = None
    r_value = None
    gp = 0
    attributes = None
    tag = None

    if data is None:
        r_type = RObjectType.NILVALUE

    elif isinstance(data, (list, tuple, dict)):
        r_type = RObjectType.VEC

        if isinstance(data, dict):
            values = data.values()
        else:
            values = data

        r_value = []
        for element in values:
            r_value.append(convert_to_r_object(element, encoding=encoding))

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
            r_value = []
            for element in data:
                r_value.append(convert_to_r_object(element, encoding=encoding))

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
            # XXX CP1252 and Latin1 are not the same
            #     Check if CharFlags.LATIN1 means actually CP1252
            #     as R on Windows mentions CP1252 as encoding
            gp = CharFlags.LATIN1
        else:
            raise NotImplementedError("unknown what gp value to use")
        r_value = data

    else:
        raise NotImplementedError(f"{type(data)}")

    return build_r_object(r_type, value=r_value, attributes=attributes, tag=tag, gp=gp)
