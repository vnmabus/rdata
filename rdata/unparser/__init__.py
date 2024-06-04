"""Utilities for unparsing a rdata file."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os
    from typing import IO, Any, Literal

    from rdata.parser import RData

    from ._ascii import UnparserASCII
    from ._xdr import UnparserXDR

    FileFormatType = Literal["xdr", "ascii"]
    FileTypeType = Literal["rds", "rda"]
    CompressionType = Literal["gzip", "bzip2", "xz", None]


def unparse_file(
        path: os.PathLike[Any] | str,
        r_data: RData,
        *,
        file_format: FileFormatType = "xdr",
        file_type: FileTypeType = "rds",
        compression: CompressionType = "gzip",
) -> None:
    """
    Unparse RData object to a file.

    Parameters
    ----------
    path:
        File path to be created
    r_data:
        RData object
    file_format:
        File format
    file_type:
        File type
    compression:
        Compression
    """
    if compression is None:
        from builtins import open  # noqa: UP029
    elif compression == "bzip2":
        from bz2 import open  # type: ignore [no-redef]
    elif compression == "gzip":
        from gzip import open  # type: ignore [no-redef]
    elif compression == "xz":
        from lzma import open  # type: ignore [no-redef]
    else:
        msg = f"Unknown compression: {compression}"
        raise ValueError(msg)

    with open(path, "wb") as f:
        unparse_fileobj(f, r_data, file_format=file_format, file_type=file_type)


def unparse_fileobj(
        fileobj: IO[Any],
        r_data: RData,
        *,
        file_format: FileFormatType = "xdr",
        file_type: FileTypeType = "rds",
) -> None:
    """
    Unparse RData object to a file object.

    Parameters
    ----------
    fileobj:
        File object
    r_data:
        RData object
    file_format:
        File format
    file_type:
        File type
    """
    Unparser: type[UnparserXDR | UnparserASCII]  # noqa: N806

    if file_format == "ascii":
        from ._ascii import UnparserASCII as Unparser
    elif file_format == "xdr":
        from ._xdr import UnparserXDR as Unparser
    else:
        msg = f"Unknown file format: {file_format}"
        raise ValueError(msg)

    unparser = Unparser(fileobj)  # type: ignore [arg-type]
    unparser.unparse_r_data(r_data, rds=file_type == "rds")


def unparse_data(
        r_data: RData,
        *,
        file_format: FileFormatType = "xdr",
        file_type: FileTypeType = "rds",
) -> bytes:
    """
    Unparse RData object to a bytestring.

    Parameters
    ----------
    r_data:
        RData object
    file_format:
        File format
    file_type:
        File type

    Returns:
    -------
    data:
        Bytestring of data
    """
    fd = io.BytesIO()
    unparse_fileobj(fd, r_data, file_format=file_format, file_type=file_type)
    return fd.getvalue()
