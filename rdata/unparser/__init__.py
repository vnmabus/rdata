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
    CompressionType = Literal["gzip", "bzip2", "xz", "none"]


def unparse_file(
        path: os.PathLike[Any] | str,
        r_data: RData,
        *,
        file_format: FileFormatType = "xdr",
        rds: bool = True,
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
    rds:
        Whether to create RDS or RDA file
    compression:
        Compression
    """
    if compression == "none":
        from builtins import open  # noqa: UP029
    elif compression == "bzip2":
        from bz2 import open  # type: ignore [no-redef]
    elif compression == "gzip":
        from gzip import open  # type: ignore [no-redef]
    elif compression == "xz":
        from lzma import open  # type: ignore [no-redef]
    else:
        msg = f"Unknown compression: {compression}"
        if compression is None:
            msg += ". Use 'none' for no compression."
        raise ValueError(msg)

    with open(path, "wb") as f:
        unparse_fileobj(f, r_data, file_format=file_format, rds=rds)


def unparse_fileobj(
        fileobj: IO[Any],
        r_data: RData,
        *,
        file_format: FileFormatType = "xdr",
        rds: bool = True,
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
    rds:
        Whether to create RDS or RDA file
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
    unparser.unparse_r_data(r_data, rds=rds)


def unparse_data(
        r_data: RData,
        *,
        file_format: FileFormatType = "xdr",
        rds: bool = True,
) -> bytes:
    """
    Unparse RData object to a bytestring.

    Parameters
    ----------
    r_data:
        RData object
    file_format:
        File format
    rds:
        Whether to create RDS or RDA file

    Returns:
    -------
    data:
        Bytestring of data
    """
    fd = io.BytesIO()
    unparse_fileobj(fd, r_data, file_format=file_format, rds=rds)
    return fd.getvalue()
