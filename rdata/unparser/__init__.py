"""Utilities for unparsing a rdata file."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from rdata.parser import (
    RData,
    RObjectType,
)

from ._ascii import UnparserASCII
from ._xdr import UnparserXDR

if TYPE_CHECKING:
    import os
    from collections.abc import Callable
    from typing import Any, Literal

    from ._unparser import WriteableBinaryFile

    FileFormat = Literal["xdr", "ascii"]
    FileType = Literal["rds", "rda"]
    Compression = Literal["gzip", "bzip2", "xz", None]


def unparse_file(
    path: os.PathLike[Any] | str,
    r_data: RData,
    *,
    file_format: FileFormat = "xdr",
    file_type: FileType = "rds",
    compression: Compression = "gzip",
) -> None:
    """
    Unparse RData object to a file.

    Args:
        path: File path to be created.
        r_data: RData object.
        file_format: File format.
        file_type: File type.
        compression: Compression.
    """
    open_with_compression: Callable[
        [os.PathLike[Any] | str, Literal["wb"]],
        WriteableBinaryFile,
    ]
    if compression is None:
        open_with_compression = open
    elif compression == "bzip2":
        from bz2 import open as open_with_compression  # noqa: PLC0415
    elif compression == "gzip":
        from gzip import open as open_with_compression  # noqa: PLC0415
    elif compression == "xz":
        from lzma import open as open_with_compression  # noqa: PLC0415
    else:
        msg = f"Unknown compression: {compression}"
        raise ValueError(msg)

    with open_with_compression(path, "wb") as f:
        unparse_fileobj(
            f,
            r_data,
            file_format=file_format,
            file_type=file_type,
        )


def unparse_fileobj(
    fileobj: WriteableBinaryFile,
    r_data: RData,
    *,
    file_format: FileFormat = "xdr",
    file_type: FileType = "rds",
) -> None:
    """
    Unparse RData object to a file object.

    Args:
        fileobj: File object.
        r_data: RData object.
        file_format: File format.
        file_type: File type.
    """
    unparser_class: type[UnparserXDR | UnparserASCII]

    if file_format == "ascii":
        unparser_class = UnparserASCII

        rda_magic = "RDA"
    elif file_format == "xdr":
        unparser_class = UnparserXDR

        rda_magic = "RDX"
    else:
        msg = f"Unknown file format: {file_format}"
        raise ValueError(msg)

    # Check that RData object for rda file is of correct kind
    if file_type == "rda":
        r_object = r_data.object
        if not (
            r_object.info.type is RObjectType.LIST
            and r_object.tag is not None
            and r_object.tag.info.type is RObjectType.SYM
        ):
            msg = "r_data object must be dictionary-like for rda file"
            raise ValueError(msg)

    # Write rda-specific magic
    if file_type == "rda":
        fileobj.write(f"{rda_magic}{r_data.versions.format}\n".encode("ascii"))

    unparser = unparser_class(fileobj)
    unparser.unparse_r_data(r_data)


def unparse_data(
    r_data: RData,
    *,
    file_format: FileFormat = "xdr",
    file_type: FileType = "rds",
) -> bytes:
    """
    Unparse RData object to a bytestring.

    Args:
        r_data: RData object.
        file_format: File format.
        file_type: File type.

    Returns:
        Bytestring of data.
    """
    fd = io.BytesIO()
    unparse_fileobj(fd, r_data, file_format=file_format, file_type=file_type)
    return fd.getvalue()
