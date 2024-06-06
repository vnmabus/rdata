"""Utilities for unparsing a rdata file."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from rdata.parser import (
    RData,
    RObjectType,
)

if TYPE_CHECKING:
    import os
    from typing import IO, Any, Literal

    from rdata.parser import RData

    from ._ascii import UnparserASCII
    from ._xdr import UnparserXDR

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
    Unparser: type[UnparserXDR | UnparserASCII]  # noqa: N806

    if file_format == "ascii":
        from ._ascii import UnparserASCII as Unparser
        rda_magic = "RDA"
    elif file_format == "xdr":
        from ._xdr import UnparserXDR as Unparser
        rda_magic = "RDX"
    else:
        msg = f"Unknown file format: {file_format}"
        raise ValueError(msg)

    # Check that RData object for rda file is of correct kind
    if file_type == "rda":
        r_object = r_data.object
        if not (r_object.info.type is RObjectType.LIST
                and r_object.tag is not None
                and r_object.tag.info.type is RObjectType.SYM):
            msg = "r_data object must be dictionary-like for rda file"
            raise ValueError(msg)

    # Write rda-specific magic
    if file_type == "rda":
        fileobj.write(f"{rda_magic}{r_data.versions.format}\n".encode("ascii"))

    unparser = Unparser(fileobj)  # type: ignore [arg-type]
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
