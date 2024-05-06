"""Utilities for writing a rdata file."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import os

    from rdata.parser import RData

    from .ascii import WriterASCII
    from .xdr import WriterXDR


def write(
        path: os.PathLike[Any] | str,
        r_data: RData,
        *,
        file_format: str = "xdr",
        rds: bool = True,
        compression: str = "gzip",
) -> None:
    """
    Write RData object to a file.

    Parameters
    ----------
    path:
        File path to be written
    r_data:
        RData object
    file_format:
        File format (ascii or xdr)
    rds:
        Whether to write RDS or RDA file
    compression:
        Compression (gzip, bzip2, xz, or none)
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
        write_file(f, r_data, file_format=file_format, rds=rds)


def write_file(
        fileobj: IO[Any],
        r_data: RData,
        *,
        file_format: str = "xdr",
        rds: bool = True,
) -> None:
    """
    Write RData object to a file object.

    Parameters
    ----------
    fileobj:
        File object
    r_data:
        RData object
    file_format:
        File format (ascii or xdr)
    """
    Writer: type[WriterXDR | WriterASCII]  # noqa: N806

    if file_format == "ascii":
        from .ascii import WriterASCII as Writer
    elif file_format == "xdr":
        from .xdr import WriterXDR as Writer
    else:
        msg = f"Unknown file format: {file_format}"
        raise ValueError(msg)

    w = Writer(fileobj)  # type: ignore [arg-type]
    w.write_r_data(r_data, rds=rds)
