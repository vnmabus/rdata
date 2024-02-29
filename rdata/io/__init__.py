from __future__ import annotations

import os
from typing import Any

from rdata.parser import RData


def write(
        path: os.PathLike[Any] | str,
        r_data: RData,
        *,
        format: str = "xdr",
        rds: bool = False,
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
    format:
        File format (ascii or xdr)
    compression:
        Compression (gzip, bzip2, xz, or none)
    """
    if format == "ascii":
        mode = "w"
    elif format == "xdr":
        mode = "wb"
    else:
        raise ValueError(f"Unknown format: {format}")

    if compression == "gzip":
        from gzip import open
    elif compression == "bzip2":
        from bz2 import open
    elif compression == "xz":
        from lzma import open
    elif compression == "none":
        import builtins
        open = builtins.open
    else:
        msg = f"Unknown compression: {compression}"
        if compression is None:
            msg += ". Use 'none' for no compression."
        raise ValueError(msg)

    with open(path, mode) as f:
        write_file(f, r_data, format=format, rds=rds)


def write_file(
        fileobj,
        r_data: RData,
        *,
        format: str = "xdr",
        rds: bool = False,
) -> None:
    """
    Write RData object to a file object.

    Parameters
    ----------
    fileobj:
        File object
    r_data:
        RData object
    format:
        File format (ascii or xdr)
    """
    if format == "ascii":
        from .ascii import WriterASCII as Writer
    elif format == "xdr":
        from .xdr import WriterXDR as Writer
    else:
        raise ValueError(f"Unknown format: {format}")

    w = Writer(fileobj)
    w.write_r_data(r_data, rds=rds)
