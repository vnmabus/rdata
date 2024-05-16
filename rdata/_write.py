"""Functions to perform conversion and unparsing in one step."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .conversion import convert_to_r_data
from .unparser import write
from .parser import RVersions

if TYPE_CHECKING:
    import os
    from typing import Any


def write_rdata(
    path: os.PathLike[Any] | str,
    data: Any,  # noqa: ANN401
    *,
    file_format: str = "xdr",
    rds: bool = True,
    compression: str = "gzip",
    encoding: str = "UTF-8",
    versions: tuple[int, int, int] | None = None,
) -> None:
    r_data = convert_to_r_data(
        data,
        rds=rds,
        encoding=encoding,
        versions=None if versions is None else RVersions(*versions),
    )
    write(
        path,
        r_data,
        rds=rds,
        file_format=file_format,
        compression=compression,
    )


def write_rds(
    path: os.PathLike[Any] | str,
    data: Any,  # noqa: ANN401
    *,
    file_format: str = "xdr",
    compression: str = "gzip",
    encoding: str = "UTF-8",
    versions: tuple[int, int, int] | None = None,
) -> None:
    """
    Write an RDS file.

    This is a convenience function that wraps
    :func:`rdata.conversion.convert_to_r_data` and :func:`rdata.io.write`,
    as it is the common use case.

    Args:
        path: File path to be written.
        data: Python data object.
        file_format: File format (xdr or ascii).
        compression: Compression (gzip, bzip2, xz, or none).
        encoding: Encoding to be used for strings within data.
        versions: Tuple of file version information
            (format_version, r_version, minimum_r_version).

    See Also:
        :func:`write_rda`: Similar function that writes an RDA or RDATA file.

    Examples:
        Write a Python object to an RDS file.

        >>> import rdata
        >>>
        >>> data = ["hello", 1, 2.2, 3.3+4.4j]
        >>> rdata.write_rds("test.rds", data)
    """
    return write_rdata(
        path=path,
        data=data,
        file_format=file_format,
        rds=True,
        compression=compression,
        encoding=encoding,
        versions=versions,
    )


def write_rda(
    path: os.PathLike[Any] | str,
    data: dict[str, Any],
    *,
    file_format: str = "xdr",
    compression: str = "gzip",
    encoding: str = "UTF-8",
    versions: tuple[int, int, int] | None = None,
) -> None:
    """
    Write an RDA or RDATA file.

    This is a convenience function that wraps
    :func:`rdata.conversion.convert_to_r_data` and :func:`rdata.io.write`,
    as it is the common use case.

    Args:
        path: File path to be written.
        data: Python dictionary with data and variable names.
        file_format: File format (xdr or ascii).
        compression: Compression (gzip, bzip2, xz, or none).
        encoding: Encoding to be used for strings within data.
        versions: Tuple of file version information
            (format_version, r_version, minimum_r_version).

    See Also:
        :func:`write_rds`: Similar function that writes an RDS file.

    Examples:
        Write a Python dictionary to an RDA file.

        >>> import rdata
        >>>
        >>> data = {"name": "hello", "values": [1, 2.2, 3.3+4.4j]}
        >>> rdata.write_rda("test.rda", data)
    """
    return write_rdata(
        path=path,
        data=data,
        file_format=file_format,
        rds=False,
        compression=compression,
        encoding=encoding,
        versions=versions,
    )

