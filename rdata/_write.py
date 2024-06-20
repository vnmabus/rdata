"""Functions to perform conversion and unparsing in one step."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .conversion import build_r_data, convert_to_r_object, convert_to_r_object_for_rda
from .conversion.to_r import DEFAULT_FORMAT_VERSION
from .unparser import unparse_file

if TYPE_CHECKING:
    import os
    from typing import Any

    from .conversion.to_r import Encoding
    from .unparser import Compression, FileFormat


def write_rds(
    path: os.PathLike[Any] | str,
    data: Any,  # noqa: ANN401
    *,
    file_format: FileFormat = "xdr",
    compression: Compression = "gzip",
    encoding: Encoding = "utf-8",
    format_version: int = DEFAULT_FORMAT_VERSION,
) -> None:
    """
    Write an RDS file.

    This is a convenience function that wraps
    :func:`rdata.conversion.convert_to_r_object`,
    :func:`rdata.conversion.build_r_data`,
    and :func:`rdata.unparser.unparse_file`,
    as it is the common use case.

    Args:
        path: File path to be written.
        data: Python data object.
        file_format: File format.
        compression: Compression.
        encoding: Encoding to be used for strings within data.
        format_version: File format version.

    See Also:
        :func:`write_rda`: Similar function that writes an RDA or RDATA file.

    Examples:
        Write a Python object to an RDS file.

        >>> import rdata
        >>>
        >>> data = ["hello", 1, 2.2, 3.3+4.4j]
        >>> rdata.write_rds("test.rds", data)
    """
    r_object = convert_to_r_object(
        data,
        encoding=encoding,
    )
    r_data = build_r_data(
        r_object,
        encoding=encoding,
        format_version=format_version,
    )
    unparse_file(
        path,
        r_data,
        file_type="rds",
        file_format=file_format,
        compression=compression,
    )


def write_rda(
    path: os.PathLike[Any] | str,
    data: dict[str, Any],
    *,
    file_format: FileFormat = "xdr",
    compression: Compression = "gzip",
    encoding: Encoding = "utf-8",
    format_version: int = DEFAULT_FORMAT_VERSION,
) -> None:
    """
    Write an RDA or RDATA file.

    This is a convenience function that wraps
    :func:`rdata.conversion.convert_to_r_object_for_rda`,
    :func:`rdata.conversion.build_r_data`,
    and :func:`rdata.unparser.unparse_file`,
    as it is the common use case.

    Args:
        path: File path to be written.
        data: Python dictionary with data and variable names.
        file_format: File format.
        compression: Compression.
        encoding: Encoding to be used for strings within data.
        format_version: File format version.

    See Also:
        :func:`write_rds`: Similar function that writes an RDS file.

    Examples:
        Write a Python dictionary to an RDA file.

        >>> import rdata
        >>>
        >>> data = {"name": "hello", "values": [1, 2.2, 3.3+4.4j]}
        >>> rdata.write_rda("test.rda", data)
    """
    r_object = convert_to_r_object_for_rda(
        data,
        encoding=encoding,
    )
    r_data = build_r_data(
        r_object,
        encoding=encoding,
        format_version=format_version,
    )
    unparse_file(
        path,
        r_data,
        file_type="rda",
        file_format=file_format,
        compression=compression,
    )
