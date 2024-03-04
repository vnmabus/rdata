"""Functions to perform parsing and conversion in one step."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .conversion._conversion import DEFAULT_CLASS_MAP, ConstructorDict, convert
from .parser._parser import (
    DEFAULT_ALTREP_MAP,
    AcceptableFile,
    AltRepConstructorMap,
    Traversable,
    parse_file,
)

if TYPE_CHECKING:
    import os
    from collections.abc import MutableMapping


def read_rdata(  # noqa: PLR0913
    file_or_path: AcceptableFile | os.PathLike[Any] | Traversable | str,
    *,
    expand_altrep: bool = True,
    altrep_constructor_dict: AltRepConstructorMap = DEFAULT_ALTREP_MAP,
    extension: str | None = None,
    constructor_dict: ConstructorDict = DEFAULT_CLASS_MAP,
    default_encoding: str | None = None,
    force_default_encoding: bool = False,
    global_environment: MutableMapping[str, Any] | None = None,
    base_environment: MutableMapping[str, Any] | None = None,
) -> Any:  # noqa: ANN401
    parsed = parse_file(
        file_or_path=file_or_path,
        expand_altrep=expand_altrep,
        altrep_constructor_dict=altrep_constructor_dict,
        extension=extension,
    )

    return convert(
        parsed,
        constructor_dict=constructor_dict,
        default_encoding=default_encoding,
        force_default_encoding=force_default_encoding,
        global_environment=global_environment,
        base_environment=base_environment,
    )


def read_rds(  # noqa: PLR0913
    file_or_path: AcceptableFile | os.PathLike[Any] | Traversable | str,
    *,
    expand_altrep: bool = True,
    altrep_constructor_dict: AltRepConstructorMap = DEFAULT_ALTREP_MAP,
    constructor_dict: ConstructorDict = DEFAULT_CLASS_MAP,
    default_encoding: str | None = None,
    force_default_encoding: bool = False,
    global_environment: MutableMapping[str, Any] | None = None,
    base_environment: MutableMapping[str, Any] | None = None,
) -> Any:  # noqa: ANN401
    """
    Read an RDS file, containing an R object.

    This is a convenience function that wraps :func:`rdata.parser.parse_file`
    and :func:`rdata.parser.convert`, as it is the common use case.

    Args:
        file_or_path: File in the RDS format.
        expand_altrep: Whether to translate ALTREPs to normal objects.
        altrep_constructor_dict: Dictionary mapping each ALTREP to
            its constructor.
        constructor_dict: Dictionary mapping names of R classes to constructor
            functions with the following prototype:

            .. code-block :: python

                def constructor(obj, attrs):
                    ...

            This dictionary can be used to support custom R classes. By
            default, the dictionary used is
            :data:`~rdata.conversion._conversion.DEFAULT_CLASS_MAP`
            which has support for several common classes.
        default_encoding: Default encoding used for strings with unknown
            encoding. If `None`, the one stored in the file will be used, or
            ASCII as a fallback.
        force_default_encoding:
            Use the default encoding even if the strings specify other
            encoding.
        global_environment: Global environment to use. By default is an empty
            environment.
        base_environment: Base environment to use. By default is an empty
            environment.

    Returns:
        Contents of the file converted to a Python object.

    See Also:
        :func:`read_rda`: Similar function that parses a RDA or RDATA file.

    Examples:
        Parse one of the included examples, containing a dataframe

        >>> import rdata
        >>>
        >>> data = rdata.read_rds(
        ...              rdata.TESTDATA_PATH / "test_dataframe.rds"
        ... )
        >>> data
              class  value
            1     a      1
            2     b      2
            3     b      3

    """
    return read_rdata(
        file_or_path=file_or_path,
        expand_altrep=expand_altrep,
        altrep_constructor_dict=altrep_constructor_dict,
        extension=".rds",
        constructor_dict=constructor_dict,
        default_encoding=default_encoding,
        force_default_encoding=force_default_encoding,
        global_environment=global_environment,
        base_environment=base_environment,
    )


def read_rda(  # noqa: PLR0913
    file_or_path: AcceptableFile | os.PathLike[Any] | Traversable | str,
    *,
    expand_altrep: bool = True,
    altrep_constructor_dict: AltRepConstructorMap = DEFAULT_ALTREP_MAP,
    constructor_dict: ConstructorDict = DEFAULT_CLASS_MAP,
    default_encoding: str | None = None,
    force_default_encoding: bool = False,
    global_environment: MutableMapping[str, Any] | None = None,
    base_environment: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Read an RDA or RDATA file, containing an R object.

    This is a convenience function that wraps :func:`rdata.parser.parse_file`
    and :func:`rdata.parser.convert`, as it is the common use case.

    Args:
        file_or_path: File in the RDA format.
        expand_altrep: Whether to translate ALTREPs to normal objects.
        altrep_constructor_dict: Dictionary mapping each ALTREP to
            its constructor.
        constructor_dict: Dictionary mapping names of R classes to constructor
            functions with the following prototype:

            .. code-block :: python

                def constructor(obj, attrs):
                    ...

            This dictionary can be used to support custom R classes. By
            default, the dictionary used is
            :data:`~rdata.conversion._conversion.DEFAULT_CLASS_MAP`
            which has support for several common classes.
        default_encoding: Default encoding used for strings with unknown
            encoding. If `None`, the one stored in the file will be used, or
            ASCII as a fallback.
        force_default_encoding:
            Use the default encoding even if the strings specify other
            encoding.
        global_environment: Global environment to use. By default is an empty
            environment.
        base_environment: Base environment to use. By default is an empty
            environment.

    Returns:
        Contents of the file converted to a Python object.

    See Also:
        :func:`read_rds`: Similar function that parses a RDS file.

    Examples:
        Parse one of the included examples, containing a dataframe

        >>> import rdata
        >>>
        >>> data = rdata.read_rda(
        ...              rdata.TESTDATA_PATH / "test_dataframe.rda"
        ... )
        >>> data
        {'test_dataframe':   class  value
        1     a      1
        2     b      2
        3     b      3}

    """
    return read_rdata(  # type: ignore[no-any-return]
        file_or_path=file_or_path,
        expand_altrep=expand_altrep,
        altrep_constructor_dict=altrep_constructor_dict,
        extension=".rda",
        constructor_dict=constructor_dict,
        default_encoding=default_encoding,
        force_default_encoding=force_default_encoding,
        global_environment=global_environment,
        base_environment=base_environment,
    )
