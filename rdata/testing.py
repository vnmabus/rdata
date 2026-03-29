"""Utilities for testing with R files."""

from __future__ import annotations

import subprocess
import tempfile
from typing import Any, Protocol

R_CODE_PREFIX = """::: """


class HasDoc(Protocol):
    """Python object having a docstring."""
    __doc__: str | None


def get_data_source(
    function_or_class: HasDoc,
    *,
    prefix: str = R_CODE_PREFIX,
) -> str:
    """
    Get the part of the docstring containing the data source.

    Args:
        function_or_class: Function or class whose docstring contains the data
            source.
        prefix: Prefix used to mark lines that contain the data source.

    Returns:
        The data source.

    """
    doc = function_or_class.__doc__
    if doc is None:
        return ""

    source = ""

    for line in doc.splitlines(keepends=True):
        stripped_line = line.lstrip()
        if stripped_line.startswith(prefix):
            source += stripped_line.removeprefix(prefix)

    return source


def execute_r_data_source(
    function_or_class: HasDoc,
    *,
    prefix: str = R_CODE_PREFIX,
    append: str = "",
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """
    Execute R data source.

    Args:
        function_or_class: Function or class whose docstring contains the data
            source.
        prefix: Prefix used to mark lines that contain the data source.
        append: Code appended to the end (for example, to save the objects).
        kwargs: Each keyword parameter corresponds to a variable to set at the
            beginning of the code.

    """
    source = get_data_source(
        function_or_class,
        prefix=prefix,
    )
    if not source:
        return

    inits = ""
    for key, value in kwargs.items():
        inits += f"{key} <- {value!r}\n"

    source = inits + source + append

    with tempfile.NamedTemporaryFile("w", delete_on_close=False) as file:
        file.write(source)
        file.flush()
        subprocess.check_call(  # noqa: S603
            ["Rscript", file.name],  # noqa: S607
        )
