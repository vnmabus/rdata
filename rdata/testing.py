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
    """Get the part of the docstring containing the data source."""
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
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Execute R data source."""
    source = get_data_source(
        function_or_class,
        prefix=prefix,
    )
    if not source:
        return

    inits = ""
    for key, value in kwargs.items():
        inits += f"{key} <- {value!r}\n"

    source = inits + source

    with tempfile.NamedTemporaryFile("w") as file:
        file.write(source)
        file.flush()
        subprocess.check_call(  # noqa: S603
            ["Rscript", file.name],  # noqa: S607
        )
