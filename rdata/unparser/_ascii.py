"""Unparser for files in ASCII format."""

from __future__ import annotations

import string
from typing import TYPE_CHECKING, Any

import numpy as np

from rdata.missing import is_na

from ._unparser import Unparser

if TYPE_CHECKING:
    import io

    import numpy.typing as npt


class UnparserASCII(Unparser):
    """Unparser for files in ASCII format."""

    def __init__(
        self,
        file: io.BytesIO,
    ) -> None:
        """Unparser for files in ASCII format."""
        self.file = file

    def _add_line(self, line: str) -> None:
        r"""Write a line with trailing \n."""
        # Write in binary mode to be compatible with
        # compression (e.g. when file = gzip.open())
        self.file.write(f"{line}\n".encode("ascii"))

    def unparse_magic(self) -> None:
        """Unparse magic bits."""
        self._add_line("A")

    def _unparse_array_values(self, array: npt.NDArray[Any]) -> None:  # noqa: C901
        # Convert boolean to int
        if np.issubdtype(array.dtype, np.bool_):
            array = array.astype(np.int32)

        # Convert complex to pairs of floats
        if np.issubdtype(array.dtype, np.complexfloating):
            assert array.dtype == np.complex128
            array = array.view(np.float64)

        # Unparse data
        for value in array:
            if np.issubdtype(array.dtype, np.integer):
                line = "NA" if value is None or np.ma.is_masked(value) else str(value)  # type: ignore [no-untyped-call]

            elif np.issubdtype(array.dtype, np.floating):
                if is_na(value):
                    line = "NA"
                elif np.isnan(value):
                    line = "NaN"
                elif value == np.inf:
                    line = "Inf"
                elif value == -np.inf:
                    line = "-Inf"
                else:
                    line = str(value)
                    if line.endswith(".0"):
                        line = line[:-2]

            else:
                msg = f"Unknown dtype: {array.dtype}"
                raise ValueError(msg)

            self._add_line(line)

    def unparse_string(self, value: bytes) -> None:
        """Unparse a string."""
        self.unparse_int(len(value))

        # Ideally we could do here the reverse of parsing,
        # i.e., value = value.decode('latin1').encode('unicode_escape').decode('ascii')
        # This would produce byte representation in hex such as '\xc3\xa4',
        # but we need to have the equivalent octal presentation '\303\244'.
        # So, we do somewhat manual conversion instead:
        s = "".join(chr(byte) if chr(byte) in string.printable else rf"\{byte:03o}"
                    for byte in value)

        self._add_line(s)
