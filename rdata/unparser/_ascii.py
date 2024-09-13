"""Unparser for files in ASCII format."""

from __future__ import annotations

import string
from typing import TYPE_CHECKING, Any

import numpy as np

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

    def _unparse_array_values(self, array: npt.NDArray[Any]) -> None:
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
                if np.isnan(value):
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

    def _unparse_string_characters(self, value: bytes) -> None:
        # Ideally we could do here the reverse of parsing,
        # i.e., output = value.decode('latin1').encode('unicode_escape').decode('ascii')
        # This would produce byte representation in hex such as '\xc3\xa4',
        # but we need to have the equivalent octal presentation '\303\244'.
        # So, we need to do somewhat manual conversion instead.

        # List of ascii characters that are written directly;
        # this is all printable ascii except
        # - ' '  that Python writes as ' ',    but R as '\040'
        # - '\v' that Python writes as '\x0b', but R as '\v'
        # - '\f' that Python writes as '\x0c', but R as '\f'
        write_raw = string.printable.replace(" ", "")\
                                    .replace("\v", "")\
                                    .replace("\f", "")

        def escape(b: bytes) -> str:
            r"""Escape string, e.g., b'\n' -> r'\\n'."""
            return b.decode("latin1").encode("unicode_escape").decode("ascii")

        # Go though the string byte-by-byte as we need to
        # convert every non-ascii character separately
        output = ""
        ascii_buffer = b""
        for byte in value:
            if chr(byte) in write_raw:
                # Collect ascii characters to substring buffer
                ascii_buffer += bytes([byte])
            else:
                # Encountered a non-ascii character!
                # Escape and add the ascii buffer
                output += escape(ascii_buffer)
                ascii_buffer = b""
                # Add '\v' or '\f' or non-ascii character in octal presentation
                if chr(byte) == "\v":
                    output += r"\v"
                elif chr(byte) == "\f":
                    output += r"\f"
                else:
                    output += rf"\{byte:03o}"
        # Escape and add the remaining ascii buffer
        output += escape(ascii_buffer)

        # Escape some more characters like R does
        output = output.replace('"', r'\"').replace("'", r"\'").replace("?", r"\?")

        self._add_line(output)
