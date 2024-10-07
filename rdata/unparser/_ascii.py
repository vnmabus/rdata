"""Unparser for files in ASCII format."""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

import numpy as np

from rdata.missing import is_na

from ._unparser import Unparser

if TYPE_CHECKING:
    import io
    from typing import Any, Final

    import numpy.typing as npt


def build_byte_to_str_map() -> tuple[str, ...]:
    """Build byte-to-string mapping for string conversion."""

    def escape(b: bytes) -> str:
        r"""Escape string, e.g., b'\n' -> r'\n'."""
        return b.decode("latin1").encode("unicode_escape").decode("ascii")

    # Fill mapping with octal codes
    byte_to_str = [rf"\{byte:03o}" for byte in range(256)]

    # Update mapping for ascii characters
    for byte in string.printable.encode("ascii"):
        # Note: indexing bytestring yields ints
        assert isinstance(byte, int)
        byte_to_str[byte] = escape(bytes([byte]))

    # Update mapping for special characters
    byte_to_str[b'"'[0]] = r'\"'
    byte_to_str[b"'"[0]] = r"\'"
    byte_to_str[b"?"[0]] = r"\?"
    byte_to_str[b" "[0]] = r"\040"
    byte_to_str[b"\v"[0]] = r"\v"
    byte_to_str[b"\f"[0]] = r"\f"

    return tuple(byte_to_str)


BYTE_TO_STR: Final = build_byte_to_str_map()


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

    def _unparse_array_values_raw(self,
        array: npt.NDArray[np.int32 | np.float64 | np.complex128],
    ) -> None:
        # Convert complex to pairs of floats
        if np.issubdtype(array.dtype, np.complexfloating):
            assert array.dtype == np.complex128
            array = array.view(np.float64)

        # Unparse data
        for value in array:
            if np.issubdtype(array.dtype, np.integer):
                line = "NA" if is_na(value) else str(value)

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

    def _unparse_string_characters(self, value: bytes) -> None:
        # Ideally we could do here the reverse of parsing,
        # i.e., output = value.decode('latin1').encode('unicode_escape').decode('ascii')
        # This would produce byte representation in hex such as '\xc3\xa4',
        # but we need to have the equivalent octal presentation '\303\244'.
        # In addition, some ascii characters need to be escaped.

        # Convert string byte-by-byte
        output = "".join(BYTE_TO_STR[byte] for byte in value)

        self._add_line(output)
