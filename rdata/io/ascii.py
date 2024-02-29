from __future__ import annotations

import string
from typing import TextIO

import numpy as np

from .base import Writer


class WriterASCII(Writer):
    """Writer for files in ASCII format."""

    def __init__(
        self,
        file: TextIO,
    ) -> None:
        self.file = file

    def _writeline(self, line) -> None:
        """Write a line with trailing \\n"""
        self.file.write(f"{line}\n")

    def write_magic(self, rda_version):
        if rda_version is not None:
            self._writeline(f"RDA{rda_version}")
        self._writeline("A")

    def _write_array_values(self, array):
        # Convert boolean to int
        if np.issubdtype(array.dtype, np.bool_):
            array = array.astype(np.int32)

        # Convert complex to pairs of floats
        if np.issubdtype(array.dtype, np.complexfloating):
            assert array.dtype == np.complex128
            array = array.view(np.float64)

        # Write data
        for value in array:
            if np.issubdtype(array.dtype, np.integer):
                if value is None or np.ma.is_masked(value):
                    line = "NA"
                else:
                    line = str(value)

            elif np.issubdtype(array.dtype, np.floating):
                line = str(value)
                if line.endswith(".0"):
                    line = line[:-2]

            else:
                msg = f"Unknown dtype: {array.dtype}"
                raise ValueError(msg)

            self._writeline(line)

    def write_string(self, value: bytes):
        self.write_int(len(value))

        # This line would produce byte representation in hex such as '\xc3\xa4':
        # value = value.decode('latin1').encode('unicode_escape').decode('ascii')
        # but we need to have the equivalent octal presentation '\303\244'.
        # So, we use a somewhat manual conversion instead:
        value = "".join(chr(byte) if chr(byte) in string.printable else rf"\{byte:03o}" for byte in value)

        self._writeline(value)
