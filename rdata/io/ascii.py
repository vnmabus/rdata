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

    def write_nullable_bool(self, value):
        if value is None or np.ma.is_masked(value):
            self._writeline("NA")
        else:
            self.write_int(int(value))

    def write_nullable_int(self, value):
        if value is None or np.ma.is_masked(value):
            self._writeline("NA")
        else:
            self._writeline(value)

    def write_double(self, value):
        s = str(value)
        if s.endswith(".0"):
            s = s[:-2]
        self._writeline(s)

    def write_string(self, value: bytes):
        self.write_int(len(value))

        # This line would produce byte representation in hex such as '\xc3\xa4':
        # value = value.decode('latin1').encode('unicode_escape').decode('ascii')
        # but we need to have the equivalent octal presentation '\303\244'.
        # So, we use a somewhat manual conversion instead:
        value = "".join(chr(byte) if chr(byte) in string.printable else rf"\{byte:03o}" for byte in value)

        self._writeline(value)
