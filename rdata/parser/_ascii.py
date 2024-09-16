from __future__ import annotations

import io
from typing import Any

import numpy as np
import numpy.typing as npt

from rdata.missing import R_FLOAT_NA, R_INT_NA

from ._parser import AltRepConstructorMap, Parser


def map_int_na(line: str) -> int:
    return R_INT_NA if line == "NA" else int(line)


def map_float_na(line: str) -> float:
    return R_FLOAT_NA if line == "NA" else float(line)


class ParserASCII(Parser):
    """Parser for data in ASCII format."""

    def __init__(
        self,
        data: memoryview,
        *,
        expand_altrep: bool,
        altrep_constructor_dict: AltRepConstructorMap,
    ) -> None:
        super().__init__(
            expand_altrep=expand_altrep,
            altrep_constructor_dict=altrep_constructor_dict,
        )
        self.file = io.TextIOWrapper(io.BytesIO(data), encoding="ascii")

    def _readline(self) -> str:
        r"""Read a line without trailing \n."""
        return self.file.readline()[:-1]

    def _parse_array_values(
            self,
            dtype: npt.DTypeLike,
            length: int,
    ) -> npt.NDArray[Any]:

        array = np.empty(length, dtype=dtype)
        value: int | float | complex

        for i in range(length):
            line = self._readline()

            if np.issubdtype(dtype, np.integer):
                value = map_int_na(line)

            elif np.issubdtype(dtype, np.floating):
                value = map_float_na(line)

            elif np.issubdtype(dtype, np.complexfloating):
                value1 = map_float_na(line)
                line2 = self._readline()
                value2 = map_float_na(line2)
                value = complex(value1, value2)

            else:
                msg = f"Unknown dtype: {dtype}"
                raise ValueError(msg)

            array[i] = value

        return array

    def parse_string(self, length: int) -> bytes:
        # Non-ascii characters in strings are written using octal byte codes,
        # for example, a string 'aä' (2 chars) in UTF-8 is written as an ascii
        # string r'a\303\244' (9 chars). We want to transform this to a byte
        # string b'a\303\244' (3 bytes) corresponding to the byte
        # representation of the original UTF-8 string.
        # Let's use this string as an example to go through the code below

        # Read the ascii string
        s = self._readline()
        # Now s = r'a\303\244' (9 chars)

        # Convert characters to bytes (all characters are ascii)
        b = s.encode("ascii")
        # Now b = br'a\303\244' (9 bytes)

        # There is a special 'unicode_escape' encoding that does
        # basically two things here:
        # 1) interpret e.g. br'\303' (4 bytes) as a single byte b'\303'
        # 2) decode so-transformed byte string to a string with latin1 encoding
        s = b.decode("unicode_escape")
        # Now s = 'aÃ¤' (3 chars)

        # We don't really want the latter latin1 decoding step done by
        # the previous line of code, so we undo it by encoding in latin1
        # back to bytes
        b = s.encode("latin1")
        # Now b = b'a\303\244' (3 bytes)

        # We return this byte representation here. Later in the code there
        # will be the decoding step from b'a\303\244' to 'aä',
        # that is, s = b.decode('utf8')
        assert len(b) == length
        return b

    def check_complete(self) -> None:
        assert self.file.read(1) == ""
