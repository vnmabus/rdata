"""Writer for files in ASCII format."""

from __future__ import annotations

import string
from typing import TYPE_CHECKING, Any

import numpy as np

from ._unparser import Writer

if TYPE_CHECKING:
    import io

    import numpy.typing as npt


class WriterASCII(Writer):
    """Writer for files in ASCII format."""

    def __init__(
        self,
        file: io.BytesIO,
    ) -> None:
        """Writer for files in ASCII format."""
        self.file = file

    def _writeline(self, line: str) -> None:
        r"""Write a line with trailing \n."""
        # Write in binary mode to be compatible with
        # compression (e.g. when file = gzip.open())
        self.file.write(f"{line}\n".encode("ascii"))

    def write_magic(self, rda_version: int | None) -> None:
        """Write magic bits."""
        if rda_version is not None:
            self._writeline(f"RDA{rda_version}")
        self._writeline("A")

    def _write_array_values(self, array: npt.NDArray[Any]) -> None:
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
                line = "NA" if value is None or np.ma.is_masked(value) else str(value)  # type: ignore [no-untyped-call]

            elif np.issubdtype(array.dtype, np.floating):
                line = str(value)
                if line.endswith(".0"):
                    line = line[:-2]

            else:
                msg = f"Unknown dtype: {array.dtype}"
                raise ValueError(msg)

            self._writeline(line)

    def write_string(self, value: bytes) -> None:
        """Write a string."""
        self.write_int(len(value))

        # Ideally we could do here the reverse of parsing,
        # i.e., value = value.decode('latin1').encode('unicode_escape').decode('ascii')
        # This would produce byte representation in hex such as '\xc3\xa4',
        # but we need to have the equivalent octal presentation '\303\244'.
        # So, we do somewhat manual conversion instead:
        s = "".join(chr(byte) if chr(byte) in string.printable else rf"\{byte:03o}"
                    for byte in value)

        self._writeline(s)
