"""Unparser for files in XDR format."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._unparser import Unparser

if TYPE_CHECKING:
    import io

    import numpy as np
    import numpy.typing as npt


class UnparserXDR(Unparser):
    """Unparser for files in XDR format."""

    def __init__(
        self,
        file: io.BytesIO,
    ) -> None:
        """Unparser for files in XDR format."""
        self.file = file

    def unparse_magic(self) -> None:
        """Unparse magic bits."""
        self.file.write(b"X\n")

    def _unparse_array_values_raw(self,
        array: npt.NDArray[np.int32 | np.float64 | np.complex128],
    ) -> None:
        # Convert to big endian if needed
        array = array.astype(array.dtype.newbyteorder(">"))

        # Create a contiguous data buffer if not already
        # 1D array should be both C and F contiguous
        assert array.flags["C_CONTIGUOUS"] == array.flags["F_CONTIGUOUS"]
        data = array.data if array.flags["C_CONTIGUOUS"] else array.tobytes()
        self.file.write(data)

    def _unparse_string_characters(self, value: bytes) -> None:
        self.file.write(value)
