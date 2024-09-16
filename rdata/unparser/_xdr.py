"""Unparser for files in XDR format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rdata.missing import R_INT_NA

from ._unparser import Unparser

if TYPE_CHECKING:
    import io

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

    def _unparse_array_values(self, array: npt.NDArray[Any]) -> None:
        # Convert boolean to int
        if np.issubdtype(array.dtype, np.bool_):
            array = array.astype(np.int32)

        # Flatten masked values and convert int arrays to int32
        if np.issubdtype(array.dtype, np.integer):
            if np.ma.is_masked(array):  # type: ignore [no-untyped-call]
                mask = np.ma.getmask(array)  # type: ignore [no-untyped-call]
                array = np.ma.getdata(array).copy()  # type: ignore [no-untyped-call]
                array[mask] = R_INT_NA
            info = np.iinfo(np.int32)
            if not all(info.min <= val <= info.max for val in array):
                msg = "Integer array not castable to int32"
                raise ValueError(msg)
            array = array.astype(np.int32)

        # Convert to big endian if needed
        array = array.astype(array.dtype.newbyteorder(">"))

        # Create a contiguous data buffer if not already
        # 1D array should be both C and F contiguous
        assert array.flags["C_CONTIGUOUS"] == array.flags["F_CONTIGUOUS"]
        data = array.data if array.flags["C_CONTIGUOUS"] else array.tobytes()
        self.file.write(data)

    def unparse_string(self, value: bytes) -> None:
        """Unparse a string."""
        if value is None:
            self.unparse_int(-1)
        else:
            self.unparse_int(len(value))
            self.file.write(value)
