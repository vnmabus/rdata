"""Writer for files in XDR format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rdata.parser import R_INT_NA

from .base import Writer

if TYPE_CHECKING:
    import io

    import numpy.typing as npt


class WriterXDR(Writer):
    """Writer for files in XDR format."""

    def __init__(
        self,
        file: io.BytesIO,
    ) -> None:
        """Writer for files in XDR format."""
        self.file = file

    def write_magic(self, rda_version: int) -> None:
        """Write magic bits."""
        if rda_version is not None:
            self.file.write(f"RDX{rda_version}\n".encode("ascii"))
        self.file.write(b"X\n")

    def _write_array_values(self, array: npt.NDArray[Any]) -> None:
        # Convert boolean to int
        if np.issubdtype(array.dtype, np.bool_):
            array = array.astype(np.int32)

        # Flatten masked values and convert int arrays to int32
        if np.issubdtype(array.dtype, np.integer):
            if np.ma.is_masked(array):
                mask = np.ma.getmask(array)
                array = np.ma.getdata(array).copy()
                array[mask] = R_INT_NA
            if not np.all([np.can_cast(val, np.int32) for val in array]):
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

    def write_string(self, value: bytes) -> None:
        """Write a string."""
        if value is None:
            self.write_int(-1)
        else:
            self.write_int(len(value))
            self.file.write(value)
