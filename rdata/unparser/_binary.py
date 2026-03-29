"""Unparser for files in native binary format."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._unparser import Unparser, WriteableBinaryFile

if TYPE_CHECKING:
    import numpy.typing as npt


class UnparserBinary(Unparser):
    """Unparser for files in native binary format."""

    def __init__(
        self,
        file: WriteableBinaryFile,
    ) -> None:
        """Unparser for files in native binary format."""
        self.file = file

    def unparse_magic(self) -> None:
        """Unparse magic bits."""
        self.file.write(b"B\n")

    def _unparse_array_values_raw(
        self,
        array: npt.NDArray[np.int32 | np.float64 | np.complex128],
    ) -> None:
        # R native binary serialization stores data in host byte order.
        native_array = np.ascontiguousarray(
            array.astype(array.dtype.newbyteorder("="), copy=False),
        )
        self.file.write(native_array.tobytes())

    def _unparse_string_characters(self, value: bytes) -> None:
        self.file.write(value)
