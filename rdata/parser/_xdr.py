from __future__ import annotations

import io
import numpy as np
import numpy.typing as npt

from typing import (
    Any,
)

from ._parser import (
    AltRepConstructorMap,
    DEFAULT_ALTREP_MAP,
    Parser,
    RData,
)


class ParserXDR(Parser):
    """Parser for data in XDR format."""

    def __init__(
        self,
        data: memoryview,
        *,
        expand_altrep: bool = True,
        altrep_constructor_dict: AltRepConstructorMap = DEFAULT_ALTREP_MAP,
    ) -> None:
        super().__init__(
            expand_altrep=expand_altrep,
            altrep_constructor_dict=altrep_constructor_dict,
        )
        self.file = io.BytesIO(data)

    def _parse_array_values(
            self,
            dtype: np.dtype,
            length: int,
    ) -> npt.NDArray[Any]:
        dtype = np.dtype(dtype)
        buffer = self.file.read(length * dtype.itemsize)
        # Read in big-endian order and convert to native byte order
        return np.frombuffer(buffer, dtype=dtype.newbyteorder('>')).astype(dtype, copy=False)

    def parse_string(self, length: int) -> bytes:
        return self.file.read(length)

    def parse_all(self) -> RData:
        rdata = super().parse_all()
        # Check that there is no more data in the file
        assert self.file.read(1) == b''
        return rdata
