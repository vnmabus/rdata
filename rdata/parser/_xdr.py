from __future__ import annotations

import io
from typing import Any

import numpy as np
import numpy.typing as npt

from ._parser import AltRepConstructorMap, Parser


class ParserXDR(Parser):
    """Parser for data in XDR format."""

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
        self.file = io.BytesIO(data)

    def _parse_array_values(
            self,
            dtype: npt.DTypeLike,
            length: int,
    ) -> npt.NDArray[Any]:
        dtype = np.dtype(dtype)
        buffer = self.file.read(length * dtype.itemsize)
        # Read in big-endian order and convert to native byte order
        return np.frombuffer(
            buffer,
            dtype=dtype.newbyteorder(">"),
        ).astype(dtype, copy=False)

    def parse_string(self, length: int) -> bytes:
        return self.file.read(length)

    def check_complete(self) -> None:
        assert self.file.read(1) == b""
