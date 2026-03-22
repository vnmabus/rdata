from __future__ import annotations

import io
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ._parser import AltRepConstructorMap, Parser

_BYTEORDER_MARKER_SIZE = 4
_SUPPORTED_FORMAT_VERSIONS = {2, 3}


class ParserBinary(Parser):
    """Parser for data in native binary format."""

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
        self.byteorder = self._detect_byteorder(data)

    @staticmethod
    def _detect_byteorder(data: memoryview) -> Literal["<", ">"]:
        if len(data) < _BYTEORDER_MARKER_SIZE:
            msg = "Unknown binary endianness"
            raise NotImplementedError(msg)

        first_int = bytes(data[:_BYTEORDER_MARKER_SIZE])
        little = int.from_bytes(first_int, byteorder="little", signed=True)
        big = int.from_bytes(first_int, byteorder="big", signed=True)

        little_is_valid = little in _SUPPORTED_FORMAT_VERSIONS
        big_is_valid = big in _SUPPORTED_FORMAT_VERSIONS

        if little_is_valid and not big_is_valid:
            return "<"
        if big_is_valid and not little_is_valid:
            return ">"

        msg = "Unknown binary endianness"
        raise NotImplementedError(msg)

    def _parse_array_values(
        self,
        dtype: npt.DTypeLike,
        length: int,
    ) -> npt.NDArray[Any]:
        dtype = np.dtype(dtype)
        buffer = self.file.read(length * dtype.itemsize)
        # `frombuffer` on bytes creates a read-only view.
        # `mask_na_values` mutates integer arrays in place, so we need a copy.
        return np.frombuffer(
            buffer,
            dtype=dtype.newbyteorder(self.byteorder),
        ).astype(dtype, copy=True)

    def parse_string(self, length: int) -> bytes:
        return self.file.read(length)

    def check_complete(self) -> None:
        assert self.file.read(1) == b""
