from __future__ import annotations

from typing import (
    Any,
    BinaryIO,
)

import numpy as np
import numpy.typing as npt

from ._parser import (
    AltRepConstructorMap,
    DEFAULT_ALTREP_MAP,
    Parser,
    RData,
)



R_INT_NA = -2**31  # noqa: WPS432
"""Value used to represent a missing integer in R."""


class ParserXDR(Parser):
    """Parser used when the integers and doubles are in XDR format."""

    def __init__(
        self,
        file: BinaryIO,
        *,
        expand_altrep: bool = True,
        altrep_constructor_dict: AltRepConstructorMap = DEFAULT_ALTREP_MAP,
    ) -> None:
        super().__init__(
            expand_altrep=expand_altrep,
            altrep_constructor_dict=altrep_constructor_dict,
        )
        self.file = file

    def _parse_array(
            self,
            dtype: np.dtype,
    ) -> npt.NDArray[Any]:  # noqa: D102
        length = self.parse_int()
        return self._parse_array_values(dtype, length)

    def _parse_array_values(
            self,
            dtype: np.dtype,
            length: int,
    ) -> npt.NDArray[Any]:  # noqa: D102
        dtype = np.dtype(dtype)
        buffer = self.file.read(length * dtype.itemsize)
        # Read in big-endian order and convert to native byte order
        return np.frombuffer(buffer, dtype=dtype.newbyteorder('>')).astype(dtype, copy=False)

    def parse_int(self) -> int:  # noqa: D102
        return int(self._parse_array_values(np.int32, 1)[0])

    def parse_double(self) -> float:  # noqa: D102
        return float(self._parse_array_values(np.float64, 1)[0])

    def parse_complex(self) -> complex:  # noqa: D102
        return complex(self._parse_array_values(np.complex128, 1)[0])

    def parse_nullable_int_array(
        self,
        fill_value: int = R_INT_NA,
    ) -> npt.NDArray[np.int32] | np.ma.MaskedArray[Any, Any]:  # noqa: D102

        data = self._parse_array(np.int32)
        mask = data == R_INT_NA
        data[mask] = fill_value

        if np.any(mask):
            return np.ma.array(  # type: ignore
                data=data,
                mask=mask,
                fill_value=fill_value,
            )

        return data

    def parse_double_array(self) -> npt.NDArray[np.float64]:  # noqa: D102
        return self._parse_array(np.float64)

    def parse_complex_array(self) -> npt.NDArray[np.complex128]:  # noqa: D102
        return self._parse_array(np.complex128)

    def parse_string(self, length: int) -> bytes:  # noqa: D102
        return self.file.read(length)

    def parse_all(self) -> RData:
        rdata = super().parse_all()
        # Check that there is no more data in the file
        assert self.file.read(1) == b''
        return rdata
