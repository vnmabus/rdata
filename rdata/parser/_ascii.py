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
    R_INT_NA,
)

class ParserASCII(Parser):
    """Parser for data in ASCII format."""

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
        self.file = io.TextIOWrapper(io.BytesIO(data), encoding='ascii')

    def _readline(self) -> str:
        """Read a line without trailing \\n"""
        return self.file.readline()[:-1]

    def _parse_array_values(
            self,
            dtype: np.dtype,
            length: int,
    ) -> npt.NDArray[Any]:

        array = np.empty(length, dtype=dtype)

        for i in range(length):
            line = self._readline()

            if np.issubdtype(dtype, np.integer):
                if line == 'NA':
                    value = R_INT_NA
                else:
                    value = int(line)

            elif np.issubdtype(dtype, np.floating):
                value = float(line)

            elif np.issubdtype(dtype, np.complexfloating):
                line2 = self._readline()
                value = complex(float(line), float(line2))

            else:
                raise ValueError(f'unknown dtype: {dtype}')

            array[i] = value

        return array

    def parse_string(self, length: int) -> bytes:
        return self._readline().encode('ascii').decode('unicode_escape').encode('latin1')

    def parse_all(self) -> RData:
        rdata = super().parse_all()
        # Check that there is no more data in the file
        assert self.file.read(1) == ''
        return rdata
