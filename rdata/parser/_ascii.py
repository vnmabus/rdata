from __future__ import annotations

import io
from typing import Any

import numpy as np
import numpy.typing as npt

from ._parser import AltRepConstructorMap, Parser, R_INT_NA


class ParserASCII(Parser):
    """Parser for data in ASCII format."""

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
        self.file = io.TextIOWrapper(io.BytesIO(data), encoding='ascii')

    def _readline(self) -> str:
        """Read a line without trailing \\n"""
        return self.file.readline()[:-1]

    def _parse_array_values(
            self,
            dtype: npt.DTypeLike,
            length: int,
    ) -> npt.NDArray[Any]:

        array = np.empty(length, dtype=dtype)
        value: int | float | complex

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
        s = self._readline().encode('ascii').decode('unicode_escape').encode('latin1')
        assert len(s) == length
        return s

    def check_complete(self) -> None:
        assert self.file.read(1) == ''
