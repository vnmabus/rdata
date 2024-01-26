from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import BinaryIO

from .base import Writer
from rdata.parser._parser import R_INT_NA


def flatten_nullable_int_array(array):
    assert array.dtype in (np.int32, int)
    if np.ma.is_masked(array):
        mask = np.ma.getmask(array)
        array = np.ma.getdata(array).copy()
        array[mask] = R_INT_NA
    return array


class WriterXDR(Writer):
    """Writer for XDR format files."""

    def __init__(
        self,
        file: BinaryIO,
    ) -> None:
        self.file = file

    def write_magic(self):
        self.file.write(b'X\n')

    def __write_array(self, array):
        self.write_int(array.size)
        self.__write_array_values(array)

    def __write_array_values(self, array):
        # XXX tobytes() creates a memory copy
        # It'd be better to check contiguity and use ndarray.data
        self.file.write(array.astype(array.dtype.newbyteorder('>')).tobytes())

    def write_nullable_bool(self, value):
        if value is None or np.ma.is_masked(value):
            value = R_INT_NA
        self.__write_array_values(np.array(value).astype(np.int32))

    def write_nullable_int(self, value):
        if value is None or np.ma.is_masked(value):
            value = R_INT_NA
        self.__write_array_values(np.array(value).astype(np.int32))

    def write_double(self, value):
        self.__write_array_values(np.array(value))

    def write_complex(self, value):
        self.__write_array_values(np.array(value))

    def write_nullable_bool_array(self, array):
        self.write_nullable_int_array(array.astype(np.int32))

    def write_nullable_int_array(self, array):
        array = flatten_nullable_int_array(array)
        self.__write_array(array.astype(np.int32))

    def write_double_array(self, array):
        self.__write_array(array)

    def write_complex_array(self, array):
        self.__write_array(array)

    def write_string(self, value: bytes):
        self.write_int(len(value))
        self.file.write(value)
