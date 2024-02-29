from __future__ import annotations

import abc
import warnings

import numpy as np

from rdata.parser._parser import (
    RExtraInfo,
    RObjectInfo,
    RObjectType,
    RVersions,
)


def pack_r_object_info(info: RObjectInfo) -> np.int32:
    if info.type == RObjectType.NILVALUE:
        bits = f"{0:24b}"
    elif info.type == RObjectType.REF:
        bits = f"{info.reference:24b}"
    else:
        bits = (f"{0:4b}"
                f"{info.gp:16b}"
                f"{0:1b}"
                f"{info.tag:1b}"
                f"{info.attributes:1b}"
                f"{info.object:1b}"
                )
    bits += f"{info.type.value:8b}"
    bits = bits.replace(" ", "0")
    assert len(bits) == 32
    info_int = np.packbits([int(b) for b in bits]).view(">i4").astype("=i4")[0]
    return info_int


class Writer(abc.ABC):
    """Writer interface for a R file."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def write_magic(self, rda_version):
        pass

    def write_header(self, versions: RVersions, extra: RExtraInfo):
        """Write header."""
        self.write_int(versions.format)
        self.write_int(versions.serialized)
        self.write_int(versions.minimum)
        if versions.format >= 3:
            self.write_string(extra.encoding.encode("ascii"))

    def write_bool(self, value):
        self.write_int(int(value))

    @abc.abstractmethod
    def write_nullable_bool(self, value):
        pass

    def write_int(self, value):
        if not isinstance(value, (int, np.int32)):
            raise RuntimeError(f"Not valid integer: {value} ({type(value)})")
        self.write_nullable_int(value)

    @abc.abstractmethod
    def write_nullable_int(self, value):
        pass

    @abc.abstractmethod
    def write_double(self, value):
        pass

    def write_complex(self, value):
        self.write_double(value.real)
        self.write_double(value.imag)

    @abc.abstractmethod
    def write_string(self, value):
        pass

    def __write_array(self, array, write_value):
        # Expect only 1D arrays here
        assert array.ndim == 1
        self.write_int(array.size)
        for value in array:
            write_value(value)

    def write_nullable_bool_array(self, array):
        """Write a boolean array."""
        self.__write_array(array, self.write_nullable_bool)

    def write_nullable_int_array(self, array):
        """Write an integer array."""
        self.__write_array(array, self.write_nullable_int)

    def write_double_array(self, array):
        """Write a double array."""
        self.__write_array(array, self.write_double)

    def write_complex_array(self, array):
        """Write a complex array."""
        self.__write_array(array, self.write_complex)

    def write_r_data(self, r_data, *, rds=True):
        self.write_magic(None if rds else r_data.versions.format)
        self.write_header(r_data.versions, r_data.extra)
        self.write_R_object(r_data.object)

    def write_R_object(self, obj):
        # Some types write attributes and tag with data while some write them
        # later. These booleans keep track of whether attributes or tag
        # has been written already
        attributes_written = False
        tag_written = False

        # Write info bytes
        info = obj.info
        self.write_int(pack_r_object_info(info))

        # Write data
        value = obj.value
        if info.type in {
           RObjectType.NIL,
           RObjectType.NILVALUE,
        }:
            # These types don't have any data
            assert value is None

        elif info.type == RObjectType.SYM:
            self.write_R_object(value)

        elif info.type in {
            RObjectType.LIST,
            # XXX Parser treats these equally as LIST.
            #     Not tested if they work
            # RObjectType.LANG,
            # RObjectType.CLO,
            # RObjectType.PROM,
            # RObjectType.DOT,
            # RObjectType.ATTRLANG,
        }:
            if info.attributes:
                self.write_R_object(obj.attributes)
                attributes_written = True

            if info.tag:
                self.write_R_object(obj.tag)
                tag_written = True

            for element in value:
                self.write_R_object(element)

        elif info.type == RObjectType.CHAR:
            self.write_string(value)

        elif info.type == RObjectType.LGL:
            self.write_nullable_bool_array(value)

        elif info.type == RObjectType.INT:
            self.write_nullable_int_array(value)

        elif info.type == RObjectType.REAL:
            self.write_double_array(value)

        elif info.type == RObjectType.CPLX:
            self.write_complex_array(value)

        elif info.type in {
            RObjectType.STR,
            RObjectType.VEC,
            # XXX Parser treats this equally as VEC.
            #     Not tested if it works
            # RObjectType.EXPR,
        }:
            self.write_int(len(value))
            for element in value:
                self.write_R_object(element)

        else:
            raise NotImplementedError(f"{info.type}")

        # Write attributes if it has not been written yet
        if info.attributes and not attributes_written:
            self.write_R_object(obj.attributes)

        # Write tag if it has not been written yet
        if info.tag and not tag_written:
            warnings.warn(
                f"Tag not implemented for type {info.type} "
                "and ignored",
            )
