"""Abstract base class for writers."""

from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from rdata.parser import (
    RData,
    RExtraInfo,
    RObject,
    RObjectInfo,
    RObjectType,
    RVersions,
)

if TYPE_CHECKING:
    import numpy.typing as npt


def pack_r_object_info(info: RObjectInfo) -> np.int32:
    """Pack RObjectInfo to an integer."""
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
    assert len(bits) == 32  # noqa: PLR2004
    return np.packbits([int(b) for b in bits]).view(">i4").astype("=i4")[0]


class Writer(abc.ABC):
    """Writer interface for a R file."""

    @abc.abstractmethod
    def write_magic(self, rda_version: int) -> None:
        """Write magic bits."""

    def write_header(self, versions: RVersions, extra: RExtraInfo) -> None:
        """Write header."""
        self.write_int(versions.format)
        self.write_int(versions.serialized)
        self.write_int(versions.minimum)
        minimum_version_with_encoding = 3
        if versions.format >= minimum_version_with_encoding:
            self.write_string(extra.encoding.encode("ascii"))

    def write_bool(self, value: bool) -> None:  # noqa: FBT001
        """Write a boolean value."""
        self.write_int(int(value))

    def write_int(self, value: int) -> None:
        """Write an integer value."""
        self._write_array_values(np.array([value]))

    def _write_array(self, array: npt.NDArray[Any]) -> None:
        """Write an array of values."""
        # Expect only 1D arrays here
        assert array.ndim == 1
        self.write_int(array.size)
        self._write_array_values(array)

    @abc.abstractmethod
    def _write_array_values(self, array: npt.NDArray[Any]) -> None:
        """Write magic bits."""

    @abc.abstractmethod
    def write_string(self, value: bytes) -> None:
        """Write a string."""

    def write_r_data(self, r_data: RData, *, rds: bool = True) -> None:
        """Write an RData object."""
        self.write_magic(None if rds else r_data.versions.format)
        self.write_header(r_data.versions, r_data.extra)
        self.write_r_object(r_data.object)

    def write_r_object(self, obj: RObject) -> None:  # noqa: C901, PLR0912
        """Write an RObject object."""
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
            self.write_r_object(value)

        elif info.type in {
            RObjectType.LIST,
            RObjectType.LANG,
            # Parser treats the following equal to LIST.
            # Not tested if they work
            # RObjectType.CLO,
            # RObjectType.PROM,
            # RObjectType.DOT,
            # RObjectType.ATTRLANG,
        }:
            if info.attributes:
                self.write_r_object(obj.attributes)
                attributes_written = True

            if info.tag:
                self.write_r_object(obj.tag)
                tag_written = True

            for element in value:
                self.write_r_object(element)

        elif info.type in {
            RObjectType.CHAR,
            RObjectType.BUILTIN,
            # Parser treats the following equal to LIST.
            # Not tested if they work
            # RObjectType.SPECIAL,
        }:
            self.write_string(value)

        elif info.type in {
            RObjectType.LGL,
            RObjectType.INT,
            RObjectType.REAL,
            RObjectType.CPLX,
        }:
            self._write_array(value)

        elif info.type in {
            RObjectType.STR,
            RObjectType.VEC,
            RObjectType.EXPR,
        }:
            self.write_int(len(value))
            for element in value:
                self.write_r_object(element)

        else:
            msg = f"{info.type}"
            raise NotImplementedError(msg)

        # Write attributes if it has not been written yet
        if info.attributes and not attributes_written:
            self.write_r_object(obj.attributes)

        # Write tag if it has not been written yet
        if info.tag and not tag_written:
            warnings.warn(  # noqa: B028
                f"Tag not implemented for type {info.type} "
                "and ignored",
            )
