"""Abstract base class for unparsers."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np

from rdata.missing import R_INT_NA
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


def pack_r_object_info(info: RObjectInfo) -> int:
    """Pack RObjectInfo to an integer."""
    if info.type == RObjectType.NILVALUE:
        bits = f"{0:24b}"
    elif info.type == RObjectType.REF:
        bits = f"{info.reference:24b}"
    else:
        bits = (
            f"{0:4b}"
            f"{info.gp:16b}"
            f"{0:1b}"
            f"{info.tag:1b}"
            f"{info.attributes:1b}"
            f"{info.object:1b}"
        )
    bits += f"{info.type.value:8b}"
    bits = bits.replace(" ", "0")
    assert len(bits) == 32  # noqa: PLR2004
    return int(f"0b{bits}", base=2)


class Unparser(abc.ABC):
    """Unparser interface for a R file."""

    @abc.abstractmethod
    def unparse_magic(self) -> None:
        """Unparse magic bits."""

    def unparse_header(self, versions: RVersions, extra: RExtraInfo) -> None:
        """Unparse header."""
        self.unparse_int(versions.format)
        self.unparse_int(versions.serialized)
        self.unparse_int(versions.minimum)
        minimum_version_with_encoding = 3
        if versions.format >= minimum_version_with_encoding:
            assert extra.encoding is not None
            self.unparse_string(extra.encoding.encode("ascii"))

    def unparse_int(self, value: int | np.int32) -> None:
        """Unparse an integer value."""
        self._unparse_array_values(np.array([value]))

    def unparse_array(self, array: npt.NDArray[Any]) -> None:
        """Unparse an array of values."""
        # Expect only 1D arrays here
        assert array.ndim == 1
        self.unparse_int(array.size)
        self._unparse_array_values(array)

    def _unparse_array_values(self, array: npt.NDArray[Any]) -> None:
        """Unparse the values of an array."""
        # Convert boolean to int
        if np.issubdtype(array.dtype, np.bool_):
            array = array.astype(np.int32)

        # Flatten masked values and convert int arrays to int32
        if np.issubdtype(array.dtype, np.integer):
            if np.ma.is_masked(array):  # type: ignore [no-untyped-call]
                mask = np.ma.getmask(array)  # type: ignore [no-untyped-call]
                array = np.ma.getdata(array).copy()  # type: ignore [no-untyped-call]
                array[mask] = R_INT_NA

            if array.dtype != np.int32:
                info = np.iinfo(np.int32)
                if np.any(array > info.max) or np.any(array < info.min):
                    msg = "Integer array not castable to int32"
                    raise ValueError(msg)
                array = array.astype(np.int32)

        assert array.dtype in (np.int32, np.float64, np.complex128)
        self._unparse_array_values_raw(array)

    @abc.abstractmethod
    def _unparse_array_values_raw(self,
        array: npt.NDArray[np.int32 | np.float64 | np.complex128],
    ) -> None:
        """Unparse the values of an array as such."""

    def unparse_string(self, value: bytes | None) -> None:
        """Unparse a string."""
        if value is None:
            self.unparse_int(-1)
            return
        self.unparse_int(len(value))
        self._unparse_string_characters(value)

    @abc.abstractmethod
    def _unparse_string_characters(self, value: bytes) -> None:
        """Unparse characters of a string (not None)."""

    def unparse_r_data(self, r_data: RData) -> None:
        """Unparse an RData object."""
        self.unparse_magic()
        self.unparse_header(r_data.versions, r_data.extra)
        self.unparse_r_object(r_data.object)

    def unparse_r_object(self, obj: RObject) -> None:  # noqa: C901, PLR0912
        """Unparse an RObject object."""
        # Some types include attributes and tag with data while some add them
        # later. These booleans keep track of whether attributes or tag
        # has been done already
        attributes_done = False
        tag_done = False

        # Unparse info bytes
        info = obj.info
        self.unparse_int(pack_r_object_info(info))

        # Unparse data
        value = obj.value
        if info.type in {
            RObjectType.NIL,
            RObjectType.NILVALUE,
            RObjectType.REF,
        }:
            # These types don't have any data
            assert value is None

        elif info.type == RObjectType.SYM:
            self.unparse_r_object(value)

        elif info.type in {
            RObjectType.LIST,
            RObjectType.LANG,
            RObjectType.ALTREP,
            # Parser treats the following equal to LIST.
            # Not tested if they work
            # RObjectType.CLO,
            # RObjectType.PROM,
            # RObjectType.DOT,
            # RObjectType.ATTRLANG,
        }:
            if info.attributes:
                assert obj.attributes is not None
                self.unparse_r_object(obj.attributes)
                attributes_done = True

            if info.tag:
                assert obj.tag is not None
                self.unparse_r_object(obj.tag)
                tag_done = True

            for element in value:
                self.unparse_r_object(element)

        elif info.type in {
            RObjectType.CHAR,
            RObjectType.BUILTIN,
            # Parser treats the following equal to CHAR.
            # Not tested if they work
            # RObjectType.SPECIAL,
        }:
            self.unparse_string(value)

        elif info.type in {
            RObjectType.LGL,
            RObjectType.INT,
            RObjectType.REAL,
            RObjectType.CPLX,
        }:
            self.unparse_array(value)

        elif info.type in {
            RObjectType.STR,
            RObjectType.VEC,
            RObjectType.EXPR,
        }:
            self.unparse_int(len(value))
            for element in value:
                self.unparse_r_object(element)

        else:
            msg = f"type {info.type} not implemented"
            raise NotImplementedError(msg)

        # Unparse attributes if it has not been done yet
        if info.attributes and not attributes_done:
            assert obj.attributes is not None
            self.unparse_r_object(obj.attributes)

        # Unparse tag if it has not been done yet
        if info.tag and not tag_done:
            msg = f"Tag not implemented for type {info.type}"
            raise NotImplementedError(msg)
