import abc
import bz2
import enum
import gzip
import lzma
import os
import pathlib
from typing import Optional, NamedTuple, Any, Union, BinaryIO
import warnings
import xdrlib

import numpy as np


class FileTypes(enum.Enum):
    """
    Type of file containing a R file.
    """
    bzip2 = "bz2"
    gzip = "gzip"
    xz = "xz"
    rdata_binary = "rdata (binary)"


magic_dict = {
    FileTypes.bzip2: b"\x42\x5a\x68",
    FileTypes.gzip: b"\x1f\x8b",
    FileTypes.xz: b"\xFD7zXZ\x00",
    FileTypes.rdata_binary: b"RDX2\n"
}


def file_type(data: memoryview):
    """
    Returns the type of the file.
    """

    for filetype, magic in magic_dict.items():
        if data[:len(magic)] == magic:
            return filetype
    return None


class RdataFormats(enum.Enum):
    """
    Format of a R file.
    """
    XDR = "XDR"
    ASCII = "ASCII"
    binary = "binary"


format_dict = {
    RdataFormats.XDR: b"X\n",
    RdataFormats.ASCII: b"A\n",
    RdataFormats.binary: b"B\n",
}


def rdata_format(data: memoryview):
    """
    Returns the format of the data.
    """

    for format_type, magic in format_dict.items():
        if data[:len(magic)] == magic:
            return format_type
    return None


class RObjectType(enum.Enum):
    """
    Type of a R object.
    """
    NIL = 0  # NULL
    SYM = 1  # symbols
    LIST = 2  # pairlists
    CLO = 3  # closures
    ENV = 4  # environments
    PROM = 5  # promises
    LANG = 6  # language objects
    SPECIAL = 7  # special functions
    BUILTIN = 8  # builtin functions
    CHAR = 9  # internal character strings
    LGL = 10  # logical vectors
    INT = 13  # integer vectors
    REAL = 14  # numeric vectors
    CPLX = 15  # complex vectors
    STR = 16  # character vectors
    DOT = 17  # dot-dot-dot object
    ANY = 18  # make “any” args work
    VEC = 19  # list (generic vector)
    EXPR = 20  # expression vector
    BCODE = 21  # byte code
    EXTPTR = 22  # external pointer
    WEAKREF = 23  # weak reference
    RAW = 24  # raw vector
    S4 = 25  # S4 classes not of simple type
    NILVALUE = 254  # NIL value
    REF = 255  # Reference


class CharFlags(enum.IntFlag):
    HAS_HASH = 1
    BYTES = 1 << 1
    LATIN1 = 1 << 2
    UTF8 = 1 << 3
    CACHED = 1 << 5
    ASCII = 1 << 6


class RVersions(NamedTuple):
    """
    R versions.
    """
    format: int
    serialized: int
    minimum: int


class RObjectInfo(NamedTuple):
    """
    Internal attributes of a R object.
    """
    type: RObjectType
    object: bool
    attributes: bool
    tag: bool
    gp: int
    reference: int


class RObject(NamedTuple):
    """
    Representation of a R object.
    """
    info: RObjectInfo
    value: Any
    attributes: 'RObject'
    tag: Optional['RObject'] = None
    referenced_object: Optional['RObject'] = None

    def _str_internal(self, indent: int=0, used_references=None):

        if used_references is None:
            used_references = set()

        string = ""

        string += f"{' ' * indent}{self.info.type}\n"

        if self.tag:
            tag_string = self.tag._str_internal(indent + 4,
                                                used_references.copy())
            string += f"{' ' * (indent + 2)}tag:\n{tag_string}\n"

        if self.info.reference:
            assert self.referenced_object
            reference_string = (f"{' ' * (indent + 4)}..."
                                if self.info.reference in used_references
                                else self.referenced_object._str_internal(
                                    indent + 4, used_references.copy()))
            string += (f"{' ' * (indent + 2)}reference: "
                       f"{self.info.reference}\n{reference_string}\n")

        string += f"{' ' * (indent + 2)}value:\n"

        if isinstance(self.value, RObject):
            string += self.value._str_internal(indent + 4,
                                               used_references.copy())
        elif isinstance(self.value, tuple) or isinstance(self.value, list):
            for elem in self.value:
                string += elem._str_internal(indent + 4,
                                             used_references.copy())
        elif isinstance(self.value, np.ndarray):
            string += " " * (indent + 4)
            if len(self.value) > 4:
                string += (f"[{self.value[0]}, {self.value[1]} ... "
                           f"{self.value[-2]}, {self.value[-1]}]\n")
            else:
                string += f"{self.value}\n"
        else:
            string += f"{' ' * (indent + 4)}{self.value}\n"

        if(self.attributes):
            attr_string = self.attributes._str_internal(
                indent + 4,
                used_references.copy())
            string += f"{' ' * (indent + 2)}attributes:\n{attr_string}\n"

        return string

    def __str__(self):
        return self._str_internal()


class RData(NamedTuple):
    """
    Data contained in a R file.
    """
    versions: RVersions
    object: RObject


class Parser(abc.ABC):
    """
    Parser interface for a R file.
    """

    def parse_bool(self) -> bool:
        """
        Parse a boolean.
        """
        return bool(self.parse_int())

    @abc.abstractmethod
    def parse_int(self) -> int:
        """
        Parse an integer.
        """
        pass

    @abc.abstractmethod
    def parse_double(self) -> float:
        """
        Parse a double.
        """
        pass

    def parse_complex(self) -> complex:
        """
        Parse a complex number.
        """
        return complex(self.parse_double(), self.parse_double())

    @abc.abstractmethod
    def parse_string(self, length) -> bytes:
        """
        Parse a string.
        """
        pass

    def parse_all(self) -> RData:
        """
        Parse all the file.
        """

        versions = self.parse_versions()
        obj = self.parse_R_object()

        return RData(versions, obj)

    def parse_versions(self) -> RVersions:
        """
        Parse the versions header.
        """

        format_version = self.parse_int()
        r_version = self.parse_int()
        minimum_r_version = self.parse_int()

        if format_version != 2:
            raise NotImplementedError(
                "Format version {format_version} unsupported")

        return RVersions(format_version, r_version, minimum_r_version)

    def parse_R_object(self, reference_list=None):
        """
        Parse a R object.
        """

        if reference_list is None:
            # Index is 1-based, so we insert a dummy object
            reference_list = [None]

        info_int = self.parse_int()

        info = parse_r_object_info(info_int)

        tag = None
        attributes = None
        referenced_object = None

        tag_read = False
        attributes_read = False
        add_reference = False

        if info.type == RObjectType.SYM:
            # Read Char
            value = self.parse_R_object(reference_list)
            # Symbols can be referenced
            add_reference = True

        elif info.type in [RObjectType.LIST, RObjectType.LANG]:
            tag = None
            if info.attributes:
                raise NotImplementedError("Attributes not suported for LIST")
            elif info.tag:
                tag = self.parse_R_object(reference_list)
                tag_read = True

            # Read CAR and CDR
            car = self.parse_R_object(reference_list)
            cdr = self.parse_R_object(reference_list)
            value = (car, cdr)

        elif info.type == RObjectType.CHAR:
            length = self.parse_int()
            if length > 0:
                value = self.parse_string(length=length)
            elif length == -1:
                value = b""
            else:
                raise NotImplementedError(
                    f"Length of CHAR can not be {length}")

        elif info.type == RObjectType.LGL:
            length = self.parse_int()

            value = np.empty(length, dtype=np.bool_)

            for i in range(length):
                value[i] = self.parse_bool()

        elif info.type == RObjectType.INT:
            length = self.parse_int()

            value = np.empty(length, dtype=np.int64)

            for i in range(length):
                value[i] = self.parse_int()

        elif info.type == RObjectType.REAL:
            length = self.parse_int()

            value = np.empty(length, dtype=np.double)

            for i in range(length):
                value[i] = self.parse_double()

        elif info.type == RObjectType.CPLX:
            length = self.parse_int()

            value = np.empty(length, dtype=np.complex_)

            for i in range(length):
                value[i] = self.parse_complex()

        elif info.type in [RObjectType.STR,
                           RObjectType.VEC, RObjectType.EXPR]:
            length = self.parse_int()

            value = [None] * length

            for i in range(length):
                value[i] = self.parse_R_object(reference_list)

        elif info.type == RObjectType.NILVALUE:
            value = None

        elif info.type == RObjectType.REF:
            value = None
            referenced_object = reference_list[info.reference]

        else:
            raise NotImplementedError(f"Type {info.type} not implemented")

        if info.tag and not tag_read:
            warnings.warn(f"Tag not implemented for type {info.type} "
                          "and ignored")
        if info.attributes and not attributes_read:
            attributes = self.parse_R_object(reference_list)

        result = RObject(info=info, tag=tag,
                         attributes=attributes,
                         value=value,
                         referenced_object=referenced_object)

        if add_reference:
            reference_list.append(result)

        return result


class ParserXDR(Parser):
    """
    Parser used when the integers and doubles are in XDR format.
    """

    def __init__(self, data, position=0):
        self.data = data
        self.position = position
        self.xdr_parser = xdrlib.Unpacker(data)

    def parse_int(self):
        self.xdr_parser.set_position(self.position)
        result = self.xdr_parser.unpack_int()
        self.position = self.xdr_parser.get_position()

        return result

    def parse_double(self):
        self.xdr_parser.set_position(self.position)
        result = self.xdr_parser.unpack_double()
        self.position = self.xdr_parser.get_position()

        return result

    def parse_string(self, length):
        result = self.data[self.position:(self.position + length)]
        self.position += length
        return bytes(result)


def parse_file(file_or_path: Union[BinaryIO, os.PathLike,
                                   str]) -> RData:
    """
    Parse a R file (.rda or .rdata).

    Parameters
    ----------
    file_or_path: file-like, str, bytes or path-like
        File in the R serialization format.

    Returns
    -------
    RData
        Data contained in the file (versions and object).

    See Also
    --------
    parse_data

    """
    if isinstance(file_or_path, (os.PathLike, str)):
        path = pathlib.Path(file_or_path)
        data = path.read_bytes()
    else:
        # file is a pre-opened file
        buffer: Optional[BinaryIO] = getattr(file_or_path, 'buffer', None)
        if buffer is not None:
            file_or_path = buffer
        data = file_or_path.read()
    return parse_data(data)


def parse_data(data: bytes) -> RData:
    """
    Parse the data of a R file, received as a sequence of bytes.

    Parameters
    ----------
    data: bytes
        Data extracted of a R file.

    Returns
    -------
    RData
        Data contained in the file (versions and object).

    See Also
    --------
    parse_file

    """
    view = memoryview(data)

    filetype = file_type(view)

    if filetype is FileTypes.bzip2:
        return parse_data(bz2.decompress(data))
    elif filetype is FileTypes.gzip:
        return parse_data(gzip.decompress(data))
    elif filetype is FileTypes.xz:
        return parse_data(lzma.decompress(data))
    elif filetype is FileTypes.rdata_binary:
        view = view[len(magic_dict[filetype]):]
        return parse_rdata_binary(view)
    else:
        raise NotImplementedError("Unknown file type")


def parse_rdata_binary(data: memoryview):
    """
    Select the appropiate parser and parse all the info.
    """
    format_type = rdata_format(data)

    if format_type:
        data = data[len(format_dict[format_type]):]

    if format_type is RdataFormats.XDR:
        parser = ParserXDR(data)
        return parser.parse_all()
    else:
        raise NotImplementedError("Unknown file format")


def bits(data, start, stop):
    """
    Read bits [start, stop) of an integer.
    """
    count = stop - start
    mask = ((1 << count) - 1) << start

    bitvalue = data & mask
    return bitvalue >> start


def is_special_r_object_type(r_object_type: RObjectType):
    """
    Check if a R type has a different serialization than the usual one.
    """
    return (r_object_type is RObjectType.NILVALUE
            or r_object_type is RObjectType.REF)


def parse_r_object_info(info_int: int) -> RObjectInfo:
    """
    Parse the internal information of an object.
    """
    type_exp = RObjectType(bits(info_int, 0, 8))

    reference = 0

    if is_special_r_object_type(type_exp):
        object_flag = False
        attributes = False
        tag = False
        gp = False
    else:
        object_flag = bool(bits(info_int, 8, 9))
        attributes = bool(bits(info_int, 9, 10))
        tag = bool(bits(info_int, 10, 11))
        gp = bits(info_int, 12, 28)

    if type_exp == RObjectType.REF:
        reference = bits(info_int, 8, 32)

    return RObjectInfo(
        type=type_exp,
        object=object_flag,
        attributes=attributes,
        tag=tag,
        gp=gp,
        reference=reference
    )
