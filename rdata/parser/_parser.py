from __future__ import annotations

import abc
import bz2
import enum
import gzip
import lzma
import os
import pathlib
import warnings
import xdrlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

import numpy as np


class FileTypes(enum.Enum):
    """Type of file containing a R file."""

    bzip2 = "bz2"
    gzip = "gzip"
    xz = "xz"
    rdata_binary_v2 = "rdata version 2 (binary)"
    rdata_binary_v3 = "rdata version 3 (binary)"


magic_dict = {
    FileTypes.bzip2: b"\x42\x5a\x68",
    FileTypes.gzip: b"\x1f\x8b",
    FileTypes.xz: b"\xFD7zXZ\x00",
    FileTypes.rdata_binary_v2: b"RDX2\n",
    FileTypes.rdata_binary_v3: b"RDX3\n",
}


def file_type(data: memoryview) -> Optional[FileTypes]:
    """Return the type of the file."""
    for filetype, magic in magic_dict.items():
        if data[:len(magic)] == magic:
            return filetype
    return None


class RdataFormats(enum.Enum):
    """Format of a R file."""

    XDR = "XDR"
    ASCII = "ASCII"
    binary = "binary"


format_dict = {
    RdataFormats.XDR: b"X\n",
    RdataFormats.ASCII: b"A\n",
    RdataFormats.binary: b"B\n",
}


def rdata_format(data: memoryview) -> Optional[RdataFormats]:
    """Return the format of the data."""
    for format_type, magic in format_dict.items():
        if data[:len(magic)] == magic:
            return format_type
    return None


class RObjectType(enum.Enum):
    """Type of a R object."""

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
    ALTREP = 238  # Alternative representations
    ATTRLIST = 239  # Bytecode attribute
    ATTRLANG = 240  # Bytecode attribute
    EMPTYENV = 242  # Empty environment
    BCREPREF = 243  # Bytecode repetition reference
    BCREPDEF = 244  # Bytecode repetition definition
    MISSINGARG = 251  # Missinf argument
    GLOBALENV = 253  # Global environment
    NILVALUE = 254  # NIL value
    REF = 255  # Reference


BYTECODE_SPECIAL_SET = {
    RObjectType.BCODE,
    RObjectType.BCREPREF,
    RObjectType.BCREPDEF,
    RObjectType.LANG,
    RObjectType.LIST,
    RObjectType.ATTRLANG,
    RObjectType.ATTRLIST,
}


class CharFlags(enum.IntFlag):
    """Flags for R objects of type char."""

    HAS_HASH = 1
    BYTES = 1 << 1
    LATIN1 = 1 << 2
    UTF8 = 1 << 3
    CACHED = 1 << 5
    ASCII = 1 << 6


@dataclass
class RVersions():
    """R versions."""

    format: int  # noqa: E701
    serialized: int
    minimum: int


@dataclass
class RExtraInfo():
    """
    Extra information.

    Contains the default encoding (only in version 3).

    """

    encoding: Optional[str] = None


@dataclass
class RObjectInfo():
    """Internal attributes of a R object."""

    type: RObjectType
    object: bool
    attributes: bool
    tag: bool
    gp: int
    reference: int


def _str_internal(
    obj: RObject | Sequence[RObject],
    indent: int = 0,
    used_references: Optional[Set[int]] = None,
) -> str:

    if used_references is None:
        used_references = set()

    small_indent = indent + 2
    big_indent = indent + 4

    indent_spaces = ' ' * indent
    small_indent_spaces = ' ' * small_indent
    big_indent_spaces = ' ' * big_indent

    string = ""

    if isinstance(obj, Sequence):
        string += f"{indent_spaces}[\n"
        for elem in obj:
            string += _str_internal(
                elem,
                big_indent,
                used_references.copy(),
            )
        string += f"{indent_spaces}]\n"

        return string

    string += f"{indent_spaces}{obj.info.type}\n"

    if obj.tag:
        tag_string = _str_internal(
            obj.tag,
            big_indent,
            used_references.copy(),
        )
        string += f"{small_indent_spaces}tag:\n{tag_string}\n"

    if obj.info.reference:
        assert obj.referenced_object
        reference_string = (
            f"{big_indent_spaces}..."
            if obj.info.reference in used_references
            else _str_internal(
                obj.referenced_object,
                indent + 4, used_references.copy())
        )
        string += (
            f"{small_indent_spaces}reference: "
            f"{obj.info.reference}\n{reference_string}\n"
        )

    string += f"{small_indent_spaces}value:\n"

    if isinstance(obj.value, RObject):
        string += _str_internal(
            obj.value,
            big_indent,
            used_references.copy(),
        )
    elif isinstance(obj.value, (tuple, list)):
        for elem in obj.value:
            string += _str_internal(
                elem,
                big_indent,
                used_references.copy(),
            )
    elif isinstance(obj.value, np.ndarray):
        string += big_indent_spaces
        if len(obj.value) > 4:
            string += (
                f"[{obj.value[0]}, {obj.value[1]} ... "
                f"{obj.value[-2]}, {obj.value[-1]}]\n"
            )
        else:
            string += f"{obj.value}\n"
    else:
        string += f"{big_indent_spaces}{obj.value}\n"

    if obj.attributes:
        attr_string = _str_internal(
            obj.attributes,
            big_indent,
            used_references.copy(),
        )
        string += f"{small_indent_spaces}attributes:\n{attr_string}\n"

    return string


@dataclass
class RObject():
    """Representation of a R object."""

    info: RObjectInfo
    value: Any
    attributes: Optional[RObject]
    tag: Optional[RObject] = None
    referenced_object: Optional[RObject] = None

    def __str__(self) -> str:
        return _str_internal(self)


@dataclass
class RData():
    """Data contained in a R file."""

    versions: RVersions
    extra: RExtraInfo
    object: RObject

    def __str__(self) -> str:
        return (
            "RData(\n"
            f"  versions: {self.versions}\n"
            f"  extra: {self.extra}\n"
            f"  object: \n{_str_internal(self.object, indent=4)}\n"
            ")\n"
        )


@dataclass
class EnvironmentValue():
    """Value of an environment."""

    locked: bool
    enclosure: RObject
    frame: RObject
    hash_table: RObject


AltRepConstructor = Callable[
    [RObject],
    Tuple[RObjectInfo, Any],
]
AltRepConstructorMap = Mapping[bytes, AltRepConstructor]


def format_float_with_scipen(number: float, scipen: int) -> bytes:
    """Format a floating point value as in R."""
    fixed = np.format_float_positional(number, trim="-")
    scientific = np.format_float_scientific(number, trim="-")

    assert isinstance(fixed, str)
    assert isinstance(scientific, str)

    return (
        scientific if len(fixed) - len(scientific) > scipen
        else fixed
    ).encode()


def deferred_string_constructor(
    state: RObject,
) -> Tuple[RObjectInfo, Any]:
    """Expand a deferred string ALTREP."""
    new_info = RObjectInfo(
        type=RObjectType.STR,
        object=False,
        attributes=False,
        tag=False,
        gp=0,
        reference=0,
    )

    object_to_format = state.value[0].value
    scipen = state.value[1].value

    value = [
        RObject(
            info=RObjectInfo(
                type=RObjectType.CHAR,
                object=False,
                attributes=False,
                tag=False,
                gp=CharFlags.ASCII,
                reference=0,
            ),
            value=format_float_with_scipen(num, scipen),
            attributes=None,
            tag=None,
            referenced_object=None,
        )
        for num in object_to_format
    ]

    return new_info, value


def compact_seq_constructor(
    state: RObject,
    *,
    is_int: bool = False,
) -> Tuple[RObjectInfo, Any]:
    """Expand a compact_seq ALTREP."""
    new_info = RObjectInfo(
        type=RObjectType.INT if is_int else RObjectType.REAL,
        object=False,
        attributes=False,
        tag=False,
        gp=0,
        reference=0,
    )

    start = state.value[1]
    stop = state.value[0]
    step = state.value[2]

    if is_int:
        start = int(start)
        stop = int(stop)
        step = int(step)

    value = np.arange(start, stop, step)

    return new_info, value


def compact_intseq_constructor(
    state: RObject,
) -> Tuple[RObjectInfo, Any]:
    """Expand a compact_intseq ALTREP."""
    return compact_seq_constructor(state, is_int=True)


def compact_realseq_constructor(
    state: RObject,
) -> Tuple[RObjectInfo, Any]:
    """Expand a compact_realseq ALTREP."""
    return compact_seq_constructor(state, is_int=False)


def wrap_constructor(
    state: RObject,
) -> Tuple[RObjectInfo, Any]:
    """Expand any wrap_* ALTREP."""
    new_info = RObjectInfo(
        type=state.value[0].info.type,
        object=False,
        attributes=False,
        tag=False,
        gp=0,
        reference=0,
    )

    value = state.value[0].value

    return new_info, value


default_altrep_map_dict: Mapping[bytes, AltRepConstructor] = {
    b"deferred_string": deferred_string_constructor,
    b"compact_intseq": compact_intseq_constructor,
    b"compact_realseq": compact_realseq_constructor,
    b"wrap_real": wrap_constructor,
    b"wrap_string": wrap_constructor,
    b"wrap_logical": wrap_constructor,
    b"wrap_integer": wrap_constructor,
    b"wrap_complex": wrap_constructor,
    b"wrap_raw": wrap_constructor,
}

DEFAULT_ALTREP_MAP = MappingProxyType(default_altrep_map_dict)


class Parser(abc.ABC):
    """Parser interface for a R file."""

    def __init__(
        self,
        *,
        expand_altrep: bool = True,
        altrep_constructor_dict: AltRepConstructorMap = DEFAULT_ALTREP_MAP,
    ):
        self.expand_altrep = expand_altrep
        self.altrep_constructor_dict = altrep_constructor_dict

    def parse_bool(self) -> bool:
        """Parse a boolean."""
        return bool(self.parse_int())

    @abc.abstractmethod
    def parse_int(self) -> int:
        """Parse an integer."""
        pass

    @abc.abstractmethod
    def parse_double(self) -> float:
        """Parse a double."""
        pass

    def parse_complex(self) -> complex:
        """Parse a complex number."""
        return complex(self.parse_double(), self.parse_double())

    @abc.abstractmethod
    def parse_string(self, length: int) -> bytes:
        """Parse a string."""
        pass

    def parse_all(self) -> RData:
        """Parse all the file."""
        versions = self.parse_versions()
        extra_info = self.parse_extra_info(versions)
        obj = self.parse_R_object()

        return RData(versions, extra_info, obj)

    def parse_versions(self) -> RVersions:
        """Parse the versions header."""
        format_version = self.parse_int()
        r_version = self.parse_int()
        minimum_r_version = self.parse_int()

        if format_version not in {2, 3}:
            raise NotImplementedError(
                f"Format version {format_version} unsupported",
            )

        return RVersions(format_version, r_version, minimum_r_version)

    def parse_extra_info(self, versions: RVersions) -> RExtraInfo:
        """
        Parse the extra info.

        Parses de encoding in version 3 format.

        """
        encoding = None

        if versions.format >= 3:
            encoding_len = self.parse_int()
            encoding = self.parse_string(encoding_len).decode("ASCII")

        return RExtraInfo(encoding)

    def expand_altrep_to_object(
        self,
        info: RObject,
        state: RObject,
    ) -> Tuple[RObjectInfo, Any]:
        """Expand alternative representation to normal object."""
        assert info.info.type == RObjectType.LIST

        class_sym = info.value[0]
        while class_sym.info.type == RObjectType.REF:
            class_sym = class_sym.referenced_object

        assert class_sym.info.type == RObjectType.SYM
        assert class_sym.value.info.type == RObjectType.CHAR

        altrep_name = class_sym.value.value
        assert isinstance(altrep_name, bytes)

        constructor = self.altrep_constructor_dict[altrep_name]
        return constructor(state)

    def _parse_bytecode_constant(
        self,
        reference_list: Optional[List[RObject]],
        bytecode_rep_list: List[RObject | None] | None = None,
    ) -> RObject:

        obj_type = self.parse_int()

        return self.parse_R_object(
            reference_list,
            bytecode_rep_list,
            info_int=obj_type,
        )

    def _parse_bytecode(
        self,
        reference_list: Optional[List[RObject]],
        bytecode_rep_list: List[RObject | None] | None = None,
    ) -> Tuple[RObject, Sequence[RObject]]:
        """Parse R bytecode."""
        if bytecode_rep_list is None:
            n_repeated = self.parse_int()

        code = self.parse_R_object(reference_list, bytecode_rep_list)

        if bytecode_rep_list is None:
            bytecode_rep_list = [None] * n_repeated

        n_constants = self.parse_int()
        constants = [
            self._parse_bytecode_constant(
                reference_list,
                bytecode_rep_list,
            )
            for _ in range(n_constants)
        ]

        return (code, constants)

    def parse_R_object(
        self,
        reference_list: List[RObject] | None = None,
        bytecode_rep_list: List[RObject | None] | None = None,
        info_int: int | None = None,
    ) -> RObject:
        """Parse a R object."""
        if reference_list is None:
            # Index is 1-based, so we insert a dummy object
            reference_list = []

        original_info_int = info_int
        if (
            info_int is not None
            and RObjectType(info_int) in BYTECODE_SPECIAL_SET
        ):
            info = parse_r_object_info(info_int)
            info.tag = info.type not in {
                RObjectType.BCREPREF,
                RObjectType.BCODE,
            }
        else:
            info_int = self.parse_int()
            info = parse_r_object_info(info_int)

        tag = None
        attributes = None
        referenced_object = None

        bytecode_rep_position = -1
        tag_read = False
        attributes_read = False
        add_reference = False

        result = None

        value: Any

        if info.type == RObjectType.BCREPDEF:
            assert bytecode_rep_list
            bytecode_rep_position = self.parse_int()
            info.type = RObjectType(self.parse_int())

        if info.type == RObjectType.NIL:
            value = None

        elif info.type == RObjectType.SYM:
            # Read Char
            value = self.parse_R_object(reference_list, bytecode_rep_list)
            # Symbols can be referenced
            add_reference = True

        elif info.type in {
            RObjectType.LIST,
            RObjectType.LANG,
            RObjectType.CLO,
            RObjectType.PROM,
            RObjectType.DOT,
            RObjectType.ATTRLANG,
        }:
            if info.type is RObjectType.ATTRLANG:
                info.type = RObjectType.LANG
                info.attributes = True

            tag = None
            if info.attributes:
                attributes = self.parse_R_object(
                    reference_list,
                    bytecode_rep_list,
                )
                attributes_read = True

            if info.tag:
                tag = self.parse_R_object(reference_list, bytecode_rep_list)
                tag_read = True

            # Read CAR and CDR
            car = self.parse_R_object(
                reference_list,
                bytecode_rep_list,
                info_int=(
                    None if original_info_int is None
                    else self.parse_int()
                ),
            )
            cdr = self.parse_R_object(
                reference_list,
                bytecode_rep_list,
                info_int=(
                    None if original_info_int is None
                    else self.parse_int()
                ),
            )
            value = (car, cdr)

        elif info.type == RObjectType.ENV:
            info.object = True

            result = RObject(
                info=info,
                tag=tag,
                attributes=attributes,
                value=None,
                referenced_object=referenced_object,
            )

            reference_list.append(result)

            locked = self.parse_bool()
            enclosure = self.parse_R_object(reference_list, bytecode_rep_list)
            frame = self.parse_R_object(reference_list, bytecode_rep_list)
            hash_table = self.parse_R_object(reference_list, bytecode_rep_list)
            attributes = self.parse_R_object(reference_list, bytecode_rep_list)

            value = EnvironmentValue(
                locked=locked,
                enclosure=enclosure,
                frame=frame,
                hash_table=hash_table,
            )

        elif info.type in {RObjectType.SPECIAL, RObjectType.BUILTIN}:
            length = self.parse_int()
            if length > 0:
                value = self.parse_string(length=length)

        elif info.type == RObjectType.CHAR:
            length = self.parse_int()
            if length > 0:
                value = self.parse_string(length=length)
            elif length == 0:
                value = b""
            elif length == -1:
                value = None
            else:
                raise NotImplementedError(
                    f"Length of CHAR cannot be {length}",
                )

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

        elif info.type in {
            RObjectType.STR,
            RObjectType.VEC,
            RObjectType.EXPR,
        }:
            length = self.parse_int()

            value = [None] * length

            for i in range(length):
                value[i] = self.parse_R_object(
                    reference_list, bytecode_rep_list)

        elif info.type == RObjectType.BCODE:
            value = self._parse_bytecode(reference_list, bytecode_rep_list)
            tag_read = True

        elif info.type == RObjectType.EXTPTR:

            result = RObject(
                info=info,
                tag=tag,
                attributes=attributes,
                value=None,
                referenced_object=referenced_object,
            )

            reference_list.append(result)
            protected = self.parse_R_object(
                reference_list,
                bytecode_rep_list,
            )
            extptr_tag = self.parse_R_object(
                reference_list,
                bytecode_rep_list,
            )

            value = (protected, extptr_tag)

        elif info.type == RObjectType.S4:
            value = None

        elif info.type == RObjectType.ALTREP:
            altrep_info = self.parse_R_object(
                reference_list,
                bytecode_rep_list,
            )
            altrep_state = self.parse_R_object(
                reference_list,
                bytecode_rep_list,
            )
            altrep_attr = self.parse_R_object(
                reference_list,
                bytecode_rep_list,
            )

            if self.expand_altrep:
                info, value = self.expand_altrep_to_object(
                    info=altrep_info,
                    state=altrep_state,
                )
                attributes = altrep_attr
            else:
                value = (altrep_info, altrep_state, altrep_attr)

        elif info.type == RObjectType.EMPTYENV:
            value = None

        elif info.type == RObjectType.BCREPREF:
            assert bytecode_rep_list
            position = self.parse_int()
            result = bytecode_rep_list[position]
            assert result
            return result

        elif info.type == RObjectType.MISSINGARG:
            value = None

        elif info.type == RObjectType.GLOBALENV:
            value = None

        elif info.type == RObjectType.NILVALUE:
            value = None

        elif info.type == RObjectType.REF:
            value = None
            # Index is 1-based
            referenced_object = reference_list[info.reference - 1]

        else:
            raise NotImplementedError(f"Type {info.type} not implemented")

        if info.tag and not tag_read:
            warnings.warn(
                f"Tag not implemented for type {info.type} "
                "and ignored",
            )
        if info.attributes and not attributes_read:
            attributes = self.parse_R_object(reference_list, bytecode_rep_list)

        if result is None:
            result = RObject(
                info=info,
                tag=tag,
                attributes=attributes,
                value=value,
                referenced_object=referenced_object,
            )
        else:
            result.info = info
            result.attributes = attributes
            result.value = value
            result.referenced_object = referenced_object

        if add_reference:
            reference_list.append(result)

        if bytecode_rep_position >= 0:
            assert bytecode_rep_list
            bytecode_rep_list[bytecode_rep_position] = result

        return result


class ParserXDR(Parser):
    """Parser used when the integers and doubles are in XDR format."""

    def __init__(
        self,
        data: memoryview,
        position: int = 0,
        *,
        expand_altrep: bool = True,
        altrep_constructor_dict: AltRepConstructorMap = DEFAULT_ALTREP_MAP,
    ) -> None:
        super().__init__(
            expand_altrep=expand_altrep,
            altrep_constructor_dict=altrep_constructor_dict,
        )
        self.data = data
        self.position = position
        self.xdr_parser = xdrlib.Unpacker(data)

    def parse_int(self) -> int:  # noqa: D102
        self.xdr_parser.set_position(self.position)
        result = self.xdr_parser.unpack_int()
        self.position = self.xdr_parser.get_position()

        return result

    def parse_double(self) -> float:  # noqa: D102
        self.xdr_parser.set_position(self.position)
        result = self.xdr_parser.unpack_double()
        self.position = self.xdr_parser.get_position()

        return result

    def parse_string(self, length: int) -> bytes:  # noqa: D102
        result = self.data[self.position:(self.position + length)]
        self.position += length
        return bytes(result)

    def parse_all(self) -> RData:
        rdata = super().parse_all()
        assert self.position == len(self.data)
        return rdata


def parse_file(
    file_or_path: Union[BinaryIO, TextIO, 'os.PathLike[Any]', str],
    *,
    expand_altrep: bool = True,
    altrep_constructor_dict: AltRepConstructorMap = DEFAULT_ALTREP_MAP,
    extension: str | None = None,
) -> RData:
    """
    Parse a R file (.rda or .rdata).

    Parameters:
        file_or_path: File in the R serialization format.
        expand_altrep: Wether to translate ALTREPs to normal objects.
        altrep_constructor_dict: Dictionary mapping each ALTREP to
            its constructor.
        extension: Extension of the file.

    Returns:
        Data contained in the file (versions and object).

    See Also:
        :func:`parse_data`: Similar function that receives the data directly.

    Examples:
        Parse one of the included examples, containing a vector

        >>> import rdata
        >>>
        >>> parsed = rdata.parser.parse_file(
        ...              rdata.TESTDATA_PATH / "test_vector.rda")
        >>> parsed
        RData(versions=RVersions(format=2,
                                 serialized=196610,
                                 minimum=131840),
              extra=RExtraInfo(encoding=None),
              object=RObject(info=RObjectInfo(type=<RObjectType.LIST: 2>,
                             object=False,
                             attributes=False,
                             tag=True,
                             gp=0,
                             reference=0),
              value=(RObject(info=RObjectInfo(type=<RObjectType.REAL: 14>,
                                              object=False,
                                              attributes=False,
                                              tag=False,
                                              gp=0,
                                              reference=0),
                             value=array([1., 2., 3.]),
                             attributes=None,
                             tag=None,
                             referenced_object=None),
                     RObject(info=RObjectInfo(type=<RObjectType.NILVALUE: 254>,
                                              object=False,
                                              attributes=False,
                                              tag=False,
                                              gp=0,
                                              reference=0),
                             value=None,
                             attributes=None,
                             tag=None,
                             referenced_object=None)),
                     attributes=None,
                     tag=RObject(info=RObjectInfo(type=<RObjectType.SYM: 1>,
                                                  object=False,
                                                  attributes=False,
                                                  tag=False,
                                                  gp=0,
                                                  reference=0),
                                 value=RObject(info=RObjectInfo(type=<RObjectType.CHAR: 9>,
                                                                object=False,
                                                                attributes=False,
                                                                tag=False,
                                                                gp=64,
                                                                reference=0),
                                               value=b'test_vector',
                                               attributes=None,
                                               tag=None,
                                               referenced_object=None),
                                 attributes=None,
                                 tag=None,
                                 referenced_object=None),
                     referenced_object=None))

    """
    if isinstance(file_or_path, (os.PathLike, str)):
        path = pathlib.Path(file_or_path)
        if extension is None:
            extension = path.suffix
        data = path.read_bytes()
    else:
        # file is a pre-opened file
        buffer: Optional[BinaryIO] = getattr(file_or_path, 'buffer', None)
        if buffer is None:
            assert isinstance(file_or_path, BinaryIO)
            binary_file: BinaryIO = file_or_path
        else:
            binary_file = buffer
        data = binary_file.read()
    return parse_data(
        data,
        expand_altrep=expand_altrep,
        altrep_constructor_dict=altrep_constructor_dict,
        extension=extension,
    )


def parse_data(
    data: bytes,
    *,
    expand_altrep: bool = True,
    altrep_constructor_dict: AltRepConstructorMap = DEFAULT_ALTREP_MAP,
    extension: str | None = None,
) -> RData:
    """
    Parse the data of a R file, received as a sequence of bytes.

    Parameters:
        data: Data extracted of a R file.
        expand_altrep: Wether to translate ALTREPs to normal objects.
        altrep_constructor_dict: Dictionary mapping each ALTREP to
            its constructor.
        extension: Extension of the file.

    Returns:
        Data contained in the file (versions and object).

    See Also:
        :func:`parse_file`: Similar function that parses a file directly.

    Examples:
        Parse one of the included examples, containing a vector

        >>> import rdata
        >>>
        >>> with open(rdata.TESTDATA_PATH / "test_vector.rda", "rb") as f:
        ...     parsed = rdata.parser.parse_data(f.read())
        >>>
        >>> parsed
        RData(versions=RVersions(format=2,
                                 serialized=196610,
                                 minimum=131840),
              extra=RExtraInfo(encoding=None),
              object=RObject(info=RObjectInfo(type=<RObjectType.LIST: 2>,
                             object=False,
                             attributes=False,
                             tag=True,
                             gp=0,
                             reference=0),
              value=(RObject(info=RObjectInfo(type=<RObjectType.REAL: 14>,
                                              object=False,
                                              attributes=False,
                                              tag=False,
                                              gp=0,
                                              reference=0),
                             value=array([1., 2., 3.]),
                             attributes=None,
                             tag=None,
                             referenced_object=None),
                     RObject(info=RObjectInfo(type=<RObjectType.NILVALUE: 254>,
                                              object=False,
                                              attributes=False,
                                              tag=False,
                                              gp=0,
                                              reference=0),
                             value=None,
                             attributes=None,
                             tag=None,
                             referenced_object=None)),
                     attributes=None,
                     tag=RObject(info=RObjectInfo(type=<RObjectType.SYM: 1>,
                                                  object=False,
                                                  attributes=False,
                                                  tag=False,
                                                  gp=0,
                                                  reference=0),
                                 value=RObject(info=RObjectInfo(type=<RObjectType.CHAR: 9>,
                                                                object=False,
                                                                attributes=False,
                                                                tag=False,
                                                                gp=64,
                                                                reference=0),
                                               value=b'test_vector',
                                               attributes=None,
                                               tag=None,
                                               referenced_object=None),
                                 attributes=None,
                                 tag=None,
                                 referenced_object=None),
                     referenced_object=None))

    """
    view = memoryview(data)

    filetype = file_type(view)

    parse_function = (
        parse_rdata_binary
        if filetype in {
            FileTypes.rdata_binary_v2,
            FileTypes.rdata_binary_v3,
            None,
        } else parse_data
    )

    if filetype is FileTypes.bzip2:
        new_data = bz2.decompress(data)
    elif filetype is FileTypes.gzip:
        new_data = gzip.decompress(data)
    elif filetype is FileTypes.xz:
        new_data = lzma.decompress(data)
    elif filetype in {FileTypes.rdata_binary_v2, FileTypes.rdata_binary_v3}:
        if extension == ".rds":
            warnings.warn(
                f"Wrong extension {extension} for file in RDATA format",
            )

        view = view[len(magic_dict[filetype]):]
        new_data = view
    else:
        new_data = view
        if extension != ".rds":
            warnings.warn("Unknown file type: assumed RDS")

    return parse_function(
        new_data,  # type: ignore
        expand_altrep=expand_altrep,
        altrep_constructor_dict=altrep_constructor_dict,
        extension=extension,
    )


def parse_rdata_binary(
    data: memoryview,
    expand_altrep: bool = True,
    altrep_constructor_dict: AltRepConstructorMap = DEFAULT_ALTREP_MAP,
    extension: str | None = None,
) -> RData:
    """Select the appropiate parser and parse all the info."""
    format_type = rdata_format(data)

    if format_type:
        data = data[len(format_dict[format_type]):]

    if format_type is RdataFormats.XDR:
        parser = ParserXDR(
            data,
            expand_altrep=expand_altrep,
            altrep_constructor_dict=altrep_constructor_dict,
        )
        return parser.parse_all()

    raise NotImplementedError("Unknown file format")


def bits(data: int, start: int, stop: int) -> int:
    """Read bits [start, stop) of an integer."""
    count = stop - start
    mask = ((1 << count) - 1) << start

    bitvalue = data & mask
    return bitvalue >> start


def is_special_r_object_type(r_object_type: RObjectType) -> bool:
    """Check if a R type has a different serialization than the usual one."""
    return (
        r_object_type is RObjectType.NILVALUE
        or r_object_type is RObjectType.REF
    )


def parse_r_object_info(info_int: int) -> RObjectInfo:
    """Parse the internal information of an object."""
    type_exp = RObjectType(bits(info_int, 0, 8))

    reference = 0

    if is_special_r_object_type(type_exp):
        object_flag = False
        attributes = False
        tag = False
        gp = 0
    else:
        object_flag = bool(bits(info_int, 8, 9))
        attributes = bool(bits(info_int, 9, 10))
        tag = bool(bits(info_int, 10, 11))  # noqa: WPS432
        gp = bits(info_int, 12, 28)  # noqa: WPS432

    if type_exp == RObjectType.REF:
        reference = bits(info_int, 8, 32)  # noqa: WPS432

    return RObjectInfo(
        type=type_exp,
        object=object_flag,
        attributes=attributes,
        tag=tag,
        gp=gp,
        reference=reference,
    )
