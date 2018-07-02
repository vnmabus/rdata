
from builtins import int
import bz2
import enum
import pathlib
import typing
import xdrlib
import numpy as np


class FileTypes(enum.Enum):
    bzip2 = "bz2"
    rdata_binary = "rdata (binary)"


class RdataFormats(enum.Enum):
    XDR = "XDR"
    ASCII = "ASCII"
    binary = "binary"


class RObjectType(enum.Enum):
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


class RVersions(typing.NamedTuple):
    format: int
    serialized: int
    minimum: int


class RObjectInfo(typing.NamedTuple):
    type: RObjectType
    object: bool
    attributes: bool
    tag: bool
    gp: int
    reference: int


class RObject(typing.NamedTuple):
    info: RObjectInfo
    value: typing.Any
    attributes: typing.Container['RObject']
    tag: typing.Union['RObject', None] = None
    referenced_object: 'RObject' = None


class RData(typing.NamedTuple):
    versions: RVersions
    object: RObject


def print_R_object(obj:RObject, indent=0):

    print(" " * indent, end="")
    print(f"{obj.info.type}")

    if obj.tag:
        print(" " * (indent + 2), end="")
        print("tag:")
        print_R_object(obj.tag, indent=indent + 4)

    if obj.info.reference:
        print(" " * (indent + 2), end="")
        print(f"reference: {obj.info.reference}")
        print_R_object(obj.referenced_object, indent=indent + 4)

    print(" " * (indent + 2), end="")
    print("value:")

    if isinstance(obj.value, RObject):
        print_R_object(obj.value, indent=indent + 4)
    elif isinstance(obj.value, tuple) or isinstance(obj.value, list):
        for elem in obj.value:
            print_R_object(elem, indent=indent + 4)
    elif isinstance(obj.value, np.ndarray):
        print(" " * (indent + 4), end="")
        if len(obj.value) > 4:
            print(f"[{obj.value[0]}, {obj.value[1]} ... {obj.value[-2]}, {obj.value[-1]}]")
        else:
            print(obj.value)
    else:
        print(" " * (indent + 4), end="")
        print(obj.value)

    if(obj.attributes):
        print(" " * (indent + 2), end="")
        print("attributes:")
        print_R_object(obj.attributes, indent=indent + 4)


class ParserXDR():

    def __init__(self, data, position=0):
        self.data = data
        self.position = position
        self.xdr_parser = xdrlib.Unpacker(data)

    @property
    def remaining_data(self):
        return self.data[self.position:]

    def parse_all(self):

        versions = self.parse_versions()
        obj = self.parse_R_object()

        return RData(versions, obj)

    def parse_versions(self):

        format_version = self.parse_int()
        r_version = self.parse_int()
        minimum_r_version = self.parse_int()

        print(format_version, r_version, minimum_r_version)

        if format_version != 2:
            raise NotImplementedError("Format version {format_version} unsupported")

        return RVersions(format_version, r_version, minimum_r_version)

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

    def parse_R_object(self, reference_list=None):

        if reference_list is None:
            reference_list = [None]  # Index is 1-based, so we insert a dummy object

        info_int = self.parse_int()

        info = parse_r_object_info(info_int)
        print(info)

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

        elif info.type == RObjectType.LIST:
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
            print(f'length={length}')
            if length > 0:
                value = self.parse_string(length=length)
            elif length == -1:
                value = b""
            else:
                raise NotImplementedError(f"Length of CHAR can not be {length}")
            print(f'string={value}')

        elif info.type == RObjectType.INT:
            length = self.parse_int()
            print(f'length={length}')

            value = np.empty(length, dtype=np.int32)

            for i in range(length):
                value[i] = self.parse_int()

        elif info.type == RObjectType.REAL:
            length = self.parse_int()
            print(f'length={length}')

            value = np.empty(length, dtype=np.double)

            for i in range(length):
                value[i] = self.parse_double()

        elif info.type == RObjectType.STR:
            length = self.parse_int()
            print(f'length={length}')

            value = [None] * length

            for i in range(length):
                value[i] = self.parse_R_object(reference_list)

        elif info.type == RObjectType.VEC:
            length = self.parse_int()
            print(f'length={length}')
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

        if info.object:
            raise NotImplementedError(f"Object not implemented")
        if info.tag and not tag_read:
            raise NotImplementedError(f"Tag not implemented")
        if info.attributes and not attributes_read:
            attributes = self.parse_R_object(reference_list)
            # raise NotImplementedError(f"Attributes not implemented")
            # print(f'attributes={attributes}')

        result = RObject(info=info, tag=tag,
                       attributes=attributes,
                       value=value,
                       referenced_object=referenced_object)

        if add_reference:
            reference_list.append(result)

        print_R_object(result)
        return result

magic_dict = {
    FileTypes.bzip2: b"\x42\x5a\x68",
    FileTypes.rdata_binary: b"RDX2\n"
}

format_dict = {
    RdataFormats.XDR: b"X\n",
    RdataFormats.ASCII: b"A\n",
    RdataFormats.binary: b"B\n",
}


def file_type(data: memoryview):
    '''
    Returns the type of the file.
    '''

    for filetype, magic in magic_dict.items():
        if data[:len(magic)] == magic:
            return filetype
    return None


def rdata_format(data: memoryview):
    '''
    Returns the format of the data.
    '''

    for format_type, magic in format_dict.items():
        if data[:len(magic)] == magic:
            return format_type
    return None


def parse_file(file):

    try:
        path = pathlib.Path(file)
        data = path.read_bytes()
    except:
        # file is a pre-opened file
        data = file.read()
    return parse_data(data)


def parse_data(data: bytes):

    data = memoryview(data)

    filetype = file_type(data)

    if filetype is FileTypes.bzip2:
        return parse_data(bz2.decompress(data))
    elif filetype is FileTypes.rdata_binary:
        data = data[len(magic_dict[filetype]):]
        return parse_rdata_binary(data)
    else:
        raise NotImplementedError("Unknown file type")


def parse_rdata_binary(data: memoryview):
    format_type = rdata_format(data)

    if format_type:
        data = data[len(format_dict[format_type]):]

    if format_type is RdataFormats.XDR:
        parser = ParserXDR(data)
        return parser.parse_all()
    else:
        raise NotImplementedError("Unknown file format")


def bits(data, start, stop):
    count = stop - start
    mask = ((1 << count) - 1) << start

    bitvalue = data & mask
    return bitvalue >> start


def is_special_r_object_type(r_object_type: RObjectType):
    return (r_object_type is RObjectType.NILVALUE
            or r_object_type is RObjectType.REF)


def parse_r_object_info(info_int: int) -> RObjectInfo:
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

