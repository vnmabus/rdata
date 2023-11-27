import string
import numpy as np

from rdata.parser._parser import (
    CharFlags,
    RData,
    RExtraInfo,
    RObject,
    RObjectInfo,
    RObjectType,
    RVersions,
)


class Converter():

    def __init__(self, encoding='UTF-8'):
        assert encoding in ['UTF-8', 'CP1252']
        self.encoding = encoding

    def convert_list(self, values):
        r_value = []
        for value in values:
            r_value.append(self.convert_to_robject(value))
        return r_value

    def convert_array(self, array):
        if array.ndim != 1:
            raise NotImplementedError(f"array ndim={array.ndim}")
        return array

    def convert_to_robject(self, data) -> RObject:
        # Default args for most types (None/False/0)
        r_info_kwargs = dict(
            object=False,
            attributes=False,
            tag=False,
            gp=0,
            reference=0,
        )
        attributes = None
        tag = None
        referenced_object = None

        if isinstance(data, list):
            r_type = RObjectType.VEC
            r_value = self.convert_list(data)

        elif isinstance(data, np.ndarray):
            if data.dtype.kind in ['U', 'S']:
                assert data.size == 1
                return self.convert_to_robject(data[0])

            r_type = {
                'b': RObjectType.LGL,
                'i': RObjectType.INT,
                'f': RObjectType.REAL,
                'c': RObjectType.CPLX,
            }[data.dtype.kind]
            r_value = self.convert_array(data)

        elif isinstance(data, str):
            r_type = RObjectType.STR
            r_value = [self.convert_to_robject(data.encode(self.encoding))]

        elif isinstance(data, bytes):
            r_type = RObjectType.CHAR
            if all(chr(byte) in string.printable for byte in data):
                gp = CharFlags.ASCII
            elif self.encoding == 'UTF-8':
                gp = CharFlags.UTF8
            elif self.encoding == 'CP1252':
                # XXX CP1252 and Latin1 are not the same
                #     Check if CharFlags.LATIN1 means actually CP1252
                #     as R on Windows mentions CP1252 as encoding
                gp = CharFlags.LATIN1
            else:
                raise NotImplementedError("unknown what gp value to use")
            r_info_kwargs.update(gp=gp)
            r_value = data

        else:
            raise NotImplementedError(f"{type(data)}")

        r_info = RObjectInfo(r_type, **r_info_kwargs)
        return RObject(r_info, r_value, attributes, tag, referenced_object)

    def convert_to_rdata(self, data) -> RData:
        versions = RVersions(3, 262657, 197888)
        extra = RExtraInfo(self.encoding)
        obj = self.convert_to_robject(data)
        return RData(versions, extra, obj)
