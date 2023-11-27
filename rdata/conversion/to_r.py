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


def build_r_object(r_type, *, value=None, attributes=None, tag=None, gp=0):
    assert r_type is not None
    r_object = RObject(RObjectInfo(r_type,
                                   object=False,
                                   attributes=attributes is not None,
                                   tag=tag is not None,
                                   gp=gp,
                                   reference=0),
                       value,
                       attributes,
                       tag,
                       None)
    return r_object


def build_r_list(key, value):
    r_list = build_r_object(
        RObjectType.LIST,
        value=[
            value,
            build_r_object(RObjectType.NILVALUE),
            ],
        tag=build_r_object(
            RObjectType.SYM,
            value=key,
            ),
        )
    return r_list


class Converter():

    def __init__(self, encoding='UTF-8'):
        assert encoding in ['UTF-8', 'CP1252']
        self.encoding = encoding

    def convert_to_robject(self, data) -> RObject:
        # Default args for most types (None/False/0)
        r_type = None
        r_value = None
        gp = 0
        attributes = None
        tag = None
        referenced_object = None

        if data is None:
            r_type = RObjectType.NILVALUE

        elif isinstance(data, (list, dict)):
            r_type = RObjectType.VEC
            r_value = []
            if isinstance(data, dict):
                values = data.values()
            else:
                values = data
            for element in values:
                r_value.append(self.convert_to_robject(element))

            if isinstance(data, dict):
                attributes = build_r_list(
                    self.convert_to_robject(b'names'),
                    self.convert_to_robject(np.array(list(data.keys()))),
                    )

        elif isinstance(data, np.ndarray):
            if data.dtype.kind in ['S']:
                assert data.ndim == 1
                r_type = RObjectType.STR
                r_value = []
                for element in data:
                    r_value.append(self.convert_to_robject(element))

            elif data.dtype.kind in ['U']:
                assert data.ndim == 1
                data = np.array([s.encode(self.encoding) for s in data])
                return self.convert_to_robject(data)

            else:
                r_type = {
                    'b': RObjectType.LGL,
                    'i': RObjectType.INT,
                    'f': RObjectType.REAL,
                    'c': RObjectType.CPLX,
                }[data.dtype.kind]

                if data.ndim == 1:
                    r_value = data
                elif data.ndim == 2:
                    # R uses column-major order like Fortran
                    r_value = np.ravel(data, order='F')
                    attributes = build_r_list(
                        self.convert_to_robject(b'dim'),
                        self.convert_to_robject(np.array(data.shape)),
                        )
                else:
                    raise NotImplementedError(f"ndim={data.ndim}")

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
            r_value = data

        else:
            raise NotImplementedError(f"{type(data)}")

        return build_r_object(r_type, value=r_value, attributes=attributes, tag=tag, gp=gp)

    def convert_to_rdata(self, data) -> RData:
        versions = RVersions(3, 262657, 197888)
        extra = RExtraInfo(self.encoding)
        obj = self.convert_to_robject(data)
        return RData(versions, extra, obj)
