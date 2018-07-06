import typing

import xarray

import numpy as np
from rdata.parser._parser import RObject

from . import parser


def expand_list(r_list: parser.RObject, function: typing.Callable=lambda x: x):
    if r_list.info.type is parser.RObjectType.NILVALUE:
        return {}
    elif r_list.info.type is not parser.RObjectType.LIST:
        raise TypeError("Must receive a LIST or NILVALUE object")

    tag = decode_tag(r_list.tag)

    return {tag: function(r_list.value[0]),
            **expand_list(r_list.value[1], function)}


def expand_attrs(attrs, function):
    return expand_list(attrs, function)


def expand_vector(r_vec: parser.RObject,
                  function: typing.Callable=lambda x: x):
    if r_vec.info.type is not parser.RObjectType.VEC:
        raise TypeError("Must receive a VEC object")

    value = [function(o) for o in r_vec.value]

    # If it has the name attribute, use a dict instead
    if r_vec.attributes:
        attrs = expand_attrs(r_vec.attributes, function)
        field_names = attrs.get('names', None)
        if field_names:
            value = dict(zip(field_names, value))

    return value


def decode_char(r_char: parser.RObject):
    if r_char.info.type is not parser.RObjectType.CHAR:
        raise TypeError("Must receive a CHAR object")

    if r_char.info.gp & parser.CharFlags.UTF8:
        return r_char.value.decode("utf_8")
    elif r_char.info.gp & parser.CharFlags.LATIN1:
        return r_char.value.decode("latin_1")
    elif r_char.info.gp & parser.CharFlags.ASCII:
        return r_char.value.decode("ascii")
    elif r_char.info.gp & parser.CharFlags.BYTES:
        return r_char.value
    else:
        raise NotImplementedError("Encoding not implemented")


def decode_tag(r_symbol: parser.RObject):
    if r_symbol.info.type is parser.RObjectType.SYM:
        return decode_char(r_symbol.value)
    elif r_symbol.info.type is parser.RObjectType.REF:
        return decode_tag(r_symbol.referenced_object)
    else:
        raise TypeError("Must receive a SYM or REF object")


def array_to_xarray(r_array: RObject):
    if r_array.info.type not in {parser.RObjectType.INT,
                                 parser.RObjectType.REAL}:
        raise TypeError("Must receive an array object")

    if r_array.attributes:
        attrs = expand_attrs(r_array.attributes, xarray_conversion)
    else:
        attrs = {}

    value = r_array.value

    shape = attrs.get('dim', None)
    if shape is not None:
        value = np.reshape(value, shape)

    dimnames = attrs.get('dimnames', None)
    if dimnames:
        dimension_names = ["dim_" + str(i) for i, _ in enumerate(dimnames)]
        coords = {dimension_names[i]: d
                  for i, d in enumerate(dimnames) if d is not None}

        value = xarray.DataArray(value, dims=dimension_names, coords=coords)

    return value


def xarray_conversion(data: typing.Union[parser.RData, parser.RObject],
                      references=None):
    if isinstance(data, parser.RData):
        obj: RObject = data.object
    else:
        obj: RObject = data

    if references is None:
        references = {}

    if obj.info.type == parser.RObjectType.SYM:

        # Return the internal string
        value = decode_char(obj.value)

    elif obj.info.type == parser.RObjectType.LIST:

        # Expand the list and process the elements
        value = expand_list(obj, xarray_conversion)

    elif obj.info.type == parser.RObjectType.CHAR:

        # Return the internal string
        value = decode_char(obj)

    elif obj.info.type == parser.RObjectType.INT:

        # Return the internal array
        value = array_to_xarray(obj)

    elif obj.info.type == parser.RObjectType.REAL:

        # Return the internal array
        value = array_to_xarray(obj)

    elif obj.info.type == parser.RObjectType.STR:

        # Convert the internal strings
        value = [xarray_conversion(o) for o in obj.value]

    elif obj.info.type == parser.RObjectType.VEC:

        # Convert the internal strings
        value = expand_vector(obj, xarray_conversion)

        return value

    elif obj.info.type == parser.RObjectType.REF:

        # Return the referenced value
        value = references[id(obj.referenced_object)]

    elif obj.info.type == parser.RObjectType.NILVALUE:

        return None

    else:
        raise NotImplementedError(f"Type {obj.info.type} not implemented")

    references[id(obj)] = value

    return value

