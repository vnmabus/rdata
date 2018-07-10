import abc
import typing
import warnings

import xarray

import numpy as np
from rdata.parser._parser import RObject

from .. import parser


def convert_list(r_list: parser.RObject,
                 conversion_function: typing.Callable=lambda x: x):
    """
    Expand a tagged R pairlist to a Python dictionary.

    Parameters
    ----------
    r_list: RObject
        Pairlist R object, with tags.
    conversion_function: Callable
        Conversion function to apply to the elements of the list. By default
        is the identity function.

    Returns
    -------
    dictionary: dict
        A dictionary with the tags of the pairwise list as keys and their
        corresponding values as values.

    See Also
    --------
    convert_vector

    """
    if r_list.info.type is parser.RObjectType.NILVALUE:
        return {}
    elif r_list.info.type is not parser.RObjectType.LIST:
        raise TypeError("Must receive a LIST or NILVALUE object")

    tag = convert_symbol(r_list.tag)

    return {tag: conversion_function(r_list.value[0]),
            **convert_list(r_list.value[1], conversion_function)}


def convert_attrs(r_obj: parser.RObject,
                  conversion_function: typing.Callable=lambda x: x):
    """
    Return the attributes of an object as a Python dictionary.

    Parameters
    ----------
    r_obj: RObject
        R object.
    conversion_function: Callable
        Conversion function to apply to the elements of the attribute list. By
        default is the identity function.

    Returns
    -------
    dictionary: dict
        A dictionary with the names of the attributes as keys and their
        corresponding values as values.

    See Also
    --------
    convert_list

    """
    if r_obj.attributes:
        attrs = convert_list(r_obj.attributes, conversion_function)
    else:
        attrs = {}
    return attrs


def convert_vector(r_vec: parser.RObject,
                   function: typing.Callable=lambda x: x):
    """
    Convert a R vector to a Python list or dictionary.

    If the vector has a ``names`` attribute, the result is a dictionary with
    the names as keys. Otherwise, the result is a Python list.

    Parameters
    ----------
    r_vec: RObject
        R vector.
    conversion_function: Callable
        Conversion function to apply to the elements of the vector. By default
        is the identity function.

    Returns
    -------
    vector: dict or list
        A dictionary with the ``names`` of the vector as keys and their
        corresponding values as values. If the vector does not have an argument
        ``names``, then a normal Python list is returned.

    See Also
    --------
    convert_list

    """
    if r_vec.info.type is not parser.RObjectType.VEC:
        raise TypeError("Must receive a VEC object")

    value = [function(o) for o in r_vec.value]

    # If it has the name attribute, use a dict instead
    attrs = convert_attrs(r_vec, function)
    field_names = attrs.get('names', None)
    if field_names:
        value = dict(zip(field_names, value))

    return value


def convert_char(r_char: parser.RObject):
    """
    Decode a R character array to a Python string or bytes.

    The bits that signal the encoding are in the general pointer. The
    string can be encoded in UTF8, LATIN1 or ASCII, or can be a sequence
    of bytes.

    Parameters
    ----------
    r_char: RObject
        R character array.

    Returns
    -------
    string: str or bytes
        Decoded string.

    See Also
    --------
    convert_symbol

    """
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


def convert_symbol(r_symbol: parser.RObject):
    """
    Decode a R symbol to a Python string or bytes.

    Parameters
    ----------
    r_symbol: RObject
        R symbol.

    Returns
    -------
    string: str or bytes
        Decoded string.

    See Also
    --------
    convert_char

    """
    if r_symbol.info.type is parser.RObjectType.SYM:
        return convert_char(r_symbol.value)
    elif r_symbol.info.type is parser.RObjectType.REF:
        return convert_symbol(r_symbol.referenced_object)
    else:
        raise TypeError("Must receive a SYM or REF object")


def convert_array(r_array: RObject,
                  conversion_function: typing.Callable=lambda x: x,
                  attrs: dict=None):
    """
    Convert a R array to a Numpy ndarray or a Xarray DataArray.

    If the array has attribute ``dimnames`` the output will be a
    Xarray DataArray, preserving the dimension names.

    Parameters
    ----------
    r_array: RObject
        R array.
    conversion_function: Callable
        Conversion function to apply to the attributes of the array.
        By default is the identity function.

    Returns
    -------
    array: ndarray or DataArray
        Array.

    See Also
    --------
    convert_vector

    """
    if attrs is None:
        attrs = {}

    if r_array.info.type not in {parser.RObjectType.INT,
                                 parser.RObjectType.REAL}:
        raise TypeError("Must receive an array object")

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


class Converter(abc.ABC):

    @abc.abstractmethod
    def convert(self, data: typing.Union[parser.RData, parser.RObject]):
        pass


class SimpleConverter(Converter):

    def __init__(self, constructor_dict=None):
        self.references = {}

        self.constructor_dict = ({} if constructor_dict is None
                                 else constructor_dict)

    def convert(self, data: typing.Union[parser.RData, parser.RObject]):
        """
        Convert a R object to a Python one.
        """
        if isinstance(data, parser.RData):
            obj: RObject = data.object
        else:
            obj: RObject = data

        attrs = convert_attrs(obj, self.convert)

        if obj.info.type == parser.RObjectType.SYM:

            # Return the internal string
            value = convert_char(obj.value)

        elif obj.info.type == parser.RObjectType.LIST:

            # Expand the list and process the elements
            value = convert_list(obj, self.convert)

        elif obj.info.type == parser.RObjectType.CHAR:

            # Return the internal string
            value = convert_char(obj)

        elif obj.info.type == parser.RObjectType.INT:

            # Return the internal array
            value = convert_array(obj, self.convert, attrs=attrs)

        elif obj.info.type == parser.RObjectType.REAL:

            # Return the internal array
            value = convert_array(obj, self.convert, attrs=attrs)

        elif obj.info.type == parser.RObjectType.STR:

            # Convert the internal strings
            value = [self.convert(o) for o in obj.value]

        elif obj.info.type == parser.RObjectType.VEC:

            # Convert the internal objects
            value = convert_vector(obj, self.convert)

            return value

        elif obj.info.type == parser.RObjectType.REF:

            # Return the referenced value
            value = self.references[id(obj.referenced_object)]

        elif obj.info.type == parser.RObjectType.NILVALUE:

            return None

        else:
            raise NotImplementedError(f"Type {obj.info.type} not implemented")

        if obj.info.object:
            classname = attrs["class"]
            assert len(classname) == 1
            classname = classname[0]

            constructor = self.constructor_dict.get(classname, None)

            if constructor:
                new_obj = constructor(obj, attrs)
            else:
                new_obj = NotImplemented

            if new_obj is NotImplemented:
                warnings.warn(f"Missing constructor for R class "
                              f"\"{classname}\". "
                              f"The underlying R object is returned instead",
                              stacklevel=1)
            else:
                obj = new_obj

        self.references[id(obj)] = value

        return value
