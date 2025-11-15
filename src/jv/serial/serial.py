from dataclasses import fields, is_dataclass
from ctypes import c_uint8, c_double, c_float
from typing import Callable, get_origin, Optional, Union, Any
import torch
import numpy as np
import struct


DTYPE_TO_NUM: dict[str, int] = {
    "bool": 0,
    "int8": 1,
    "uint8": 2,
    "int16": 3,
    "uint16": 4,
    "int32": 5,
    "uint32": 6,
    "int64": 7,
    "uint64": 8,
    "float16": 9,
    "float32": 10,
    "float64": 11,
    "complex64": 12,
    "complex128": 13,
}


def format_single_char(c: str) -> tuple[str, str]:
    return ("c", c)


def format_single_int(i: int) -> tuple[str, int]:
    return ("i", i)


def format_list(lst: list) -> tuple[str, list]:
    format = ""
    values = []

    for item in lst:

        fmt, val = format_single_char("|")
        format += fmt
        values.extend([b"|"])

        if is_dataclass(item):
            fmt, val = get_format_and_values_dataclass(item)
            format += fmt
            values.extend(val)
        else:

            try:
                fmt, val = get_format_and_values(type(item), item)
                format += fmt
                values.extend(val)
            except KeyError:
                raise Exception(f"Item of type {type(item)} not supported.")

    return format, values


def unique_format(object):

    format = ""
    value = []

    if isinstance(object, list):

        # Add `^` to indicate start of the list
        indic_fmt, indic_val = format_single_char("^")
        format += indic_fmt
        value.extend([indic_val.encode()])

        # Add the number of elements
        len_fmt, len_val = format_single_int(len(object))
        format += len_fmt
        value.extend([len_val])

        # Add list elements separated by `|`
        fmt, val = format_list(object)
        format += fmt
        value.extend(val)

        # Add `^` to indicate end of the list
        format += indic_fmt
        value.extend([indic_val.encode()])

    elif isinstance(object, torch.Tensor) or isinstance(object, np.ndarray):

        if isinstance(object, torch.Tensor):
            object = object.numpy()

        # Add `$` to indicate start of array
        indic_fmt, indic_val = format_single_char("$")
        format += indic_fmt
        value.extend([indic_val.encode()])

        # Add type information
        type_fmt, type_val = format_single_int(DTYPE_TO_NUM[str(object.dtype)])
        format += type_fmt
        value.extend([type_val])

        # Add shape information
        paren_fmt, paren_val = format_single_char("(")
        format += paren_fmt
        value.extend([paren_val.encode()])

        for dim in object.shape:
            type_fmt, type_val = format_single_int(dim)
            format += type_fmt
            value.extend([type_val])

        paren_fmt, paren_val = format_single_char(")")
        format += paren_fmt
        value.extend([paren_val.encode()])

        # Add byte string representation of array
        b_string = object.tobytes()
        format += f"{len(b_string)}s"
        value.extend([b_string])

        # Add `$` to indicate end of array
        format += indic_fmt
        value.extend([indic_val.encode()])

    else:
        raise Exception(f"Unique object {type(object)} is not supported.")

    return format, value


TYPE_TO_FORMAT_STR: dict[type, Callable] = {
    int: lambda o: ('i', [o]),
    c_uint8: lambda o: ("H", [o]),
    float: lambda o: ('d', [o]),
    c_float: lambda o: ("f", [o]),
    c_double: lambda o: ("d", [o]),
    bool: lambda o: ('?', [o]),
    str: lambda s: (f'{len(s)}s', [s.encode('utf-8')]),
    list: lambda lst: unique_format(lst),
    torch.Tensor: lambda a: unique_format(a),
    np.ndarray: lambda a: unique_format(a),
}


def get_format_and_values(typ, object) -> tuple[str, Any]:

    if get_origin(typ) is list:
        return TYPE_TO_FORMAT_STR[list](object)
    elif get_origin(typ) is Optional or get_origin(typ) is Union:
        raise Exception("Union or optional types are not allowed.")

    return TYPE_TO_FORMAT_STR[typ](object)


def get_format_and_values_dataclass(dc) -> tuple[str, list]:

    format = ""
    values = []

    for field in fields(dc):

        typ, name = field.type, field.name
        object = getattr(dc, name)

        if is_dataclass(typ):
            serialize_dataclass(field)
        else:
            try:
                fmt, val = get_format_and_values(typ, object)
                format += fmt
                values.extend(val)
            except Exception as e:
                raise Exception(f"Unhandled type from dataclass: {typ}, {e}")

    return (format, values)


def serialize_dataclass(dc) -> bytes:
    """
    Serializes a dataclass by converting every unique type into an object.
    The dataclass can be specified with ctypes.

    Given the dataclass definition it will generate the pack format, and
    then split the dataclass into its unique objects so it can be passed into
    pack. Dataclasses with Optional types are not allowed.

    For iterables, the prescence and end is denoted with `^`, and then the number
    of items, then items are added with a delimiter before them `|`.

    For Tensor or ndarrays, the prescence and end is denoted with `$`, and then
    the type is given with a unique integer id from DTYPE_TO_NUM, and the shape
    is given with parentheses and 32-bit integers for shapes `(x1x2xn)`.
    Tensors or ndarrays are converted into byte strings. For a Tensor this first
    requires converting to an ndarray.

    Mappings are provided in `TYPE_TO_FORMAT_STR`.
    """

    format, values = get_format_and_values_dataclass(dc)
    return struct.pack("="+format, *values)  # use native but no padding
