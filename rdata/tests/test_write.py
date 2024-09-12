"""Tests of writing, unparsing, and Python-to-R conversion."""

from __future__ import annotations

import tempfile
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

import rdata
from rdata.conversion import ConverterFromPythonToR
from rdata.unparser import unparse_data

if TYPE_CHECKING:
    from rdata.conversion.to_r import Encoding
    from rdata.unparser import Compression, FileFormat, FileType


TESTDATA_PATH = rdata.TESTDATA_PATH

valid_compressions = [None, "bzip2", "gzip", "xz"]
valid_formats = ["xdr", "ascii"]


def decompress_data(data: bytes) -> bytes:
    """Decompress bytes."""
    from rdata.parser._parser import FileTypes, file_type

    filetype = file_type(memoryview(data))

    if filetype is FileTypes.bzip2:
        from bz2 import decompress
    elif filetype is FileTypes.gzip:
        from gzip import decompress
    elif filetype is FileTypes.xz:
        from lzma import decompress
    else:
        return data

    return decompress(data)


fnames = sorted([fpath.name for fpath in Path(str(TESTDATA_PATH)).glob("*.rd?")])

def parse_file_type_and_format(data: bytes) -> tuple[FileType, FileFormat]:
    """Parse file type and format from data."""
    from rdata.parser._parser import (
        FileTypes,
        RdataFormats,
        file_type,
        magic_dict,
        rdata_format,
    )
    view = memoryview(data)

    file_type_str: FileType
    file_format_str: FileFormat

    filetype = file_type(view)
    if filetype in {
            FileTypes.rdata_binary_v2,
            FileTypes.rdata_binary_v3,
            FileTypes.rdata_ascii_v2,
            FileTypes.rdata_ascii_v3,
            }:
        file_type_str = "rda"
        magic = magic_dict[filetype]
        view = view[len(magic):]
    else:
        file_type_str = "rds"

    rdataformat = rdata_format(view)
    file_format_str = "xdr" if rdataformat is RdataFormats.XDR else "ascii"

    return file_type_str, file_format_str


@pytest.mark.parametrize("fname", fnames, ids=fnames)
def test_unparse(fname: str) -> None:
    """Test unparsing RData object to a file."""
    with (TESTDATA_PATH / fname).open("rb") as f:
        data = decompress_data(f.read())
        file_type, file_format = parse_file_type_and_format(data)
        r_data = rdata.parser.parse_data(data, expand_altrep=False)

        try:
            out_data = unparse_data(
                r_data, file_format=file_format, file_type=file_type)
        except NotImplementedError as e:
            pytest.xfail(str(e))

        if file_format == "ascii":
            data = data.replace(b"\r\n", b"\n")

        assert data == out_data


@pytest.mark.parametrize("fname", fnames, ids=fnames)
@pytest.mark.parametrize("expand_altrep", [True, False])
def test_convert_to_r(fname: str, expand_altrep: bool) -> None:  # noqa: FBT001
    """Test converting Python data to RData object."""
    with (TESTDATA_PATH / fname).open("rb") as f:
        # Skip test files without unique R->py->R transformation
        if fname in [
            "test_encodings.rda",     # encoding not kept in Python
            "test_encodings_v3.rda",  # encoding not kept in Python
            "test_list_attrs.rda",    # attributes not kept in Python
            "test_file.rda",          # attributes not kept in Python
        ]:
            pytest.skip("ambiguous R->py->R transformation")

        data = decompress_data(f.read())
        file_type, file_format = parse_file_type_and_format(data)

        r_data = rdata.parser.parse_data(data, expand_altrep=expand_altrep)

        try:
            py_data = rdata.conversion.convert(r_data)
        except NotImplementedError as e:
            pytest.skip(str(e))

        encoding: Encoding
        encoding = r_data.extra.encoding  # type: ignore [assignment]
        if encoding is None:
            encoding = "cp1252" if "win" in fname else "utf-8"
        else:
            encoding = encoding.lower()  # type: ignore [assignment]

        converter = ConverterFromPythonToR(
            encoding=encoding,
            format_version=r_data.versions.format,
            r_version_serialized=r_data.versions.serialized,
        )
        if fname in [
            "test_dataframe_dtypes.rds",
            "test_dataframe_int_rownames.rds",
            "test_dataframe_range_rownames.rds",
            "test_dataframe_rownames.rda",
        ]:
            converter.df_attr_order = ["names", "class", "row.names"]

        try:
            if file_type == "rds":
                r_obj = converter.convert_to_r_object(py_data)
            else:
                r_obj = converter.convert_to_r_object_for_rda(py_data)
        except NotImplementedError as e:
            pytest.xfail(str(e))

        new_r_data = converter.build_r_data(r_obj)

        assert str(r_data) == str(new_r_data)
        assert r_data == new_r_data

        # Check futher that the resulting unparsed data is correct to ensure that
        # Python-to-R conversion hasn't created any odd objects that can't be unparsed
        if not expand_altrep:
            file_type, file_format = parse_file_type_and_format(data)
            out_data = unparse_data(
                new_r_data, file_format=file_format, file_type=file_type)

            if file_format == "ascii":
                data = data.replace(b"\r\n", b"\n")

            assert data == out_data


def test_convert_to_r_bad_rda() -> None:
    """Test checking that data for RDA has variable names."""
    py_data = "hello"
    converter = ConverterFromPythonToR()
    with pytest.raises(TypeError, match="(?i)data must be a dictionary"):
        converter.convert_to_r_object_for_rda(py_data)  # type: ignore [arg-type]


def test_convert_to_r_empty_rda() -> None:
    """Test checking that data for RDA has variable names."""
    py_data: dict[str, Any] = {}
    converter = ConverterFromPythonToR()
    with pytest.raises(ValueError, match="(?i)data must not be empty"):
        converter.convert_to_r_object_for_rda(py_data)


def test_unparse_bad_rda() -> None:
    """Test checking that data for RDA has variable names."""
    py_data = "hello"
    converter = ConverterFromPythonToR()
    r_obj = converter.convert_to_r_object(py_data)
    r_data = converter.build_r_data(r_obj)
    with pytest.raises(ValueError, match="(?i)must be dictionary-like"):
        unparse_data(r_data, file_type="rda")


def test_convert_to_r_bad_encoding() -> None:
    """Test checking encoding."""
    converter = ConverterFromPythonToR(encoding="non-existent")  # type: ignore [arg-type]
    with pytest.raises(LookupError, match="(?i)unknown encoding"):
        converter.convert_to_r_object("ä")


def test_convert_to_r_unsupported_encoding() -> None:
    """Test checking encoding."""
    converter = ConverterFromPythonToR(encoding="cp1250")  # type: ignore [arg-type]
    with pytest.raises(ValueError, match="(?i)unsupported encoding"):
        converter.convert_to_r_object("ä")


def test_unparse_big_int() -> None:
    """Test checking too large integers."""
    big_int = 2**32
    converter = ConverterFromPythonToR()
    r_obj = converter.convert_to_r_object(big_int)
    r_data = converter.build_r_data(r_obj)
    with pytest.raises(ValueError, match="(?i)not castable"):
        unparse_data(r_data, file_format="xdr")


def test_convert_dataframe_pandas_dtypes() -> None:
    """Test converting dataframe with pandas dtypes."""
    df1 = pd.DataFrame(
        {
            "int": np.array([10, 20, 30], dtype=np.int32),
            "float": [1.1, 2.2, 3.3],
            "string": ["x" ,"y", "z"],
            "bool": [True, False, True],
            "complex": [4+5j, 6+7j, 8+9j],
        },
        index=range(3),
    )

    df2 = pd.DataFrame(
        {
            "int": pd.Series([10, 20, 30], dtype=pd.Int32Dtype()),
            "float": pd.Series([1.1, 2.2, 3.3], dtype=pd.Float64Dtype()),
            "string": pd.Series(["x" ,"y", "z"], dtype=pd.StringDtype()),
            "bool": pd.Series([True, False, True], dtype=pd.BooleanDtype()),
            "complex": pd.Series([4+5j, 6+7j, 8+9j], dtype=complex),
        },
        index=pd.RangeIndex(3),
    )

    r_obj1 = ConverterFromPythonToR().convert_to_r_object(df1)
    r_obj2 = ConverterFromPythonToR().convert_to_r_object(df2)

    assert str(r_obj1) == str(r_obj2)
    assert r_obj1 == r_obj2


@pytest.mark.parametrize("compression", [*valid_compressions, "fail"])
@pytest.mark.parametrize("file_format", [*valid_formats, None, "fail"])
@pytest.mark.parametrize("file_type", ["rds", "rda"])
def test_write_file(
    compression: Compression,
    file_format: FileFormat,
    file_type: FileType,
) -> None:
    """Test writing RData object to a real file with compression."""
    expectation: AbstractContextManager[Any] = nullcontext()
    if file_format not in valid_formats:
        expectation = pytest.raises(ValueError, match="(?i)unknown file format")
    if compression not in valid_compressions:
        expectation = pytest.raises(ValueError, match="(?i)unknown compression")

    py_data = {"key": "Hello", "none": None}
    suffix = ".rds" if file_type == "rds" else ".rda"
    read = rdata.read_rds if file_type == "rds" else rdata.read_rda
    write = rdata.write_rds if file_type == "rds" else rdata.write_rda
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / f"file{suffix}"

        with expectation:
            write(fpath, py_data, file_format=file_format, compression=compression)
            assert py_data == read(fpath)
