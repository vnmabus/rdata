"""Tests of writing, unparsing, and Python-to-R conversion."""

from __future__ import annotations

import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import rdata
from rdata.unparser import unparse_data

if TYPE_CHECKING:
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
def test_convert_to_r(fname: str) -> None:
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

        r_data = rdata.parser.parse_data(data, expand_altrep=False)

        try:
            py_data = rdata.conversion.convert(r_data)
        except NotImplementedError as e:
            pytest.skip(str(e))

        encoding = r_data.extra.encoding
        if encoding is None:
            encoding = "CP1252" if "win" in fname else "UTF-8"

        try:
            if file_type == "rds":
                r_obj = rdata.conversion.convert_to_r_object(
                    py_data, encoding=encoding)
            else:
                r_obj = rdata.conversion.convert_to_r_object_for_rda(
                    py_data, encoding=encoding)
            new_r_data = rdata.conversion.build_r_data(
                r_obj,
                encoding=encoding,
                format_version=r_data.versions.format,
                r_version_serialized=r_data.versions.serialized,
            )
        except NotImplementedError as e:
            pytest.xfail(str(e))

        assert r_data == new_r_data
        assert str(r_data) == str(new_r_data)


def test_convert_to_r_bad_rda() -> None:
    """Test checking that data for RDA has variable names."""
    py_data = "hello"
    with pytest.raises(TypeError, match="(?i)data must be a dictionary"):
        rdata.conversion.convert_to_r_object_for_rda(py_data)  # type: ignore [arg-type]


def test_convert_to_r_empty_rda() -> None:
    """Test checking that data for RDA has variable names."""
    py_data: dict[str, Any] = {}
    with pytest.raises(ValueError, match="(?i)data must not be empty"):
        rdata.conversion.convert_to_r_object_for_rda(py_data)


def test_unparse_bad_rda() -> None:
    """Test checking that data for RDA has variable names."""
    py_data = "hello"
    r_obj = rdata.conversion.convert_to_r_object(py_data)
    r_data = rdata.conversion.build_r_data(r_obj)
    with pytest.raises(ValueError, match="(?i)must be dictionary-like"):
        unparse_data(r_data, file_type="rda")


def test_convert_to_r_bad_encoding() -> None:
    """Test checking encoding."""
    with pytest.raises(LookupError, match="(?i)unknown encoding"):
        rdata.conversion.convert_to_r_object("ä", encoding="non-existent")


def test_convert_to_r_unsupported_encoding() -> None:
    """Test checking encoding."""
    with pytest.raises(ValueError, match="(?i)unsupported encoding"):
        rdata.conversion.convert_to_r_object("ä", encoding="CP1250")


def test_unparse_big_int() -> None:
    """Test checking too large integers."""
    big_int = 2**32
    r_obj = rdata.conversion.convert_to_r_object(big_int)
    r_data = rdata.conversion.build_r_data(r_obj)
    with pytest.raises(ValueError, match="(?i)not castable"):
        unparse_data(r_data, file_format="xdr")


@pytest.mark.parametrize("compression", [*valid_compressions, "fail"])
@pytest.mark.parametrize("file_format", [*valid_formats, None, "fail"])
@pytest.mark.parametrize("file_type", ["rds", "rda"])
def test_write_file(
    compression: Compression,
    file_format: FileFormat,
    file_type: FileType,
) -> None:
    """Test writing RData object to a real file with compression."""
    expectation = nullcontext()
    if file_format not in valid_formats:
        expectation = pytest.raises(ValueError, match="(?i)unknown file format")  # type: ignore [assignment]
    if compression not in valid_compressions:
        expectation = pytest.raises(ValueError, match="(?i)unknown compression")  # type: ignore [assignment]

    py_data = {"key": "Hello", "none": None}
    suffix = ".rds" if file_type == "rds" else ".rda"
    read = rdata.read_rds if file_type == "rds" else rdata.read_rda
    write = rdata.write_rds if file_type == "rds" else rdata.write_rda
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / f"file{suffix}"

        with expectation:
            write(fpath, py_data, file_format=file_format, compression=compression)
            assert py_data == read(fpath)
