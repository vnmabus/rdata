"""Tests of writing and Python-to-R conversion."""

from __future__ import annotations

import io
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import rdata
import rdata.io

if TYPE_CHECKING:
    from collections.abc import Generator

TESTDATA_PATH = rdata.TESTDATA_PATH

valid_compressions = ["none", "bzip2", "gzip", "xz"]
valid_formats = ["xdr", "ascii"]


@contextmanager
def no_error() -> Generator[Any, Any, Any]:
    """Context manager that does nothing but returns no_error.

    This context manager can be used like pytest.raises()
    when no error is expected.
    """
    yield no_error


def decompress_data(data: memoryview) -> bytes:
    """Decompress bytes."""
    from rdata.parser._parser import FileTypes, file_type

    filetype = file_type(data)

    if filetype is FileTypes.bzip2:
        from bz2 import decompress
    elif filetype is FileTypes.gzip:
        from gzip import decompress
    elif filetype is FileTypes.xz:
        from lzma import decompress
    else:
        return data

    return decompress(data)


fnames = sorted([fpath.name for fpath in TESTDATA_PATH.glob("*.rd?")])

@pytest.mark.parametrize("fname", fnames, ids=fnames)
def test_write(fname: str) -> None:
    """Test writing RData object to a file."""
    with (TESTDATA_PATH / fname).open("rb") as f:
        data = decompress_data(f.read())
        rds = data[:2] != b"RD"
        fmt = "ascii" if data.isascii() else "xdr"

        r_data = rdata.parser.parse_data(data, expand_altrep=False)

        fd = io.BytesIO()
        try:
            rdata.io.write_file(fd, r_data, file_format=fmt, rds=rds)
        except NotImplementedError as e:
            pytest.xfail(str(e))

        out_data = fd.getvalue()

        if fmt == "ascii":
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
        rds = data[:2] != b"RD"

        r_data = rdata.parser.parse_data(data, expand_altrep=False)

        try:
            py_data = rdata.conversion.convert(r_data)
        except NotImplementedError as e:
            pytest.skip(str(e))

        encoding = r_data.extra.encoding
        if encoding is None:
            encoding = "CP1252" if "win" in fname else "UTF-8"

        try:
            new_r_data = rdata.conversion.convert_to_r_data(
                py_data, rds=rds, versions=r_data.versions, encoding=encoding,
                )
        except NotImplementedError as e:
            pytest.xfail(str(e))

        assert r_data == new_r_data
        assert str(r_data) == str(new_r_data)


def test_convert_to_r_bad_rda() -> None:
    """Test checking that data for RDA has variable names."""
    py_data = "hello"
    with pytest.raises(ValueError, match="(?i)data must be a dictionary"):
        rdata.conversion.convert_to_r_data(py_data, rds=False)


def test_convert_to_r_bad_encoding() -> None:
    """Test checking encoding."""
    with pytest.raises(LookupError, match="(?i)unknown encoding"):
        rdata.conversion.convert_to_r_data("ä", encoding="non-existent")


def test_convert_to_r_unsupported_encoding() -> None:
    """Test checking encoding."""
    with pytest.raises(ValueError, match="(?i)unsupported encoding"):
        rdata.conversion.convert_to_r_data("ä", encoding="CP1250")


@pytest.mark.parametrize("compression", [*valid_compressions, None, "fail"])
@pytest.mark.parametrize("fmt", [*valid_formats, None, "fail"])
@pytest.mark.parametrize("rds", [True, False])
def test_write_real_file(compression: str, fmt: str, rds: bool) -> None:  # noqa: FBT001
    """Test writing RData object to a real file with compression."""
    expectation = no_error()
    if fmt not in valid_formats:
        expectation = pytest.raises(ValueError, match="(?i)unknown file format")  # type: ignore [assignment]
    if compression not in valid_compressions:
        expectation = pytest.raises(ValueError, match="(?i)unknown compression")  # type: ignore [assignment]

    py_data = {"key": "Hello", "none": None}
    suffix = ".rds" if rds else ".rda"
    read = rdata.read_rds if rds else rdata.read_rda
    write = rdata.write_rds if rds else rdata.write_rda
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / f"file{suffix}"

        with expectation as status:
            write(fpath, py_data, file_format=fmt, compression=compression)

        if status is no_error:
            assert py_data == read(fpath)
