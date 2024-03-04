"""Tests of writing and Python-to-R conversion."""

from __future__ import annotations

import io

import pytest

import rdata
import rdata.io

TESTDATA_PATH = rdata.TESTDATA_PATH


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
        data: bytes | str
        fd: io.BytesIO | io.StringIO
        data = decompress_data(f.read())
        rds = data[:2] != b"RD"
        fmt = "ascii" if data.isascii() else "xdr"

        r_data = rdata.parser.parse_data(data, expand_altrep=False)

        if fmt == "ascii":
            fd = io.StringIO()
            data = data.decode("ascii")
            data = data.replace("\r\n", "\n")
        else:
            fd = io.BytesIO()

        try:
            rdata.io.write_file(fd, r_data, format=fmt, rds=rds)
        except NotImplementedError as e:
            pytest.xfail(str(e))

        out_data = fd.getvalue()

        if data != out_data:
            print(r_data)
            print("in")
            print(data)
            print("out")
            print(out_data)

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

        if r_data != new_r_data:
            print("ref")
            print(r_data)
            print("py")
            print(py_data)
            print("new")
            print(new_r_data)

        assert r_data == new_r_data
        assert str(r_data) == str(new_r_data)
