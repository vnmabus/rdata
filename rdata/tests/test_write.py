import io
import pytest

import rdata
import rdata.io


TESTDATA_PATH = rdata.TESTDATA_PATH


def decompress_data(data):
    from rdata.parser._parser import file_type, FileTypes

    filetype = file_type(data)

    if filetype is FileTypes.bzip2:
        from bz2 import decompress
    elif filetype is FileTypes.gzip:
        from gzip import decompress
    elif filetype is FileTypes.xz:
        from xz import decompress
    else:
        decompress = lambda x: x

    return decompress(data)


@pytest.mark.parametrize("fname", [
    "test_vector.rda",
    "test_full_named_matrix.rds",
    "test_ascii_v2.rda",
    ])
def test_write(fname):
    with (TESTDATA_PATH / fname).open("rb") as f:
        data = decompress_data(f.read())
        rds = data[:2] != b'RD'
        format = 'ascii' if data.isascii() else 'xdr'

        r_data = rdata.parser.parse_data(data)

        if format == 'ascii':
            fd = io.StringIO()
            data = data.decode('ascii')
        else:
            fd = io.BytesIO()
        rdata.io.write_file(fd, r_data, format=format, rds=rds)

        assert data == fd.getvalue()
