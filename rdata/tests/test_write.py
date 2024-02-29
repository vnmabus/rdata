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
    ])
def test_write(fname):
    with (TESTDATA_PATH / fname).open("rb") as f:
        data = decompress_data(f.read())
        rds = data[:2] != b'RD'

        r_data = rdata.parser.parse_data(data)

        ofpath = 'output.rda'
        rdata.io.write(ofpath, r_data, format='xdr', compression='none', rds=rds)

        with open(ofpath, 'rb') as ff:
            out_data = ff.read()

        assert data == out_data




