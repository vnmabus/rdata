"""rdata: Read R datasets from Python."""
import errno as _errno
import os as _os
import pathlib as _pathlib

from . import conversion, parser


def _get_test_data_path() -> _pathlib.Path:
    return _pathlib.Path(_os.path.dirname(__file__)) / "tests" / "data"


TESTDATA_PATH = _get_test_data_path()
"""
Path of the test data.

"""

try:
    with open(
        _pathlib.Path(_os.path.dirname(__file__)) / 'VERSION',
        'r',
    ) as version_file:
        __version__ = version_file.read().strip()
except IOError as e:
    if e.errno != _errno.ENOENT:
        raise

    __version__ = "0.0"
