import os as _os
import pathlib as _pathlib

from . import conversion, parser


def _get_test_data_path() -> _pathlib.Path:
    return _pathlib.Path(_os.path.dirname(__file__)) / "tests" / "data"


TESTDATA_PATH = _get_test_data_path()
"""
Path of the test data.

"""
