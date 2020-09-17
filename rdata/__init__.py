from . import conversion
from . import parser


def _get_test_data_path():
    import os
    import pathlib

    return pathlib.Path(os.path.dirname(__file__)) / "tests" / "data"


TESTDATA_PATH = _get_test_data_path()
"""
Path of the test data.

"""
