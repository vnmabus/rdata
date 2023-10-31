"""rdata: Read R datasets from Python."""
from importlib.resources import files
from importlib.resources.abc import Traversable

from . import conversion, parser


def _get_test_data_path() -> Traversable:
    return files(__name__) / "tests" / "data"


TESTDATA_PATH = _get_test_data_path()
"""
Path of the test data.

"""

__version__ = "0.9.2.dev1"
