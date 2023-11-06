"""rdata: Read R datasets from Python."""
from __future__ import annotations

from importlib.resources import files
from typing import Final

from . import conversion, parser
from .parser._parser import Traversable


def _get_test_data_path() -> Traversable:
    return files(__name__) / "tests" / "data"


TESTDATA_PATH: Final[Traversable] = _get_test_data_path()
"""
Path of the test data.

"""

__version__ = "0.10.0"
