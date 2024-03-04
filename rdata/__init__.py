"""rdata: Read R datasets from Python."""
from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING, Final

from . import conversion as conversion, parser as parser, testing as testing
from ._read import read_rda as read_rda, read_rds as read_rds

if TYPE_CHECKING:
    from .parser._parser import Traversable


def _get_test_data_path() -> Traversable:
    return files(__name__) / "tests" / "data"


TESTDATA_PATH: Final[Traversable] = _get_test_data_path()
"""
Path of the test data.

"""

__version__ = "0.11.2"
