"""Utilities for missing (NA) values in R."""

from typing import Final

import numpy as np

#: Value used to represent a missing integer in R.
R_INT_NA: Final[int] = -2**31

#: Value used to represent a missing float in R.
#  This is a NaN with a particular payload, but it's not the same as np.nan.
R_FLOAT_NA: Final[float] = np.frombuffer(b"\x7f\xf0\x00\x00\x00\x00\x07\xa2", dtype=">f8").astype("=f8")[0]  # noqa: E501


def is_float_na(value: float) -> bool:
    """Check if value is NA value."""
    return np.array(value).tobytes() == np.array(R_FLOAT_NA).tobytes()
