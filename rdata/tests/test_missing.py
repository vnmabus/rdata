"""Tests of missing value functionality."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rdata.missing import R_FLOAT_NA, R_INT_NA, is_na, mask_na_values


def test_int_is_na() -> None:
    """Test checking NA values in int array."""
    array = np.array([1, 2, R_INT_NA], dtype=np.int32)
    ref_mask = np.array([0, 0, 1], dtype=np.bool_)

    mask = is_na(array)
    np.testing.assert_array_equal(mask, ref_mask)


def test_float_is_na() -> None:
    """Test checking NA values in float array."""
    array = np.array([1, 2, R_FLOAT_NA, np.nan], dtype=np.float64)
    ref_mask = np.array([0, 0, 1, 0], dtype=np.bool_)

    mask = is_na(array)
    np.testing.assert_array_equal(mask, ref_mask)


@pytest.mark.parametrize("value", [R_INT_NA, R_FLOAT_NA])
def test_value_is_na(value: Any) -> None:  # noqa: ANN401
    """Test checking single NA values."""
    assert is_na(value)


@pytest.mark.parametrize("value", [
    np.int32(0), 0, np.float64(0.0), 0.0, np.nan,
])
def test_value_is_not_na(value: Any) -> None:  # noqa: ANN401
    """Test checking single NA values."""
    assert not is_na(value)


def test_int64() -> None:
    """Test checking int64."""
    with pytest.raises(NotImplementedError):
        is_na(2**32)
    with pytest.raises(NotImplementedError):
        is_na(-2**32)


def test_wrong_type() -> None:
    """Test checking int64."""
    with pytest.raises(NotImplementedError):
        is_na("test")


def test_masked_array() -> None:
    """Test checking masked array creation."""
    array = np.array([1, 2, R_FLOAT_NA, np.nan], dtype=np.float64)
    ref_mask = np.array([0, 0, 1, 0], dtype=np.bool_)
    ref_data = array.copy()

    masked = mask_na_values(array)
    assert isinstance(masked, np.ma.MaskedArray)
    np.testing.assert_array_equal(masked.data, ref_data)
    np.testing.assert_array_equal(masked.mask, ref_mask)


def test_masked_array_fill() -> None:
    """Test checking masked array creation."""
    array = np.array([1, 2, R_FLOAT_NA, np.nan], dtype=np.float64)
    ref_mask = np.array([0, 0, 1, 0], dtype=np.bool_)
    ref_data = array.copy()
    ref_data[ref_mask] = 42

    masked = mask_na_values(array, fill_value=42)
    assert isinstance(masked, np.ma.MaskedArray)
    np.testing.assert_array_equal(masked.data, ref_data)
    np.testing.assert_array_equal(masked.mask, ref_mask)


def test_nonmasked_array() -> None:
    """Test checking masked array no-op."""
    array = np.array([1, 2, np.nan, np.nan], dtype=np.float64)

    masked = mask_na_values(array)
    assert not isinstance(masked, np.ma.MaskedArray)
    np.testing.assert_array_equal(masked, array)
