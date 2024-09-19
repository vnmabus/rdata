"""Utilities for missing (NA) values in R."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any, Final

    import numpy.typing as npt


#: Value used to represent a missing integer in R.
R_INT_NA: Final[int] = np.int32(-2**31)  # type: ignore [assignment]

#: Value used to represent a missing float in R.
#  This is a NaN with a particular payload, but it's not the same as np.nan.
R_FLOAT_NA: Final[float] = np.uint64(0x7ff00000000007a2).view(np.float64)  # type: ignore [assignment]


def get_na_value(dtype: np.dtype[Any]) -> Any:  # noqa: ANN401
    """
    Get NA value for a given type.

    Args:
        dtype: NumPy dtype.

    Returns:
        NA value of given dtype.
    """
    if dtype == np.int32:
        return R_INT_NA
    if dtype == np.float64:
        return R_FLOAT_NA
    msg = f"NA for numpy dtype {dtype} not implemented"
    raise NotImplementedError(msg)


def is_na(
    array: Any | npt.NDArray[Any],  # noqa: ANN401
) -> bool | npt.NDArray[np.bool_]:
    """
    Check if the array elements are NA.

    Args:
        array: NumPy array or single value.

    Returns:
        Boolean mask of NA values in the array.
    """
    if isinstance(array, np.ndarray):
        dtype = array.dtype
        na = get_na_value(dtype)
        if dtype == np.int32:
            # Use the native dtype for comparison when possible;
            # slightly faster than the steps below
            return array == na  # type: ignore [no-any-return]
        raw_dtype = f"V{array.dtype.itemsize}"
        return array.view(raw_dtype) == np.array(na).view(raw_dtype)  # type: ignore [no-any-return]

    if isinstance(array, int):
        try:
            return is_na(np.array(array, dtype=np.int32))
        except OverflowError:
            return is_na(np.array(array))

    if isinstance(array, (float, np.int32, np.float64)):
        return is_na(np.array(array))

    msg = f"NA for {type(array)} not implemented"
    raise NotImplementedError(msg)


def mask_na_values(
    array: npt.NDArray[Any],
    *,
    fill_value: Any | None = None,  # noqa: ANN401
) -> npt.NDArray[Any] | np.ma.MaskedArray[Any, Any]:
    """
    Mask NA elements of the array.

    Args:
        array: NumPy array.
        fill_value: Fill value for the masked array.
            Defaults to the NA value.

    Returns:
        NumPy masked array with NA values as the mask
        or the original array if there is no NA elements.
    """
    mask = is_na(array)
    if np.any(mask):
        if fill_value is None:
            fill_value = get_na_value(array.dtype)

        array[mask] = fill_value
        return np.ma.array(  # type: ignore [no-untyped-call,no-any-return]
            data=array,
            mask=mask,
            fill_value=fill_value,
        )
    return array
