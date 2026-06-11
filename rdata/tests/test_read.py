"""Tests of parsing and conversion."""

from collections import ChainMap
from contextlib import AbstractContextManager, nullcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import pytest
import xarray

import rdata
from rdata.missing import R_FLOAT_NA

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from rdata.parser._parser import AcceptableFile

TESTDATA_PATH = rdata.TESTDATA_PATH
NATIVE_BINARY_FORMAT_VERSION = 2

##############################################################################
# Special behavior tests
# ----------------------
#
# The tests below check for special behaviors of the reading functions.
###############################################################################


@pytest.mark.read_test("test_open_file")
def test_open_file(
    dataset_path: str,
    rdata_format_suffix: Literal["rda", "rds"],
    subtests: pytest.Subtests,
) -> None:
    """
    Tests the different ways of passing a file to be opened by ```parse_file``.

    Test ``parse_file`` passing the following options:
        1. The file path as a string.
        2. The file path as a path object.
        3. The file already opened as a binary file.

    For this test the data contents do not really matter.

    Code for creating the data:

    ::: test_open_file <- c(1.0, 2.0, 3.0)

    """
    open_options: Mapping[
        str,
        Callable[[], AbstractContextManager[str | Path | AcceptableFile]],
    ] = {
        "string": lambda: nullcontext(dataset_path),
        "path": lambda: nullcontext(Path(dataset_path)),
        "opened file": lambda: Path(dataset_path).open("rb"),  # noqa: SIM115
    }

    for option_name, open_option in open_options.items():
        with subtests.test(msg=option_name):
            with open_option() as f:
                parsed = rdata.parser.parse_file(
                    f,
                    extension=f".{rdata_format_suffix}",
                )

            converted = rdata.conversion.convert(parsed)

            reference = np.array([1., 2., 3.])

            match rdata_format_suffix:
                case "rds":
                    np.testing.assert_equal(
                        converted,
                        reference,
                    )
                case "rda":
                    np.testing.assert_equal(
                        converted,
                        {"test_open_file": reference},
                    )


def test_native_binary_rds_little_endian() -> None:
    """Test parsing native binary RDS in little-endian byte order."""
    def i32(value: int) -> bytes:
        return int(value).to_bytes(4, byteorder="little", signed=True)

    data = b"".join((
        b"B\n",
        i32(2),
        i32(0x00030002),
        i32(0x00020300),
        i32(254),
    ))
    parsed = rdata.parser.parse_data(data, extension=".rds")

    assert parsed.versions.format == NATIVE_BINARY_FORMAT_VERSION
    assert parsed.object.info.type == rdata.parser.RObjectType.NILVALUE


def test_native_binary_rds_big_endian() -> None:
    """Test parsing native binary RDS in big-endian byte order."""
    def i32(value: int) -> bytes:
        return int(value).to_bytes(4, byteorder="big", signed=True)

    data = b"".join((
        b"B\n",
        i32(2),
        i32(0x00030002),
        i32(0x00020300),
        i32(254),
    ))
    parsed = rdata.parser.parse_data(data, extension=".rds")

    assert parsed.versions.format == NATIVE_BINARY_FORMAT_VERSION
    assert parsed.object.info.type == rdata.parser.RObjectType.NILVALUE


def test_native_binary_rda_header() -> None:
    """Test parsing native binary RDA with RDB wrapper magic."""
    def i32(value: int) -> bytes:
        return int(value).to_bytes(4, byteorder="little", signed=True)

    data = b"".join((
        b"RDB2\n",
        b"B\n",
        i32(2),
        i32(0x00030002),
        i32(0x00020300),
        i32(254),
    ))
    parsed = rdata.parser.parse_data(data, extension=".rda")

    assert parsed.versions.format == NATIVE_BINARY_FORMAT_VERSION
    assert parsed.object.info.type == rdata.parser.RObjectType.NILVALUE


###############################################################################
# Tests of parsing/conversion of objects
# --------------------------------------
#
# The tests below check for correct parsing and conversion of different kinds
# of R objects.
###############################################################################


@pytest.mark.read_test("test_logical")
def test_logical(
    dataset_object: np.typing.NDArray[np.bool],
) -> None:
    """
    Test parsing of logical vectors.

    Code for creating the data:

    ::: test_logical <- c(TRUE, TRUE, FALSE, TRUE, FALSE)

    """
    np.testing.assert_equal(
        dataset_object,
        np.array([True, True, False, True, False]),
    )


@pytest.mark.read_test("test_nullable_logical")
def test_nullable_logical(
    dataset_object: np.ma.MaskedArray[Any, np.dtype[np.bool]],
) -> None:
    """
    Test parsing of logical vectors containing NA.

    Code for creating the data:

    ::: test_nullable_logical <- c(TRUE, FALSE, NA)

    """
    np.testing.assert_array_equal(
        dataset_object.data,
        np.array([True, False, True]),
    )
    np.testing.assert_array_equal(
        dataset_object.mask,
        np.array([False, False, True]),
    )


@pytest.mark.read_test("test_nullable_int")
def test_nullable_int(
    dataset_object: np.ma.MaskedArray[Any, np.dtype[np.int32]],
) -> None:
    """
    Test parsing of integer vectors containing NA.

    Code for creating the data:

    ::: test_nullable_int <- c(313L, -12L, NA)

    """
    np.testing.assert_array_equal(
        dataset_object.data,
        np.array([313, -12, -2**31]),
    )
    np.testing.assert_array_equal(
        dataset_object.mask,
        np.array([False, False, True]),
    )


@pytest.mark.read_test("test_vector")
def test_vector(
    dataset_object: np.typing.NDArray[np.float64],
) -> None:
    """
    Test parsing of numerical vectors.

    Code for creating the data:

    ::: test_vector <- c(1.0, 2.0, 3.0)

    """
    np.testing.assert_equal(
        dataset_object,
        np.array([1.0, 2.0, 3.0]),
    )


@pytest.mark.read_test("test_named_vector")
def test_named_vector(
    dataset_object: xarray.DataArray,
) -> None:
    """
    Test parsing of vectors with names.

    Code for creating the object:

    ::: test_named_vector <- c(a=1, b=2, c=3)

    """
    xarray.testing.assert_identical(
        dataset_object,
        xarray.DataArray(
            [1.0, 2.0, 3.0],
            coords=[["a", "b", "c"]],
        ),
    )


@pytest.mark.read_test("test_empty_string")
def test_empty_string(
    dataset_object: np.typing.NDArray[np.str_],
) -> None:
    """
    Test that the empty string is parsed correctly.

    Code for creating the data:

    ::: test_empty_string <- ""

    """
    np.testing.assert_array_equal(
        dataset_object,
        np.array([""]),
    )


@pytest.mark.read_test("test_na_string")
def test_na_string(
    dataset_object: np.typing.NDArray[np.object_],
) -> None:
    """
    Test that the NA string is parsed correctly.

    Code for creating the data:

    ::: test_na_string <- as.character(NA)

    """
    np.testing.assert_array_equal(
        dataset_object,
        np.array([None]),
    )


@pytest.mark.read_test("test_complex")
def test_complex(
    dataset_object: np.typing.NDArray[np.complex64],
) -> None:
    """
    Test that complex numbers can be parsed.

    Code for creating the data:

    ::: test_complex <- c(1 + 2i, 2, 0, 1 + 3i, -1i)

    """
    np.testing.assert_equal(
        dataset_object,
        np.array([1 + 2j, 2, 0, 1 + 3j, -1j]),
    )


@pytest.mark.read_test("test_matrix")
def test_matrix(
    dataset_object: np.typing.NDArray[np.float64],
) -> None:
    """
    Test that a matrix can be parsed.

    Code for creating the data:

    ::: test_matrix <- matrix(1:6, nrow=2, byrow=TRUE)

    """
    np.testing.assert_equal(
        dataset_object,
        np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]),
    )


@pytest.mark.read_test("test_named_matrix")
def test_named_matrix(
    dataset_object: xarray.DataArray,
) -> None:
    """
    Test that a named matrix can be parsed.

    Code for creating the data:

    ::: dimnames <- list(
    :::     c("dim0_0", "dim0_1"),
    :::     c("dim1_0", "dim1_1", "dim1_2")
    ::: )
    ::: test_named_matrix <- matrix(1:6, nrow=2, byrow=TRUE, dimnames=dimnames)

    """
    reference = xarray.DataArray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dims=["dim_0", "dim_1"],
        coords={
            "dim_0": ["dim0_0", "dim0_1"],
            "dim_1": ["dim1_0", "dim1_1", "dim1_2"],
        },
    )

    xarray.testing.assert_identical(
        dataset_object,
        reference,
    )


@pytest.mark.read_test("test_half_named_matrix")
def test_half_named_matrix(
    dataset_object: xarray.DataArray,
) -> None:
    """
    Test that a named matrix with no name for a dim can be parsed.

    Code for creating the data:

    ::: test_half_named_matrix <- matrix(
    :::     1:6,
    :::     nrow=2,
    :::     byrow=TRUE,
    :::     dimnames=list(c("dim0_0", "dim0_1"))
    ::: )

    """
    reference = xarray.DataArray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dims=["dim_0", "dim_1"],
        coords={
            "dim_0": ["dim0_0", "dim0_1"],
        },
    )

    xarray.testing.assert_identical(
        dataset_object,
        reference,
    )


@pytest.mark.read_test("test_full_named_matrix")
def test_full_named_matrix(
    dataset_object: xarray.DataArray,
) -> None:
    """
    Test that a named matrix with dim names can be parsed.

    Code for creating the data:

    ::: dimnames <- list(
    :::     my_dim_0=c("dim0_0", "dim0_1"),
    :::     my_dim_1=c("dim1_0", "dim1_1", "dim1_2")
    ::: )
    ::: test_full_named_matrix <- matrix(
    :::     1:6,
    :::     nrow=2,
    :::     byrow=TRUE,
    :::     dimnames=dimnames
    ::: )

    """
    reference = xarray.DataArray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dims=["my_dim_0", "my_dim_1"],
        coords={
            "my_dim_0": ["dim0_0", "dim0_1"],
            "my_dim_1": ["dim1_0", "dim1_1", "dim1_2"],
        },
    )

    xarray.testing.assert_identical(
        dataset_object,
        reference,
    )


@pytest.mark.read_test("test_empty_list")
def test_empty_list(
    dataset_object: list[object],
) -> None:
    """
    Test parsing the empty list.

    Code for creating the data:

    ::: test_empty_list <- list()

    """
    assert dataset_object == []


@pytest.mark.read_test("test_empty_named_list")
def test_empty_named_list(
    dataset_object: dict[str, object],
) -> None:
    """
    Test parsing the empty named list.

    Code for creating the data:

    ::: test_empty_named_list <- setNames(list(), character(0))

    """
    assert dataset_object == {}


@pytest.mark.read_test("test_list")
def test_list(
    dataset_object: list[np.typing.NDArray[np.float64 |np.str_]],
) -> None:
    """
    Test that list can be parsed.

    Code for creating the data:

    ::: test_list <- list(1, c("a", "b", "c"), c(2, 3), "hi")

    """
    np.testing.assert_equal(
        dataset_object,
        [
            np.array([1.0]),
            np.array(["a", "b", "c"]),
            np.array([2.0, 3.0]),
            np.array(["hi"]),
        ],
    )


@pytest.mark.read_test("test_list_attrs")
def test_list_attrs(
    dataset_object: list[np.typing.NDArray[np.str_ | np.int32]],
) -> None:
    """
    Test that lists accept attributes.

    Code for creating the data:

    ::: test_list_attrs <- list("list", 5)
    ::: attr(test_list_attrs, "my_attr") <- "attr_value"

    """
    np.testing.assert_equal(
        dataset_object,
        [np.array(["list"]), np.array([5])],
    )


@pytest.mark.filterwarnings("ignore:Missing constructor")
@pytest.mark.read_test("test_file")
def test_file(
    dataset_object: list[int],
) -> None:
    """
    Test that external pointers can be parsed.

    Code for creating the data:

    ::: test_file <- file()
    """
    np.testing.assert_equal(
        dataset_object,
        np.array([3]),
    )


@pytest.mark.read_test("test_expression")
def test_expression(
    dataset_object: rdata.conversion.RExpression,
) -> None:
    """
    Test that expressions can be parsed.

    Code for creating the data:

    ::: test_expression <- expression(base^exponent)

    """
    assert dataset_object == rdata.conversion.RExpression([
        rdata.conversion.RLanguage(
            ["^", "base", "exponent"],
            attributes={},
        ),
    ])


@pytest.mark.read_test("test_builtin")
def test_builtin(
    dataset_object: rdata.conversion.RBuiltin,
) -> None:
    """
    Test that builtin functions can be parsed.

    Code for creating the data:

    ::: test_builtin <- abs

    """
    assert dataset_object == rdata.conversion.RBuiltin(name="abs")


@pytest.mark.read_test("test_minimal_function_uncompiled")
def test_minimal_function_uncompiled(
    dataset_object: rdata.conversion.RFunction,
) -> None:
    """
    Test that a minimal function can be parsed.

    Code for creating the data:

    ::: options(keep.source = TRUE)
    ::: test_minimal_function_uncompiled <- function() NULL

    """
    assert isinstance(
        dataset_object,
        rdata.conversion.RFunction,
    )

    np.testing.assert_equal(dataset_object.environment, ChainMap({}))
    np.testing.assert_equal(dataset_object.formals, None)
    np.testing.assert_equal(dataset_object.body, None)
    np.testing.assert_equal(
        dataset_object.source,
        "test_minimal_function_uncompiled <- function() NULL\n",
    )


@pytest.mark.filterwarnings("ignore:Missing constructor")
@pytest.mark.read_test("test_minimal_function")
def test_minimal_function(
    dataset_object: rdata.conversion.RFunction,
) -> None:
    """
    Test that a minimal function (compiled) can be parsed.

    Code for creating the data:

    ::: options(keep.source = TRUE)
    ::: test_minimal_function <- function() NULL
    ::: library(compiler)
    ::: test_minimal_function <- cmpfun(test_minimal_function)

    """
    assert isinstance(
        dataset_object,
        rdata.conversion.RFunction,
    )

    np.testing.assert_equal(dataset_object.environment, ChainMap({}))
    np.testing.assert_equal(dataset_object.formals, None)

    converted_body = dataset_object.body

    assert isinstance(
        converted_body,
        rdata.conversion.RBytecode,
    )

    np.testing.assert_equal(converted_body.code, np.array([12, 17, 1]))
    np.testing.assert_equal(converted_body.attributes, {})

    np.testing.assert_equal(
        dataset_object.source,
        "test_minimal_function <- function() NULL\n",
    )


@pytest.mark.read_test("test_empty_function_uncompiled")
def test_empty_function_uncompiled(
    dataset_object: rdata.conversion.RFunction,
) -> None:
    """
    Test that a simple function can be parsed.

    Code for creating the data:

    ::: options(keep.source = TRUE)
    ::: test_empty_function_uncompiled <- function() {}

    """
    assert isinstance(
        dataset_object,
        rdata.conversion.RFunction,
    )

    np.testing.assert_equal(dataset_object.environment, ChainMap({}))
    np.testing.assert_equal(dataset_object.formals, None)
    assert isinstance(dataset_object.body, rdata.conversion.RLanguage)
    np.testing.assert_equal(
        dataset_object.source,
        "test_empty_function_uncompiled <- function() {}\n",
    )


@pytest.mark.filterwarnings("ignore:Missing constructor")
@pytest.mark.read_test("test_empty_function")
def test_empty_function(
    dataset_object: rdata.conversion.RFunction,
) -> None:
    """
    Test that a simple function (compiled) can be parsed.

    Code for creating the data:

    ::: options(keep.source = TRUE)
    ::: test_empty_function <- function() {}
    ::: library(compiler)
    ::: test_empty_function <- cmpfun(test_empty_function)

    """
    assert isinstance(
        dataset_object,
        rdata.conversion.RFunction,
    )

    np.testing.assert_equal(dataset_object.environment, ChainMap({}))
    np.testing.assert_equal(dataset_object.formals, None)

    converted_body = dataset_object.body

    assert isinstance(
        converted_body,
        rdata.conversion.RBytecode,
    )

    np.testing.assert_equal(converted_body.code, np.array([12, 17, 1]))
    np.testing.assert_equal(converted_body.attributes, {})

    np.testing.assert_equal(
        dataset_object.source,
        "test_empty_function <- function() {}\n",
    )


@pytest.mark.filterwarnings("ignore:Missing constructor")
@pytest.mark.read_test("test_function")
def test_function(
    dataset_object: rdata.conversion.RFunction,
) -> None:
    """
    Test that functions can be parsed.

    Code for creating the data:

    ::: options(keep.source = TRUE)
    ::: test_function <- function() {print("Hello")}
    ::: library(compiler)
    ::: test_function <- cmpfun(test_function)

    """
    assert isinstance(
        dataset_object,
        rdata.conversion.RFunction,
    )

    np.testing.assert_equal(dataset_object.environment, ChainMap({}))
    np.testing.assert_equal(dataset_object.formals, None)

    converted_body = dataset_object.body

    assert isinstance(
        converted_body,
        rdata.conversion.RBytecode,
    )

    np.testing.assert_equal(
        converted_body.code,
        np.array([12, 23, 1, 34, 4, 38, 2, 1]),
    )
    np.testing.assert_equal(converted_body.attributes, {})

    np.testing.assert_equal(
        dataset_object.source,
        "test_function <- function() {print(\"Hello\")}\n",
    )


@pytest.mark.filterwarnings("ignore:Missing constructor")
@pytest.mark.read_test("test_function_arg")
def test_function_arg(
    dataset_object: rdata.conversion.RFunction,
) -> None:
    """
    Test that functions can be parsed.

    Code for creating the data:

    ::: options(keep.source = TRUE)
    ::: test_function_arg <- function(a) {print(a)}
    ::: library(compiler)
    ::: test_function_arg <- cmpfun(test_function_arg)

    """
    assert isinstance(
        dataset_object,
        rdata.conversion.RFunction,
    )

    np.testing.assert_equal(dataset_object.environment, ChainMap({}))
    np.testing.assert_equal(dataset_object.formals, {"a": NotImplemented})

    converted_body = dataset_object.body

    assert isinstance(
        converted_body,
        rdata.conversion.RBytecode,
    )

    np.testing.assert_equal(
        converted_body.code,
        np.array([12, 23, 1, 29, 4, 38, 2, 1]),
    )
    np.testing.assert_equal(converted_body.attributes, {})

    np.testing.assert_equal(
        dataset_object.source,
        "test_function_arg <- function(a) {print(a)}\n",
    )


@pytest.mark.read_test("test_empty_dataframe")
def test_empty_dataframe(
    dataset_object: pd.DataFrame,
) -> None:
    """
    Test conversion of an empty dataframe.

    Code for creating the object:

    ::: test_empty_dataframe <- data.frame()

    """
    pd.testing.assert_frame_equal(
        dataset_object,
        pd.DataFrame(columns=[], index=np.array([], dtype=np.int32)),
    )


@pytest.mark.read_test("test_empty_dataframe_without_names")
def test_empty_dataframe_without_names(
    dataset_object: pd.DataFrame,
) -> None:
    """
    Test conversion of an empty dataframe without names.

    Code for creating the object:

    ::: test_empty_dataframe_without_names <- data.frame()
    ::: attr(test_empty_dataframe_without_names, "names") <- NULL

    """
    pd.testing.assert_frame_equal(
        dataset_object,
        pd.DataFrame(index=np.array([], dtype=np.int32)),
    )


@pytest.mark.read_test("test_dataframe")
def test_dataframe(
    dataset_object: pd.DataFrame,
) -> None:
    """
    Test dataframe conversion.

    Code for creating the object:

    ::: test_dataframe <- data.frame(
    :::     class=factor(c("a", "b", "b")),
    :::     value=c(1L, 2L, 3L)
    ::: )

    """
    pd.testing.assert_frame_equal(
        dataset_object,
        pd.DataFrame(
            {
                "class": pd.Categorical(
                    ["a", "b", "b"],
                ),
                "value": pd.Series(
                    [1, 2, 3],
                    dtype=pd.Int32Dtype(),
                ).array,
            },
            index=pd.RangeIndex(start=1, stop=4),
        ),
    )


@pytest.mark.read_test("test_dataframe_rownames")
def test_dataframe_rownames(
    dataset_object: pd.DataFrame,
) -> None:
    """
    Test dataframe conversion.

    Code for creating the object:

    ::: test_dataframe_rownames <- data.frame(
    :::     class=factor(c("a", "b", "b")),
    :::     value=c(1L, 2L, 3L),
    :::     row.names=c("Madrid", "Frankfurt", "Herzberg am Harz")
    ::: )

    """
    pd.testing.assert_frame_equal(
        dataset_object,
        pd.DataFrame(
            {
                "class": pd.Categorical(
                    ["a", "b", "b"],
                ),
                "value": pd.Series(
                    [1, 2, 3],
                    dtype=pd.Int32Dtype(),
                ).array,
            },
            index=("Madrid", "Frankfurt", "Herzberg am Harz"),
        ),
    )


@pytest.mark.read_test("test_dataframe_int_rownames")
def test_dataframe_int_rownames(
    dataset_object: pd.DataFrame,
) -> None:
    """
    Test dataframe conversion.

    Code for creating the object:

    ::: test_dataframe_int_rownames <- data.frame(
    :::     col1=c(10, 20, 30),
    :::     row.names=c(3L, 6L, 9L)
    ::: )

    """
    index = np.array([3, 6, 9], dtype=np.int32)
    reference = pd.DataFrame(
        {
            "col1": pd.Series(
                [10., 20., 30.],
                dtype=float, index=index),
        },
        index=index,
    )
    pd.testing.assert_frame_equal(dataset_object, reference)


@pytest.mark.read_test("test_dataframe_range_rownames")
def test_dataframe_range_rownames(
    dataset_object: pd.DataFrame,
) -> None:
    """
    Test dataframe conversion.

    Code for creating the object:

    ::: test_dataframe_range_rownames <- data.frame(
    :::     col1=c(10, 20, 30),
    :::     row.names=2:4
    ::: )

    """
    index = pd.Index([2, 3, 4], dtype=np.int32)
    reference = pd.DataFrame(
        {
            "col1": pd.Series(
                [10., 20., 30.],
                dtype=float, index=index),
        },
        index=index,
    )
    pd.testing.assert_frame_equal(dataset_object, reference)


@pytest.mark.read_test("test_dataframe_dtypes")
def test_dataframe_dtypes(
    dataset_object: pd.DataFrame,
) -> None:
    """
    Test dataframe conversion.

    Code for creating the object:

    ::: test_dataframe_dtypes <- data.frame(
    :::     int=c(10L, 20L, 30L),
    :::     float=c(1.1, 2.2, 3.3),
    :::     string=c("x", "y", "z"),
    :::     bool=as.logical(c(1, 0, 1)),
    :::     complex=c(4+5i, 6+7i, 8+9i)
    ::: )

    """
    index = pd.RangeIndex(1, 4)
    reference = pd.DataFrame(
        {
            "int": pd.Series(
                [10, 20, 30],
                dtype=pd.Int32Dtype(), index=index),
            "float": pd.Series(
                [1.1, 2.2, 3.3],
                dtype=float, index=index),
            "string": pd.Series(
                ["x", "y", "z"],
                dtype=pd.StringDtype(), index=index),
            "bool": pd.Series(
                [True, False, True],
                dtype=pd.BooleanDtype(), index=index),
            "complex": pd.Series(
                [4+5j, 6+7j, 8+9j],
                dtype=complex, index=index),
        },
        index=index,
    )
    pd.testing.assert_frame_equal(dataset_object, reference)


@pytest.mark.read_test("test_dataframe_dtypes_with_na")
def test_dataframe_dtypes_with_na(
    dataset_object: pd.DataFrame,
) -> None:
    """
    Test dataframe conversion.

    Code for creating the object:

    ::: test_dataframe_dtypes_with_na <- data.frame(
    :::     int=c(10L, 20L, 30L, NA),
    :::     float=c(1.1, 2.2, 3.3, NA),
    :::     string=c("x", "y", "z", NA),
    :::     bool=as.logical(c(1, 0, 1, NA)),
    :::     complex=c(4+5i, 6+7i, 8+9i, NA)
    ::: )

    """
    index = pd.RangeIndex(1, 5)
    reference = pd.DataFrame(
        {
            "int": pd.Series(
                [10, 20, 30, pd.NA],
                dtype=pd.Int32Dtype(), index=index),
            "float": pd.Series(
                [1.1, 2.2, 3.3, R_FLOAT_NA],
                dtype=float, index=index),
            "string": pd.Series(
                ["x", "y", "z", pd.NA],
                dtype=pd.StringDtype(), index=index),
            "bool": pd.Series(
                [True, False, True, pd.NA],
                dtype=pd.BooleanDtype(), index=index),
            "complex": pd.Series(
                [4+5j, 6+7j, 8+9j, R_FLOAT_NA],
                dtype=complex, index=index),
        },
        index=index,
    )

    with np.errstate(invalid="ignore"):
        # Comparing complex arrays with R_FLOAT_NA gives warning
        pd.testing.assert_frame_equal(dataset_object, reference)


@pytest.mark.read_test("test_dataframe_float_with_na_nan")
def test_dataframe_float_with_na_nan(
    dataset_object: pd.DataFrame,
) -> None:
    """
    Test dataframe conversion.

    Code for creating the object:

    ::: test_dataframe_float_with_na_nan <- data.frame(
    :::     float=c(1.1, 2.2, 3.3, NA, NaN, Inf, -Inf)
    ::: )

    """
    index = pd.RangeIndex(1, 8)
    reference = pd.DataFrame(
        {
            "float": pd.Series(
                [1.1, 2.2, 3.3, R_FLOAT_NA, np.nan, np.inf, -np.inf],
                dtype=float, index=index),
        },
        index=index,
    )
    pd.testing.assert_frame_equal(dataset_object, reference)


@pytest.mark.read_test("test_factor")
def test_factor(
    dataset_object: pd.DataFrame,
) -> None:
    """
    Test factor conversion.

    Code for creating the object:

    ::: test_factor <- factor(c("a", "b", "b"))

    """
    pd.testing.assert_frame_equal(
        pd.DataFrame(dataset_object),
        pd.DataFrame(pd.Categorical(["a", "b", "b"])),
    )


@pytest.mark.read_test("test_ts")
def test_ts(
    dataset_object: pd.Series,
) -> None:
    """
    Test time series conversion.

    Code for creating the object:

    ::: test_ts <- ts(c(1, 2, 3), start=c(2000, 3), frequency = 12)

    """
    pd.testing.assert_series_equal(
        dataset_object,
        pd.Series({
            2000 + Fraction(2, 12): 1.0,
            2000 + Fraction(3, 12): 2.0,
            2000 + Fraction(4, 12): 3.0,
        }),
    )


@pytest.mark.read_test("test_s4")
def test_s4(
    recwarn: pytest.WarningsRecorder,
    dataset_object: pd.Series,
) -> None:
    """
    Test parsing of S4 classes.

    Code for creating the object:

    ::: setClass("Person", representation(name = "character", age = "numeric"))
    ::: test_s4 <- new("Person", name = "Carlos", age = 28)

    """
    assert len(recwarn) == 1
    w = recwarn.pop(UserWarning)
    assert issubclass(w.category, UserWarning)
    assert str(w.message).startswith("Missing constructor")

    np.testing.assert_equal(
        dataset_object,
        SimpleNamespace(
            age=np.array(28),
            name=["Carlos"],
            **{"class": ["Person"]},
        ),
    )


@pytest.mark.read_test("test_environment_global_default")
def test_environment_global_default(
    dataset_object: ChainMap[str, object],
) -> None:
    """
    Test parsing of environments, with default global environment.

    Code for creating the object:

    ::: test_environment_global_default <- new.env()
    ::: assign("string", "test", envir = test_environment_global_default)

    """
    dict_env = {"string": np.array(["test"])}
    empty_global_env: dict[str, np.typing.NDArray[np.str_]] = {}

    assert dataset_object == ChainMap(dict_env, ChainMap(empty_global_env))


@pytest.mark.read_test(
        "test_environment_global_argument",
        global_environment={"global": ["test"]},
)
def test_environment_global_argument(
    dataset_object: ChainMap[str, object],
) -> None:
    """
    Test parsing of environments, with default global environment.

    Code for creating the object:

    ::: test_environment_global_argument <- new.env()
    ::: assign("string", "test", envir = test_environment_global_argument)

    """
    dict_env = {"string": ["test"]}
    global_env = {"global": ["test"]}

    assert dataset_object == ChainMap(dict_env, ChainMap(global_env))


@pytest.mark.read_test("test_emptyenv")
def test_emptyenv(
    dataset_object: ChainMap[str, object],
) -> None:
    """
    Test parsing the empty environment.

    Code for creating the object:

    ::: test_emptyenv <- emptyenv()

    """
    assert dataset_object == ChainMap({})


@pytest.mark.read_test("test_altrep_compact_intseq")
def test_altrep_compact_intseq(
    dataset_object: np.typing.NDArray[np.int32],
) -> None:
    """
    Test alternative representation of sequences of ints.

    Code for creating the object:

    ::: test_altrep_compact_intseq <- 0:999

    """
    np.testing.assert_equal(
        dataset_object,
        np.arange(1000),
    )


@pytest.mark.read_test("test_altrep_compact_intseq_asymmetric")
def test_altrep_compact_intseq_asymmetric(
    dataset_object: np.typing.NDArray[np.int32],
) -> None:
    """
    Test alternative representation of sequences of ints.

    This test an origin different from 0, to reproduce
    issue #29.

    Code for creating the object:

    ::: test_altrep_compact_intseq_asymmetric <- -5:5

    """
    np.testing.assert_equal(
        dataset_object,
        np.arange(-5, 6),
    )


@pytest.mark.read_test("test_altrep_compact_realseq")
def test_altrep_compact_realseq(
    dataset_object: np.typing.NDArray[np.float64],
) -> None:
    """
    Test alternative representation of sequences of ints.

    Code for creating the object:

    ::: test_altrep_compact_realseq <- seq(0, 999, by=1)

    """
    np.testing.assert_equal(
        dataset_object,
        np.arange(1000.0),
    )


@pytest.mark.read_test("test_altrep_compact_realseq_asymmetric")
def test_altrep_compact_realseq_asymmetric(
    dataset_object: np.typing.NDArray[np.float64],
) -> None:
    """
    Test alternative representation of sequences of reals.

    This test an origin different from 0, to reproduce
    issue #29.

    Code for creating the object:

    ::: test_altrep_compact_realseq_asymmetric <- seq(-5, 5, by=1)

    """
    np.testing.assert_equal(
        dataset_object,
        np.arange(-5.0, 6.0),
    )



@pytest.mark.read_test("test_altrep_deferred_string")
def test_altrep_deferred_string(
    dataset_object: np.typing.NDArray[np.str_],
) -> None:
    """
    Test alternative representation of deferred strings.

    Code for creating the object:

    ::: test_altrep_deferred_string <- as.character(
    :::     c(1, 2.3, 10000, 1e+05, -10000, -1e+05, 0.001, 1e-04, 1e-05)
    ::: )

    """
    np.testing.assert_array_equal(
        dataset_object,
        np.array([
            "1", "2.3", "10000",
            "1e+05", "-10000", "-1e+05",
            "0.001", "1e-04", "1e-05",
        ]),
    )


@pytest.mark.read_test("test_altrep_wrap_real")
def test_altrep_wrap_real(
    dataset_object_parsed: rdata.parser.RObject,
    dataset_object: np.typing.NDArray[np.int32],
) -> None:
    """
    Test alternative representation of wrap_real.

    Code for creating the object:

    ::: test_altrep_wrap_real <- .Internal(wrap_meta(3, 0, 0))

    """
    parsed = dataset_object_parsed
    assert parsed.info.type == rdata.parser.RObjectType.REAL  # sanity check
    assert not parsed.info.object
    assert not parsed.info.attributes
    assert parsed.attributes is None

    np.testing.assert_equal(
        dataset_object,
        np.array([3]),
    )


@pytest.mark.read_test("test_altrep_wrap_real_attributes")
def test_altrep_wrap_real_attributes(
    dataset_object_parsed: rdata.parser.RObject,
    dataset_object: np.typing.NDArray[np.int32],
) -> None:
    """
    Test alternative representation of wrap_real with attributes.

    Code for creating the object:

    ::: test_altrep_wrap_real_attributes <- .Internal(
    :::     wrap_meta(c(1, 2, 3), 0, 0)
    ::: )
    ::: attr(test_altrep_wrap_real_attributes, "foo") <- "bar"

    """
    parsed = dataset_object_parsed
    assert parsed.info.type == rdata.parser.RObjectType.REAL  # sanity check
    assert not parsed.info.object
    assert parsed.info.attributes
    assert parsed.attributes is not None
    assert parsed.attributes.tag is not None
    assert parsed.attributes.tag.value.value == b"foo"
    assert parsed.attributes.value[0].value[0].value == b"bar"

    np.testing.assert_equal(
        dataset_object,
        np.array([1., 2., 3.]),
    )


@pytest.mark.filterwarnings("ignore:Missing constructor")
@pytest.mark.read_test("test_altrep_wrap_real_class_attribute")
def test_altrep_wrap_real_class_attribute(
    dataset_object_parsed: rdata.parser.RObject,
    dataset_object: np.typing.NDArray[np.int32],
) -> None:
    """
    Test altrep of wrap_real with class attribute.

    Code for creating the object:

    ::: test_altrep_wrap_real_class_attribute <- .Internal(
    :::     wrap_meta(c(1, 2, 3), 0, 0)
    ::: )
    ::: attr(test_altrep_wrap_real_class_attribute, "class") <- "Date"

    """
    parsed = dataset_object_parsed
    assert parsed.info.type == rdata.parser.RObjectType.REAL  # sanity check
    assert parsed.info.object
    assert parsed.info.attributes
    assert parsed.attributes is not None
    assert parsed.attributes.tag is not None
    assert parsed.attributes.tag.value.value == b"class"
    assert parsed.attributes.value[0].value[0].value == b"Date"

    np.testing.assert_equal(
        dataset_object,
        np.array([1., 2., 3.]),
    )


@pytest.mark.read_test("test_altrep_wrap_string")
def test_altrep_wrap_string(
    dataset_object: np.typing.NDArray[np.str_],
) -> None:
    """
    Test alternative representation of wrap_string.

    Code for creating the object:

    ::: test_altrep_wrap_string <- .Internal(wrap_meta("Hello", 0, 0))

    """
    np.testing.assert_array_equal(
        dataset_object,
        np.array(["Hello"]),
    )


@pytest.mark.read_test("test_altrep_wrap_logical")
def test_altrep_wrap_logical(
    dataset_object: np.typing.NDArray[np.bool],
) -> None:
    """
    Test alternative representation of wrap_logical.

    Code for creating the object:

    ::: test_altrep_wrap_logical <- .Internal(wrap_meta(TRUE, 0, 0))

    """
    np.testing.assert_equal(
        dataset_object,
        np.array([True]),
    )


@pytest.mark.read_test("test_ascii_characters")
def test_ascii_characters(
    dataset_object: str,
) -> None:
    r"""
    Test reading string with all ascii printable characters.

    Code for creating the object:

    ::: test_ascii_characters <- "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\v\f\r\n"

    """  # noqa: E501
    assert dataset_object == "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\v\f\r\n"  # noqa: E501


@pytest.mark.read_test("test_nan_inf")
def test_nan_inf(
    dataset_object: np.typing.NDArray[np.float64],
) -> None:
    """
    Test reading nan and inf.

    Code for creating the object:

    ::: test_nan_inf <- c(0, -0, NaN, Inf, -Inf)

    """
    np.testing.assert_equal(
        dataset_object,
        np.array([0., -0., np.nan, np.inf, -np.inf]),
    )
    np.testing.assert_equal(
        np.signbit(dataset_object),
        np.array([False, True, False, False, True]),
    )


@pytest.mark.read_test("test_encoding_utf8")
def test_encoding_utf8(
    dataset_object: np.typing.NDArray[np.str_],
) -> None:
    r"""
    Test of character vector with UTF-8 encoding.

    Code for creating the object:

    ::: test_encoding_utf8 <- "e\xc4\xa5o\xc5\x9dan\xc4\x9do \xc4\x89iu\xc4\xb5a\xc5\xadde"
    ::: Encoding(test_encoding_utf8) <- "UTF-8"

    """  # noqa: E501
    np.testing.assert_array_equal(
        dataset_object,
        np.array(["eĥoŝanĝo ĉiuĵaŭde"]),
    )


@pytest.mark.read_test("test_encoding_latin1")
def test_encoding_latin1(
    dataset_object: np.typing.NDArray[np.str_],
) -> None:
    r"""
    Test of character vector with latin1 encoding.

    Code for creating the object:

    ::: test_encoding_latin1 <- "ca\xf1\xf3n"
    ::: Encoding(test_encoding_latin1) <- "latin1"

    """
    np.testing.assert_array_equal(
        dataset_object,
        np.array(["cañón"]),
    )


@pytest.mark.read_test("test_encoding_bytes")
def test_encoding_bytes(
    dataset_object: np.typing.NDArray[np.bytes_],
) -> None:
    r"""
    Test of character vector with bytes encoding.

    Code for creating the object:

    ::: test_encoding_bytes <- "reba\xf1o"
    ::: Encoding(test_encoding_bytes) <- "bytes"

    """
    np.testing.assert_array_equal(
        dataset_object,
        np.array([b"reba\xf1o"]),
    )


@pytest.mark.filterwarnings("ignore:Unknown encoding. Assumed ASCII.")
@pytest.mark.filterwarnings("ignore:Exception while decoding")
@pytest.mark.read_test("test_encoding_unknown")
def test_encoding_unknown(
    dataset_object: np.typing.NDArray[np.str_ | np.bytes_],
    rdata_format_version: Literal[2, 3],
) -> None:
    r"""
    Test of character vector with unknown encoding.

    Treatment of unknown encoding differs between versions:
        - For version 2 it is treated as ascii if possible, else bytes (with a
          warning).
        - For version 3 the default encoding is used.

    Code for creating the object:

    ::: latin1_str <- "\xcd\xf1igo"
    ::: Encoding(latin1_str) <- "latin1"
    ::: test_encoding_unknown <-  enc2native(latin1_str)
    ::: Encoding(test_encoding_unknown) <- "unknown"

    """
    match rdata_format_version:
        case 2:
            # There are 2 possibilities, depending on the native encoding
            reference_utf8 = np.array([b"\xc3\x8d\xc3\xb1igo"])
            reference_latin1 = np.array([b"\xcd\xf1igo"])

            matches_utf8 = np.equal(dataset_object, reference_utf8)
            matches_latin1 = np.equal(dataset_object, reference_latin1)

            assert matches_utf8 or matches_latin1
        case 3:
            np.testing.assert_array_equal(
                dataset_object,
                np.array(["Íñigo"]),
            )


@pytest.mark.read_test("test_namespace")
def test_namespace(
    dataset_object: rdata.conversion.RNamespace,
) -> None:
    r"""
    Test of namespace objects.

    Code for creating the object:

    ::: test_namespace <- asNamespace("stats")

    """
    assert isinstance(dataset_object, rdata.conversion.RNamespace)
    assert dataset_object.name == "stats"
    assert isinstance(dataset_object.version, str)
