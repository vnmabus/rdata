"""Tests of parsing and conversion."""

import itertools
import unittest
from collections import ChainMap
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray

import rdata

TESTDATA_PATH = rdata.TESTDATA_PATH


class SimpleTests(unittest.TestCase):
    """Collection of simple test cases."""

    def test_opened_file(self) -> None:
        """Test that an opened file can be passed to parse_file."""
        with (TESTDATA_PATH / "test_vector.rda").open("rb") as f:
            parsed = rdata.parser.parse_file(f)
            converted = rdata.conversion.convert(parsed)

            assert isinstance(converted, dict)

    def test_opened_string(self) -> None:
        """Test that a string can be passed to parse_file."""
        parsed = rdata.parser.parse_file(
            str(TESTDATA_PATH / "test_vector.rda"),
        )
        converted = rdata.conversion.convert(parsed)

        assert isinstance(converted, dict)

    def test_logical(self) -> None:
        """Test parsing of logical vectors."""
        data = rdata.read_rda(TESTDATA_PATH / "test_logical.rda")

        np.testing.assert_equal(data, {
            "test_logical": np.array([True, True, False, True, False]),
        })

    def test_nullable_logical(self) -> None:
        """Test parsing of logical vectors containing NA."""
        data = rdata.read_rda(TESTDATA_PATH / "test_nullable_logical.rda")

        array = data["test_nullable_logical"]
        np.testing.assert_array_equal(
            array.data,
            np.array([True, False, True]),
        )
        np.testing.assert_array_equal(
            array.mask,
            np.array([False, False, True]),
        )

    def test_nullable_int(self) -> None:
        """Test parsing of integer vectors containing NA."""
        data = rdata.read_rda(TESTDATA_PATH / "test_nullable_int.rda")

        array = data["test_nullable_int"]
        np.testing.assert_array_equal(
            array.data,
            np.array([313, -12, -2**31]),
        )
        np.testing.assert_array_equal(
            array.mask,
            np.array([False, False, True]),
        )

    def test_vector(self) -> None:
        """Test parsing of numerical vectors."""
        data = rdata.read_rda(TESTDATA_PATH / "test_vector.rda")

        np.testing.assert_equal(data, {
            "test_vector": np.array([1.0, 2.0, 3.0]),
        })

    def test_empty_string(self) -> None:
        """Test that the empty string is parsed correctly."""
        data = rdata.read_rda(TESTDATA_PATH / "test_empty_str.rda")

        np.testing.assert_equal(data, {
            "test_empty_str": [""],
        })

    def test_ascii_empty_string(self) -> None:
        """Test that the empty string is parsed correctly from ascii file."""
        data = rdata.read_rds(TESTDATA_PATH / "test_ascii_empty_str.rds")
        np.testing.assert_equal(data, [""])

    def test_na_string(self) -> None:
        """Test that the NA string is parsed correctly."""
        data = rdata.read_rda(TESTDATA_PATH / "test_na_string.rda")

        np.testing.assert_equal(data, {
            "test_na_string": [None],
        })

    def test_ascii_na_string(self) -> None:
        """Test that the NA string is parsed correctly."""
        # File created in R with
        # saveRDS(as.character(NA), file="test_ascii_na_string.rds", ascii=TRUE, compress=FALSE)  # noqa: E501
        data = rdata.read_rds(TESTDATA_PATH / "test_ascii_na_string.rds")
        np.testing.assert_equal(data, [None])

    def test_complex(self) -> None:
        """Test that complex numbers can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_complex.rda")

        np.testing.assert_equal(data, {
            "test_complex": np.array([1 + 2j, 2, 0, 1 + 3j, -1j]),
        })

    def test_matrix(self) -> None:
        """Test that a matrix can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_matrix.rda")

        np.testing.assert_equal(data, {
            "test_matrix": np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]),
        })

    def test_named_matrix(self) -> None:
        """Test that a named matrix can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_named_matrix.rda")

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
            data["test_named_matrix"],
            reference,
        )

    def test_half_named_matrix(self) -> None:
        """Test that a named matrix with no name for a dim can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_half_named_matrix.rda")

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
            data["test_half_named_matrix"],
            reference,
        )

    def test_full_named_matrix(self) -> None:
        """Test that a named matrix with dim names can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_full_named_matrix.rda")

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
            data["test_full_named_matrix"],
            reference,
        )

    def test_full_named_matrix_rds(self) -> None:
        """Test that a named matrix with dim names can be parsed."""
        data = rdata.read_rds(TESTDATA_PATH / "test_full_named_matrix.rds")

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
            data,
            reference,
        )

    def test_list(self) -> None:
        """Test that list can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_list.rda")

        np.testing.assert_equal(data, {
            "test_list":
                [
                    np.array([1.0]),
                    ["a", "b", "c"],
                    np.array([2.0, 3.0]),
                    ["hi"],
                ],
        })

    @pytest.mark.filterwarnings("ignore:Missing constructor")
    def test_file(self) -> None:
        """Test that external pointers can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_file.rda")

        np.testing.assert_equal(data, {
            "test_file": [5],
        })

    def test_expression(self) -> None:
        """Test that expressions can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_expression.rda")

        np.testing.assert_equal(data, {
            "test_expression": rdata.conversion.RExpression([
                rdata.conversion.RLanguage(
                    ["^", "base", "exponent"],
                    attributes={},
                ),
            ]),
        })

    def test_builtin(self) -> None:
        """Test that builtin functions can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_builtin.rda")

        np.testing.assert_equal(data, {
            "test_builtin": rdata.conversion.RBuiltin(name="abs"),
        })

    def test_minimal_function_uncompiled(self) -> None:
        """Test that a minimal function can be parsed."""
        data = rdata.read_rda(
            TESTDATA_PATH / "test_minimal_function_uncompiled.rda",
        )

        converted_fun = data["test_minimal_function_uncompiled"]

        assert isinstance(
            converted_fun,
            rdata.conversion.RFunction,
        )

        np.testing.assert_equal(converted_fun.environment, ChainMap({}))
        np.testing.assert_equal(converted_fun.formals, None)
        np.testing.assert_equal(converted_fun.body, None)
        np.testing.assert_equal(
            converted_fun.source,
            "test_minimal_function_uncompiled <- function() NULL\n",
        )

    @pytest.mark.filterwarnings("ignore:Missing constructor")
    def test_minimal_function(self) -> None:
        """Test that a minimal function (compiled) can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_minimal_function.rda")

        converted_fun = data["test_minimal_function"]

        assert isinstance(
            converted_fun,
            rdata.conversion.RFunction,
        )

        np.testing.assert_equal(converted_fun.environment, ChainMap({}))
        np.testing.assert_equal(converted_fun.formals, None)

        converted_body = converted_fun.body

        assert isinstance(
            converted_body,
            rdata.conversion.RBytecode,
        )

        np.testing.assert_equal(converted_body.code, np.array([12, 17, 1]))
        np.testing.assert_equal(converted_body.attributes, {})

        np.testing.assert_equal(
            converted_fun.source,
            "test_minimal_function <- function() NULL\n",
        )

    def test_empty_function_uncompiled(self) -> None:
        """Test that a simple function can be parsed."""
        data = rdata.read_rda(
            TESTDATA_PATH / "test_empty_function_uncompiled.rda",
        )

        converted_fun = data["test_empty_function_uncompiled"]

        assert isinstance(
            converted_fun,
            rdata.conversion.RFunction,
        )

        np.testing.assert_equal(converted_fun.environment, ChainMap({}))
        np.testing.assert_equal(converted_fun.formals, None)
        assert isinstance(converted_fun.body, rdata.conversion.RLanguage)
        np.testing.assert_equal(
            converted_fun.source,
            "test_empty_function_uncompiled <- function() {}\n",
        )

    @pytest.mark.filterwarnings("ignore:Missing constructor")
    def test_empty_function(self) -> None:
        """Test that a simple function (compiled) can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_empty_function.rda")

        converted_fun = data["test_empty_function"]

        assert isinstance(
            converted_fun,
            rdata.conversion.RFunction,
        )

        np.testing.assert_equal(converted_fun.environment, ChainMap({}))
        np.testing.assert_equal(converted_fun.formals, None)

        converted_body = converted_fun.body

        assert isinstance(
            converted_body,
            rdata.conversion.RBytecode,
        )

        np.testing.assert_equal(converted_body.code, np.array([12, 17, 1]))
        np.testing.assert_equal(converted_body.attributes, {})

        np.testing.assert_equal(
            converted_fun.source,
            "test_empty_function <- function() {}\n",
        )

    @pytest.mark.filterwarnings("ignore:Missing constructor")
    def test_function(self) -> None:
        """Test that functions can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_function.rda")

        converted_fun = data["test_function"]

        assert isinstance(
            converted_fun,
            rdata.conversion.RFunction,
        )

        np.testing.assert_equal(converted_fun.environment, ChainMap({}))
        np.testing.assert_equal(converted_fun.formals, None)

        converted_body = converted_fun.body

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
            converted_fun.source,
            "test_function <- function() {print(\"Hello\")}\n",
        )

    @pytest.mark.filterwarnings("ignore:Missing constructor")
    def test_function_arg(self) -> None:
        """Test that functions can be parsed."""
        data = rdata.read_rda(TESTDATA_PATH / "test_function_arg.rda")

        converted_fun = data["test_function_arg"]

        assert isinstance(
            converted_fun,
            rdata.conversion.RFunction,
        )

        np.testing.assert_equal(converted_fun.environment, ChainMap({}))
        np.testing.assert_equal(converted_fun.formals, {"a": NotImplemented})

        converted_body = converted_fun.body

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
            converted_fun.source,
            "test_function_arg <- function(a) {print(a)}\n",
        )

    def test_encodings(self) -> None:
        """Test of differents encodings."""
        with self.assertWarns(
            UserWarning,
            msg="Unknown encoding. Assumed ASCII.",
        ):
            data = rdata.read_rda(
                TESTDATA_PATH / "test_encodings.rda",
            )

            np.testing.assert_equal(data, {
                "test_encoding_utf8": ["eĥoŝanĝo ĉiuĵaŭde"],
                "test_encoding_latin1": ["cañón"],
                "test_encoding_bytes": [b"reba\xf1o"],
                "test_encoding_latin1_implicit": [b"\xcd\xf1igo"],
            })

    def test_encodings_v3(self) -> None:
        """Test encodings in version 3 format."""
        data = rdata.read_rda(TESTDATA_PATH / "test_encodings_v3.rda")

        np.testing.assert_equal(data, {
            "test_encoding_utf8": ["eĥoŝanĝo ĉiuĵaŭde"],
            "test_encoding_latin1": ["cañón"],
            "test_encoding_bytes": [b"reba\xf1o"],
            "test_encoding_latin1_implicit": ["Íñigo"],
        })

    def test_dataframe(self) -> None:
        """Test dataframe conversion."""
        for f in ("test_dataframe.rda", "test_dataframe_v3.rda"):
            with self.subTest(file=f):
                data = rdata.read_rda(TESTDATA_PATH / f)

                pd.testing.assert_frame_equal(
                    data["test_dataframe"],
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

    def test_dataframe_rds(self) -> None:
        """Test dataframe conversion."""
        for f in ("test_dataframe.rds", "test_dataframe_v3.rds"):
            with self.subTest(file=f):
                data = rdata.read_rds(TESTDATA_PATH / f)

                pd.testing.assert_frame_equal(
                    data,
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

    def test_dataframe_rownames(self) -> None:
        """Test dataframe conversion."""
        data = rdata.read_rda(TESTDATA_PATH / "test_dataframe_rownames.rda")

        pd.testing.assert_frame_equal(
            data["test_dataframe_rownames"],
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

    def test_ts(self) -> None:
        """Test time series conversion."""
        data = rdata.read_rda(TESTDATA_PATH / "test_ts.rda")

        pd.testing.assert_series_equal(
            data["test_ts"],
            pd.Series({
                2000 + Fraction(2, 12): 1.0,
                2000 + Fraction(3, 12): 2.0,
                2000 + Fraction(4, 12): 3.0,
            }),
        )

    def test_s4(self) -> None:
        """Test parsing of S4 classes."""
        with pytest.warns(UserWarning, match="Missing constructor"):
            data = rdata.read_rda(TESTDATA_PATH / "test_s4.rda")

        np.testing.assert_equal(data, {
            "test_s4": SimpleNamespace(
                age=np.array(28),
                name=["Carlos"],
                **{"class": ["Person"]},
            ),
        })

    def test_environment(self) -> None:
        """Test parsing of environments."""
        parsed = rdata.parser.parse_file(
            TESTDATA_PATH / "test_environment.rda",
        )
        converted = rdata.conversion.convert(parsed)

        dict_env = {"string": ["test"]}
        empty_global_env: dict[str, Any] = {}

        np.testing.assert_equal(converted, {
            "test_environment": ChainMap(dict_env, ChainMap(empty_global_env)),
        })

        global_env = {"global": "test"}

        converted_global = rdata.conversion.convert(
            parsed,
            global_environment=global_env,
        )

        np.testing.assert_equal(converted_global, {
            "test_environment": ChainMap(dict_env, ChainMap(global_env)),
        })

    def test_emptyenv(self) -> None:
        """Test parsing the empty environment."""
        data = rdata.read_rda(TESTDATA_PATH / "test_emptyenv.rda")

        assert data == {
            "test_emptyenv": ChainMap({}),
        }

    def test_empty_list(self) -> None:
        """Test parsing the empty list."""
        data = rdata.read_rds(TESTDATA_PATH / "test_empty_list.rds")
        assert data == []

    def test_empty_named_list(self) -> None:
        """Test parsing the empty list."""
        data = rdata.read_rds(TESTDATA_PATH / "test_empty_named_list.rds")
        assert data == {}

    def test_list_attrs(self) -> None:
        """Test that lists accept attributes."""
        data = rdata.read_rda(TESTDATA_PATH / "test_list_attrs.rda")

        np.testing.assert_equal(data, {
            "test_list_attrs": [["list"], [5]],
        })

    def test_altrep_compact_intseq(self) -> None:
        """Test alternative representation of sequences of ints."""
        data = rdata.read_rda(TESTDATA_PATH / "test_altrep_compact_intseq.rda")

        np.testing.assert_equal(data, {
            "test_altrep_compact_intseq": np.arange(1000),
        })

    def test_altrep_compact_intseq_asymmetric(self) -> None:
        """
        Test alternative representation of sequences of ints.

        This test an origin different from 0, to reproduce
        issue #29.
        """
        data = rdata.read_rda(
            TESTDATA_PATH / "test_altrep_compact_intseq_asymmetric.rda",
        )

        np.testing.assert_equal(data, {
            "test_altrep_compact_intseq_asymmetric": np.arange(-5, 6),
        })

    def test_altrep_compact_realseq(self) -> None:
        """Test alternative representation of sequences of ints."""
        data = rdata.read_rda(
            TESTDATA_PATH / "test_altrep_compact_realseq.rda",
        )

        np.testing.assert_equal(data, {
            "test_altrep_compact_realseq": np.arange(1000.0),
        })

    def test_altrep_compact_realseq_asymmetric(self) -> None:
        """
        Test alternative representation of sequences of ints.

        This test an origin different from 0, to reproduce
        issue #29.
        """
        data = rdata.read_rda(
            TESTDATA_PATH / "test_altrep_compact_realseq_asymmetric.rda",
        )

        np.testing.assert_equal(data, {
            "test_altrep_compact_realseq_asymmetric": np.arange(-5.0, 6.0),
        })

    def test_altrep_deferred_string(self) -> None:
        """Test alternative representation of deferred strings."""
        data = rdata.read_rda(
            TESTDATA_PATH / "test_altrep_deferred_string.rda",
        )

        np.testing.assert_equal(data, {
            "test_altrep_deferred_string": [
                "1", "2.3", "10000",
                "1e+05", "-10000", "-1e+05",
                "0.001", "1e-04", "1e-05",
            ],
        })

    def test_altrep_wrap_real(self) -> None:
        """Test alternative representation of wrap_real."""
        data = rdata.read_rda(
            TESTDATA_PATH / "test_altrep_wrap_real.rda",
        )

        np.testing.assert_equal(data, {
            "test_altrep_wrap_real": [3],
        })

    def test_altrep_wrap_string(self) -> None:
        """Test alternative representation of wrap_string."""
        data = rdata.read_rda(TESTDATA_PATH / "test_altrep_wrap_string.rda")

        np.testing.assert_equal(data, {
            "test_altrep_wrap_string": ["Hello"],
        })

    def test_altrep_wrap_logical(self) -> None:
        """Test alternative representation of wrap_logical."""
        data = rdata.read_rda(TESTDATA_PATH / "test_altrep_wrap_logical.rda")

        np.testing.assert_equal(data, {
            "test_altrep_wrap_logical": [True],
        })

    def test_ascii(self) -> None:
        """Test ascii files."""
        ref_ma = np.ma.array(  # type: ignore[no-untyped-call]
            data=[True],
            mask=[True],
            fill_value=True,
        )
        ref = [[1.1], [2], [3. + 4.j], ref_ma, ["aä"]]

        for tag, v, ext in itertools.product(
                ("", "win_"),
                (2, 3),
                ("rda", "rds"),
        ):
            f = f"test_ascii_{tag}v{v}.{ext}"
            with self.subTest(file=f):
                parsed = rdata.parser.parse_file(
                    TESTDATA_PATH / f,
                )
                converted = rdata.conversion.convert(parsed)

                if ext == "rda":
                    np.testing.assert_equal(converted, {"data": ref})
                    ma = converted["data"][3]
                else:
                    np.testing.assert_equal(converted, ref)
                    ma = converted[3]

                # Test masked array separately
                np.testing.assert_equal(ma.data, ref_ma.data)
                np.testing.assert_equal(ma.mask, ref_ma.mask)
                np.testing.assert_equal(ma.mask, ref_ma.mask)
                np.testing.assert_equal(ma.get_fill_value(),
                                        ref_ma.get_fill_value())

    def test_ascii_characters(self) -> None:
        """Test reading string with all ascii printable characters."""
        data = rdata.read_rds(TESTDATA_PATH / "test_ascii_chars.rds")
        assert data == "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\v\f\r\n", data  # noqa: E501

    def test_ascii_ascii_characters(self) -> None:
        """Test reading string with all ascii printable characters."""
        data = rdata.read_rds(TESTDATA_PATH / "test_ascii_ascii_chars.rds")
        assert data == "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\v\f\r\n", data  # noqa: E501

    def test_nan_inf(self) -> None:
        """Test reading nan and inf."""
        data = rdata.read_rds(TESTDATA_PATH / "test_nan_inf.rds")
        np.testing.assert_equal(data, [0., np.nan, np.inf, -np.inf])

    def test_ascii_nan_inf(self) -> None:
        """Test reading nan and inf in ascii."""
        data = rdata.read_rds(TESTDATA_PATH / "test_ascii_nan_inf.rds")
        np.testing.assert_equal(data, [0., np.nan, np.inf, -np.inf])



if __name__ == "__main__":
    unittest.main()
