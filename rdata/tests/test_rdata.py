"""Tests of parsing and conversion."""

import unittest
from collections import ChainMap
from fractions import Fraction
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pandas as pd

import rdata

TESTDATA_PATH = rdata.TESTDATA_PATH


class SimpleTests(unittest.TestCase):
    """Collection of simple test cases."""

    def test_opened_file(self) -> None:
        """Test that an opened file can be passed to parse_file."""
        with open(TESTDATA_PATH / "test_vector.rda") as f:
            parsed = rdata.parser.parse_file(f)
            converted = rdata.conversion.convert(parsed)

            self.assertIsInstance(converted, dict)

    def test_opened_string(self) -> None:
        """Test that a string can be passed to parse_file."""
        parsed = rdata.parser.parse_file(
            str(TESTDATA_PATH / "test_vector.rda"),
        )
        converted = rdata.conversion.convert(parsed)

        self.assertIsInstance(converted, dict)

    def test_logical(self) -> None:
        """Test parsing of logical vectors."""
        parsed = rdata.parser.parse_file(TESTDATA_PATH / "test_logical.rda")
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_logical": np.array([True, True, False, True, False]),
        })

    def test_vector(self) -> None:
        """Test parsing of numerical vectors."""
        parsed = rdata.parser.parse_file(TESTDATA_PATH / "test_vector.rda")
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_vector": np.array([1.0, 2.0, 3.0]),
        })

    def test_empty_string(self) -> None:
        """Test that the empty string is parsed correctly."""
        parsed = rdata.parser.parse_file(TESTDATA_PATH / "test_empty_str.rda")
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_empty_str": [""],
        })

    def test_na_string(self) -> None:
        """Test that the NA string is parsed correctly."""
        parsed = rdata.parser.parse_file(
            TESTDATA_PATH / "test_na_string.rda",
        )
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_na_string": [None],
        })

    def test_complex(self) -> None:
        """Test that complex numbers can be parsed."""
        parsed = rdata.parser.parse_file(TESTDATA_PATH / "test_complex.rda")
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_complex": np.array([1 + 2j, 2, 0, 1 + 3j, -1j]),
        })

    def test_matrix(self) -> None:
        """Test that a matrix can be parsed."""
        parsed = rdata.parser.parse_file(TESTDATA_PATH / "test_matrix.rda")
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_matrix": np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]),
        })

    def test_list(self) -> None:
        """Test that list can be parsed."""
        parsed = rdata.parser.parse_file(TESTDATA_PATH / "test_list.rda")
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_list":
                [
                    np.array([1.0]),
                    ['a', 'b', 'c'],
                    np.array([2.0, 3.0]),
                    ['hi'],
                ],
        })

    def test_expression(self) -> None:
        """Test that expressions can be parsed."""
        parsed = rdata.parser.parse_file(TESTDATA_PATH / "test_expression.rda")
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_expression": rdata.conversion.RExpression([
                rdata.conversion.RLanguage(['^', 'base', 'exponent']),
            ]),
        })

    def test_encodings(self) -> None:
        """Test of differents encodings."""
        with self.assertWarns(
            UserWarning,
            msg="Unknown encoding. Assumed ASCII.",
        ):
            parsed = rdata.parser.parse_file(
                TESTDATA_PATH / "test_encodings.rda",
            )
            converted = rdata.conversion.convert(parsed)

            np.testing.assert_equal(converted, {
                "test_encoding_utf8": ["eĥoŝanĝo ĉiuĵaŭde"],
                "test_encoding_latin1": ["cañón"],
                "test_encoding_bytes": [b"reba\xf1o"],
                "test_encoding_latin1_implicit": [b"\xcd\xf1igo"],
            })

    def test_encodings_v3(self) -> None:
        """Test encodings in version 3 format."""
        parsed = rdata.parser.parse_file(
            TESTDATA_PATH / "test_encodings_v3.rda",
        )
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_encoding_utf8": ["eĥoŝanĝo ĉiuĵaŭde"],
            "test_encoding_latin1": ["cañón"],
            "test_encoding_bytes": [b"reba\xf1o"],
            "test_encoding_latin1_implicit": ["Íñigo"],
        })

    def test_dataframe(self) -> None:
        """Test dataframe conversion."""
        for f in ("test_dataframe.rda", "test_dataframe_v3.rda"):
            with self.subTest(file=f):
                parsed = rdata.parser.parse_file(
                    TESTDATA_PATH / f,
                )
                converted = rdata.conversion.convert(parsed)

                pd.testing.assert_frame_equal(
                    converted["test_dataframe"],
                    pd.DataFrame({
                        "class": pd.Categorical(
                            ["a", "b", "b"],
                        ),
                        "value": [1, 2, 3],
                    }),
                )

    def test_ts(self) -> None:
        """Test time series conversion."""
        parsed = rdata.parser.parse_file(TESTDATA_PATH / "test_ts.rda")
        converted = rdata.conversion.convert(parsed)

        pd.testing.assert_series_equal(
            converted["test_ts"],
            pd.Series({
                2000 + Fraction(2, 12): 1.0,
                2000 + Fraction(3, 12): 2.0,
                2000 + Fraction(4, 12): 3.0,
            }),
        )

    def test_s4(self) -> None:
        """Test parsing of S4 classes."""
        parsed = rdata.parser.parse_file(TESTDATA_PATH / "test_s4.rda")
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_s4": SimpleNamespace(
                age=np.array(28),
                name=["Carlos"],
                **{'class': ["Person"]},  # noqa: WPS517
            ),
        })

    def test_environment(self) -> None:
        """Test parsing of environments."""
        parsed = rdata.parser.parse_file(
            TESTDATA_PATH / "test_environment.rda",
        )
        converted = rdata.conversion.convert(parsed)

        dict_env = {'string': ['test']}
        empty_global_env: Dict[str, Any] = {}

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
        parsed = rdata.parser.parse_file(
            TESTDATA_PATH / "test_emptyenv.rda",
        )
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_emptyenv": ChainMap({}),
        })

    def test_list_attrs(self) -> None:
        """Test that lists accept attributes."""
        parsed = rdata.parser.parse_file(TESTDATA_PATH / "test_list_attrs.rda")
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_list_attrs": [['list'], [5]],
        })

    def test_altrep_compact_intseq(self) -> None:
        """Test alternative representation of sequences of ints."""
        parsed = rdata.parser.parse_file(
            TESTDATA_PATH / "test_altrep_compact_intseq.rda",
        )
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_altrep_compact_intseq": np.arange(1000),
        })

    def test_altrep_compact_realseq(self) -> None:
        """Test alternative representation of sequences of ints."""
        parsed = rdata.parser.parse_file(
            TESTDATA_PATH / "test_altrep_compact_realseq.rda",
        )
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_altrep_compact_realseq": np.arange(1000.0),
        })

    def test_altrep_deferred_string(self) -> None:
        """Test alternative representation of deferred strings."""
        parsed = rdata.parser.parse_file(
            TESTDATA_PATH / "test_altrep_deferred_string.rda",
        )
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_altrep_deferred_string": [  # noqa: WPS317
                "1", "2.3", "10000",
                "1e+05", "-10000", "-1e+05",
                "0.001", "1e-04", "1e-05",
            ],
        })

    def test_altrep_wrap_real(self) -> None:
        """Test alternative representation of wrap_real."""
        parsed = rdata.parser.parse_file(
            TESTDATA_PATH / "test_altrep_wrap_real.rda",
        )
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_altrep_wrap_real": [3],
        })

    def test_altrep_wrap_string(self) -> None:
        """Test alternative representation of wrap_string."""
        parsed = rdata.parser.parse_file(
            TESTDATA_PATH / "test_altrep_wrap_string.rda",
        )
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_altrep_wrap_string": ["Hello"],
        })

    def test_altrep_wrap_logical(self) -> None:
        """Test alternative representation of wrap_logical."""
        parsed = rdata.parser.parse_file(
            TESTDATA_PATH / "test_altrep_wrap_logical.rda",
        )
        converted = rdata.conversion.convert(parsed)

        np.testing.assert_equal(converted, {
            "test_altrep_wrap_logical": [True],
        })


if __name__ == "__main__":
    unittest.main()
