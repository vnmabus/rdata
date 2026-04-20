"""Configuration for Pytest."""
from typing import Literal, get_args

import pytest

from rdata import TESTDATA_PATH, read_rda, read_rds
from rdata.parser import parse_file
from rdata.testing import execute_r_data_source

SerializationType = Literal["xdr", "ascii", "binary"]
SuffixType = Literal["rda", "rds"]
VersionType = Literal[2, 3]


def pytest_configure(config: pytest.Config) -> None:
    """
    Add custom markers.

    Add a marker ``read_test`` to mark tests that read an R object.

    """
    config.addinivalue_line(
        "markers",
        "read_test: mark test that reads a R object from a file.",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Add an option to generate the R datasets.

    RScript must be installed to use this option.

    Args:
        parser: The Pytest parser.

    """
    parser.addoption(
        "--generate-datasets",
        action="store_true",
        help=(
            "Generate the R datasets. "
            "RScript must be installed in the system."
        ),
    )


# Shared Fixtures

# Based on https://docs.pytest.org/en/stable/how-to/fixtures.html#using-markers-to-pass-data-to-fixtures  # noqa: E501
@pytest.fixture
def dataset_base_filename(request: pytest.FixtureRequest) -> str | None:
    """
    Fixture to get the base filename corresponding to a test.

    The base filename is that passed to the ``read_test`` marker.

    """
    # Untyped, see: https://github.com/pytest-dev/pytest/issues/13888
    node = request.node
    assert isinstance(node, pytest.Item | pytest.Collector)
    marker = node.get_closest_marker("read_test")
    if marker is None:
        return None

    name = marker.args[0]
    assert isinstance(name, str)
    return name


@pytest.fixture(params=get_args(VersionType))
def rdata_format_version(request: pytest.FixtureRequest) -> VersionType:
    """Version of the Rdata format to use."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=get_args(SuffixType))
def rdata_format_suffix(request: pytest.FixtureRequest) -> SuffixType:
    """Suffix of the Rdata format to use."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=get_args(SerializationType))
def rdata_format_serialization(
    rdata_format_suffix: SuffixType,
    request: pytest.FixtureRequest,
) -> SerializationType:
    """Serialization type of the Rdata format to use."""
    format_type = request.param
    if format_type == "binary" and rdata_format_suffix == "rda":
        pytest.skip("RDA format does not support native binary serialization.")

    return format_type  # type: ignore[no-any-return]


@pytest.fixture
def dataset_filename_internal(
    dataset_base_filename: str | None,
    rdata_format_version: VersionType,
    rdata_format_serialization: SerializationType,
    rdata_format_suffix: SuffixType,
) -> str:
    """
    Fixture to get the dataset filename.

    The dataset filename includes the base filename and a suffix depending on
    the format.
    This is parameterized with all possible formats, to test all combinations.

    Warning:
        This is not intended to be used in tests: use `dataset_filename`
        instead.

    """
    assert dataset_base_filename

    return (
        f"{dataset_base_filename}__{rdata_format_serialization}__version_"
        f"{rdata_format_version}.{rdata_format_suffix}"
    )


@pytest.fixture
def dataset_path_internal(
    dataset_filename_internal: str,
) -> str:
    """
    Fixture to get the full dataset path.

    The path is of the form <TESTDATA_PATH>/generated/<dataset_filename>.

    """
    return str(TESTDATA_PATH / "generated" / dataset_filename_internal)


def to_r_string(string: str) -> str:
    """
    Convert string object to a R source string.

    This surrounds the string with the proper quotations for the R language
    and escapes special characters.

    """
    escaped = string.replace("\\", "\\\\")
    return f"\"{escaped}\""


@pytest.fixture
def create_dataset(
    dataset_base_filename: str | None,
    rdata_format_version: VersionType,
    rdata_format_serialization: SerializationType,
    rdata_format_suffix: SuffixType,
    dataset_path_internal: str,
    request: pytest.FixtureRequest,
) -> None:
    """
    This fixture creates the dataset if ``create-datasets`` is passed in CLI.

    It extracts the docstring from the test to find the code that generates
    the test object, and executes it, saving the object with the corresponding
    format.

    """
    if dataset_base_filename is None or not request.config.getoption(
        "--generate-datasets",
    ):
        return

    save_code_rds = f"""
    con <- file({to_r_string(dataset_path_internal)}, "wb")
    serialize(
        {dataset_base_filename},
        con,
        ascii = {"TRUE" if rdata_format_serialization == "ascii" else "FALSE"},
        xdr = {"TRUE" if rdata_format_serialization == "xdr" else "FALSE"},
        version = {rdata_format_version}
    )
    close(con)
    """

    save_code_rda = f"""
    save(
        {dataset_base_filename},
        file = {to_r_string(dataset_path_internal)},
        ascii = {"TRUE" if rdata_format_serialization == "ascii" else "FALSE"},
        version = {rdata_format_version}
    )
    """

    save_code = (
        save_code_rds if rdata_format_suffix == "rds"
        else save_code_rda
    )

    execute_r_data_source(
        request.function,
        append=save_code,
    )


@pytest.fixture
def dataset_filename(
    dataset_filename_internal: str,
    create_dataset: None,  # noqa: ARG001
) -> str:
    """
    Fixture to get the dataset filename.

    The dataset filename includes the base filename and a suffix depending on
    the format.
    This is parameterized with all possible formats, to test all combinations.

    Note:
        Using this fixtures creates the dataset if specified in the command
        line.

    """
    return dataset_filename_internal


@pytest.fixture
def dataset_path(
    dataset_path_internal: str,
    create_dataset: None,  # noqa: ARG001
) -> str:
    """
    Fixture to get the full dataset path.

    The path is of the form <TESTDATA_PATH>/generated/<dataset_filename>.

    Note:
        Using this fixtures creates the dataset if specified in the command
        line.

    """
    return dataset_path_internal


@pytest.fixture
def dataset_object_parsed(
    dataset_base_filename: str,
    dataset_path: str,
    rdata_format_suffix: SuffixType,
    create_dataset: None,  # noqa: ARG001
) -> object:
    """
    Fixture to get the parsed object in the dataset.

    For RDA, the object is extracted from the dictionary.

    Note:
        Using this fixtures creates the dataset if specified in the command
        line.

    """
    parsed = parse_file(
        dataset_path,
        extension=f".{rdata_format_suffix}",
    )

    match rdata_format_suffix:
        case "rds":
            return parsed.object
        case "rda":
            tag = parsed.object.tag
            assert tag
            assert tag.value.value == dataset_base_filename.encode("utf8")
            return parsed.object.value[0]


@pytest.fixture
def dataset_object(
    dataset_base_filename: str,
    dataset_path: str,
    rdata_format_suffix: SuffixType,
    create_dataset: None,  # noqa: ARG001
    request: pytest.FixtureRequest,
) -> object:
    """
    Fixture to get the stored object in the dataset.

    For RDA, the object is extracted from the dictionary.

    Note:
        Using this fixtures creates the dataset if specified in the command
        line.

    """
    # Untyped, see: https://github.com/pytest-dev/pytest/issues/13888
    node = request.node
    assert isinstance(node, pytest.Item | pytest.Collector)
    marker = node.get_closest_marker("read_test")
    if marker is None:
        msg = "No \"read_test\" marker found."
        raise ValueError(msg)

    kwargs = marker.kwargs

    match rdata_format_suffix:
        case "rds":
            return read_rds(dataset_path, **kwargs)
        case "rda":
            namespace = read_rda(dataset_path, **kwargs)
            assert len(namespace) == 1
            return namespace[dataset_base_filename]
