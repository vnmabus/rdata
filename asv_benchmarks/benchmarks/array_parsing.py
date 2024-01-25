"""Benchmarks for array parsing time."""
import rdata
from rdata.testing import execute_r_data_source


class TimeArrayParsing:
    """
    A test for the time that it takes to parse arrays of different sizes.

    The following R code is used to create arrays of different sizes:

    ::: for (i in 1:MAX_TESTS) {
    :::     n = 2^i * 1024^2
    :::     saveRDS(
    :::         runif(n),
    :::         file=sprintf("array_%s.rds", i),
    :::         compress=FALSE,
    :::     )
    ::: }
    """
    MAX_TESTS = 5
    params = range(MAX_TESTS)

    def setup_cache(self) -> None:
        """Initialize the data."""
        execute_r_data_source(self, MAX_TESTS=self.MAX_TESTS)

    def time_array(self, i: int) -> None:
        """Test the time that it takes to parse an array."""
        rdata.parser.parse_file(f"array_{i + 1}.rds")
