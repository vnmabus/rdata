# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]


## [1.1.0] - 2026-06-11

### Added

- ![Documentation] Add changelog.
- ![R parsing] Parsing of binary RDA and RDS files is now supported.
- ![R parsing] Parsing of namespace objects is now supported.

## [1.0.0] - 2025-08-15

### Added

- ![Documentation] Add accompanying paper.
- ![Python to R conversion] Add functionality for writing RDS and RDA files
  from Python objects.
- ![R to Python conversion] Add support for named vectors.

## [0.11.2] - 2024-03-04

### Added

- ![Documentation] Add issue and PR templates.
- ![Documentation] Add page listing default conversions.
- ![R parsing] Parsing of ASCII RDA and RDS files is now supported.
- ![R to Python conversion] Add convenience functions for performing parsing
  and conversion in one step.

### Changed

- ![R parsing] XDR uses now a faster parser that does not depend on the
  (deprecated) module `xdrlib` of the standard library.

### Fixed

- ![R parsing] Fixed bug in ALTREP for compact sequences.

## [0.10.0] - 2023-11-09

### Added

- ![Documentation] Add contributing guide.
- ![Documentation] Add code of conduct.
- ![Documentation] Add instructions for installing the development version.
- ![Documentation] Add example of loading a RDS dataset from a URL.
- ![Documentation] Add example of loading a RDA dataset from CRAN.

### Changed

- ![R to Python conversion] Improve NA handling for logical and integer types.

## [0.9.1] - 2023-10-19

### Added

- ![Documentation] Add version selection in the documentation.
- ![Documentation] Add examples to the documentation.
- ![Infrastructure] Add functionality to try the package online.

### Changed

- ![Documentation] Improve landing page of the documentation.

## [0.9] - 2022-09-02

### Added

- ![R parsing] Add support for R functions.
- ![R parsing] Add support for external pointers.

## [0.8] - 2022-06-07

### Added

- ![R parsing] Add support RDS file format.

## [0.7] - 2022-03-24

### Added

- ![Infrastructure] Add citation file.
- ![R to Python conversion] Add support for matrices with named dimensions.

### Fixed

- ![Infrastructure] The packaged version can now be introspected properly from
  inside Python.

## [0.6] - 2021-11-03

### Changed

- ![R to Python conversion] Preserve row names of `data.frame` objects.

## [0.5] - 2021-04-22

### Added

- ![R parsing] Add support for parsing of ALTREPs.
- ![R parsing] Parse ALTREP of compact sequences.
- ![R parsing] Parse ALTREP of deferred strings.

### Changed

- ![R parsing] Parse attributes for pairlist objects.

## [0.4] - 2021-02-28

### Added

- ![R parsing] Add support for reading S4 objects.
- ![R parsing] Add support for reading environment objects.

## [0.3] - 2021-01-18

### Added

- ![R parsing] Version 3 of R files can now be correctly parsed.
- ![R to Python conversion] Add option `default_encoding` to use a default
  encoding, overriding the one specified in the file, if any.
- ![R to Python conversion] NA strings are now converted to `None`.
- ![R to Python conversion] Malformed strings are now accepted and converted
  to `bytes` with a warning.
- ![R to Python conversion] Add option `force_default_encoding` to force all
  strings to be decoded using the default encoding, even when they include
  encoding information.

### Changed

- ![R parsing] The provided default encoding is now used for strings with no
  defined encoding.

### Fixed

- ![R parsing] The empty string is now returned also for strings with length
  equal to -1.

## [0.2.3] - 2020-12-19

### Added

- ![Documentation] Add installation guide.
- ![Typing] The package now provides type hints usable by end users.

### Fixed

- ![R parsing] The empty string is now correctly parsed.

## [0.2.2] - 2020-09-18

### Added

- ![Documentation] Add basic usage guide and examples.

### Changed

- ![Infrastructure] Move project configuration to pyproject.toml, instead of
  setuptools.

## [0.2.1] - 2018-10-16

### Fixed

- ![R to Python conversion] Multidimensional matrices were previously being
  read using C memory layout, but R uses FORTRAN memory layout.

## [0.2] - 2018-09-10

### Added

- ![R parsing] Added support for simple R expressions.
- ![R to Python conversion] Added support for `ordered` objects, and
  improved `factor` conversion.

## [0.1.2] - 2018-08-25

### Added

- ![R parsing] Added support for logical (boolean) vectors.
- ![R parsing] Added support for complex numbers.


## [0.1.1] - 2018-08-09

### Added

- ![R to Python conversion] Unknown strings are now accepted with a warning.
- ![R to Python conversion] Added support for class inheritance.
- ![R to Python conversion] Add support for ordered columns in dataframes.
- ![R to Python conversion] Add support for `factor` objects with missing data.

### Changed

- ![R to Python conversion] `SimpleConverter` objects can now be reused.

## [0.1] - 2018-08-08

Initial version of the package.

[Documentation]: https://img.shields.io/badge/Documentation-orange
[Infrastructure]: https://img.shields.io/badge/Infrastructure-grey
[R parsing]: https://img.shields.io/badge/R%20parsing-darkgreen
[R to Python conversion]: https://img.shields.io/badge/R%20to%20Python%20conversion-blue
[Python to R conversion]: https://img.shields.io/badge/Python%20to%20R%20conversion-red
[Typing]: https://img.shields.io/badge/Typing-purple