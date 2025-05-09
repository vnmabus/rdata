rdata
=====

|build-status| |docs| |coverage| |repostatus| |versions| |pypi| |conda| |zenodo| |pyOpenSci| |joss|

A Python library for R datasets.

..
	Github does not support include in README for dubious security reasons, so
	we copy-paste instead. Also Github does not understand Sphinx directives.
	.. include:: docs/index.rst
	.. include:: docs/usage.rst

The package rdata offers a lightweight way in Python to import and export R datasets/objects stored
in the ".rda" and ".rds" formats.
Its main advantages are:

- It is a pure Python implementation, with no dependencies on the R language or
  related libraries.
  Thus, it can be used anywhere where Python is supported, including the web
  using `Pyodide <https://pyodide.org/>`__.
- It attempts to support all objects that can be meaningfully translated between R and Python.
  As opposed to other solutions, you are no limited to import dataframes or
  data with a particular structure.
- It allows users to easily customize the conversion of R classes to Python
  ones and vice versa.
  Does your data use custom R classes?
  Worry no longer, as it is possible to define custom conversions to the Python
  classes of your choosing.
- It has a permissive license (MIT). As opposed to other packages that depend
  on R libraries and thus need to adhere to the GPL license, you can use rdata
  as a dependency on MIT, BSD or even closed source projects.

Installation
============

Installing a stable release
---------------------------

The rdata package is on PyPi and can be installed using :code:`pip`:

.. code::

   pip install rdata

The package is also available for :code:`conda` using the :code:`conda-forge` channel:

.. code::

   conda install -c conda-forge rdata

Installing a develop version
----------------------------

The current version from the develop branch can be installed as

.. code::

   pip install git+https://github.com/vnmabus/rdata.git@develop

Documentation
=============

The documentation of rdata is in
`ReadTheDocs <https://rdata.readthedocs.io/>`__.

Examples
========

Examples of use are available in
`ReadTheDocs <https://rdata.readthedocs.io/en/stable/auto_examples/>`__.

Citing rdata
============

Please, if you find this software useful in your work, reference it citing the following paper:

.. code-block::

  @article{ramos-carreno+rossi_2024_rdata,
      author = {Ramos-Carreño, Carlos and Rossi, Tuomas},
      doi = {10.21105/joss.07540},
      journal = {Journal of Open Source Software},
      month = dec,
      number = {104},
      pages = {1--4},
      title = {{rdata: A Python library for R datasets}},
      url = {https://joss.theoj.org/papers/10.21105/joss.07540#},
      volume = {9},
      year = {2024}
  }

You can additionally cite the software repository itself using:

.. code-block::

  @misc{ramos-carreno++_2024_rdata-repo,
    author = {The rdata developers},
    doi = {10.5281/zenodo.6382237},
    month = dec,
    title = {rdata: A Python library for R datasets},
    url = {https://github.com/vnmabus/rdata},
    year = {2024}
  }

If you want to reference a particular version for reproducibility, check the version-specific DOIs available in Zenodo.

Usage
=====

Read an R dataset
-----------------

The common way of reading an rds file is:

.. code:: python

    import rdata

    converted = rdata.read_rds(rdata.TESTDATA_PATH / "test_dataframe.rds")
    print(converted)

which returns the read dataframe:

.. code:: none

      class  value
    1     a      1
    2     b      2
    3     b      3

The analog rda file can be read in a similar way:

.. code:: python

    import rdata

    converted = rdata.read_rda(rdata.TESTDATA_PATH / "test_dataframe.rda")
    print(converted)

which returns a dictionary mapping the variable name defined in the file (:code:`test_dataframe`) to the dataframe:

.. code:: none

    {'test_dataframe':   class  value
    1     a      1
    2     b      2
    3     b      3}

Under the hood, these reading functions are equivalent to the following two-step code:

.. code:: python

    import rdata

    parsed = rdata.parser.parse_file(rdata.TESTDATA_PATH / "test_dataframe.rda")
    converted = rdata.conversion.convert(parsed)
    print(converted)

This consists of two steps:

#. First, the file is parsed using the function
   `rdata.parser.parse_file <https://rdata.readthedocs.io/en/latest/modules/rdata.parser.parse_file.html>`__.
   This provides a literal description of the
   file contents as a hierarchy of Python objects representing the basic R
   objects. This step is unambiguous and always the same.
#. Then, each object must be converted to an appropriate Python object. In this
   step there are several choices on which Python type is the most appropriate
   as the conversion for a given R object. Thus, we provide a default
   `rdata.conversion.convert <https://rdata.readthedocs.io/en/latest/modules/rdata.conversion.convert.html>`__
   routine, which tries to select Python
   objects that preserve most information of the original R object. For custom
   R classes, it is also possible to specify conversion routines to Python
   objects as exemplified in
   `the documentation <https://rdata.readthedocs.io/en/latest/usage.html#converting>`__.

Write an R dataset
------------------

The common way of writing data to an rds file is:

.. code:: python

    import pandas as pd
    import rdata

    df = pd.DataFrame({"class": pd.Categorical(["a", "b", "b"]), "value": [1, 2, 3]})
    print(df)

    rdata.write_rds("data.rds", df)

which writes the dataframe to file :code:`data.rds`:

.. code:: none

      class  value
    0     a      1
    1     b      2
    2     b      3

Similarly, the dataframe can be written to an rda file with a given variable name:

.. code:: python

    import pandas as pd
    import rdata

    df = pd.DataFrame({"class": pd.Categorical(["a", "b", "b"]), "value": [1, 2, 3]})
    data = {"my_dataframe": df}
    print(data)

    rdata.write_rda("data.rda", data)

which writes the name-dataframe dictionary to file :code:`data.rda`:

.. code:: none

    {'my_dataframe':   class  value
    0     a      1
    1     b      2
    2     b      3}

Under the hood, these writing functions are equivalent to the following two-step code:

.. code:: python

    import pandas as pd
    import rdata

    df = pd.DataFrame({"class": pd.Categorical(["a", "b", "b"]), "value": [1, 2, 3]})
    data = {"my_dataframe": df}

    r_data = rdata.conversion.convert_python_to_r_data(data, file_type="rda")
    rdata.unparser.unparse_file("data.rda", r_data, file_type="rda")

This consists of two steps (reverse to reading):

#. First, each Python object is converted to an appropriate R object.
   Like in reading, there are several choices, and the default
   `rdata.conversion.convert_python_to_r_data <https://rdata.readthedocs.io/en/latest/modules/rdata.conversion.convert_python_to_r_data.html>`__.
   routine tries to select
   R objects that preserve most information of the original Python object.
   For Python classes, it is also possible to specify custom conversion routines
   to R classes as exemplified in
   `the documentation <https://rdata.readthedocs.io/en/latest/usage.html#converting>`__.
#. Then, the created RData representation is unparsed to a file using the function
   `rdata.unparser.unparse_file <https://rdata.readthedocs.io/en/latest/modules/rdata.unparser.unparse_file.html>`__.


Additional examples
===================

Additional examples illustrating the functionalities of this package can be
found in the
`ReadTheDocs documentation <https://rdata.readthedocs.io/en/latest/auto_examples/index.html>`__.


.. |build-status| image:: https://github.com/vnmabus/rdata/actions/workflows/main.yml/badge.svg?branch=master
    :alt: build status
    :target: https://github.com/vnmabus/rdata/actions/workflows/main.yml

.. |docs| image:: https://readthedocs.org/projects/rdata/badge/?version=latest
    :alt: Documentation Status
    :target: https://rdata.readthedocs.io/en/latest/?badge=latest

.. |coverage| image:: http://codecov.io/github/vnmabus/rdata/coverage.svg?branch=develop
    :alt: Coverage Status
    :target: https://codecov.io/gh/vnmabus/rdata/branch/develop

.. |repostatus| image:: https://www.repostatus.org/badges/latest/active.svg
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.
   :target: https://www.repostatus.org/#active

.. |versions| image:: https://img.shields.io/pypi/pyversions/rdata
   :alt: PyPI - Python Version

.. |pypi| image:: https://badge.fury.io/py/rdata.svg
    :alt: Pypi version
    :target: https://pypi.python.org/pypi/rdata/

.. |conda| image:: https://anaconda.org/conda-forge/rdata/badges/version.svg
    :alt: Conda version
    :target: https://anaconda.org/conda-forge/rdata

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6382237.svg
    :alt: Zenodo DOI
    :target: https://doi.org/10.5281/zenodo.6382237

.. |pyOpenSci| image:: https://tinyurl.com/y22nb8up
    :alt: pyOpenSci: Peer reviewed
    :target: https://github.com/pyOpenSci/software-submission/issues/144

.. |joss| image:: https://joss.theoj.org/papers/10.21105/joss.07540/status.svg
   :target: https://doi.org/10.21105/joss.07540
