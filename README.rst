rdata
=====

|build-status| |docs| |coverage| |repostatus| |versions| |pypi| |conda| |zenodo|

Read R datasets from Python.

..
	Github does not support include in README for dubious security reasons, so
	we copy-paste instead. Also Github does not understand Sphinx directives.
	.. include:: docs/index.rst
	.. include:: docs/simpleusage.rst

The package rdata offers a lightweight way to import R datasets/objects stored
in the ".rda" and ".rds" formats into Python.
Its main advantages are:

- It is a pure Python implementation, with no dependencies on the R language or
  related libraries.
  Thus, it can be used anywhere where Python is supported, including the web
  using `Pyodide <https://pyodide.org/>`_.
- It attempt to support all R objects that can be meaningfully translated.
  As opposed to other solutions, you are no limited to import dataframes or
  data with a particular structure.
- It allows users to easily customize the conversion of R classes to Python
  ones.
  Does your data use custom R classes?
  Worry no longer, as it is possible to define custom conversions to the Python
  classes of your choosing.
- It has a permissive license (MIT). As opposed to other packages that depend
  on R libraries and thus need to adhere to the GPL license, you can use rdata
  as a dependency on MIT, BSD or even closed source projects.
	
Installation
============

rdata is on PyPi and can be installed using :code:`pip`:

.. code::

   pip install rdata

It is also available for :code:`conda` using the :code:`conda-forge` channel:

.. code::

   conda install -c conda-forge rdata
   
Installing the develop version
------------------------------

The current version from the develop branch can be installed as

.. code::

   pip install git+https://github.com/vnmabus/rdata.git@develop

Documentation
=============

The documentation of rdata is in
`ReadTheDocs <https://rdata.readthedocs.io/>`_.

Examples
========

Examples of use are available in
`ReadTheDocs <https://rdata.readthedocs.io/en/stable/auto_examples/>`_.
	
Simple usage
============

Read a R dataset
----------------

The common way of reading an R dataset is the following one:

.. code:: python

    import rdata

    converted = rdata.read_rda(rdata.TESTDATA_PATH / "test_vector.rda")
    converted
    
which results in

.. code::

    {'test_vector': array([1., 2., 3.])}

Under the hood, this is equivalent to the following code:

.. code:: python

    import rdata

    parsed = rdata.parser.parse_file(rdata.TESTDATA_PATH / "test_vector.rda")
    converted = rdata.conversion.convert(parsed)
    converted
    
This consists on two steps: 

#. First, the file is parsed using the function
   `rdata.parser.parse_file <https://rdata.readthedocs.io/en/latest/modules/rdata.parser.parse_file.html>`_.
   This provides a literal description of the
   file contents as a hierarchy of Python objects representing the basic R
   objects. This step is unambiguous and always the same.
#. Then, each object must be converted to an appropriate Python object. In this
   step there are several choices on which Python type is the most appropriate
   as the conversion for a given R object. Thus, we provide a default
   `rdata.conversion.convert <https://rdata.readthedocs.io/en/latest/modules/rdata.conversion.convert.html>`_
   routine, which tries to select Python objects that preserve most information
   of the original R object. For custom R classes, it is also possible to
   specify conversion routines to Python objects.
   
Convert custom R classes
------------------------

The basic
`convert <https://rdata.readthedocs.io/en/latest/modules/rdata.conversion.convert.html>`_
routine only constructs a
`SimpleConverter <https://rdata.readthedocs.io/en/latest/modules/rdata.conversion.SimpleConverter.html>`_
object and calls its
`convert <https://rdata.readthedocs.io/en/latest/modules/rdata.conversion.SimpleConverter.html#rdata.conversion.SimpleConverter.convert>`_
method. All arguments of
`convert <https://rdata.readthedocs.io/en/latest/modules/rdata.conversion.convert.html>`_
are directly passed to the
`SimpleConverter <https://rdata.readthedocs.io/en/latest/modules/rdata.conversion.SimpleConverter.html>`_
initialization method.

It is possible, although not trivial, to make a custom
`Converter <https://rdata.readthedocs.io/en/latest/modules/rdata.conversion.Converter.html>`_
object to change the way in which the
basic R objects are transformed to Python objects. However, a more common
situation is that one does not want to change how basic R objects are
converted, but instead wants to provide conversions for specific R classes.
This can be done by passing a dictionary to the
`SimpleConverter <https://rdata.readthedocs.io/en/latest/modules/rdata.conversion.SimpleConverter.html>`_
initialization method, containing
as keys the names of R classes and as values, callables that convert a
R object of that class to a Python object. By default, the dictionary used
is
`DEFAULT_CLASS_MAP <https://rdata.readthedocs.io/en/latest/modules/rdata.conversion.DEFAULT_CLASS_MAP.html>`_,
which can convert commonly used R classes such as
`data.frame <https://www.rdocumentation.org/packages/base/topics/data.frame>`_
and `factor <https://www.rdocumentation.org/packages/base/topics/factor>`_.

As an example, here is how we would implement a conversion routine for the
factor class to
`bytes <https://docs.python.org/3/library/stdtypes.html#bytes>`_
objects, instead of the default conversion to
Pandas
`Categorical <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Categorical.html#pandas.Categorical>`_ objects:

.. code:: python

    import rdata

    def factor_constructor(obj, attrs):
        values = [bytes(attrs['levels'][i - 1], 'utf8')
                  if i >= 0 else None for i in obj]
   
        return values

    new_dict = {
        **rdata.conversion.DEFAULT_CLASS_MAP,
        "factor": factor_constructor
    }

    converted = rdata.read_rda(
        rdata.TESTDATA_PATH / "test_dataframe.rda",
        constructor_dict=new_dict,
    )
    converted
    
which has the following result:

.. code::

    {'test_dataframe':   class  value
        1     b'a'      1
        2     b'b'      2
        3     b'b'      3}
    
Additional examples
===================

Additional examples illustrating the functionalities of this package can be
found in the
`ReadTheDocs documentation <https://rdata.readthedocs.io/en/latest/auto_examples/index.html>`_.


.. |build-status| image:: https://github.com/vnmabus/rdata/actions/workflows/main.yml/badge.svg?branch=master
    :alt: build status
    :scale: 100%
    :target: https://github.com/vnmabus/rdata/actions/workflows/main.yml

.. |docs| image:: https://readthedocs.org/projects/rdata/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://rdata.readthedocs.io/en/latest/?badge=latest
    
.. |coverage| image:: http://codecov.io/github/vnmabus/rdata/coverage.svg?branch=develop
    :alt: Coverage Status
    :scale: 100%
    :target: https://codecov.io/gh/vnmabus/rdata/branch/develop

.. |repostatus| image:: https://www.repostatus.org/badges/latest/active.svg
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.
   :target: https://www.repostatus.org/#active

.. |versions| image:: https://img.shields.io/pypi/pyversions/rdata
   :alt: PyPI - Python Version
   :scale: 100%
    
.. |pypi| image:: https://badge.fury.io/py/rdata.svg
    :alt: Pypi version
    :scale: 100%
    :target: https://pypi.python.org/pypi/rdata/

.. |conda| image:: https://anaconda.org/conda-forge/rdata/badges/version.svg
    :alt: Conda version
    :scale: 100%
    :target: https://anaconda.org/conda-forge/rdata

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6382237.svg
    :alt: Zenodo DOI
    :scale: 100%
    :target: https://doi.org/10.5281/zenodo.6382237
