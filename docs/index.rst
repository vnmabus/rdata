rdata version |version|
=======================

|build-status| |docs| |coverage| |pypi| |zenodo|

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

.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Contents:

   installation
   simpleusage
   apilist
   auto_examples/index
   Try online! <https://rdata.readthedocs.io/en/latest/lite/lab/?path=auto_examples/plot_example.ipynb>
   conversions
   contributors

The package rdata is developed `on Github <http://github.com/vnmabus/rdata>`_.
Please report `issues <https://github.com/vnmabus/rdata/issues>`_ there
as well.

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
    
.. |pypi| image:: https://badge.fury.io/py/rdata.svg
    :alt: Pypi version
    :scale: 100%
    :target: https://pypi.python.org/pypi/rdata/
    
.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6382237.svg
    :alt: Zenodo DOI
    :scale: 100%
    :target: https://doi.org/10.5281/zenodo.6382237
