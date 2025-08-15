.. _default_conversions:

Default conversions
===================

This page list the default conversions between R and Python objects.

Basic types
-----------

The conversion of basic types is performed directly by the
:func:`~rdata.conversion.convert` and :func:`~rdata.conversion.convert_python_to_r_data` functions
(or the underlying :class:`~rdata.conversion.Converter` and :class:`~rdata.conversion.ConverterFromPythonToR`).
Thus, changing the conversion for basic types currently requires creating a custom converter class.
The default converters perform the following conversions:

================== ================================================================================================
R type             Python type
================== ================================================================================================
builtin function   :class:`rdata.conversion.RBuiltin`.
bytecode           :class:`rdata.conversion.RBytecode`.
char (internal)    :class:`str` or :class:`bytes` (depending on the encoding flags).
closure            :class:`rdata.conversion.RFunction`.
complex            :class:`numpy.ndarray` with 128-bits complex dtype.

                   :class:`xarray.DataArray` if it contains labeled dimensions.
environment        :class:`rdata.conversion.REnvironment`.
                   There are three special cases: the empty, base and global environments, which are
                   all empty by default. The base and global environments may be supplied to the
                   converter.
expression         :class:`rdata.conversion.RExpression`.
external pointer   :class:`rdata.conversion.RExternalPointer`.
integer            :class:`numpy.ndarray` with 32-bits integer dtype.

                   :class:`numpy.ma.MaskedArray` with 32-bits integer dtype if it contains NA values.

                   :class:`xarray.DataArray` if it contains labeled dimensions.
language           :class:`rdata.conversion.RLanguage`.
list / vector      :class:`list` (if untagged).

                   :class:`dict` (if tagged). Empty lists are considered tagged.
logical (boolean)  :class:`numpy.ndarray` with boolean dtype.

                   :class:`numpy.ma.MaskedArray` with boolean dtype if it contains NA values.

                   :class:`xarray.DataArray` if it contains labeled dimensions.
missing argument   :data:`NotImplemented`.
NULL               :data:`None`.
real               :class:`numpy.ndarray` with 64-bits floating point dtype.

                   :class:`xarray.DataArray` if it contains labeled dimensions.
reference          The referenced value, that is, an object already converted.
S4 object          :class:`types.SimpleNamespace`.
special function   :class:`rdata.conversion.RBuiltin`.
string             :class:`numpy.ndarray` with a suitable fixed-length string dtype.
symbol             :class:`str`.
================== ================================================================================================

Custom classes
--------------

In addition, R objects containing a `"class"` attribute, or Python objects not listed above, are passed to an R-to-Python or Python-to-R constructor function, respectively.
When the "class" attribute contains several class names, these are tried in order.
A dictionary of constructor functions can be supplied to the converters as exemplified in :ref:`converting`.
The default constructor dictionaries perform the following conversions:

================== ================================================================================================
R class            Python class
================== ================================================================================================
data.frame         :class:`pandas.DataFrame`.
factor             :class:`pandas.Categorical`.
ordered            :class:`pandas.Categorical` (with ordered categories).
srcfile            :class:`rdata.conversion.SrcFile`.
srcfilecopy        :class:`rdata.conversion.SrcFileCopy`.
srcref             :class:`rdata.conversion.SrcRef`.
ts                 :class:`pandas.Series`.
================== ================================================================================================
