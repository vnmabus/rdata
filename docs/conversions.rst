Default conversions
===================

This page list the default conversions applied to R objects to convert them to
Python objects.

Basic types
-----------

The conversion of basic types is performed directly by the
:class:`~rdata.conversion.Converter` used.
Thus, changing the conversion for basic types currently requires creating a
custom :class:`~rdata.conversion.Converter` class.
The default :class:`~rdata.conversion.SimpleConverter` realizes the following
conversions:

================== ================================================================================================
R object type      Python conversion
================== ================================================================================================
builtin function   :class:`rdata.conversion.RBuiltin`.
bytecode           :class:`rdata.conversion.RBytecode`.
char (internal)    :class:`str` or :class:`bytes` (depending on the encoding flags).
closure            :class:`rdata.conversion.RFunction`.
complex            :class:`numpy.ndarray` with 128-bits complex dtype.

                   :class:`numpy.ma.MaskedArray` with 128-bits complex dtype if it contains NA values.

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
list               :class:`list` (if untagged).

                   :class:`dict` (if tagged). Empty lists are considered tagged.
logical (boolean)  :class:`numpy.ndarray` with boolean dtype.

                   :class:`numpy.ma.MaskedArray` with boolean dtype if it contains NA values.

                   :class:`xarray.DataArray` if it contains labeled dimensions.
missing argument   :data:`NotImplemented`.
NULL               :data:`None`.
real               :class:`numpy.ndarray` with 64-bits floating point dtype.

                   :class:`numpy.ma.MaskedArray` with 64-bits floating point dtype if it contains NA values.

                   :class:`xarray.DataArray` if it contains labeled dimensions.
reference          The referenced value, that is, an object already converted.
S4 object          :class:`types.SimpleNamespace`.
special function   :class:`rdata.conversion.RBuiltin`.
string             :class:`numpy.ndarray` with suitable fixed-length string dtype.
symbol             :class:`str`.
vector             :class:`list` (if untagged).

                   :class:`dict` (if tagged). Empty lists are considered tagged.
================== ================================================================================================

Custom classes
--------------

In addition, objects containing a `"class"` attribute are passed to a "constructor function", if one is available.
A dictionary of constructor functions can be supplied to the converter, where the key of each element corresponds
to the class name.
When the `"class"` attribute contains several class names, these are tried in order. 
The default constructor dictionary allows to convert the following R classes:

================== ================================================================================================
R class            Python conversion
================== ================================================================================================
data.frame         :class:`pandas.DataFrame`.
factor             :class:`pandas.Categorical`.
ordered            :class:`pandas.Categorical` (with ordered categories).
srcfile            :class:`rdata.conversion.SrcFile`.
srcfilecopy        :class:`rdata.conversion.SrcFileCopy`.
srcref             :class:`rdata.conversion.SrcRef`.
ts                 :class:`pandas.Series`.
================== ================================================================================================
