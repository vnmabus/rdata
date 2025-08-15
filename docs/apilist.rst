API List
========

List of functions and structures
--------------------------------
A complete list of all functions and structures provided by rdata.

Convenience functions
^^^^^^^^^^^^^^^^^^^^^
Functions that read and write an :code:`.rds` or :code:`.rda` file, performing parsing/unparsing and conversion with
one line of code.

.. autosummary::
   :toctree: modules

   rdata.read_rds
   rdata.read_rda
   rdata.write_rds
   rdata.write_rda

Parsing files
^^^^^^^^^^^^^
Functions for parsing data in the :code:`.rds` / :code:`.rda` format.
These functions return a structure representing
the contents of the file, without transforming it to more appropriate Python objects. Thus, if a different
way of converting R objects to Python objects is needed, it can be done from this structure.

.. autosummary::
   :toctree: modules

   rdata.parser.parse_file
   rdata.parser.parse_data

Unparsing files
^^^^^^^^^^^^^^^
Functions for unparsing data in the :code:`.rds` / :code:`.rda` format.
These functions do the opposite operation of the parsing functions, that is, they take as an input
the rdata's structure representing the contents of the file, and write it to a file or a bytestring.

.. autosummary::
   :toctree: modules

   rdata.unparser.unparse_file
   rdata.unparser.unparse_data

Conversion between R and Python objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These objects and functions convert the parsed R objects to appropriate Python objects and vice versa.
The Python/R objects are chosen to preserve most original properties, but the mappings between objects
could change in the future, if more fitting object mappings are found.

.. autosummary::
   :toctree: modules

   rdata.conversion.Converter
   rdata.conversion.SimpleConverter
   rdata.conversion.convert
   rdata.conversion.ConverterFromPythonToR
   rdata.conversion.convert_python_to_r_data
   rdata.conversion.convert_python_to_r_object

Auxiliary structures
^^^^^^^^^^^^^^^^^^^^
These classes are used to represent R objects which have no clear analog in Python, so that the information
therein can be retrieved.

.. autosummary::
   :toctree: modules

   rdata.conversion.RBuiltin
   rdata.conversion.RBytecode
   rdata.conversion.RFunction
   rdata.conversion.REnvironment
   rdata.conversion.RExpression
   rdata.conversion.RExternalPointer
   rdata.conversion.RLanguage
   rdata.conversion.SrcFile
   rdata.conversion.SrcFileCopy
   rdata.conversion.SrcRef
