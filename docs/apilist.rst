API List
========

List of functions and structures
--------------------------------
A complete list of all functions and structures provided by rdata.

Convenience functions
^^^^^^^^^^^^^^^^^^^^^
Functions that read and transform a `.rds` or `.rda` file, performing parsing and conversion with
one line of code.

.. autosummary::
   :toctree: modules
   
   rdata.read_rds
   rdata.read_rda

Parse :code:`.rda` format
^^^^^^^^^^^^^^^^^^^^^^^^^
Functions for parsing data in the :code:`.rda` format. These functions return a structure representing
the contents of the file, without transforming it to more appropriate Python objects. Thus, if a different
way of converting R objects to Python objects is needed, it can be done from this structure. 

.. autosummary::
   :toctree: modules
   
   rdata.parser.parse_file
   rdata.parser.parse_data
   
Conversion of the R objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^
These objects and functions convert the parsed R objects to appropriate Python objects. The Python object
corresponding to a R object is chosen to preserve most original properties, but it could change in the 
future, if a more fitting Python object is found.

.. autosummary::
   :toctree: modules
   
   rdata.conversion.Converter
   rdata.conversion.SimpleConverter
   rdata.conversion.convert

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
