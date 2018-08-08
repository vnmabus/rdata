API List
========

List of functions and structures
--------------------------------
A complete list of all functions and structures provided by rdata.

Parse :code:`.rda` format
^^^^^^^^^^^^^^^^^^^^^^^^^
Functions for parsing data in the :code:`.rda` format. These functions return a structure representing
the contents of the file, without transforming it to more appropiate Python objects. Thus, if a different
way of converting R objects to Python objects is needed, it can be done from this structure. 

.. autosummary::
   :toctree: functions
   
   rdata.parser.parse_file
   rdata.parser.parse_data
   
Conversion of the R objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^
These objects and functions convert the parsed R objects to appropiate Python objects. The Python object
corresponding to a R object is chosen to preserve most original properties, but it could change in the 
future, if a more fitting Python object is found.

.. autosummary::
   :toctree: functions
   
   rdata.conversion.Converter
   rdata.conversion.SimpleConverter
   rdata.conversion.convert
   rdata.conversion.DEFAULT_CLASS_MAP

