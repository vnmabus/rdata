Simple usage
============

Read a R dataset
----------------
The common way of reading an R dataset is the following one:

>>> import rdata

>>> parsed = rdata.parser.parse_file(<file>)
>>> converted = rdata.conversion.convert(parsed)
>>> converted
    {'example_dataset': array([1., 2., 3.])}