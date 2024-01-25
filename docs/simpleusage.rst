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

.. code:: python

    {'test_vector': array([1., 2., 3.])}

Under the hood, this is equivalent to the following code:

.. code:: python

    import rdata

    parsed = rdata.parser.parse_file(rdata.TESTDATA_PATH / "test_vector.rda")
    converted = rdata.conversion.convert(parsed)
    converted
    
This consists on two steps: 

#. First, the file is parsed using the function
   :func:`rdata.parser.parse_file`. This provides a literal description of the
   file contents as a hierarchy of Python objects representing the basic R
   objects. This step is unambiguous and always the same.
#. Then, each object must be converted to an appropriate Python object. In this
   step there are several choices on which Python type is the most appropriate
   as the conversion for a given R object. Thus, we provide a default
   :func:`rdata.conversion.convert` routine, which tries to select Python
   objects that preserve most information of the original R object. For custom
   R classes, it is also possible to specify conversion routines to Python
   objects.
   
Convert custom R classes
------------------------

The basic :func:`~rdata.conversion.convert` routine only constructs a
:class:`~rdata.conversion.SimpleConverter` object and calls its
:meth:`~rdata.conversion.SimpleConverter.convert` method. All arguments of
:func:`~rdata.conversion.convert` are directly passed to the
:class:`~rdata.conversion.SimpleConverter` initialization method.

It is possible, although not trivial, to make a custom
:class:`~rdata.conversion.Converter` object to change the way in which the
basic R objects are transformed to Python objects. However, a more common
situation is that one does not want to change how basic R objects are
converted, but instead wants to provide conversions for specific R classes.
This can be done by passing a dictionary to the
:class:`~rdata.conversion.SimpleConverter` initialization method, containing
as keys the names of R classes and as values, callables that convert a
R object of that class to a Python object. By default, the dictionary used
is :data:`~rdata.conversion.DEFAULT_CLASS_MAP`, which can convert
commonly used R classes such as
`data.frame <https://www.rdocumentation.org/packages/base/topics/data.frame>`_
and `factor <https://www.rdocumentation.org/packages/base/topics/factor>`_.

As an example, here is how we would implement a conversion routine for the
factor class to :class:`bytes` objects, instead of the default conversion to
Pandas :class:`~pandas.Categorical` objects:

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

.. code:: python

    {'test_dataframe':   class  value
        1     b'a'      1
        2     b'b'      2
        3     b'b'      3}