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

A similar rda file can be read similarly:

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
   :func:`rdata.parser.parse_file`. This provides a literal description of the
   file contents as a hierarchy of Python objects representing the basic R
   objects. This step is unambiguous and always the same.
#. Then, each object must be converted to an appropriate Python object. In this
   step there are several choices on which Python type is the most appropriate
   as the conversion for a given R object. Thus, we provide a default
   :func:`rdata.conversion.convert` routine, which tries to select Python
   objects that preserve most information of the original R object. For custom
   R classes, it is also possible to specify conversion routines to Python
   objects as exemplified in :ref:`converting`.


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
   :func:`rdata.conversion.convert_python_to_r_data` routine tries to select
   R objects that preserve most information of the original Python object.
   For Python classes, it is also possible to specify custom conversion routines
   to R classes as exemplified in :ref:`converting`.
#. Then, the created RData representation is unparsed to a file using the function
   :func:`rdata.unparser.unparse_file`.


.. _converting:

Converting between R and Python classes
---------------------------------------

The :func:`~rdata.conversion.convert` and :func:`~rdata.conversion.convert_python_to_r_data` functions
implement the conversion of common data types and arrays (see :ref:`default_conversions`).
It is also possible to provide custom conversions for specific R and Python classes
by passing a dictionary of constructor functions to the conversion function.
The default dictionaries contains constructors for commonly used R classes such as
`data.frame <https://www.rdocumentation.org/packages/base/topics/data.frame>`_
and `factor <https://www.rdocumentation.org/packages/base/topics/factor>`_.


As an example, here we demonstrate how to implement an R-to-Python and Python-to-R conversion routines
for the R factor class to our custom class, instead of the default conversion to
Pandas :class:`~pandas.Categorical` class.

An example custom Python class representing an R factor is:

.. code:: python

    class MyFactor:
        """My custom class representing R factor."""
        def __init__(self, values, levels):
            self.values = np.asarray(values)
            self.levels = np.asarray(levels)

        def __getitem__(self, i):
            return self.levels[self.values[i]]

        def __len__(self):
            return len(self.values)

        def __str__(self):
            return f"MyFactor with: " + ", ".join(self[i] for i in range(len(self)))

Reading
^^^^^^^

Let's read an rds file using a custom constructor mapping R factor to :code:`MyFactor`:

.. code:: python

    import rdata

    def r_to_py_factor_constructor(obj, attrs):
        """Custom constructor."""
        return MyFactor(obj - 1, attrs["levels"])

    # Use the custom constructor for factor
    r_to_py_constructors = rdata.conversion.DEFAULT_CLASS_MAP.copy()
    r_to_py_constructors["factor"] = r_to_py_factor_constructor

    # Read data
    print("Read")
    data = rdata.read_rds(
        #rdata.TESTDATA_PATH / "test_dataframe.rds",
        "factor.rds",
        constructor_dict=r_to_py_constructors,
    )
    print(f"Done: {data}")

which produces the following printout:

.. code:: none

    Read
    Done: MyFactor with: a, b, b

Writing
^^^^^^^

Let's write an rds file using a custom constructor mapping :code:`MyFactor` to R factor:

.. code:: python

    import rdata

    def py_to_r_factor_constructor(obj, converter):
        """Custom constructor."""
        return rdata.conversion.to_r.build_r_object(
            rdata.parser.RObjectType.INT,
            value=obj.values + 1,
            is_object=True,
            attributes=converter.convert_to_r_attributes({
                "levels": obj.levels,
                "class": "factor",
            }),
        )

    # Use the custom constructor for MyFactor
    py_to_r_constructors = rdata.conversion.to_r.DEFAULT_CLASS_MAP.copy()
    py_to_r_constructors[MyFactor] = py_to_r_factor_constructor

    # Write data
    data = MyFactor([0, 1, 1], ["a", "b"])
    print(f"Write: {data}")
    rdata.write_rds("test.rds", data, constructor_dict=py_to_r_constructors)
    print("Done")

which produces a file :code:`test.rds` and the following printout:

.. code:: none

    Write: MyFactor with: a, b, b
    Done

