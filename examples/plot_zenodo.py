"""
Loading a RDS file from a URL
=============================

A simple example showing how to read a dataset in the RDS format from a URL.

"""

# sphinx_gallery_thumbnail_path = '_static/download.png'

# %%
# If the data to read is accesible at a particular URL, we can open it as a
# file using the function :external+python:func:`urllib.request.urlopen`.
# Thus, we need to import that function as well as the rdata package.

from urllib.request import urlopen

import rdata

# %%
# For this example we will use a dataset hosted at
# `Zenodo <https://zenodo.org/records/7425539>`_.
# This is a small dataset containing information about some fungal pathogens.
#
dataset_url = (
    "https://zenodo.org/records/7425539/files/core_fungal_pathogens.rds"
)

# %%
# The object resulting from calling
# :external+python:func:`~urllib.request.urlopen` can be then
# passed to :func:`~rdata.parser.parse_file` as if it were a normal file.

with urlopen(dataset_url) as dataset:
    parsed = rdata.parser.parse_file(dataset)

# %%
# RDS files do not have a special magic number that identifies them.
# Thus, when reading a RDS file, rdata has to suppose that the file is a valid
# RDS file, and warns about that.
# We can omit this warning by passing manually the extension of the file
# instead.
with urlopen(dataset_url) as dataset:
    parsed = rdata.parser.parse_file(dataset, extension="rds")

# %%
# This parsed object contains a lossless representation of the internal data
# contained in the file.
# This data mimics the internal format used in R, and is thus not directly
# usable.
# However, we can retrieve some information about the file that will be lost
# after the conversion to a Python object, such as the version of the format
# employed or the encoding used for the strings.

print(parsed.versions.format)
print(parsed.extra.encoding)

# %%
# In order to convert it to Python objects we need to use the function
# :func:`rdata.conversion.convert`.
converted = rdata.conversion.convert(parsed)

# %%
# RDS files contain just one R object.
# In this particular case, it is a R dataframe object, that will be converted
# to a Pandas dataframe by default.
converted
