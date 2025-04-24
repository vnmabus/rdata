"""
Writing data to RDA and RDS files
=================================

An example showing how to write dataframes to RDA and RDS files.

"""

# sphinx_gallery_thumbnail_path = '_static/R_logo.svg'

# %%
# Let's import Pandas to create dataframe and rdata to write to file.

import pandas as pd
import rdata

# %%
# Let's create a toy dataframe

df = pd.DataFrame({"class": pd.Categorical(["a", "b", "b"]), "value": [1, 2, 3]})
df

# %%
# Let's write this dataframe to an RDS file.

rdata.write_rds("data.rds", df)

# %%
# By default, the file is written in a binary XDR format with gzip compression.
# We can write the file in different formats and compressions too.

rdata.write_rds("data_uncompressed.rds", df, compression=None)
rdata.write_rds("data_ascii.rds", df, file_format="ascii", compression=None)

# %%
# We can also write multiple variables with given names to an RDA file.

df1 = pd.DataFrame({"class": pd.Categorical(["a", "b", "b"]), "value": [1, 2, 3]})
df2 = pd.DataFrame({"color": ["red", "green", "blue"], "count": [12, 34, 56]})
data = {"df1": df1, "df2": df2}

rdata.write_rda("data.rda", data)
