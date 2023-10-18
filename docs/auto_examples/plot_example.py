# fmt: off
"""
R data loading
==============

Shows how to load R packages.
"""

# Author: Carlos Ramos Carreño
# License: MIT

# %%
# Use the file uploader to convert files to Python.
#
import rdata
from ipywidgets import FileUpload, interact

@interact(files=FileUpload(accept="*.rd*", multiple=True))
def convert_from_file(files):
    for f in files:
        parsed = rdata.parser.parse_data(f.content)
        converted = rdata.conversion.convert(parsed)
        for key, value in converted.items():
            print(f"{key}:")
            print(value)
