# fmt: off
"""
R data loading
==============

Use the file uploader to convert files to Python.
"""

# sphinx_gallery_thumbnail_path = '_static/R_logo.svg'

from ipywidgets import FileUpload, interact

import rdata


@interact(files=FileUpload(accept="*.rd*", multiple=True))
def convert_from_file(files):
    """Open a rds or rdata file and display its contents as Python objects."""
    for f in files:
        parsed = rdata.parser.parse_data(f.content)
        converted = rdata.conversion.convert(parsed)
        for key, value in converted.items():
            print(f"{key}:")
            print(value)
