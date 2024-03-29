"""Configuration of the Sphinx documentation."""

# rdata documentation build configuration file, created by
# sphinx-quickstart on Tue Aug 7 12:49:32 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import importlib.metadata
import os
import sys
import textwrap

import rdata

# General information about the project.
project = "rdata"
author = "Carlos Ramos Carreño"
copyright = "2018, Carlos Ramos Carreño"  # noqa: A001
github_url = "https://github.com/vnmabus/rdata"
rtd_version = os.environ.get("READTHEDOCS_VERSION")
rtd_version_type = os.environ.get("READTHEDOCS_VERSION_TYPE")

switcher_version = rtd_version
if switcher_version == "latest":
    switcher_version = "dev"
elif rtd_version_type not in {"branch", "tag"}:
    switcher_version = rdata.__version__

rtd_branch = os.environ.get(" READTHEDOCS_GIT_IDENTIFIER", "develop")
language = "en"

try:
    release = importlib.metadata.version("rdata")
except importlib.metadata.PackageNotFoundError:
    print(  # noqa: T201
        f"To build the documentation, The distribution information of\n"
        f"{project} has to be available.  Either install the package\n"
        f"into your development environment or run 'setup.py develop'\n"
        f"to setup the metadata.  A virtualenv is recommended!\n",
    )
    sys.exit(1)

version = ".".join(release.split(".")[:2])

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx_codeautolink",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "jupyterlite_sphinx",  # Move after sphinx gallery
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


add_module_names = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "use_edit_page_button": True,
    "github_url": github_url,
    "switcher": {
        "json_url": (
            "https://rdata.readthedocs.io/en/latest/_static/switcher.json"
        ),
        "version_match": switcher_version,
    },
    "show_version_warning_banner": True,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/rdata",
            "icon": "https://avatars.githubusercontent.com/u/2964877",
            "type": "url",
        },
        {
            "name": "Anaconda",
            "url": "https://anaconda.org/conda-forge/rdata",
            "icon": "https://avatars.githubusercontent.com/u/3571983",
            "type": "url",
        },
    ],
    "logo": {
        "text": "🗃 rdata",
    },
}

html_context = {
    "github_user": "vnmabus",
    "github_repo": "rdata",
    "github_version": "develop",
    "doc_path": "docs",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for LaTeX output ---------------------------------------------

latex_engine = "lualatex"

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "rdata.tex",
     "rdata Documentation", author, "manual"),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, "rdata", "rdata Documentation", [author], 1),
]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "rdata",
        "rdata documentation",
        author,
        "rdata",
        "Read R datasets from Python.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# -- Options for "sphinx.ext.autodoc.typehints" --

autodoc_preserve_defaults = True
autodoc_typehints = "description"

# -- Options for "sphinx.ext.autosummary" --

autosummary_generate = True

# -- Options for "sphinx.ext.intersphinx" --

intersphinx_mapping = {
    "igraph": ("https://python.igraph.org/en/stable/api", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
}

# -- Options for "sphinx.ext.todo" --

todo_include_todos = True


# -- Options for "sphinx_gallery.gen_gallery" --

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "reference_url": {
        "rdata": None,
    },
    "doc_module": "rdata",
    "jupyterlite": {
        "use_jupyter_lab": True,
    },
    "first_notebook_cell": textwrap.dedent("""\
    %pip install lzma
    %pip install rdata
    %pip install ipywidgets
    import pyodide_http
    pyodide_http.patch_all()
    """),
}
