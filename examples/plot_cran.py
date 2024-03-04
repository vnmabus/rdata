"""
Loading a RDA file with custom types from CRAN
==============================================

A more advanced example showing how to read a dataset in the RDATA format from
the CRAN repository of R packages that include custom R types.

"""

# %%
# We will show how to load the graph of the classical
# `seven bridges of Königsberg problem
# <https://en.wikipedia.org/wiki/Seven_Bridges_of_K%C3%B6nigsberg>`_ from the
# R package igraphdata.
#
# .. warning::
#     This is for illustration purposes only. If you plan to use the same
#     dataset repeatedly it is better to download it, or to use a package that
#     caches it, such as
#     `scikit-datasets <https://daviddiazvico.github.io/scikit-datasets/>`_.
#
# We will make use of the function
# :external+python:func:`urllib.request.urlopen` to load the url, as well as
# the package rdata.
# The package is a tar file so we need also to import the
# :external+python:mod:`tarfile` module.
# We will use the package `igraph <https://python.igraph.org/en/stable/>`_ for
# constructing the graph in Python.
# Finally, we will import some plotting routines from Matplotlib.

import tarfile
from urllib.request import urlopen

import igraph
import igraph.drawing
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import rdata

# %%
# The following URL contains the link to download the package from CRAN.

pkg_url = (
    "https://cran.r-project.org/src/contrib/Archive/"
    "igraphdata/igraphdata_1.0.0.tar.gz"
)

# %%
# The dataset is contained in the "data" folder, as it is common for
# R packages.
# The file is named Koenisberg and it is in the RDATA format
# (.rda extension).
data_path = "igraphdata/data/Koenigsberg.rda"

# %%
# We proceed to open the package using
# :external+python:func:`~urllib.request.urlopen`
# and :external+python:mod:`tarfile`.


with urlopen(pkg_url) as package:
    with tarfile.open(fileobj=package, mode="r|gz") as package_tar:
        for member in package_tar:
            if member.name == data_path:
                dataset = package_tar.extractfile(member)
                assert dataset
                with dataset:
                    parsed = rdata.parser.parse_file(dataset)
                break

# %%
# We could try to convert this dataset to Python objects.

converted = rdata.conversion.convert(parsed)
print(converted)

# %%
# From this representation, we can see that .rda files contain a mapping
# of variable names to objects, and not just one object as .rds files.
# In this case there is just one variable called "Koenigsberg", as the
# dataset itself, but that is not necessarily always the case.

# %%
# We can also see that there is no default conversion for the "igraph"
# class, representing a graph.
# Thus, the converted object is a list of the underlying vectors used
# by this type.

# %%
# It is however possible to define our own conversion routines for R classes
# using the package rdata.
# For that purpose we need to create a "constructor" function, that accepts
# as arguments the underlying object to convert and its attributes, and
# returns the converted object.

# %%
# In this example, the object will be received as a list, corresponding to
# the `igraph_t structure defined by the igraph package
# <https://github.com/igraph/igraph/blob/
# 50d46370fd677128cf758e4dd5c1de61dae9a3ef/include/
# igraph_datatype.h#L110-L121>`_.
# We will convert it to a :external+igraph:class:`~igraph.Graph` object from
# the `Python version of the igraph package
# <https://python.igraph.org/en/stable/>`_.
# The attrs dict is empty and will not be used.


def graph_constructor(obj, attrs):
    """Construct graph object from R representation."""
    n_vertices = int(obj[0][0])
    is_directed = obj[1]
    edge_from = obj[2].astype(int)
    edge_to = obj[3].astype(int)

    # output_edge_index = obj[4]
    # input_edge_index = obj[5]
    # output_vertex_edge_index = obj[6]
    # input_vertex_edge_index = obj[7]

    graph_attrs = obj[8][1]
    vertex_attrs = obj[8][2]
    edge_attrs = obj[8][3]

    return igraph.Graph(
        n=n_vertices,
        directed=is_directed,
        edges=list(zip(edge_from, edge_to)),
        graph_attrs=graph_attrs,
        vertex_attrs=vertex_attrs,
        edge_attrs=edge_attrs,
    )


# %%
# We create a dict with all the constructors that we want to apply.
# In this case, we include first the default constructors (which
# provide transformations for common R classes) and our newly created
# constructor.
# The key used for the dictionary entries should be the name of the
# corresponding R class.
constructor_dict = {
    **rdata.conversion.DEFAULT_CLASS_MAP,
    "igraph": graph_constructor,
}

# %%
# We can now call the :func:`rdata.conversion.convert` functtion, supplying
# the dictionary of constructors to use.
converted = rdata.conversion.convert(parsed, constructor_dict=constructor_dict)

# %%
# Finally, we check the constructed graph by plotting it using the
# :external+igraph:func:`igraph.drawing.plot` function.
fig, axes = plt.subplots()
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
igraph.drawing.plot(
    converted["Koenigsberg"],
    target=axes,
    vertex_label=converted["Koenigsberg"].vs["name"],
    vertex_label_size=8,
    vertex_size=120,
    vertex_color=to_hex("tab:blue"),
    edge_label=converted["Koenigsberg"].es["name"],
    edge_label_size=8,
)
plt.show()
