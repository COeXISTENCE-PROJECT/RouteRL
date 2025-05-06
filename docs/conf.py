# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'RouteRL'
copyright = '2024, Anastasia Psarou, Ahmet Onur Akman, Łukasz Gorczyca'
author = 'Anastasia Psarou, Ahmet Onur Akman, Łukasz Gorczyca'

# The full version, including alpha/beta/rc tags
release = '1.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',    # Enables the auto module directive
    'sphinx.ext.napoleon',   # Supports Google-style or NumPy-style docstrings
    'sphinx.ext.viewcode',   # Adds links to the source code
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    'myst_nb',
    'furo.sphinxext',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
]

source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb"}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_image",
    "colon_fence",
]

suppress_warnings = ["myst.header"]

autodoc_default_options = {
    "members": True,
    #"undoc-members": False,  # Exclude undocumented methods
    "private-members": False,  # Exclude private methods (starting with _)
    #"special-members": False,  # Exclude special methods (__init__, __str__, etc.)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

pygments_style = 'sphinx'  

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Napoleon settings
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

master_doc = 'index' 

nb_execution_mode = "off"
pygments_style = "friendly"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_theme_options = {
    "source_repository": "https://github.com/COeXISTENCE-PROJECT/RouteRL",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ["kwargs_box.css"]
html_logo = "_static/logo.png"