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
import sphinx_rtd_theme
import os
import sys

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('sphinxext'))


# -- Project information -----------------------------------------------------

project = 'Flow'
copyright = '2020, Bram Berendsen'
author = 'Bram Berendsen'

# The full version, including alpha/beta/rc tags
release = 'v1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.coverage',
              'sphinx.ext.imgmath',
              'sphinx.ext.mathjax',
              'sphinx.ext.doctest',
              'sphinx.ext.githubpages',
              'sphinx_rtd_theme',
              'matplotlib.sphinxext.plot_directive',
              'IPython.sphinxext.ipython_directive',
              'IPython.sphinxext.ipython_console_highlighting',
              'sphinx.ext.inheritance_diagram',
              'sphinxcontrib.bibtex',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx'
              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Math settings
math_number_all = True

# Latex settings
pngmath_latex=r"C:\Program Files\MiKTeX 2.9\miktex\bin\x64\latex.exe"
pngmath_dvipng=r"C:\Program Files\MiKTeX 2.9\miktex\bin\x64\dvipng.exe"

# Theme settings
html_logo = './_static/drop.png'
html_favicon = './_static/drop.bmp'

# Napoleon settings
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

autodoc_default_flags = ['members',
                         'undoc-members',
                         'private-members',
                         'inherited-members',
                         'show-inheritance']

# intersphinx settings
intersphinx_mapping = {'numpy': ('https://docs.scipy.org/doc/numpy', None),
                       'functools': ('https://docs.python.org/3.7/library/functools.html', None)}
