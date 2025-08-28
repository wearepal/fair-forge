# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "fair-forge"
copyright = "2025, PAL developers"
author = "PAL developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",  # google-style docstrings
    "sphinx.ext.viewcode",  # add a buttone for viewing the source code
]

# templates_path = ["_templates"]
exclude_patterns = []
autosummary_generate = True  # let autosummary generate stubs
autosummary_imported_members = True  # include imports in the summary...
autosummary_ignore_module_all = False  # ...but if `__all__` is set, respect it
# We cannot set the following because the numpy docs imported through intersphinx
# are of course written in numpy style.
# napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
}

autodoc_default_options = {
    # We set `members` to true so that the autosummary of modules actually contains
    # the documentation of the module's members.
    "members": True,
    # `imported-members` is necessary so that classes and functions from submodules
    # are documented as well.
    "imported-members": True,
    # We want to show the documentation for `__call__`.
    "special-members": "__call__",
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
