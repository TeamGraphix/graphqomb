"""
Sphinx documentation configuration file for the graphix-zx project.

This file configures the behavior of Sphinx, including the theme,
extensions, and documentation settings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))

# -- Project information -----------------------------------------------------

project = "graphix-zx"
# copyright = "2025, Team Graphix"
author = "Masato Fukushima, to be added"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosectionlabel_prefix_document = True
default_role = "any"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = " "
# html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_context = {
    "mode": "production",
}

pygments_style = "sphinx"
pygments_dark_style = "monokai"

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["gallery"],
    "filename_pattern": "/",
    "thumbnail_size": (800, 550),
}

suppress_warnings = ["config.cache"]
