# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'LongshotEvolve'
copyright = '2024, Vibrant Matrix'
author = 'bluebread'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx.ext.intersphinx',
]

# Add support for Markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# Theme is set to furo (already configured at line 80)
# html_theme = 'furo'  # Already set at line 80
# html_static_path = ['_static']  # Already set at line 81

# Furo-specific theme options
html_theme_options = {
    # Furo theme options
    'light_css_variables': {
        'color-brand-primary': '#007bff',
        'color-brand-content': '#007bff',
    },
    'dark_css_variables': {
        'color-brand-primary': '#007bff',
        'color-brand-content': '#007bff',
    },
}

# -- Extension configuration -------------------------------------------------
# Napoleon settings for Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# MyST parser configuration for Markdown support
myst_enable_extensions = [
    "deflist",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
]
myst_heading_anchors = 3  # Generate anchors for h1, h2, h3

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

html_theme = 'furo'
html_static_path = ['_static']