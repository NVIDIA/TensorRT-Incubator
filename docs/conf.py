import sys
import os

ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.path.pardir)
sys.path.insert(0, ROOT_DIR)
import tripy

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# Want to be able to generate docs with no dependencies installed
autodoc_mock_imports = []


autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

autodoc_member_order = "bysource"

autodoc_inherit_docstrings = True

add_module_names = False

autosummary_generate = True

source_suffix = [".rst"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "Tripy"
copyright = "2023, NVIDIA"
author = "NVIDIA"

version = tripy.__version__
# The full version, including alpha/beta/rc tags.
release = version

# Style
pygments_style = "colorful"

html_theme = "sphinx_rtd_theme"

# Use the TRT theme and NVIDIA logo
html_static_path = ["_static"]

html_logo = "_static/img/nvlogo_white.png"

# Hide source link
html_show_sourcelink = False

# Output file base name for HTML help builder.
htmlhelp_basename = "TripyDoc"

# Template files to extend default Sphinx templates.
# See https://www.sphinx-doc.org/en/master/templating.html for details.
templates_path = ["_templates"]

# For constructor arguments to show up in Sphinx generated doc
autoclass_content = "both"

# Unlimited depth sidebar.
html_theme_options = {"navigation_depth": -1}

html_sidebars = {"**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]}


# Allows us to override the default page width in the Sphinx theme.
def setup(app):
    app.add_css_file("style.css")
    LATEX_BUILDER = "sphinx.builders.latex"
    if LATEX_BUILDER in app.config.extensions:
        app.config.extensions.remove(LATEX_BUILDER)
