import sys
import os

ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.path.pardir)
sys.path.insert(0, ROOT_DIR)
import tripy as tp

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Want to be able to generate docs with no dependencies installed
autodoc_mock_imports = []


autodoc_default_options = {
    "members": True,
    "no-undoc-members": True,
    "show-inheritance": True,
}

autodoc_member_order = "bysource"

autodoc_inherit_docstrings = True

add_module_names = True

autosummary_generate = True

source_suffix = [".rst"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "Tripy"
copyright = "2023, NVIDIA"
author = "NVIDIA"

version = tp.__version__
# The full version, including alpha/beta/rc tags.
release = version

# Style
pygments_style = "colorful"

html_theme = "sphinx_rtd_theme"

# Use the TRT theme and NVIDIA logo
html_static_path = ["_static"]

html_logo = "_static/img/nvlogo_white.png"

# Hide source link
html_show_sourcelink = True

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

# Myst will complain about relative links in our top-level README
suppress_warnings = ["myst.xref_missing"]

# Ignore docs/README.md as that's for developers and not supposed to be included in the public docs.
exclude_patterns = ["README.md"]


def setup(app):
    # A note on aliases: if you rename a class via an import statement, e.g. `import X as Y`,
    # the documentation generated for `Y` will just be: "Alias of X"
    # To get the real documentation, you can make Sphinx think that `Y` is not an alias but instead a real
    # class/function. To do so, you just need to define the __name__ attribute in this function (*not* in tripy code!):
    #   Y.__name__ = "Y"

    app.add_css_file("style.css")
    LATEX_BUILDER = "sphinx.builders.latex"
    if LATEX_BUILDER in app.config.extensions:
        app.config.extensions.remove(LATEX_BUILDER)
