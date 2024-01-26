import os
import sys

ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.path.pardir)
sys.path.insert(0, ROOT_DIR)
import contextlib
import io
from textwrap import dedent, indent

import tripy as tp
from tests import helper

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
    "special-members": "True",
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
copyright = "2024, NVIDIA"
author = "NVIDIA"

version = tp.__version__
# The full version, including alpha/beta/rc tags.
release = version

# Style
pygments_style = "colorful"
pygments_dark_style = "one-dark"

html_static_path = ["_static"]

html_theme = "sphinx_nefertiti"

html_theme_options = {
    "style": "green",
    "show_powered_by": False,
    "documentation_font_size": "16px",
    "monospace_font_size": "14px",
}

html_sidebars = {"**": ["globaltoc.html"]}

# Hide source link
html_show_sourcelink = True

# Output file base name for HTML help builder.
htmlhelp_basename = "TripyDoc"

# Template files to extend default Sphinx templates.
# See https://www.sphinx-doc.org/en/master/templating.html for details.
templates_path = ["_templates"]

# For constructor arguments to show up in Sphinx generated doc
autoclass_content = "both"

# Myst will complain about relative links in our top-level README
suppress_warnings = ["myst.xref_missing"]

# Ignore docs/README.md as that's for developers and not supposed to be included in the public docs.
exclude_patterns = ["README.md"]


def process_docstring(app, what, name, obj, options, lines):
    doc = "\n".join(lines).strip()
    blocks = helper.consolidate_code_blocks(doc)

    lines.clear()
    for block in blocks:
        if isinstance(block, helper.CodeBlock):
            # Add back the code block after removing assertions.
            remove_tags = ["assert "]
            OMIT_COMMENT = "# doc: omit"

            def should_omit(line):
                line = line.strip()
                return any(line.startswith(tag) for tag in remove_tags) or line.endswith(OMIT_COMMENT)

            lines.extend([block_line for block_line in block.splitlines() if not should_omit(block_line)])

            # Add output as a separate code block.
            outfile = io.StringIO()

            def get_stdout():
                outfile.flush()
                outfile.seek(0)
                return outfile.read().strip()

            try:
                with contextlib.redirect_stdout(outfile):
                    helper.exec_doc_example(dedent(block))
            except:
                print(f"Failed while processing docstring for: {what}: {name} ({obj})")
                print(f"Note: Code example was:\n{block}")
                print(get_stdout())
                raise

            stdout = get_stdout()
            if stdout:
                line = block.splitlines()[1]
                indentation = len(line) - len(line.lstrip())

                out = (
                    indent("\nOutput:\n:: \n\n" + indent(f"{stdout}", prefix=" " * 4), prefix=" " * (indentation - 4))
                    + "\n\n"
                )
                lines.extend(out.splitlines())
        else:
            lines.append(block)


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

    app.connect("autodoc-process-docstring", process_docstring)
