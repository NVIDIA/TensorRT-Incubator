import os
import sys

ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.path.pardir)
sys.path.insert(0, ROOT_DIR)
import inspect
import re
from textwrap import dedent, indent

import tripy as tp
from tests import helper


PARAM_PAT = re.compile(":param .*?:")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_toolbox.collapse",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "myst_parser",
]

# Move type annotations to the description and don't use fully qualified names.
autodoc_typehints = "both"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

autodoc_default_options = {
    "members": True,
    "no-undoc-members": True,
    "show-inheritance": True,
    "special-members": "__call__",
}

autodoc_member_order = "bysource"

autodoc_inherit_docstrings = True

add_module_names = True

autosummary_generate = True

# For constructor arguments to show up in Sphinx generated doc
autoclass_content = "both"

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
pygments_style = "xcode"
pygments_dark_style = "one-dark"

html_static_path = ["_static"]

html_theme = "sphinx_nefertiti"

html_theme_options = {
    "style": "green",
    "show_powered_by": False,
    "documentation_font_size": "1.0rem",
    "monospace_font_size": "0.85rem",
    "repository_url": "https://gitlab-master.nvidia.com/TensorRT/poc/tripy",
    "repository_name": "Tripy",
}

html_sidebars = {"**": ["globaltoc.html"]}

# Show source link
html_show_sourcelink = True

# Output file base name for HTML help builder.
htmlhelp_basename = "TripyDoc"

html_css_files = ["style.css"]

# Myst will complain about relative links in our top-level README
suppress_warnings = ["myst.xref_missing"]

myst_fence_as_directive = ["mermaid"]

myst_url_schemes = {
    "http": None,
    "https": None,
    "source": "https://gitlab-master.nvidia.com/TensorRT/poc/tripy/-/blob/main/{{path}}",
}
myst_number_code_blocks = ["py", "rst"]

exclude_patterns = ["*.md"]

# When class documentation is generated, 'process_docstring' is called twice - once for
# the class docstring and again for the '__init__' docstring. We only want to check the
# function signature for the latter.
seen_classes = set()


def process_docstring(app, what, name, obj, options, lines):
    doc = "\n".join(lines).strip()
    blocks = helper.consolidate_code_blocks(doc)

    # Check signature for functions/methods and class constructors.
    if what in {"function", "method"} or (what == "class" and name in seen_classes):
        signature = inspect.signature(obj)

        # We don't currently check overload dispatchers since this would require manual parsing of the docstring.
        if not hasattr(obj, "is_overload_dispatcher"):
            # The docstring has been processed at this point such that parameters appear as `:param <name>:`
            documented_args = {
                match.replace(":param ", "").rstrip(":").replace("\\", "") for match in PARAM_PAT.findall(doc)
            }

            for pname, param in signature.parameters.items():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    pname = "**" + pname
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    pname = "*" + pname

                if pname == "self":
                    # Don't want a type annotation for the self parameter.
                    assert (
                        param.annotation == signature.empty
                    ), f"Avoid using type annotations for the `self` parameter since this will corrupt the rendered documentation!"
                else:
                    assert (
                        pname in documented_args
                    ), f"Missing documentation for parameter: '{pname}' in: '{obj}'. Please ensure you've included this in the `Args:` section. Note: Documented parameters were: {documented_args}"

                    assert (
                        param.annotation != signature.empty
                    ), f"Missing type annotation for parameter: '{pname}' in: '{obj}'. Please update the signature with type annotations"

                    assert not inspect.ismodule(
                        param.annotation
                    ), f"Type annotation cannot be a module, but got: '{param.annotation}' for parameter: '{pname}' in: '{obj}'. Please specify a type!"

            assert signature.return_annotation != signature.empty, (
                f"Missing return type annotation for: '{obj}'. "
                f"Hint: If this interface does not return anything, use a type annotation of `-> None`."
            )

            if signature.return_annotation != None:
                assert (
                    ":returns:" in doc
                ), f"For: {obj}, return value is not documented. Please ensure you've included a `Returns:` section"

    seen_classes.add(name)

    def allow_no_example():
        return (
            what in {"attribute", "module", "class", "data"}
            or
            # Modules include examples in their constructors
            (what == "method" and obj.__name__ == "__call__")
        )

    if not allow_no_example():
        assert (
            ".. code-block:: python" in doc
        ), f"For: {obj} (which is a: '{what}'), no example was provided. Please add an example!"

    lines.clear()
    for block in blocks:
        if not isinstance(block, helper.DocstringCodeBlock):
            lines.append(block)
            continue

        code_block_lines, _ = helper.update_code_block_with_outputs_and_locals(
            block,
            block.code(),
            err_msg=f"Failed while processing docstring for: {what}: {name} ({obj})",
            format_contents=lambda title, contents, lang: f"\n\n.. code-block:: {lang}\n"
            + indent((f":caption: {title}" if title else "") + f"\n\n{contents}", prefix=" " * helper.TAB_SIZE),
        )

        # Grab the caption from the example code block.
        for line in code_block_lines:
            caption_marker = ":caption:"
            if caption_marker in line:
                _, _, caption = line.partition(caption_marker)
                caption = caption.strip()
                if caption != "Example":
                    caption = f"Example: {caption}"
                break
        else:
            assert False, f"For: {obj}, example does not have a caption. Please add a caption to each example!"

        # Put the entire code block + output under a collapsible section to save space.
        line = code_block_lines[0]
        indentation = len(line) - len(line.lstrip())
        default_open = what != "method" and what != "property"
        lines.extend(
            indent(
                f"\n.. collapse:: {caption}"
                + (("\n" + (" " * helper.TAB_SIZE) + ":open:") if default_open else "")
                + "\n\n",
                prefix=" " * indentation,
            ).splitlines()
        )
        code_block_lines = indent("\n".join(code_block_lines) + "\n", prefix=" " * helper.TAB_SIZE).splitlines()
        lines.extend(code_block_lines)


def setup(app):
    # A note on aliases: if you rename a class via an import statement, e.g. `import X as Y`,
    # the documentation generated for `Y` will just be: "Alias of X"
    # To get the real documentation, you can make Sphinx think that `Y` is not an alias but instead a real
    # class/function. To do so, you just need to define the __name__ attribute in this function (*not* in tripy code!):
    #   Y.__name__ = "Y"

    app.connect("autodoc-process-docstring", process_docstring)
