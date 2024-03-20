import os
import sys

ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.path.pardir)
sys.path.insert(0, ROOT_DIR)
import contextlib
import inspect
import io
import re
from textwrap import dedent, indent

import tripy as tp
from tests import helper

TAB_SIZE = 4

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

myst_url_schemes = {"source": "https://gitlab-master.nvidia.com/TensorRT/poc/tripy/-/blob/main/{{path}}"}
myst_number_code_blocks = ["py", "rst"]

# Ignore most markdown files as they are not part of the API reference documentation.
exclude_patterns = ["README.md", "development/*.md"]

# When class documentation is generated, 'process_docstring' is called twice - once for
# the class docstring and again for the '__init__' docstring. We only want to check the
# function signature for the latter.
seen_classes = set()


def process_docstring(app, what, name, obj, options, lines):
    doc = "\n".join(lines).strip()
    blocks = helper.consolidate_code_blocks(doc)

    TRIPY_CLASSES = [tripy_obj for tripy_obj in helper.discover_tripy_objects() if inspect.isclass(tripy_obj)]

    # Check signature for functions/methods and class constructors.
    if what in {"function", "method"} or (what == "class" and name in seen_classes):
        signature = inspect.signature(obj)

        # We don't currently check overload dispatchers since this would require manual parsing of the docstring.
        if not hasattr(obj, "is_overload_dispatcher"):
            # The docstring has been processed at this point such that parameters appear as `:param <name>:`
            documented_args = {match.replace(":param ", "").rstrip(":") for match in PARAM_PAT.findall(doc)}

            for pname, param in signature.parameters.items():
                if pname == "self":
                    # Don't want a type annotation for the self parameter.
                    assert (
                        param.annotation == signature.empty
                    ), f"Avoid using type annotations for the `self` parameter since this will corrupt the rendered documentation!"
                else:
                    assert (
                        pname in documented_args
                    ), f"Missing documentation for parameter: '{pname}' in: '{obj}'. Please ensure you've included this in the `Args:` section"

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
            what in {"attribute", "module", "class"}
            or
            # Modules include examples in their constructors
            (what == "method" and obj.__name__ == "__call__")
        )

    if not allow_no_example():
        assert ".. code-block:: python" in doc, f"For: {obj} no example was provided. Please add an example!"

    lines.clear()
    for block in blocks:
        if not isinstance(block, helper.DocstringCodeBlock):
            lines.append(block)
            continue

        # Add back the code block after removing assertions.
        NO_PRINT_LOCALS = "# doc: no-print-locals"
        PRINT_LOCALS = "# doc: print-locals"
        REMOVE_TAGS = ["assert ", NO_PRINT_LOCALS, PRINT_LOCALS]
        OMIT_COMMENT = "# doc: omit"

        should_append_locals = True
        # By default, we print all local variables. If `print_vars` it not empty,
        # then we'll only print those that appear in it.
        print_vars = set()

        code_block_lines = []
        for block_line in block.splitlines():
            if block_line.strip() == NO_PRINT_LOCALS:
                should_append_locals = False

            if block_line.strip().startswith(PRINT_LOCALS):
                _, _, names = block_line.strip().partition(PRINT_LOCALS)
                print_vars.update(names.strip().split(" "))

            if any(block_line.strip().startswith(tag) for tag in REMOVE_TAGS) or block_line.endswith(OMIT_COMMENT):
                continue

            code_block_lines.append(block_line)

        def add_block(title, contents, lang="python"):
            line = block.splitlines()[1]
            indentation = len(line) - len(line.lstrip())

            out = (
                indent(
                    f"\n\n.. code-block:: {lang}\n"
                    + indent((f":caption: {title}" if title else "") + f"\n\n{contents}", prefix=" " * TAB_SIZE),
                    prefix=" " * (indentation - 4),
                )
                + "\n\n"
            )
            code_block_lines.extend(out.splitlines())

        # Add output as a separate code block.
        outfile = io.StringIO()

        def get_stdout():
            outfile.flush()
            outfile.seek(0)
            return outfile.read().strip()

        code = dedent(block.code())
        try:
            with contextlib.redirect_stdout(outfile), contextlib.redirect_stderr(outfile):
                code_locals = helper.exec_code(code)
        except:
            print(f"Failed while processing docstring for: {what}: {name} ({obj})")
            print(f"Note: Code example was:\n{code}")
            print(get_stdout())
            raise

        # Add local variables as a separate code block
        locals_str = ""
        if should_append_locals:
            for name, obj in code_locals.items():

                def should_print():
                    if name in print_vars:
                        return True

                    if print_vars and name not in print_vars:
                        return False

                    # Skip over any non-tripy types.
                    if not any(isinstance(obj, tripy_obj) for tripy_obj in TRIPY_CLASSES):
                        return False

                    EXCLUDE_OBJECTS = [tp.jit]

                    if any(isinstance(obj, exclude_obj) for exclude_obj in EXCLUDE_OBJECTS):
                        return False

                    return True

                if not should_print():
                    continue

                def pretty_str_from_dict(dct):
                    if not dct:
                        return r"{}"
                    ret = "{\n"
                    for key, value in dct.items():
                        ret += indent(f"{key}: {value},\n", prefix=" " * TAB_SIZE)
                    ret += "}"
                    return ret

                locals_str += f"\n>>> {name}"
                if isinstance(obj, tp.Module):
                    locals_str += f".state_dict()\n{pretty_str_from_dict(obj.state_dict())}"
                elif isinstance(obj, dict):
                    locals_str += f"\n{pretty_str_from_dict(obj)}"
                else:
                    locals_str += f"\n{obj}"

        if locals_str:
            add_block("", locals_str)

        stdout = get_stdout() or ""

        if stdout:
            add_block("Output:", stdout, lang="")

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
                f"\n.. collapse:: {caption}" + (("\n" + (" " * TAB_SIZE) + ":open:") if default_open else "") + "\n\n",
                prefix=" " * indentation,
            ).splitlines()
        )
        code_block_lines = indent("\n".join(code_block_lines) + "\n", prefix=" " * TAB_SIZE).splitlines()
        lines.extend(code_block_lines)


def setup(app):
    # A note on aliases: if you rename a class via an import statement, e.g. `import X as Y`,
    # the documentation generated for `Y` will just be: "Alias of X"
    # To get the real documentation, you can make Sphinx think that `Y` is not an alias but instead a real
    # class/function. To do so, you just need to define the __name__ attribute in this function (*not* in tripy code!):
    #   Y.__name__ = "Y"

    app.connect("autodoc-process-docstring", process_docstring)
