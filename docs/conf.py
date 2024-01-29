import os
import sys

ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.path.pardir)
sys.path.insert(0, ROOT_DIR)
import contextlib
import io
from textwrap import dedent, indent

import tripy as tp
from tests import helper
import inspect
import re

PARAM_PAT = re.compile(":param .*?:")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
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
    "documentation_font_size": "1.0rem",
    "monospace_font_size": "0.8rem",
}

html_sidebars = {"**": ["globaltoc.html"]}

# Show source link
html_show_sourcelink = True

# Output file base name for HTML help builder.
htmlhelp_basename = "TripyDoc"

# For constructor arguments to show up in Sphinx generated doc
autoclass_content = "both"

# Myst will complain about relative links in our top-level README
suppress_warnings = ["myst.xref_missing"]

# Ignore docs/README.md as that's for developers and not supposed to be included in the public docs.
exclude_patterns = ["README.md"]


def process_docstring(app, what, name, obj, options, lines):
    doc = "\n".join(lines).strip()
    blocks = helper.consolidate_code_blocks(doc)

    TRIPY_CLASSES = [obj for obj in helper.discover_tripy_objects() if inspect.isclass(obj)]

    if inspect.isfunction(obj):
        signature = inspect.signature(obj)

        # We don't currently check overload dispatchers since this would require manual parsing of the docstring.
        if not hasattr(obj, "is_overload_dispatcher"):
            # The docstring has been processed at this point such that parameters appear as `:param <name>:`
            documented_args = {match.replace(":param ", "").rstrip(":") for match in PARAM_PAT.findall(doc)}

            for name, param in signature.parameters.items():
                if name == "self":
                    # Don't want a type annotation for the self parameter.
                    assert (
                        param.annotation == signature.empty
                    ), f"Avoid using type annotations for the `self` parameter since this will corrupt the rendered documentation!"
                else:
                    assert name in documented_args, f"Missing documentation for parameter: '{name}' in: '{obj}'"

                    assert (
                        param.annotation != signature.empty
                    ), f"Missing type annotation for parameter: '{name}' in: '{obj}'"

            assert signature.return_annotation != signature.empty, (
                f"Missing return type annotation for: '{obj}'. "
                f"Hint: If this interface does not return anything, use a type annotation of `-> None`."
            )

            if signature.return_annotation != None:
                assert ":returns:" in doc, f"For: {obj}, return value is not documented."

    lines.clear()
    for block in blocks:
        if not isinstance(block, helper.CodeBlock):
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

        for block_line in block.splitlines():
            if block_line.strip() == NO_PRINT_LOCALS:
                should_append_locals = False

            if block_line.strip().startswith(PRINT_LOCALS):
                _, _, names = block_line.strip().partition(PRINT_LOCALS)
                print_vars.update(names.strip().split(" "))

            if any(block_line.strip().startswith(tag) for tag in REMOVE_TAGS) or block_line.endswith(OMIT_COMMENT):
                continue

            lines.append(block_line)

        # Add output as a separate code block.
        outfile = io.StringIO()

        def get_stdout():
            outfile.flush()
            outfile.seek(0)
            return outfile.read().strip()

        try:
            with contextlib.redirect_stdout(outfile), contextlib.redirect_stderr(outfile):
                code_locals = helper.exec_doc_example(dedent(block))
        except:
            print(f"Failed while processing docstring for: {what}: {name} ({obj})")
            print(f"Note: Code example was:\n{block}")
            print(get_stdout())
            raise

        stdout = get_stdout() or ""

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
                        ret += indent(f"{key}: {value},\n", prefix=" " * 4)
                    ret += "}"
                    return ret

                if isinstance(obj, tp.nn.Module):
                    locals_str += f"\n{name}.state_dict(): {pretty_str_from_dict(obj.state_dict())}\n"
                elif isinstance(obj, dict):
                    locals_str += f"\n{name}: {pretty_str_from_dict(obj)}\n"
                else:
                    locals_str += f"\n{name}: {obj}\n"

        def add_block(title, contents):
            line = block.splitlines()[1]
            indentation = len(line) - len(line.lstrip())

            out = (
                indent(
                    f"\n{title}:\n:: \n\n{indent(contents, prefix=' ' * 4)}",
                    prefix=" " * (indentation - 4),
                )
                + "\n\n"
            )
            lines.extend(out.splitlines())

        if stdout:
            add_block("Output", stdout)
        if locals_str:
            add_block("Variable Values", locals_str)


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
