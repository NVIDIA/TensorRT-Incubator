#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys

ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.path.pardir)
sys.path.insert(0, ROOT_DIR)
import inspect
import re
from textwrap import indent

from tests import helper

import tripy as tp
from tripy.common.datatype import DATA_TYPES
from tripy.constraints import TYPE_VERIFICATION

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

html_static_path = ["_static"]

html_theme = "furo"

html_title = f"Tripy {tp.__version__}"

html_theme_options = {
    "light_logo": "logo-light-mode.png",
    "dark_logo": "logo-dark-mode.png",
    "light_css_variables": {
        "color-api-pre-name": "#4e9a06",
        "color-api-name": "#4e9a06",
        "color-api-background": "#e8e8e8",
    },
    "dark_css_variables": {
        "color-api-background": "#303030",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/TensorRT-Incubator",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

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
    "source": "https://github.com/NVIDIA/TensorRT-Incubator/tree/main/tripy/{{path}}",
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
    unqual_name = name.split(".")[-1]

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

                # Type annotations are optional for the `self` parameter unless the API has to be type-verified.
                if pname != "self" or unqual_name in TYPE_VERIFICATION:
                    assert (
                        pname in documented_args
                    ), f"Missing documentation for parameter: '{pname}' in: '{obj}'. Please ensure you've included this in the `Args:` section. Note: Documented parameters were: {documented_args} {doc}"
                    assert (
                        pname in documented_args
                    ), f"Missing documentation for parameter: '{pname}' in: '{obj}'. Please ensure you've included this in the `Args:` section. Note: Documented parameters were: {documented_args}"

                    assert (param.annotation != signature.empty) or param.kind in {
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    }, f"Missing type annotation for parameter: '{pname}' in: '{obj}'. Please update the signature with type annotations"

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

    if unqual_name in TYPE_VERIFICATION:
        add_text_index = -1
        for index, block in enumerate(blocks):

            def insert_block(text):
                nonlocal index

                blocks.insert(index, text)
                index += 1

            if re.search(r".. code-block::", block):
                type_dict = TYPE_VERIFICATION[unqual_name].dtypes
                insert_block("TYPE CONSTRAINTS:")
                # Add the dtype constraint name and the dtypes that correlate.
                for type_name, dt in type_dict.items():
                    insert_block(
                        f"    - **{type_name}**: :class:`"
                        + "`, :class:`".join(
                            sorted(
                                set(dt),
                                key=lambda dtype: (
                                    tuple(typ.__name__ for typ in DATA_TYPES[dtype].__bases__),
                                    DATA_TYPES[dtype].itemsize,
                                ),
                            )
                        )
                        + "`",
                    )
                insert_block("\n")

                if TYPE_VERIFICATION[unqual_name].dtype_exceptions:
                    # Add the dtype exceptions.
                    insert_block("UNSUPPORTED TYPE COMBINATIONS:")
                    for exception_dict in TYPE_VERIFICATION[unqual_name].dtype_exceptions:
                        insert_block(
                            "    - "
                            + ", ".join([f"**{key}**\ =\ :class:`{val}`" for key, val in exception_dict.items()]),
                        )
                    insert_block("\n")
                break

            if re.search(r":param \w+: ", block):
                param_name = re.match(r":param (\w+): ", block).group(1)
                # Add dtype constraint to start of each parameter description.
                if TYPE_VERIFICATION[unqual_name].dtype_constraints.get(param_name, None):
                    add_text_index = re.search(r":param \w+: ", block).span()[1]
                    blocks[index] = (
                        f"{block[0:add_text_index]}[*dtype=*\ **{TYPE_VERIFICATION[unqual_name].dtype_constraints[param_name]}**\ ] {block[add_text_index:]}"
                    )

            if TYPE_VERIFICATION[unqual_name].return_dtype is not None and re.search(r":returns:", block):
                add_text_index = re.search(r":returns:", block).span()[1] + 1
                # Add dtype constraint to start of returns description.
                blocks[index] = (
                    f"{block[0:add_text_index]}[*dtype=*\ **{TYPE_VERIFICATION[unqual_name].return_dtype}**\ ] {block[add_text_index:]}"
                )

    seen_classes.add(name)

    def allow_no_example():
        # `tp.Module`s include examples in their constructors, so their __call__ methods don't require examples.
        is_tripy_module_call_method = False
        if what == "method" and obj.__name__ == "__call__":
            class_name = name.rpartition(".")[0]
            # Class names will be prefixed with tripy.<...>, so we need to import it here to make eval() work.
            import tripy

            is_tripy_module_call_method = issubclass(eval(class_name), tp.Module)

        return what in {"attribute", "module", "class", "data"} or is_tripy_module_call_method

    if not allow_no_example():
        assert (
            ".. code-block:: python" in doc
        ), f"For: {obj} (which is a: '{what}'), no example was provided. Please add an example!"

    lines.clear()
    for block in blocks:
        if not isinstance(block, helper.DocstringCodeBlock):
            lines.append(block)
            continue

        code_block_lines, local_var_lines, output_lines, _ = helper.process_code_block_for_outputs_and_locals(
            block,
            block.code(),
            format_contents=lambda title, contents, lang: f"\n\n.. code-block:: {lang}\n"
            + indent((f":caption: {title}" if title else "") + f"\n\n{contents}", prefix=" " * helper.TAB_SIZE),
            err_msg=f"Failed while processing docstring for: {what}: {name} ({obj})",
            strip_assertions=True,
        )

        code_block_lines += local_var_lines + output_lines

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
