# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import glob
import inspect
import os
import re
import subprocess as sp
from collections import defaultdict
from dataclasses import dataclass
from textwrap import dedent, indent
from typing import Dict, List, Set

import nvtripy as tp
from nvtripy.export import PUBLIC_APIS
from tests import helper


@dataclass
class Order:
    """
    Describes how markdown guides should be ordered in the index.
    """

    is_before_api_ref: bool
    index: int


@dataclass
class GuideSet:
    """
    Represents a set of markdown guides in the `docs/` directory
    """

    title: str
    guides: List[str]
    order: Order


def to_snake_case(string):
    return re.sub("([A-Z]+)", r"_\1", string).lower().lstrip("_")


def to_title(string):
    return string.replace("_", " ").title()


def get_name(api):
    return api.qualname.partition(f"{tp.__name__}.")[-1]


def make_heading(title):
    heading_separator = "#" * len(title)

    return f"{heading_separator}\n{title}\n{heading_separator}\n\n"


def build_api_doc(api, include_heading=True):
    automod = "autodata"
    name = get_name(api)
    if inspect.isfunction(api.obj):
        automod = "autofunction"
    elif inspect.isclass(api.obj):
        automod = "autoclass"
    elif inspect.ismodule(api.obj):
        automod = "automodule"

    return (
        "\n"
        + (make_heading(name) if include_heading else "")
        + dedent(
            f"""
        .. {automod}:: {api.qualname}
        """
        ).strip()
        + indent(("\n" + "\n".join(api.autodoc_options)) if api.autodoc_options else "", prefix=" " * 4)
        + "\n"
    )


def build_rst_markdown_include(source_path):
    return dedent(
        f"""
                .. include:: {source_path}
                    :parser: myst_parser.sphinx_
                  """
    ).strip()


def process_index_constituents(constituents):
    def process_consituent(constituent):
        if constituent.endswith(".rst"):
            return os.path.splitext(constituent)[0]
        return f"{constituent}/index"

    return indent(
        "\n\n" + "\n".join(process_consituent(constituent) for constituent in sorted(constituents)), prefix=" " * 4
    )


def build_root_index_file(constituents, guide_sets, processed_markdown_dirname):
    contents = (
        (
            # We want to include the top-level README in our main index file.
            dedent(
                f"""
                .. include:: {os.path.join(processed_markdown_dirname, "README.md")}
                    :parser: myst_parser.sphinx_
                """
            )
        ).strip()
        + "\n\n"
    )

    sorted_guide_sets = list(sorted(guide_sets, key=lambda guide_set: guide_set.order.index))

    def add_guide_set(guide_set):
        nonlocal contents
        contents += (
            dedent(
                f"""
                .. toctree::
                    :hidden:
                    :caption: {guide_set.title}
                    :maxdepth: 1
                """
            )
            + indent("\n" + "\n".join(guide_set.guides), prefix=" " * 4)
            + "\n\n"
        )

    for guide_set in filter(lambda guide_set: guide_set.order.is_before_api_ref, sorted_guide_sets):
        add_guide_set(guide_set)

    contents += (
        (
            dedent(
                f"""
                .. toctree::
                    :hidden:
                    :caption: API Reference
                    :maxdepth: 1
                """
            )
        ).strip()
        + process_index_constituents(constituents)
        + "\n\n"
    )
    for guide_set in filter(lambda guide_set: not guide_set.order.is_before_api_ref, sorted_guide_sets):
        add_guide_set(guide_set)
    return contents


def build_index_file(name, constituents, include_heading=True, caption=None):

    return (
        (make_heading(name) if include_heading else "\n\n")
        + dedent(
            f"""
        .. toctree::
            :maxdepth: 1
            {f":caption: {caption}" if caption else ""}
        """
        ).strip()
        + process_index_constituents(constituents)
    )


def write_if_different(filepath, content):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing_content = f.read()
        if existing_content == content:
            print(f"No changes for {filepath}")
            # Update the file timestamp so we know its up-to-date in timestamp checks.
            os.utime(filepath, None)
            return
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Updated {filepath}")


def process_guide(guide_path: str, processed_guide_path: str):
    if os.path.exists(processed_guide_path) and os.path.getmtime(guide_path) <= os.path.getmtime(processed_guide_path):
        print(f"Skipping {guide_path} as it is not newer than {processed_guide_path}")
        return

    os.makedirs(os.path.dirname(processed_guide_path), exist_ok=True)

    new_blocks = []
    blocks = helper.consolidate_code_blocks_from_readme(guide_path)

    # We need to maintain all the code we've seen so far since future
    # code might rely on things that were already defined in previous code blocks.
    code_locals = {}
    for index, block in enumerate(blocks):
        print(f"Processing block {index} (lang={block.lang}) in: {guide_path}: ", end="")

        should_eval = not block.has_marker("doc: no_eval_or_format")
        if should_eval and block.lang.startswith("py"):
            print("Evaluating Python block")

            def add_block(kind, contents, lang, code_indentation):
                # Indent our block to match the indentation of the code in the original block.

                if kind == helper.BlockKind.LOCAL_VARS:
                    return indent(
                        "\n```{eval-rst}"
                        + "\n.. collapse:: Local Variables\n"
                        + indent(
                            "\n::\n" + indent(contents, prefix=" " * helper.TAB_SIZE),
                            prefix=" " * (helper.TAB_SIZE * 2),
                        )
                        + "\n```\n",
                        prefix=" " * code_indentation,
                    )

                assert kind == helper.BlockKind.OUTPUT
                return indent(
                    "\n"
                    # Only include the "Output:" title when the code block is rendered.
                    # Hidden code blocks can be used to dynamically generate documentation.
                    + ("Output:" if not block.has_marker("doc: omit") else "")
                    + f"\n```{lang}\n{dedent(contents).strip()}\n```",
                    prefix=" " * code_indentation,
                )

            code_block_lines, local_var_lines, output_lines, code_locals = (
                helper.process_code_block_for_outputs_and_locals(
                    block.raw_str(),
                    format_contents=add_block,
                    err_msg=f"Error while executing code block {index} (line {block.line_number}) from {guide_path}. ",
                    local_vars=code_locals,
                    force_no_print_locals=block.has_marker("doc: no_print_locals"),
                )
            )

            if not block.has_marker("doc: omit"):
                new_blocks.extend(code_block_lines)

            if not block.has_marker("doc: no_print_locals"):
                new_blocks.extend(local_var_lines)
            new_blocks.extend(output_lines)

        else:
            if should_eval and (block.lang.startswith("sh") or block.lang.startswith("bash")):
                print("Evaluating shell block")
                commands = str(block).splitlines()
                for command in commands:
                    status = sp.run(command.strip(), shell=True)
                    status.check_returncode()

            if not block.has_marker("doc: omit"):
                print("Adding block")
                # All non-Python blocks are appended as-is.
                new_blocks.append(block.raw_str())
            else:
                print("Omitting block")

    write_if_different(processed_guide_path, "\n".join(new_blocks))


def main():
    parser = argparse.ArgumentParser(
        description="Uses the metadata populated by the `export.public_api()` decorator "
        "to automatically generate .rst files that can be used with Sphinx to generate documentation."
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory for generated .rst files. Any existing files will be deleted!",
        default=os.path.join("build", "doc_sources"),
    )

    args = parser.parse_args()

    def make_output_path(*components):
        return os.path.join(args.output, *components)

    doc_hierarcy: Dict[str, Set[str]] = defaultdict(set)

    def is_file(document_under):
        return bool(os.path.splitext(document_under)[1])

    rst_files = {}
    seen_apis = set()
    for api in PUBLIC_APIS:
        name = get_name(api)
        # Overloads will show up as multiple separate objects in PUBLIC_APIS, but
        # we only need to document them once.
        if name in seen_apis:
            continue
        seen_apis.add(name)

        if is_file(api.document_under):
            rst_path = make_output_path(*api.document_under.split("/"))
            rst_filename = os.path.basename(rst_path)
        else:
            rst_filename = f"{to_snake_case(name)}.rst"
            rst_path = make_output_path(*api.document_under.split("/"), rst_filename)

        path = ""
        if api.document_under:
            for component in api.document_under.split("/"):
                # Do not add files under document hierarchy.
                if is_file(component):
                    continue

                doc_hierarcy[path.rstrip("/")].add(component)
                path += f"{component}/"

        # We make a special exception for `index.rst` files since those are included automatically below.
        if rst_filename != "index.rst":
            doc_hierarcy[path.rstrip("/")].add(rst_filename)

        os.makedirs(os.path.dirname(rst_path), exist_ok=True)

        # Only include a heading if we didn't already create the file:
        include_heading = False
        if rst_path not in rst_files:
            include_heading = True
            rst_files[rst_path] = ""
        rst_files[rst_path] += build_api_doc(api, include_heading=include_heading)

    def str_from_hierarchy(obj):
        if isinstance(obj, dict):
            ret = ""
            for key, value in obj.items():
                ret += f"\n{key}/\n" + str_from_hierarchy(value) + "\n"
            return ret
        return "\n".join(f"\t- {val}" for val in obj)

    print(f"Generating documentation hierarchy:\n{str_from_hierarchy(doc_hierarcy)}")

    EXCLUDE_DIRS = ["_static", "_templates"]

    guide_sets: List[GuideSet] = []
    guide_dirs = list(
        filter(
            lambda path: not any(exclude in path for exclude in EXCLUDE_DIRS),
            # Include top-level directory for the main README.md
            [""] + [path for path in glob.iglob(os.path.join("docs", "*")) if os.path.isdir(path)],
        )
    )

    processed_markdown_dirname = "processed_mds"
    processed_markdown_dir = make_output_path(processed_markdown_dirname)
    for dir_path in sorted(guide_dirs):
        is_top_level_dir = dir_path == ""

        # Extract order information from directory name.
        basename = os.path.basename(dir_path)
        order, _, title = basename.partition("_")
        # Exempt top-level README from pre/post prefix requirements since they are not actually added to the final docs.
        if not is_top_level_dir:
            assert order.startswith("pre") or order.startswith(
                "post"
            ), f"Guide directories must start with a pre<N>/post<N> prefix, but got: {dir_path}"
            is_before_api_ref = "pre" in order
            index = int(order.replace("pre", "").replace("post", ""))

        title = to_title(title)
        guides = []

        if is_top_level_dir:
            # We only want the main README from the top-level directory.
            guide_mds = [os.path.join(dir_path, "README.md")]
        else:
            guide_mds = sorted(glob.iglob(os.path.join(dir_path, "*.md")))

        for guide in guide_mds:
            # Copy guide to build directory
            guide_filename = os.path.basename(guide)
            processed_guide = os.path.join(processed_markdown_dir, basename, guide_filename)
            process_guide(guide, processed_guide)
            guide_out_path = make_output_path(basename, os.path.basename(processed_guide).replace(".md", ".rst"))
            os.makedirs(os.path.dirname(guide_out_path), exist_ok=True)

            # Do not create RSTs for top-level README
            if not is_top_level_dir:
                content = build_rst_markdown_include(
                    source_path=os.path.relpath(processed_guide, os.path.dirname(guide_out_path))
                )
                rst_files[guide_out_path] = content

            guides.append(os.path.join(basename, os.path.splitext(guide_filename)[0]))

        # We have special treatment for the files in the top-level directory.
        if not is_top_level_dir:
            guide_sets.append(GuideSet(title, guides, Order(is_before_api_ref, index)))

    for path, constituents in doc_hierarcy.items():
        is_root = path == ""

        index_path = make_output_path(path, "index.rst")
        pre_existing_file = index_path in rst_files

        if is_root:
            assert (
                not pre_existing_file
            ), f"APIs should *not* target the root index file directly! Please remove any `document_under='index.rst'` arguments!"
            content = build_root_index_file(constituents, guide_sets, processed_markdown_dirname)
        else:
            content = build_index_file(
                name=to_title(os.path.basename(os.path.dirname(index_path))),
                constituents=constituents,
                include_heading=not pre_existing_file,
                caption="See also:" if pre_existing_file else None,
            )
        if not pre_existing_file:
            rst_files[index_path] = ""
        rst_files[index_path] += content

    # Delete any existing RST files that are not in rst_files
    for existing_file in glob.glob(os.path.join(args.output, "**", "*.rst"), recursive=True):
        if existing_file not in rst_files:
            print(f"Removing stale file: {existing_file}")
            os.remove(existing_file)

    # Write the new/updated RST files
    for path, content in rst_files.items():
        write_if_different(path, content)


if __name__ == "__main__":
    main()
