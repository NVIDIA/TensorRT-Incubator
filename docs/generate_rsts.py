
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from textwrap import dedent, indent
from typing import Dict, List, Set

import tripy as tp
from tests import helper
from tripy.export import PUBLIC_APIS


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


def build_markdown_doc(source_path):
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


def build_root_index_file(constituents, guide_sets):
    contents = (
        (
            # We want to include the top-level README in our main index file.
            dedent(
                f"""
                .. include:: {os.path.join(os.path.pardir, os.path.pardir, "README.md")}
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


def process_guide(guide_path: str, processed_guide_path: str):

    os.makedirs(os.path.dirname(processed_guide_path), exist_ok=True)

    new_blocks = []
    blocks = helper.consolidate_code_blocks_from_readme(guide_path)

    # We need to maintain all the code we've seen so far since future
    # code might rely on things that were already defined in previous code blocks.
    code_locals = {}
    for block in blocks:
        if block.lang.startswith("sh"):
            subprocess.call(str(block), shell=True)

        if not block.lang.startswith("py"):
            new_blocks.append(block.raw_str())
            continue

        def add_block(title, contents, lang):
            print(f"Appending output block for code in: {guide_path}")
            # Only include the "Output:" heading when the code block is actually rendered in the documentation.
            return (
                "\n"
                + (title if not block.has_marker("comment") else "")
                + f"\n```{lang}\n{dedent(contents).strip()}\n```"
            )

        code_block_lines, code_locals = helper.update_code_block_with_outputs_and_locals(
            block.raw_str(),
            str(block),
            err_msg=f"Error while executing code block from {guide_path}.",
            format_contents=add_block,
            local_vars=code_locals,
        )

        new_blocks.extend(code_block_lines)

    with open(processed_guide_path, "w") as fout:
        fout.write("\n".join(new_blocks))


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

    shutil.rmtree(args.output, ignore_errors=True)

    def make_output_path(*components):
        return os.path.join(args.output, *components)

    doc_hierarcy: Dict[str, Set[str]] = defaultdict(set)

    def is_file(document_under):
        return bool(os.path.splitext(document_under)[1])

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

        pre_existing_file = os.path.exists(rst_path)
        with open(rst_path, "a") as f:
            f.write(build_api_doc(api, include_heading=not pre_existing_file))

    def str_from_hierarchy(obj):
        if isinstance(obj, dict):
            ret = ""
            for key, value in obj.items():
                ret += f"\n{key}/\n" + str_from_hierarchy(value) + "\n"
            return ret
        return "\n".join(f"\t- {val}" for val in obj)

    print(f"Generating documentation hierarchy:\n{str_from_hierarchy(doc_hierarcy)}")

    guide_sets: List[GuideSet] = []
    guide_dirs = list(
        filter(
            lambda path: not path.startswith("_"),
            [os.path.basename(path) for path in glob.iglob(os.path.join("docs", "*")) if os.path.isdir(path)],
        )
    )

    processed_markdown_dirname = "processed_mds"
    processed_markdown_dir = make_output_path(processed_markdown_dirname)
    for dir_path in sorted(guide_dirs):
        # Extract order information from directory name.
        order, _, title = dir_path.partition("_")
        assert order.startswith("pre") or order.startswith(
            "post"
        ), f"Guide directories must start with a pre<N>/post<N> prefix, but got: {dir_path}"
        is_before_api_ref = "pre" in order
        index = int(order.replace("pre", "").replace("post", ""))

        title = to_title(title)
        guides = []
        for guide in sorted(glob.iglob(os.path.join("docs", dir_path, "*.md"))):
            # Copy guide to build directory
            guide_filename = os.path.basename(guide)
            processed_guide = os.path.join(processed_markdown_dir, dir_path, guide_filename)
            process_guide(guide, processed_guide)
            guide_out_path = make_output_path(dir_path, os.path.basename(processed_guide).replace(".md", ".rst"))
            os.makedirs(os.path.dirname(guide_out_path), exist_ok=True)
            with open(guide_out_path, "w") as f:
                f.write(
                    build_markdown_doc(
                        source_path=os.path.join(os.path.pardir, processed_markdown_dirname, dir_path, guide_filename)
                    )
                )
            guides.append(f"{dir_path}/{os.path.splitext(guide_filename)[0]}")

        guide_sets.append(GuideSet(title, guides, Order(is_before_api_ref, index)))

    for path, constituents in doc_hierarcy.items():
        is_root = path == ""

        index_path = make_output_path(path, "index.rst")
        pre_existing_file = os.path.exists(index_path)

        if is_root:
            assert (
                not pre_existing_file
            ), f"APIs should *not* target the root index file directly! Please remove any `document_under='index.rst'` arguments!"

        with open(index_path, "a") as f:
            f.write(
                build_root_index_file(constituents, guide_sets)
                if is_root
                else build_index_file(
                    name=to_title(os.path.basename(os.path.dirname(index_path))),
                    constituents=constituents,
                    include_heading=not pre_existing_file,
                    caption="See also:" if pre_existing_file else None,
                )
            )


if __name__ == "__main__":
    main()
