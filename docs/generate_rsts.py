import argparse
import glob
import inspect
import os
import re
import shutil
from collections import defaultdict
from textwrap import dedent, indent
from typing import Dict, Set

from tripy.export import PUBLIC_APIS


def to_snake_case(string):
    return re.sub("([A-Z]+)", r"_\1", string).lower().lstrip("_")


def make_heading(title):
    heading_separator = "#" * len(title)

    return f"{heading_separator}\n{title}\n{heading_separator}\n\n"


def build_api_doc(api, include_heading=True):

    automod = "autodata"
    if inspect.isfunction(api.obj):
        automod = "autofunction"
    elif inspect.isclass(api.obj):
        automod = "autoclass"

    return (
        "\n"
        + (make_heading(api.obj.__name__) if include_heading else "")
        + dedent(
            f"""
        .. {automod}:: tripy.{api.obj.__name__}
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


def build_root_index_file(constituents, developer_guides):
    return (
        (
            # We want to include the top-level README in our main index file.
            dedent(
                f"""
                .. include:: {os.path.join(os.path.pardir, os.path.pardir, "README.md")}
                    :parser: myst_parser.sphinx_

                .. toctree::
                    :caption: API Reference
                    :maxdepth: 1
                """
            )
        ).strip()
        + process_index_constituents(constituents)
        + "\n\n"
        + (
            # We want to include the top-level README in our main index file.
            dedent(
                f"""
                .. toctree::
                    :caption: Developer Guides
                    :maxdepth: 1
                """
            )
        )
        + indent("\n" + "\n".join(developer_guides), prefix=" " * 4)
    )


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
        # Overloads will show up as multiple separate objects in PUBLIC_APIS, but
        # we only need to document them once.
        if api.obj.__name__ in seen_apis:
            continue
        seen_apis.add(api.obj.__name__)

        if is_file(api.document_under):
            rst_path = make_output_path(*api.document_under.split("/"))
            rst_filename = os.path.basename(rst_path)
        else:
            rst_filename = f"{to_snake_case(api.obj.__name__)}.rst"
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
        with open(rst_path, "a") as f:
            f.write(build_api_doc(api, include_heading=api.include_heading))

    def str_from_hierarchy(obj):
        if isinstance(obj, dict):
            ret = ""
            for key, value in obj.items():
                ret += f"\n{key}/\n" + str_from_hierarchy(value) + "\n"
            return ret
        return "\n".join(f"\t- {val}" for val in obj)

    print(f"Generating documentation hierarchy:\n{str_from_hierarchy(doc_hierarcy)}")

    developer_guides = []
    # This is hard-coded to support only the development docs for now, but can be generalized
    # to support arbitrary markdown files in the docs directory.
    for guide in glob.iglob(os.path.join("docs", "development", "*.md")):
        guide_file = os.path.basename(guide)
        guide_out_path = make_output_path("development", os.path.basename(guide).replace(".md", ".rst"))
        os.makedirs(os.path.dirname(guide_out_path), exist_ok=True)
        with open(guide_out_path, "w") as f:
            f.write(
                build_markdown_doc(
                    source_path=os.path.join(
                        os.path.pardir, os.path.pardir, os.path.pardir, "docs", "development", guide_file
                    )
                )
            )
        developer_guides.append(f"development/{os.path.splitext(guide_file)[0]}")

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
                build_root_index_file(constituents, developer_guides)
                if is_root
                else build_index_file(
                    name=os.path.basename(os.path.dirname(index_path)).replace("_", " ").title(),
                    constituents=constituents,
                    include_heading=not pre_existing_file,
                    caption="See also:" if pre_existing_file else None,
                )
            )


if __name__ == "__main__":
    main()
