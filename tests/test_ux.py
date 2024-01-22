"""
Tests that ensure the user experience is nice. For example, making sure that
README links work.
"""

import glob
import inspect
import os
import re
from textwrap import dedent

import pytest
import requests

from tests import helper
from tests.helper import ROOT_DIR
from tripy.frontend import Tensor
from tripy.frontend.trace import Trace


class TestReadme:
    README_TEST_CASES = [
        path
        for path in glob.glob(os.path.join(ROOT_DIR, "**", "*.md"), recursive=True)
        if not path.startswith(
            (
                os.path.join(ROOT_DIR, "build"),
                os.path.join(ROOT_DIR, "mlir-tensorrt"),
                os.path.join(ROOT_DIR, "stablehlo"),
            )
        )
    ]

    @pytest.mark.parametrize("readme", README_TEST_CASES)
    def test_links_valid(self, readme):
        MD_LINK_PAT = re.compile(r"\[.*?\]\((.*?)\)")

        readme_dir = os.path.dirname(readme)
        with open(readme, "r") as f:
            links = MD_LINK_PAT.findall(f.read())

        for link in links:
            link, _, _ = link.partition("#")  # Ignore section links for now
            if link.startswith("https://"):
                assert requests.get(link).status_code == 200
            else:
                assert os.path.pathsep * 2 not in link, "Duplicate slashes break links in GitHub"
                link_abs_path = os.path.abspath(os.path.join(readme_dir, link))
                assert os.path.exists(
                    link_abs_path
                ), f"In README: '{readme}', link: '{link}' does not exist. Note: Full path was: '{link_abs_path}'"


# In order to test docstrings, we need to recursively discover all submodules
# and any classes/functions contained in those submodules.


# Returns a list of all classes, functions, and methods defined in Tripy.
def get_all_tripy_interfaces():
    all_objects = set()
    for obj in helper.discover_tripy_objects():
        all_objects.add(obj)
        all_objects.update(
            {
                member
                for _, member in inspect.getmembers(
                    obj,
                    lambda member: inspect.isfunction(member)
                    or isinstance(member, property)
                    or inspect.isclass(member),
                )
            }
        )

    # Some sanity checks to make sure we're actually getting all the objects we expect
    assert Tensor in all_objects
    assert Tensor.shape in all_objects
    assert Trace in all_objects

    return all_objects


def get_all_docstrings_with_examples():
    def get_qualname(obj):
        if isinstance(obj, property):
            return obj.fget.__qualname__
        return obj.__qualname__

    # NOTE: If you edit the parsing logic here, please also update `tests/README.md`.
    docstrings = []
    ids = []
    for obj in get_all_tripy_interfaces():
        if not obj.__doc__:
            print(f"Skipping {get_qualname(obj)} because no docstring was present")
            continue

        blocks = [
            dedent(block)
            for block in helper.consolidate_code_blocks(obj.__doc__)
            if isinstance(block, helper.CodeBlock)
        ]
        if blocks is None:
            print(f"Skipping {get_qualname(obj)} because no example was present in the docstring")
            continue

        docstrings.extend(blocks)
        ids.extend([f"{get_qualname(obj)}:{idx}" for idx in range(len(blocks))])

    return docstrings, ids


DOCSTRING_TEST_CASES, DOCSTRING_IDS = get_all_docstrings_with_examples()


class TestDocstrings:
    @pytest.mark.parametrize("example_code", DOCSTRING_TEST_CASES, ids=DOCSTRING_IDS)
    def test_examples_in_docstrings(self, example_code):
        assert example_code, "Example code is empty! Is the formatting correct? Refer to `tests/README.md`."
        for banned_module in ["numpy", "tripy"]:
            assert (
                f"import {banned_module}" not in example_code
            ), f"Avoid importing {banned_module} in example docstrings"
            assert f"from {banned_module}" not in example_code, f"Avoid importing {banned_module} in example docstrings"

        helper.exec_doc_example(example_code)
