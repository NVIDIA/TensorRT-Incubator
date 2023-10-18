"""
Tests that ensure the user experience is nice. For example, making sure that
README links work.
"""

import glob
import importlib
import inspect
import os
import pkgutil
import re
from textwrap import dedent

import pytest
import requests

import tripy
from tests.helper import ROOT_DIR
from tripy.flat_ir import FlatIR
from tripy.frontend import TensorExpression


class TestReadme:
    README_TEST_CASES = [
        path
        for path in glob.glob(os.path.join(ROOT_DIR, "**", "*.md"), recursive=True)
        if not path.startswith((os.path.join(ROOT_DIR, "build"), os.path.join(ROOT_DIR, "mlir-tensorrt")))
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
def discover_modules():
    mods = [tripy]
    while mods:
        mod = mods.pop(0)

        yield mod

        if hasattr(mod, "__path__"):
            mods.extend(
                [
                    importlib.import_module(f"{mod.__name__}.{submod.name}")
                    for submod in pkgutil.iter_modules(mod.__path__)
                ]
            )


def discover_tripy_objects():
    for mod in discover_modules():
        yield from [
            obj for obj in mod.__dict__.values() if hasattr(obj, "__module__") and obj.__module__.startswith("tripy")
        ]


# Returns a list of all classes, functions, and methods defined in Tripy.
def get_all_tripy_interfaces():
    all_objects = set()
    for obj in discover_tripy_objects():
        all_objects.add(obj)
        all_objects.update({member for _, member in inspect.getmembers(obj, inspect.isfunction)})

    # Some sanity checks to make sure we're actually getting all the objects we expect
    assert TensorExpression in all_objects
    assert FlatIR in all_objects

    return all_objects


def get_all_docstrings_with_examples():
    docstrings = []
    ids = []
    for obj in get_all_tripy_interfaces():
        if not obj.__doc__ or "Example:" not in obj.__doc__:
            print(f"Skipping {obj.__qualname__} because no example was present in the docstring")
            continue

        # NOTE: For now, we assume that the example is the last thing in the docstring.
        # This simplifies the parsing logic. If we add other things after the example,
        # please update this function!
        example_code = dedent(obj.__doc__.partition("Example:")[-1].partition("::")[-1])

        docstrings.append(example_code)
        ids.append(obj.__qualname__)

    return docstrings, ids


DOCSTRING_TEST_CASES, DOCSTRING_IDS = get_all_docstrings_with_examples()


class TestDocstrings:
    @pytest.mark.parametrize("example_code", DOCSTRING_TEST_CASES, ids=DOCSTRING_IDS)
    def test_examples_in_docstrings(self, example_code):
        # Don't inherit variables from the current environment so we can be sure the docstring examples
        # work in total isolation.
        exec(example_code, {}, {})
