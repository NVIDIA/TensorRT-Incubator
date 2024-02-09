"""
Tests that ensure the user experience is nice. For example, making sure that
README links work.
"""

import os
import re

import pytest
import requests

import tripy as tp
from tests import helper


class TestReadme:

    @pytest.mark.parametrize("readme", helper.MARKDOWN_FILES)
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


DOCSTRING_TEST_CASES, DOCSTRING_IDS = helper.get_all_docstrings_with_examples()


class TestDocstrings:
    @pytest.mark.parametrize("example_code", DOCSTRING_TEST_CASES, ids=DOCSTRING_IDS)
    def test_examples_in_docstrings(self, example_code):
        assert example_code, "Example code is empty! Is the formatting correct? Refer to `tests/README.md`."
        for banned_module in ["numpy", "tripy", "torch"]:
            assert (
                f"import {banned_module}" not in example_code
            ), f"Avoid importing {banned_module} in example docstrings"
            assert f"from {banned_module}" not in example_code, f"Avoid importing {banned_module} in example docstrings"

        helper.exec_code(example_code)


class TestMissingAttributes:
    # When we try to access a missing attribute, tripy should issue a nice error
    # if it exists under a different class/submodule.
    @pytest.mark.parametrize(
        "get_func, message",
        [
            (lambda: tp.exp, "tripy.Tensor.exp"),
            (lambda: tp.softmax, "tripy.nn.softmax"),
            (lambda: tp.nn.gather, "tripy.Tensor.gather"),
        ],
    )
    def test_nice_error_for_similar_attributes(self, get_func, message):
        with pytest.raises(AttributeError, match=f"Did you mean: '{message}'?"):
            get_func()

    def test_no_inifinite_looping_for_invalid_attributes(self):
        with pytest.raises(AttributeError):
            tp.no_way_this_will_ever_be_a_real_function_name
