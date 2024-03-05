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

        with open(readme, "r") as f:
            links = MD_LINK_PAT.findall(f.read())

        readme_dir = os.path.dirname(readme)
        for link in links:
            link, _, _ = link.partition("#")  # Ignore section links for now

            if not link:
                continue

            if link.startswith("https://"):
                assert requests.get(link).status_code == 200
            else:
                assert os.path.sep * 2 not in link, f"Duplicate slashes break links in GitHub. Link was: {link}"
                SOURCE_TAG = "source:"
                if link.startswith(SOURCE_TAG):
                    _, _, link = link.partition(SOURCE_TAG)

                    assert (
                        link.startswith(os.path.sep) and os.path.pardir not in link
                    ), f"All links to paths that are not markdown files must be absolute, but got: {link}"

                else:
                    # Omit `docs/README.md` since it's not part of the rendered documentation (but rather describes the documentation).
                    assert ("docs/" not in readme or "docs/README.md" in readme) or link.endswith(
                        ".md"
                    ), f"For markdown files in the `docs/` directory, only links to markdown files can omit the leading {SOURCE_TAG}."

                if link.startswith(os.path.sep):
                    link_full_path = os.path.join(helper.ROOT_DIR, link.lstrip(os.path.sep))
                else:
                    link_full_path = os.path.abspath(os.path.join(readme_dir, link))

                assert os.path.exists(
                    link_full_path
                ), f"In README: '{readme}', link: '{link}' does not exist. Note: Full path was: {link_full_path}"


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

    @pytest.mark.parametrize(
        "get_func, message",
        [
            (lambda x: x.ones_like, "tripy.ones_like"),
            (lambda x: x.softmax, "tripy.nn.softmax"),
        ],
    )
    def test_nice_error_for_tensor_attr(self, get_func, message):
        a = tp.Tensor([1, 2])
        with pytest.raises(AttributeError, match=f"Did you mean: '{message}'?"):
            get_func(a)

    def test_no_inifinite_looping_for_invalid_tensor_attributes(self):
        a = tp.Tensor([1, 2])
        with pytest.raises(AttributeError):
            a.no_way_this_will_ever_be_a_real_tensor_attr_name
