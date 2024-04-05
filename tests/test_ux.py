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
from tripy.export import PUBLIC_APIS


class TestReadme:
    @pytest.mark.parametrize("readme", helper.MARKDOWN_FILES)
    def test_links_valid(self, readme):
        MD_LINK_PAT = re.compile(r"\[.*?\]\((.*?)\)")
        DOC_REFERENCE_PAT = re.compile(r"\{[a-z]+\}`(.*?)`")

        with open(readme, "r") as f:
            contents = f.read()
            links = MD_LINK_PAT.findall(contents)
            doc_references = DOC_REFERENCE_PAT.findall(contents)

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
                PROJECT_REF_TAG = "project:"
                if link.startswith(SOURCE_TAG) or link.startswith(PROJECT_REF_TAG):
                    if link.startswith(SOURCE_TAG):
                        _, _, link = link.partition(SOURCE_TAG)

                        assert (
                            link.startswith(os.path.sep) and os.path.pardir not in link
                        ), f"All links to paths that are not markdown files must be absolute, but got: {link}"
                    else:
                        _, _, link = link.partition(PROJECT_REF_TAG)

                        assert not link.startswith(
                            os.path.sep
                        ), f"All links to rendered markdown files must be relative, but got: {link}"

                else:
                    # Omit `docs/README.md` since it's not part of the rendered documentation (but rather describes the documentation).
                    assert "docs/" not in readme or "docs/README.md" in readme, (
                        f"All links must include a leading '{SOURCE_TAG}' or {PROJECT_REF_TAG}."
                        "Use the former to link to source files and the latter to link to other markdown files "
                        "that are part of the *rendered* docs (for markdown files in the source code, use the former tag.)."
                    )

                if link.startswith(os.path.sep):
                    link_full_path = os.path.join(helper.ROOT_DIR, link.lstrip(os.path.sep))
                else:
                    link_full_path = os.path.abspath(os.path.join(readme_dir, link))

                assert os.path.exists(
                    link_full_path
                ), f"In README: '{readme}', link: '{link}' does not exist. Note: Full path was: {link_full_path}"

        for doc_reference in doc_references:
            # Each doc reference should point to a fully qualified name using the name "tripy"
            try:
                exec(doc_reference, {"tripy": tp}, {})
            except:
                print(f"Error while processing document reference: {doc_reference}")
                raise


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

    @pytest.mark.parametrize("api", PUBLIC_APIS)
    def test_all_public_apis_have_docstrings(self, api):
        assert api.obj.__doc__, f"All public APIs must include docstrings but: {api.obj} has no docstring!"


class TestMissingAttributes:
    def test_no_inifinite_looping_for_invalid_attributes(self):
        with pytest.raises(AttributeError):
            tp.no_way_this_will_ever_be_a_real_function_name

    @pytest.mark.parametrize(
        "get_func, message",
        [
            (lambda x: x.ones_like, "tripy.ones_like"),
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
