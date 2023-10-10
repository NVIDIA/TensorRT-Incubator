"""
Tests that ensure the user experience is nice. For example, making sure that
README links work.
"""

import glob
import os
import re

import pytest
import requests

from tests.helper import ROOT_DIR

README_TEST_CASES = [
    path
    for path in glob.glob(os.path.join(ROOT_DIR, "**", "*.md"), recursive=True)
    if not path.startswith((os.path.join(ROOT_DIR, "build"), os.path.join(ROOT_DIR, "mlir-tensorrt")))
]


class TestReadme:
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
