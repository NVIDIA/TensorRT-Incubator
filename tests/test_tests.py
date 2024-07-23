"""
Tests that ensure that all tests do not have any randomness calls
"""

import glob
import os

import pytest
from tests import helper


class TestImports:
    INVALID_TEXT = ["random", "randn"]

    @pytest.mark.parametrize(
        "file_path",
        [file_path for file_path in glob.iglob(os.path.join(helper.ROOT_DIR, "tests", "**", "*.py"), recursive=True)],
    )
    def test_no_invalid_imports(self, file_path):
        if file_path.endswith("test_test.py"):
            with open(file_path, "r") as file:
                content = file.read()
                for invalid_text in self.INVALID_TEXT:
                    assert invalid_text not in content, f"Found '{invalid_text}' in {file_path}"
