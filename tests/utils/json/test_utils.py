import pytest

import tripy as tp


def test_load_json_errors_if_file_nonexistent():
    with pytest.raises(FileNotFoundError, match="No such file"):
        tp.load("nonexistent-path")
