import pytest

import tripy as tp


# TODO (#142): Unwaive test once serialization is actually enabled.
@pytest.mark.skip("Serialization is not enabled")
def test_load_json_errors_if_file_nonexistent():
    with pytest.raises(FileNotFoundError, match="No such file"):
        tp.load("nonexistent-path")
