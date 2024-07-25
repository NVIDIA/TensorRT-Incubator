import tripy as tp


def test_timing_cache(tmp_path):
    # Create a dummy file using pytest fixture for timing cache.
    dummy_file = tmp_path / "dummy.txt"
    tp.config.timing_cache_file_path = str(dummy_file)

    def func(a, b):
        c = a + b
        return c

    compiler = tp.Compiler(func)

    compiler.compile(tp.InputInfo((2,), dtype=tp.float32), tp.InputInfo((2,), dtype=tp.float32))

    assert dummy_file.parent.exists(), "Cache directory was not created."
    assert dummy_file.exists(), "Cache file was not created."

    dummy_file = tmp_path / "dummy1.txt"
    tp.config.timing_cache_file_path = str(dummy_file)

    # Check if new timing cache file is created when we recompile.
    compiler.compile(tp.InputInfo((1,), dtype=tp.float32), tp.InputInfo((1,), dtype=tp.float32))

    assert dummy_file.exists(), "Cache file was not created."
    assert tp.config.timing_cache_file_path == str(
        dummy_file
    ), f"get_timing_cache_file() path does not match the user provided path {str(dummy_file)}"
