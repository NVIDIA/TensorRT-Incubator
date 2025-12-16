# -*- Python -*-
# Configuration file for the 'lit' test runner for the unittests (GTest-based C++ tests).

import os

import lit.formats

# name: The name of this test suite.
config.name = "MLIR-TensorRT-PJRT-Unit"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.pjrt_obj_root, "test", "Unit")
config.test_source_root = config.test_exec_root

lit_config.parallelism_groups["normal"] = 1
config.parallelism_group = "normal"

# testFormat: The test format to use to interpret tests.
# The last parameter "Tests" indicates that all executables that are GTest tests
# have the suffix "Tests".
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")

if config.enable_asan:
    config.environment["ASAN_OPTIONS"] = "protect_shadow_gap=0,detect_leaks=0"

# If we compiled with a particular installation of TensorRT, then it's libraries
# won't by default be on the OS's dynamic lib search path.We forward the
# installation path given to the cmake configuration command here.
if config.tensorrt_lib_dir:
    ld_library_path = [config.tensorrt_lib_dir]
    if "LD_LIBRARY_PATH" in config.environment:
        ld_library_path += [config.environment["LD_LIBRARY_PATH"]]
    config.environment["LD_LIBRARY_PATH"] = os.pathsep.join(ld_library_path)
