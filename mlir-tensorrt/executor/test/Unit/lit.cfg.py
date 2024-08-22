# -*- Python -*-
# Configuration file for the 'lit' test runner for the unittests (GTest-based C++ tests).

import os

import lit.formats

# name: The name of this test suite.
config.name = "MLIR-Executor-Unit"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.executor_obj_root, "test", "Unit")
config.test_source_root = config.test_exec_root

# testFormat: The test format to use to interpret tests.
# The last parameter "Tests" indicates that all executables that are GTest tests
# have the suffix "Tests".
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")

if config.enable_asan:
    config.environment["ASAN_OPTIONS"] = "protect_shadow_gap=0,detect_leaks=0"
