# -*- Python -*-

from pathlib import Path
import importlib.util
import os
import sys

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "MLIR_EXECUTOR"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".lua"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]
config.substitutions.append(("%executor_libs", config.executor_libs_dir))
if config.enable_asan:
    config.environment["ASAN_OPTIONS"] = "protect_shadow_gap=0,detect_leaks=0"

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.executor_tools_dir, config.llvm_tools_dir]
tools = ["executor-opt", "executor-translate", "executor-runner"]

llvm_config.add_tool_substitutions(tools, tool_dirs)


def load_gpu_tools_module():
    assert Path(config.gpu_tools_script).exists(), "gpu_tools.py script does not exist"
    spec = importlib.util.spec_from_file_location("gpu_tools", config.gpu_tools_script)
    gpu_tools = importlib.util.module_from_spec(spec)
    sys.modules["gpu_tools"] = gpu_tools
    spec.loader.exec_module(gpu_tools)
    return gpu_tools


gpu_tools = load_gpu_tools_module()
config.num_cuda_devices = gpu_tools.get_num_cuda_devices()

if config.enable_assertions:
    config.available_features.add("debug-print")

for i in range(config.num_cuda_devices):
    config.available_features.add(f"host-has-at-least-{i+1}-gpus")
