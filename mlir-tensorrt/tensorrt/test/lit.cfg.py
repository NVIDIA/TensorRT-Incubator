# -*- Python -*-

import os
import sys

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "MLIR_TENSORRT_DIALECT"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.tensorrt_dialect_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]
config.tensorrt_dialect_tools_dir = os.path.join(
    config.tensorrt_dialect_obj_root, "bin"
)
config.tensorrt_dialect_libs_dir = os.path.join(config.tensorrt_dialect_obj_root, "lib")
config.substitutions.append(
    ("%tensorrt_dialect_libs", config.tensorrt_dialect_libs_dir)
)
if config.enable_asan:
    config.environment["ASAN_OPTIONS"] = "protect_shadow_gap=0,detect_leaks=0"

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment(
    "PYTHONPATH",
    [config.gpu_tools_package_path],
    append_path=True,
)


llvm_config.with_environment(
    "LD_LIBRARY_PATH",
    [config.tensorrt_lib_dir, config.tensorrt_dialect_libs_dir],
    append_path=True,
)


tool_dirs = [config.tensorrt_dialect_tools_dir, config.llvm_tools_dir]
tools = [
    "tensorrt-opt",
    ToolSubst(
        "%pick-one-gpu",
        f"CUDA_VISIBLE_DEVICES=$(python3 -m mlir_tensorrt.tools.gpu_tools pick-device)",
    ),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Setup information about CUDA devices on the host.
gpu_tools = sys.modules["gpu_tools"]
config.num_cuda_devices = gpu_tools.get_num_cuda_devices()


def all_gpus_have_fp8_support() -> bool:
    try:
        with gpu_tools.nvml_context() as _:
            return gpu_tools.has_fp8_support()
    except Exception as e:
        return False


if all_gpus_have_fp8_support():
    config.available_features.add(f"all-gpus-support-fp8")
for i in range(config.num_cuda_devices):
    config.available_features.add(f"host-has-at-least-{i+1}-gpus")

# Setup features related to the TensorRT version
trt_version = config.mlir_tensorrt_compile_time_version.split(".")
trt_version_major, trt_version_minor = int(trt_version[0]), int(trt_version[1])
if trt_version_major < 9:
    config.available_features.add("tensorrt-version-lt-9.0")
if trt_version_major == 9:
    config.available_features.add("tensorrt-version-eq-9")
if trt_version_major >= 10:
    config.available_features.add("tensorrt-version-ge-10.0")
