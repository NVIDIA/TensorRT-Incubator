# -*- Python -*-

import os
import sys

import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
import psutil

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "MLIR_TENSORRT_DIALECT"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

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
config.substitutions.append(("%tensorrt_dialect_libs", config.tensorrt_dialect_lib_dir))

if config.enable_asan:
    config.environment["ASAN_OPTIONS"] = "protect_shadow_gap=0,detect_leaks=0"


def make_tool_with_preload_prefix(tool: str):
    # Returns a tool prefixed with setting the right LD_PRELOAD for ensuring that the
    # right ASAN runtime is loaded (in the case where shared libraries produced by the project are
    # being dynamically loaded, e.g. PyBind modules).
    return f"LD_PRELOAD=$({config.host_cxx} -print-file-name=libclang_rt.asan-{config.host_arch}.so) {tool}"


LINUX_ASAN_ENABLED = "Linux" in config.host_os and config.enable_asan

# If ASAN is enabled, then configure the Python executable to actually refer to a combination of setting LD_PRELOAD and invoking python.
python_executable = config.python_executable
if LINUX_ASAN_ENABLED:
    python_executable = make_tool_with_preload_prefix(python_executable)

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment(
    "PYTHONPATH",
    [config.gpu_tools_package_path],
    append_path=True,
)


llvm_config.with_environment(
    "LD_LIBRARY_PATH",
    os.pathsep.join([config.tensorrt_lib_dir, config.tensorrt_dialect_lib_dir]),
)


tool_dirs = [config.tensorrt_dialect_tools_dir, config.llvm_tools_dir]
tools = [
    "tensorrt-opt",
    ToolSubst(
        "%pick-one-gpu",
        f"CUDA_VISIBLE_DEVICES=$(python3 -m mlir_tensorrt.tools.gpu_tools pick-device)",
    ),
    ToolSubst("%PYTHON", python_executable, unresolved="ignore"),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)


if not config.enable_asan:
    config.available_features.add("no-asan")


def estimate_paralllelism(
    gb_gpu_mem_required: float, gb_sys_mem_required: float
) -> int:
    try:
        parallelism = 2
        with gpu_tools.nvml_context() as (devices, _):
            parallelism = gpu_tools.estimate_parallelism_from_memory(
                devices, gb_gpu_mem_required
            )
        return int(
            min(
                parallelism,
                (psutil.virtual_memory().available / (1024**3)) // gb_sys_mem_required,
            )
        )
    except:
        return 2


# Setup the parallelism groups.
lit_config.parallelism_groups["translation-tests"] = estimate_paralllelism(
    8.0, gb_sys_mem_required=3.0
)
config.parallelism_group = None
