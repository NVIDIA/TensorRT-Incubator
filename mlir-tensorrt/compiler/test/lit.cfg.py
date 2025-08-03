# -*- Python -*-
# pyright: reportAttributeAccessIssue=false
import importlib.util
import os
import shutil
import sys
from pathlib import Path

from lit.LitConfig import LitConfig
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.TestingConfig import TestingConfig
import lit.formats
import lit.util
import psutil

config: TestingConfig = config  # type: ignore
lit_config: LitConfig = lit_config  # type: ignore

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "MLIR_TENSORRT"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = {".mlir", ".py", ".test"}

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)


def estimate_paralllelism(
    gb_gpu_mem_required: float, gb_sys_mem_required: float
) -> int:
    try:
        parallelism = 2
        if config.enable_cuda:
            with gpu_tools.nvml_context() as (devices, _):
                parallelism = gpu_tools.estimate_parallelism_from_memory(
                    devices, gb_gpu_mem_required
                )
        return int(
            max(
                min(
                    parallelism,
                    (0.5 * psutil.virtual_memory().available / (1024**3))
                    // gb_sys_mem_required,
                ),
                1,
            )
        )
    except:
        return 2


# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_tensorrt_obj_root, "compiler", "test")

# mlir_tensorrt_tools_dir: binary output path for tool executables
config.mlir_tensorrt_tools_dir = os.path.join(config.mlir_tensorrt_obj_root, "bin")

# Add additional expansions that can be used in the `RUN:` commands.
config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%test_src_root", config.test_source_root))
config.substitutions.append(("%mtrt_src_dir", config.mlir_tensorrt_root))
config.substitutions.append(
    ("%trt_include_dir", os.path.join(config.tensorrt_lib_dir, "..", "include"))
)
config.substitutions.append(("%trt_lib_dir", config.tensorrt_lib_dir))
config.substitutions.append(("%stablehlo_src_dir", config.stablehlo_source_root))
config.substitutions.append(
    (
        "%cuda_toolkit_linux_cxx_flags",
        config.cuda_toolkit_linux_cxx_flags,
    )
)


# Setup the parallelism groups. Note that just instantiating the TRT builder
# requires ~2.5 GB of system memory, so we use 3.0 as a baseline limit.
lit_config.parallelism_groups["non-collective"] = estimate_paralllelism(
    2.0, gb_sys_mem_required=5.0
)
lit_config.parallelism_groups["collective"] = 1
lit_config.parallelism_groups["models"] = estimate_paralllelism(
    8.0, gb_sys_mem_required=6.0
)
config.parallelism_group = "non-collective"

print(f"Parallelism Groups: {lit_config.parallelism_groups}", file=sys.stderr)

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = {
    "Inputs",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "lit.cfg.py",
    "test_utils.py",
    "models",
}


# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.mlir_tensorrt_tools_dir, append_path=True)

if config.enable_asan:
    config.environment["ASAN_OPTIONS"] = "protect_shadow_gap=0,detect_leaks=0"


# Enable CUDA lazy function loading.
llvm_config.with_environment("CUDA_MODULE_LOADING", "LAZY")


def make_tool_with_preload_prefix(tool: str):
    # Returns a tool prefixed with setting the right LD_PRELOAD for ensuring that the
    # right ASAN runtime is loaded (in the case where shared libraries produced by the project are
    # being dynamically loaded, e.g. PyBind modules).
    return f"LD_PRELOAD=$({config.host_cxx} -print-file-name=libclang_rt.asan-{config.host_arch}.so) {tool}"


def make_tool_with_single_device_selection(tool: str):
    return (
        f"CUDA_VISIBLE_DEVICES=$(python3 -m mlir_tensorrt.tools.gpu_tools pick-device)"
    )


tool_dirs = [config.mlir_tensorrt_tools_dir]
tools = [
    ToolSubst("%mlir_src_dir", config.mlir_src_root, unresolved="ignore"),
    ToolSubst(
        "%pick-one-gpu",
        f"CUDA_VISIBLE_DEVICES=$(python3 -m mlir_tensorrt.tools.gpu_tools pick-device)",
    ),
    "mlir-tensorrt-compiler",
    "mlir-tensorrt-opt",
    "mlir-tensorrt-runner",
    "mlir-tensorrt-translate",
    ToolSubst("%host_cxx", command=config.host_cxx, unresolved="warn"),
]

LINUX_ASAN_ENABLED = "Linux" in config.host_os and config.enable_asan

# If ASAN is enabled, then configure the Python executable to actually refer to a combination of setting LD_PRELOAD and invoking python.
python_executable = config.python_executable
if LINUX_ASAN_ENABLED:
    python_executable = make_tool_with_preload_prefix(python_executable)
tools.extend(
    [
        ToolSubst("%PYTHON", python_executable, unresolved="ignore"),
    ]
)

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Add MLIR TensorRT Python module to the path
llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(
            config.mlir_tensorrt_obj_root, "python_packages", "mlir_tensorrt_compiler"
        ),
        os.path.join(
            config.mlir_tensorrt_obj_root, "python_packages", "mlir_tensorrt_runtime"
        ),
        os.path.join(
            config.mlir_tensorrt_obj_root, "python_packages", "mlir_tensorrt_tools"
        ),
    ],
    append_path=True,
)

extra_ld_lib_path = [
    # Ensure we can find DSOs containing test plugins for `tensorrt.opaque_plugin`.
    config.mlir_tensorrt_lib_dir
]

if config.tensorrt_lib_dir:
    # If we compiled with a particular installation of TensorRT, then it's libraries
    # won't by default be on the OS's dynamic lib search path.We forward the
    # installation path given to the cmake configuration command here.
    extra_ld_lib_path.append(config.tensorrt_lib_dir)

llvm_config.with_environment(
    "LD_LIBRARY_PATH",
    extra_ld_lib_path,
    append_path=True,
)


# Add configuration features that depend on the host or flags defined with the
# `-D[flag-name]=[value]` option via the `llvm-lit` CLI. These features can be
# used to predicate tests by adding "REQUIRES: feature-name" to the top of the
# test file near the RUN command.
if shutil.which("nsys"):
    config.available_features.add("host-has-nsight-systems")
if lit.util.pythonize_bool(lit_config.params.get("enable_benchmark_suite", None)):
    config.available_features.add("enable_benchmark_suite")
if lit.util.pythonize_bool(lit_config.params.get("enable_functional_suite", None)):
    config.available_features.add("enable_functional_suite")
if not config.enable_asan:
    config.available_features.add("no-asan")


# Some of our tests utilize checks against debug output in order to verify
# that flags were correctly propagated from e.g. Python all the way to the TensorRT
# translation pass. This is a bit brittle, but until we replace with a better
# solution, we can only run those tests when debug printing is available.
if config.enable_assertions:
    config.available_features.add("debug-print")
