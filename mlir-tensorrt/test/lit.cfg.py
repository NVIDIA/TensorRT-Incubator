# -*- Python -*-
# pyright: reportAttributeAccessIssue=false
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any

import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.LitConfig import LitConfig
from lit.TestingConfig import TestingConfig

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
config.gpu_tools_script = os.path.join(
    config.test_source_root,
    "gpu_tools.py",
)


def load_gpu_tools_module():
    assert Path(config.gpu_tools_script).exists(), "gpu_tools.py script does not exist"
    spec = importlib.util.spec_from_file_location("gpu_tools", config.gpu_tools_script)
    gpu_tools = importlib.util.module_from_spec(spec)
    sys.modules["gpu_tools"] = gpu_tools
    spec.loader.exec_module(gpu_tools)
    return gpu_tools


gpu_tools = load_gpu_tools_module()


def estimate_paralllelism(mem_required: float) -> int:
    try:
        with gpu_tools.nvml_context() as devices:
            return gpu_tools.estimate_parallelism_from_memory(devices, mem_required)
    except:
        return 2


# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_tensorrt_obj_root, "test")

# mlir_tensorrt_tools_dir: binary output path for tool executables
config.mlir_tensorrt_tools_dir = os.path.join(config.mlir_tensorrt_obj_root)

# Add additional expansions that can be used in the `RUN:` commands.
config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%test_src_root", config.test_source_root))

# Setup the parallelism groups.
lit_config.parallelism_groups["non-collective"] = estimate_paralllelism(2.0)
lit_config.parallelism_groups["collective"] = 1
lit_config.parallelism_groups["models"] = estimate_paralllelism(8.0)
lit_config.parallelism_group = None

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
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
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
    "mlir-tensorrt-opt",
    ToolSubst(
        "%pick-one-gpu",
        f"CUDA_VISIBLE_DEVICES=$(python3 -m mlir_tensorrt.tools.gpu_tools pick-device)",
    ),
    "mlir-tensorrt-translate",
    "mlir-tensorrt-runner",
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
    os.path.join(config.mlir_tensorrt_obj_root, "tensorrt/lib"),
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


def get_num_cuda_devices() -> int:
    try:
        with gpu_tools.nvml_context() as devices:
            return len(devices)
    except:
        return 0


def all_gpus_have_fp8_support() -> bool:
    try:
        with gpu_tools.nvml_context() as _:
            return gpu_tools.has_fp8_support()
    except:
        return False


# Add configuration features that depend on the host or flags defined with the
# `-D[flag-name]=[value]` option via the `llvm-lit` CLI. These features can be
# used to predicate tests by adding "REQUIRES: feature-name" to the top of the
# test file near the RUN command.
trt_version = config.mlir_tensorrt_compile_time_version.split(".")
trt_version_major, trt_version_minor = int(trt_version[0]), int(trt_version[1])
for i in range(1, get_num_cuda_devices() + 1):
    config.available_features.add(f"host-has-at-least-{i}-gpus")

if all_gpus_have_fp8_support():
    config.available_features.add(f"all-gpus-support-fp8")

if shutil.which("nsys"):
    config.available_features.add("host-has-nsight-systems")
if lit.util.pythonize_bool(lit_config.params.get("enable_benchmark_suite", None)):
    config.available_features.add("enable_benchmark_suite")
if lit.util.pythonize_bool(lit_config.params.get("enable_functional_suite", None)):
    config.available_features.add("enable_functional_suite")
if trt_version_major < 9:
    config.available_features.add("tensorrt-version-lt-9.0")
if trt_version_major == 9:
    config.available_features.add("tensorrt-version-eq-9")
if trt_version_major >= 10:
    config.available_features.add("tensorrt-version-ge-10.0")
if not config.enable_asan:
    config.available_features.add("no-asan")

# Some of our tests utilize checks against debug output in order to verify
# that flags were correctly propagated from e.g. Python all the way to the TensorRT
# translation pass. This is a bit brittle, but until we replace with a better
# solution, we can only run those tests when debug printing is available.
if config.enable_assertions:
    config.available_features.add("debug-print")
