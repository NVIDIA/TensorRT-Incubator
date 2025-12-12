# -*- Python -*-
import importlib.util
import os
import sys
import shutil
from pathlib import Path

import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
import psutil

LINUX_ASAN_ENABLED = "Linux" in config.host_os and config.enable_asan


def make_tool_with_preload_prefix(tool: str):
    if not LINUX_ASAN_ENABLED:
        return tool
    # Returns a tool prefixed with setting the right LD_PRELOAD for ensuring that the
    # right ASAN runtime is loaded (in the case where shared libraries produced by the project are
    # being dynamically loaded, e.g. PyBind modules).
    return f"LD_PRELOAD=$({config.host_cxx} -print-file-name=libclang_rt.asan-{config.host_arch}.so) {tool}"


# name: The name of this test suite.
config.name = "MLIR_TENSORRT_PJRT"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.pjrt_obj_root, "test")
config.mlir_trt_jax_py_executable = make_tool_with_preload_prefix("mlir-trt-jax-py")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

# Setup PJRT flags
# User can pass '-Dpjrt_flags="...."' to the `llvm-lit` command line to
# update flags.
pjrt_flags = ["-tensorrt-builder-opt-level=0"] + lit_config.params.get(
    "pjrt_flags", ""
).split(" ")
llvm_config.with_environment(
    "MLIR_TRT_FLAGS",
    " ".join(list(filter(lambda x: len(x) > 0, pjrt_flags))),
)

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()


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
            max(
                min(
                    parallelism,
                    (0.5 * (psutil.virtual_memory().available / (1024**3)))
                    // gb_sys_mem_required,
                ),
                1,
            )
        )
    except:
        return 2


# Setup the parallelism groups.
ESTIMATED_GPU_MEMORY_PER_JAX_TEST_MB = 2048.0
ESTIMATED_GPU_MEMORY_PER_JAX_MODEL_TEST_MB = 8192.0

# For JAX tests, based on some experimentation, we want to allow up to 10
# concurrent tests if the GPU allows. The "pick-one-gpu" trick used in the command
# line will ensure that the test doesn't launch unless at least 'ESTIMATED_GPU_MEMORY_PER_JAX_TEST_MB'
# are available on the target device.
lit_config.parallelism_groups["non-collective"] = min(
    estimate_paralllelism(
        ESTIMATED_GPU_MEMORY_PER_JAX_TEST_MB / 1024.0, gb_sys_mem_required=5.0
    ),
    10,
)
lit_config.parallelism_groups["collective"] = 1

# For models, reduce parallelism further.
lit_config.parallelism_groups["models"] = min(
    estimate_paralllelism(
        ESTIMATED_GPU_MEMORY_PER_JAX_MODEL_TEST_MB / 1024.0, gb_sys_mem_required=5.0
    ),
    5,
)
config.parallelism_group = "non-collective"

print(f"Parallelism Groups: {lit_config.parallelism_groups}", file=sys.stderr)


# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "Examples",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "lit.cfg.py",
    "conftest.py",
    "test_utils.py",
    "models",
    "mtrt_jax_pytest_plugin.py",
]
config.pjrt_libs_dir = os.path.join(config.pjrt_obj_root, "lib")
config.substitutions.append(("%pjrt_libs", config.pjrt_libs_dir))
if config.enable_asan:
    config.environment["ASAN_OPTIONS"] = "protect_shadow_gap=0,detect_leaks=0"

config.environment["MTRT_LIT_MANAGED_ENV"] = "1"

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment(
    "PATH", os.path.join(config.mlir_tensorrt_obj_root, "bin"), append_path=True
)
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

tool_dirs = [config.llvm_tools_dir, os.path.join(config.mlir_tensorrt_obj_root, "bin")]
tools = [
    ToolSubst(
        "%PYTHON",
        make_tool_with_preload_prefix(config.python_executable),
        unresolved="ignore",
    ),
    ToolSubst(
        "%pick-one-gpu",
        f"CUDA_VISIBLE_DEVICES=$(python3 -m mlir_tensorrt.tools.gpu_tools pick-device --required-memory {ESTIMATED_GPU_MEMORY_PER_JAX_TEST_MB})",
    ),
    ToolSubst(
        "%pick-one-gpu-for-model-test",
        f"CUDA_VISIBLE_DEVICES=$(python3 -m mlir_tensorrt.tools.gpu_tools pick-device --required-memory {ESTIMATED_GPU_MEMORY_PER_JAX_MODEL_TEST_MB})",
    ),
    ToolSubst(
        "%mlir-trt-jax-py", config.mlir_trt_jax_py_executable, unresolved="ignore"
    ),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)


# -----------------------------------------------------------------------------
# Feature Support
# Add configuration features that depend on the host or flags defined with the
# `-D[flag-name]=[value]` option via the `llvm-lit` CLI. These features can be
# used to predicate tests by adding "REQUIRES: feature-name" to the top of the
# test file near the RUN command.
# -----------------------------------------------------------------------------
if lit.util.pythonize_bool(lit_config.params.get("enable_benchmark_suite", None)):
    config.available_features.add("enable_benchmark_suite")
if lit.util.pythonize_bool(lit_config.params.get("run_mini_benchmark", None)):
    config.available_features.add("mini_benchmark")
if lit.util.pythonize_bool(lit_config.params.get("enable_functional_suite", None)):
    config.available_features.add("enable_functional_suite")
if not config.enable_asan:
    config.available_features.add("no-asan")
if config.enable_nccl:
    config.available_features.add("nccl")

try:
    import flashinfer_jit_cache

    config.available_features.add("flashinfer_jit_cache")
except:
    pass
