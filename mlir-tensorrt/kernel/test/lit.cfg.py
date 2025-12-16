# -*- Python -*-
import importlib.util
import os
from pathlib import Path

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "MLIR_KERNEL"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# gpu_tools_script: path to the gpu_tools.py script
config.gpu_tools_script = os.path.join(
    config.test_source_root,
    "../../integrations/python/mlir_tensorrt_tools/mlir_tensorrt/tools/gpu_tools.py",
)

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

LINUX_ASAN_ENABLED = "Linux" in config.host_os and config.enable_asan

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
]
config.kernel_libs_dir = os.path.join(config.kernel_obj_root, "lib")
config.substitutions.append(("%kernel_libs", config.kernel_libs_dir))
if config.enable_asan:
    config.environment["ASAN_OPTIONS"] = "protect_shadow_gap=0,detect_leaks=0"

# Add Python package module to the path
llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(config.project_obj_dir, "python_packages", "mlir_tensorrt_kernel"),
    ],
    append_path=True,
)

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.kernel_tools_dir, config.llvm_tools_dir]
tools = ["kernel-opt", "kernel-translate"]


def make_tool_with_preload_prefix(tool: str):
    # Returns a tool prefixed with setting the right LD_PRELOAD for ensuring that the
    # right ASAN runtime is loaded (in the case where shared libraries produced by the project are
    # being dynamically loaded, e.g. PyBind modules).
    return f"LD_PRELOAD=$({config.host_cxx} -print-file-name=libclang_rt.asan-{config.host_arch}.so) {tool}"


if LINUX_ASAN_ENABLED:
    config.python_executable = make_tool_with_preload_prefix(config.python_executable)


def load_gpu_tools_module():
    assert Path(config.gpu_tools_script).exists(), "gpu_tools.py script does not exist"
    spec = importlib.util.spec_from_file_location("gpu_tools", config.gpu_tools_script)
    gpu_tools = importlib.util.module_from_spec(spec)
    sys.modules["gpu_tools"] = gpu_tools
    spec.loader.exec_module(gpu_tools)
    return gpu_tools


if config.enable_cuda:
    try:
        config.available_features.add("cuda")
        gpu_tools = load_gpu_tools_module()
        for i in range(1, gpu_tools.get_num_cuda_devices() + 1):
            config.available_features.add(f"host-has-at-least-{i}-gpus")
    except:
        print(
            f"In {__file__}, 'config.enable_cuda' is true, but an error was "
            "encountered when detecting host GPU capabilities. "
            "Tests that require a GPU will be skipped. Check 'nvidia-smi' to "
            "ensure that the CUDA driver is loaded "
            "and can detect the host's GPUs."
        )


tools.extend(
    [
        ToolSubst("%PYTHON", config.python_executable, unresolved="ignore"),
    ]
)

llvm_config.add_tool_substitutions(tools, tool_dirs)
