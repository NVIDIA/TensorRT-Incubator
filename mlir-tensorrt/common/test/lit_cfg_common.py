import os
from pathlib import Path
import shutil

config.gpu_tools_script = os.path.join(
    Path(__file__).parent,
    "../../integrations/python/mlir_tensorrt_tools/mlir_tensorrt/tools/gpu_tools.py",
)


def load_gpu_tools_module():
    from pathlib import Path
    import importlib.util
    import sys

    assert Path(config.gpu_tools_script).exists(), "gpu_tools.py script does not exist"
    spec = importlib.util.spec_from_file_location("gpu_tools", config.gpu_tools_script)
    gpu_tools = importlib.util.module_from_spec(spec)
    sys.modules["gpu_tools"] = gpu_tools
    spec.loader.exec_module(gpu_tools)
    return gpu_tools


try:
    if config.enable_cuda:
        try:
            print(
                f"CUDA Toolkit Version: {config.cuda_toolkit_version}", file=sys.stderr
            )
            cuda_toolkit_major = int(config.cuda_toolkit_version.split(".")[0])
        except:
            cuda_toolkit_major = 12

        if cuda_toolkit_major <= 12:
            config.available_features.add("cuda-toolkit-major-version-lte-12")

        if shutil.which("nsys"):
            config.available_features.add("host-has-nsight-systems")

        config.available_features.add("cuda")

        gpu_tools = load_gpu_tools_module()

        for i in range(1, gpu_tools.get_num_cuda_devices() + 1):
            config.available_features.add(f"host-has-at-least-{i}-gpus")

        for sm_version in gpu_tools.get_supported_sm_versions():
            feature_name = f"has-gpu-sm-gte-{sm_version}"
            config.available_features.add(feature_name)

        for ptx_version in gpu_tools.get_supported_ptx_versions():
            feature_name = f"has-support-for-ptx-gte-{ptx_version}"
            config.available_features.add(feature_name)

        if gpu_tools.all_gpus_have_fp8_support():
            config.available_features.add(f"all-gpus-support-fp8")

        if gpu_tools.all_gpus_have_fp4_support():
            config.available_features.add(f"all-gpus-support-fp4")


except Exception as e:
    print(
        f"In {__file__}, 'config.enable_cuda' is true, but an error was "
        f"encountered when loading the 'gpu_tools.py'  module: {e}."
        "Tests that require a GPU will be skipped.\n"
        "1. Check that the 'pynvml' package is installed in the environment used by the build:\n"
        "> python3 -m pip install pynvml\n"
        "2. Ensure that the CUDA driver is loaded and can detect the host's GPUs:\n"
        "> nvidia-smi"
    )


try:
    if config.target_tensorrt:
        config.available_features.add("tensorrt")

        trt_version = config.mlir_tensorrt_compile_time_version.split(".")
        trt_version_major, trt_version_minor = int(trt_version[0]), int(trt_version[1])

        if trt_version_major < 9:
            config.available_features.add("tensorrt-version-lt-9.0")
        if trt_version_major == 9:
            config.available_features.add("tensorrt-version-eq-9")
        if trt_version_major >= 10:
            for minor in range(0, 16):
                if trt_version_minor >= minor:
                    config.available_features.add(f"tensorrt-version-ge-10.{minor}")
except Exception as e:
    print(
        f"In {__file__}, 'config.target_tensorrt' is true, but an error was "
        f"encountered when detecting the TensorRT version: {e}"
    )

if config.enable_assertions:
    config.available_features.add("asserts")
    config.available_features.add("debug-print")
else:
    config.available_features.add("noasserts")
