"""
Utility functions used in the setup.py files for the Python packages.
"""

import datetime
import os
import re
import shutil
import sys
from pathlib import Path
import tempfile
import setuptools
import subprocess
import atexit

TENSORRT_VERSION = os.getenv("MLIR_TRT_DOWNLOAD_TENSORRT_VERSION", "10.12")


def log(*args):
    # When running the build, stdout may be captured, so print to stderr for debug info.
    print(*args, file=sys.stderr)


def get_cuda_version() -> str:
    try:
        cuda_version_environment_variable = os.environ["CUDA_VERSION"]
        cuda_major_version = cuda_version_environment_variable.split(".")[0]
        log(f"CUDA_VERSION: {cuda_version_environment_variable}")
        log(f"CUDA_MAJOR_VERSION: {cuda_major_version}")
        return cuda_major_version
    except:
        log("CUDA_VERSION not set, using default of 12")
        return "12"


def get_base_version() -> str:
    # Derive the version from the Version.cmake file.
    project_root = Path(__file__).parent.parent.parent.resolve()
    version_cmake = project_root / "Version.cmake"
    if not version_cmake.exists():
        raise RuntimeError(f"Version.cmake not found at {version_cmake}")
    version_str = ""
    with open(version_cmake, "r") as f:
        data = f.read().strip()
        for component in ["MAJOR", "MINOR", "PATCH"]:
            match = re.search(
                f'set\\(MLIR_TENSORRT_VERSION_{component}\\s+"(\\d+)"\\)', data
            )
            if not match:
                raise RuntimeError(f"Invalid Version.cmake file: {version_cmake}")
            version_str += match.group(1) + "."
    return version_str.rstrip(".")


def append_version_feature_flags(version: str) -> str:
    version += f"+cuda{get_cuda_version()}"
    version += f"-trt{TENSORRT_VERSION.replace('.', '')}"
    return version


def get_nightly_version() -> str:
    # For development builds, we append the date to the version.
    datestring = datetime.date.today().strftime("%Y%m%d")
    return append_version_feature_flags(f"{get_base_version()}.dev{datestring}")


def cleanup_dir(dir: Path, should_cleanup: bool, comment: str = ""):
    prefix = "Cleaning up" if should_cleanup else "Not cleaning up"
    log(f"{prefix} {comment} at {dir}")
    if should_cleanup:
        shutil.rmtree(dir, ignore_errors=True)


def copy_python_package_to_staging_dir(
    pkg_name: str, install_dir: Path, staging_dir: Path
):
    """
    Copy a python package contents from the CMake installation directory to the
    wheel build staging directory. The staging directory is created by setuptools.
    """
    # Copy installed package to build directory
    package_install_path = install_dir / "python_packages" / pkg_name
    if not package_install_path.exists():
        raise RuntimeError(f"Expected package not found at {package_install_path}")

    # Create build/lib directory if it doesn't exist
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Copy package contents to build directory
    for item in package_install_path.iterdir():
        if item.is_dir():
            shutil.copytree(item, staging_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, staging_dir / item.name)

    log(f"Package copied to {staging_dir}")


def run_cmake_build(python_package_name: str, python_wheel_staging_dir: Path):
    """
    Run CMake build and configuration steps.
    """
    # Get project root (3 levels up from this setup.py)
    project_root = Path(__file__).parent.parent.parent.resolve()

    # Environment variable overrides
    cmake_preset = os.environ.get("MLIR_TRT_CMAKE_PRESET", "python-wheel-build")
    install_prefix = os.environ.get("MLIR_TRT_INSTALL_DIR", None)
    build_dir = os.environ.get("MLIR_TRT_BUILD_DIR", None)
    parallel_jobs = os.environ.get("MLIR_TRT_PARALLEL_JOBS", str(os.cpu_count() or 1))

    # Additional CMake options from environment
    cmake_options = []
    if "MLIR_TRT_ENABLE_HLO" in os.environ:
        cmake_options.append(
            f'-DMLIR_TRT_ENABLE_HLO={os.environ["MLIR_TRT_ENABLE_HLO"]}'
        )
    if "MLIR_TRT_ENABLE_TORCH" in os.environ:
        cmake_options.append(
            f'-DMLIR_TRT_ENABLE_TORCH={os.environ["MLIR_TRT_ENABLE_TORCH"]}'
        )
    if "MLIR_TRT_ENABLE_NCCL" in os.environ:
        cmake_options.append(
            f'-DMLIR_TRT_ENABLE_NCCL={os.environ["MLIR_TRT_ENABLE_NCCL"]}'
        )
    if "MLIR_TRT_ENABLE_CUBLAS" in os.environ:
        cmake_options.append(
            f'-DMLIR_TRT_ENABLE_CUBLAS={os.environ["MLIR_TRT_ENABLE_CUBLAS"]}'
        )
    # Override TensorRT version if specified
    cmake_options.append(f"-DMLIR_TRT_DOWNLOAD_TENSORRT_VERSION={TENSORRT_VERSION}")

    # Create temporary directories for build and install
    cleanup_install = True
    cleanup_build = True

    if install_prefix:
        install_dir = Path(install_prefix).resolve()
        install_dir.mkdir(parents=True, exist_ok=True)
        cleanup_install = False
    else:
        install_dir = Path(tempfile.mkdtemp(prefix="mlir-tensorrt-install-"))
        cleanup_install = True

    if build_dir:
        build_dir = Path(build_dir).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        cleanup_build = False
    else:
        build_dir = Path(tempfile.mkdtemp(prefix="mlir-tensorrt-build-"))
        cleanup_build = True

    atexit.register(cleanup_dir, install_dir, cleanup_install, "install directory")
    atexit.register(cleanup_dir, build_dir, cleanup_build, "build directory")

    log(f"Building MLIR-TensorRT in {build_dir}")
    log(f"Installing to {install_dir}")

    # Retrieve the path to the isolated Python site where the wheel build is occuring.
    # That is where the Python build dependencies (the packages listed in
    # 'build-requires' in the pyproject.toml) are available. The actual CMake build
    # may invoke the Python interpreter (in order to e.g. generate custom MLIR files
    # or other tasks, so it needs to be aware of this environment. This is done by setting
    # the PYTHONPATH to this location when invoking the CMake process below.
    isolated_env_path = os.path.dirname(os.path.dirname(setuptools.__file__))

    # Configure with CMake
    configure_cmd = [
        "cmake",
        "--preset",
        cmake_preset,
        "--fresh",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        # Force use of the current Python interpreter.
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-B",
        str(build_dir),
        f"-S",
        str(project_root),
    ]
    configure_cmd.extend(cmake_options)

    log(f"Configure command: {' '.join(configure_cmd)}")
    subprocess.check_call(
        configure_cmd,
        cwd=project_root,
        # Make sure to set the PYTHONPATH to the isolated build environment
        # created by setuptools.
        env={
            **os.environ,
            "PYTHONPATH": isolated_env_path,
        },
    )

    # Build all targets
    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "all",
        "--parallel",
        parallel_jobs,
        "--",
        "--quiet",
    ]

    log(f"Build command: {' '.join(build_cmd)}")
    subprocess.check_call(build_cmd, cwd=project_root)

    # Install
    install_cmd = ["cmake", "--install", str(build_dir)]

    log(f"Install command: {' '.join(install_cmd)}")
    subprocess.check_call(install_cmd, cwd=project_root)

    copy_python_package_to_staging_dir(
        python_package_name, install_dir, python_wheel_staging_dir
    )
