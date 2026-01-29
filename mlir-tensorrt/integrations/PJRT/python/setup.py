#!/usr/bin/env python3
from setuptools import Distribution, setup, find_namespace_packages
import importlib
import os
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution


def load_setup_utils():
    spec = importlib.util.spec_from_file_location(
        "setup_utils",
        Path(os.path.dirname(__file__)).parent.parent / "python" / "setup_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup_utils = load_setup_utils()
log = setup_utils.log


class BinaryDistribution(Distribution):
    # Forces distutils to use Python, ABI, and platform tags specific to the host.
    def has_ext_modules(self):
        return True


USE_PYPI_VERSION = str(os.environ.get("MLIR_TRT_PYPI", "")).lower() == "1"
if USE_PYPI_VERSION:
    PKG_VERSION = setup_utils.get_pypi_version()
    log(f"Building for PyPI upload. Using package version: {PKG_VERSION}")
else:
    PKG_VERSION = setup_utils.get_nightly_version()
    log(f"Building nightly/development wheel. Using package version: {PKG_VERSION}")


class CMakeBuild(build_py):
    """Custom build command that invokes CMake to build the C++ components."""

    def run(self):
        setup_utils.run_cmake_build(
            "mlir_tensorrt_jax",
            Path(self.build_lib),
        )


class CMakeBuildExt(build_ext):
    """Custom build_ext that does nothing since we handle everything in build_py."""

    def run(self):
        # build_py handles the CMake build, so we don't need to do anything here
        pass


def main():
    setup(
        version=PKG_VERSION,
        zip_safe=False,
        distclass=BinaryDistribution,
        cmdclass={
            "build_py": CMakeBuild,
            "build_ext": CMakeBuildExt,
        },
    )


if __name__ == "__main__":
    main()
