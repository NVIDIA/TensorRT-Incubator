import importlib
import os
from pathlib import Path

from setuptools import setup
from setuptools.dist import Distribution


def load_setup_utils():
    spec = importlib.util.spec_from_file_location(
        "setup_utils",
        Path(os.path.dirname(__file__)).parent / "setup_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup_utils = load_setup_utils()

VERSION = setup_utils.get_wheel_version()


class BinaryDistribution(Distribution):
    # Forces distutils to use Python, ABI, and platform tags specific to the host.
    def has_ext_modules(self):
        return True


setup(
    distclass=BinaryDistribution,
    version=VERSION,
    install_requires=[
        "click >= 8.1.0",
        "sh >= 2.0.0",
        "numpy",
        "matplotlib",
        "pandas",
        "nvtx",
    ],
)
