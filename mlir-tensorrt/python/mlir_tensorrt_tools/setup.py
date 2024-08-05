import distutils.util
import re
import sys

import requests
from setuptools import setup
from setuptools.dist import Distribution

# PKG_VERSION will be supplied by CMake configuration when this file is populated
# into the build directory. However, to make sure this file is still usable in
# the source directory, the logic below populates a default version number.
VERSION = "@PKG_VERSION@"
if "@" in VERSION:
    VERSION = "0.0.1"


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
