from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    # Forces distutils to use Python, ABI, and platform tags specific to the host.
    def has_ext_modules(self):
        return True


# PKG_VERSION will be supplied by CMake configuration when this file is populated
# into the build directory. However, to make sure this file is still usable in
# the source directory, the logic below populates a default version number.
VERSION = "@PKG_VERSION@"
if "@" in VERSION:
    VERSION = "0.0.1"

setup(distclass=BinaryDistribution, version=VERSION)
