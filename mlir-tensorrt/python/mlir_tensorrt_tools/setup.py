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


def get_cpu_torch_wheel_url(version: str):
    # Setuptools doesn't allow specifying the index URL, so this function
    # queries the pytorch package index to find a matching package given by the
    # version (regex).
    torch_cpu_url = "https://download.pytorch.org/whl/torch"
    index = requests.get(torch_cpu_url)

    version_info = sys.version_info
    py_major = version_info.major
    py_minor = version_info.minor
    # Pytorch wheels use "cpMajorMinor" for both abi and python tags.
    py_tag = f"cp{py_major}{py_minor}"
    abi_tag = py_tag
    platform_tag = distutils.util.get_platform().replace("-", "_").replace(".", "_")

    regex = "".join(
        [
            r'href="/(whl/cpu/torch-',
            version,
            f"-{py_tag}-{abi_tag}-{platform_tag}",
            r'\.whl)#sha256=.*">',
        ]
    )
    items = list(re.findall(regex, index.content.decode()))
    assert len(items) == 1, "expected single PyTorch wheel package"

    return f"https://download.pytorch.org/{items[0]}"


torch_wheel_url = get_cpu_torch_wheel_url(version=r"2.1.0.*cpu")


class BinaryDistribution(Distribution):
    # Forces distutils to use Python, ABI, and platform tags specific to the host.
    def has_ext_modules(self):
        return True


setup(
    distclass=BinaryDistribution,
    version=VERSION,
    install_requires=[
        f"torch @ {torch_wheel_url}",
        "click >= 8.1.0",
        "sh >= 2.0.0",
        "numpy",
        "matplotlib",
        "pandas",
        "nvtx",
    ],
)
