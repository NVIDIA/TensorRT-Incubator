import toml
import os
import subprocess

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

import tripy

# Read the pyproject.toml file from the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))
pyproject_path = os.path.join(current_directory, "pyproject.toml")

with open(pyproject_path, "r") as f:
    pyproject = toml.load(f)

project = pyproject.get("project", {})
dependencies = project.get("dependencies", [])


class BuildPyCommand(build_py):
    def run(self):
        # Run the standard build_py.
        build_py.run(self)

        # Generate stubs
        self.generate_stubs()

    def generate_stubs(self):
        package_name = self.distribution.metadata.name
        output_dir = os.path.join(self.build_lib, package_name)
        try:
            subprocess.run(
                ["stubgen", "-m", package_name, "--include-docstrings", "--inspect-mode", "-o", output_dir], check=True
            )
            with open(os.path.join(output_dir, "__init__.pyi"), "w") as f:
                pass
        except subprocess.CalledProcessError as e:
            print(f"Error generating stubs: {e}")


if __name__ == "__main__":
    setup(
        name="tripy",
        version=tripy.__version__,
        description="Tripy: Python programming model for TensorRT",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        url="https://github.com/NVIDIA/tensorrt-incubator/tripy/",
        author="NVIDIA",
        author_email="svc_tensorrt@nvidia.com",
        classifiers=[
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
        ],
        install_requires=dependencies,
        cmdclass={
            "build_py": BuildPyCommand,
        },
        license="Apache 2.0",
        packages=find_packages(exclude=("tests", "tests.*")),
        package_data={
            "tripy": ["py.typed", "*.pyi"],
        },
        zip_safe=True,
        python_requires=">=3.8",
    )
