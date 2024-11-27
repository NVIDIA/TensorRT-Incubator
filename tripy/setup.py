#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import subprocess

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.discovery import FlatLayoutPackageFinder


class BuildPyCommand(build_py):
    def run(self):
        # Run the standard build_py.
        build_py.run(self)

        # Generate stubs
        self.generate_stubs()

    def generate_stubs(self):
        package_name = self.distribution.metadata.name
        try:
            subprocess.run(
                [
                    "stubgen",
                    "-m",
                    package_name,
                    "--include-docstrings",
                    "--inspect-mode",
                    "--export-less",
                    "-o",
                    self.build_lib,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error generating stubs: {e}")


default_excludes = list(FlatLayoutPackageFinder.DEFAULT_EXCLUDE) + ["notebooks"]

if __name__ == "__main__":
    setup(
        cmdclass={"build_py": BuildPyCommand},
        packages=find_packages(exclude=default_excludes),  # Use the dynamically updated exclude list
    )
