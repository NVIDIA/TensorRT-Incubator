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
import glob
import os
import subprocess as sp
import tempfile

import pytest
from tests import helper

import tripy as tp


def check_wheel(virtualenv, wheel_dir):
    wheel_paths = glob.glob(os.path.join(wheel_dir, "*.whl"))
    assert len(wheel_paths) == 1, "Expected exactly one wheel file."

    wheel_path = wheel_paths[0]
    assert os.path.basename(wheel_path) == f"tripy-{tp.__version__}-py3-none-any.whl"

    virtualenv.run(
        [
            virtualenv.python,
            "-m",
            "pip",
            "install",
            wheel_path,
            # Needed to install MLIR-TRT:
            "-f",
            "https://nvidia.github.io/TensorRT-Incubator/packages.html",
            # For some reason, using the cache causes pip not to correctly install TRT
            # if this test is run multiple times in succession.
            "--no-cache-dir",
        ]
    )

    assert "tripy" in virtualenv.installed_packages()
    tripy_pkg = virtualenv.installed_packages()["tripy"]
    assert tripy_pkg.version == tp.__version__

    # Check that we only package things we actually want.
    # If tests are packaged, they'll end up in a higher-level directory.
    assert not os.path.exists(os.path.join(tripy_pkg.source_path, "tests"))

    # Lastly check we can actually import it and run a simple sanity test:
    virtualenv.run(
        [
            virtualenv.python,
            "-c",
            "import tripy as tp; x = tp.ones((5,), dtype=tp.int32); assert x.tolist() == [1] * 5",
        ]
    )
    return tripy_pkg


@pytest.mark.l1
def test_isolated_wheel_packaging_and_install(virtualenv):
    # Tests wheel packaging and installation as an external user would do it.
    with helper.raises(Exception, "returned non-zero exit status"):
        virtualenv.run([virtualenv.python, "-c", "import tripy"])

    virtualenv.install_package("build")

    with tempfile.TemporaryDirectory() as tmp:
        virtualenv.run([virtualenv.python, "-m", "build", ".", "-o", tmp], cwd=helper.ROOT_DIR)

        check_wheel(virtualenv, tmp)


@pytest.mark.l1
def test_container_wheel_packaging_and_install(virtualenv):
    # Tests wheel packaging as we do it in the development container.
    with tempfile.TemporaryDirectory() as tmp:
        sp.run(["python3", "-m", "build", ".", "-n", "-o", tmp], cwd=helper.ROOT_DIR)

        tripy_pkg = check_wheel(virtualenv, tmp)

        # Check that stubs are included
        assert os.path.exists(os.path.join(tripy_pkg.source_path, tp.__name__, "__init__.pyi"))
