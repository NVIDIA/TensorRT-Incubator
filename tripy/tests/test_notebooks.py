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

import pytest
from tests import paths

NOTEBOOKS_ROOT = os.path.join(paths.ROOT_DIR, "notebooks")

NOTEBOOKS = glob.glob(os.path.join(NOTEBOOKS_ROOT, "*.ipynb"))
# Paranoid check:
assert os.path.join(NOTEBOOKS_ROOT, "resnet50.ipynb") in NOTEBOOKS


# Like our example, we want to test notebooks with both the release package and TOT.
@pytest.mark.l1
@pytest.mark.l1_release_package
@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebooks(notebook_path, tripy_virtualenv):
    # Install the minimum packages needed to execute notebook tests.
    # The notebooks should install everything else themselves.
    for package in ["notebook==7.2.2", "pytest==7.1.3", "pytest-notebook==0.10.0"]:
        tripy_virtualenv.install_package(package)
    tripy_virtualenv.run(f"python3 -m pytest --nb-test-files {notebook_path}")
