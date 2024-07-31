#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import glob
import os
import subprocess as sp
from typing import Optional

import pytest
import torch

import tripy as tp
from tests.helper import ROOT_DIR
from tripy.common.datatype import DATA_TYPES

skip_if_older_than_sm89 = pytest.mark.skipif(
    torch.cuda.get_device_capability() < (8, 9), reason="Some features (e.g. fp8) are not available before SM90"
)

skip_if_older_than_sm80 = pytest.mark.skipif(
    torch.cuda.get_device_capability() < (8, 0), reason="Some features (e.g. bfloat16) are not available before SM80"
)

DATA_TYPE_TEST_CASES = [
    dtype if dtype not in [tp.float8] else pytest.param(tp.float8, marks=skip_if_older_than_sm89)
    for dtype in DATA_TYPES.values()
]


@pytest.fixture()
def sandboxed_install_run(virtualenv):
    """
    A special fixture that runs commands, but sandboxes any `pip install`s in a virtual environment.
    Packages from the test environment are still usable, but those in the virtual environment take precedence
    """

    VENV_PYTHONPATH = glob.glob(os.path.join(virtualenv.virtualenv, "lib", "python*", "site-packages"))[0]

    class StatusWrapper:
        def __init__(self, stdout=None, stderr=None, success=None) -> None:
            self.stdout = stdout
            self.stderr = stderr
            self.success = success

    def run_impl(command: str, cwd: Optional[str] = None):
        env = copy.copy(os.environ)
        # Always prioritize our own copy of TriPy over anything in the venv.
        env["PYTHONPATH"] = ROOT_DIR + os.pathsep + VENV_PYTHONPATH

        print(f"Running command: {command}")

        status = StatusWrapper()
        if "pip" in command:
            virtualenv.run(command, cwd=cwd)
            status.success = True
        else:
            sp_status = sp.run(command, cwd=cwd, env=env, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)

            def try_decode(inp):
                try:
                    return inp.decode()
                except UnicodeDecodeError:
                    return inp

            status.stdout = try_decode(sp_status.stdout)
            status.stderr = try_decode(sp_status.stderr)
            status.success = sp_status.returncode == 0

        return status

    return run_impl
