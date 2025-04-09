#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import nvtripy as tp
import pytest
import torch
from nvtripy.common.datatype import DATA_TYPES

skip_if_older_than_sm89 = pytest.mark.skipif(
    torch.cuda.get_device_capability() < (8, 9), reason="Some features (e.g. float8) are not available before SM90"
)

skip_if_older_than_sm80 = pytest.mark.skipif(
    torch.cuda.get_device_capability() < (8, 0), reason="Some features (e.g. bfloat16) are not available before SM80"
)

DATA_TYPE_TEST_CASES = [
    dtype if dtype not in [tp.float8] else pytest.param(tp.float8, marks=skip_if_older_than_sm89)
    for dtype in DATA_TYPES.values()
]


@pytest.fixture()
def tripy_virtualenv(virtualenv):
    """
    A virtual environment that inherits the PYTHONPATH from the host.
    """
    virtualenv.env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    # The tensorrt_bindings package doesn't install correctly if there are cache entries.
    virtualenv.run("pip cache remove tensorrt*")
    return virtualenv
