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

import torch

import tripy as tp


class TestPooling:
    pass


from tripy.logging import logger

logger.verbosity = "ir"

tp_pool = tp.MaxPool((2, 2), (1, 1), ((1, 1), (1, 1)))

inp_torch = torch.arange(9, dtype=torch.float32).reshape((1, 1, 3, 3))
inp = tp.Tensor(inp_torch)
out = tp_pool(inp)
print(out)
