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
import nvtripy as tp
import torch


def tripy_truncated_normal(x: tp.Tensor, mean: float, std: float) -> tp.Tensor:
    x = torch.from_dlpack(x)
    size = x.shape
    tmp = x.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    x.data.mul_(std).add_(mean)
    return tp.Tensor(x)


def tripy_init_mlp_xavier_weights_zero_bias(m: tp.Module) -> None:
    if isinstance(m, tp.Linear):
        weight = torch.from_dlpack(m.weight)
        torch.nn.init.xavier_uniform(weight)
        m.weight = tp.Tensor(weight)
        if getattr(m, "bias", None) is not None:
            bias = torch.from_dlpack(m.bias)
            bias.data.fill_(0.0)
            m.bias = tp.Tensor(bias)
