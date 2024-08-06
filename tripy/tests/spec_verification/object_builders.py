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

import numbers
import numbers
import tripy as tp

from typing import Union, Any, Tuple, List, Sequence, Optional
from tripy.common import datatype

def tensor_builder(input_values, namespace):
    init = input_values.init
    if init:
        # Have to eval for "init" to force tensor to be constant which is currently an issue for quantize.
        temp = tp.Tensor(init, dtype=namespace[input_values.dtype])
        temp.eval()
        return temp
    shape = input_values.shape
    if not shape:
        shape = (3,2)
    return tp.ones(dtype=namespace[input_values.dtype], shape=shape)


def dtype_builder(input_values, namespace):
    return namespace[input_values.dtype]


def tensor_list_builder(input_values, namespace):
    count = input_values.count
    if not count:
        count = 2
    return [tensor_builder(input_values, namespace) for _ in range(count)]


def device_builder(input_values, namespace):
    target = input_values.target
    if not target:
        target = "gpu"
    return tp.device(target)


def default_builder(input_values, namespace):
    return input_values.init


find_func = {
    "tripy.Tensor": tensor_builder,
    "tripy.Shape": tensor_builder,
    Sequence[int]: default_builder,
    numbers.Number: default_builder,
    int: default_builder,
    "tripy.dtype": dtype_builder,
    datatype.dtype: dtype_builder,
    Tuple: default_builder,
    List[Union["tripy.Tensor"]]: tensor_list_builder,
    "tripy.device": device_builder,
    bool: default_builder,
    float: default_builder,
    "tripy.Tensor": tensor_builder,
    "tripy.Shape": tensor_builder,
    Sequence[int]: default_builder,
    numbers.Number: default_builder,
    int: default_builder,
    "tripy.dtype": dtype_builder,
    datatype.dtype: dtype_builder,
    Tuple: default_builder,
    List[Union["tripy.Tensor"]]: tensor_list_builder,
    "tripy.device": device_builder,
    bool: default_builder,
    float: default_builder,
}


def create_obj(param_name, input_desc, namespace):
def create_obj(param_name, input_desc, namespace):
    param_type = list(input_desc.keys())[0]
    create_obj_func = find_func.get(param_type, None)
    if create_obj_func:
        namespace[param_name] = create_obj_func(input_desc[param_type], namespace)
        return namespace[param_name]
    return None
    create_obj_func = find_func.get(param_type, None)
    if create_obj_func:
        namespace[param_name] = create_obj_func(input_desc[param_type], namespace)
        return namespace[param_name]
    return None
