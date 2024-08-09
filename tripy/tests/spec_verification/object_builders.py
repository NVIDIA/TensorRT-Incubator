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
import tripy as tp

from typing import Union, Tuple, List, Sequence
from tripy.common import datatype

def tensor_builder(input_values, namespace):
    init = input_values.init
    if init is None:
        return tp.ones(dtype=namespace[input_values.dtype], shape=(3,2))
    elif not isinstance(init, tp.Tensor):
        assert input_values.dtype == None
        return init
    return tp.cast(init,dtype=namespace[input_values.dtype])


def dtype_builder(input_values, namespace):
    return namespace[input_values.dtype]


def tensor_list_builder(input_values, namespace):
    init = input_values.init
    if init is None:
        return [tp.ones(shape=(3,2),dtype=namespace[input_values.dtype]) for _ in range(2)]
    else:
        return [tp.cast(tens,dtype=namespace[input_values.dtype]) for tens in init]


def device_builder(input_values, namespace):
    target = input_values.init
    if target is None:
        return tp.device("gpu")
    return target


def default_builder(input_values, namespace):
    return input_values.init


find_func = {
    "tripy.Tensor": tensor_builder,
    "tripy.Shape": tensor_builder,
    "tripy.dtype": dtype_builder,
    datatype.dtype: dtype_builder,
    List[Union["tripy.Tensor"]]: tensor_list_builder,
    "tripy.device": device_builder,
}


def create_obj(param_name, input_desc, namespace):
    param_type = list(input_desc.keys())[0]
    create_obj_func = find_func.get(param_type, default_builder)
    if create_obj_func:
        namespace[param_name] = create_obj_func(input_desc[param_type], namespace)
        return namespace[param_name]
    
