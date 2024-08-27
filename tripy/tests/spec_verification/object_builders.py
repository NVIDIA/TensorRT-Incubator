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

import tripy as tp
import math


def tensor_builder(func_obj, input_values, namespace):
    shape = input_values.get("shape", None)
    if not shape:
        shape = (3, 2)
    return tp.ones(dtype=namespace[input_values["dtype"]], shape=shape)


def shape_tensor_builder(func_obj, input_values, namespace):
    follow_tensor = input_values.get("follow_tensor", None)
    return (math.prod((namespace[follow_tensor]).shape.tolist()),)


def dtype_builder(func_obj, input_values, namespace):
    dtype = input_values.get("dtype", None)
    return namespace[dtype]


def int_builder(func_obj, input_values, namespace):
    return input_values.get("value", None)


find_func = {
    "Tensor": tensor_builder,
    "shape_tensor": shape_tensor_builder,
    "dtype": dtype_builder,
    "int": int_builder,
}


def create_obj(func_obj, param_name, input_desc, namespace):
    param_type = list(input_desc.keys())[0]
    create_obj_func = find_func[param_type]
    namespace[param_name] = create_obj_func(func_obj, input_desc[param_type], namespace)
    return namespace[param_name]
