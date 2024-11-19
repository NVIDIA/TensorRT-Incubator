#
# SPDX-FileCopyrightText: Copyright (c) 2024-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

import tripy as tp


@pytest.fixture(params=["compile", "eager"])
def eager_or_compiled(request):
    def wrapper(func, *args, **kwargs):
        def get_input_info(x: tp.Tensor):
            return tp.InputInfo(list(map(int, x.shape)), dtype=x.dtype)

        if request.param == "eager":
            return func(*args, **kwargs)

        assert request.param == "compile"

        compile_args = []
        for arg in args:
            # We don't want to feed DimensionSize as a dynamic input to the compiler (https://github.com/NVIDIA/TensorRT-Incubator/issues/65).
            if isinstance(arg, tp.Tensor) and not isinstance(arg, tp.DimensionSize):
                compile_args.append(get_input_info(arg))
            else:
                compile_args.append(arg)
        compile_args = tuple(compile_args)

        compile_kwargs = dict(
            (
                k,
                ((get_input_info(v) if isinstance(v, tp.Tensor) and not isinstance(v, tp.DimensionSize) else v)),
            )
            for k, v in kwargs.items()
        )

        compiled_func = tp.compile(func, args=compile_args, kwargs=compile_kwargs)

        tensor_args = tuple(x for x in args if isinstance(x, tp.Tensor) and not isinstance(x, tp.DimensionSize))

        tensor_kwargs = {
            k: v for k, v in kwargs.items() if isinstance(v, tp.Tensor) and not isinstance(v, tp.DimensionSize)
        }

        return compiled_func(*tensor_args, **tensor_kwargs)

    return wrapper
