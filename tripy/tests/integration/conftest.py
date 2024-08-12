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

import pytest

import tripy as tp


@pytest.fixture(params=["compile", "eager"])
def mode(request):
    return request.param


@pytest.fixture
def compile_fixture(mode):
    def wrapper(func, *args, **kwargs):
        def get_shape(x: tp.Tensor):
            x.eval()
            return tp.InputInfo(x.trace_tensor.shape, dtype=x.dtype)

        if mode == "compile":
            compiler = tp.Compiler(func)
            # Cast appropriate args / kwargs to use `tp.InputInfo`
            compile_args = tuple(map(lambda x: get_shape(x) if isinstance(x, tp.Tensor) else x, list(args)))
            compile_kwargs = dict((k, get_shape(v) if isinstance(v, tp.Tensor) else v) for k, v in kwargs.items())
            compiled_func = compiler.compile(*compile_args, **compile_kwargs)
            # Remove baked in args, aka, only keep tp.Tensor's
            args = tuple(filter(lambda x: isinstance(x, tp.Tensor), args))
            kwargs = dict(filter(lambda kv: isinstance(kv[1], tp.Tensor), kwargs.items()))
            return compiled_func(*args, **kwargs)
        elif mode == "eager":
            return func(*args, **kwargs)

    return wrapper
