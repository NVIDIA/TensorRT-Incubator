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
from tripy.frontend import Tensor


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def returns_non_tensor(a):
    return "not a tensor"


def returns_nothing(a):
    return


def accepts_nothing():
    return Tensor([1])


def returns_multiple_tensors(a, b):
    return a + b, a - b


def variadic_positional(*args):
    pass


def variadic_keyword(**kwargs):
    pass
