
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

import numpy as np
import cupy as cp
import pytest

import tripy as tp


class DummyNestedOp(tp.Module):
    def __init__(self, tensor):
        super().__init__()
        self.param = tp.Parameter(tensor)

    def __call__(self):
        return self.param


class DummyOp(tp.Module):
    def __init__(self, tensor):
        super().__init__()
        self.nested = DummyNestedOp(tensor)

    def __call__(self):
        return self.nested()


class Network(tp.Module):
    def __init__(self):
        super().__init__()
        self.param = tp.Parameter(tp.ones((2,), dtype=tp.float32))
        self.dummy1 = DummyOp(tp.zeros((2,), dtype=tp.float32))
        self.dummy2 = DummyOp(tp.arange(2, dtype=tp.float32))

    def __call__(self):
        return self.param + self.dummy1() + self.dummy2()


class ListNetwork(tp.Module):
    def __init__(self):
        super().__init__()
        self.params = [tp.Parameter(tp.ones((2,), dtype=tp.float32))]
        self.dummy_list = [DummyOp(tp.zeros((2,), dtype=tp.float32)), DummyOp(tp.arange(2, dtype=tp.float32))]

    def __call__(self):
        out = self.param
        for op in self.dummy_list:
            out = out + op()
        return out


class DictNetwork(tp.Module):
    def __init__(self):
        super().__init__()
        self.params = {"param": tp.Parameter(tp.ones((2,), dtype=tp.float32))}
        self.dummy_dict = {
            "op0": DummyOp(tp.zeros((2,), dtype=tp.float32)),
            "op1": DummyOp(tp.arange(2, dtype=tp.float32)),
        }

    def __call__(self):
        out = self.param
        for op_name in self.dummy_dict:
            out = out + self.dummy_dict[op_name]
        return out


class ComplexNetwork(tp.Module):
    def __init__(self):
        super().__init__()
        self.nets = {
            "dict_net": DictNetwork(),
            "list_net": ListNetwork(),
        }

    def __call__(self):
        out1 = self.nets["dict_net"]()
        out2 = self.nets["list_net"]()
        return out1 + out2


@pytest.fixture(params=[(Network, ())])
def all_network_modes(request):
    call_args = request.param[1]
    inputs = [tp.Tensor(cp.full(2, v, dtype=np.float32), device=tp.device("gpu")) for v in call_args]
    yield request.param[0](), call_args, inputs


@pytest.fixture
def network():
    yield Network()


@pytest.fixture
def list_network():
    yield ListNetwork()


@pytest.fixture
def dict_network():
    yield DictNetwork()


@pytest.fixture
def complex_network():
    yield ComplexNetwork()
