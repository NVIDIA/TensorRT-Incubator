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

# SPDX-LicenseCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np

import tripy as tp
from tests import helper
from textwrap import dedent

class TestSequential:
    def test_basic_structure(self, sequential_network):
        assert len(sequential_network) == 2

        assert isinstance(sequential_network[0], tp.Linear)
        assert  np.array_equal(cp.from_dlpack(sequential_network[0].weight),  cp.from_dlpack(sequential_network[0].weight))
        assert  np.array_equal(cp.from_dlpack(sequential_network[0].bias),  cp.from_dlpack(sequential_network[0].bias))

    def test_named_children(self, sequential_network):
        expected_names = [("0", sequential_network[0]), ("1", sequential_network[1])]
        assert list(sequential_network.named_children()) == expected_names

    # not valid test since sequential_network.named_parameters() is empty
    # def test_named_parameters(self, sequential_network):
    #     param_names = [f"{i}.{name}" for i in range(len(sequential_network)) for name, _ in sequential_network[i].named_parameters()]
    #     for name, _ in sequential_network.named_parameters(): # 
    #         assert name in param_names

    def test_forward_pass(self, sequential_network):
        input_data = tp.Tensor([1.0])
        output = sequential_network(input_data)
        assert output.shape == tp.Shape([1,2])

    def test_state_dict(self, sequential_network):
        state_dict = sequential_network.state_dict()
        param_count = sum(len(dict(m.named_parameters())) for m in sequential_network)
        assert len(state_dict) == param_count

        expected_state_dict_keys = ["0.weight", "0.bias", "1.weight", "1.bias"]
        assert list(state_dict.keys()) == expected_state_dict_keys

    def test_load_state_dict(self, sequential_network):
        new_state_dict = {
            "0.weight": tp.Parameter(tp.ones((3, 1)))
        }
        sequential_network.load_state_dict(new_state_dict)
        assert np.array_equal(cp.from_dlpack(sequential_network[0].weight),  cp.from_dlpack(new_state_dict["0.weight"]))

    def test_modify_parameters(self, sequential_network):
        new_param = tp.Parameter(tp.ones((2, 3)))
        sequential_network[1].weight = new_param
        assert sequential_network[1].weight is new_param

    def test_invalid_index_access(self, sequential_network):
        with helper.raises(ValueError, match="Key 2 not found in modules"):
            _ = sequential_network[2]

    def test_str_representation(self, sequential_network):
        expected_str = dedent(
            """\
            Sequential(
              0=
                Linear(
                 weight=shape(3, 1),
                 bias=shape(3),
                ),
              1=
                Linear(
                 weight=shape(2, 3),
                 bias=shape(2),
                ),
            )"""
        )
        assert str(sequential_network) == expected_str

class TestNestedSequential:
    def test_basic_structure(self, nested_sequential_network):
        # Check that the top-level Sequential has two layers and that one of them is a nested Sequential
        assert len(nested_sequential_network) == 2
        assert isinstance(nested_sequential_network[1], tp.Sequential)

    def test_named_children_top_level(self, nested_sequential_network):
        expected_names = [
            ("0", nested_sequential_network[0]),
            ("1", nested_sequential_network[1]),
        ]
        assert list(nested_sequential_network.named_children()) == expected_names

    def test_named_children_nested(self, nested_sequential_network):
        expected_names = [
            ("0", nested_sequential_network[1][0]),
            ("1", nested_sequential_network[1][1]),
        ]
        assert list(nested_sequential_network[1].named_children()) == expected_names

    def test_load_state_dict_nested(self, nested_sequential_network):
        # Loading state dict with parameters for both top-level and nested modules
        new_state_dict = {
            "1.1.weight": tp.Parameter(tp.ones((1, 3))),
        }
        nested_sequential_network.load_state_dict(new_state_dict)
        assert np.array_equal(cp.from_dlpack(nested_sequential_network[1][1].weight),  cp.from_dlpack(new_state_dict["1.1.weight"]))

    def test_str_representation(self, nested_sequential_network):
        expected_str = dedent(
            """\
            Sequential(
              0=
                Linear(
                 weight=shape(4, 2),
                 bias=shape(4),
                ),
              1=
                Sequential(
                  0=
                    Linear(
                     weight=shape(3, 4),
                     bias=shape(3),
                    ),
                  1=
                    Linear(
                     weight=shape(1, 3),
                     bias=shape(1),
                    ),
                ),
            )"""
        )
        assert str(nested_sequential_network) == expected_str
