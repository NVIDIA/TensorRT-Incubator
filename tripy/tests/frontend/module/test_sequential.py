#
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
#

# SPDX-LicenseCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest

import nvtripy as tp
from tests import helper
from textwrap import dedent


def get_linear_layers():
    linear0 = tp.Linear(1, 3)
    linear0.weight = tp.ones(linear0.weight.shape)
    linear0.bias = tp.ones(linear0.bias.shape)

    linear1 = tp.Linear(3, 2)
    linear1.weight = tp.ones(linear1.weight.shape)
    linear1.bias = tp.ones(linear1.bias.shape)

    return [linear0, linear1]


@pytest.fixture
def sequential_network():
    yield tp.Sequential(*get_linear_layers())


@pytest.fixture
def dict_sequential_network():
    linear0, linear1 = get_linear_layers()
    yield tp.Sequential({"layer1": linear0, "layer2": linear1})


@pytest.fixture
def mixed_container_sequential_network():
    conv = tp.Conv(in_channels=2, out_channels=2, kernel_dims=(1, 1), stride=(1, 1))
    conv.weight = tp.ones(conv.weight.shape)
    conv.bias = tp.ones(conv.bias.shape)

    linear = tp.Linear(2, 1)
    linear.weight = tp.ones(linear.weight.shape)
    linear.bias = tp.ones(linear.bias.shape)

    yield tp.Sequential(
        conv,
        lambda x: tp.avgpool(x, kernel_dims=(2, 2), stride=(1, 1)),
        lambda x: tp.flatten(x, start_dim=1),
        linear,
    )


@pytest.fixture
def nested_sequential_network():
    linear0 = tp.Linear(2, 1)
    linear0.weight = tp.ones(linear0.weight.shape)
    linear0.bias = tp.ones(linear0.bias.shape)

    linear1, linear2 = get_linear_layers()
    yield tp.Sequential(linear0, tp.Sequential(linear1, linear2))


class TestSequential:
    def test_basic_structure(self, sequential_network):
        assert len(sequential_network) == 2

        assert isinstance(sequential_network[0], tp.Linear)

    def test_named_children(self, sequential_network):
        expected_names = [("0", sequential_network[0]), ("1", sequential_network[1])]
        assert list(sequential_network.named_children()) == expected_names

    def test_forward_pass(self, sequential_network):
        input_data = tp.Tensor([1.0])
        output = sequential_network(input_data)
        assert output.shape == (1, 2)

    def test_state_dict(self, sequential_network):
        state_dict = sequential_network.state_dict()
        param_count = sum(len(dict(m.named_parameters())) for m in sequential_network)
        assert len(state_dict) == param_count

        expected_state_dict_keys = ["0.weight", "0.bias", "1.weight", "1.bias"]
        assert list(state_dict.keys()) == expected_state_dict_keys

    def test_load_state_dict(self, sequential_network):
        new_state_dict = {"0.weight": tp.ones((3, 1))}
        sequential_network.load_state_dict(new_state_dict, strict=False)
        assert tp.equal(sequential_network[0].weight, new_state_dict["0.weight"])

    def test_modify_parameters(self, sequential_network):
        new_param = tp.ones((2, 3))
        sequential_network[1].weight = new_param
        assert sequential_network[1].weight is new_param

    def test_invalid_index_access(self, sequential_network):
        with helper.raises(tp.TripyException, match="Key: '2' not found in modules"):
            _ = sequential_network[2]

    def test_str_representation(self, sequential_network):
        expected_str = dedent(
            """
            Sequential(
                0: Module = Linear(
                    weight: Parameter = (shape=(3, 1), dtype=float32),
                    bias: Parameter = (shape=(3,), dtype=float32),
                ),
                1: Module = Linear(
                    weight: Parameter = (shape=(2, 3), dtype=float32),
                    bias: Parameter = (shape=(2,), dtype=float32),
                ),
            )
            """
        ).strip()

        assert str(sequential_network) == expected_str


class TestDictSequential:
    def test_basic_structure(self, dict_sequential_network):
        assert len(dict_sequential_network) == 2
        assert isinstance(dict_sequential_network["layer1"], tp.Linear)
        assert isinstance(dict_sequential_network["layer2"], tp.Linear)

    def test_named_children(self, dict_sequential_network):
        expected_names = [("layer1", dict_sequential_network["layer1"]), ("layer2", dict_sequential_network["layer2"])]
        assert list(dict_sequential_network.named_children()) == expected_names

    def test_forward_pass(self, dict_sequential_network):
        input_data = tp.Tensor([[1.0]])
        output = dict_sequential_network(input_data)
        assert output.shape == (1, 2)

    def test_state_dict(self, dict_sequential_network):
        state_dict = dict_sequential_network.state_dict()
        expected_keys = ["layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"]
        assert list(state_dict.keys()) == expected_keys

    def test_load_state_dict(self, dict_sequential_network):
        new_state_dict = {"layer1.weight": tp.ones((3, 1))}
        dict_sequential_network.load_state_dict(new_state_dict, strict=False)
        assert tp.equal(dict_sequential_network["layer1"].weight, new_state_dict["layer1.weight"])

    def test_modify_parameters(self, dict_sequential_network):
        new_weight = tp.ones((2, 3))
        dict_sequential_network["layer2"].weight = new_weight
        assert dict_sequential_network["layer2"].weight is new_weight

    def test_str_representation(self, dict_sequential_network):
        expected_str = dedent(
            """
            Sequential(
                layer1: Module = Linear(
                    weight: Parameter = (shape=(3, 1), dtype=float32),
                    bias: Parameter = (shape=(3,), dtype=float32),
                ),
                layer2: Module = Linear(
                    weight: Parameter = (shape=(2, 3), dtype=float32),
                    bias: Parameter = (shape=(2,), dtype=float32),
                ),
            )
            """
        ).strip()
        assert str(dict_sequential_network) == expected_str


class TestMixedContainerSequential:
    def test_basic_structure(self, mixed_container_sequential_network):
        assert len(mixed_container_sequential_network) == 4
        assert isinstance(mixed_container_sequential_network[0], tp.Module)
        assert callable(mixed_container_sequential_network[1])
        assert callable(mixed_container_sequential_network[2])
        assert isinstance(mixed_container_sequential_network[3], tp.Module)

    def test_forward_pass(self, mixed_container_sequential_network):
        input_data = tp.ones((1, 2, 2, 2), dtype=tp.float32)
        output = mixed_container_sequential_network(input_data)
        assert output.shape == (1, 1)

    def test_named_children(self, mixed_container_sequential_network):
        expected_names = [("0", mixed_container_sequential_network[0]), ("3", mixed_container_sequential_network[3])]
        assert list(mixed_container_sequential_network.named_children()) == expected_names

    def test_state_dict(self, mixed_container_sequential_network):
        state_dict = mixed_container_sequential_network.state_dict()
        expected_keys = set(["0.bias", "0.weight", "3.weight", "3.bias"])
        assert set(state_dict.keys()) == expected_keys

    def test_load_state_dict(self, mixed_container_sequential_network):
        new_state_dict = {
            "0.weight": tp.ones((2, 2, 1, 1), dtype=tp.float32),
            "0.bias": tp.zeros((2,), dtype=tp.float32),
            "3.weight": tp.zeros((1, 2), dtype=tp.float32),
            "3.bias": tp.zeros((1,), dtype=tp.float32),
        }
        mixed_container_sequential_network.load_state_dict(new_state_dict, strict=False)

        assert np.array_equal(
            cp.from_dlpack(mixed_container_sequential_network[0].weight), cp.from_dlpack(new_state_dict["0.weight"])
        )
        assert np.array_equal(
            cp.from_dlpack(mixed_container_sequential_network[0].bias), cp.from_dlpack(new_state_dict["0.bias"])
        )
        assert np.array_equal(
            cp.from_dlpack(mixed_container_sequential_network[3].weight), cp.from_dlpack(new_state_dict["3.weight"])
        )
        assert np.array_equal(
            cp.from_dlpack(mixed_container_sequential_network[3].bias), cp.from_dlpack(new_state_dict["3.bias"])
        )

    def test_str_representation(self, mixed_container_sequential_network):
        expected_str = dedent(
            """\
            Sequential(
                0: Module = Conv(
                    bias: Parameter = (shape=(2,), dtype=float32),
                    weight: Parameter = (shape=(2, 2, 1, 1), dtype=float32),
                ),
                3: Module = Linear(
                    weight: Parameter = (shape=(1, 2), dtype=float32),
                    bias: Parameter = (shape=(1,), dtype=float32),
                ),
            )"""
        )
        assert str(mixed_container_sequential_network) == expected_str


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
            "1.1.weight": tp.ones((2, 3)),
        }
        nested_sequential_network.load_state_dict(new_state_dict, strict=False)
        assert tp.equal(nested_sequential_network[1][1].weight, new_state_dict["1.1.weight"])

    def test_str_representation(self, nested_sequential_network):
        expected_str = dedent(
            """
            Sequential(
                0: Module = Linear(
                    weight: Parameter = (shape=(1, 2), dtype=float32),
                    bias: Parameter = (shape=(1,), dtype=float32),
                ),
                1: Module = Sequential(
                    0: Module = Linear(
                        weight: Parameter = (shape=(3, 1), dtype=float32),
                        bias: Parameter = (shape=(3,), dtype=float32),
                    ),
                    1: Module = Linear(
                        weight: Parameter = (shape=(2, 3), dtype=float32),
                        bias: Parameter = (shape=(2,), dtype=float32),
                    ),
                ),
            )
            """
        ).strip()
        assert str(nested_sequential_network) == expected_str
