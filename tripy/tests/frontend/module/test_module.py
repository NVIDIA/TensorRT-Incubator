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

import cupy as cp
import numpy as np

import tripy as tp
from tests import helper
from textwrap import dedent


class TestModule:
    def test_basic(self, all_network_modes):
        test_net, call_args, inputs = all_network_modes
        assert len(dict(test_net.named_parameters())) == 1
        assert len(dict(test_net.named_children())) == 2

        result = np.array([1.0, 2.0]) + np.full(2, sum(call_args), dtype=np.float32)
        assert np.array_equal(cp.from_dlpack(test_net(*inputs)).get(), result)

    def test_get_set_attr(self, network):
        network.new_attr = True
        assert hasattr(network, "new_attr")

        network.new_param = tp.Parameter(0.0)
        assert "new_param" in dict(network.named_parameters())

        network.param = tp.Parameter([0.0, 1.0])
        network.dummy1 = None
        assert cp.from_dlpack(dict(network.named_parameters())["param"]).get().tolist() == [0.0, 1.0]
        assert "dummy1" not in dict(network.named_children())

    def test_incompatible_parameter_cannot_be_set(self, network):
        with helper.raises(
            tp.TripyException, match="New parameter shape: \[2, 3\] is not compatible with current shape: \[2\]"
        ):
            network.param = tp.Parameter(tp.ones((2, 3)))

    def test_named_children(self, network):
        # Children should only return immediate children
        assert list(network.named_children()) == [("dummy1", network.dummy1), ("dummy2", network.dummy2)]
        assert list(network.dummy1.named_children()) == [("nested", network.dummy1.nested)]

    def test_state_dict(self, network):
        # state_dict should recurse through all children
        assert network.state_dict() == {
            "param": network.param,
            "dummy1.nested.param": network.dummy1.nested.param,
            "dummy2.nested.param": network.dummy2.nested.param,
        }

    def test_load_from_state_dict_top_level_param(
        self,
        network,
    ):
        state_dict = {"param": tp.Parameter(tp.zeros((2,), dtype=tp.float32))}
        network.load_from_state_dict(state_dict)
        assert network.param is state_dict["param"]

    def test_load_from_state_dict_nested_param(
        self,
        network,
    ):
        state_dict = {"dummy1.nested.param": tp.Parameter(tp.arange(2, dtype=tp.float32))}
        network.load_from_state_dict(state_dict)
        assert network.dummy1.nested.param is state_dict["dummy1.nested.param"]

    def test_load_from_state_dict_with_different_shapes_fails(
        self,
        network,
    ):
        state_dict = {"param": tp.Parameter(tp.zeros((3,), dtype=tp.float32))}

        with helper.raises(
            tp.TripyException, match=r"New parameter shape: \[3\] is not compatible with current shape: \[2\]"
        ):
            network.load_from_state_dict(state_dict)

    def test_load_from_state_dict_with_different_dtype_fails(
        self,
        network,
    ):
        state_dict = {"param": tp.Parameter(tp.ones((2,), dtype=tp.float16))}

        with helper.raises(
            tp.TripyException, match="New parameter dtype: float16 is not compatible with current dtype: float32"
        ):
            network.load_from_state_dict(state_dict)

    def test_mixed_collections_not_registered(self, network):
        network.mix_param_list = [True, tp.Parameter(1)]
        network.mix_param_dict = {0: True, "a": tp.Parameter(1)}
        network.mix_module_list = [True, tp.Module()]
        network.mix_module_dict = {0: True, "a": tp.Module()}

        assert "mix_params_list" not in dict(network.named_parameters())
        assert "mix_params_dict" not in dict(network.named_parameters())
        assert "mix_modules_list" not in dict(network.named_children())
        assert "mix_modules_dict" not in dict(network.named_children())

    def test_module_print(self, network):
        expected_output = dedent(
            """\
            Network(
              dummy1=
                DummyOp(
                  nested=
                    DummyNestedOp(
                     param=shape(2),
                    ),
                ),
              dummy2=
                DummyOp(
                  nested=
                    DummyNestedOp(
                     param=shape(2),
                    ),
                ),
             param=shape(2),
            )"""
        )


class TestModuleWithList:
    def test_named_children(self, list_network):
        # Children should only return immediate children
        assert list(list_network.named_children()) == [
            ("dummy_list.0", list_network.dummy_list[0]),
            ("dummy_list.1", list_network.dummy_list[1]),
        ]
        assert list(list_network.dummy_list[0].named_children()) == [("nested", list_network.dummy_list[0].nested)]
        assert list(list_network.dummy_list[1].named_children()) == [("nested", list_network.dummy_list[1].nested)]

    def test_state_dict(self, list_network):
        # state_dict should recurse through all children
        assert list_network.state_dict() == {
            "params.0": list_network.params[0],
            "dummy_list.0.nested.param": list_network.dummy_list[0].nested.param,
            "dummy_list.1.nested.param": list_network.dummy_list[1].nested.param,
        }

    def test_load_from_state_dict_top_level_param(
        self,
        list_network,
    ):
        state_dict = {"params.0": tp.Parameter(tp.zeros((2,), dtype=tp.float32))}
        list_network.load_from_state_dict(state_dict)
        assert list_network.params[0] is state_dict["params.0"]

    def test_load_from_state_dict_nested_param(
        self,
        list_network,
    ):
        state_dict = {"dummy_list.0.nested.param": tp.Parameter(tp.arange(2, dtype=tp.float32))}
        list_network.load_from_state_dict(state_dict)
        assert list_network.dummy_list[0].nested.param is state_dict["dummy_list.0.nested.param"]

    def test_modify_list_param(self, list_network):
        new_param = tp.Parameter(5.0)
        list_network.params[0] = new_param
        assert dict(list_network.named_parameters())["params.0"] is new_param

    def test_add_list_params(self, list_network):
        param0 = tp.Parameter(0.0)
        param1 = tp.Parameter(1.0)
        list_network.new_params = [param0, param1]
        tripy_params = dict(list_network.named_parameters())
        assert tripy_params["new_params.0"] is param0
        assert tripy_params["new_params.1"] is param1


class TestModuleWithDict:
    def test_named_children(self, dict_network):
        # Children should only return immediate children
        assert list(dict_network.named_children()) == [
            ("dummy_dict.op0", dict_network.dummy_dict["op0"]),
            ("dummy_dict.op1", dict_network.dummy_dict["op1"]),
        ]
        assert list(dict_network.dummy_dict["op0"].named_children()) == [
            ("nested", dict_network.dummy_dict["op0"].nested)
        ]
        assert list(dict_network.dummy_dict["op1"].named_children()) == [
            ("nested", dict_network.dummy_dict["op1"].nested)
        ]

    def test_state_dict(self, dict_network):
        # state_dict should recurse through all children
        assert dict_network.state_dict() == {
            "params.param": dict_network.params["param"],
            "dummy_dict.op0.nested.param": dict_network.dummy_dict["op0"].nested.param,
            "dummy_dict.op1.nested.param": dict_network.dummy_dict["op1"].nested.param,
        }

    def test_load_from_state_dict_top_level_param(
        self,
        dict_network,
    ):
        state_dict = {"params.param": tp.Parameter(tp.zeros((2,), dtype=tp.float32))}
        dict_network.load_from_state_dict(state_dict)
        assert dict_network.params["param"] is state_dict["params.param"]

    def test_load_from_state_dict_nested_param(
        self,
        dict_network,
    ):
        state_dict = {"dummy_dict.op0.nested.param": tp.Parameter(tp.arange(2, dtype=tp.float32))}
        dict_network.load_from_state_dict(state_dict)
        assert dict_network.dummy_dict["op0"].nested.param is state_dict["dummy_dict.op0.nested.param"]

    def test_modify_dict_param(self, dict_network):
        new_param = tp.Parameter(5.0)
        dict_network.params["param"] = new_param
        assert dict(dict_network.named_parameters())["params.param"] is new_param

    def test_add_dict_params(self, dict_network):
        param0 = tp.Parameter(0.0)
        param1 = tp.Parameter(1.0)
        dict_network.new_params = {"param0": param0, "param1": param1}
        tripy_params = dict(dict_network.named_parameters())
        assert tripy_params["new_params.param0"] is param0
        assert tripy_params["new_params.param1"] is param1


class TestComplexModule:
    def test_basic_structure(self, complex_network):
        module = complex_network
        assert hasattr(module, "nets")
        assert isinstance(module.nets, dict)
        assert isinstance(module.nets["dict_net"], tp.Module)
        assert isinstance(module.nets["list_net"], tp.Module)

    def test_state_dict(self, complex_network):
        module = complex_network
        tensor = tp.ones((2,))
        external_state_dict = {
            "nets.dict_net.params.param": tensor,
            "nets.dict_net.dummy_dict.op0.nested.param": tensor,
            "nets.list_net.params.0": tensor,
            "nets.list_net.dummy_list.0.nested.param": tensor,
        }
        module.load_from_state_dict(external_state_dict)
        assert module.nets["dict_net"].params["param"] is tensor
        assert module.nets["list_net"].params[0] is tensor
        assert module.nets["dict_net"].dummy_dict["op0"].nested.param is tensor
        assert module.nets["list_net"].dummy_list[0].nested.param is tensor
