import numpy as np

import tripy as tp


class TestModule:
    def test_basic(self, all_network_modes):
        test_net, call_args, inputs = all_network_modes
        assert len(test_net._params.keys()) == 1
        assert len(test_net._modules.keys()) == 2

        result = np.array([1.0, 2.0]) + np.full(2, sum(call_args), dtype=np.float32)
        assert np.array_equal(test_net(*inputs).numpy(), result)

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
        state_dict = {"param": tp.nn.Parameter(tp.zeros(2, dtype=tp.float32))}
        network.load_from_state_dict(state_dict)
        assert network.param is state_dict["param"]

    def test_load_from_state_dict_nested_param(
        self,
        network,
    ):
        state_dict = {"dummy1.nested.param": tp.nn.Parameter(tp.arange(2, dtype=tp.float32))}
        network.load_from_state_dict(state_dict)
        assert network.dummy1.nested.param is state_dict["dummy1.nested.param"]


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
        state_dict = {"params.0": tp.nn.Parameter(tp.zeros(2, dtype=tp.float32))}
        list_network.load_from_state_dict(state_dict)
        assert list_network.params[0] is state_dict["params.0"]

    def test_load_from_state_dict_nested_param(
        self,
        list_network,
    ):
        state_dict = {"dummy_list.0.nested.param": tp.nn.Parameter(tp.arange(2, dtype=tp.float32))}
        list_network.load_from_state_dict(state_dict)
        assert list_network.dummy_list[0].nested.param is state_dict["dummy_list.0.nested.param"]


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
        state_dict = {"params.param": tp.nn.Parameter(tp.zeros(2, dtype=tp.float32))}
        dict_network.load_from_state_dict(state_dict)
        assert dict_network.params["param"] is state_dict["params.param"]

    def test_load_from_state_dict_nested_param(
        self,
        dict_network,
    ):
        state_dict = {"dummy_dict.op0.nested.param": tp.nn.Parameter(tp.arange(2, dtype=tp.float32))}
        dict_network.load_from_state_dict(state_dict)
        assert dict_network.dummy_dict["op0"].nested.param is state_dict["dummy_dict.op0.nested.param"]


class TestComplexModule:
    def test_basic_structure(self, complex_network):
        module = complex_network
        assert hasattr(module, "nets")
        assert isinstance(module.nets, dict)
        assert isinstance(module.nets["dict_net"], tp.nn.Module)
        assert isinstance(module.nets["list_net"], tp.nn.Module)

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
