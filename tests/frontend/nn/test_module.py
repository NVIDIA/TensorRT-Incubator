import numpy as np


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
