from typing import Any, Dict

from tripy.frontend.nn.parameter import Parameter


class Module:
    """
    Base class used to create all neural network modules.
    Module class allows accessing all the parameters associated within nested Modules.

    The implementation currently assumes that Parameters are associated with the Module
    as an attribute and are not part of nested List or Dict.

    Specifically, this is allowed:

    ``self.param = Parameter(Tensor([1,2,3]))``

    Whereas this is not currently supported:

    ``self.param = {"param1": Parameter(Tensor([1,2,3]))}``

    Example:
    ::

        class AddBias(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = tp.nn.Parameter(tp.Tensor([1.0, 1.0], dtype=tp.float32))

            def __call__(self, x):
                return x + self.bias

        add_bias = AddBias()

        print(f"bias: {add_bias.bias}")

        inp = tp.Tensor([1.0, 1.0], dtype=tp.float32)
        out = add_bias(inp)

        print(f"out: {out}")
        assert np.array_equal(out.numpy(), np.array([2.0, 2.0]))
    """

    def __init__(self):
        self._params: Dict[str, Parameter] = {}
        self._modules: Dict[str, "Module"] = {}

    def __repr__(self) -> str:
        pass

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if name in self._params:
            return self._params[name]
        elif name in self._modules:
            return self._modules[name]

        raise AttributeError(f"No attribute {name} found in {self.__class__.__name__} module")
