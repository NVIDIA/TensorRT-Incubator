import copy
from typing import Any, Dict, Iterator, Tuple

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

    def state_dict(self) -> Dict[str, Parameter]:
        r"""
        Returns a dictionary mapping names to parameters in the module.
        This will recurse over any nested child modules.

        Returns:
            A dictionary mapping names to parameters.

        Example:
        ::

            class MyModule(tp.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.param = tp.nn.Parameter(tp.ones(2, dtype=tp.float32))
                    self.linear1 = tp.nn.Linear(2, 2)
                    self.linear2 = tp.nn.Linear(2, 2)

            module = MyModule()

            state_dict = module.state_dict()
            print(state_dict.keys())
            assert set(state_dict.keys()) == {"param", "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"}
        """
        state_dict = copy.copy(self._params)

        for child_name, child in self.named_children():
            child_state_dict = child.state_dict()
            for name, param in child_state_dict.items():
                # We add a prefix for any parameters coming from nested modules
                # so they can be disambiguated correctly in higher level modules.
                state_dict[f"{child_name}.{name}"] = param

        return state_dict

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        r"""
        Returns an iterator over immediate children of this module, yielding tuples
        containing the name of the child module and the child module.

        Yields:
            A tuple containing the name of the child module and the child module

        Example:
        ::

            class StackedLinear(tp.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = tp.nn.Linear(2, 2)
                    self.linear2 = tp.nn.Linear(2, 2)

            stacked_linear = StackedLinear()

            for name, module in stacked_linear.named_children():
                print(f"{name}: {type(module)}")

            assert [name for name, _ in stacked_linear.named_children()] == ["linear1", "linear2"]
        """
        yield from self._modules.items()
