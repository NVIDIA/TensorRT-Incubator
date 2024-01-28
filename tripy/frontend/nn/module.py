import copy
from typing import Any, Dict, Iterator, Tuple
import operator

from tripy.frontend.nn.parameter import Parameter


class Module:
    """
    Base class used to define neural network modules.
    You can nest modules by assigning them as attributes of other modules.

    The implementation currently assumes that :class:`tripy.nn.Parameter` s are associated
    with the Module as direct attributes and not contained in other data structures.

    Specifically, this is allowed:
    ::

        self.param = Parameter(Tensor([1,2,3]))

    Whereas this is not currently supported:
    ::

        self.param = {"param1": Parameter(Tensor([1,2,3]))}

    Example:

    .. code:: python
        :number-lines:

        class AddBias(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = tp.nn.Parameter(tp.Tensor([1.0, 1.0], dtype=tp.float32))

            def __call__(self, x):
                return x + self.bias

        add_bias = AddBias()

        input = tp.Tensor([1.0, 1.0], dtype=tp.float32)
        output = add_bias(input)

        assert np.array_equal(output.numpy(), np.array([2.0, 2.0]))
    """

    def __init__(self):
        self._params: Dict[str, Parameter] = {}
        self._modules: Dict[str, "Module"] = {}

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

        .. code:: python
            :number-lines:

            # doc: print-locals state_dict

            class MyModule(tp.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.param = tp.nn.Parameter(tp.ones(2, dtype=tp.float32))
                    self.linear1 = tp.nn.Linear(2, 2)
                    self.linear2 = tp.nn.Linear(2, 2)

            module = MyModule()

            state_dict = module.state_dict()

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

    def load_from_state_dict(self, state_dict: Dict[str, Parameter]):
        r"""
        Loads parameters from the provided ``state_dict`` into the current module.
        This will recurse over any nested child modules.

        Args:
            state_dict: A dictionary mapping names to parameters.

        For example, using the module defined in the example in :func:`state_dict` :

        .. code:: python
            :number-lines:

            # doc: no-print-locals

            class MyModule(tp.nn.Module): # doc: omit
                def __init__(self): # doc: omit
                    super().__init__() # doc: omit
                    self.param = tp.nn.Parameter(tp.ones(2, dtype=tp.float32)) # doc: omit
                    self.linear1 = tp.nn.Linear(2, 2) # doc: omit
                    self.linear2 = tp.nn.Linear(2, 2) # doc: omit
            module = MyModule() # doc: omit
            state_dict = module.state_dict() # doc: omit

            print(f"Before: {module.param}")

            state_dict["param"] = tp.nn.Parameter(tp.Tensor(np.zeros(2, dtype=np.float32)))
            module.load_from_state_dict(state_dict)

            print(f"After: {module.param}")

            assert np.array_equal(module.state_dict()["param"].numpy(), np.array(np.zeros(2, dtype=np.float32)))
        """
        for nested_attr_name, param in state_dict.items():
            submodule_name, _, param_name = nested_attr_name.rpartition(".")
            # If there is no submodule, it means we are accessing a parameter of self
            module = self
            if submodule_name:
                module = operator.attrgetter(submodule_name)(self)
            setattr(module, param_name, param)

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        r"""
        Returns an iterator over immediate children of this module, yielding tuples
        containing the name of the child module and the child module.

        Yields:
            A tuple containing the name of the child module and the child module

        Example:

        .. code:: python
            :number-lines:

            # doc: no-print-locals

            class StackedLinear(tp.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = tp.nn.Linear(2, 2)
                    self.linear2 = tp.nn.Linear(2, 2)

            stacked_linear = StackedLinear()

            for name, module in stacked_linear.named_children():
                print(f"{name}: {module}")

            assert [name for name, _ in stacked_linear.named_children()] == ["linear1", "linear2"]
        """
        yield from self._modules.items()
