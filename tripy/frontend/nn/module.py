import copy
from typing import Any, Dict, List, Iterator, Tuple, Union
import operator

from tripy.common import logger
from tripy.frontend.nn.parameter import Parameter
from tripy.common.exception import raise_error


class Module:
    r"""
    Base class used to define neural network modules.
    You can nest modules by assigning them as attributes of other modules.

    Child modules or :class:`tripy.nn.Parameter` s may be contained in Python ``list``\s or ``dict``\s.
    If using ``dict``\s, the keys must be strings.
    Nested data structures (for example, ``list``\s of ``list``\s) are not supported.
    Taking child modules as an example, this is allowed:
    ::

        self.linear = tp.nn.Linear(2, 2)
        self.list_modules = [tp.nn.Linear(2, 2), tp.nn.Linear(2, 2)]
        self.dict_modules = {
            "linear": tp.nn.Linear(2, 2),
            "layernorm": tp.nn.LayerNorm(2),
        }

    Whereas this is not supported:
    ::

        self.list_modules = [[tp.nn.Linear(2, 2)], [tp.nn.Linear(2, 2)]]
        self.dict_modules = {
            (1, "linear"): tp.nn.Linear(2, 2),
        }

    .. code-block:: python
        :linenos:
        :caption: Example

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
            if not value:
                return

            def _check_types(objs: Union[List, Dict], typ: Any):
                if isinstance(objs, List):
                    return all(isinstance(obj, typ) for obj in objs)
                else:
                    return all(isinstance(obj, typ) for _, obj in objs.items())

            # register modules/params from container if all elements are Modules/Parameters
            if isinstance(value, List):
                if _check_types(value, Module):
                    for idx, v in enumerate(value):
                        self._modules[f"{name}.{idx}"] = v
                elif _check_types(value, Parameter):
                    for idx, v in enumerate(value):
                        self._params[f"{name}.{idx}"] = v
                else:
                    logger.warning("A list of mixed types will not get registered to module's state_dict().")
            elif isinstance(value, Dict):
                if not all(isinstance(k, str) for k in value):
                    logger.warning("A dict with non-string keys will not get registered to module's state_dict().")
                elif _check_types(value, Module):
                    for k, v in value.items():
                        self._modules[f"{name}.{k}"] = v
                elif _check_types(value, Parameter):
                    for k, v in value.items():
                        self._params[f"{name}.{k}"] = v
                else:
                    logger.warning("A dict of mixed types will not get registered to module's state_dict().")

    def __getattr__(self, name: str) -> Any:
        if name in self._params:
            return self._params[name]
        elif name in self._modules:
            return self._modules[name]

        raise AttributeError(f"No attribute '{name}' found in '{self.__class__.__name__}' module")

    def state_dict(self) -> Dict[str, Parameter]:
        r"""
        Returns a dictionary mapping names to parameters in the module.
        This will recurse over any nested child modules.

        Returns:
            A dictionary mapping names to parameters.

        .. code-block:: python
            :linenos:
            :caption: Example

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

    def load_from_state_dict(self, state_dict: Dict[str, Parameter]) -> None:
        r"""
        Loads parameters from the provided ``state_dict`` into the current module.
        This will recurse over any nested child modules.

        Args:
            state_dict: A dictionary mapping names to parameters.

        .. code-block:: python
            :linenos:
            :caption: Example

            # doc: no-print-locals

            class MyModule(tp.nn.Module): # doc: omit
                def __init__(self): # doc: omit
                    super().__init__() # doc: omit
                    self.param = tp.nn.Parameter(tp.ones(2, dtype=tp.float32)) # doc: omit
                    self.linear1 = tp.nn.Linear(2, 2) # doc: omit
                    self.linear2 = tp.nn.Linear(2, 2) # doc: omit
            module = MyModule() # doc: omit
            state_dict = module.state_dict() # doc: omit

            # Using the `module` and `state_dict` from the `state_dict()` example:
            print(f"Before: {module.param}")

            state_dict["param"] = tp.nn.Parameter(tp.Tensor(np.zeros(2, dtype=np.float32)))
            module.load_from_state_dict(state_dict)

            print(f"After: {module.param}")

            assert np.array_equal(module.state_dict()["param"].numpy(), np.array(np.zeros(2, dtype=np.float32)))

        .. seealso:: :func:`state_dict`
        """

        def _find_module(module: Union[Module, List, Dict], sub_strs: List[str]):
            while sub_strs:
                child_name = sub_strs.pop(0)
                if isinstance(module, list):
                    module = module[int(child_name)]
                elif isinstance(module, dict):
                    module = module[child_name]
                elif isinstance(module, Module):
                    module = operator.attrgetter(child_name)(module)
            return module

        def _check_param_type(cls, original_param, new_param, param_name):
            # TODO: add check for dtype when https://gitlab-master.nvidia.com/TensorRT/poc/tripy/-/merge_requests/215 is merged.
            original_param_shape = original_param.shape.eval()
            new_param_shape = new_param.shape.eval()
            if original_param_shape != new_param_shape:
                raise_error(
                    "Shape of new parameter does not match shape of existing parameter.",
                    details=[
                        f"For parameter '{param_name}', currently assigned parameter has shape: '{original_param_shape}' while new parameter has shape: '{new_param_shape}'"
                    ],
                )

        for nested_attr_name, param in state_dict.items():
            submodule_name, _, param_name = nested_attr_name.rpartition(".")
            # If there is no submodule, it means we are accessing a parameter of self
            module = self
            if submodule_name:
                try:
                    # try to access module.submodule_name as it's the most common case
                    module = operator.attrgetter(submodule_name)(self)
                except AttributeError:
                    logger.verbose(f"Cannot access {submodule_name} directly, trying to find the correct module.")
                    # find module starting from the beginning
                    module = _find_module(module, submodule_name.split("."))

            if isinstance(module, Module):
                _check_param_type(self, getattr(module, param_name), param, nested_attr_name)
                setattr(module, param_name, param)
            elif isinstance(module, list):
                _check_param_type(self, module[int(param_name)], param, nested_attr_name)
                module[int(param_name)] = param
            elif isinstance(module, dict):
                _check_param_type(self, module[param_name], param, nested_attr_name)
                module[param_name] = param

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        r"""
        Returns an iterator over immediate children of this module, yielding tuples
        containing the name of the child module and the child module itself.

        Returns:
            An iterator over tuples containing the name of the child module and the child module itself.

        .. code-block:: python
            :linenos:
            :caption: Example

            # doc: no-print-locals

            class StackedLinear(tp.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = tp.nn.Linear(2, 2)
                    self.linear2 = tp.nn.Linear(2, 2)

            stacked_linear = StackedLinear()

            for name, module in stacked_linear.named_children():
                print(f"{name}: {type(module).__name__}")

            assert [name for name, _ in stacked_linear.named_children()] == ["linear1", "linear2"]
        """
        yield from self._modules.items()
