import copy
import operator
from typing import Any, Dict, Iterator, List, Tuple, Union

from tripy import export
from tripy.common.exception import raise_error
from tripy.frontend.module.parameter import Parameter
from tripy.logging import logger


def _check_param_compatible(original_param, new_param, param_name):
    if not isinstance(original_param, Parameter):
        return

    is_compatible = original_param._is_compatible(new_param)
    if not is_compatible:
        raise_error(
            f"For parameter: {param_name}, new parameter is not compatible with the existing parameter.",
            details=is_compatible.error_details,
        )


@export.public_api(document_under="modules/index.rst")
class Module:
    r"""
    Base class used to define neural network modules.
    You can nest modules by assigning them as attributes of other modules.

    Child modules or :class:`tripy.Parameter` s may be contained in Python ``list``\s or ``dict``\s.
    If using ``dict``\s, the keys must be strings.
    Nested data structures (for example, ``list``\s of ``list``\s) are not supported.
    Taking child modules as an example, this is allowed:
    ::

        self.linear = tp.Linear(2, 2)
        self.list_modules = [tp.Linear(2, 2), tp.Linear(2, 2)]
        self.dict_modules = {
            "linear": tp.Linear(2, 2),
            "layernorm": tp.LayerNorm(2),
        }

    Whereas this is not supported:
    ::

        self.list_modules = [[tp.Linear(2, 2)], [tp.Linear(2, 2)]]
        self.dict_modules = {
            (1, "linear"): tp.Linear(2, 2),
        }

    .. code-block:: python
        :linenos:
        :caption: Example

        class AddBias(tp.Module):
            def __init__(self):
                super().__init__()
                self.bias = tp.Parameter(tp.Tensor([1.0, 1.0], dtype=tp.float32))

            def __call__(self, x):
                return x + self.bias

        add_bias = AddBias()

        input = tp.Tensor([1.0, 1.0], dtype=tp.float32)
        output = add_bias(input)

        assert np.array_equal(output.numpy(), np.array([2.0, 2.0]))
    """

    def __init__(self):
        # Avoid name clashes with members of child classes:
        self._tripy_params: Dict[str, Parameter] = {}
        self._tripy_modules: Dict[str, "Module"] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter) or ("_tripy_params" in self.__dict__ and name in self._tripy_params):
            _check_param_compatible(getattr(self, name, None), value, name)
            self._tripy_params[name] = value
        elif isinstance(value, Module) or ("_tripy_modules" in self.__dict__ and name in self._tripy_modules):
            self._tripy_modules[name] = value
        else:
            super().__setattr__(name, value)
            # avoid infinite recursion during initialization
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
                        self._tripy_modules[f"{name}.{idx}"] = v
                elif _check_types(value, Parameter):
                    for idx, v in enumerate(value):
                        key = f"{name}.{idx}"
                        _check_param_compatible(self._tripy_params.get(key), v, name)
                        self._tripy_params[key] = v
                else:
                    logger.warning("A list of mixed types will not get registered to module's state_dict().")
            elif isinstance(value, Dict):
                if not all(isinstance(k, str) for k in value):
                    logger.warning("A dict with non-string keys will not get registered to module's state_dict().")
                elif _check_types(value, Module):
                    for k, v in value.items():
                        self._tripy_modules[f"{name}.{k}"] = v
                elif _check_types(value, Parameter):
                    for k, v in value.items():
                        key = f"{name}.{k}"
                        _check_param_compatible(self._tripy_params.get(key), v, name)
                        self._tripy_params[key] = v
                else:
                    logger.warning("A dict of mixed types will not get registered to module's state_dict().")

    def __getattr__(self, name: str) -> Any:
        if name in self._tripy_params:
            return self._tripy_params[name]
        elif name in self._tripy_modules:
            return self._tripy_modules[name]

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

            class MyModule(tp.Module):
                def __init__(self):
                    super().__init__()
                    self.param = tp.Parameter(tp.ones(2, dtype=tp.float32))
                    self.linear1 = tp.Linear(2, 2)
                    self.linear2 = tp.Linear(2, 2)

            module = MyModule()

            state_dict = module.state_dict()

            assert set(state_dict.keys()) == {"param", "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"}
        """
        state_dict = copy.copy(self._tripy_params)

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

            class MyModule(tp.Module): # doc: omit
                def __init__(self): # doc: omit
                    super().__init__() # doc: omit
                    self.param = tp.Parameter(tp.ones(2, dtype=tp.float32)) # doc: omit
                    self.linear1 = tp.Linear(2, 2) # doc: omit
                    self.linear2 = tp.Linear(2, 2) # doc: omit
            module = MyModule() # doc: omit
            state_dict = module.state_dict() # doc: omit

            # Using the `module` and `state_dict` from the `state_dict()` example:
            print(f"Before: {module.param}")

            state_dict["param"] = tp.Parameter(tp.Tensor(np.zeros(2, dtype=np.float32)))
            module.load_from_state_dict(state_dict)

            print(f"After: {module.param}")

            assert np.array_equal(module.state_dict()["param"].numpy(), np.array(np.zeros(2, dtype=np.float32)))

        .. seealso:: :func:`state_dict`
        """

        def find_module(module: Union[Module, List, Dict], sub_strs: List[str]):
            while sub_strs:
                child_name = sub_strs.pop(0)
                if isinstance(module, list):
                    module = module[int(child_name)]
                elif isinstance(module, dict):
                    module = module[child_name]
                elif isinstance(module, Module):
                    module = operator.attrgetter(child_name)(module)
            return module

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
                    module = find_module(module, submodule_name.split("."))

            if isinstance(module, Module):
                _check_param_compatible(getattr(module, param_name), param, nested_attr_name)
                setattr(module, param_name, param)
            elif isinstance(module, list):
                _check_param_compatible(module[int(param_name)], param, nested_attr_name)
                module[int(param_name)] = param
            elif isinstance(module, dict):
                _check_param_compatible(module[param_name], param, nested_attr_name)
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

            class StackedLinear(tp.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = tp.Linear(2, 2)
                    self.linear2 = tp.Linear(2, 2)

            stacked_linear = StackedLinear()

            for name, module in stacked_linear.named_children():
                print(f"{name}: {type(module).__name__}")

            assert [name for name, _ in stacked_linear.named_children()] == ["linear1", "linear2"]
        """
        yield from self._tripy_modules.items()
