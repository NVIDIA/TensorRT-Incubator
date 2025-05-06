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

import abc
import copy
import inspect
import operator
from typing import Any, Dict, Iterator, List, Set, Tuple, Union

from nvtripy import export, utils, config
from nvtripy.common.exception import raise_error
from nvtripy.frontend.module.parameter import ParameterBase, DefaultParameter, OptionalParameter
from nvtripy.frontend.tensor import Tensor
from nvtripy.logging import logger
from nvtripy.utils.function_registry import type_str_from_arg


def _check_param_compatible(original_param, new_param, param_name):
    # We want to check the incoming parameter type even when the original parameter is not a tensor.
    if not isinstance(new_param, Tensor):
        raise_error(
            "Unrecognized type for module parameter.",
            f"Expected a tensor for parameter: '{param_name}', but got: {type_str_from_arg(new_param)}",
        )

    if not isinstance(original_param, (Tensor, ParameterBase)):
        # Allow values to be initialized to non-tensor types and changed later.
        # Note that this is required for the constructor to work since `original_param` will not be set.
        return

    is_compatible = utils.result.Result.ok()

    new_shape = tuple(map(int, new_param.shape))
    if not isinstance(original_param, ParameterBase) or original_param.shape is not None:
        # We need to evaluate here anyway, so we map the entire shape to numbers upfront to save us from recomputing
        # them again later.
        original_shape = tuple(map(int, original_param.shape))
        if original_shape != new_shape:
            is_compatible = utils.result.Result.err(
                ["New parameter shape: ", new_shape, " is not compatible with current shape: ", original_shape]
            )

    # Once we know the concrete shape of the parameter, we can update the trace tensor accordingly.
    # This not only makes the trace more informative, but is actually required for some APIs, like
    # addConvolutionND (to set kernelDims correctly).
    new_param.trace_tensor.shape = new_shape

    original_dtype = original_param.dtype
    new_dtype = new_param.dtype
    if original_dtype != new_dtype:
        is_compatible = utils.result.Result.err(
            ["New parameter dtype: ", new_dtype, " is not compatible with current dtype: ", original_dtype]
        )

    if not is_compatible:
        raise_error(
            f"For parameter: {param_name}, new parameter is not compatible with the existing parameter.",
            details=is_compatible.error_details,
        )


@export.public_api(document_under="modules/index.rst")
class Module:
    r"""
    Base class used to define neural network modules.
    Child classes must implement the :func:`forward` method.

    You can nest modules by assigning them as attributes of other modules.

    Child modules, :class:`nvtripy.Tensor` s, or other callables/lambda functions may be contained
    in Python ``list``\ s or ``dict``\ s.

    If using ``dict``\ s, the keys must be strings.
    Nested data structures (for example, ``list``\s of ``list``\s) are not supported.
    Taking child modules as an example, this is allowed:
    ::

        self.linear = tp.Linear(2, 2)
        self.list_modules = [tp.Linear(2, 2), tp.Linear(2, 2)]
        self.dict_modules = {
            "linear": tp.Linear(2, 2),
            "layernorm": tp.LayerNorm(2),
        }

    This is another valid example with a wrapped :class:`nvtripy.avgpool` lambda function
    ::

        self.dict_modules = {
            "convolution": tp.Conv(in_channels=2, out_channels=2, kernel_dims=(1,1)),
            "pool": lambda x: tp.avgpool(x, kernel_dims=(2,2))
        }

    Whereas this is not supported:
    ::

        self.list_modules = [[tp.Linear(2, 2)], [tp.Linear(2, 2)]]
        self.dict_modules = {
            (1, "linear"): tp.Linear(2, 2),
        }

    .. code-block:: python
        :linenos:

        class AddBias(tp.Module):
            def __init__(self):
                super().__init__()
                self.bias = tp.Tensor([1.0, 1.0], dtype=tp.float32)

            def forward(self, x):
                return x + self.bias

        add_bias = AddBias()

        input = tp.Tensor([1.0, 1.0], dtype=tp.float32)
        output = add_bias(input)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 2.0]))
    """

    def __init__(self):
        # Set up the signature so that `inspect.signature(module)` will return
        # the signature of the `forward` method - this way no information is lost
        # in the signature when compiling a Module's `__call__` method.
        self.__signature__ = inspect.signature(self.forward)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Tensor):
            _check_param_compatible(getattr(self, name, None), value, name)

        super().__setattr__(name, value)

    def state_dict(self) -> Dict[str, Tensor]:
        r"""
        Returns a dictionary mapping names to parameters in the module.
        This will recurse over any nested child modules.

        Returns:
            A dictionary mapping names to parameters.

        .. code-block:: python
            :linenos:

            # doc: print-locals state_dict

            class MyModule(tp.Module):
                def __init__(self):
                    super().__init__()
                    self.param = tp.ones((2,), dtype=tp.float32)
                    self.linear1 = tp.Linear(2, 2)
                    self.linear2 = tp.Linear(2, 2)

            module = MyModule()

            state_dict = module.state_dict()

            assert set(state_dict.keys()) == {"param", "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"}
        """
        state_dict = copy.copy(dict(self.named_parameters()))

        for child_name, child in self.named_children():
            child_state_dict = child.state_dict()
            for name, param in child_state_dict.items():
                # We add a prefix for any parameters coming from nested modules
                # so they can be disambiguated correctly in higher level modules.
                state_dict[f"{child_name}.{name}"] = param

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True) -> Tuple[Set[str], Set[str]]:
        r"""
        Loads parameters from the provided ``state_dict`` into the current module.
        This will recurse over any nested child modules.

        Args:
            state_dict: A dictionary mapping names to parameters.
            strict: If True, keys in ``state_dict`` must exactly match those in this module. If not,
                an error will be raised.

        Returns:
            A ``tuple`` of two ``set``\s of strings representing:
            - missing_keys: keys that are expected by this module but not provided in ``state_dict``.
            - unexpected_keys: keys that are not expected by this module but provided in ``state_dict``.

        .. code-block:: python
            :linenos:

            # doc: no-print-locals

            class MyModule(tp.Module):
                def __init__(self):
                    super().__init__()
                    self.param = tp.ones((2,), dtype=tp.float32)

            module = MyModule()

            print(f"Before: {module.param}")

            module.load_state_dict({"param": tp.zeros((2,), dtype=tp.float32)})

            print(f"After: {module.param}")

            assert np.array_equal(cp.from_dlpack(module.state_dict()["param"]).get(), np.array(np.zeros((2,), dtype=np.float32)))

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

        original_sd = self.state_dict()
        expected_keys = set(original_sd)
        optional_keys = set(k for k, v in original_sd.items() if isinstance(v, OptionalParameter))
        provided_keys = set(state_dict)
        missing_keys = expected_keys - optional_keys - provided_keys
        unexpected_keys = provided_keys - expected_keys
        if strict and (missing_keys or unexpected_keys):
            details = []
            if missing_keys:
                details.append(f"Missing keys: {sorted(missing_keys)}\n")
            if unexpected_keys:
                details.append(
                    f"Unexpected keys:\n{sorted(unexpected_keys)}\n\nNote: Expected keys were:\n{sorted(expected_keys)}"
                )
            raise_error(
                "state_dict is incompatible.",
                details,
            )

        for nested_attr_name, param in state_dict.items():
            if nested_attr_name in unexpected_keys:
                continue
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
        return (missing_keys, unexpected_keys)

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        r"""
        Returns an iterator over immediate children of this module, yielding tuples
        containing the name of the child module and the child module itself.

        Returns:
            An iterator over tuples containing the name of the child module and the child module itself.

        .. code-block:: python
            :linenos:

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
        yield from self._iterate_members_of_types({Module})

    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        r"""
        Returns:
            An iterator over tuples containing the name of a parameter and the parameter itself.

        .. code-block:: python
            :linenos:

            # doc: no-print-locals

            class MyModule(tp.Module):
                def __init__(self):
                    super().__init__()
                    self.alpha = tp.Tensor(1)
                    self.beta = tp.Tensor(2)

            linear = MyModule()

            for name, parameter in linear.named_parameters():
                print(f"{name}: {parameter}")

            assert [name for name, _ in linear.named_parameters()] == ["alpha", "beta"]
        """
        yield from self._iterate_members_of_types({Tensor, ParameterBase})

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        r"""
        Calls the module with the specified arguments.

        Args:
            *args: Positional arguments to the module.
            **kwargs: Keyword arguments to the module.

        Returns:
            The outputs computed by the module.

        .. code-block:: python
            :linenos:

            class Module(tp.Module):
                def forward(self, x):
                    return tp.relu(x)

            module = Module()

            input = tp.arange(-3, 3)
            out = module(input) # Note that we do not call `forward` directly.
        """
        state_dict = self.state_dict()

        not_set_parameters = {name: param for name, param in state_dict.items() if isinstance(param, DefaultParameter)}

        if not_set_parameters:
            if "all" in config.extra_error_information:
                params_errors = ["\n\n"]
                for name, param in not_set_parameters.items():
                    params_errors.extend([f"'{name}': defined here:", param])
            else:
                params_errors = [", ".join(not_set_parameters.keys())]

            raise_error(
                "Some parameters were never set in the module. Please assign tensors for these parameters.",
                ["Note: Not set parameters were: "] + params_errors,
            )

        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""
        Implements the forward pass of this module.
        Child classes must implement this method.

        Args:
            *args: Positional arguments to the module.
            **kwargs: Keyword arguments to the module.

        Returns:
            The outputs computed by the module.

        .. code-block:: python
            :linenos:

            class Module(tp.Module):
                def forward(self, x):
                    return tp.relu(x)

            module = Module()

            input = tp.arange(-3, 3)
            out = module(input) # Note that we do not call `forward` directly.
        """
        ...

    def _iterate_members_of_types(self, types: type) -> Iterator[Tuple[str, Any]]:
        def isinstance_any(obj):
            return any(isinstance(obj, typ) for typ in types)

        for name, value in vars(self).items():
            if isinstance_any(value):
                yield name, value
            elif isinstance(value, List):
                for i, obj in enumerate(value):
                    if isinstance_any(obj):
                        yield f"{name}.{i}", obj
            elif isinstance(value, Dict):
                for key, obj in value.items():
                    if isinstance_any(obj):
                        yield f"{name}.{key}", obj

    def __str__(self):
        from textwrap import indent

        class_name = self.__class__.__name__
        module_str = f"{class_name}(\n"

        body_str = ""
        for name, param in self.named_parameters():
            body_str += f"{name}: Parameter = (shape={param.shape}, dtype={param.dtype}),\n"

        for name, child in self.named_children():
            body_str += f"{name}: Module = {str(child).strip()},\n"

        module_str += indent(body_str, " " * 4)
        module_str += f")"
        return module_str

    def initialize_dummy_parameters(self) -> None:
        """
        Initializes any uninitialized parameters in the module with dummy values.
        This is useful for debugging and testing purposes.

        .. code-block:: python
            :linenos:

            # doc: no-print-locals
            linear = tp.Linear(2, 2)
            print(linear.state_dict())
            assert not isinstance(linear.weight, tp.Tensor)
            assert not isinstance(linear.bias, tp.Tensor)

            linear.initialize_dummy_parameters()
            print(linear.state_dict())
            assert isinstance(linear.weight, tp.Tensor)
            assert isinstance(linear.bias, tp.Tensor)
        """
        from nvtripy.frontend.ops.ones import ones

        state_dict = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, DefaultParameter):
                state_dict[name] = ones(shape=param.shape, dtype=param.dtype)

        self.load_state_dict(state_dict)
