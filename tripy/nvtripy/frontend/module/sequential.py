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
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Tuple, Union

from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.frontend.module import Module

ModuleLike = Union[Module, Callable[["nvtripy.Tensor"], "nvtripy.Tensor"]]


@export.public_api(document_under="modules/sequential.rst")
@dataclass
class Sequential(Module):
    r"""
    A module to stack multiple callable layers or modules in a sequential order. The `Sequential`
    container can accept either a list of modules/callable objects or a dictionary of named modules/callable objects.
    Layers are added in the order they are passed, and each is called sequentially during the forward pass.
    """

    def __init__(self, *modules: Union[ModuleLike, Dict[str, ModuleLike]]) -> None:
        r"""
        Args:
            *modules: The module(s) or callable(s) to include in the sequence.
                These must take exactly one input and return exactly one output.
                Can be passed as individual positional arguments or as a single dictionary of named modules.

        .. code-block:: python
            :linenos:
            :caption: Sequential with Positional Arguments

            model = tp.Sequential(tp.Linear(1, 3), tp.Linear(3, 2))

            model.load_state_dict({
                "0.weight": tp.ones((3, 1)),
                "0.bias": tp.ones((3, )),
                "1.weight": tp.ones((2, 3)),
                "1.bias": tp.ones((2, )),
            })

            input = tp.Tensor([1.0])
            output = model(input)

        .. code-block:: python
            :linenos:
            :caption: Sequential with a Dictionary

            model = tp.Sequential({'layer1': tp.Linear(1, 3), 'layer2': tp.Linear(3, 2)})

            model.load_state_dict({
                "layer1.weight": tp.ones((3, 1)),
                "layer1.bias": tp.ones((3, )),
                "layer2.weight": tp.ones((2, 3)),
                "layer2.bias": tp.ones((2, )),
            })

            input = tp.Tensor([1.0])
            output = model(input)

        .. code-block:: python
            :linenos:
            :caption: Sequential with Callables

            model = tp.Sequential(
                tp.relu,
                lambda x: tp.avgpool(x, kernel_dims=(2,2), stride=(1,1))
            )

            input = tp.ones((1,2,2,2), dtype=tp.float32)
            output = model(input)
        """
        super().__init__()
        self.modules = {}

        if len(modules) == 1 and isinstance(modules[0], dict):
            self.modules = copy.copy(modules[0])
        else:
            for idx, module in enumerate(modules):
                self.modules[str(idx)] = module

    def forward(self, input: "nvtripy.Tensor") -> "nvtripy.Tensor":
        r"""
        Defines the forward pass by applying each module in the container sequentially to `input`

        Args:
            input: The input tensor to pass through the sequence of modules.

        Returns:
            The output tensor after passing through each module in sequence.
        """
        for module in self.modules.values():
            input = module(input)
        return input

    def __getattr__(self, name: str) -> Any:
        """
        Custom __getattr__ to search both in `modules` dictionary and in other attributes. This is for handling
        `module = operator.attrgetter(child_name)(module)` calls in nvtripy/frontend/module/module.py:load_state_dict
        """
        if name in self.modules:
            return self.modules[name]

        # Fallback to regular attribute access if not found in modules
        return super().__getattr__(name)

    def __len__(self) -> int:
        r"""
        Returns the total number of modules in the sequence.

        Returns:
            The number of modules in the sequence.

        .. code-block:: python
            :linenos:

            # doc: print-locals model length

            model = tp.Sequential(tp.Linear(1, 64), tp.Linear(64, 128))
            length = len(model)
            assert length == 2
        """
        return len(self.modules)

    def __iter__(self) -> Iterator[Module]:
        r"""
        Returns an iterator over the modules in the sequence.

        Returns:
            An iterator over the modules.

        .. code-block:: python
            :linenos:

            model = tp.Sequential(tp.Linear(1, 3), tp.Linear(3, 2))
            for layer in model:
                print(layer)
        """
        return iter(self.modules.values())

    def __getitem__(self, idx: Union[int, str]) -> Module:
        r"""
        Accesses a module by index (int) or name (str).

        Args:
            idx: The index or name of the module to retrieve.

        Returns:
            The module at the specified index or name.

        Raises:
            TypeError: If `idx` is not an int or str.

        .. code-block:: python
            :linenos:

            model = tp.Sequential(tp.Linear(1, 3), tp.Linear(3, 2))
            print(model[1])
        """
        key = str(idx) if isinstance(idx, int) else idx

        if key not in self.modules:
            raise_error(
                f"Key: '{key}' not found in modules.", [f"Note: Available keys were: {list(self.modules.keys())}"]
            )

        return self.modules[key]

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        r"""
        Returns an iterator over all the first-order modules in this `Sequential` container.
        Each child module is represented by its name and the module object itself.

        Returns:
            An iterator over tuples containing
            the name and module of each child.

        .. code-block:: python
            :linenos:

            model = tp.Sequential(tp.Linear(1, 3), tp.Linear(3, 2))

            for name, child in model.named_children():
                print(f"{name}: {type(child).__name__}")

        """
        # Overriding the base implementation to prevent displaying every child module
        # with the 'modules' prefix in the state_dict. This change ensures compatibility
        # with PyTorch's naming conventions.
        for name, module in self.modules.items():
            if isinstance(module, Module):
                yield name, module
