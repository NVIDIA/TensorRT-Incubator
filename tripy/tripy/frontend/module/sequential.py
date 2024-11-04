from typing import Union, Tuple, Dict, Iterator, Any
from dataclasses import dataclass

from tripy import export
from tripy.frontend.module.parameter import Parameter
from tripy.frontend.module import Module


@export.public_api(document_under="modules/sequential.rst")
@dataclass
class Sequential(Module):
    r"""
    A module to stack multiple layers or modules in a sequential order. The `Sequential`
    container can accept either a list of modules or a dictionary of named modules. Modules are
    added in the order they are passed, and each is called sequentially during the forward pass.
    """

    modules: Union[Module, Dict[str, Module]]
    r"""The modules to include in the sequence"""

    def __init__(self, *modules: Union[Module, Dict[str, Module]]) -> None:
        r"""
        Args:
            *modules: The modules to include in the sequence.
                Can be passed as individual positional arguments or as a single dictionary of named modules.

        .. code-block:: python
            :linenos:
            :caption: Example

            import tripy as tp

            # Sequential with layers passed as arguments
            model = tp.Sequential(tp.Linear(1, 3), tp.Linear(3, 2))

            # Sequential with named dictionary of layers
            model_named = tp.Sequential({'layer1': tp.Linear(1, 3), 'layer2': tp.Linear(3, 2)})

            x = tp.Tensor([1.0])
            output = model(x)
        """
        super().__init__()
        self.modules = {}

        if len(modules) == 1 and isinstance(modules[0], dict):
            for name, module in modules[0].items():
                self.modules[name] = module
        else:
            for idx, module in enumerate(modules):
                self.modules[str(idx)] = module

    def __call__(self, x: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Defines the forward pass by applying each module in the container sequentially to input `x`.

        Args:
            x (Tensor): The input tensor to pass through the sequence of modules.

        Returns:
            Tensor: The output tensor after passing through each module in sequence.
        """
        for module in self.modules.values():
            x = module(x)
        return x

    def __getattr__(self, name) -> Any:
        """
        Custom __getattr__ to search both in `modules` dictionary and in other attributes. This is for handling
        `module = operator.attrgetter(child_name)(module)` calls in tripy/frontend/module/module.py:load_state_dict
        """
        # Check if `name` is a key in the modules dictionary
        if "modules" in self.__dict__ and name in self.modules:
            return self.modules[name]

        # Fallback to regular attribute access if not found in modules
        try:
            return super().__getattr__(name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute or module named '{name}'")

    def __len__(self) -> int:
        r"""
        Returns the total number of modules in the sequence.

        Returns:
            int: The number of modules in the sequence.

        .. code-block:: python
            :linenos:
            :caption: Example

            model = tp.Sequential(tp.Linear(1, 64), tp.Linear(64, 128))
            assert len(model) == 2
        """
        return len(self.modules)

    def __iter__(self) -> Iterator[Module]:
        r"""
        Returns an iterator over the modules in the sequence.

        Returns:
            Iterator[Module]: An iterator over the modules.

        .. code-block:: python
            :linenos:
            :caption: Example

            model = tp.Sequential(tp.Linear(1, 3), tp.Linear(3, 2))
            for layer in model:
                print(layer)
        """
        return iter(self.modules.values())

    def __getitem__(self, idx: Union[int, str]) -> Module:
        r"""
        Accesses a module by index (int) or name (str).

        Args:
            idx (Union[int, str]): The index or name of the module to retrieve.

        Returns:
            Module: The module at the specified index or name.

        Raises:
            TypeError: If `idx` is not an int or str.

        .. code-block:: python
            :linenos:
            :caption: Example

            model = tp.Sequential(tp.Linear(1, 3), tp.Linear(3, 2))
            print(model[1])
        """
        if not isinstance(idx, (int, str)):
            raise TypeError("Index must be an int or str.")

        key = str(idx) if isinstance(idx, int) else idx

        if key not in self.modules:
            raise ValueError(f"Key {key} not found in modules.")

        return self.modules[key]

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        r"""
        Returns an iterator over all child modules in this `Sequential` container.
        Each child module is represented by its name and the module object itself.

        Returns:
            Iterator[Tuple[str, Module]]: An iterator over tuples containing
            the name and module of each child.

        .. code-block:: python
            :linenos:
            :caption: Example

            model = tp.Sequential(tp.Linear(1, 3), tp.Linear(3, 2))

            for name, child in model.named_children():
                print(f"{name}: {type(child).__name__}")

        """
        for name, module in self.modules.items():
            if isinstance(module, Module):
                yield name, module
            elif isinstance(module, Sequential):
                # Traverse deeper into Sequential containers within Sequential
                for child_name, child in module.named_children():
                    yield f"{name}.{child_name}", child
