from typing import Union, List, Dict, Iterator
from dataclasses import dataclass

from tripy import export
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
            *modules (Union[Module, Dict[str, Module]]): The modules to include in the sequence.
                Can be passed as individual positional arguments or as a single dictionary of named modules.

        .. code-block:: python
            :linenos:
            :caption: Example

            import tripy as tp

            # Sequential with layers passed as arguments
            model = tp.Sequential(tp.Linear(1, 3), tp.Linear(3, 2))

            # Sequential with named dictionary of layers
            model_named = tp.Sequential({'layer1': tp.Linear(1, 3), 'layer2': tp.Linear(3, 2)})

            # Forward pass
            x = tp.Tensor([1.0])
            output = model(x)
        """
        super().__init__()
        self._modules = {}

        if len(modules) == 1 and isinstance(modules[0], dict):
            for name, module in modules[0].items():
                self.add_module(name, module)
        else:
            for idx, module in enumerate(modules):
                self.add_module(str(idx), module)

    def __call__(self, x: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Defines the forward pass by applying each module in the container sequentially to input `x`.

        Args:
            x (Tensor): The input tensor to pass through the sequence of modules.

        Returns:
            Tensor: The output tensor after passing through each module in sequence.
        """
        for module in self._modules.values():
            x = module(x)
        return x
    
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
        return len(self._modules)


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
        return iter(self._modules.values())

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
            layer = model[1]
            assert isinstance(layer, tp.Linear)
        """
        if isinstance(idx, int):
            return list(self._modules.values())[idx]
        elif isinstance(idx, str):
            return self._modules[idx]
        else:
            raise TypeError("Index must be an int or str")

    def __setitem__(self, idx: Union[int, str], module: Module) -> None:
        r"""
        Replaces a module at a specific index or name.

        Args:
            idx (Union[int, str]): The index or name of the module to replace.
            module (Module): The new module to set at the specified position.

        Raises:
            TypeError: If `idx` is not an int or str, or if `module` is not an instance of `Module`.
        
        .. code-block:: python
            :linenos:
            :caption: Example

            model = tp.Sequential(tp.Linear(1, 3), tp.Linear(3, 2))
            model[1] = tp.Linear(3, 1)
            assert isinstance(model[1], tp.Linear)
        """
        if not isinstance(module, Module):
            raise TypeError(f"{module} is not of type Module")

        if isinstance(idx, int):
            name = list(self._modules.keys())[idx]
        elif isinstance(idx, str):
            name = idx
        else:
            raise TypeError("Index must be an int or str")

        self._modules[name] = module
        setattr(self, name, module)

    def __delitem__(self, idx: Union[int, str]) -> None:
        r"""
        Deletes a module by index or name.
        
        Args:
            idx (Union[int, str]): The index or name of the module to delete.

        Raises:
            TypeError: If `idx` is not an int or str.
        
        .. code-block:: python
            :linenos:
            :caption: Example

            model = tp.Sequential(tp.Linear(1, 3), tp.Linear(3, 2))
            del model[1]
            assert len(model) == 1
        """
        if isinstance(idx, int):
            name = list(self._modules.keys())[idx]
        elif isinstance(idx, str):
            name = idx
        else:
            raise TypeError("Index must be an int or str")

        del self._modules[name]
        delattr(self, name)

    def add_module(self, name: str, module: Module) -> None:
        """
        Adds a module with a specified name to the sequence.
        
        Args:
            name (str): The name of the module to be added.
            module (Module): The module to be added to the sequence.
        
        Returns:
            None

        Raises:
            TypeError: If `module` is not an instance of `Module`.

        .. code-block:: python
            :linenos:
            :caption: Example

            model = tp.Sequential()
            model.add_module("layer1", tp.Linear(1, 3))
            model.add_module("layer2", tp.Linear(3, 2))

            assert len(model) == 2
        """
        if not isinstance(module, Module):
            raise TypeError(f"{module} is not of type Module")
        
        self._modules[name] = module
        setattr(self, name, module)

    def append(self, module: Module) -> "Sequential":
        """
        Appends a module to the end of the sequence using an auto-generated name.

        Args:
            module (Module): The module to append to the sequence.

        Returns:
            Sequential: The instance of `Sequential` with the added module, allowing for chaining.
        
        Raises:
            TypeError: If `module` is not an instance of `Module`.

        .. code-block:: python
            :linenos:
            :caption: Example

            model = tp.Sequential()
            model.append(tp.Linear(1, 3))
            model.append(tp.Linear(3, 2))

            assert len(model) == 2
        """
        self.add_module(str(len(self)), module)
        return self
   
    def extend(self, *modules: Union[Module, Dict[str, Module]]) -> "Sequential":
        r"""
        Appends modules to the sequence, accepting either a dictionary of named modules
        or individual modules as arguments.

        Args:
            *modules (Union[Module, Dict[str, Module]]): Additional modules to append to the sequence.
                Can be individual modules as positional arguments or a single dictionary of named modules.
        
        Returns:
            Sequential: The instance of `Sequential` with the added module, allowing for chaining.
    
        .. code-block:: python
            :linenos:
            :caption: Example

            model = tp.Sequential(tp.Linear(1, 64))
            model.extend(tp.Linear(64, 128), tp.Linear(128, 256))
            assert len(model) == 3
        """
        if len(modules) == 1 and isinstance(modules[0], dict):
            for name, module in modules[0].items():
                self.add_module(name, module)
        else:
            for layer in modules:
                self.append(layer)

    def __str__(self) -> str:
        r"""
        Returns a string representation of the Sequential container, showing each module's name and type.

        Returns:
            str: The string representation of the Sequential container.
        
        .. code-block:: python
            :linenos:
            :caption: Example

            model = tp.Sequential(tp.Linear(1, 3), tp.Linear(3, 2))
            print(model)
        """
        module_str = "\n".join(f"({name}): {module}" for name, module in self._modules.items())
        return f"{self.__class__.__name__}(\n{module_str}\n)"
