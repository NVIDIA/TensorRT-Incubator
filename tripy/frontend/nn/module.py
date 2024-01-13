from typing import Any, Dict

import pickle

import tripy as tp
from tripy.frontend.nn.parameter import Parameter


class Module:
    """
    Base class used to create all neural network modules.
    Module class allows accessing all the parameters associated within nested Modules.

    The implementation currently assumes that Parameters are associated with the Module
    as an attribute and are not part of nested List or Dict.
    Ex:
    self.param = Parameter(Tensor([1,2,3])) # This is allowed
    self.param = {"param1": Parameter(Tensor([1,2,3]))} # This is not allowed

    Example:
    ::

        import numpy as np

        class AddBias(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = tp.nn.Parameter(tp.Tensor([1.0, 1.0], dtype=tp.float32))

            def __call__(self, x):
                return x + self.bias

        add_bias = AddBias()
        inp = tp.Tensor([1.0, 1.0], dtype=tp.float32)
        out = add_bias(inp)

        print(out)
        assert (out.numpy() == np.array([2.0, 2.0])).all()
    """

    _params: Dict[str, Parameter]
    _modules: Dict[str, "Module"]

    def __init__(self):
        self._params = {}
        self._modules = {}

    def __repr__(self) -> str:
        pass

    def save_weights(self, file_name: str):
        """
        Save Module parameters to the specified path.

        Args:
            file_name: The file name at which to save weights.
        """
        param_dict = {}
        stack = [("", self)]
        while stack:
            m_prefix, module = stack.pop()
            current_params = module._params
            current_params = {m_prefix + str(key): val for key, val in current_params.items()}
            param_dict = {**param_dict, **current_params}
            for m in module._modules:
                stack.append((m_prefix + m + ".", module._modules[m]))

        # todo: fix cpu_view and should support all types without explicit param.
        weights_dict = {key: value.numpy() for key, value in param_dict.items()}
        with open(file_name, "wb") as f:
            pickle.dump(weights_dict, f)

    def load_weights(self, file_name: str):
        """
        Load Module parameters from the specified path.

        Args:
            file_name: The file name from which to load weights.
        """
        weights_dict = None
        with open(file_name, "rb") as f:
            weights_dict = pickle.load(f)

        stack = [("", self)]
        while stack:
            m_prefix, module = stack.pop()
            for key, val in module._params.items():
                new_key = m_prefix + str(key)
                module._params[key] = Parameter(tp.Tensor(weights_dict[new_key]))

            for m in module._modules:
                stack.append((m_prefix + m + ".", module._modules[m]))

    def apply(self, fn: callable, recurse=True):
        """
        Apply user function (fn) to all parameters of this Module and all nested Modules (if recurse is enabled).
        Args:
            fn: User defined function to be applied to parameters
            recurse: Recursively accesses all submodules associated with this module.
        """
        param_dict = {}
        stack = [("", self)]
        while stack:
            m_prefix, module = stack.pop()
            for key, val in module._params.items():
                module._params[key] = fn(val)

            if recurse:
                for m in module._modules:
                    stack.append((m_prefix + m + ".", module._modules[m]))

    def parameters(self, recurse=True) -> Dict[str, Parameter]:
        """
        Returns all parameters associated with this Module and any nested Modules.
        Args:
            recurse: Recursively accesses all submodules associated with this module.
        """
        param_dict = {}
        stack = [("", self)]
        while stack:
            m_prefix, module = stack.pop()
            current_params = module._params
            current_params = {m_prefix + str(key): val for key, val in current_params.items()}
            param_dict = {**param_dict, **current_params}
            if recurse:
                for m in module._modules:
                    stack.append((m_prefix + m + ".", module._modules[m]))

        return param_dict

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Parameter):
            self.__dict__["_params"][__name] = __value
        elif isinstance(__value, Module):
            self.__dict__["_modules"][__name] = __value
        else:
            super().__setattr__(__name, __value)

    def __getattr__(self, __name: str) -> Any:
        if __name in self.__dict__["_params"]:
            return self.__dict__["_params"][__name]
        elif __name in self.__dict__["_modules"]:
            return self.__dict__["_modules"][__name]

        raise AttributeError(f"No attribute {__name} found in {self.__class__.__name__} module")
