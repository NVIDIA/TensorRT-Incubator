from typing import Any, Dict

from tripy.frontend.nn.parameter import Parameter


class Module:
    """
    Base class for neural network modules.

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

        assert (out.eval().cpu_view(np.float32) == np.array([2.0, 2.0])).all()
    """

    _params: Dict[str, Any]
    _modules: Dict[str, Any]

    def __init__(self):
        self._params = {}
        self._modules = {}

    def __repr__(self) -> str:
        pass

    def save_weights(self, path: str):
        """
        Save Module parameters to the specified path.

        Args:
            path: The path at which to save weights.
        """
        pass

    def load_weights(self, path: str):
        """
        Load Module parameters from the specified path.

        Args:
            path: The path from which to load weights.
        """
        pass

    def parameters(self) -> Dict[str, Any]:
        """
        Returns all parameters associated with this Module and any nested Modules.
        """
        pass

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

        raise AttributeError(f"No attribute {__name} found.")
