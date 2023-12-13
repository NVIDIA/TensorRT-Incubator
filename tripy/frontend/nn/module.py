from typing import Any, Dict

from tripy.frontend.tensor import Tensor
from tripy.common.exception import TripyException


class Module:
    """Base class used to create all neural network modules.
    Module class allows accessing all the parameters associated within nested Modules.
    Example:
    ::
        import numpy as np
        from tripy.frontend.nn.module import Module

        class Network(Module):
            def __init__(self):
                super().__init__()
                from tripy.frontend.nn.module import Parameter
                from tripy.frontend import Tensor
                import numpy as np
                self.param = Parameter(Tensor(np.ones([2])))

            def __call__(self, x):
                return x + self.param

        net = Network()
        def infer(net):
            from tripy.frontend import Tensor
            import numpy as np
            x = Tensor(np.ones(2))
            return net(x)

        assert(infer(net).eval() == np.array([2.0, 2.0])).all()
    """

    _params: Dict[str, Any]
    _modules: Dict[str, Any]

    def __init__(self):
        self._params = {}
        self._modules = {}

    def __repr__(self) -> str:
        return f"Module({self._modules}{self._params})"

    def save_weights(self, file_name: str):
        """Save Module parameters to file."""
        pass

    def load_weights(self, file_name: str):
        """Load Module parameters from file."""
        pass

    def parameters(self):
        """Returns the list of all parameters associated with this Module and parameters associated within nested Modules"""
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


# todo: add metaclass to allow isintance(cls, Parameter) to return true if object is Tensor.
class Parameter(Tensor):
    """Parameters are regular tripy tensors along with an extra attribute that helps the underlying
    compiler to optimize the network for performance.
    """

    def __new__(cls, data=None):
        if data is None:
            # Create an empty tensor
            assert False, "tripy does not support creating an empty tensor as of 12/12/2023"
        data._is_param = True
        return data

    def __repr__(self) -> str:
        return repr(self.val)

    def __str__(self) -> str:
        return str(self.val)
