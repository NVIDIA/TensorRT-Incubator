from tripy.frontend.tensor import Tensor, TensorMeta


class ParamMeta(TensorMeta):
    def __instancecheck__(self, instance):
        # Return True if the instance is an instance of the parent Tensor class
        return super().__instancecheck__(instance) or (
            isinstance(instance, Tensor) and getattr(instance, "_is_param", False)
        )


class Parameter(Tensor, metaclass=ParamMeta):
    """
    A Parameter is a special kind of :class:`tripy.Tensor` that treated by the compiler as a
    constant, allowing for additional optimization opportunities.

    Example:
    ::

        param = tp.nn.Parameter(tp.Tensor([1.0, 1.0], dtype=tp.float32))
        print(param)
        assert isinstance(param, tp.nn.Parameter)
        assert isinstance(param, tp.Tensor)
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
