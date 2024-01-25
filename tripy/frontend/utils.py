import functools


# Decorator to preprocess inputs of a function and convert numpy, python types to tripy tensors.
def convert_inputs_to_tensors():
    """
    Decorator that converts all arguments to Tensors before passing them along
    to the decorated function.
    """

    # NOTE: At some point we will need to make it so we can exclude arguments to convert.
    # To do so, we should add a new parameter: `exclude: List[str]` which contains the names
    # of arguments. Then, we can use `inspect.signature` to see which args or kwargs this corresponds
    # to.
    def impl(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from tripy.frontend.tensor import Tensor

            # Only convert args to tripy tensor. kwargs are allowed to be of non-tripy tensor type.
            new_args = [Tensor(arg) if not isinstance(arg, Tensor) else arg for arg in args]
            new_kwargs = {name: Tensor(arg) if not isinstance(arg, Tensor) else arg for name, arg in kwargs.items()}
            return func(*new_args, **new_kwargs)

        return wrapper

    return impl
