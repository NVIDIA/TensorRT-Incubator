import functools
from tripy import utils


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

            def update_arg(arg, index, is_kwarg):
                if not isinstance(arg, Tensor):
                    arg = Tensor(arg)

                    # Try to include correct column offsets for non-tensor arguments.
                    try:
                        user_frame_index = arg._stack_info.get_first_user_frame_index()
                        callee_info = arg._stack_info[user_frame_index - 1]
                        source_info = arg._stack_info[user_frame_index]

                        source_info.column_range = utils.get_arg_column_offset(
                            source_info.code,
                            index,
                            len(args),
                            callee_info._dispatch_target if callee_info._dispatch_target else func.__name__,
                            is_kwarg,
                        )
                    except:
                        pass

                return arg

            new_args = [update_arg(arg, index, is_kwarg=False) for index, arg in enumerate(args)]
            new_kwargs = {
                name: update_arg(arg, index, is_kwarg=True) for index, (name, arg) in enumerate(kwargs.items())
            }
            return func(*new_args, **new_kwargs)

        return wrapper

    return impl
