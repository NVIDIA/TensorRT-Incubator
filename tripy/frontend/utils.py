import functools
from tripy import utils
from tripy.logging import logger


# Decorator to preprocess inputs of a function and convert numpy, python types to tripy tensors.
# TODO(#135): Improve the type casting for python numbers
def convert_inputs_to_tensors(allow_type_casting=False):
    """
    Decorator that converts all arguments to Tensors before passing them along
    to the decorated function.

    Args:
        allow_type_casting: If true, cast python numbers arguments (if any) to
            the first Tensor argument's dtype.
    """

    # NOTE: At some point we will need to make it so we can exclude arguments to convert.
    # To do so, we should add a new parameter: `exclude: List[str]` which contains the names
    # of arguments. Then, we can use `inspect.signature` to see which args or kwargs this corresponds
    # to.
    def impl(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from tripy.frontend.tensor import Tensor

            def add_column_info_for_non_tensor(arg, index, is_kwarg, dtype):
                if isinstance(arg, Tensor):
                    return arg

                arg = Tensor(arg, dtype=dtype)

                # Try to include correct column offsets for non-tensor arguments.
                try:
                    # This is the stack depth in arg.stack_info where we find the function
                    # that's decorated with `convert_inputs_to_tensors()`. If this changes, tests in test_utils.py should fail.
                    WRAPPER_STACK_DEPTH = 5
                    # Find the first caller of this function that is NOT the function registry.
                    # Also save the last dispatch target we see.
                    dispatch_target = None
                    for idx, source_info in enumerate(arg.stack_info[WRAPPER_STACK_DEPTH:]):
                        dispatch_target = source_info._dispatch_target or dispatch_target
                        if source_info.module != "tripy.utils.function_registry":
                            frame_index = idx + WRAPPER_STACK_DEPTH
                            break
                    else:
                        # Fallback path is just to look at the user code
                        frame_index = arg.stack_info.get_first_user_frame_index()
                        dispatch_target = arg.stack_info[frame_index - 1]._dispatch_target

                    source_info = arg.stack_info[frame_index]

                    # The reverse binary ops need special handling since they will be swapped out for the non-reverse
                    # variants and the order of operands will be inverted.
                    REVERSE_BIN_OPS = {
                        "__radd__",
                        "__rmul__",
                        "__rsub__",
                        "__rpow__",
                        "__rtruediv__",
                    }

                    if dispatch_target in REVERSE_BIN_OPS:
                        assert index in [0, 1]
                        index = 0 if index == 1 else 1
                        dispatch_target = dispatch_target.replace("__r", "__")

                    candidates = utils.get_arg_candidate_column_offsets(
                        source_info.code,
                        index,
                        len(args),
                        dispatch_target if dispatch_target else func.__name__,
                        is_kwarg,
                    )

                    # Only set column range if there is exactly one candidate, otherwise we can't reliably determine
                    # what we should be pointing at.
                    if len(candidates) == 1:
                        source_info.column_range = candidates[0]
                except Exception as e:
                    logger.verbose(
                        f"Exception while trying to determine column offsets in 'convert_inputs_to_tensors': {e}"
                    )
                    pass

                return arg

            cast_dtype = args[0].dtype if allow_type_casting and isinstance(args[0], Tensor) else None

            new_args = [
                add_column_info_for_non_tensor(arg, index, is_kwarg=False, dtype=cast_dtype)
                for index, arg in enumerate(args)
            ]

            new_kwargs = {
                name: add_column_info_for_non_tensor(arg, index, is_kwarg=True, dtype=cast_dtype)
                for index, (name, arg) in enumerate(kwargs.items())
            }
            return func(*new_args, **new_kwargs)

        return wrapper

    return impl
