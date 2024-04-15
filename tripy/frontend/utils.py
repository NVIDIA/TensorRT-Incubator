import functools
from typing import List, Tuple
from tripy import utils
from tripy.logging import logger
from collections import OrderedDict


# Decorator to preprocess inputs of a function and convert numpy, python types to tripy tensors.
def convert_inputs_to_tensors(sync_arg_types: List[Tuple[str]] = None, exclude: List[str] = None):
    """
    Decorator that converts all arguments to Tensors before passing them along
    to the decorated function.

    Args:
        sync_arg_types: A list of tuples of strings indicating the parameter indices for parameters
            that must share a type. For example, `sync_arg_types=[("a", "b"), ("c", "d")]` indicates that
            arguments `a` and `b` and arguments `c` and `d` should have the same types. Type casting is only
            enabled for Python numbers, and at least one of the arguments in each tuple must be a Tensor.

        exclude: A list of names of arguments to skip over. These arguments will not be modified.

    """

    sync_arg_types = utils.default(sync_arg_types, [])
    exclude = utils.default(exclude, [])

    def impl(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from tripy.common.exception import raise_error
            from tripy.frontend.tensor import Tensor

            def add_column_info_for_non_tensor(arg, index, is_kwarg, dtype):
                assert not isinstance(arg, Tensor)
                arg = Tensor(arg, dtype=dtype)

                # Try to include correct column offsets for non-tensor arguments.
                try:
                    # This is the stack depth in arg.stack_info where we find the function
                    # that's decorated with `convert_inputs_to_tensors()`. If this changes, tests in test_utils.py should fail.
                    WRAPPER_STACK_DEPTH = 4
                    # Find the first caller of this function that is NOT the function registry.
                    # Also save the last dispatch target we see.
                    dispatch_target = None
                    for idx, source_info in enumerate(arg.stack_info[WRAPPER_STACK_DEPTH:]):
                        import tripy.function_registry

                        dispatch_target = source_info._dispatch_target or dispatch_target
                        if source_info.module != tripy.function_registry.__name__:
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

            # Merge args with kwargs, preserving order.
            all_args = OrderedDict(zip(func.__code__.co_varnames, args))
            all_args.update(kwargs)

            new_kwargs = {}
            for index, (name, arg) in enumerate(all_args.items()):
                if name in exclude or isinstance(arg, Tensor):
                    new_kwargs[name] = arg
                    continue

                cast_dtype = None
                for sync_tuple in sync_arg_types:
                    if name not in sync_tuple:
                        continue

                    for tensor_name in sync_tuple:
                        sync_tensor = all_args[tensor_name]
                        # If multiple Tensors exist in a tuple,
                        # leave the dtype check to frontend ops
                        if isinstance(all_args[tensor_name], Tensor):
                            break
                    else:
                        sync_args = {arg_name: all_args[arg_name] for arg_name in sync_tuple}
                        raise_error(
                            f"At least one of the arguments: {sync_tuple} must be a `tripy.Tensor`.",
                            [f"Got {sync_args}"],
                        )
                    cast_dtype = sync_tensor.dtype
                    break

                is_kwarg = name in kwargs
                new_kwargs[name] = add_column_info_for_non_tensor(arg, index, is_kwarg=is_kwarg, dtype=cast_dtype)

            return func(**new_kwargs)

        return wrapper

    return impl
