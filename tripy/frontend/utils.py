import functools
import inspect
from typing import List, Optional, Sequence, Tuple, Union

from tripy import utils
from tripy.frontend.trace.ops import BaseTraceOp
from tripy.flat_ir.ops import BaseFlatIROp


# Decorator to preprocess inputs of a function and convert numpy, python types to tripy tensors.
def convert_inputs_to_tensors(
    sync_arg_types: Optional[List[Tuple[str]]] = None,
    exclude: Optional[List[str]] = None,
    unpack_argument: Optional[List[str]] = None,
    skip_num_stack_entries: int = 0,
):
    """
    Decorator that converts all arguments to Tensors before passing them along
    to the decorated function.

    Args:
        sync_arg_types: A list of tuples of strings indicating the parameter indices for parameters
            that must share a type. For example, `sync_arg_types=[("a", "b"), ("c", "d")]` indicates that
            arguments `a` and `b` and arguments `c` and `d` should have the same types. Type casting is only
            enabled for Python numbers, and at least one of the arguments in each tuple must be a Tensor.
            For arguments that are lists included in `convert_lists`,
            the syncing will be done for each member of the specified lists.

        exclude: A list of names of arguments to skip over. These arguments will not be modified.

        unpack_argument: If an argument name is included and it is a list,
          the members of the list will each be individually converted into tensors,
          rather than having the whole list converted into a tensor.

        skip_num_stack_entries: If the decorator is used on a function that is *called by*
            a function that the user invokes, it will be necessary to skip stack entries
            in order to get the column info from the user code. The number of entries skipped
            should be equal to the nesting depth from a function called by user code
            (if the decorated function is called by the user the depth is 0;
            if the decorated function is called from a user function, the depth is 1; etc.)
    """

    sync_arg_types = utils.default(sync_arg_types, [])
    exclude = utils.default(exclude, [])
    unpack_argument = utils.default(unpack_argument, [])

    def impl(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from tripy.common.exception import raise_error
            from tripy.frontend.tensor import Tensor

            # Try to include correct column offsets for non-tensor arguments.
            def add_column_info_for_non_tensor(arg, arg_index, is_kwarg, dtype, list_index=None):
                assert not isinstance(arg, Tensor)
                arg = Tensor(arg, dtype=dtype)

                # This is the stack depth in arg.stack_info where we find the function
                # that's decorated with `convert_inputs_to_tensors()`.
                for idx, source_info in enumerate(arg.stack_info):
                    if source_info.function == "wrapper" and source_info.module == __name__:
                        WRAPPER_STACK_DEPTH = idx + 1
                        break
                else:
                    assert (
                        False
                    ), "`wrapper` function was not found in the call stack. Please update the check above if the name of the wrapper function has changed."

                # Find the first caller of this function that is NOT the function registry.
                # Also save the last dispatch target we see.
                dispatch_target = None
                for idx, source_info in enumerate(arg.stack_info[WRAPPER_STACK_DEPTH + skip_num_stack_entries :]):
                    import tripy.function_registry

                    dispatch_target = source_info._dispatch_target or dispatch_target
                    if source_info.module != tripy.function_registry.__name__:
                        frame_index = idx + WRAPPER_STACK_DEPTH + skip_num_stack_entries
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
                    assert arg_index in [0, 1]
                    arg_index = 0 if arg_index == 1 else 1
                    dispatch_target = dispatch_target.replace("__r", "__")

                # Special case for __getitem__: It is variadic. Argument 0 is the tensor, argument 1 is the indices,
                # and all subsequent arguments are slice parameters (in start, stop, step order).
                # Hence, we subtract two to get the index of the slice parameters
                if dispatch_target == "__getitem__":
                    arg_index -= 2

                candidates = utils.get_arg_candidate_column_offsets(
                    source_info.code,
                    arg_index,
                    len(args),
                    dispatch_target if dispatch_target else func.__name__,
                    is_kwarg,
                    list_index=list_index,
                )

                # Only set column range if there is exactly one candidate, otherwise we can't reliably determine
                # what we should be pointing at.
                if len(candidates) == 1:
                    source_info.column_range = candidates[0]

                return arg

            # Merge positional and keyword arguments, trying to determine names where possible.
            # In the case of variadic positional arguments, we cannot determine names, so we use
            # None instead.
            signature = inspect.signature(func)
            arg_names = []
            for name, param in signature.parameters.items():
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    # Positional arguments cannot follow variadic positional arguments
                    # (they would just be absorbed into the variadic argument).
                    break

                arg_names.append(name)

            arg_names += [None] * len(args)
            all_args = list(zip(arg_names, args))
            all_args.extend(kwargs.items())

            def get_arg(name: str):
                for arg_name, arg in all_args:
                    if name == arg_name:
                        return arg

                assert (
                    False
                ), "Cannot retrieve unnamed argument. This could be because the argument is a variadic argument."

            new_args = []
            new_kwargs = {}
            for index, (name, arg) in enumerate(all_args):

                def add_arg(arg):
                    if name not in kwargs:
                        new_args.append(arg)
                    else:
                        new_kwargs[name] = arg

                def find_sync_target_dtype(arg_name):

                    for sync_tuple in sync_arg_types:
                        if arg_name not in sync_tuple:
                            continue

                        for tensor_name in sync_tuple:
                            sync_target = get_arg(tensor_name)
                            # If multiple Tensors exist in a tuple,
                            # leave the dtype check to frontend ops

                            if isinstance(sync_target, Sequence):
                                for member in sync_target:
                                    if isinstance(member, Tensor):
                                        return member.dtype
                            elif isinstance(sync_target, Tensor):
                                return sync_target.dtype
                        else:
                            sync_args = {sync_arg_name: get_arg(sync_arg_name) for sync_arg_name in sync_tuple}
                            raise_error(
                                f"At least one of the arguments: {sync_tuple} must be a `tripy.Tensor`.",
                                [f"Got {sync_args}"],
                            )
                    return None

                def convert_nontensor_arg(arg, list_index=None):
                    cast_dtype = find_sync_target_dtype(name)
                    return add_column_info_for_non_tensor(
                        arg, index, is_kwarg=name in kwargs, dtype=cast_dtype, list_index=list_index
                    )

                if name in exclude or isinstance(arg, Tensor):
                    add_arg(arg)
                    continue
                if name in unpack_argument and isinstance(arg, Sequence):
                    new_list = [
                        member if isinstance(member, Tensor) else convert_nontensor_arg(member, list_index=i)
                        for i, member in enumerate(arg)
                    ]
                    if isinstance(arg, tuple):
                        new_list = tuple(new_list)
                    add_arg(new_list)
                    continue
                add_arg(convert_nontensor_arg(arg))

            return func(*new_args, **new_kwargs)

        return wrapper

    return impl


def topological_sort(ops: List[Union[BaseTraceOp, BaseFlatIROp]]) -> List[Union[BaseTraceOp, BaseFlatIROp]]:
    """
    This utility to topologically sort a graph that can be a Trace or a FlatIR graph.
    """
    stack = list()
    visited_layer_ids = set()
    id_ops = [id(op) for op in ops]

    def add_to_stack(op, stack):
        visited_layer_ids.add(id(op))
        for ip in op.inputs:
            if ip.producer is not None and id(ip.producer) not in visited_layer_ids and id(ip.producer) in id_ops:
                add_to_stack(ip.producer, stack)

        stack.append(op)

    for op in ops:
        if id(op) not in visited_layer_ids:
            add_to_stack(op, stack)

    assert len(ops) == len(
        stack
    ), f"Num original ops {len(ops)}, got num {len(stack)}, {len(set([id(op) for op in ops]))}"
    return stack
