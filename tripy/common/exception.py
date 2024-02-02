import ast
import inspect
from textwrap import indent
from typing import Any, List, Tuple

from colored import Fore, attr


class TripyException(Exception):
    pass


# Grab column offsets for a given frame based on information from its callee.
def _get_candidate_column_offsets(cur_frame, callee):
    def get_callee_func_name():
        callee_name = callee.function
        # Some functions (e.g. tensor methods) are routed through a function registry.
        # We don't actually care about the dispatch function, so we look at the `key`
        # to determine which underlying method we're actually calling.
        if callee._dispatch_target:
            callee_name = callee._dispatch_target
        return callee_name

    callee_name = get_callee_func_name()

    candidate_column_offsets = []

    # Need to dedent before parsing, so save indent.
    raw_code = cur_frame.code
    code = raw_code.lstrip()
    indentation = len(raw_code) - len(code)

    parsed_ast = ast.parse(code)
    for node in ast.walk(parsed_ast):

        def get_ast_node_func_name(node):
            if isinstance(node, ast.BinOp):
                MAPPING = {
                    ast.Add: "__add__",
                    ast.Sub: "__sub__",
                    ast.Mult: "__mul__",
                    ast.Div: "__truediv__",
                    ast.FloorDiv: "__floordiv__",
                    ast.Mod: "__mod__",
                    ast.Pow: "__pow__",
                    ast.BitAnd: "__and__",
                    ast.BitOr: "__or__",
                    ast.BitXor: "__xor__",
                    ast.LShift: "__lshift__",
                    ast.RShift: "__rshift__",
                }
                return MAPPING.get(type(node.op))

            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Call):
                    return get_ast_node_func_name(func)

                if isinstance(func, ast.Attribute):
                    return func.attr

                return func.id
            return None

        try:
            ast_node_name = get_ast_node_func_name(node)
        except:
            continue

        if ast_node_name is None:
            continue

        def check_name_matches():
            # We need special checking for __init__ methods since the AST node will just be the class name, e.g. `Tensor`.
            if callee_name != "__init__":
                return ast_node_name == callee_name

            # We hardcode names of some common classes here to avoid creating an import dependency:
            if ast_node_name in {"Tensor"}:
                return True
            return False

        # Since there could be multiple different function calls on the same line, we use the callee name
        # to determine which one(s) to look at.
        if check_name_matches():
            candidate_column_offsets.append((indentation + node.col_offset, indentation + node.end_col_offset))

    return candidate_column_offsets


def _make_stack_info_message(stack_info: "utils.StackInfo", enable_color: bool = True) -> str:
    import tripy.utils.function_registry
    from tripy.frontend.utils import convert_inputs_to_tensors

    EXCLUDE_FUNCTIONS = [convert_inputs_to_tensors]

    def should_exclude(frame):
        for func in EXCLUDE_FUNCTIONS:
            filename = inspect.getsourcefile(func)
            lines, start_line = inspect.getsourcelines(func)

            if frame.file != filename:
                return False

            if frame.line < start_line or frame.line > (start_line + len(lines)):
                return False
            return True

    def apply_color(inp, color):
        if not enable_color:
            return inp
        return f"{color}{inp}{attr('reset')}"

    frame_strs = []
    num_frames_printed = 0
    for index, frame in enumerate(stack_info):
        if not frame.code:
            continue

        if frame.module == tripy.utils.function_registry.__name__:
            continue

        if should_exclude(frame):
            continue

        line_info = f"{apply_color(frame.file, Fore.yellow)}:{frame.line}"

        line_no = f"{frame.line:>4} "
        indent = " " * len(line_no)

        frame_info = ""
        if num_frames_printed == 0:
            frame_info += "\n\n"

        frame_info += f"--> {line_info}\n{indent}|\n{line_no}| {frame.code}"

        column_range = None
        if index > 0:
            # With multiple calls to the same function name on the same line,
            # it is not possible for us to determine which column offset is correct, so we
            # won't include it in that case.
            try:
                candidate_column_offsets = _get_candidate_column_offsets(frame, stack_info[index - 1])
            except:
                pass
            else:
                if len(candidate_column_offsets) == 1:
                    column_range = candidate_column_offsets[0]

        if column_range:
            start, end = column_range
            size = end - start
            frame_info += f"\n{indent}| " + " " * start + apply_color("^" * (size), Fore.red)
            if num_frames_printed > 0:
                frame_info += " --- required from here"
        else:
            if num_frames_printed > 0:
                frame_info = "Required from:\n" + frame_info
        frame_info += "\n\n"
        frame_strs.append(frame_info)
        num_frames_printed += 1

    if frame_strs:
        return "".join(frame_strs)
    return "\n\n<No stack information available>\n\n"


def raise_error(summary: str, details: List[Any] = []):
    """
    Raises a Tripy exception with a formatted message.

    Args:
        summary: A summary of the error message. This will be displayed before any other details.
        details: Details on the error. This function handles objects in this list as follows:
            - If they include a `stack_info` member, then information on the first user frame is displayed,
                including file/line information as well as the line of code.

                IMPORTANT: Any stack frames from the function registry are not displayed since
                the function registry is an implementation detail used to dispatch to the real functions
                we care about. Additionally, any code defined in the functions listed in ``EXCLUDE_FUNCTIONS``
                is omitted.

            - In all other cases, the object is just converted to a string.

    Raises:
        TripyException
    """
    detail_msg = ""
    for detail in details:
        if hasattr(detail, "stack_info"):
            detail_msg += _make_stack_info_message(detail.stack_info)
        else:
            detail_msg += str(detail)

    msg = f"{summary}\n" + indent(detail_msg, " " * 4)
    raise TripyException(msg)


def search_for_missing_attr(module_name: str, name: str, look_in: List[Tuple[Any, str]]):
    """
    Searches for an attribute in the given modules/objects and then raises an AttributeError
    including a hint on where to find the attribute.

    This is intended to be called from a module/submodule-level `__getattr__` override.

    Args:
        module_name: The name of the current module or submodule.
        name: The name of the attribute we are searching for.
        look_in: A list containing elements of (obj/submodule, obj_name) to search under.

    Raises:
        AttributeError: This will potentially include a hint if the attribute was found under
            one of the objects in look_in.
    """
    import inspect

    # We look at the call stack to prevent infinite recursion.
    # Consider the case where `tripy` searches under `tripy.nn` and vice-versa.
    # If the attribute is not present in either, they will keep ping-ponging back
    # and forth since `hasattr` in this function will call `__getattr__` which will
    # then call `search_for_missing_attr` ad infinitum.
    stack = inspect.stack()

    stack_modules = []
    stack_classes = []
    for frame in stack:
        module = inspect.getmodule(frame.frame)
        if module:
            stack_modules.append(module)

        self_arg = frame.frame.f_locals.get("self")
        if self_arg is not None:
            try:
                class_type = self_arg.__class__
            except:
                pass
            else:
                stack_classes.append(class_type)

    stack_modules = list(filter(lambda mod: mod is not None, [inspect.getmodule(frame.frame) for frame in stack]))
    stack_classes = list([])

    msg = f"Module: {module_name} does not have attribute: {name}. "
    # If a symbol isn't found in the top-level, we'll look at specific classes/modules
    # in case there's a similar symbol there.
    # We provide the names as well since the object name will be the fully qualified name,
    # which is not necessarily what the user uses (e.g. tripy.nn vs. tripy.frontend.nn).

    for obj, obj_name in look_in:
        # Avoid infinite recursion - see comment above.
        if obj in stack_modules + stack_classes:
            continue

        if hasattr(obj, name):
            msg += f"Did you mean: '{obj_name}.{name}'?"
            break

    raise AttributeError(msg.strip())
