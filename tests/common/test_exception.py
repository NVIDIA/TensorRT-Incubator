from dataclasses import dataclass
from textwrap import dedent

from tests import helper
from tripy.common.exception import TripyException, _make_stack_info_message, raise_error
from tripy.utils import StackInfo, get_stack_info
from tripy.utils.stack_info import SourceInfo


@dataclass
class ObjWithStackInfo:
    stack_info: StackInfo


class TestRaiseError:
    def test_obj_with_stack_info(self):
        obj = ObjWithStackInfo(get_stack_info())
        # Fow now we'll just test that the file name is included in the error message.
        # This proves that the stack info is being accessed.
        with helper.raises(TripyException, match=f"{__file__}"):
            raise_error("Test message", details=[obj])

    def test_can_determine_column_range(self):
        # This is derived from a simple expression:
        # a = tp.zeros((2, 3)) - tp.ones((2, 4))
        stack_info = StackInfo(
            [
                SourceInfo(
                    module="tripy.frontend.tensor",
                    file="/tripy/tripy/frontend/tensor.py",
                    line=52,
                    function="_finalize",
                    code="",
                    _dispatch_target="",
                ),
                SourceInfo(
                    module="tripy.frontend.tensor",
                    file="/tripy/tripy/frontend/tensor.py",
                    line=74,
                    function="build",
                    code="",
                    _dispatch_target="",
                ),
                SourceInfo(
                    module="tripy.frontend.ops.binary_elementwise",
                    file="/tripy/tripy/frontend/ops/binary_elementwise.py",
                    line=175,
                    function="sub",
                    code="",
                    _dispatch_target="",
                ),
                SourceInfo(
                    module="tripy.frontend.utils",
                    file="/tripy/tripy/frontend/utils.py",
                    line=23,
                    function="wrapper",
                    code="            return func(*new_args, **new_kwargs)",
                    _dispatch_target="",
                ),
                SourceInfo(
                    module="tripy.utils.function_registry",
                    file="/tripy/tripy/utils/function_registry.py",
                    line=143,
                    function="__call__",
                    code="        return self.func(*args, **kwargs)",
                    _dispatch_target="",
                ),
                SourceInfo(
                    module="tripy.utils.function_registry",
                    file="/tripy/tripy/utils/function_registry.py",
                    line=237,
                    function="wrapper",
                    code="                        return self.find_overload(key, args, kwargs)(*args, **kwargs)",
                    _dispatch_target="__sub__",
                ),
                SourceInfo(
                    module="__main__",
                    file="/tripy/tmp.py",
                    line=3,
                    function="<module>",
                    code="a = tp.zeros((2, 3)) - tp.ones((2, 4))",
                    _dispatch_target="",
                ),
            ]
        )

        error_msg = _make_stack_info_message(stack_info, enable_color=False)
        assert (
            dedent(
                """
                --> /tripy/tmp.py:3
                     |
                   3 | a = tp.zeros((2, 3)) - tp.ones((2, 4))
                     |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                """
            ).strip()
            in dedent(error_msg).strip()
        )
