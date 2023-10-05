import sys

import tripy.util

# Internal-only imports
from tripy.util.stack_info import SourceInfo


def test_get_stack_info():
    def func():
        # Make sure these two lines remain adjacent since we need to know the offset to use for the line number.
        expected_line_num = sys._getframe().f_lineno + 1
        return tripy.util.get_stack_info(), expected_line_num

    # Make sure these two lines remain adjacent since we need to know the offset to use for the line number.
    expected_outer_line_num = sys._getframe().f_lineno + 1
    stack_info, expected_inner_line_num = func()

    assert stack_info[0] == SourceInfo(__name__, file=__file__, line=expected_inner_line_num, function=func.__name__)
    assert stack_info[1] == SourceInfo(
        __name__, file=__file__, line=expected_outer_line_num, function=test_get_stack_info.__name__
    )
