from tripy.common.exception import TripyException, raise_error

from tripy.utils import get_stack_info, StackInfo
from dataclasses import dataclass
import pytest
import tripy as tp


class TestRaiseError:
    def test_obj_with_stack_info(self):
        @dataclass
        class ObjWithStackInfo:
            stack_info: StackInfo

        obj = ObjWithStackInfo(get_stack_info())
        # Fow now we'll just test that the file name is included in the error message.
        # This proves that the stack info is being accessed.
        with pytest.raises(TripyException, match=f"{__file__}"):
            raise_error("Test message", details=[obj])

    def test_datatype(self):
        # Data types should print human-readable names instead of something cryptic like `<class 'abc.float32'>`
        with pytest.raises(TripyException, match="Error:\n    'float32'"):
            raise_error("Error:", details=[tp.float32])