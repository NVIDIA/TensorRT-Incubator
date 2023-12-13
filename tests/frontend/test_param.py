from typing import Any
from tripy import Tensor, jit
from tripy.frontend.nn import Module, Parameter

import numpy as np
import pytest


class TestParam:
    def test_param_const_fold(self):
        param = Parameter(Tensor([1, 2, 3]))

        @jit
        def func(param):
            return param + param

        ret = func(param)

        # Check that param is const folded into jitted function.
        assert func._const_args == (0,)
