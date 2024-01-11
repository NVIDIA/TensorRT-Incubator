import numpy as np
import pytest

from tripy import Tensor
from tripy import device, jit
from tripy.frontend.nn import Module, Parameter, Linear


class TestLinear:
    def test_linear_params(self):
        linear = Linear(20, 30)
        assert isinstance(linear, Linear)
        params = linear.parameters()
        assert len(params.keys()) == 2
        assert params["weight"].numpy().shape == (30, 20)
        assert params["bias"].numpy().shape == (1, 30)
