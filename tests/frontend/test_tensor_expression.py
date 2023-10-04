from tripy.frontend import TensorExpression
import pytest

# Internal-only imports
from tripy.frontend.parameters import ValueParameters, BinaryElementwiseParameters


class TestTensorExpression:
    def test_tensor(self):
        VALUES = [1, 2, 3]
        a = TensorExpression.tensor(VALUES)

        assert isinstance(a, TensorExpression)
        assert a.inputs == []
        assert isinstance(a.params, ValueParameters)
        assert a.params.values == VALUES

    @pytest.mark.parametrize("func, operation", [(lambda a, b: a + b, BinaryElementwiseParameters.Operation.SUM)])
    def test_binary_elementwise(self, func, operation):
        a = TensorExpression.tensor([1])
        b = TensorExpression.tensor([2])

        out = func(a, b)
        assert isinstance(out, TensorExpression)
        assert isinstance(out.params, BinaryElementwiseParameters)
        assert out.params.operation == operation
