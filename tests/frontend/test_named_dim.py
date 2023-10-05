from tripy.frontend import NamedDim
import pytest


class TestNamedDim:

    def test_named_dim_repr(self):
        dim = NamedDim("dim", (2,3,4))
        assert repr(dim) == "NamedDim(name=\"dim\", dim_range=(2, 3, 4))"
        eval_dim = eval(repr(dim))
        assert eval_dim.min == 2 and eval_dim.opt == 3 and eval_dim.max == 4

    @pytest.mark.parametrize("shape", [2, (2, 3), (2, 3, 4)])
    def test_named_dim_profile(self, shape):
        dim = NamedDim("dim", shape)
        assert dim.min == shape if isinstance(shape, int) else shape[0]
        assert dim.opt == shape if isinstance(shape, int) else shape[1]
        assert dim.max == shape if isinstance(shape, int) else shape[-1]

    @pytest.mark.parametrize("shape", [{2}, (2, 3, 5, 6), 3.4])
    def test_named_dim_rejects_invalid_shape(self, shape):
        with pytest.raises(ValueError):
            dim = NamedDim("dim", shape)

    # Verify that read only property can not be set
    def test_named_dim_rejects_setting_dim_range(self):
        with pytest.raises(AttributeError):
            dim = NamedDim("dim", 2)
            dim.min = 4
            dim.opt = 4
            dim.max = 4
