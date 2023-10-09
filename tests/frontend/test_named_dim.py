from tripy.frontend import NamedDim
import pytest


class TestNamedDim:
    @pytest.mark.parametrize("range", [2, 3])
    def test_named_dim_repr(self, range):
        dim = NamedDim("dim", 2, max=range)
        assert repr(dim) == f'NamedDim(name="dim", runtime_value=2, min=2, opt=2, max={range})'
        eval_dim = eval(repr(dim))
        assert eval_dim.min == 2 and eval_dim.opt == 2 and eval_dim.max == range and eval_dim.runtime_value == 2

    @pytest.mark.parametrize("dim_range", [{"min": 3, "max": 5}, {"min": 3, "max": 5, "opt": 4}])
    def test_named_dim_profile_dict(self, dim_range):
        dim = NamedDim("dim", runtime_value=3, **dim_range)
        assert dim.min == dim_range["min"]
        assert dim.max == dim_range["max"]
        assert dim.opt == dim_range.get("opt", int((dim_range["max"] + dim_range["min"]) / 2.0))

    @pytest.mark.parametrize("runtime_value", [2])
    @pytest.mark.parametrize("dim_range", [(3)])
    def test_named_dim_profile_out_of_range(self, runtime_value, dim_range):
        with pytest.raises(AssertionError):
            dim = NamedDim("dim", runtime_value, min=3)

    def test_named_dim_validate_set_runtime_value(self):
        dim = NamedDim("dim", runtime_value=3, min=2, opt=4, max=6)
        dim.runtime_value = 5

    def test_named_dim_validate_rejects_invalid_runtime_value(self):
        with pytest.raises(AssertionError):
            dim = NamedDim("dim", runtime_value=3, min=2, opt=4, max=6)
            dim.runtime_value = 1

    # Verify that read only property can not be set
    def test_named_dim_rejects_setting_dim_range(self):
        with pytest.raises(AttributeError):
            dim = NamedDim("dim", 2)
            dim.min = 4
            dim.opt = 4
            dim.max = 4

    def test_incomplete_profile(self):
        with pytest.raises(AssertionError):
            dim = NamedDim("d", 2, opt=3)
