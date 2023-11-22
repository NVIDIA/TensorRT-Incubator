from tripy.frontend import Dim
import pytest


class TestDim:
    @pytest.mark.parametrize("range", [2, 3])
    def test_repr(self, range):
        dim = Dim(2, max=range)
        assert repr(dim) == f"Dim(runtime_value=2, min=2, opt=2, max={range})"
        eval_dim = eval(repr(dim))
        assert eval_dim.min == 2 and eval_dim.opt == 2 and eval_dim.max == range and eval_dim.runtime_value == 2

    @pytest.mark.parametrize("dim_range", [{"min": 3, "max": 5}, {"min": 3, "max": 5, "opt": 4}])
    def test_profile_dict(self, dim_range):
        dim = Dim(runtime_value=3, **dim_range)
        assert dim.min == dim_range["min"]
        assert dim.max == dim_range["max"]
        assert dim.opt == dim_range.get("opt", int((dim_range["max"] + dim_range["min"]) / 2.0))

    def test_profile_out_of_range(self):
        with pytest.raises(AssertionError):
            _ = Dim(2, min=3)

    def test_validate_set_runtime_value(self):
        dim = Dim(runtime_value=3, min=2, opt=4, max=6)
        dim.runtime_value = 5

    def test_validate_rejects_invalid_runtime_value(self):
        with pytest.raises(AssertionError):
            dim = Dim(runtime_value=3, min=2, opt=4, max=6)
            dim.runtime_value = 1

    # Verify that read only property can not be set
    def test_rejects_setting_dim_range(self):
        with pytest.raises(AttributeError):
            dim = Dim(2)
            dim.min = 4
            dim.opt = 4
            dim.max = 4

    def test_incomplete_profile(self):
        with pytest.raises(AssertionError):
            _ = Dim(2, opt=3)
