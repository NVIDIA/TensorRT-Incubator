from tripy.utils import Result

from tests import helper


class TestResult:
    def test_cannot_retrieve_value_of_error(self):
        result: Result[int] = Result.err(["error!"])
        with helper.raises(AssertionError):
            result.value

    def test_cannot_retrieve_error_details_of_ok(self):
        result: Result[int] = Result.ok(0)
        with helper.raises(AssertionError):
            result.error_details
