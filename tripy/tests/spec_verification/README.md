# Introduction

Spec verification is designed to ensure that the datatypes documented in the operator documentation are accurate.

# How to Verify an Operation

To run the verification program on an operation, add the decorator `@constraints.dtype_info` to the operation. The inputs to the decorator will help the verifier determine the constraints on the inputs and the datatypes to verify.

To learn more about how to use `@constraints.dtype_info` check out `tripy/constraints.py` and `tests/spec_verification/test_dtype_constraints.py`.

After the decorator is set up, it will automatically run verification test cases alongside the regular test cases. If you only want to run the verifier, execute `pytest -s -v` within the tests/spec_verification folder.
