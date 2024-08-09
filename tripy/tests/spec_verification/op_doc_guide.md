# Introduction

Spec verification is designed to ensure that the datatypes documented in the operator documentation are accurate.

# How to Verify an Operation

To run the verification program on an operation, add the decorator `@dtype_info.dtype_info` to the operation. The inputs to the decorator will help the verifier determine the constraints on the inputs and the datatypes to verify.

There are four optional inputs for the decorator `dtype_variables`, `dtype_constraints`, `param_type_specification`, and `function_name`. To learn more about how these work go to `tripy/dtype_info.py`

If your function requires setting up some input variables you can do so by adding an element to `default_constraints_all` dictionary within `tests/spec_verification/test_dtype_constraints.py`. You can find more information about how `default_constraints_all` works withing `tests/spec_verification/test_dtype_constraints.py`.

After the decorator is set up, it will automatically run verification test cases alongside the regular test cases. If you only want to run the verifier, execute `pytest -s -v` within the tests/spec_verification folder.

Currently `int4` and `float8` are not being verified since int4 can not be printed/evaled and float8 has exceptions with some functions that interfer with verification.
