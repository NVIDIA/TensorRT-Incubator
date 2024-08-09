# Introduction

Spec verification is designed to ensure that the datatypes documented in the operator documentation are accurate.

# How to Verify an Operation

To run the verification program on an operation, add the decorator @dtype_info.dtype_info to the operation. The inputs to the decorator will help the verifier determine the constraints on the inputs and the datatypes to verify.

There are four optional inputs for the decorator:

 -  `dtype_variables`: This input must be a dictionary with the names of groups of variables as the keys and lists of datatypes as the values. Example: dtype_variables={"T": ["float32", "float16", "int8", "int32", "int64", "bool"], "T1": ["int32"]}. Any datatype not included will be tested to ensure it fails the test cases.
 - `dtype_constraints`: This input assigns inputs and return parameters to variable groups. It must be a dictionary with parameter names as keys and variable group names as values. For assigning the return value, the key must be dtype_info.RETURN_VALUE. Example: dtype_constraints={"input": "T", "index": "T1", dtype_info.RETURN_VALUE: "T"}.. 
 - `param_type_specification`: This parameter addresses situations where the type hint is not defined or linked to an internal object builder. It also allows the verifier to use a type other than the first option in a Union type hint. Example: `param_type_specification={"self": "tripy.Tensor"}`. Here is a list of type hints that are currently being used: `"tripy.Tensor", "tripy.Shape", Sequence[int], numbers.Number, int, "tripy.dtype", datatype.dtype, Tuple, List[Union["tripy.Tensor"]], "tripy.device", bool, float`
 - `function_name`: This parameter is only needed if a function is being mapped to multiple APIs. Takes a string with the function name as input.

If your function requires setting up some input variables you can do so by adding an element to `default_constraints_all` dictionary within `tests/spec_verification/test_dtype_constraints.py`.

 - `default_constraints`: This dictionary helps set specific constraints and values for parameters. These constraints correspond to the type hint of each parameter. Each type has different constraints that can be set, and some have default values, so you might not need to pass other_constraints for every operation. If there is no default, you must specify an initialization value, or the testcase may fail. The dictionary's keys must be the name of the function that they are constraining and the value must be a dictionary with the constraints. Here is the list of possible parameter types and constraints:
    - **tensor** - constraints: `init` and `shape` default: `tp.ones(shape=(3,2))`. If `init` is passed then value must be in the form of a `list`. Example: `"scale": {"init": [1, 1, 1]}` or `"scale": {"shape": (3,3)}`
    - **int** - constraints: `init` default: **no default**. Example: `"dim": {"init": 0}`.
    - **dtype** - constraints: **no constraints** default: **no default**. Dtype parameters will be set using `dtype_constraints` input.
    - **tuple** - constraints: `init` default: **no default**. Example: `"dims": {"init": (3,3)}`. 
    - **list/sequence of tensors** - constraints: `count`, `init`, and `shape` default: `count=2, shape=(3,2)`. Example: `"dim": {"count": 3}`. No default means that you must specify an initialization value or an error will be thrown. This will create a list/sequence of tensors of size `count` and each tensor will follow the `init` and `shape` value similar to tensor parameters.
    - **device** - constraints: `target` default: `target="gpu"`. Example: `{"device": {"target": "cpu"}}`.
    - **int list** - constraints: `init` default: **no default**. Example: `"dim": {"init": [1, 2, 3]}`.
    - **bool** - constraints: `init` default: **no default**. Example: `"dim": {"init": True}`. 
    - **float** - constraints: `init` default: **no default**. Example: `"dim": {"init": 1.23}`

After the decorator is set up, it will automatically run verification test cases alongside the regular test cases. If you only want to run the verifier, execute `pytest -s -v` within the tests/spec_verification folder.