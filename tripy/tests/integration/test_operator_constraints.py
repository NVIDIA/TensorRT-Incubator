# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for operator constraints.

This test suite validates that Tripy's operator constraints correctly predict what the underlying
software stack (MLIR-TRT/TensorRT) will accept or reject. For each operator with constraints, we:

1. Generate all possible combinations of data types for parameters that accept Tensors or dtypes
2. Use Tripy's input_requirements to predict whether each combination should be valid
3. Call the operator with Tripy's validation disabled, so errors come from the underlying stack
4. For valid combinations: verify the outputs match the output_guarantees
5. For invalid combinations: ensure the underlying stack raises an exception during evaluation

This ensures Tripy's constraint system stays in sync with the underlying implementation.
"""

import contextlib
import inspect
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

import nvtripy as tp
from nvtripy.common.datatype import DATA_TYPES
from nvtripy.frontend.wrappers import OPERATOR_CONSTRAINTS
from nvtripy.utils.types import str_from_type_annotation
from nvtripy.utils.utils import make_list
from tests import helper
from tests.conftest import skip_if_older_than_sm89


@dataclass
class OperatorConstraintCase:
    func: Callable
    dtype_values: Dict[str, tp.dtype]  # The specific dtype combination for this test

    def __str__(self):
        param_str = "_".join(f"{key}-{val}" for key, val in self.dtype_values.items())
        return f"{self.func.__name__}-{param_str}"


def get_dtype_constrained_params(func: Callable) -> List[str]:
    sig = inspect.signature(func)
    return [
        param_name
        for param_name, param in sig.parameters.items()
        if param.annotation is not inspect.Parameter.empty
        and (type_str := str_from_type_annotation(param.annotation))
        and ("Tensor" in type_str or "nvtripy.dtype" in type_str)
    ]


def generate_test_cases() -> List[OperatorConstraintCase]:
    cases = []

    for op_constraint in OPERATOR_CONSTRAINTS:
        func = op_constraint.func
        dtype_params = get_dtype_constrained_params(func)

        if not dtype_params:
            continue

        all_dtypes = list(DATA_TYPES.values())

        for dtype_combination in itertools.product(all_dtypes, repeat=len(dtype_params)):
            dtype_values = dict(zip(dtype_params, dtype_combination))

            # Skip FP8 on older hardware
            marks = [skip_if_older_than_sm89()] if any(dtype == tp.float8 for dtype in dtype_values.values()) else []

            cases.append(pytest.param(OperatorConstraintCase(func, dtype_values), marks=marks))

    # Sort for deterministic ordering
    cases.sort(key=lambda case: str(case.values[0]))
    return cases


CUSTOM_VALUES = {
    "__getitem__": {"index": 0},
    "arange": {"start": 0, "stop": 10, "step": 1},
    "avgpool": {"kernel_dims": [2, 2]},
    "convolution": {
        "weight": tp.Tensor(np.ones((2, 2, 3, 3), dtype=np.float32)),
        "bias": tp.Tensor([1.0, 2.0]),
        "padding": ((0, 0), (0, 0)),
        "stride": [1, 1],
        "groups": 1,
        "dilation": [1, 1],
    },
    "copy": {
        "input": tp.ones((2, 2)),
        "device": tp.device("cpu"),
    },
    "deconvolution": {
        "weight": tp.Tensor(np.ones((2, 2, 3, 3), dtype=np.float32)),
        "bias": tp.Tensor([1.0, 2.0]),
        "padding": ((0, 0), (0, 0)),
        "stride": [1, 1],
        "groups": 1,
        "dilation": [1, 1],
    },
    "dequantize": {"scale": tp.Tensor([1.0, 2.0]), "dim": 1},
    "expand": {"sizes": tp.Tensor((2, 2, 5, 5))},
    "full": {"shape": tp.Tensor([2, 2]), "value": tp.Tensor(1.0)},
    "full_like": {"value": tp.Tensor(1.0)},
    "gather": {"index": tp.Tensor([1])},
    "instancenorm": {
        "num_channels": 2,
        "weight": tp.Tensor(np.ones((2,), dtype=np.float32)),
        "bias": tp.Tensor(np.zeros((2,), dtype=np.float32)),
    },
    "iota": {"shape": tp.Tensor([2, 2])},
    "maxpool": {"kernel_dims": [2, 2]},
    "ones": {"shape": [2, 2]},
    "outer": {"vec1": tp.Tensor([1, 2, 3]), "vec2": tp.Tensor([1, 2, 3])},
    "pad": {"pad": [(0, 1), (1, 0), (1, 1), (0, 0)]},
    "permute": {"perm": [1, 0, 3, 2]},
    "quantize": {"scale": tp.Tensor([1.0, 2.0]), "dim": 1},
    "repeat": {"repeats": 2, "dim": 0},
    "reshape": {"shape": tp.Tensor([2, 25])},
    "resize": {
        "mode": "nearest",
        "output_shape": tp.Tensor((1, 2, 10, 10)),
        "scales": [1, 1, 2, 2],
    },
    "squeeze": {"dims": 0},
    "transpose": {"dim0": 0, "dim1": 1},
    "zeros": {"shape": [2, 2]},
}

# Arguments that must be constants on CPU
REQUIRES_CPU_CONST = {
    "dequantize": {"scale"},
    "quantize": {"scale"},
}

# Some operations require input shapes to be known
REQUIRES_KNOWN_SHAPES = {
    "convolution": {"input", "weight", "bias"},
    "deconvolution": {"input", "weight", "bias"},
}


def _apply_tensor_adjustments(tensor: tp.Tensor, func_name: str, param_name: str) -> tp.Tensor:
    if func_name in REQUIRES_CPU_CONST and param_name in REQUIRES_CPU_CONST[func_name]:
        if tensor.device.kind != "cpu":
            tensor = tp.copy(tensor, device=tp.device("cpu"))

    if func_name in REQUIRES_KNOWN_SHAPES and param_name in REQUIRES_KNOWN_SHAPES[func_name]:
        if any(dim == tp.constants.DYNAMIC_DIM for dim in tensor.trace_tensor.shape):
            tensor.trace_tensor.shape = tuple(map(int, tensor.shape))

    return tensor


def generate_input_values(case: OperatorConstraintCase) -> Dict[str, Any]:
    if tp.int4 in case.dtype_values.values():
        pytest.skip(f"#579: Cannot generate INT4 inputs")

    inputs = {}
    sig = inspect.signature(case.func)
    func_name = case.func.__name__

    for param_name, param in sig.parameters.items():
        dtype = case.dtype_values.get(param_name)
        param_type = str_from_type_annotation(param.annotation)

        # Handle custom values first
        if func_name in CUSTOM_VALUES and param_name in CUSTOM_VALUES[func_name]:
            inputs[param_name] = CUSTOM_VALUES[func_name][param_name]
            if isinstance(inputs[param_name], tp.Tensor) and dtype is not None:
                inputs[param_name] = tp.cast(inputs[param_name], dtype=dtype)
                inputs[param_name] = _apply_tensor_adjustments(inputs[param_name], func_name, param_name)
            continue

        # Skip optional parameters unless they need a specific dtype
        if param.default is not inspect.Parameter.empty and dtype is None:
            continue

        # Generate values based on parameter type
        if "Tensor" in param_type:
            assert dtype is not None, f"Tensor parameter '{param_name}' must have a dtype constraint"
            base_tensor = tp.cast(tp.Tensor(np.ones((1, 2, 5, 5), dtype=np.float32)), dtype=dtype)

            if "Sequence" in param_type or "List" in param_type:
                inputs[param_name] = [_apply_tensor_adjustments(base_tensor, func_name, param_name) for _ in range(2)]
            else:
                inputs[param_name] = _apply_tensor_adjustments(base_tensor, func_name, param_name)
        elif "nvtripy.dtype" in param_type:
            assert dtype is not None, f"dtype parameter '{param_name}' must have a dtype constraint"
            inputs[param_name] = dtype
        elif "numbers.Number" in param_type or "int" in param_type or "float" in param_type:
            inputs[param_name] = 1

    return inputs


OPERATOR_CONSTRAINT_CASES = generate_test_cases()


@pytest.mark.parametrize("case", OPERATOR_CONSTRAINT_CASES, ids=lambda case: str(case))
def test_operator_constraints(case: OperatorConstraintCase):
    op_constraint = next((oc for oc in OPERATOR_CONSTRAINTS if oc.func == case.func), None)
    assert op_constraint is not None, f"Could not find constraints for {case.func.__name__}"

    # If input validation is enabled, negative tests will trivially pass (we will throw an
    # error before even trying to call the underlying implementation).
    with helper.config("enable_input_validation", False):
        inputs = generate_input_values(case)
        merged_args = list(inputs.items())

        # Some operators may only define output guarantees.
        # In that case, we cannot predict input validity via constraints.
        is_valid = (
            True if op_constraint.input_requirements is None else bool(op_constraint.input_requirements(merged_args))
        )

        with contextlib.ExitStack() as stack:
            if not is_valid:
                stack.enter_context(helper.raises(Exception))

            outputs = make_list(case.func(**inputs))

            for out in outputs:
                if isinstance(out, tp.Tensor):
                    # Avoid evaluating CPU constants since some types (e.g. FP8) don't allow them to be used outside of
                    # certain operations.
                    out._eval_for_internal_methods()

            if is_valid and op_constraint.output_guarantees is not None:
                output_result = op_constraint.output_guarantees(merged_args, tuple(outputs))
                assert output_result, f"Output guarantees not met for {case.func.__name__}: " + " ".join(
                    output_result.error_details
                )
