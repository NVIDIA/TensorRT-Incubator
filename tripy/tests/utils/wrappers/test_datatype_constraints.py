# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import contextlib
import inspect
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import nvtripy as tp
import pytest
from nvtripy.common.datatype import DATA_TYPES
from nvtripy.utils import wrappers
from nvtripy.utils.types import str_from_type_annotation
from nvtripy.utils.utils import make_list
from nvtripy.utils.wrappers import DATA_TYPE_CONSTRAINTS
from tests import helper
from tests.conftest import skip_if_older_than_sm89


@dataclass
class DtypeConstraintCase:
    func: Callable
    constraints: Dict[str, str]
    variables: Dict[str, tp.dtype]
    negative: bool  # Whether this is a negative test case

    def __str__(self):
        return f"{self.func.__name__}-{'_'.join(f'{key}-{val}' for key, val in self.variables.items())}" + (
            "-invalid" if self.negative else "-valid"
        )


DTYPE_CONSTRAINT_CASES: List[DtypeConstraintCase] = []

for dtc in DATA_TYPE_CONSTRAINTS:
    keys, values = zip(*dtc.variables.items())

    def add_cases(combinations, negative):
        for combination in combinations:
            dtype_combination = dict(zip(keys, combination))
            DTYPE_CONSTRAINT_CASES.append(
                pytest.param(
                    DtypeConstraintCase(dtc.func, dtc.constraints, dtype_combination, negative),
                    marks=(
                        skip_if_older_than_sm89()
                        if any(dtype == "float8" for dtype in dtype_combination.values())
                        else []
                    ),
                )
            )

    # Positive cases:
    positive_combinations = list(itertools.product(*values))
    positive_combinations = [
        comb
        for comb in positive_combinations
        if not any(all(dtc.variables[key] == val for key, val in exception.items()) for exception in dtc.exceptions)
    ]
    add_cases(positive_combinations, negative=False)

    # Negative cases - we do this by simply generating all possible combinations and removing the positive ones:
    total_dtypes = set(map(str, DATA_TYPES.values()))
    negative_combinations = list(itertools.product(*(total_dtypes for _ in values)))
    negative_combinations = list(comb for comb in negative_combinations if comb not in positive_combinations)
    add_cases(negative_combinations, negative=True)


DTYPE_CONSTRAINT_CASES.sort(key=lambda case: str(case))


def generate_input_values(case: DtypeConstraintCase):
    # In some cases, we need to use custom values so that the code is valid.
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

    inputs = {}
    for param_name, param in inspect.signature(case.func).parameters.items():
        param_type = str_from_type_annotation(param.annotation)

        dtype = None
        if param_name in case.constraints:
            dtype = DATA_TYPES[case.variables[case.constraints[param_name]]]

        if dtype == tp.int4:
            # TODO (#579): Enable int4 inputs
            pytest.skip(f"#579: Cannot generate INT4 inputs")

        if case.func.__name__ in CUSTOM_VALUES and param_name in CUSTOM_VALUES[case.func.__name__]:
            inputs[param_name] = CUSTOM_VALUES[case.func.__name__][param_name]
            if isinstance(inputs[param_name], tp.Tensor) and dtype is not None:
                inputs[param_name] = tp.cast(inputs[param_name], dtype=dtype)
                inputs[param_name].eval()
            continue

        if param.default is not inspect.Parameter.empty and dtype is None:
            continue  # Skip optional parameters unless we explicitly need to set a datatype for them.

        if "nvtripy.Tensor" in param_type:
            assert dtype is not None, "Tensors must have type annotations"
            # Need to cast here because `ones` does not support all types.
            tensor = tp.Tensor(np.ones((1, 2, 5, 5), dtype=np.float32))
            if "Sequence" in param_type:
                inputs[param_name] = [tp.cast(tensor, dtype=dtype) for _ in range(2)]
            else:
                inputs[param_name] = tp.cast(tensor, dtype=dtype)
        elif "nvtripy.dtype" in param_type:
            assert dtype is not None, "Data types must have type annotations"
            inputs[param_name] = dtype
        elif "numbers.Number" in param_type or "int" in param_type or "float" in param_type:
            inputs[param_name] = 1
        else:
            assert False, f"Unsupported parameter type: {param_type}"
    return inputs


@pytest.mark.parametrize("case", DTYPE_CONSTRAINT_CASES, ids=lambda case: str(case))
def test_datatype_constraints(case: DtypeConstraintCase):

    # If data type checking is enabled, negative tests will trivially pass (we will throw an
    # error before even trying to call the function).
    with helper.config("enable_dtype_checking", False):
        inputs = generate_input_values(case)

        with contextlib.ExitStack() as stack:
            if case.negative:
                stack.enter_context(helper.raises(Exception))

            outputs = make_list(case.func(**inputs))

            # Some APIs do not generate Tensor outputs (e.g. `allclose`), so we don't need to evaluate those.
            # For DimensionSizes, we don't need to check the type either (they can only be one type)
            if not any(type(out) == tp.Tensor for out in outputs):
                return

            expected_return_types = [
                case.variables[cons] for cons in make_list(case.constraints[wrappers.RETURN_VALUE])
            ]
            assert expected_return_types, "Return value must have a constraint"
            if len(expected_return_types) < len(outputs):
                expected_return_types += [expected_return_types[-1]] * (len(outputs) - len(expected_return_types))

            for out, expected_type in zip(outputs, expected_return_types):
                out.eval()
                assert out.dtype == DATA_TYPES[expected_type], f"Expected {expected_type}, got {out.dtype}"
