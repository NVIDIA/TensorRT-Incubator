#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

import pytest
from tests import helper
import tripy as tp


# Tests to ensure that we're able to map errors from MLIR-TRT back to the Python code cleanly.
class TestErrorMapping:
    def test_invalid_slice(self):
        values = tp.Tensor([1, 2, 3])
        sliced = values[4]

        with helper.raises(
            tp.TripyException,
            r"limit index 5 is larger than dimension size 3 in dimension 0",
            has_stack_info_for=[values],
        ):
            sliced.eval()

    def test_reshape_invalid_volume(self):
        tensor = tp.ones((2, 2))
        reshaped = tp.reshape(tensor, (3, 3))

        with helper.raises(
            tp.TripyException,
            r"number of output elements \(9\) doesn't match expected number of elements \(4\)",
            has_stack_info_for=[tensor, reshaped],
        ):
            reshaped.eval()

    def test_reason_context(self):
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.backend.mlir.compiler import map_error_to_user_code_and_raise
        from tripy.common.exception import TripyException
        from tripy.frontend.trace import Trace

        with FlatIRTensor.context(["This is the first level of context"]):
            with FlatIRTensor.context(["This is the second level of context"]):
                # We need to emit an error from one of the internally created `FlatIRTensor`s to see the context
                a = tp.ones((1,))
                b = tp.ones((1,))
                trace = Trace([a + b])
                flat_ir = trace.to_flat_ir()
                producer = flat_ir.outputs[0].producer.inputs[0]
                flat_ir_inputs = ",".join(map(lambda i: i.name, producer.producer.inputs))
                trace_inputs = ",".join(producer.producer.trace_input_names)
                trace_output = producer.producer.trace_output_names[0]
                err_str = f'loc("{flat_ir_inputs};;<out>;;{producer.name};;<trace_in>;;{trace_inputs};;<trace_out>;;{trace_output}"): Test error'

                with pytest.raises(
                    TripyException,
                    match=".*This is the first level of context\n    This is the second level of context\n.*",
                ) as exc:
                    map_error_to_user_code_and_raise(flat_ir, exc, err_str)

    def test_layer_metadata_callback(self):
        # TODO: Finish this:
        inp = tp.ones((2, 2))
        out = tp.gelu(inp)

        out.eval()
