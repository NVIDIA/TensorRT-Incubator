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

from textwrap import dedent

import cupy as cp

import nvtripy as tp
from tests import helper
from nvtripy.frontend.trace import Trace


class TestTrace:
    def test_single_layer_structure(self):
        a = tp.Tensor([0], name="a")

        trace = Trace([a.trace_tensor])

        assert len(trace.ops) == 1
        layer = trace.ops[0]

        assert layer == a.trace_tensor.producer
        assert layer.inputs == []
        # Check that name propagates
        assert layer.outputs[0].name == "a"

    def test_trace_recurses_inputs(self):
        a = tp.Tensor([0], name="a")
        b = tp.Tensor([1], name="b")

        c = a + b
        c.name = "c"

        trace = Trace([c.trace_tensor])

        assert len(trace.ops) == 3
        names = {layer.outputs[0].name for layer in trace.ops}

        assert names == {"a", "b", "c"}

    def test_layers_are_topologically_sorted(self):
        a = tp.Tensor([0], name="a")
        b = tp.Tensor([1], name="b")

        c = a + b
        c.name = "c"

        trace = Trace([c.trace_tensor])

        assert len(trace.ops) == 3

        # The final layer should be 'c'. The ordering of 'a' and 'b' doesn't matter.
        assert trace.ops[-1].outputs[0].name == "c"

    def test_duplicate_traces_are_skipped(self):
        a = tp.Tensor([0])
        b = tp.Tensor([1])

        c = a + b
        # In a naive implementation, we might end up tracing the `c` expression twice.
        # Our implementation should not do that.
        d = c + c

        trace = Trace([d.trace_tensor])

        # If we end up tracing `c` twice, we'll end up with 7 layers: [a, b, a, b, c, c, d].
        # Without duplication, we should just have [a, b, c, d].
        assert len(trace.ops) == 4

    def test_str(self):
        a = tp.Tensor([0], name="a")
        b = tp.Tensor([1], name="b")

        c = a + b
        c.name = "c"

        trace = Trace([c.trace_tensor])

        print(trace)  # Makes it easier to debug when the test fails.
        assert (
            str(trace)
            == dedent(
                """
                a = storage(shape=(1,), dtype=int32, device=gpu:0, data_str=[0])
                b = storage(shape=(1,), dtype=int32, device=gpu:0, data_str=[1])
                c = a + b
                outputs:
                    c: [shape=([-1]), dtype=(int32), loc=(gpu:0)]
                """
            ).strip()
        )

    def test_infer_tensor_info(self):
        shape = (5, 5)
        a = tp.Tensor(cp.ones(shape, dtype=cp.float32))
        b = tp.Tensor(cp.ones(shape, dtype=cp.float32))

        c = a + b

        trace = Trace([c.trace_tensor])

        assert trace.ops[-1].outputs[0].dtype == a.trace_tensor.producer.dtype
        assert trace.ops[-1].outputs[0].device == tp.device("gpu")

    def test_multiple_outputs(self):
        shape = 1
        a = tp.Tensor(cp.ones(shape, dtype=cp.float32), name="a")
        b = tp.Tensor(cp.ones(shape, dtype=cp.float32), name="b")

        c = a + b
        c.name = "c"
        d = c + c
        d.name = "d"

        # The order c,d is important to test topological sort correctness, since if its d,c the dependencies are managed automatically.
        trace = Trace([c.trace_tensor, d.trace_tensor])
        print(trace)
        assert (
            str(trace)
            == dedent(
                """
                a = storage(shape=(1,), dtype=float32, device=gpu:0, data_str=[1.])
                b = storage(shape=(1,), dtype=float32, device=gpu:0, data_str=[1.])
                c = a + b
                d = c + c
                outputs:
                    c: [shape=([-1]), dtype=(float32), loc=(gpu:0)]
                    d: [shape=([-1]), dtype=(float32), loc=(gpu:0)]
                """
            ).strip()
        )

    def test_input_output(self):
        a = tp.Tensor([1, 1])
        # a is an input
        trace = Trace([a.trace_tensor], [a.trace_tensor])
        assert len(trace.inputs) == 1
        assert len(trace.outputs) == 1
        assert len(trace.ops) == 0

    def test_all_inputs(self):
        shape = 1
        # Need explicit data type here since by default dtype is cp.float64 which is not yet supported.
        # (38): Add cast operation to support unsupported backend types.
        a = tp.Tensor(cp.ones(shape, dtype=cp.float32), name="a")
        b = tp.Tensor(cp.ones(shape, dtype=cp.float32), name="b")

        c = a + b
        c.name = "c"
        trace = Trace([c.trace_tensor], [a.trace_tensor, b.trace_tensor])
        print(trace)
        assert (
            str(trace)
            == dedent(
                """
                inputs:
                    a: [shape=([1]), dtype=(float32), loc=(gpu:0)]
                    b: [shape=([1]), dtype=(float32), loc=(gpu:0)]
                c = a + b
                outputs:
                    c: [shape=([-1]), dtype=(float32), loc=(gpu:0)]
                """
            ).strip()
        )

    def test_const_and_input(self):
        shape = 1
        a = tp.Tensor(cp.ones(shape, dtype=cp.float32), name="a")
        b = tp.Tensor(cp.ones(shape, dtype=cp.float32), name="b")

        c = a + b
        c.name = "c"
        trace = Trace([c.trace_tensor], [a.trace_tensor])
        print(trace)
        assert (
            str(trace)
            == dedent(
                """
                inputs:
                    a: [shape=([1]), dtype=(float32), loc=(gpu:0)]
                b = storage(shape=(1,), dtype=float32, device=gpu:0, data_str=[1.])
                c = a + b
                outputs:
                    c: [shape=([-1]), dtype=(float32), loc=(gpu:0)]
                """
            ).strip()
        )

    def test_duplicate_tensor_names_fails(self):
        a = tp.ones((2, 3), dtype=tp.float32)
        b = tp.ones((2, 3), dtype=tp.float32)

        a.name = "a"
        b.name = "a"

        c = a + b

        with helper.raises(
            tp.TripyException,
            match="Found distinct tensors with the same name: 'a'.",
            has_stack_info_for=[a, b],
        ):
            Trace([c.trace_tensor])
