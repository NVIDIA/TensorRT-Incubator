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
import inspect
import os
import tempfile
from typing import Tuple

import numpy as np
import nvtripy as tp
import pytest
from tests import helper
from tests.backend.api.conftest import *


@pytest.fixture(scope="session")
def single_return_executable():
    return tp.compile(add, args=[tp.InputInfo((2, 2), dtype=tp.float32), tp.InputInfo((2, 2), dtype=tp.float32)])


@pytest.fixture(scope="session")
def multiple_return_executable():
    return tp.compile(
        returns_multiple_tensors, args=[tp.InputInfo((2, 2), dtype=tp.float32), tp.InputInfo((2, 2), dtype=tp.float32)]
    )


class TestExecutable:
    def test_args(self, single_return_executable):
        args = [tp.ones((2, 2), dtype=tp.float32).eval(), tp.ones((2, 2), dtype=tp.float32).eval()]
        out = single_return_executable(*args)

        assert tp.allclose(out, add(*args))

    def test_kwargs(self, single_return_executable):
        kwargs = {"a": tp.ones((2, 2), dtype=tp.float32).eval(), "b": tp.ones((2, 2), dtype=tp.float32).eval()}
        out = single_return_executable(**kwargs)

        assert tp.allclose(out, add(**kwargs))

    @pytest.mark.parametrize(
        "args,kwargs,expected_error",
        [
            ([tp.ones((2, 2), dtype=tp.float32)], {}, "Missing argument: b"),
            (
                [
                    tp.ones((2, 2), dtype=tp.float32),
                    tp.ones((2, 2), dtype=tp.float32),
                    tp.ones((2, 2), dtype=tp.float32),
                ],
                {},
                "Incorrect number of arguments.",
            ),
            (
                [tp.ones((2, 2), dtype=tp.float32), tp.ones((2, 2), dtype=tp.float32)],
                {"b": tp.ones((2, 2), dtype=tp.float32)},
                r"Extra keyword arguments: \['b'\]",
            ),
            (
                [tp.ones((2, 2), dtype=tp.float32), tp.ones((2, 2), dtype=tp.float32)],
                {"c": tp.ones((2, 2), dtype=tp.float32)},
                r"Extra keyword arguments: \['c'\]",
            ),
        ],
    )
    def test_incorrect_arguments(self, args, kwargs, expected_error, single_return_executable):
        with helper.raises(tp.TripyException, match=expected_error):
            single_return_executable(*args, **kwargs)

    def test_signature(self, single_return_executable):
        signature = inspect.signature(single_return_executable)

        assert list(signature.parameters.keys()) == ["a", "b"]
        for param in signature.parameters.values():
            assert param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            assert param.annotation == tp.Tensor

        assert signature.return_annotation == tp.Tensor

    def test_signature_multiple_return_values(self, multiple_return_executable):
        signature = inspect.signature(multiple_return_executable)

        assert signature.return_annotation == Tuple[tp.Tensor, tp.Tensor]

    def test_file_io(self, single_return_executable):
        with tempfile.TemporaryDirectory() as temp_dir:
            exe_file = os.path.join(temp_dir, "executable.json")
            single_return_executable.save(exe_file)
            assert os.path.exists(exe_file)
            loaded_executable = tp.Executable.load(exe_file)
            assert loaded_executable.__signature__ == single_return_executable.__signature__

            inp = tp.iota((2, 2), dtype=tp.float32).eval()
            out1 = single_return_executable(inp, inp)
            out2 = loaded_executable(inp, inp)
            assert tp.equal(out1, out2)

            assert loaded_executable.input_infos == single_return_executable.input_infos

    def test_tensorrt_engine(self, single_return_executable):
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        trt_engine = single_return_executable.serialized_tensorrt_engine
        load_engine = EngineFromBytes(trt_engine)
        with TrtRunner(load_engine) as runner:
            inp_data0 = np.random.rand(2, 2).astype(np.float32)
            inp_data1 = np.random.rand(2, 2).astype(np.float32)
            output = runner.infer(feed_dict={"arg0": inp_data0, "arg1": inp_data1})["result0"]
            tripy_output = single_return_executable(tp.Tensor(inp_data0).eval(), tp.Tensor(inp_data1).eval())
            assert tp.equal(tripy_output, tp.Tensor(output))

    def test_input_info(self, single_return_executable):
        input_infos = single_return_executable.input_infos
        assert len(input_infos) == 2
        assert input_infos["a"].shape_bounds.min == (2, 2)
        assert input_infos["a"].shape_bounds.opt == (2, 2)
        assert input_infos["a"].shape_bounds.max == (2, 2)
        assert input_infos["a"].dtype == tp.float32

        assert input_infos["b"].shape_bounds.min == (2, 2)
        assert input_infos["b"].shape_bounds.opt == (2, 2)
        assert input_infos["b"].shape_bounds.max == (2, 2)
        assert input_infos["b"].dtype == tp.float32

    def test_incorrect_dtype_rejected(self):
        a = tp.ones((2, 2), dtype=tp.int32).eval()

        compiled_add = tp.compile(
            add, args=[tp.InputInfo((2, 2), dtype=tp.float32), tp.InputInfo((2, 2), dtype=tp.float32)]
        )
        with helper.raises(tp.TripyException, "Unexpected tensor data type."):
            compiled_add(a, a)

    def test_incorrect_shape_rejected(self):
        a = tp.ones((1, 2), dtype=tp.float32).eval()

        compiled_add = tp.compile(
            add, args=[tp.InputInfo((2, 2), dtype=tp.float32), tp.InputInfo((2, 2), dtype=tp.float32)]
        )
        with helper.raises(tp.TripyException, "Unexpected tensor shape.", has_stack_info_for=[a]):
            compiled_add(a, a)
