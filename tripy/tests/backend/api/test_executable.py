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
        args = [tp.ones((2, 2), dtype=tp.float32), tp.ones((2, 2), dtype=tp.float32)]
        out = single_return_executable(*args)

        assert tp.allclose(out, add(*args))

    def test_kwargs(self, single_return_executable):
        kwargs = {"a": tp.ones((2, 2), dtype=tp.float32), "b": tp.ones((2, 2), dtype=tp.float32)}
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

    def test_io_tensor_info(self, multiple_return_executable):
        input_info = multiple_return_executable._get_input_info()
        assert len(input_info) == 2
        for i in range(2):
            assert input_info[i].shape_bounds == ((2, 2), (2, 2))
            assert input_info[i].dtype == tp.float32
        output_info = multiple_return_executable._get_output_info()
        assert len(output_info) == 2
        for i in range(2):
            assert output_info[i].shape_bounds == ((2, 2), (2, 2))
            assert output_info[i].dtype == tp.float32

    def test_file_io(self, single_return_executable):
        with tempfile.TemporaryDirectory() as temp_dir:
            exe_file = os.path.join(temp_dir, "executable.json")
            single_return_executable.save(exe_file)
            assert os.path.exists(exe_file)
            loaded_executable = tp.Executable.load(exe_file)
            assert loaded_executable.__signature__ == single_return_executable.__signature__

            inp = tp.iota((2, 2), dtype=tp.float32)
            out1 = single_return_executable(inp, inp)
            out2 = loaded_executable(inp, inp)
            assert tp.equal(out1, out2)
