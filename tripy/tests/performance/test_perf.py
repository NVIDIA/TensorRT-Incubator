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
import time
from textwrap import dedent
from typing import Callable

import pytest
import torch

# Need to import cases in order to populate PERF_CASES and load pytest fixtures
from tests.performance.cases import *
from tests.performance.conftest import PERF_CASES

import nvtripy as tp


def run_timed_trials(thunk: Callable[[], None], warm_up_runs=10, iterations=1000):
    """
    Returns the average time measured for calls to the thunk (the function intended to be timed)
    in microseconds. First performs the specified number of untimed warm-ups.
    """

    for _ in range(warm_up_runs):
        thunk()

    start = time.perf_counter_ns()
    for _ in range(iterations):
        thunk()
    end = time.perf_counter_ns()
    return (end - start) / (iterations * 1000.0)


@pytest.mark.parametrize("perf_case", PERF_CASES)
def test_perf_regression(perf_case, benchmark):
    compiled_tripy_module, _, inputs, _ = perf_case

    def run_inference():
        compiled_tripy_module(**inputs)
        compiled_tripy_module.stream.synchronize()

    benchmark(run_inference)


@pytest.mark.parametrize("perf_case", PERF_CASES)
def test_perf_comparative(perf_case):
    compiled_tripy_module, compiled_torch_module, inputs, perf_threshold = perf_case

    WARM_UP_RUNS = 10
    ITERATIONS = 250

    # Time Tripy
    stream = tp.default_stream()

    for _ in range(WARM_UP_RUNS):
        compiled_tripy_module(**inputs)
    stream.synchronize()

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        tripy_out = compiled_tripy_module(**inputs)
    stream.synchronize()
    end = time.perf_counter()

    # Torch will report time in ms:
    tripy_time = (end - start) * 1000

    # Time Torch
    torch_inputs = {key: torch.from_dlpack(value).to(device="cuda") for key, value in inputs.items()}

    with torch.no_grad():
        for _ in range(WARM_UP_RUNS):
            compiled_torch_module(**torch_inputs)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(ITERATIONS):
            torch_out = compiled_torch_module(**torch_inputs)
        end.record()
        torch.cuda.synchronize()

        torch_time = start.elapsed_time(end)

    # If the outputs don't match, then we're either not comparing apples-to-apples
    # or there is an accuracy bug somewhere - either way we want to catch it here.
    assert torch.allclose(torch_out, torch.from_dlpack(tripy_out))

    print(f"Tripy was {torch_time / float(tripy_time)}x faster than Torch")
    assert (tripy_time * perf_threshold) < torch_time


def test_tripy_overhead():
    def measure_overhead(num_io, warm_up_runs=10, iterations=1000):
        """
        Returns the overhead introduced by Tripy code for the specified number
        of input/output tensors of a function in microseconds.
        """
        import nvtripy as tp

        assert num_io > 0

        arg_str = ", ".join(f"arg{num}" for num in range(num_io))
        exec(
            dedent(
                f"""
                def func({arg_str}):
                    return [{arg_str}]
                """
            ),
            locals(),
            globals(),
        )

        # By using an empty shape, we ensure that we are measuring nothing
        # except Tripy Python overheads.
        SHAPE = (0,)
        compiled_one_io = tp.compile(func, args=[tp.InputInfo(shape=SHAPE, dtype=tp.float32) for _ in range(num_io)])

        inputs = [tp.iota(shape=SHAPE, dtype=tp.float32) for _ in range(num_io)]
        for input in inputs:
            input.eval()

        def measure_thunk():
            return compiled_one_io(*inputs)

        return run_timed_trials(measure_thunk, warm_up_runs=warm_up_runs, iterations=iterations)

    assert measure_overhead(1) < 250.0

    # Check that the overhead increases at most linearly as we increase number of I/O tensors.
    overheads = [measure_overhead(i) for i in range(3, 10)]
    deltas = [n - p for p, n in zip(overheads[:-1], overheads[1:])]
    print(f"overheads: {overheads}")
    print(f"deltas: {deltas}")
    assert all(delta < 45 for delta in deltas)

    # Ensure all deltas are within a few microseconds of each other
    average_delta = sum(deltas) / float(len(deltas))
    assert all(abs(delta - average_delta) < 10 for delta in deltas)


def test_tripy_param_update(benchmark):
    m = tp.Module()
    m.param = tp.Tensor([1, 2, 3, 4])

    # Leave the instantiation outside of the measured section to avoid overhead from registry calls.
    new_param = tp.Tensor([5, 6, 7, 8])

    def measure_thunk():
        m.param = new_param

    benchmark(measure_thunk)
