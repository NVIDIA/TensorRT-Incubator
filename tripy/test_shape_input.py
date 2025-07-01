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
import nvtripy as tp
import cupy as cp

from nvtripy.logging import logger

logger.verbosity = "ir"
import mlir_tensorrt.runtime.api as runtime


def func(x, y):
    x = x + x
    x = tp.reshape(x, (-1, y))
    return x


compiled_func = tp.compile(
    func, args=[tp.InputInfo(shape=((2, 4, 6), 4), dtype=tp.float32), tp.DimensionInputInfo(value_bounds=(1, 2, 3))]
)

print("compilation complete.")

sig = compiled_func._executable_signature

for idx in range(2):

    arg = sig.get_arg(idx)
    memref = runtime.MemRefType(arg)
    print(f"Arg {idx}: ", memref.address_space)

    print("Shape: ", memref.shape)
    bound = sig.get_arg_bound(idx)
    print(f"Bound: {bound.min()}, {bound.max()}")


# import pdb
# pdb.set_trace()


inp = cp.ones((4, 4), dtype=cp.float32)
inp = tp.Tensor(inp)
dim_inp = tp.DimensionSize(2)
out = compiled_func(inp, dim_inp)
print(out)
