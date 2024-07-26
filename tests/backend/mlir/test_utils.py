
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from mlir_tensorrt.compiler import ir

import tripy
from tripy.backend.mlir import utils as mlir_utils
from tripy.common.datatype import DATA_TYPES
import os


class TestUtils:
    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_convert_dtype(self, dtype):
        if dtype in {tripy.bool}:
            pytest.skip("Bool is not working correctly yet")

        with mlir_utils.make_ir_context():
            assert (
                mlir_utils.get_mlir_dtype(dtype)
                == {
                    "float64": ir.F64Type.get(),
                    "float32": ir.F32Type.get(),
                    "float16": ir.F16Type.get(),
                    "float8": ir.Float8E4M3FNType.get(),
                    "bfloat16": ir.BF16Type.get(),
                    "int4": ir.IntegerType.get_signless(4),
                    "int8": ir.IntegerType.get_signless(8),
                    "int16": ir.IntegerType.get_signless(16),
                    "int32": ir.IntegerType.get_signless(32),
                    "int64": ir.IntegerType.get_signless(64),
                    "bool": ir.IntegerType.get_signless(1),
                }[dtype.name]
            )
