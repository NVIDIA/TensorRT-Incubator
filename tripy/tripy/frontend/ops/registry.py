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

from typing import Any

from tripy.function_registry import FunctionRegistry


# We use the tensor method registry to define methods on the `Tensor` class out of line.
# This lets the method live alongside the trace operation and makes it a bit more modular
# to add new operations. This can only be used for magic methods.
class TensorMethodRegistry(FunctionRegistry):
    def __call__(self, key: Any):
        # We make a special exception for "shape" since we actually do want that to be a property
        allowed_methods = ["numpy", "cupy", "shape"]
        assert (
            key in allowed_methods or key.startswith("__") and key.endswith("__")
        ), f"The tensor method registry should only be used for magic methods, but was used for: {key}"

        return super().__call__(key)


TENSOR_METHOD_REGISTRY = TensorMethodRegistry()
