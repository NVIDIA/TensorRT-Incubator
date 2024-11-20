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

from typing import Any, Callable

# We use the tensor method registry to define methods on the `Tensor` class out of line.
# This lets the method live alongside the trace operation and makes it a bit more modular
# to add new operations. This can only be used for magic methods.
TENSOR_METHOD_REGISTRY = {}


def register_tensor_method(name: str):
    """
    Decorator to add the method to the tensor method registry with the name specified.
    This does not use the FunctionRegistry decorator because every tensor method would also be
    registered in the public function registry and we would prefer to avoid having overhead
    from having to dispatch overloads and check types twice.
    """

    # We make a special exception for "shape" since we actually do want that to be a property
    allowed_methods = ["shape"]
    assert name in allowed_methods or name.startswith(
        "__"
    ), f"The tensor method registry should only be used for magic methods, but was used for: {name}"

    def impl(func: Callable[..., Any]) -> Callable[..., Any]:
        TENSOR_METHOD_REGISTRY[name] = func
        return func

    return impl
