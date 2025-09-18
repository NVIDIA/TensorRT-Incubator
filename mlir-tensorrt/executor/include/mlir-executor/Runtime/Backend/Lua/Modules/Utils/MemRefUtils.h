//===- MemRefUtils.h --------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// Utilities for handling memref arguments in Lua.
///
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <vector>

namespace sol {
struct this_state;
struct variadic_args;
} // namespace sol

namespace mtrt {

/// Populates `offset`, `shape` and `strides` with memref info derived from the
/// list `args` and returns pointer. This method differs from the "safe" version
/// above in that it doesn't check `tracker` that the allocation is tracked.
uintptr_t getMemRefInfo(sol::this_state state, sol::variadic_args args,
                        unsigned rank, unsigned memrefIdx, int64_t &offset,
                        std::vector<int64_t> &shape,
                        std::vector<int64_t> &strides);

/// Return the number of arguments that are required to build a memref of rank
/// `rank`.
unsigned getNumArgsPerMemRef(unsigned rank);

/// Return the argument index within a parameter pack representing a memref that
/// corresponds to the the start of the strides vector.
unsigned getStrideBeginArgNum(unsigned rank);

} // namespace mtrt
