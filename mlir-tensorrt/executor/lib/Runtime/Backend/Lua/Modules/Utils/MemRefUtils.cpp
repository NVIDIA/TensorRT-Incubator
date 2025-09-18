//===- MemRefUtils.cpp ----- ----------------------------------------------===//
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
#include "mlir-executor/Runtime/Backend/Lua/Modules/Utils/MemRefUtils.h"
#include "mlir-executor/Runtime/Backend/Common/CommonRuntime.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaErrorHandling.h"
#include "mlir-executor/Runtime/Backend/Lua/SolAdaptor.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/Support/raw_ostream.h"

namespace mrt = mtrt;
using namespace mtrt;

constexpr unsigned kAlignedPtrArgNum = 1;
constexpr unsigned kShapeBeginArgNum = 3;
constexpr unsigned kOffsetArgNum = 2;

unsigned mrt::getNumArgsPerMemRef(unsigned rank) { return 3 + 2 * rank; }
unsigned mrt::getStrideBeginArgNum(unsigned rank) {
  return kShapeBeginArgNum + rank;
}

/// Returns the memref aligned ptr (and offset, shape, and strides through
/// the relevant arguments) given a variadic argument container. The
/// variadic args could contain multiple unpacked memrefs (of the same
/// rank), so `memrefIdx` indicates which memref is desired.
uintptr_t mrt::getMemRefInfo(sol::this_state state, sol::variadic_args args,
                             unsigned rank, unsigned memrefIdx, int64_t &offset,
                             std::vector<int64_t> &shape,
                             std::vector<int64_t> &strides) {
  const auto &arg =
      args[getNumArgsPerMemRef(rank) * memrefIdx + kAlignedPtrArgNum];
  if (!arg.is<uintptr_t>()) {
    luaL_error(state, "unexpected type for executor pointer argument");
    return 0;
  }

  const auto &offsetArg =
      args[getNumArgsPerMemRef(rank) * memrefIdx + kOffsetArgNum];
  if (!offsetArg.is<int>()) {
    luaL_error(state, "unexpected type for executor memref offset argument");
    return 0;
  }
  offset = offsetArg.as<int>();

  unsigned startOffset =
      getNumArgsPerMemRef(rank) * memrefIdx + kShapeBeginArgNum;
  for (unsigned i = startOffset, e = startOffset + rank; i < e; i++) {
    if (!args[i].is<int>()) {
      luaL_error(state, "unexpected type for executor shape size argument");
      return 0;
    }
    shape.push_back(args[i].as<int>());
  }

  // Strides
  startOffset =
      getNumArgsPerMemRef(rank) * memrefIdx + getStrideBeginArgNum(rank);
  for (unsigned i = startOffset, e = startOffset + rank; i < e; i++) {
    if (!args[i].is<int>()) {
      luaL_error(state, "unexpected type for executor shape size argument");
      return 0;
    }
    strides.push_back(args[i].as<int>());
  }

  return arg.as<uintptr_t>();
}
