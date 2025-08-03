//===- LuaAllocation.h ------------------------------------------*- C++ -*-===//
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
/// Declarations for Lua variable allocation.
///
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {

/// The type of allocation for a value.
///   - Local: A local variable is allocated.
///   - Global: A global variable is allocated.
///   - Spill: A spill variable is allocated (i.e. a table should be used).
///     This is used if local allocation exceeds the maximum number (200).
enum class LuaAllocationType {
  Local,
  Global,
  Spill,
};

class LuaAllocation {
public:
  LuaAllocation(func::FuncOp funcOp) : funcOp(funcOp) { allocate(funcOp); }

  struct AllocationResult {
    AllocationResult(LuaAllocationType type, unsigned id)
        : type(type), id(id) {}

    LuaAllocationType type;
    /// The ID of the allocated variable.
    unsigned id;
  };

  /// Get the allocation result for the given value.
  AllocationResult getAllocationResult(Value value) const {
    return allocationResults.at(value);
  }

  bool hasSpill() const { return spill; }

  /// Print the allocation results to the given stream.
  void print(llvm::raw_ostream &os) const;

private:
  /// Allocate Lua variables for the given function.
  void allocate(func::FuncOp funcOp);

private:
  func::FuncOp funcOp;
  DenseMap<Value, AllocationResult> allocationResults;
  bool spill{false};
};

} // namespace mlir
