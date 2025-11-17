//===- SlotAssignment.h -----------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Declarations for slot assignment algorithms that map SSA values to
/// function frame slots/registers for register-based VM backends.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_TARGET_SLOTASSIGNMENT_SLOTASSIGNMENT_H
#define MLIR_EXECUTOR_TARGET_SLOTASSIGNMENT_SLOTASSIGNMENT_H

#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {

class FunctionOpInterface;

/// SlotAssignmentManager encapsulates how an MLIR function is analyzed and
/// SSA values are mapped to a register-based VM's notion of registers, which
/// we call "slots" here. Slots should correspond function-frame-local
/// registers. In Lua, this corresponds to local variables. Generally there is
/// a hard upper limit on the number of local variables, which we set to 200
/// here by default.
class SlotAssignmentManager {
public:
  /// Encapsulates options for the SlotAssignmentManager.
  struct Options {
    /// Whether to coalesce live-ranges of the operands and results of pure ops
    /// (e.g. identity operations or unary math ops).
    bool enablePureOpCoalescing = true;

    /// Whether to coalesce live-ranges of block arguments with their
    /// predecessors. This is a key optimization and should only be disabled for
    /// testing purposes. The slot assignments produced when this is set to
    /// 'off' are still correct, but backends may have to deal with more cycles
    /// in "slot swaps" needed at branching instructions.
    bool enableBlockArgumentCoalescing = true;

    /// The maximum number of local slots to use.
    unsigned maxLocalSlots = 199;
  };

  SlotAssignmentManager(FunctionOpInterface funcOp, Options options);

  /// The type of allocation for a value.
  ///   - Local: A slot/register local to the function frame
  ///   - Spill: The value is stored in a memory slot.
  enum class SlotType : uint8_t {
    Local,
    Spill,
  };

  /// Holds information regarding how an SSA value was mapped to a slot.
  struct SlotAssignment {
    SlotAssignment(SlotType type, uint32_t id) : type(type), id(id) {}
    SlotType type;
    /// The ID of the allocated variable.
    uint32_t id;
  };

  /// Get the allocation result for the given value.
  SlotAssignment getAllocationResult(Value value) const {
    return assignment.at(value);
  }

  /// Returns true if any SSA value was assigned to a "spill" or memory slot.
  bool hasSpill() const { return spill; }

  /// Print the allocation results to the given stream.
  void print(llvm::raw_ostream &os) const;

private:
  Options options;

  FunctionOpInterface funcOp;
  /// Holds the final mapping from value to slot.
  DenseMap<Value, SlotAssignment> assignment;
  /// Indicates whether any SSA value was assigned to a "spill" or memory slot.
  /// How this achieved may vary depending on the actual target backend.
  bool spill{false};
};

void emitSlotSwap(
    llvm::MutableArrayRef<int32_t> sourceSlots,
    llvm::ArrayRef<int32_t> targetSlots, int32_t tempSlotId,
    llvm::function_ref<void(int32_t sourceSlot, int32_t targetSlot)> emitMove);

void emitSlotSwap(
    llvm::MutableArrayRef<StringRef> sourceSlots,
    llvm::ArrayRef<StringRef> targetSlots, StringRef tempSlotId,
    llvm::function_ref<void(StringRef sourceSlot, StringRef targetSlot)>
        emitMove);

} // namespace mlir

#endif // MLIR_EXECUTOR_TARGET_SLOTASSIGNMENT_SLOTASSIGNMENT_H
