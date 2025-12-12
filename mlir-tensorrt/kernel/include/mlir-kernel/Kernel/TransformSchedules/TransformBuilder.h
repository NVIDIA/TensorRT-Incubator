//===- TransformBuilder.h -------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Builder utilities for Transform IR operations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_KERNEL_TRANSFORMSCHEDULES_TRANSFORMBUILDER
#define MLIR_KERNEL_KERNEL_TRANSFORMSCHEDULES_TRANSFORMBUILDER

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

namespace mlir::kernel {

/// Builder utilities for Transform IR operations.
class TransformIRBuilder : public mlir::ImplicitLocOpBuilder {
public:
  using ImplicitLocOpBuilder::ImplicitLocOpBuilder;

  /// Given the handle to an operation and the operand index, this creates the
  /// sequence `transform.get_operand`, `transform.get_defining_op` to retrieve
  /// the handle of the producer. It should only be used in situations where the
  /// sequence is guaranteed to succeed (e.g. operand is defined by an op and is
  /// not a BlockArgument).
  Value operandHandle(Value consumerHandle, int64_t operandIndex);

  /// Create a `transform.apply_patterns` op with the given pattern sets.
  template <typename... Patterns>
  void applyPatterns(Value funcHandle) {
    create<transform::ApplyPatternsOp>(
        funcHandle, [](OpBuilder &b, Location loc) {
          static_assert(sizeof...(Patterns) > 0, "no patterns provided");
          (b.create<Patterns>(loc), ...);
        });
  }

  /// Create the "apply common subexpression elimination" operation to the given
  /// handle (must be isolated from above).
  void cse(Value handle);

  /// Create IR to fuse all the non-null handles in `fusableOperandHandles` into
  /// `containingOpHandle`. The handles that are fused are replaced with the
  /// new producer inside the containing op. The final containing op is returned
  /// (original may have been replaced).
  Value fuseProducersIntoContainingOp(Value containingOpHandle,
                                      MutableArrayRef<Value> producerHandles);

  /// Create IR to get and return the parent `scf.forall` handle for the given
  /// op handle.
  Value getParentForallOp(Value opHandle);

  /// Create the "tile to forall" operation to the given handle.
  transform::TileUsingForallOp tileToForall(Value opHandle,
                                            ArrayRef<int64_t> numThreads,
                                            transform::NumThreadsSpec dispatch);

  transform::TileUsingForallOp tileToForall(Value opHandle,
                                            ArrayRef<int64_t> tileShape);

  /// Create the "sequence" operation to the given handle.
  transform::SequenceOp sequence(
      Value funcHandle,
      std::function<void(TransformIRBuilder &, BlockArgument)> bodyBuilder);

  /// Create a `transform.structured.match` operation.
  template <typename MatchOpType>
  Value structuredMatch(Value containerHandle,
                        ArrayRef<NamedAttribute> attrsToMatch = {}) {
    transform::MatchOp matchOp{};
    if constexpr (std::is_same_v<MatchOpType, linalg::LinalgOp>) {
      matchOp = create<transform::MatchOp>(
          anyOpType, containerHandle, ArrayAttr{},
          getAttr<transform::MatchInterfaceEnumAttr>(
              transform::MatchInterfaceEnum::LinalgOp),
          DictionaryAttr{}, TypeAttr{}, ArrayAttr{});
    } else {
      matchOp = create<transform::MatchOp>(
          containerHandle,
          ArrayRef<StringRef>(MatchOpType::getOperationName()));
    }
    if (attrsToMatch.empty())
      return matchOp;
    matchOp.setOpAttrsAttr(getDictionaryAttr(attrsToMatch));
    return matchOp;
  }

  /// Cached `!transform.any_op` type.
  const Type anyOpType{transform::AnyOpType::get(getContext())};

  /// Cached `!transform.any_value` type.
  const Type anyValueType{transform::AnyValueType::get(getContext())};
};
} // namespace mlir::kernel

#endif // MLIR_KERNEL_KERNEL_TRANSFORMSCHEDULES_TRANSFORMBUILDER
