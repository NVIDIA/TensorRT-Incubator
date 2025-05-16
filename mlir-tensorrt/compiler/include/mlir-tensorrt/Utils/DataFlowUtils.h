//===- DataFlowUtils.h ------------------------------------------*- C++ -*-===//
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
/// Utilities associated with the MLIR dataflow framework.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_UTILS_DATAFLOWUTILS
#define MLIR_TENSORRT_UTILS_DATAFLOWUTILS

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

/// A rewrite listener that is aware of a DataFlowSolver state. When operations
/// are replaced or erased, the corresponding adjustments are made to the solver
/// state.
template <typename... LatticeTypes>
class SolverStateListener : public RewriterBase::Listener {
public:
  SolverStateListener(DataFlowSolver &solver) : solver(solver) {}

  /// Remap relevant analysis state of type T from `original` to `replacement`.
  template <typename T>
  void copyLatticeState(Value from, Value to) {
    if constexpr (!std::is_same_v<T, dataflow::Executable>) {
      if (const T *lattice = solver.lookupState<T>(from)) {
        T *latticeReplacement = solver.getOrCreateState<T>(to);
        latticeReplacement->getValue() = lattice->getValue();
      }
    } else {
      // do nothing for liveness analysis for the moment except create the state
      if (const auto *oldState =
              solver.lookupState<dataflow::Executable>(from)) {
        dataflow::Executable *newState = solver.getOrCreateState<T>(to);
        // Set to live if old state is live. We ignore change status.
        if (oldState->isLive())
          (void)newState->setToLive();
      }
    }
  }

  void copyLatticeStates(Value from, Value to) {
    (copyLatticeState<LatticeTypes>(from, to), ...);
  }

  /// This is not a method of `RewriterBase::Listener` but can be used to
  /// simplify state copying when users are cloning operations. It must be
  /// invoked manually.
  void notifyOperationCloned(Operation *op, Operation *clone) {
    for (auto [originalResult, clonedResult] :
         llvm::zip_equal(op->getResults(), clone->getResults()))
      copyLatticeStates(originalResult, clonedResult);
  }

protected:
  void notifyOperationReplaced(Operation *op,
                               ValueRange replacements) override {
    for (auto [original, replacement] :
         llvm::zip_equal(op->getResults(), replacements))
      copyLatticeStates(original, replacement);
    solver.eraseState(solver.getProgramPointAfter(op));
  }

  void notifyOperationReplaced(Operation *op, Operation *replacement) override {
    notifyOperationReplaced(op, replacement->getResults());
  }

  void notifyOperationErased(Operation *op) override {
    solver.eraseState(solver.getProgramPointAfter(op));
    for (Value res : op->getResults())
      solver.eraseState(res);
  }

  DataFlowSolver &solver;
};

} // namespace mlir

#endif // MLIR_TENSORRT_UTILS_DATAFLOWUTILS
