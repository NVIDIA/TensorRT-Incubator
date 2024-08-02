//===- TensorKindOpInterfaceImpl.cpp --------------------------------------===//
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
/// TensorKindOpInterface implementation for TensorRT operations. See also the
/// [`TensorKindAnalysis` documentation](docs/Analysis/TensorKindAnalysis.md)
/// for more information.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"

using namespace mlir;
using namespace mlir::tensorrt;

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//
namespace {
struct SliceOpTensorKindOpInterfaceImpl
    : public TensorKindOpInterface::ExternalModel<
          SliceOpTensorKindOpInterfaceImpl, tensorrt::SliceOp> {
  void inferOperandKind(
      Operation *op, ArrayRef<TensorKindLattice *> operands,
      ArrayRef<const TensorKindLattice *> results,
      llvm::function_ref<void(OpOperand &, TensorKind)> setOperandKind) const {
    auto sliceOp = cast<SliceOp>(op);
    if (!results[0]->getValue().isUninitialized()) {
      setOperandKind(sliceOp.getInputMutable(),
                     results[0]->getValue().getKind());
      if (sliceOp.getFill())
        setOperandKind(sliceOp.getFillMutable()[0],
                       results[0]->getValue().getKind());
    }

    for (auto optionalOperandRange :
         {sliceOp.getStartMutable(), sliceOp.getSizeMutable(),
          sliceOp.getStrideMutable()}) {
      if (optionalOperandRange.size() == 1)
        setOperandKind(optionalOperandRange[0], TensorKind::Host);
    }
  }

  TensorKind getStaticOperandTensorKind(Operation *sliceOp,
                                        OpOperand &operand) const {
    int64_t nonSliceParams = cast<SliceOp>(sliceOp).getFill() ? 2 : 1;
    return operand.getOperandNumber() >= nonSliceParams ? TensorKind::Host
                                                        : TensorKind::Unknown;
  }
};

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//
struct ShuffleOpTensorKindOpInterfaceImpl
    : public TensorKindOpInterface::ExternalModel<
          ShuffleOpTensorKindOpInterfaceImpl, tensorrt::ShuffleOp> {
  void inferOperandKind(
      Operation *op, ArrayRef<TensorKindLattice *> operands,
      ArrayRef<const TensorKindLattice *> results,
      llvm::function_ref<void(OpOperand &, TensorKind)> setOperandKind) const {
    auto shuffleOp = cast<ShuffleOp>(op);
    setOperandKind(shuffleOp.getInputMutable(),
                   results[0]->getValue().getKind());
    if (shuffleOp.getDynamicReshape())
      setOperandKind(shuffleOp.getDynamicReshapeMutable()[0], TensorKind::Host);
  }

  TensorKind getStaticOperandTensorKind(Operation *op,
                                        OpOperand &operand) const {
    auto shuffleOp = cast<ShuffleOp>(op);
    return shuffleOp.getDynamicReshape() &&
                   operand.get() == shuffleOp.getDynamicReshape()
               ? TensorKind::Host
               : TensorKind::Unknown;
  }
};

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//
struct ShapeTensorKindOpInterfaceImpl
    : public TensorKindOpInterface::ExternalModel<
          ShapeTensorKindOpInterfaceImpl, tensorrt::ShapeOp> {
  void inferOperandKind(
      Operation *shapeOp, ArrayRef<TensorKindLattice *> operands,
      ArrayRef<const TensorKindLattice *> results,
      llvm::function_ref<void(OpOperand &, TensorKind)> setOperandKind) const {
    // Since the shape operation only uses the dimensions of the input,
    // we can't infer whether it is a shape or execution tensor here.
    setOperandKind(cast<ShapeOp>(shapeOp).getInputMutable(),
                   mlir::TensorKind::Unknown);
  }

  TensorKind getStaticOperandTensorKind(Operation *op,
                                        OpOperand &operand) const {
    return TensorKind::Unknown;
  }
};

//===----------------------------------------------------------------------===//
// LinspaceOp
//===----------------------------------------------------------------------===//
struct LinspaceTensorKindOpInterfaceImpl
    : public TensorKindOpInterface::ExternalModel<
          LinspaceTensorKindOpInterfaceImpl, LinspaceOp> {
  void inferOperandKind(
      Operation *op, ArrayRef<TensorKindLattice *> operands,
      ArrayRef<const TensorKindLattice *> results,
      llvm::function_ref<void(OpOperand &, TensorKind)> setOperandKind) const {
    auto linspaceOp = cast<LinspaceOp>(op);
    if (linspaceOp.getShape())
      setOperandKind(linspaceOp.getShapeMutable()[0], TensorKind::Host);
    if (linspaceOp.getStart())
      setOperandKind(linspaceOp.getStartMutable()[0],
                     results[0]->getValue().getKind());
    if (linspaceOp.getStep())
      setOperandKind(linspaceOp.getStepMutable()[0],
                     results[0]->getValue().getKind());
  }

  TensorKind getStaticOperandTensorKind(Operation *op,
                                        OpOperand &operand) const {
    auto linspaceOp = cast<LinspaceOp>(op);
    if (linspaceOp.getShape() && operand.get() == linspaceOp.getShape())
      return TensorKind::Host;
    return TensorKind::Unknown;
  }
};

} // namespace

void tensorrt::registerTensorKindOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, tensorrt::TensorRTDialect *dialect) {
        SliceOp::attachInterface<SliceOpTensorKindOpInterfaceImpl>(*ctx);
        ShuffleOp::attachInterface<ShuffleOpTensorKindOpInterfaceImpl>(*ctx);
        ShapeOp::attachInterface<ShapeTensorKindOpInterfaceImpl>(*ctx);
        LinspaceOp::attachInterface<LinspaceTensorKindOpInterfaceImpl>(*ctx);
      });
}
