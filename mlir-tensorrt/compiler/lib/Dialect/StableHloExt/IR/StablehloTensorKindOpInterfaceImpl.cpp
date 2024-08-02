//===- StablehloTensorKindOpInterfaceImpl.cpp -----------------------------===//
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
/// Implementation of TensorKindOpInterface for Stable HLO ops.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir-tensorrt/Dialect/StableHloExt/IR/StableHloExt.h"
#include "mlir/IR/DialectRegistry.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// DynamicSliceOp
//===----------------------------------------------------------------------===//
namespace {

template <typename T>
constexpr std::pair<int64_t, int64_t> OpToHostParametersOffsetAndSize() {

  // Simple macro for creating the appearance of a table.
#define CASE(OpType, start, size)                                              \
  if constexpr (std::is_same_v<T, OpType>)                                     \
  return {start, size}

  CASE(stablehlo::DynamicIotaOp, 0, 1);
  // Note that `stablehlo.dynamic_slice` and `stablehlo.dynamic_update_slice`
  // have a variadic number of 0-d integer tensors which are host params.
  // The variadic segment is always at the end, so we use `ShapedType::kDynamic`
  // to signal this.
  CASE(stablehlo::DynamicSliceOp, 1, ShapedType::kDynamic);
  CASE(stablehlo::DynamicUpdateSliceOp, 2, ShapedType::kDynamic);
  CASE(stablehlo::DynamicBroadcastInDimOp, 1, 1);
  CASE(stablehlo::DynamicReshapeOp, 1, 1);
  CASE(stablehlo::RealDynamicSliceOp, 1, 3);
  CASE(stablehlo::DynamicPadOp, 2, 3);
  CASE(stablehlo::DynamicGatherOp, 2, 1);
  CASE(stablehlo::DynamicConvOp, 2, 1);

#undef CASE

  return {-1, -1};
}

template <typename OpType>
struct SimpleTensorKindOpInterfaceImpl
    : public TensorKindOpInterface::ExternalModel<
          SimpleTensorKindOpInterfaceImpl<OpType>, OpType> {
  static constexpr int64_t kHostParamSegmentOffset =
      std::get<0>(OpToHostParametersOffsetAndSize<OpType>());
  static constexpr int64_t kHostParamSegmentSize =
      std::get<1>(OpToHostParametersOffsetAndSize<OpType>());

  static_assert(kHostParamSegmentOffset >= 0 &&
                    (kHostParamSegmentSize >= 1 ||
                     kHostParamSegmentSize == ShapedType::kDynamic),
                "invalid operand segment offset/size specification");

  static constexpr int64_t operandMustHaveHostVisibility(int64_t operandIdx) {
    return operandIdx >= kHostParamSegmentOffset &&
           (kHostParamSegmentSize == ShapedType::kDynamic ||
            operandIdx < kHostParamSegmentOffset + kHostParamSegmentSize);
  }

  void inferOperandKind(
      Operation *op, ArrayRef<TensorKindLattice *> operands,
      ArrayRef<const TensorKindLattice *> results,
      llvm::function_ref<void(OpOperand &, TensorKind)> setOperandKind) const {
    for (auto [idx, operand] : llvm::enumerate(operands)) {
      if (operandMustHaveHostVisibility(idx)) {
        setOperandKind(op->getOpOperand(idx), mlir::TensorKind::Host);
        continue;
      }
      if (!results[0] || results[0]->getValue().isUninitialized())
        continue;
      setOperandKind(op->getOpOperand(idx), results[0]->getValue().getKind());
    }
  }

  TensorKind getStaticOperandTensorKind(Operation *op,
                                        OpOperand &operand) const {
    return operandMustHaveHostVisibility(operand.getOperandNumber())
               ? TensorKind::Host
               : TensorKind::Unknown;
  }
};

template <typename... Args>
void attachTensorKindOpInterfaceExternalModels(MLIRContext *ctx) {
  (Args::template attachInterface<SimpleTensorKindOpInterfaceImpl<Args>>(*ctx),
   ...);
}

} // namespace

void stablehlo::registerTensorKindOpInterfaceExternalModels(
    DialectRegistry &registry) {

  registry.addExtension(
      +[](MLIRContext *ctx, stablehlo::StablehloDialect *dialect) {
        attachTensorKindOpInterfaceExternalModels<
            // clang-format off
            stablehlo::DynamicIotaOp,
            stablehlo::DynamicSliceOp,
            stablehlo::DynamicUpdateSliceOp,
            stablehlo::DynamicBroadcastInDimOp,
            stablehlo::DynamicReshapeOp,
            stablehlo::RealDynamicSliceOp,
            stablehlo::DynamicPadOp,
            stablehlo::DynamicGatherOp,
            stablehlo::DynamicConvOp
            // clang-format on
            >(ctx);
      });
}
