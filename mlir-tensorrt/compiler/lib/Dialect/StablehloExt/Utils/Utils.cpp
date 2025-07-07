//===- Utils.cpp ----------- ----------------------------------------------===//
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
#include "mlir-tensorrt/Dialect/StablehloExt/Utils/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::stablehlo;

bool stablehlo::canUpdateTypeWithoutCast(
    OpOperand &use, const std::function<bool(OpOperand &)> &otherCases) {
  Operation *consumer = use.getOwner();
  Value value = use.get();
  // There is one limitation to StableHlo ops being updated in place:
  // all operations that accept shape tensor argument should have shape tensor
  // with known rank.
  if (isa<stablehlo::CompositeOp>(consumer))
    return false;

  if (auto iotaOp = dyn_cast<stablehlo::DynamicIotaOp>(consumer))
    return false; // shape tensor is the only argument for iotaOp, always
                  // insert a cast op.

  if (auto sliceOp = dyn_cast<stablehlo::DynamicSliceOp>(consumer))
    return sliceOp.getOperand() == value;

  if (auto dynamicUpdateSliceOp =
          dyn_cast<stablehlo::DynamicUpdateSliceOp>(consumer))
    return dynamicUpdateSliceOp.getOperand() == value;

  if (auto dynamicBroadcastInDimOp =
          dyn_cast<stablehlo::DynamicBroadcastInDimOp>(consumer))
    return dynamicBroadcastInDimOp.getOperand() == value;

  if (auto reshapeOp = dyn_cast<stablehlo::DynamicReshapeOp>(consumer))
    return reshapeOp.getOperand() == value;

  if (auto sliceOp = dyn_cast<stablehlo::RealDynamicSliceOp>(consumer))
    return sliceOp.getOperand().getType() == value.getType();

  if (auto padOp = dyn_cast<stablehlo::DynamicPadOp>(consumer))
    return padOp.getOperand() == value;

  if (auto gatherOp = dyn_cast<stablehlo::DynamicGatherOp>(consumer))
    return gatherOp.getOperand() == value;

  if (auto convOp = dyn_cast<stablehlo::DynamicConvOp>(consumer))
    return convOp.getOperand(0) == value;

  return isa<stablehlo::StablehloDialect>(consumer->getDialect()) ||
         (otherCases && otherCases(use));
}

bool stablehlo::canUpdateTypeWithoutCast(
    Value result, const std::function<bool(Operation *)> &otherCases) {
  return llvm::all_of(result.getUses(), [&](OpOperand &use) {
    return canUpdateTypeWithoutCast(use) ||
           (otherCases && otherCases(use.getOwner()));
  });
}

bool stablehlo::canConvertToLinalg(Operation *op) {
  auto returnTrue = [](Operation *op) { return true; };
  // Elementwise ops. Taken from `StablehloToLinalgPointwise.cpp`.
  return llvm::TypeSwitch<Operation *, bool>(op)
      // clang-format off
      .Case<
         stablehlo::AbsOp,
         stablehlo::AddOp,
         stablehlo::AndOp,
         stablehlo::Atan2Op,
         stablehlo::BitcastConvertOp,
         stablehlo::CbrtOp,
         stablehlo::CeilOp,
         stablehlo::ClampOp,
         stablehlo::ClzOp,
         stablehlo::CompareOp,
         stablehlo::ComplexOp,
         stablehlo::ConvertOp,
         stablehlo::CosineOp,
         stablehlo::DivOp,
         stablehlo::ExpOp,
         stablehlo::Expm1Op,
         stablehlo::FloorOp,
         stablehlo::ImagOp,
         stablehlo::IsFiniteOp,
         stablehlo::Log1pOp,
         stablehlo::LogOp,
         stablehlo::LogisticOp,
         stablehlo::MaxOp,
         stablehlo::MinOp,
         stablehlo::MulOp,
         stablehlo::NegOp,
         stablehlo::NotOp,
         stablehlo::OrOp,
         stablehlo::PopulationCountOp,
         stablehlo::PowOp,
         stablehlo::RealOp,
         stablehlo::ReducePrecisionOp,
         stablehlo::RemOp,
         stablehlo::RoundNearestEvenOp,
         stablehlo::RoundOp,
         stablehlo::RsqrtOp,
         stablehlo::SelectOp,
         stablehlo::ShiftLeftOp,
         stablehlo::ShiftRightArithmeticOp,
         stablehlo::ShiftRightLogicalOp,
         stablehlo::SignOp,
         stablehlo::SineOp,
         stablehlo::SqrtOp,
         stablehlo::SubtractOp,
         stablehlo::TanhOp,
         stablehlo::XorOp>
      // clang-format on
      (returnTrue)
      // Random ops. Taken from `StablehloToLinalgRandom.cpp`.
      .Case<stablehlo::RngBitGeneratorOp, stablehlo::RngOp>(returnTrue)
      // Reductions.
      .Case<stablehlo::ReduceOp, stablehlo::ReduceWindowOp>(returnTrue)
      // Basic Contractions.
      .Case<stablehlo::DotOp, stablehlo::DotGeneralOp>(returnTrue)
      // Convolutions.
      .Case<stablehlo::ConvolutionOp>(returnTrue)
      // Other ops.
      // From `StablehloLegalizeToLinalg.cpp`.
      // Note: Left out `stablehlo.reverse` since currently it generates IR with
      // index maps which cannot be tiled.
      // clang-format off
      .Case<stablehlo::BitcastConvertOp,
            stablehlo::BroadcastInDimOp,
            stablehlo::BroadcastOp,
            stablehlo::ConcatenateOp,
            stablehlo::ConstantOp,
            stablehlo::DynamicBroadcastInDimOp,
            stablehlo::DynamicIotaOp,
            stablehlo::DynamicSliceOp,
            stablehlo::DynamicUpdateSliceOp,
            stablehlo::EinsumOp,
            stablehlo::GatherOp,
            stablehlo::IotaOp,
            stablehlo::MapOp,
            stablehlo::PadOp,
            stablehlo::RealDynamicSliceOp,
            stablehlo::ReshapeOp,
            stablehlo::SelectAndScatterOp,
            stablehlo::SetDimensionSizeOp,
            stablehlo::SliceOp,
            stablehlo::TorchIndexSelectOp,
            stablehlo::TransposeOp
            >(returnTrue)
      // clang-format on
      .Default([](Operation *op) { return false; });
}
