//===- Verification.cpp  --------------------------------------------------===//
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
/// Implementation of verification routines for TensorRT ops.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::tensorrt;

/// Verify operands of matrix multiply op.
/// If input tensors does not have equal ranks, they are reshaped to be
/// broadcastable.
static LogicalResult validateMatMulOperands(MatrixMultiplyOp op) {
  TensorType input0Type = op.getInput0().getType();
  TensorType input1Type = op.getInput1().getType();
  MatrixOperation input0MatOp = op.getOp0();
  MatrixOperation input1MatOp = op.getOp1();

  ArrayRef<int64_t> input0CollectionDims = op.getCollectionDims();
  ArrayRef<int64_t> input1CollectionDims = op.getCollectionDims(1);
  if (input0CollectionDims.size() != input1CollectionDims.size())
    return op->emitOpError("Input0 and Input1 number of collection dims "
                           "doesn't match. Input0, MatrixOp: ")
           << static_cast<int>(input0MatOp)
           << ", Collection dims: " << input0CollectionDims
           << " ; Input1, MatrixOp: " << static_cast<int>(input1MatOp)
           << ", Collection dims: " << input1CollectionDims;

  // Check if collection dims are broadcastable.
  FailureOr<SmallVector<int64_t>> broadcastedCollectionDims =
      getBroadcastedShape(input0CollectionDims, input1CollectionDims);
  if (failed(broadcastedCollectionDims))
    return op->emitOpError(
        "collection (batch) dimensions of \"input0\" and \"input1\""
        " are not broadcastable");

  SmallVector<int64_t, 2> contractionDimSizes;
  if (input0MatOp == MatrixOperation::kVECTOR) {
    contractionDimSizes.push_back(
        input0Type.getDimSize(input0Type.getRank() - 1));
  } else {
    // Add the appropriate dimension size to the output shape, accounting for
    // transpose.
    auto matShape = input0Type.getShape().take_back(2);
    int64_t parDim = input0MatOp == MatrixOperation::kNONE ? 0 : 1;
    int64_t contractionDim = (parDim + 1) % 2;
    contractionDimSizes.push_back(matShape[contractionDim]);
  }

  if (input1MatOp == MatrixOperation::kVECTOR) {
    contractionDimSizes.push_back(
        input1Type.getDimSize(input1Type.getRank() - 1));
  } else {
    // Add the appropriate dimension size to the output shape, accounting for
    // transpose.
    auto matShape = input1Type.getShape().take_back(2);
    int64_t parDim = input1MatOp == MatrixOperation::kNONE ? 1 : 0;
    int64_t contractionDim = (parDim + 1) % 2;
    contractionDimSizes.push_back(matShape[contractionDim]);
  }

  // Check that all the contraction dims agree. This check should be omitted if
  // either dim is dynamic.
  if (!llvm::all_equal(contractionDimSizes) &&
      llvm::none_of(contractionDimSizes,
                    [](int64_t d) { return d == ShapedType::kDynamic; }))
    return op->emitOpError("operand shapes do not have consistent sizes for "
                           "the contraction dimension");

  return success();
}

/// Given the reassoiciation indices and `type` that is being rank-reduced,
/// check if each group in the reassoication each has only one non-unit
/// dimension.
static LogicalResult isValidRankReducingSqueezeReassociationMap(
    RankedTensorType type, ArrayRef<ReassociationIndices> reassociation) {
  for (const auto &indices : reassociation) {
    if (llvm::count_if(indices, [&](int64_t dim) {
          assert(dim < type.getRank() && "dim out-of-bounds");
          return type.getDimSize(dim) > 1;
        }) > 1)
      return failure();
  }
  return success();
}

LogicalResult tensorrt::LinspaceOp::verify() {
  // For rank > 1 linspace operations, we require that "start" be a scalar and
  // "step" be a 1D tensor of same size as rank as the result.
  if (getType().getRank() > 1) {
    if (getStep() == nullptr)
      return emitOpError("dynamic `step` must be specified if the result is "
                         "greater than rank 1");
    if (getStep().getType().getDimSize(0) != getType().getRank())
      return emitOpError("dynamic `step` type dimension 0 length must be the "
                         "same size as the rank of the result type");
  }
  if (getStep() != nullptr &&
      getType().getElementType() != getStep().getType().getElementType())
    return emitOpError(
        "`step` element type and result element type should be the same");
  if ((getStart() == nullptr && getStep() != nullptr) ||
      (getStart() == nullptr && getStep() != nullptr))
    return emitOpError(
        "`start` and `step` must either both be dynamic or both be static");
  if (getStart() != nullptr && getStep() != nullptr &&
      getStart().getType().getElementType() !=
          getStep().getType().getElementType())
    return emitOpError(
        "start and step tensor types must have the same element type");
  if (getShape() == nullptr &&
      llvm::any_of(getType().getShape(),
                   [](int64_t dim) { return dim == ShapedType::kDynamic; }))
    return emitOpError(
        "If `result` has dynamic dims, `shape` tensor must be present");
  return success();
}

LogicalResult tensorrt::RandomUniformOp::verify() {
  if ((getLow() == nullptr && getHigh() != nullptr) ||
      (getHigh() == nullptr && getLow() != nullptr))
    return emitOpError(
        "`low` and `high` must either both be dynamic or both be static");
  if (getLow() != nullptr && getHigh() != nullptr &&
      getLow().getType().getElementType() !=
          getHigh().getType().getElementType())
    return emitOpError(
        "`low` and `high` tensor types must have the same element type");
  if (getLow() != nullptr &&
      (getType().getElementType() != getLow().getType().getElementType()))
    return emitOpError(
        "`low`, `high` and `result` element type should be the same");
  if (getShape() == nullptr &&
      llvm::any_of(getType().getShape(),
                   [](int64_t dim) { return dim == ShapedType::kDynamic; }))
    return emitOpError(
        "If `result` has dynamic dims, `shape` tensor must be present");
  return success();
}

LogicalResult tensorrt::RandomNormalOp::verify() {
  if ((getMean() == nullptr && getStd() != nullptr) ||
      (getStd() == nullptr && getMean() != nullptr))
    return emitOpError(
        "`mean` and `std` must either both be dynamic or both be static");
  if (getMean() != nullptr && getStd() != nullptr &&
      getMean().getType().getElementType() !=
          getStd().getType().getElementType())
    return emitOpError(
        "`mean` and `std` tensor types must have the same element type");
  if (getMean() != nullptr &&
      (getType().getElementType() != getMean().getType().getElementType()))
    return emitOpError(
        "`mean`, `std` and `result` element type should be the same");
  if (getShape() == nullptr &&
      llvm::any_of(getType().getShape(),
                   [](int64_t dim) { return dim == ShapedType::kDynamic; }))
    return emitOpError(
        "If `result` has dynamic dims, `shape` tensor must be present");
  return success();
}

LogicalResult tensorrt::NormalizationOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType scaleType = getScale().getType();
  RankedTensorType biasType = getBias().getType();
  ArrayRef<int64_t> axis = getAxis();
  std::optional<uint32_t> numGroups = getNumGroups();

  auto checkScaleAndBiasShape = [](RankedTensorType type,
                                   int64_t expectedChannelDim) {
    // Check all elements are 1 except one dim
    if (llvm::count(type.getShape(), 1) != type.getRank() - 1)
      return failure();
    if (type.getDimSize(1) != expectedChannelDim)
      return failure();
    return success();
  };

  if (axis.back() >= inputType.getRank())
    return emitOpError("`axis` value ") << axis.back()
                                        << " is out of the bound"
                                           " for input tensor of rank "
                                        << inputType.getRank();

  // Verify Group Normalization cases
  if (numGroups.has_value() && *numGroups > 1) {
    // The channel dimension is considered to be the second dimension in a [N,
    // C, H, W, ...] formatted tensor
    if (numGroups.has_value() && *numGroups > 1 &&
        inputType.getShape()[1] % *numGroups != 0)
      return emitOpError(
          "It is an error to set `num_groups` to a value that does "
          "not evenly divide into the number of channels of the "
          "input tensor.");

    if (numGroups.has_value() && *numGroups > 1 &&
        !llvm::equal(axis, llvm::seq<int64_t>(2, inputType.getRank()))) {
      return emitOpError(
          "If num_groups != 1, it is expected that axis array contains "
          "all dimensions after the channel dimension (which is 1).");
    }

    if (numGroups.has_value() && *numGroups > 1 &&
        (failed(checkScaleAndBiasShape(scaleType, *numGroups)) ||
         failed(checkScaleAndBiasShape(biasType, *numGroups))))
      return emitOpError(
                 "If num_groups != 1, scale and bias shape is expected "
                 "to be [1, num_groups, 1, 1, ... N] where N is rank of input "
                 "tensor i.e. ")
             << inputType.getRank();
    return success();
  }

  if (axis.size() > 1) {
    // Verify Instance Normalization case
    // If scale, bias are in the form [1, C, 1, 1, .. N], which is the case for
    // instance normalization, check whether correct axis [2, 3, 4, .. N] are
    // provided.
    if (succeeded(checkScaleAndBiasShape(scaleType, inputType.getDimSize(1))) &&
        succeeded(checkScaleAndBiasShape(biasType, inputType.getDimSize(1))) &&
        !llvm::equal(axis, llvm::seq<int64_t>(2, inputType.getRank())))
      return emitOpError("If more than one axis is provided and scale/bias are "
                         "in the form [1, C, 1, 1, .. N]"
                         ", this is the case for instance normalization. Array "
                         "`axis` should contain "
                         "all the axis after channel.");
    return success();
  } else {
    // Verify batch normalization case, when axis[0] == 0
    if (axis[0] == 0 &&
        (failed(checkScaleAndBiasShape(scaleType, inputType.getDimSize(1))) ||
         failed(checkScaleAndBiasShape(biasType, inputType.getDimSize(1)))))
      return emitOpError(
          "In case of batch normalization (axis=0), scale and bias shape "
          "is expected to be [1, C, 1, 1, ..] where input is in the form [N, "
          "C, "
          "H, W, ...]");
    // When axis[0] != 0, it can be either instance norm or layer norm, based on
    // scale and bias shape.
  }
  return success();
}

LogicalResult tensorrt::ExpandRankOp::verify() {
  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto resultType = cast<RankedTensorType>(getType());
  if (inputType.getRank() > resultType.getRank())
    return emitOpError(
        "result rank should be greater than or equal to input rank");

  std::optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForReshape(inputType, resultType);
  // Check that each reassociation group has at most one non-unit dim. Here we
  // treat the operation as "collapsing" the result type to the input type.
  if (!reassociation.has_value() ||
      failed(isValidRankReducingSqueezeReassociationMap(resultType,
                                                        *reassociation)))
    return emitOpError("the reshape is not a valid rank expansion produced "
                       "from inserting 1's");

  return success();
}

LogicalResult tensorrt::CollapseRankOp::verify() {
  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto resultType = cast<RankedTensorType>(getType());
  const int64_t inputRank = inputType.getRank();
  const int64_t resultRank = resultType.getRank();
  if (inputRank < resultRank)
    return emitOpError(
        "input type rank should be greater than or equal to result type rank");

  // Get the reshape reassociation map. This is a result-rank-sized set of
  // vectors that map input dimension positions to output dimension positions.
  std::optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForReshape(inputType, resultType);
  if (!reassociation)
    return emitOpError(
        "failed to compute a reassociation map from input to output type");

  // Check that each reassociation group has at most one non-unit dim.
  if (failed(isValidRankReducingSqueezeReassociationMap(inputType,
                                                        *reassociation)))
    return emitOpError("expected to only drop unit-dims from the shape");

  return success();
}

LogicalResult tensorrt::BroadcastOp::verify() {
  TensorType inputType = getInput().getType();
  TensorType resultType = getType();

  // Check broadcasting dimension constraints.
  ArrayRef<int64_t> broadcastDims = getBroadcastDims();
  if (static_cast<int64_t>(broadcastDims.size()) != inputType.getRank())
    return emitOpError("expected ")
           << getBroadcastDimsAttrName() << " size to equal input type rank ("
           << inputType.getRank() << ")";
  if (!llvm::all_of(broadcastDims, [&](int64_t broadcastDimValue) {
        return broadcastDimValue >= 0 &&
               broadcastDimValue < resultType.getRank();
      }))
    return emitOpError() << "all " << getBroadcastDimsAttrName()
                         << " values must be in the range [0,"
                         << resultType.getRank() << ")";

  if (llvm::SmallSetVector<int64_t, 4>(broadcastDims.begin(),
                                       broadcastDims.end())
          .size() != broadcastDims.size())
    return emitOpError() << getBroadcastDimsAttrName()
                         << " contains duplicate values";

  // Dynamic result type is only allowed in the precense of shape operand.
  if (!resultType.hasStaticShape() && !getShape())
    return emitOpError()
           << "if the result type has unknown dimensions, a shape "
              "operand must be provided";

  // Check that the broadcasting makes sense.
  for (auto [inputShapeIdx, resultShapeIdx] :
       llvm::enumerate(getBroadcastDims())) {
    if (failed(checkLhsShapeBroadcastableToRhs(
            inputType.getShape().slice(inputShapeIdx, 1),
            resultType.getShape().slice(resultShapeIdx, 1))))
      return emitOpError() << "expected input shape dimension " << inputShapeIdx
                           << " (" << inputType.getDimSize(inputShapeIdx)
                           << ") to be broadcastable to result type dimension "
                           << resultShapeIdx << " ("
                           << resultType.getDimSize(resultShapeIdx) << ")";
  }

  return success();
}

LogicalResult tensorrt::TransposeOp::verify() {
  TensorType inputType = getInput().getType();
  AffineMap perm = getPermutation();
  if (perm.getNumResults() != inputType.getRank() || !perm.isPermutation())
    return emitOpError("expected \"permutation\" to be a permutation of rank ")
           << inputType.getRank();
  return success();
}

static LogicalResult verifyTopKOperation(Operation *op, TensorType inputType,
                                         int64_t reductionDim, int64_t k,
                                         TensorType valuesType) {

  // Check validity of the axis.
  if (reductionDim >= inputType.getRank() || reductionDim < 0)
    return op->emitOpError("\"axis\" attribute is ")
           << reductionDim << ", but this is out of bounds for input of rank "
           << inputType.getRank();

  if (valuesType.getDimSize(reductionDim) != k)
    return op->emitOpError("expected result shapes dim ")
           << reductionDim << " to equal " << k;

  // If the rank of the reduction dimension is known, then we can
  // verify that k is not OOB.
  if (!inputType.isDynamicDim(reductionDim) &&
      inputType.getDimSize(reductionDim) < k)
    return op->emitOpError() << "expected K attribute value to be smaller than "
                                "the dimension specified by \"axis\"";
  return success();
}

LogicalResult tensorrt::ArgMinOp::verify() {
  return verifyTopKOperation(this->getOperation(), getInput().getType(),
                             static_cast<int64_t>(getAxis()), /*k=*/1,
                             getValues().getType());
}

LogicalResult tensorrt::ArgMaxOp::verify() {
  return verifyTopKOperation(this->getOperation(), getInput().getType(),
                             static_cast<int64_t>(getAxis()), /*k=*/1,
                             getValues().getType());
}

// impl end

LogicalResult tensorrt::ConvolutionOp::verify() {
  // Convolution impl start
  if (!getKernel() && !getKernelStatic().has_value())
    return emitOpError(
        "kernel operand or kernelStatic attribute must be specified");

  if (getKernel() && getKernelStatic().has_value())
    return emitOpError("only one of kernel operand or kernelStatic attribute "
                       "can be specified");

  for (ArrayRef<int64_t> arrayAttrRef :
       {getStride(), getPrePadding(), getPostPadding()}) {
    if (static_cast<int64_t>(arrayAttrRef.size()) != getNumSpatialDims())
      return emitOpError() << "stride/pre_padding/post_padding should have "
                              "size equal to the number of spatial dimensions";
  }
  if (std::optional<ArrayRef<int64_t>> dilation = getDilation()) {
    if (static_cast<int64_t>(dilation->size()) != getNumSpatialDims())
      return emitOpError("dilation should have size equal to the number of "
                         "spatial dimensions");
  }

  if (getNumGroups() != 1) {
    ArrayRef<int64_t> kernelShape =
        getKernelStatic().has_value()
            ? getKernelStaticAttr().getShapedType().getShape()
            : cast<RankedTensorType>(getKernel().getType()).getShape();

    if (getInput().getType().getDimSize(1) % getNumGroups() != 0 ||
        kernelShape[0] % getNumGroups() != 0)
      return emitOpError("both input channels and output channels must be "
                         "divisible by ")
             << getNumGroupsAttrName();

    if (getInput().getType().getDimSize(1) / getNumGroups() != kernelShape[1])
      return emitOpError("for ")
             << getNumGroupsAttrName() << " = " << getNumGroups()
             << " and input channels = " << getInput().getType().getDimSize(1)
             << ", second (idx = 1) kernel dimension should be "
             << getInput().getType().getDimSize(1) / getNumGroups();
  }

  if (getBias() != nullptr &&
      (getBias().getType().getRank() != 1 ||
       getBias().getType().getDimSize(0) != getType().getDimSize(1)))
    return emitOpError(
        "bias type should be a rank-1 tensor type with size "
        "equal to the number of channels (dim 1) of the result tensor type");

  // Convolution impl end
  return success();
} // LogicalResult tensorrt::ConvolutionOp::verify()

LogicalResult tensorrt::ActivationOp::verify() {
  ActivationType act = getActivationType();
  if ((getInput().getType().getElementType().isSignlessInteger(32) ||
       getInput().getType().getElementType().isSignlessInteger(64)) &&
      (act != ActivationType::kRELU))
    return emitOpError()
           << "int32 and int64 types are supported only for RELU activation.";
  std::optional<APFloat> alpha = getAlpha();
  std::optional<APFloat> beta = getBeta();
  bool expectAlpha = requiresAlphaAttribute(act);
  bool expectBeta = requiresBetaAttribute(act);

  if ((expectAlpha && !getAlpha()) || (!expectAlpha && getAlpha()) ||
      (expectBeta && !getBeta()) || (!expectBeta && getBeta())) {
    return emitOpError() << (expectAlpha ? "expected `alpha` attribute"
                                         : "expected no `alpha` attribute")
                         << " and "
                         << (expectBeta ? "expected `beta` attribute"
                                        : "expected no `beta` attribute")
                         << " for `activationType="
                         << stringifyActivationType(act);
  }
  return success();
}

bool tensorrt::ActivationOp::requiresAlphaAttribute(ActivationType act) {
  return act != ActivationType::kRELU && act != ActivationType::kSIGMOID &&
         act != ActivationType::kTANH && act != ActivationType::kSOFTSIGN &&
         act != ActivationType::kGELU_TANH && act != ActivationType::kGELU_ERF;
}

bool tensorrt::ActivationOp::requiresBetaAttribute(ActivationType act) {
  return act != ActivationType::kRELU && act != ActivationType::kELU &&
         act != ActivationType::kLEAKY_RELU &&
         act != ActivationType::kSIGMOID && act != ActivationType::kTANH &&
         act != ActivationType::kSOFTSIGN &&
         act != ActivationType::kTHRESHOLDED_RELU &&
         act != ActivationType::kGELU_TANH && act != ActivationType::kGELU_ERF;
}

LogicalResult tensorrt::PoolingOp::verify() {
  // Pooling impl start
  TensorType inputType = getInput().getType();
  TensorType resultType = getType();
  if (resultType.getRank() != inputType.getRank())
    return emitOpError(
        "input tensor type and result tensor type should have equal rank");
  if (inputType.getRank() != 4 && resultType.getRank() != 5)
    return emitOpError("expected input tensor type to be rank 4 (2D pooling) "
                       "or 5 (3D pooling)");
  if ((getBlendFactor().has_value() &&
       getPoolingType() != PoolingType::kMAX_AVERAGE_BLEND) ||
      (!getBlendFactor().has_value() &&
       getPoolingType() == PoolingType::kMAX_AVERAGE_BLEND))
    return emitOpError(
        "blendFactor is required when pooling type is "
        "\"kMAX_AVERAGE_BLEND\", otherwise it should not be present");
  if ((getAverageCountExcludesPadding().has_value() &&
       getPoolingType() != PoolingType::kMAX_AVERAGE_BLEND &&
       getPoolingType() != PoolingType::kAVERAGE) ||
      (!getAverageCountExcludesPadding().has_value() &&
       getPoolingType() != PoolingType::kMAX))
    return emitOpError("\"averageCountExcludesPadding\" must be "
                       "provided when pooling type is "
                       "\"kMAX_AVERAGE_BLEND\" or \"kMAX\", otherwise it "
                       "should not be provided");
  if (getStride().size() != getWindowSize().size())
    return emitOpError("\"stride\" array size should be equal to the size of "
                       "the \"windowSize\" array");
  if ((getPrePadding().size() != getWindowSize().size()) ||
      (getPostPadding().size() != getWindowSize().size()))
    return emitOpError("\"prePadding\" and \"postPadding\" array sizes should "
                       "be equal to the size of "
                       "the \"windowSize\" array");
  // Pooling impl end
  return success();
} // LogicalResult tensorrt::PoolingOp::verify()

LogicalResult tensorrt::SoftMaxOp::verify() {
  // SoftMax impl start
  const int64_t inputRank = getInput().getType().getRank();
  int64_t axis = static_cast<int64_t>(getAxis());
  if (axis >= inputRank || axis < 0)
    return emitOpError("expected axis to be non-negative and less than ")
           << inputRank;
  // SoftMax impl end
  return success();
} // LogicalResult tensorrt::SoftMaxOp::verify()

LogicalResult tensorrt::ConcatenationOp::verify() {
  auto input0Type = cast<RankedTensorType>(getInputs()[0].getType());
  int64_t input0Rank = input0Type.getRank();

  auto concatAxis = getAxis();
  if (concatAxis >= input0Rank)
    return emitOpError("concat axis exceeds the dimension of input tensor.");

  // Verify that all inputs have the same rank and that shapes are equal (except
  // for concat axis dimension).
  SmallVector<SmallVector<uint32_t>> inputDimensions;
  for (size_t i = 1; i < getInputs().size(); ++i) {
    auto curInputType = cast<RankedTensorType>(getInputs()[i].getType());
    auto curInputRank = curInputType.getRank();
    if (curInputRank != input0Rank)
      return emitOpError("input rank at input[")
             << i << "] is " << curInputRank
             << " which is different from the rank of input tensor at index 0";
    for (unsigned j = 0; j < curInputType.getShape().size(); ++j) {
      if (j == concatAxis)
        continue;
      // If one of dimension is dynamic, continue
      if (curInputType.isDynamicDim(j) || input0Type.isDynamicDim(j))
        continue;
      if (curInputType.getDimSize(j) != input0Type.getDimSize(j))
        return emitOpError("input tensor[")
               << i << "] has size " << curInputType.getShape()[j]
               << " at dimension " << j << ", while the value should be "
               << input0Type.getDimSize(j);
    }
  }
  return success();
}

LogicalResult tensorrt::DeconvolutionOp::verify() {
  if (!getKernelWeights() && !getKernelWeightsStatic().has_value())
    return emitOpError("kernelWeights operand or kernelWeightsStatic attribute "
                       "must be specified");

  if (getKernelWeights() && getKernelWeightsStatic().has_value())
    return emitOpError(
        "only one of kernelWeights operand or kernelWeightsStatic attribute "
        "can be specified");

  for (ArrayRef<int64_t> arrayAttrRef :
       {getStride(), getPrePadding(), getPostPadding()}) {
    if (static_cast<int64_t>(arrayAttrRef.size()) != getNumSpatialDims())
      return emitOpError() << "stride/pre_padding/post_padding should have "
                              "size equal to the number of spatial dimensions";
  }
  if (std::optional<ArrayRef<int64_t>> dilation = getDilation()) {
    if (static_cast<int64_t>(dilation->size()) != getNumSpatialDims())
      return emitOpError("dilation should have size equal to the number of "
                         "spatial dimensions");
  }

  if (getNumGroups() != 1 &&
      (getInput().getType().getShape()[1] % getNumGroups() != 0))
    return emitOpError("input channels must be divisible by ")
           << getNumGroupsAttrName();

  if (getBiasWeights() != nullptr &&
      (getBiasWeights().getType().getRank() != 1 ||
       getBiasWeights().getType().getDimSize(0) != getType().getDimSize(1)))
    return emitOpError(
        "bias type should be a rank-1 tensor type with size "
        "equal to the number of channels (dim 1) of the result tensor type");
  return success();
} // LogicalResult tensorrt::DeconvolutionOp::verify()

static LogicalResult verifyElementWiseDataTypes(ElementWiseOp op) {
  /// It's expected that the ODS generated validator
  /// checks the restricted set of datatypes that TRT supports. This checks edge
  /// cases for bool and i32 types.
  ElementWiseOperation opType = op.getElementwiseOperation();

  SmallVector<Type, 2> inputElementTypes{
      cast<RankedTensorType>(op.getInput1().getType()).getElementType(),
      cast<RankedTensorType>(op.getInput2().getType()).getElementType()};
  Type resultElType = op.getType().getElementType();

  // Lambda returns true if the given type is an i1 (bool).
  auto isBoolType = [](Type t) { return t.isInteger(1); };
  auto isI32OrBoolType = [](Type t) {
    return t.isInteger(32) || t.isInteger(1);
  };

  switch (opType) {
  // These operations only support boolean inputs and should have boolean result
  // type.
  case ElementWiseOperation::kAND:
  case ElementWiseOperation::kOR:
  case ElementWiseOperation::kXOR: {
    if (!llvm::all_of(inputElementTypes, isBoolType))
      return op->emitOpError("ElementWiseOperation type " +
                             stringifyElementWiseOperation(opType) +
                             " expected all input types to be i1 (bool)");
    if (!isBoolType(resultElType))
      return op->emitOpError("result element type expected to be i1 (bool)");
    return success();
  }

  // Comparison operations do not support boolean types but should have boolean
  // result type.
  case ElementWiseOperation::kGREATER:
  case ElementWiseOperation::kLESS:
  case ElementWiseOperation::kEQUAL: {
    if (llvm::any_of(inputElementTypes, isBoolType))
      return op->emitOpError("ElementWiseOperation type " +
                             stringifyElementWiseOperation(opType) +
                             " does not support boolean input types");
    if (!isBoolType(resultElType))
      return op->emitOpError("result element type expected to be i1 (bool)");
    return success();
  }

  // These operations do not support boolean types
  case ElementWiseOperation::kDIV:
  case ElementWiseOperation::kFLOOR_DIV:
  case ElementWiseOperation::kMAX:
  case ElementWiseOperation::kMIN:
  case ElementWiseOperation::kPROD:
  case ElementWiseOperation::kSUB:
  case ElementWiseOperation::kSUM: {
    if (llvm::any_of(inputElementTypes, isBoolType))
      return op->emitOpError("ElementWiseOperation type " +
                             stringifyElementWiseOperation(opType) +
                             " does not support boolean input types");
    return success();
  }

  // Pow does not support bool or INT32 types
  case ElementWiseOperation::kPOW: {
    if (llvm::any_of(inputElementTypes, isI32OrBoolType))
      return op->emitOpError("ElementWiseOperation type " +
                             stringifyElementWiseOperation(opType) +
                             " does not allow i32 or i1 (bool) input types");
    return success();
  }
  }
  llvm_unreachable("unhandled enumeration value");
}

LogicalResult tensorrt::ElementWiseOp::verify() {
  if (failed(verifyElementWiseDataTypes(*this)))
    return failure();
  TensorType input1Type = getInput1().getType();
  TensorType input2Type = getInput2().getType();
  if (failed(checkShapesBroadcastable(input1Type, input2Type)))
    return emitOpError() << "shapes are not broadcastable: " << input1Type
                         << " vs " << input2Type;
  return success();
}

LogicalResult tensorrt::PaddingOp::verify() {
  // Padding impl start
  const int64_t inputRank = this->getInput().getType().getRank();
  ArrayRef<int64_t> inputShape = this->getInput().getType().getShape();
  if (inputRank < 4)
    return emitOpError("input rank should be greater or equal than 4");

  // TensorRT only supports padding along two innermost dimensions
  if (getPrePadding().size() != 2 || getPostPadding().size() != 2)
    return emitOpError() << "padding exactly two innermost dimensions is "
                            "supported but received "
                         << getPrePaddingAttrName()
                         << " input of size: " << getPrePadding().size()
                         << " and " << getPostPaddingAttrName()
                         << " input of size: " << getPostPadding().size();

  if (inputShape[inputRank - 1] != ShapedType::kDynamic &&
      inputShape[inputRank - 2] != ShapedType::kDynamic) {
    const int64_t firstPre = getPrePadding()[0];
    const int64_t secondPre = getPrePadding()[1];
    const int64_t firstPost = getPostPadding()[0];
    const int64_t secondPost = getPostPadding()[1];
    if (firstPre + inputShape[inputRank - 2] < 0 ||
        firstPost + inputShape[inputRank - 2] < 0 ||
        firstPre + firstPost + inputShape[inputRank - 2] < 0)
      return emitOpError() << "shape error on dimension " << inputRank - 2
                           << " : "
                           << "pre padding amount: " << firstPre << "; "
                           << "post padding amount: " << firstPost << "; "
                           << "origin dimension: " << inputShape[inputRank - 2];
    if (secondPre + inputShape[inputRank - 1] < 0 ||
        secondPost + inputShape[inputRank - 1] < 0 ||
        secondPre + secondPost + inputShape[inputRank - 1] < 0)
      return emitOpError() << "shape error on dimension " << inputRank - 1
                           << " : "
                           << "pre padding amount: " << secondPre << "; "
                           << "post padding amount: " << secondPost << "; "
                           << "origin dimension: " << inputShape[inputRank - 1];
  }
  // Padding impl end
  return success();
} // LogicalResult tensorrt::PaddingOp::verify()

LogicalResult tensorrt::ShuffleOp::verify() {
  TensorType operandType = getInput().getType();
  TensorType outputType = getType();

  if (operandType.hasStaticShape() && outputType.hasStaticShape() &&
      operandType.getNumElements() != outputType.getNumElements())
    return emitOpError(
        "operand and output tensor types should have same number of elements");

  return success();
}

LogicalResult tensorrt::ReshapeOp::verify() {
  TensorType inputType = getInput().getType();
  TensorType resultType = getType();

  // If we have static shapes, then we can check that the volumes are equal.
  if (inputType.hasStaticShape() && resultType.hasStaticShape() &&
      inputType.getNumElements() != resultType.getNumElements())
    return emitOpError("input and result tensor types should have the same "
                       "number of elements");

  if (!getShape() && resultType.getNumDynamicDims() > 1)
    return emitOpError("result type may have at most one dynamic dimension "
                       "if the 'shape' operand is not provided");

  // For dynamic reshapes, check that the shape tensor is of known shape and is
  // congruent with result rank.
  if (getShape()) {
    TensorType shapeType = getShape().getType();
    if (!shapeType.hasStaticShape())
      return emitOpError()
             << "dynamic reshape must be a 1D tensor of known shape";
    if (shapeType.getDimSize(0) != resultType.getRank())
      return emitOpError() << "reshape tensor size (" << shapeType.getDimSize(0)
                           << ") does not match result type rank ("
                           << resultType.getRank() << ")";
  }

  return success();
}

LogicalResult tensorrt::ReduceOp::verify() {
  // Reduce impl start
  if (llvm::any_of(getReduceAxes(), [this](int64_t dimension) {
        return dimension >= getInput().getType().getRank();
      }))
    return emitOpError("expected each element of reduceAxes to be smaller than "
                       "input type rank (")
           << getInput().getType().getRank() << ")";
  // Reduce impl end
  return success();
} // LogicalResult tensorrt::ReduceOp::verify()

LogicalResult tensorrt::TopKOp::verify() {
  // TopK impl start
  if (failed(verifyTopKOperation(this->getOperation(), getInput().getType(),
                                 static_cast<int64_t>(getAxis()),
                                 static_cast<int64_t>(getK()),
                                 getValues().getType())))
    return failure();
  // TopK impl end
  return success();
} // LogicalResult tensorrt::TopKOp::verify()

LogicalResult tensorrt::RaggedSoftMaxOp::verify() {
  // RaggedSoftMax impl start

  // RaggedSoftMax impl end
  return success();
} // LogicalResult tensorrt::RaggedSoftMaxOp::verify()

LogicalResult tensorrt::OneHotOp::verify() {
  // OneHot impl start
  const int64_t depthRank = getDepth().getType().getRank();
  if (depthRank != 0)
    return emitOpError("expected depth to be of rank 0");

  // OneHot impl end
  return success();
} // LogicalResult tensorrt::OneHotOp::verify()

LogicalResult tensorrt::MatrixMultiplyOp::verify() {
  // MatrixMultiply impl start
  Value input0 = this->getInput0();
  auto input0Type = cast<RankedTensorType>(input0.getType());
  Value input1 = this->getInput1();
  auto input1Type = cast<RankedTensorType>(input1.getType());
  if (input0Type.getRank() == 1 || input1Type.getRank() == 1) {
    if (input0Type.getRank() == 1 && this->getOp0() != MatrixOperation::kVECTOR)
      return emitOpError("Input 0 has rank one. Expected TRT MatOp kVCETOR ");
    if (input1Type.getRank() == 1 && this->getOp1() != MatrixOperation::kVECTOR)
      return emitOpError("Input 1 has rank one. Expected TRT MatOp kVCETOR ");
  }
  // Validate operand and result shapes.
  if (failed(validateMatMulOperands(*this)))
    return failure();
  // MatrixMultiply impl end
  return success();
} // LogicalResult tensorrt::MatrixMultiplyOp::verify()

LogicalResult tensorrt::ConstantOp::verify() {
  // Constant impl start
  if (getType() != getWeights().getType())
    return emitOpError("expected weights type to match result type");
  // Constant impl end
  return success();
} // LogicalResult tensorrt::ConstantOp::verify()

LogicalResult tensorrt::IdentityOp::verify() {
  // Identity impl start

  // There are three separate rules according to the documentation:
  // (kFLOAT | kHALF | kINT32 | kBOOL | kBFLOAT16) -> (kFLOAT | kHALF | kINT32 |
  // kBOOL | kBFLOAT16)
  // (kFLOAT | kHALF) -> kUINT8
  // kUINT8 -> (kFLOAT | kHALF)

  Type dstElType = getType().getElementType();
  Type srcElType = getInput().getType().getElementType();
  // Rule #2
  if (dstElType.isUnsignedInteger(8) &&
      (!srcElType.isF32() && !srcElType.isF16()))
    return emitOpError(
        "if result element type is ui8, input element type must be f32 or f16");
  // Rule #3
  if (srcElType.isUnsignedInteger(8) &&
      (!dstElType.isF32() && !dstElType.isF16()))
    return emitOpError(
        "if input element type is ui8, result element type must be f32 or f16");

  // Otherwise, Rule#1 is satisfied given existing ODS constraints.

  // Identity impl end
  return success();
} // LogicalResult tensorrt::IdentityOp::verify()

LogicalResult tensorrt::SliceOp::verify() {

  TensorType inputType = getInput().getType();

  // For dynamic values, the ODS verifier will check that the index tensor is
  // rank 1 and static shaped.
  for (TypedValue<TensorType> dynIndexTensor :
       {getStart(), getStride(), getSize()}) {
    if (dynIndexTensor &&
        dynIndexTensor.getType().getDimSize(0) != inputType.getRank())
      return emitOpError(
          "all dynamic index tensors should have a static size equal to "
          "the rank of the input tensor type");
  }

  for (const std::optional<ArrayRef<int32_t>> &it :
       {getStaticStart(), getStaticSize(), getStaticStride()}) {
    if (it.has_value() &&
        static_cast<int64_t>(it->size()) != inputType.getRank())
      return emitOpError(
          "the size of all static index arrays (start, size, stride) should "
          "be equal to the rank of the input tensor type");
  }

  // If a fill value is provided, then the slice mode must be kFILL
  if (getFill() && getMode() != SliceMode::kFILL)
    return emitOpError(
        "when a fill value is provided, the slice \"mode\" must be kFILL");

  // CLAMP, FILL and REFLECT modes do not support bfloat16
  if (inputType.getElementType().isBF16() &&
      (getMode() == SliceMode::kCLAMP || getMode() == SliceMode::kFILL ||
       getMode() == SliceMode::kREFLECT))
    return emitOpError(
        "kCLAMP, kFILL and kREFLECT modes do not support bfloat16 type");
  return success();
}

LogicalResult tensorrt::ParametricReLUOp::verify() {
  const int64_t inputRank = this->getInput().getType().getRank();
  ArrayRef<int64_t> inputShape = this->getInput().getType().getShape();
  ArrayRef<int64_t> slopeShape = this->getSlope().getType().getShape();
  for (int i = 0; i < inputRank; ++i) {
    if (slopeShape[i] != ShapedType::kDynamic &&
        inputShape[i] != ShapedType::kDynamic &&
        slopeShape[i] != inputShape[i] && slopeShape[i] != 1)
      return emitOpError(
          "expected dimensions of slope tensor should be either 1 or "
          "equal to correspond dimension of the input tensor");
  }
  return success();
} // LogicalResult tensorrt::ParametricReLUOp::verify()

LogicalResult tensorrt::ResizeNearestOp::verify() {
  // ResizeNearestOp impl start
  TensorType inputType = getInput().getType();
  TensorType outputType = getType();

  auto outputRank = outputType.getRank();
  const int64_t resizeDim = std::min(static_cast<int64_t>(3), outputRank);
  for (int64_t i = outputRank - 1; i >= 0; --i) {
    if (inputType.isDynamicDim(i) || outputType.isDynamicDim(i))
      continue;
    if (inputType.getDimSize(i) != outputType.getDimSize(i))
      if (outputRank - i > resizeDim)
        return emitOpError("only supports resizing on the innermost min(3, "
                           "rank(input)) dimensions");
  }

  if (getScales().has_value()) {
    if (static_cast<int64_t>(getScales().value().size()) != outputRank)
      return emitOpError("scales parameter must have same number of dimensions "
                         "as input/output");
    for (int i = 0; i < outputRank - resizeDim; i++)
      if (getScales().value()[i] != 1)
        return emitOpError(
            "all scale values except innermost min(3, rank(input)) must be 1");
  }

  if (!getOutputShape()) {
    for (int64_t i = 0; i < resizeDim; ++i) {
      // output dims must be static or
      // scales is given and input dims are static
      if (outputType.isDynamicDim(outputRank - 1 - i) &&
          (inputType.isDynamicDim(outputRank - 1 - i) ||
           !getScales().has_value()))
        return emitOpError(
            "input innermost min(3, rank(input)) dimension that resize on "
            "cannot be dynamic when output_shape parameter is NOT "
            "specified and it cannot be inferred statically");
    }
  }
  return success();
} // LogicalResult tensorrt::ResizeNearestOp::verify()

LogicalResult tensorrt::ResizeLinearOp::verify() {
  // ResizeLinearOp impl start
  TensorType inputType = getInput().getType();
  TensorType outputType = getType();

  auto outputRank = outputType.getRank();
  const int64_t resizeDim = std::min(static_cast<int64_t>(3), outputRank);
  for (int64_t i = outputRank - 1; i >= 0; --i) {
    if (inputType.isDynamicDim(i) || outputType.isDynamicDim(i))
      continue;
    if (inputType.getDimSize(i) != outputType.getDimSize(i))
      if (outputRank - i > resizeDim)
        return emitOpError("only supports resizing on the innermost min(3, "
                           "rank(input)) dimensions");
  }

  if (getScales().has_value()) {
    if (static_cast<int64_t>(getScales().value().size()) != outputRank)
      return emitOpError("scales parameter must have same number of dimensions "
                         "as input/output");
    for (int i = 0; i < outputRank - resizeDim; i++)
      if (getScales().value()[i] != 1)
        return emitOpError(
            "all scale values except innermost min(3, rank(input)) must be 1");
  }

  if (!getOutputShape()) {
    for (int64_t i = 0; i < resizeDim; ++i) {
      // output dims must be static or
      // scales is given and input dims are static
      if (outputType.isDynamicDim(outputRank - 1 - i) &&
          (inputType.isDynamicDim(outputRank - 1 - i) ||
           !getScales().has_value()))
        return emitOpError(
            "input innermost min(3, rank(input)) dimension that resize on "
            "cannot be dynamic when output_shape parameter is NOT "
            "specified and it cannot be inferred statically");
    }
  }
  // ResizeLinearOp impl end
  return success();
} // LogicalResult tensorrt::ResizeLinearOp::verify()

LogicalResult tensorrt::ResizeCubicOp::verify() {
  // ResizeLinearOp impl start
  TensorType inputType = getInput().getType();
  TensorType outputType = getType();

  auto outputRank = outputType.getRank();
  if (outputRank < 2)
    return emitOpError(
        "does not support resizing on a tensor that has rank < 2");

  for (int64_t i = outputRank - 3; i >= 0; --i) {
    if (inputType.isDynamicDim(i) || outputType.isDynamicDim(i))
      continue;
    if (inputType.getDimSize(i) != outputType.getDimSize(i))
      return emitOpError(
          "only supports resizing on the innermost 2 dimensions");
  }

  if (getScales().has_value()) {
    if (static_cast<int64_t>(getScales().value().size()) != outputRank)
      return emitOpError("scales parameter must have same number of dimensions "
                         "as input/output");
    for (int i = 0; i < inputType.getRank() - 2; i++)
      if (getScales().value()[i] != 1)
        return emitOpError("all scale values except 2 innermost must be 1");
  }

  if (!getOutputShape()) {
    for (int64_t i = 0; i < 2; ++i) {
      // output dims must be static or
      // scales is given and input dims are static
      if (outputType.isDynamicDim(outputRank - 1 - i) &&
          (inputType.isDynamicDim(outputRank - 1 - i) ||
           !getScales().has_value()))
        return emitOpError(
            "input innermost 2 dimensions that resize on "
            "cannot be dynamic when output_shape parameter is NOT "
            "specified and it cannot be inferred statically");
    }
  }

  // ResizeLinearOp impl end
  return success();
} // LogicalResult tensorrt::ResizeOp::verify()

//===----------------------------------------------------------------------===//
// OpaquePluginOp
//===----------------------------------------------------------------------===//

LogicalResult OpaquePluginOp::verify() {
  if (getCreatorFunc() && !getDsoPath())
    return emitOpError() << getCreatorFuncAttrName() << " is provided but "
                         << getDsoPathAttrName() << " was not specified";
  return success();
}

LogicalResult OpaquePluginOp::verifyRegions() {
  Region &shapesRegion = getShapesRegion();
  if (getShapesRegion().empty())
    return success();

  if (shapesRegion.getBlocks().size() > 1)
    return emitOpError()
           << "expected that 'shapes' region contain 0 or 1 blocks";

  unsigned numScalarBlockArgsExpected = 0;
  unsigned numScalarYieldsExpected = 0;
  for (Value v : getInputs()) {
    RankedTensorType tensorType = cast<RankedTensorType>(v.getType());
    numScalarBlockArgsExpected += tensorType.getRank();
  }
  for (Value v : getResults()) {
    RankedTensorType tensorType = cast<RankedTensorType>(v.getType());
    numScalarYieldsExpected += tensorType.getRank();
  }

  // There should only be 'arith' operations in the region. This can
  // change in the future if required.
  for (Operation &op : shapesRegion.front().without_terminator()) {
    if (!isa<arith::ArithDialect>(op.getDialect()))
      return emitOpError() << "expected only 'arith' dialect ops and "
                              "'tensorrt.yield' (terminator) in the "
                              "'shapes' region, but an op of type '"
                           << op.getName() << "' is present";
  }

  if (shapesRegion.getNumArguments() != numScalarBlockArgsExpected ||
      !llvm::all_of(shapesRegion.getArgumentTypes(),
                    [](Type t) { return t.isInteger(64); })) {
    return emitOpError() << "expected " << numScalarBlockArgsExpected
                         << " i64 block arguments but got "
                         << shapesRegion.getNumArguments()
                         << " arguments  of types "
                         << TypeRange(getShapesRegion().getArgumentTypes());
  }

  // Verify the count and type of the yielded arguments. In addition, we can use
  // this opportunity to check whether there is a mismatch static shapes
  // calculated.
  auto yieldOp = cast<tensorrt::YieldOp>(shapesRegion.front().getTerminator());
  if (yieldOp->getNumOperands() != numScalarYieldsExpected ||
      !llvm::all_of(yieldOp->getOperandTypes(),
                    [](Type t) { return t.isInteger(64); })) {
    return emitOpError()
           << "expected " << numScalarYieldsExpected
           << " i64 values to be yielded from the 'shapes' region but got "
           << yieldOp.getNumOperands() << " values of types "
           << yieldOp.getOperandTypes();
  }

  if (std::optional<SmallVector<ShapedTypeComponents>> inferredComponents =
          inferShapeComponentsFromShapesRegion()) {
    if (failed(tensorrt::detail::verifyInferredTensorTypesWithPartialInfo(
            *this,
            [&](MLIRContext *, std::optional<Location>, ValueShapeRange,
                DictionaryAttr, OpaqueProperties, RegionRange,
                SmallVectorImpl<ShapedTypeComponents> &components)
                -> LogicalResult {
              components = std::move(*inferredComponents);
              return success();
            },
            /*shapesEqualUpToDynamicDim=*/true)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::SelectOp::verify() {
  // Select impl start

  // Select impl end
  return success();
} // LogicalResult tensorrt::SelectOp::verify()

LogicalResult tensorrt::AssertionOp::verify() {
  // Assertion impl start
  const int64_t conditionRank = getCondition().getType().getRank();
  if (conditionRank > 1)
    return emitOpError("expected condition to be of rank 0 or 1");

  // Assertion impl end
  return success();
} // LogicalResult tensorrt::AssertionOp::verify()

LogicalResult tensorrt::DequantizeOp::verify() {
  auto inputType = getInput().getType();
  const int64_t inputRank = getInput().getType().getRank();
  auto scale = getScale().getType();
  if (auto axis = getAxis()) {
    int32_t axisI32 = *axis;
    if (axisI32 >= inputRank || axis < 0)
      return emitOpError("expected axis to be non-negative and less than ")
             << inputRank;
    // Scale is 1D for per-channel quantization
    if (scale.getRank() != 1)
      return emitOpError("if axis is provided, scale must be a 1D tensor for "
                         "per channel quantization");
    // size of scales must match the size of input on quantization
    // axis
    if (!inputType.isDynamicDim(axisI32) &&
        (inputType.getDimSize(axisI32) != scale.getDimSize(0)))
      return emitOpError("expected the scales size to match the dequantization "
                         "axis of input tensor");
  } else {
    if (scale.getRank() == 2) {
      if (!inputType.getElementType().isInteger(4))
        return emitOpError(
            "2D scale is supported only for dequantizing INT4 input");
      if (inputType.getRank() != 2)
        return emitOpError("INT4 block-dequantization needs 2D input.");
      if (!inputType.isDynamicDim(scale.getRank() - 1) &&
          (scale.getDimSize(scale.getRank() - 1) !=
           inputType.getDimSize(scale.getRank() - 1)))
        return emitOpError(
            " The last dimension of scale and input tensors must "
            "match (INT4 block-dequantization is performed over the first "
            "dimension)");
    } else {
      if (scale.getRank() != 0)
        return emitOpError(
            "if no axis is provided and input is not INT4, dequantization is "
            "per-tensor. In this case, `scale` must be a scalar i.e. 0 dim "
            "tensor.");
    }
  }
  return success();
} // LogicalResult tensorrt::DequantizeOp::verify()

LogicalResult tensorrt::ScatterOp::verify() {
  auto inputDataType = getData().getType();
  auto indicesDataType = getIndices().getType();
  auto updatesDataType = getUpdates().getType();

  int64_t inputRank = inputDataType.getRank();
  int64_t indicesRank = indicesDataType.getRank();
  int64_t updatesRank = updatesDataType.getRank();

  auto indicesShape = indicesDataType.getShape();
  if (indicesShape.empty())
    return emitOpError("expected indices to have rank >= 1");

  int64_t indicesLastAxisShape = indicesShape.back();
  int64_t expectedUpdatesRank =
      inputRank + indicesRank - indicesLastAxisShape - 1;

  if (indicesLastAxisShape > inputRank)
    return emitOpError("indexing tuple rank cannot be larger than input rank");
  if (updatesRank != expectedUpdatesRank)
    return emitOpError("expected updates tensor rank to be ")
           << expectedUpdatesRank;
  return success();
}

LogicalResult tensorrt::ScatterElementsOp::verify() {
  const int64_t inputDataRank = getData().getType().getRank();
  if (inputDataRank < 1)
    return emitOpError("expected data to have rank >= 1, got ")
           << inputDataRank;

  if (getAxis()) {
    int64_t axis = static_cast<int64_t>(getAxis().value());

    if (axis >= inputDataRank || axis < 0)
      return emitOpError("expected axis to be in the range [0, ")
             << inputDataRank << ")";
  }

  return success();
} // LogicalResult tensorrt::ScatterElementsOp::verify()

LogicalResult tensorrt::QuantizeOp::verify() {
  // axis must be non-negative and must be smaller than input's Rank
  auto inputType = getInput().getType();
  auto scale = getScale().getType();
  const int64_t inputRank = getInput().getType().getRank();
  if (auto axis = getAxis()) {
    int32_t axisI32 = *axis;
    if (axisI32 >= inputRank || axis < 0)
      return emitOpError("expected axis to be non-negative and less than ")
             << inputRank;
    // Scale is 1D for per-channel quantization
    if (scale.getRank() != 1)
      return emitOpError("if axis is provided, scale must be a 1D tensor for "
                         "per channel quantization");
    // size of scales must match the size of input on quantization
    // axis
    if (!inputType.isDynamicDim(axisI32) &&
        inputType.getDimSize(axisI32) != scale.getDimSize(0))
      return emitOpError("expected the scales size to match the "
                         "quantization "
                         "axis of input tensor");
  } else {
    // If scale is 2D, its INT4 block quantization
    if (scale.getRank() == 2) {
      if (!getType().getElementType().isInteger(4))
        return emitOpError(
            "2D scale is supported only for quantizing INT4 output");
      if (inputType.getRank() != 2)
        return emitOpError("INT4 block-quantization needs 2D input.");
      if (!inputType.isDynamicDim(scale.getRank() - 1) &&
          (scale.getDimSize(scale.getRank() - 1) !=
           inputType.getDimSize(scale.getRank() - 1)))
        return emitOpError(
            " The last dimension of scale and input tensors must "
            "match (INT4 block-quantization is performed over the first "
            "dimension)");
    } else {
      // It has to be per-tensor quantization now
      if (scale.getRank() != 0)
        return emitOpError(
            "if no axis is provided and input is not INT4, quantization is "
            "per-tensor. In this case, `scale` must be a scalar i.e. 0 dim "
            "tensor.");
    }
  }
  return success();
} // LogicalResult tensorrt::QuantizeOp::verify()

/// Return the disjunction of the callables invoked with the element type of
/// `t`.
template <typename... Callable>
bool hasElementType(TensorType t, Callable... funcs) {
  return (funcs(t.getElementType()) || ...);
}

/// Checks that the given unary operation obeys type constraint requirements.
/// Each unary operation type has a different set of allowed scalar element
/// types.
/// TODO: this should go away when we break out the TensorRT unary operation
/// into separate ops.
static LogicalResult verifyAllowedDataTypes(UnaryOp op) {
  // TensorRT unary op doesn't accept scalar.
  if (cast<RankedTensorType>(op.getInput().getType()).getRank() == 0)
    return op->emitOpError("TensorRT Unary ops need at least 1D input");

  // Names of the lambdas appear in the error message using the macro below.
  auto I8 = [](Type t) { return isTensorRTInt8Type(t); };
  auto I32 = [](Type t) { return t.isInteger(32); };
  auto I64 = [](Type t) { return t.isInteger(64); };
  auto I1 = [](Type t) { return t.isInteger(1); };
  auto F16 = [](Type t) { return t.isF16(); };
  auto F32 = [](Type t) { return t.isF32(); };
  auto BF16 = [](Type t) { return t.isBF16(); };

#define HANDLE_CASE(unaryEnumCase, ...)                                        \
  case UnaryOperation::k##unaryEnumCase:                                       \
    return hasElementType(op.getType(), __VA_ARGS__)                           \
               ? success()                                                     \
               : op->emitOpError() << "expected element type to be one of "    \
                                      "the following: " #__VA_ARGS__;

  switch (op.getUnaryOperation()) {
    HANDLE_CASE(ABS, I8, I32, I64, F16, F32, BF16)
    HANDLE_CASE(ACOS, F16, F32)
    HANDLE_CASE(ACOSH, F16, F32)
    HANDLE_CASE(ASIN, F16, F32)
    HANDLE_CASE(ASINH, F16, F32)
    HANDLE_CASE(ATAN, F16, F32)
    HANDLE_CASE(ATANH, F16, F32)
    HANDLE_CASE(CEIL, F16, F32, BF16)
    HANDLE_CASE(COS, F16, F32, BF16)
    HANDLE_CASE(COSH, F16, F32)
    HANDLE_CASE(ERF, F16, F32, BF16)
    HANDLE_CASE(EXP, F16, F32, BF16)
    HANDLE_CASE(FLOOR, F16, F32, BF16)
    HANDLE_CASE(LOG, F16, F32, BF16)
    HANDLE_CASE(NEG, I8, I32, I64, F16, F32, BF16)
    HANDLE_CASE(RECIP, F16, F32, BF16)
    HANDLE_CASE(ROUND, F16, F32, BF16)
    HANDLE_CASE(SIN, F16, F32, BF16)
    HANDLE_CASE(SINH, F16, F32)
    HANDLE_CASE(SQRT, F16, F32, BF16)
    HANDLE_CASE(NOT, I1)
    HANDLE_CASE(SIGN, I8, I32, I64, F16, F32, BF16)
    HANDLE_CASE(TAN, F16, F32)
  }
  llvm_unreachable("unhandled unary operation type");

#undef HANDLE_CASE
}

LogicalResult tensorrt::UnaryOp::verify() {
  return verifyAllowedDataTypes(*this);
}
