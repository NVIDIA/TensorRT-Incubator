//===- ShapeInference.cpp -- ----------------------------------------------===//
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
/// Type inference interfaces for TensorRT operations.
///
//===----------------------------------------------------------------------===//
#include "EinsumHelper.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;
using namespace mlir::tensorrt;

//===----------------------------------------------------------------------===//
// ActivationOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::ActivationOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ActivationOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
  inferredReturnShapes.emplace_back(
      /*vec=*/inputType.getShape(),
      /*elementType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::UnaryOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  UnaryOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
  inferredReturnShapes.emplace_back(
      /*vec=*/inputType.getShape(),
      /*elementType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// ElementWiseOp
//===----------------------------------------------------------------------===//

/// Return the result type of an elementwise op given the kind of binary
/// operation and the first input element type. It is either a binary element
/// type or the same element type as the input.
static Type getElementWiseResultElementTypeType(MLIRContext *ctx,
                                                ElementWiseOperation opType,
                                                Type inputElementType) {
  Type boolType = IntegerType::get(ctx, 1);
  switch (opType) {
  case ElementWiseOperation::kGREATER:
  case ElementWiseOperation::kEQUAL:
  case ElementWiseOperation::kLESS:
  case ElementWiseOperation::kAND:
  case ElementWiseOperation::kOR:
  case ElementWiseOperation::kXOR:
    return boolType;
  case ElementWiseOperation::kDIV:
  case ElementWiseOperation::kFLOOR_DIV:
  case ElementWiseOperation::kMAX:
  case ElementWiseOperation::kMIN:
  case ElementWiseOperation::kPROD:
  case ElementWiseOperation::kSUB:
  case ElementWiseOperation::kSUM:
  case ElementWiseOperation::kPOW:
    return inputElementType;
  }
  llvm_unreachable("unhandled or invalid elementwise operation type");
  return nullptr;
}

LogicalResult tensorrt::ElementWiseOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ElementWiseOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto input1Type = cast<RankedTensorType>(adaptor.getInput1().getType());
  auto input2Type = cast<RankedTensorType>(adaptor.getInput2().getType());

  FailureOr<SmallVector<int64_t>> expectedShape =
      getBroadcastedShape(input1Type, input2Type);
  if (failed(expectedShape))
    return emitOptionalError(loc, "failed to determine expected shape");

  // The element type depends on the elementwise operation.
  inferredReturnShapes.emplace_back(
      /*vec=*/*expectedShape,
      /*elementType=*/getElementWiseResultElementTypeType(
          ctx, adaptor.getElementwiseOperation(), input1Type.getElementType()));

  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::CastOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  CastOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto rtt = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!rtt)
    return emitOptionalError(loc, "expected input to be a ranked tensor");
  inferredReturnShapes.emplace_back(/*vec=*/rtt.getShape(),
                                    /*elementType=*/nullptr);
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::ConcatenationOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // Use the `Adaptor` class to interpret `operands`, `attributes`, and
  // `regions`.
  ConcatenationOp::Adaptor adaptor(operands, attributes, properties, regions);

  SmallVector<RankedTensorType> inputTypes =
      llvm::to_vector(llvm::map_range(adaptor.getInputs(), [](Value v) {
        return cast<RankedTensorType>(v.getType());
      }));

  // Perform some simple verification since this can run before op verifier.
  if (adaptor.getInputs().empty())
    return emitOptionalError(loc, "expected at least one input");

  // Check inputs have correct rank.
  if (!llvm::all_of(inputTypes, [&](RankedTensorType t) {
        return t.getRank() == inputTypes.front().getRank();
      }))
    return emitOptionalError(loc, "all inputs must have equal rank");

  // Check axis.
  RankedTensorType input0Type = inputTypes.front();
  const int32_t axis = static_cast<int32_t>(adaptor.getAxis());
  if (axis < 0 || axis > input0Type.getRank())
    return emitOptionalError(
        loc, "expected axis to be in the range of [0, input rank)");

  // Calculate the result shape.
  SmallVector<int64_t> resultShape(inputTypes.front().getShape());
  resultShape[axis] = 0;
  for (RankedTensorType rtt : inputTypes) {
    if (ShapedType::isDynamic(resultShape[axis]) ||
        ShapedType::isDynamic(rtt.getDimSize(axis))) {
      resultShape[axis] = ShapedType::kDynamic;
      continue;
    }
    resultShape[axis] += rtt.getDimSize(axis);
  }

  // For each result, push back the expected return type.
  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/input0Type.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// ConvolutionOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::ConvolutionOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ConvolutionOp::Adaptor adaptor(operands, attributes, properties, regions);
  RankedTensorType inputType =
      cast<RankedTensorType>(adaptor.getInput().getType());
  auto inpShapeComp = ConvDeconvPoolTensorShapeComponents::createFromInputShape(
      inputType.getShape());
  if (failed(inpShapeComp))
    return emitOptionalError(
        loc, "failed to create input shape components. Input must be 4D or 5D");

  ArrayRef<int64_t> stride = adaptor.getStride();
  ArrayRef<int64_t> prePadding = adaptor.getPrePadding();
  ArrayRef<int64_t> postPadding = adaptor.getPostPadding();
  std::optional<ArrayRef<int64_t>> dilation = adaptor.getDilation();
  int32_t numGroups = adaptor.getNumGroups();

  auto layerComp = ConvDeconvPoolLayerComponents()
                       .setStride(stride)
                       .setPrePadding(prePadding)
                       .setPostPadding(postPadding)
                       .setDilation(dilation)
                       .setNumGroups(numGroups);

  ArrayRef<int64_t> kernelShape =
      adaptor.getKernelStatic().has_value()
          ? adaptor.getKernelStatic()->getShapedType().getShape()
          : cast<RankedTensorType>(adaptor.getKernel().getType()).getShape();

  auto kernelShapeComp = ConvDeconvKernelShapeComponents::createFromKernelShape(
      kernelShape, /*isOpConv=*/true, numGroups);
  if (failed(kernelShapeComp))
    return emitOptionalError(
        loc,
        "failed to create kernel shape components. Kernel must be 4D or 5D");
  auto resultShape =
      getConvDeconvOpOutputShape(*inpShapeComp, *kernelShapeComp, layerComp);
  if (failed(resultShape))
    return emitOptionalError(
        loc, "failed to compute output shape of convolution operation");
  inferredReturnShapes.emplace_back(
      /*vec=*/*resultShape->getShape(),
      /*elementType=*/inputType.getElementType());

  return success();
}

//===----------------------------------------------------------------------===//
// EinsumOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::EinsumOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  EinsumOp::Adaptor adaptor(operands, attributes, properties, regions);

  auto errorFn = [&](std::optional<Location> loc, const Twine &message) {
    return emitOptionalError(loc, message);
  };
  auto outputShape = tensorrt::einsum::inferOutputShape(
      adaptor.getEquation(), adaptor.getInputs().getType(), loc, errorFn);
  if (failed(outputShape))
    return failure();

  inferredReturnShapes.emplace_back(
      /*vec=*/*outputShape,
      /*elementType=*/cast<RankedTensorType>(
          adaptor.getInputs().front().getType())
          .getElementType());

  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::GatherOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  GatherOp::Adaptor adaptor(operands, attributes, properties, regions);

  RankedTensorType inputType =
      cast<RankedTensorType>(adaptor.getData().getType());
  int64_t axis = adaptor.getAxis();
  int64_t numBroadcastDims = adaptor.getNumBroadcastDims();
  if (axis < 0 || axis >= inputType.getRank())
    return emitOptionalError(loc, "axis must obey 0 <= axis < rank(input).");
  RankedTensorType indices =
      cast<RankedTensorType>(adaptor.getIndices().getType());
  SmallVector<int64_t> resultShape;
  resultShape.reserve(inputType.getRank() + indices.getRank() - 1 -
                      numBroadcastDims);
  FailureOr<SmallVector<int64_t>> broadcastedDims =
      getBroadcastedShape(inputType.getShape().take_front(numBroadcastDims),
                          indices.getShape().take_front(numBroadcastDims));
  if (failed(broadcastedDims))
    return emitOptionalError(
        loc, "Failed to broadcast numBroadcastDims dimensions.");

  llvm::append_range(resultShape, *broadcastedDims);
  llvm::append_range(
      resultShape,
      inputType.getShape().slice(numBroadcastDims, axis - numBroadcastDims));
  llvm::append_range(resultShape,
                     indices.getShape().drop_front(numBroadcastDims));
  llvm::append_range(resultShape, inputType.getShape().drop_front(axis + 1));

  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// GatherNdOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::GatherNdOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  GatherNdOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto inputType = cast<RankedTensorType>(adaptor.getData().getType());
  auto indicesType = cast<RankedTensorType>(adaptor.getIndices().getType());
  SmallVector<int64_t> resultShape;

  const int64_t r = inputType.getRank();
  const int64_t q = indicesType.getRank();

  const int64_t indexVectorSize = indicesType.getShape().back();

  if (r < 1 || q < 1)
    return emitOptionalError(loc, "input rank and indices rank must be >= 1");
  if (ShapedType::isDynamic(indexVectorSize) || indexVectorSize == 0 ||
      indexVectorSize > r)
    return emitOptionalError(loc,
                             "the extent of the last dimension of 'indices' "
                             "shape must be greater than zero and "
                             "less-than-or-equal-to 'data' rank");

  resultShape.reserve(q + r - indicesType.getShape().back() - 1);
  llvm::append_range(resultShape, indicesType.getShape().drop_back(1));
  llvm::append_range(resultShape,
                     inputType.getShape().take_back(r - indexVectorSize));
  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// GatherElementsOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::GatherElementsOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  GatherElementsOp::Adaptor adaptor(operands, attributes, properties, regions);
  RankedTensorType inputType =
      cast<RankedTensorType>(adaptor.getData().getType());
  RankedTensorType indicesType =
      cast<RankedTensorType>(adaptor.getIndices().getType());
  inferredReturnShapes.emplace_back(
      /*vec=*/indicesType.getShape(),
      /*elementsType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::IdentityOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  IdentityOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto rtt = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!rtt)
    return emitOptionalError(loc, "expected input to be a ranked tensor");
  inferredReturnShapes.emplace_back(/*vec=*/rtt.getShape(),
                                    /*elementType=*/nullptr);
  return success();
}

//===----------------------------------------------------------------------===//
// RandomNormalOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::RandomNormalOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  RandomNormalOp::Adaptor adaptor(operands, attributes, properties, regions);

  if (!adaptor.getShape()) {
    inferredReturnShapes.emplace_back(ShapedTypeComponents());
    return success();
  }

  RankedTensorType shapeType =
      cast<RankedTensorType>(adaptor.getShape().getType());

  if (shapeType.getRank() != 1 || shapeType.isDynamicDim(0))
    return emitOptionalError(
        loc, "shape input should be rank 1 with static dim size");

  SmallVector<int64_t> resultShape(shapeType.getDimSize(0),
                                   ShapedType::kDynamic);

  inferredReturnShapes.emplace_back(/*vec=*/resultShape,
                                    /*elementType=*/nullptr);
  return success();
}

//===----------------------------------------------------------------------===//
// LinspaceOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::LinspaceOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  LinspaceOp::Adaptor adaptor(operands, attributes, properties, regions);

  if (!adaptor.getShape()) {
    inferredReturnShapes.emplace_back(ShapedTypeComponents());
    return success();
  }

  RankedTensorType shapeType =
      cast<RankedTensorType>(adaptor.getShape().getType());

  if (shapeType.getRank() != 1 || shapeType.isDynamicDim(0))
    return emitOptionalError(
        loc, "shape input should be rank 1 with static dim size");

  SmallVector<int64_t> resultShape(shapeType.getDimSize(0),
                                   ShapedType::kDynamic);

  inferredReturnShapes.emplace_back(/*vec=*/resultShape,
                                    /*elementType=*/nullptr);
  return success();
}

//===----------------------------------------------------------------------===//
// RandomUniformOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::RandomUniformOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  RandomUniformOp::Adaptor adaptor(operands, attributes, properties, regions);

  if (!adaptor.getShape()) {
    inferredReturnShapes.emplace_back(ShapedTypeComponents());
    return success();
  }

  RankedTensorType shapeType =
      cast<RankedTensorType>(adaptor.getShape().getType());

  if (shapeType.getRank() != 1 || shapeType.isDynamicDim(0))
    return emitOptionalError(
        loc, "shape input should be rank 1 with static dim size");

  SmallVector<int64_t> resultShape(shapeType.getDimSize(0),
                                   ShapedType::kDynamic);

  inferredReturnShapes.emplace_back(/*vec=*/resultShape,
                                    /*elementType=*/nullptr);
  return success();
}

//===----------------------------------------------------------------------===//
// PoolingOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::PoolingOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  PoolingOp::Adaptor adaptor(operands, attributes, properties, regions);

  RankedTensorType inputType =
      cast<RankedTensorType>(adaptor.getInput().getType());
  auto inpShapeComp = ConvDeconvPoolTensorShapeComponents::createFromInputShape(
      inputType.getShape());
  if (failed(inpShapeComp))
    return emitOptionalError(
        loc, "failed to create input shape components. Input must be 4D or 5D");

  int64_t spatialDimsSize = inputType.getRank() - 2;
  ArrayRef<int64_t> windowSize = adaptor.getWindowSize();
  ArrayRef<int64_t> stride = adaptor.getStride();
  ArrayRef<int64_t> prePadding = adaptor.getPrePadding();
  ArrayRef<int64_t> postPadding = adaptor.getPostPadding();
  if ((static_cast<int64_t>(windowSize.size()) != spatialDimsSize) ||
      (static_cast<int64_t>(stride.size()) != spatialDimsSize) ||
      (static_cast<int64_t>(prePadding.size()) != spatialDimsSize) ||
      (static_cast<int64_t>(postPadding.size()) != spatialDimsSize))
    return emitOptionalError(
        loc, "\"windowSize\", \"stride\" ,\"prePadding\", and \"postPadding\", "
             "should have size equal to number of spatial dimensions.");
  auto layerComp = ConvDeconvPoolLayerComponents()
                       .setStride(stride)
                       .setPoolingWindow(windowSize)
                       .setPrePadding(prePadding)
                       .setPostPadding(postPadding);

  auto resultShape = getPoolingOpOutputShape(*inpShapeComp, layerComp);
  if (failed(resultShape))
    return emitOptionalError(
        loc, "failed to compute output shape of pooling operation");
  inferredReturnShapes.emplace_back(
      /*vec=*/*resultShape->getShape(),
      /*elementType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::ReduceOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ReduceOp::Adaptor adaptor(operands, attributes, properties, regions);
  bool keepDims = adaptor.getKeepDimensions();
  auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
  ArrayRef<int64_t> reduceAxes = adaptor.getReduceAxes();

  // Calculate the result shape.
  SmallVector<int64_t> resultShape;
  resultShape.reserve(inputType.getRank());
  if (keepDims) {
    llvm::append_range(resultShape, inputType.getShape());
    for (int64_t index : reduceAxes)
      resultShape[index] = 1;
  } else {
    for (int i = 0; i < inputType.getRank(); ++i) {
      if (llvm::find(reduceAxes, i) == reduceAxes.end())
        resultShape.push_back(inputType.getDimSize(i));
    }
  }
  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::SelectOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  SelectOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto thenInputType = cast<RankedTensorType>(adaptor.getThenInput().getType());
  auto elseInputType = cast<RankedTensorType>(adaptor.getElseInput().getType());
  auto condInputType = cast<RankedTensorType>(adaptor.getCondition().getType());

  FailureOr<SmallVector<int64_t>> expectedShape =
      getBroadcastedShape({thenInputType.getShape(), elseInputType.getShape(),
                           condInputType.getShape()});
  if (failed(expectedShape))
    return emitOptionalError(loc, "failed to determine expected shape");

  // The result type is the same with the thenInput/elseInput element type.
  inferredReturnShapes.emplace_back(
      /*vec=*/*expectedShape,
      /*elementType=*/thenInputType.getElementType());

  return success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::SliceOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  SliceOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!inputType)
    return emitOptionalError(loc, "expected input to be a ranked tensor");

  // In the static size case, we are given the shape. The other parameters don't
  // matter.
  if (std::optional<ArrayRef<int32_t>> shape = adaptor.getStaticSize()) {
    inferredReturnShapes.emplace_back(
        /*vec=*/llvm::to_vector(llvm::map_range(
            *shape, [](int32_t x) { return static_cast<int64_t>(x); })),
        /*elementType=*/inputType.getElementType());
    return success();
  }

  // Otherwise, we can only infer the rank.
  auto sizeTensorType = dyn_cast<RankedTensorType>(adaptor.getSize().getType());
  if (!sizeTensorType || sizeTensorType.getRank() != 1 ||
      sizeTensorType.isDynamicDim(0))
    return emitOptionalError(
        loc, "expected size operand to be 1D tensor of known size");
  inferredReturnShapes.emplace_back(
      /*vec=*/SmallVector<int64_t>(sizeTensorType.getDimSize(0),
                                   ShapedType::kDynamic),
      /*elementType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// SoftMaxOp
//===----------------------------------------------------------------------===//
LogicalResult tensorrt::SoftMaxOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  SoftMaxOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
  inferredReturnShapes.emplace_back(
      /*vec=*/inputType.getShape(),
      /*elementType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// OneHotOp
//===----------------------------------------------------------------------===//
LogicalResult tensorrt::OneHotOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  OneHotOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto indicesType = cast<RankedTensorType>(adaptor.getIndices().getType());
  const int64_t indicesRank = indicesType.getRank();
  int64_t axis = adaptor.getAxis();

  if ((axis > indicesRank) || (axis < -indicesRank - 1))
    return emitOptionalError(
        loc,
        "expected axis to be in the range [-rank(indices)-1, rank(indices)]");

  // Convert a negative axis to a positive number
  // e.g., when indicesRank=4, the valid range for axis is [-5,4].
  // axis = -1 becomes axis = 4
  // axis = -5 becoems axis = 0
  axis = (axis + (indicesRank + 1)) % (indicesRank + 1);
  auto valuesType = cast<RankedTensorType>(adaptor.getValues().getType());
  const int64_t valuesRank = valuesType.getRank();
  if (valuesRank != 1)
    return emitOptionalError(loc, "expected values to be of rank 1");

  const int64_t valuesDimSize = valuesType.getDimSize(0);
  if (valuesDimSize != 2)
    return emitOptionalError(loc, "expected values to have two elements");

  // Calculate the result shape.
  SmallVector<int64_t> resultShape(indicesType.getShape());
  resultShape.insert(resultShape.begin() + axis, ShapedType::kDynamic);

  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/valuesType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// RaggedSoftMaxOp
//===----------------------------------------------------------------------===//
LogicalResult tensorrt::RaggedSoftMaxOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  RaggedSoftMaxOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
  const int64_t inputRank = inputType.getRank();
  auto boundsRank =
      cast<RankedTensorType>(adaptor.getBounds().getType()).getRank();

  // As of TRT 8.6.10.0, `input` and `bounds` must be a 3D tensor in the
  // explicit batch mode.
  if (!((inputRank == 3) && (boundsRank == 3)))
    return emitOptionalError(loc, "expected input and bounds to be of rank 3");

  inferredReturnShapes.emplace_back(
      /*vec=*/inputType.getShape(),
      /*elementType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// MatrixMultiplyOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::MatrixMultiplyOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  MatrixMultiplyOp::Adaptor adaptor(operands, attributes, properties, regions);

  TensorType input0Type = cast<TensorType>(adaptor.getInput0().getType());
  TensorType input1Type = cast<TensorType>(adaptor.getInput1().getType());
  MatrixOperation input0MatOp = adaptor.getOp0();
  MatrixOperation input1MatOp = adaptor.getOp1();

  ArrayRef<int64_t> input0CollectionDims =
      tensorrt::MatrixMultiplyOp::getCollectionDimsImpl(input0Type,
                                                        input0MatOp);
  ArrayRef<int64_t> input1CollectionDims =
      tensorrt::MatrixMultiplyOp::getCollectionDimsImpl(input1Type,
                                                        input1MatOp);
  if (input0CollectionDims.size() != input1CollectionDims.size())
    return emitOptionalError(
        loc, "\"input0\" and \"input1\" number of collection dims "
             "doesn't match");
  // Broadcast the collection (batch) dimensions.
  FailureOr<SmallVector<int64_t>> outputShape =
      getBroadcastedShape(input0CollectionDims, input1CollectionDims);
  if (failed(outputShape))
    return emitOptionalError(
        loc, "collection (batch) dimensions of \"input0\" and \"input1\""
             " are not broadcastable");

  // Add contraction output shapes for both inputs considering transpose
  // operation.
  if (input0MatOp != MatrixOperation::kVECTOR) {
    auto input0ContractionShape = input0Type.getShape().take_back(2);
    int64_t parDim = input0MatOp == MatrixOperation::kNONE ? 0 : 1;
    outputShape->push_back(input0ContractionShape[parDim]);
  }

  if (input1MatOp != MatrixOperation::kVECTOR) {
    auto input1ContractionShape = input1Type.getShape().take_back(2);
    int64_t parDim = input1MatOp == MatrixOperation::kNONE ? 1 : 0;
    outputShape->push_back(input1ContractionShape[parDim]);
  }

  inferredReturnShapes.emplace_back(
      /*vec=*/*outputShape,
      /*elementsType=*/input0Type.getElementType());

  return success();
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//
LogicalResult tensorrt::TopKOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  TopKOp::Adaptor adaptor(operands, attributes, properties, regions);

  auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());

  // Calculate the result shape.
  SmallVector<int64_t> resultShape(inputType.getShape());
  resultShape[adaptor.getAxis()] = adaptor.getK();

  // For each result, push back the expected return type.
  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/inputType.getElementType());

  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/IntegerType::get(ctx, 32));

  return success();
}

//===----------------------------------------------------------------------===//
// PaddingOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::PaddingOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  PaddingOp::Adaptor adaptor(operands, attributes, properties, regions);

  RankedTensorType inputType =
      cast<RankedTensorType>(adaptor.getInput().getType());
  int64_t inputRank = inputType.getRank();
  ArrayRef<int64_t> prePadding = adaptor.getPrePadding();
  ArrayRef<int64_t> postPadding = adaptor.getPostPadding();
  SmallVector<int64_t> resultShape(inputType.getShape());

  if (prePadding.size() != 2 || postPadding.size() != 2)
    return emitOptionalError(
        loc, "padding exactly two innermost dimensions is supported.");

  if (!inputType.isDynamicDim(inputRank - 2))
    resultShape[inputRank - 2] =
        inputType.getDimSize(inputRank - 2) + prePadding[0] + postPadding[0];
  if (!inputType.isDynamicDim(inputRank - 1))
    resultShape[inputRank - 1] =
        inputType.getDimSize(inputRank - 1) + prePadding[1] + postPadding[1];

  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// NonZeroOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::NonZeroOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  NonZeroOp::Adaptor adaptor(operands, attributes, properties, regions);

  RankedTensorType inputType =
      cast<RankedTensorType>(adaptor.getInput().getType());

  SmallVector<int64_t> resultShape(2);
  resultShape[0] = inputType.getRank();
  resultShape[1] = ShapedType::kDynamic;
  inferredReturnShapes.emplace_back(/*vec=*/resultShape,
                                    /*elementType=*/IntegerType::get(ctx, 32));
  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::IfOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  tensorrt::IfOp::Adaptor adaptor(operands, attributes, properties, regions);

  auto trueRegionYieldOp = &(adaptor.getTrueRegion().back().back());
  auto falseRegionYieldOp = &(adaptor.getFalseRegion().back().back());

  if (trueRegionYieldOp->getNumOperands() !=
      falseRegionYieldOp->getNumOperands())
    return emitOptionalError(
        loc, "number of output tensors in true and false regions must be same");

  for (const auto &[idx, regionOutputTensors] : llvm::enumerate(
           llvm::zip(trueRegionYieldOp->getOperands().getType(),
                     falseRegionYieldOp->getOperands().getType()))) {
    auto trueRegionOutRankedTensor =
        cast<RankedTensorType>(std::get<0>(regionOutputTensors));
    auto falseRegionOutRankedTensor =
        cast<RankedTensorType>(std::get<1>(regionOutputTensors));
    if (trueRegionOutRankedTensor.getElementType() !=
            falseRegionOutRankedTensor.getElementType() ||
        !trueRegionOutRankedTensor.getShape().equals(
            falseRegionOutRankedTensor.getShape()))
      return emitOptionalError(loc,
                               "true and false regions must yield equivalent "
                               "types");
    inferredReturnShapes.emplace_back(
        /*vec=*/trueRegionOutRankedTensor.getShape(),
        /*elementType=*/trueRegionOutRankedTensor.getElementType());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::ShapeOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
  SmallVector<int64_t> inputShape = {inputType.getRank()};
  inferredReturnShapes.emplace_back(
      /*vec=*/inputShape,
      /*elementsType=*/IntegerType::get(ctx, 32));
  return success();
}

//===----------------------------------------------------------------------===//
// ParametricReLUOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::ParametricReLUOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {

  ParametricReLUOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto input = cast<RankedTensorType>(adaptor.getInput().getType());
  auto slope = cast<RankedTensorType>(adaptor.getInput().getType());

  LogicalResult isSlopeBroadcastable =
      checkLhsShapeBroadcastableToRhs(slope.getShape(), input.getShape());
  if (failed(isSlopeBroadcastable))
    return emitOptionalError(loc, " \"slope\" for " + getOperationName() +
                                      " must be broadcastable with \"input\"");

  inferredReturnShapes.emplace_back(
      /*vec=*/input.getShape(),
      /*elementType=*/input.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//
LogicalResult tensorrt::ShuffleOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShuffleOp::Adaptor adaptor(operands, attributes, properties, regions);

  RankedTensorType inputType =
      cast<RankedTensorType>(adaptor.getInput().getType());

  // `tensorrt.shuffle` layer applies 3 operations on input in sequence:
  // transpose => reshape(optional) => transpose
  // Verification and shape inference are executed for each individual step

  /// verification and shape inference for 1st transpose
  if (!isPermutationMap(adaptor.getFirstTranspose(), ctx) ||
      static_cast<unsigned>(inputType.getRank()) !=
          adaptor.getFirstTranspose().size())
    return emitOptionalError(
        loc,
        Twine("first transpose array is not a permutation of input rank ") +
            Twine(inputType.getRank()));

  auto firstTransposePerm =
      getAsPermutationMap(ctx, adaptor.getFirstTranspose());

  SmallVector<int64_t> ShapeAfterFirstTranspose =
      applyPermutationMap(firstTransposePerm, inputType.getShape());

  // verification and shape inference for reshape
  SmallVector<int64_t> ShapeAfterReshape = ShapeAfterFirstTranspose;

  if (std::optional<ArrayRef<int64_t>> staticReshape = adaptor.getReshape()) {
    // static reshape
    // Enforce the condition that the reshape specification can only have one -1
    // up front. The type inference procedure depends on this.
    auto numNegOne = llvm::count(*staticReshape, -1);
    if (numNegOne > 1)
      return emitOptionalError(
          loc, "invalid reshape specification - at most one '-1' is allowed");
    // Don't allow -1 and 0 values in placeholder mode.
    if (!adaptor.getZeroIsPlaceholder() && numNegOne > 0 &&
        llvm::count(*staticReshape, 0) > 0)
      return emitOptionalError(
          loc, "invalid reshape specification - 0 and -1 cannot be "
               "used when not in \"zero is placeholder\" mode");

    // Shape inference for static reshape
    FailureOr<SmallVector<int64_t>> reshapedOutput = inferReshapeResultShape(
        RankedTensorType::get(ShapeAfterFirstTranspose,
                              inputType.getElementType()),
        *adaptor.getReshape(), adaptor.getZeroIsPlaceholder(),
        [&](const std::string &msg) -> LogicalResult {
          return emitOptionalError(loc, msg);
        });
    if (failed(reshapedOutput))
      return failure();
    ShapeAfterReshape = reshapedOutput.value();

  } else if (adaptor.getDynamicReshape()) {
    // Dynamic reshape
    ShapeAfterReshape = SmallVector<int64_t>(
        cast<RankedTensorType>(adaptor.getDynamicReshape().getType())
            .getDimSize(0),
        ShapedType::kDynamic);
  }

  // Verification and shape inference for 2nd transpose
  if (!isPermutationMap(adaptor.getSecondTranspose(), ctx) ||
      static_cast<unsigned>(ShapeAfterReshape.size()) !=
          adaptor.getSecondTranspose().size())
    return emitOptionalError(
        loc,
        Twine("second transpose array is not a permutation of reshaped rank ") +
            Twine(ShapeAfterReshape.size()));

  auto secondTransposePerm =
      getAsPermutationMap(ctx, adaptor.getSecondTranspose());

  SmallVector<int64_t> ShapeAfterSecondTranspose = applyPermutationMap(
      secondTransposePerm, ArrayRef<int64_t>(ShapeAfterReshape));

  inferredReturnShapes.emplace_back(
      /*vec=*/ShapeAfterSecondTranspose,
      /*elementType=*/inputType.getElementType());

  return success();
}

//===----------------------------------------------------------------------===//
// DeconvolutionOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::DeconvolutionOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  DeconvolutionOp::Adaptor adaptor(operands, attributes, properties, regions);

  RankedTensorType inputType =
      cast<RankedTensorType>(adaptor.getInput().getType());

  auto inpShapeComp = ConvDeconvPoolTensorShapeComponents::createFromInputShape(
      inputType.getShape());
  if (failed(inpShapeComp))
    return emitOptionalError(
        loc, "failed to create input shape components. Input must be 4D or 5D");
  ArrayRef<int64_t> stride = adaptor.getStride();
  ArrayRef<int64_t> prePadding = adaptor.getPrePadding();
  ArrayRef<int64_t> postPadding = adaptor.getPostPadding();
  std::optional<ArrayRef<int64_t>> dilation = adaptor.getDilation();
  int32_t numGroups = adaptor.getNumGroups();
  auto layerComp = ConvDeconvPoolLayerComponents()
                       .setStride(stride)
                       .setPrePadding(prePadding)
                       .setPostPadding(postPadding)
                       .setDilation(dilation)
                       .setNumGroups(numGroups);
  ArrayRef<int64_t> kernelShape =
      adaptor.getKernelWeightsStatic().has_value()
          ? adaptor.getKernelWeightsStaticAttr().getShapedType().getShape()
          : cast<RankedTensorType>(adaptor.getKernelWeights().getType())
                .getShape();
  auto kernelShapeConstruct =
      ConvDeconvKernelShapeComponents::createFromKernelShape(kernelShape, false,
                                                             numGroups);
  if (failed(kernelShapeConstruct))
    return emitOptionalError(
        loc,
        "failed to create kernel shape components. Kernel must be 4D or 5D");
  auto resultShape = getConvDeconvOpOutputShape(
      *inpShapeComp, *kernelShapeConstruct, layerComp, /*isConv=*/false);
  if (failed(resultShape))
    return emitOptionalError(
        loc, "failed to compute output shape of deconvolution operation");
  inferredReturnShapes.emplace_back(
      /*vec=*/*resultShape->getShape(),
      /*elementType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::TransposeOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  TransposeOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());

  if (!inputType)
    return emitOptionalError(loc, "expected " + getOperationName() +
                                      " input to have ranked tensor type");
  auto perm = adaptor.getPermutation();
  if (perm.getNumResults() != inputType.getRank() || !perm.isPermutation())
    return emitOptionalError(
        loc, Twine("expected \"permutation\" to be a permutation of rank ") +
                 Twine(inputType.getRank()));
  SmallVector<int64_t> resultShape =
      applyPermutationMap(adaptor.getPermutation(), inputType.getShape());
  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementsType=*/inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// ExpandRankOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CollapseRankOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::BroadcastOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!inputType)
    return emitOptionalError(loc, "expected input to be RankedTensorType");
  // In the static shape case, we can only infer the element type, not the rank.
  if (!adaptor.getShape()) {
    inferredReturnShapes.emplace_back(
        /*elementType=*/inputType.getElementType());
    return success();
  }
  // In the dynamic shape case, we can infer the rank.
  auto shapeType = dyn_cast<RankedTensorType>(adaptor.getShape().getType());
  if (!shapeType || shapeType.getRank() != 1 || shapeType.isDynamicDim(0))
    return emitOptionalError(loc,
                             "expected shape to be 1D tensor of known size");
  inferredReturnShapes.emplace_back(
      SmallVector<int64_t>(shapeType.getDimSize(0), ShapedType::kDynamic),
      inputType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// ArgMinOp / ArgMaxOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::ArgMaxOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // Use the `Adaptor` class to interpret `operands`, `attributes`, and
  // `regions`.
  ArgMaxOp::Adaptor adaptor(operands, attributes, properties, regions);

  RankedTensorType inputType =
      cast<RankedTensorType>(adaptor.getInput().getType());

  // Check axis.
  const int32_t axis = static_cast<int32_t>(adaptor.getAxis());
  if (axis < 0 || axis >= inputType.getRank())
    return emitOptionalError(
        loc, "expected axis to be in the range of [0, input rank)");

  // Calculate the result shape.
  SmallVector<int64_t> resultShape(inputType.getShape());
  resultShape[adaptor.getAxis()] = 1;
  Type I32Type = IntegerType::get(ctx, 32);
  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/inputType.getElementType());
  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/I32Type);
  return success();
}

LogicalResult tensorrt::ArgMinOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // Use the `Adaptor` class to interpret `operands`, `attributes`, and
  // `regions`.
  ArgMinOp::Adaptor adaptor(operands, attributes, properties, regions);

  RankedTensorType inputType =
      cast<RankedTensorType>(adaptor.getInput().getType());

  // Check axis.
  const int32_t axis = static_cast<int32_t>(adaptor.getAxis());
  if (axis < 0 || axis > inputType.getRank())
    return emitOptionalError(
        loc, "expected axis to be in the range of [0, input rank)");

  // Calculate the result shape.
  SmallVector<int64_t> resultShape(inputType.getShape());
  resultShape[adaptor.getAxis()] = 1;
  Type I32Type = IntegerType::get(ctx, 32);
  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/inputType.getElementType());
  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/I32Type);
  return success();
}

//===----------------------------------------------------------------------===//
// ExecuteRegionOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ResizeNearestOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::ResizeNearestOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ResizeNearestOp::Adaptor adaptor(operands, attributes, properties, regions);

  auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
  int64_t resizeDims = std::min(static_cast<int64_t>(3), inputType.getRank());
  if (adaptor.getScales().has_value()) {
    if (static_cast<int64_t>(adaptor.getScales().value().size()) !=
        inputType.getRank())
      return emitOptionalError(loc, "scales parameter must have same number of "
                                    "dimensions as input/output");
    for (int64_t i = 0; i < inputType.getRank() - resizeDims; i++)
      if (adaptor.getScales().value()[i] != 1)
        return emitOptionalError(
            loc, "all scale values except 3 innermost must be 1");
    SmallVector<int64_t> resultShape(inputType.getShape());
    for (unsigned i = 0; i < resultShape.size(); i++)
      resultShape[i] = inputType.isDynamicDim(i)
                           ? resultShape[i]
                           : adaptor.getScales().value()[i] * resultShape[i];
    inferredReturnShapes.emplace_back(
        /*vec=*/resultShape,
        /*elementType=*/inputType.getElementType());
  } else {
    // Output shape can't be inferred, we check only output rank and element
    // type. A tensor of same rank and element type as that of input but with
    // all dynamic dimensions is inferred as output.
    SmallVector<int64_t> resultShape(inputType.getRank(), ShapedType::kDynamic);
    inferredReturnShapes.emplace_back(
        /*vec=*/resultShape,
        /*elementType=*/inputType.getElementType());
  }
  return success();
}

LogicalResult tensorrt::ResizeNearestOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &result) {
  Location loc = getLoc();
  RankedTensorType resultType = getType();
  int64_t rank = resultType.getRank();

  // Case 1: if `output_shape` is specified, then we just extract the scalars
  // from that shape.
  if (TypedValue<TensorType> outputShape = getOutputShape()) {
    // 'tensor.extract' %source [%index]
    SmallVector<OpFoldResult> extents;
    for (int64_t i = 0; i < rank; i++) {
      Value index = b.create<arith::ConstantOp>(getLoc(), b.getIndexAttr(i));
      Value extractedShape =
          b.create<tensor::ExtractOp>(loc, outputShape, index).getResult();
      extents.push_back(
          b.create<arith::IndexCastOp>(loc, b.getIndexType(), extractedShape)
              .getResult());
    }
    result.emplace_back(std::move(extents));
    return success();
  }

  SmallVector<OpFoldResult> extents;
  extents.reserve(rank);

  // This number of trailing dimensions are the special dimensions.
  const int64_t resizeDims =
      std::min(static_cast<int64_t>(3), resultType.getRank());

  for (auto [idx, extent] : llvm::enumerate(resultType.getShape())) {

    // If dimension is known, just materialize the extent as constant.
    if (!ShapedType::isDynamic(extent)) {
      extents.push_back(b.getIndexAttr(extent));
      continue;
    }

    // Otherwise, the extent is equal to sentinel value (ShapedType::kDynamic),
    // then we use `tensor.dim` on the input operand.
    // Batch dimensions can only be leading dim.
    if (static_cast<int64_t>(idx) >= rank - resizeDims)
      return failure();

    Value index = b.create<arith::ConstantOp>(loc, b.getIndexAttr(idx));
    extents.push_back(
        b.create<tensor::DimOp>(loc, getInput(), index).getResult());
  }
  result.emplace_back(std::move(extents));
  return success();
}

//===----------------------------------------------------------------------===//
// ResizeLinearOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::ResizeLinearOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ResizeLinearOp::Adaptor adaptor(operands, attributes, properties, regions);

  auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
  int64_t resizeDims = std::min(static_cast<int64_t>(3), inputType.getRank());
  if (adaptor.getScales().has_value()) {
    if (static_cast<int64_t>(adaptor.getScales().value().size()) !=
        inputType.getRank())
      return emitOptionalError(loc, "scales parameter must have same number of "
                                    "dimensions as input/output");
    for (int64_t i = 0; i < inputType.getRank() - resizeDims; i++)
      if (adaptor.getScales().value()[i] != 1)
        return emitOptionalError(
            loc, "all scale values except 3 innermost must be 1");

    SmallVector<int64_t> resultShape(inputType.getShape());
    for (unsigned i = 0; i < resultShape.size(); i++)
      resultShape[i] = inputType.isDynamicDim(i)
                           ? resultShape[i]
                           : adaptor.getScales().value()[i] * resultShape[i];
    inferredReturnShapes.emplace_back(
        /*vec=*/resultShape,
        /*elementType=*/inputType.getElementType());
  } else {
    // Output shape can't be inferred, we check only output rank and element
    // type. A tensor of same rank and element type as that of input but with
    // all dynamic dimensions is inferred as output.
    SmallVector<int64_t> resultShape(inputType.getRank(), ShapedType::kDynamic);
    inferredReturnShapes.emplace_back(
        /*vec=*/resultShape,
        /*elementType=*/inputType.getElementType());
  }
  return success();
}

LogicalResult tensorrt::ResizeLinearOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &result) {
  Location loc = getLoc();
  RankedTensorType resultType = getType();
  int64_t rank = resultType.getRank();

  // Case 1: if `output_shape` is specified, then we just extract the scalars
  // from that shape.
  if (TypedValue<TensorType> outputShape = getOutputShape()) {
    // 'tensor.extract' %source [%index]
    SmallVector<OpFoldResult> extents;
    for (int64_t i = 0; i < rank; i++) {
      Value index = b.create<arith::ConstantOp>(getLoc(), b.getIndexAttr(i));
      Value extractedShape =
          b.create<tensor::ExtractOp>(loc, outputShape, index).getResult();
      extents.push_back(
          b.create<arith::IndexCastOp>(loc, b.getIndexType(), extractedShape)
              .getResult());
    }
    result.emplace_back(std::move(extents));
    return success();
  }

  SmallVector<OpFoldResult> extents;
  extents.reserve(rank);

  // This number of trailing dimensions are the special dimensions.
  const int64_t resizeDims =
      std::min(static_cast<int64_t>(3), resultType.getRank());

  for (auto [idx, extent] : llvm::enumerate(resultType.getShape())) {

    // If dimension is known, just materialize the extent as constant.
    if (!ShapedType::isDynamic(extent)) {
      extents.push_back(b.getIndexAttr(extent));
      continue;
    }

    // Otherwise, the extent is equal to sentinel value (ShapedType::kDynamic),
    // then we use `tensor.dim` on the input operand.
    // Batch dimensions can only be leading dim.
    if (static_cast<int64_t>(idx) >= rank - resizeDims)
      return failure();

    Value index = b.create<arith::ConstantOp>(loc, b.getIndexAttr(idx));
    extents.push_back(
        b.create<tensor::DimOp>(loc, getInput(), index).getResult());
  }
  result.emplace_back(std::move(extents));
  return success();
}

//===----------------------------------------------------------------------===//
// ResizeCubicOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::ResizeCubicOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ResizeCubicOp::Adaptor adaptor(operands, attributes, properties, regions);

  auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
  if (inputType.getRank() < 2)
    return emitOptionalError(
        loc, "does not support resizing on a tensor that has rank < 2");
  if (adaptor.getScales().has_value()) {
    if (static_cast<int64_t>(adaptor.getScales().value().size()) !=
        inputType.getRank())
      return emitOptionalError(loc, "scales parameter must have same number of "
                                    "dimensions as input/output");
    for (int64_t i = 0; i < inputType.getRank() - 2; i++)
      if (adaptor.getScales().value()[i] != 1)
        return emitOptionalError(
            loc, "all scale values except 2 innermost must be 1");

    SmallVector<int64_t> resultShape(inputType.getShape());
    for (unsigned i = 0; i < resultShape.size(); i++)
      resultShape[i] = inputType.isDynamicDim(i)
                           ? resultShape[i]
                           : adaptor.getScales().value()[i] * resultShape[i];
    inferredReturnShapes.emplace_back(
        /*vec=*/resultShape,
        /*elementType=*/inputType.getElementType());
  } else {
    // Output shape can't be inferred, we check only output rank and element
    // type. A tensor of same rank and element type as that of input but with
    // all dynamic dimensions is inferred as output.
    SmallVector<int64_t> resultShape(inputType.getRank(), ShapedType::kDynamic);
    inferredReturnShapes.emplace_back(
        /*vec=*/resultShape,
        /*elementType=*/inputType.getElementType());
  }
  return success();
}

LogicalResult tensorrt::ResizeCubicOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &result) {
  Location loc = getLoc();
  RankedTensorType resultType = getType();
  int64_t rank = resultType.getRank();

  // Case 1: if `output_shape` is specified, then we just extract the scalars
  // from that shape.
  if (TypedValue<TensorType> outputShape = getOutputShape()) {
    // 'tensor.extract' %source [%index]
    SmallVector<OpFoldResult> extents;
    for (int64_t i = 0; i < rank; i++) {
      Value index = b.create<arith::ConstantOp>(getLoc(), b.getIndexAttr(i));
      Value extractedShape =
          b.create<tensor::ExtractOp>(loc, outputShape, index).getResult();
      extents.push_back(
          b.create<arith::IndexCastOp>(loc, b.getIndexType(), extractedShape)
              .getResult());
    }
    result.emplace_back(std::move(extents));
    return success();
  }

  SmallVector<OpFoldResult> extents;
  extents.reserve(rank);

  // This number of trailing dimensions are the special dimensions.
  const int64_t resizeDims =
      std::min(static_cast<int64_t>(3), resultType.getRank());

  for (auto [idx, extent] : llvm::enumerate(resultType.getShape())) {

    // If dimension is known, just materialize the extent as constant.
    if (!ShapedType::isDynamic(extent)) {
      extents.push_back(b.getIndexAttr(extent));
      continue;
    }

    // Otherwise, the extent is equal to sentinel value (ShapedType::kDynamic),
    // then we use `tensor.dim` on the input operand.
    // Batch dimensions can only be leading dim.
    if (static_cast<int64_t>(idx) >= rank - resizeDims)
      return failure();

    Value index = b.create<arith::ConstantOp>(loc, b.getIndexAttr(idx));
    extents.push_back(
        b.create<tensor::DimOp>(loc, getInput(), index).getResult());
  }
  result.emplace_back(std::move(extents));
  return success();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::ScatterOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ScatterOp::Adaptor adaptor(operands, attributes, properties, regions);

  TensorType inputDataType =
      cast<RankedTensorType>(adaptor.getData().getType());
  TensorType indicesDataType =
      cast<RankedTensorType>(adaptor.getIndices().getType());
  TensorType updatesDataType =
      cast<RankedTensorType>(adaptor.getUpdates().getType());

  int64_t inputRank = inputDataType.getRank();
  int64_t indicesRank = indicesDataType.getRank();
  int64_t updatesRank = updatesDataType.getRank();

  auto indicesShape = indicesDataType.getShape();
  if (indicesShape.empty())
    return emitOptionalError(loc, ScatterOp::getOperationName(),
                             " indices must have rank >= 1");

  int64_t indexVectorSize = indicesShape.back();
  if (ShapedType::isDynamic(indexVectorSize))
    return emitOptionalError(
        loc, "the last dimension in ", ScatterOp::getOperationName(),
        " indices tensor (the index vector size) must be static");

  if (indexVectorSize > inputRank)
    return emitOptionalError(
        loc, ScatterOp::getOperationName(),
        " index vector size cannot be larger than the input rank");

  int64_t expectedUpdatesRank = inputRank + indicesRank - indexVectorSize - 1;
  if (updatesRank != expectedUpdatesRank)
    return emitOptionalError(loc, ScatterOp::getOperationName(),
                             " expected updates tensor rank to be ",
                             expectedUpdatesRank);

  int64_t leadingIndexDims = std::max<int64_t>(0, indicesRank - 1);

  if (failed(mlir::verifyCompatibleShape(
          updatesDataType.getShape().drop_front(leadingIndexDims),
          inputDataType.getShape().drop_front(indexVectorSize))))
    return emitOptionalError(loc, ScatterOp::getOperationName(),
                             " input tensor shape is incompatible with the "
                             "shape of the updates tensor");

  inferredReturnShapes.emplace_back(ShapeAdaptor(inputDataType));
  return success();
}

//===----------------------------------------------------------------------===//
// ScatterElementsOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::ScatterElementsOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {

  ScatterElementsOp::Adaptor adaptor(operands, attributes, properties, regions);
  TensorType inputDataType = cast<TensorType>(adaptor.getData().getType());

  SmallVector<int64_t> resultShape(inputDataType.getShape());
  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/inputDataType.getElementType());

  return success();
}

//===----------------------------------------------------------------------===//
// NormalizationOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::NormalizationOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {

  NormalizationOp::Adaptor adaptor(operands, attributes, properties, regions);
  TensorType inputDataType = cast<TensorType>(adaptor.getInput().getType());

  SmallVector<int64_t> resultShape(inputDataType.getShape());
  inferredReturnShapes.emplace_back(
      /*vec=*/resultShape,
      /*elementType=*/inputDataType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// QuantizeOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::QuantizeOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  QuantizeOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto rtt = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!rtt)
    return emitOptionalError(loc, "expected input to be a ranked tensor");
  inferredReturnShapes.emplace_back(/*vec=*/rtt.getShape(),
                                    /*elementType=*/nullptr);
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicQuantizeOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::DynamicQuantizeOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  DynamicQuantizeOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto rtt = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!rtt)
    return emitOptionalError(loc, "expected input to be a ranked tensor");
  // `result` has same shape as input but f4 type.
  inferredReturnShapes.emplace_back(/*vec=*/rtt.getShape(),
                                    /*elementType=*/Float4E2M1FNType::get(ctx));
  // `scale` has same shape as input except `axis` dim and has f8 type.
  // `axis` dimension is `shape(input)[axis] / 16` i.e. divided by block size.
  SmallVector<int64_t> scaleShape = llvm::to_vector(rtt.getShape());
  auto quantizationAxis = adaptor.getAxis();
  if (!rtt.isDynamicDim(quantizationAxis))
    scaleShape[quantizationAxis] = scaleShape[quantizationAxis] / 16;
  inferredReturnShapes.emplace_back(/*vec=*/scaleShape,
                                    /*elementType=*/Float8E4M3FNType::get(ctx));
  return success();
}

//===----------------------------------------------------------------------===//
// DequantizeOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::DequantizeOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  DequantizeOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto rtt = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!rtt)
    return emitOptionalError(loc, "expected input to be a ranked tensor");
  inferredReturnShapes.emplace_back(/*vec=*/rtt.getShape(),
                                    /*elementType=*/nullptr);
  return success();
}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

LogicalResult tensorrt::AttentionOp::inferReturnTypeComponents(
    MLIRContext *ctx, std::optional<Location> loc, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  AttentionOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto queryType = dyn_cast<RankedTensorType>(adaptor.getQuery().getType());
  if (!queryType)
    return emitOptionalError(loc, "expected query to be a ranked tensor");
  inferredReturnShapes.emplace_back(
      /*vec=*/queryType.getShape(),
      /*elementType=*/queryType.getElementType());
  return success();
}
