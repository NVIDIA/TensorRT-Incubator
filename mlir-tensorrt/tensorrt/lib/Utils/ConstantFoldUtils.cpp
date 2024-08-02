//===- ConstantFoldUtils.cpp ----------------------------------------------===//
//
// The constant fold transpose logic is adapted from the LLVM project
// `llvm-project/mlir/lib/Dialect/Tosa/Transforms/TosaFolders.cpp` and has
// the original license:
// Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Changes are copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of common constant-folding operations.
/// TODO: these should be moved into a common location upstream
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Utils/ConstantFoldUtils.h"
#include "mlir-tensorrt-dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APSInt.h"

using namespace mlir;

template <typename ElementValueType>
DenseElementsAttr transposeImpl(mlir::ElementsAttr attr, ShapedType inputType,
                                ShapedType outputType, AffineMap perm) {
  // Handle the trivial scalar case.
  if (inputType.getNumElements() == 0)
    return DenseElementsAttr::get(outputType, ArrayRef<ElementValueType>{});

  ArrayRef<int64_t> inputShape = inputType.getShape();
  SmallVector<int64_t> outputStrides =
      mlir::computeStrides(outputType.getShape());
  SmallVector<int64_t> permutedStrides =
      mlir::inversePermutation(perm).compose(outputStrides);
  SmallVector<ElementValueType> outputValues(
      inputType.getNumElements(), *attr.getValues<ElementValueType>().begin());
  for (const auto &[srcLinearIndex, value] :
       llvm::enumerate(attr.getValues<ElementValueType>())) {
    uint64_t dstLinearIndex = 0;
    // Delinearize and permute the linear index.
    for (int64_t dim = inputShape.size() - 1; dim >= 0; --dim) {
      uint64_t sourceIndexForDim = srcLinearIndex % inputShape[dim];
      srcLinearIndex /= inputShape[dim];
      dstLinearIndex += permutedStrides[dim] * sourceIndexForDim;
    }
    outputValues[dstLinearIndex] = value;
  }

  return DenseElementsAttr::get(outputType, ArrayRef(outputValues));
}

// A type specialized transposition of an ElementsAttr.
// This implementation tries to operate on the underlying data in its raw
// representation when possible to avoid allocating a large number of Attribute
// objects.
ElementsAttr mlir::constantFoldTranspose(ElementsAttr attr,
                                         AffineMap permutation) {
  if (!isa<IntegerType, FloatType>(attr.getElementType()))
    return {};

  ShapedType inputType = attr.getShapedType();

  // Calculate the output type.
  SmallVector<int64_t> outputShape = permutation.compose(inputType.getShape());
  ShapedType outputType = inputType.clone(outputShape);

  // If the constant is elided, then we can't do anything. Just simulate the
  // transposition. This ensures we get the same effect in tests with elided
  // weights.
  if (std::optional<DenseResourceElementsHandle> handle =
          mlir::getElidedResourceElementsAttr(attr))
    return DenseResourceElementsAttr::get(outputType, *handle);

  // If the constant is a splat, then we can just change the shape directly.
  if (attr.isSplat() && attr.isa<DenseElementsAttr>())
    return attr.cast<DenseElementsAttr>().reshape(outputType);

  Type elementType = inputType.getElementType();
  if (auto intType = llvm::dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 1:
      return transposeImpl<bool>(attr, inputType, outputType, permutation);
    case 8:
      return transposeImpl<int8_t>(attr, inputType, outputType, permutation);
    case 16:
      return transposeImpl<int16_t>(attr, inputType, outputType, permutation);
    case 32:
      return transposeImpl<int32_t>(attr, inputType, outputType, permutation);
    case 64:
      return transposeImpl<int64_t>(attr, inputType, outputType, permutation);
    default:
      return transposeImpl<APInt>(attr, inputType, outputType, permutation);
    }
  }
  assert(isa<FloatType>(elementType) && "expected FloatType");

  if (elementType.isF32())
    return transposeImpl<float>(attr, inputType, outputType, permutation);

  return transposeImpl<APFloat>(attr, inputType, outputType, permutation);
}

ElementsAttr mlir::constantFoldReshape(ShapedType newType, ElementsAttr attr) {
  ShapedType inputType = dyn_cast<ShapedType>(attr.getType());
  if (!inputType)
    return nullptr;

  // If the constant is elided, then we can't do anything. Just simulate the
  // transposition. This ensures we get the same effect in tests with elided
  // weights.
  if (std::optional<DenseResourceElementsHandle> handle =
          mlir::getElidedResourceElementsAttr(attr))
    return DenseResourceElementsAttr::get(newType, *handle);

  auto els = dyn_cast<DenseElementsAttr>(attr);
  if (!els)
    return nullptr;

  if (els.getType().getNumElements() != newType.getNumElements())
    return nullptr;
  return els.reshape(newType);
}

static ElementsAttr constantFoldConvertFromFloatType(Type newElementType,
                                                     ElementsAttr attr) {
  ShapedType inputType = attr.getShapedType();
  // FloatType -> FloatType
  if (auto newType = dyn_cast<FloatType>(newElementType)) {
    if (attr.isSplat() && attr.isa<DenseElementsAttr>()) {
      APFloat in = attr.getSplatValue<APFloat>();
      bool losesInfo{false};
      in.convert(newType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                 &losesInfo);
      return DenseElementsAttr::get(inputType.clone(newType), in);
    }

    auto els = dyn_cast<DenseElementsAttr>(attr);
    if (!els)
      return nullptr;
    return els.mapValues(newType, [&](const APFloat &floatVal) -> APInt {
      APFloat convertedFloat = floatVal;
      bool losesInfo = false;
      convertedFloat.convert(newType.getFloatSemantics(),
                             APFloat::rmNearestTiesToEven, &losesInfo);
      return convertedFloat.bitcastToAPInt();
    });
  }

  // FloatType -> IntegerType
  if (auto newType = dyn_cast<IntegerType>(newElementType)) {
    bool isIntegerTypeUnsigned =
        newType.isUnsignedInteger() || newType.isInteger(1);
    if (attr.isSplat() && attr.isa<DenseElementsAttr>()) {
      APFloat in = attr.getSplatValue<APFloat>();
      APSInt out(newType.getIntOrFloatBitWidth(), isIntegerTypeUnsigned);
      bool isExact;
      in.convertToInteger(out, APFloat::rmTowardZero, &isExact);
      return DenseElementsAttr::get(inputType.clone(newElementType), out);
    }

    auto els = dyn_cast<DenseElementsAttr>(attr);
    if (!els)
      return nullptr;
    return els.mapValues(newElementType, [&](const APFloat &floatVal) -> APInt {
      APSInt out(newType.getIntOrFloatBitWidth(), isIntegerTypeUnsigned);
      bool isExact;
      floatVal.convertToInteger(out, APFloat::rmTowardZero, &isExact);
      return std::move(out);
    });
  }

  return nullptr;
}

static ElementsAttr constantFoldConvertFromIntegerType(Type newElementType,
                                                       ElementsAttr attr) {
  ShapedType inputType = attr.getShapedType();
  Type oldElementType = inputType.getElementType();
  bool isInputTypeUnsigned =
      oldElementType.isUnsignedInteger() || oldElementType.isInteger(1);

  // IntType -> FloatType
  if (auto newType = dyn_cast<FloatType>(newElementType)) {
    if (attr.isSplat() && attr.isa<DenseElementsAttr>()) {
      APInt in = attr.getSplatValue<APInt>();
      APFloat floatVal(newType.getFloatSemantics(),
                       APInt::getZero(newType.getWidth()));
      floatVal.convertFromAPInt(in, !isInputTypeUnsigned,
                                APFloat::rmTowardZero);
      return DenseElementsAttr::get(inputType.clone(newType), floatVal);
    }

    // Otherwise, convert one-by-one.
    auto els = dyn_cast<DenseElementsAttr>(attr);
    if (!els)
      return nullptr;
    return els.mapValues(newType, [&](const APInt &intVal) -> APInt {
      APFloat floatVal(newType.getFloatSemantics(),
                       APInt::getZero(newType.getWidth()));
      floatVal.convertFromAPInt(intVal, !isInputTypeUnsigned,
                                APFloat::rmTowardZero);
      return floatVal.bitcastToAPInt();
    });
  }

  // Int -> Int conversion
  if (auto newType = dyn_cast<IntegerType>(newElementType)) {
    if (attr.isSplat() && attr.isa<DenseElementsAttr>()) {
      APSInt out(attr.getSplatValue<APInt>(), isInputTypeUnsigned);
      return DenseElementsAttr::get(
          inputType.clone(newElementType),
          out.extOrTrunc(newElementType.getIntOrFloatBitWidth()));
    }

    auto els = dyn_cast<DenseElementsAttr>(attr);
    if (!els)
      return nullptr;
    return els.mapValues(newElementType, [&](const APInt &intVal) -> APInt {
      return APSInt(intVal, isInputTypeUnsigned)
          .extOrTrunc(newElementType.getIntOrFloatBitWidth());
    });
  }
  return nullptr;
}

ElementsAttr mlir::constantFoldConvert(Type newElementType, ElementsAttr attr) {
  ShapedType inputType = dyn_cast<ShapedType>(attr.getType());
  if (!inputType)
    return nullptr;
  // We don't handle complex values.
  if (!isa<IntegerType, FloatType>(newElementType) ||
      !isa<IntegerType, FloatType>(inputType.getElementType()))
    return nullptr;

  if (inputType.getElementType() == newElementType)
    return attr;

  if (std::optional<DenseResourceElementsHandle> handle =
          mlir::getElidedResourceElementsAttr(attr))
    return DenseResourceElementsAttr::get(inputType.clone(newElementType),
                                          *handle);

  if (auto floatType = dyn_cast<FloatType>(inputType.getElementType()))
    return constantFoldConvertFromFloatType(newElementType, attr);
  if (auto intType = dyn_cast<IntegerType>(inputType.getElementType()))
    return constantFoldConvertFromIntegerType(newElementType, attr);

  return nullptr;
}

template <typename T>
static ElementsAttr stridedSliceImpl(ElementsAttr src, RankedTensorType dstType,
                                     ArrayRef<int64_t> offsets,
                                     ArrayRef<int64_t> limits,
                                     ArrayRef<int64_t> strides) {
  // Handle edge case of empty source tensor.
  if (src.getShapedType().getNumElements() == 0)
    return {};

  // Handle edge case of rank-0 tensor.
  const int64_t rank = src.getShapedType().getRank();
  if (rank == 0)
    return DenseElementsAttr::get(dstType, *src.getValues<T>().begin());

  SmallVector<int64_t> srcStrides =
      mlir::computeSuffixProduct(src.getShapedType().getShape());
  SmallVector<int64_t> dstStrides =
      mlir::computeSuffixProduct(dstType.getShape());

  // The logic below comes from the original MLIR CRunner implementation of
  // strided memref copy. It could be further optimized if any of the
  // dimensions have unit stride.
  SmallVector<T> result;
  result.reserve(dstType.getNumElements());
  SmallVector<int64_t> indices(offsets);
  assert(static_cast<int64_t>(indices.size()) == rank &&
         "mismatched src rank and offsets rank");
  int64_t readIndex = mlir::linearize(indices, srcStrides);
  while (true) {
    result.push_back(src.getValues<T>()[readIndex]);
    assert(static_cast<int64_t>(result.size()) <= dstType.getNumElements());
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      int64_t newIndex = indices[axis] + strides[axis];
      indices[axis] = newIndex;
      readIndex += srcStrides[axis] * strides[axis];
      if (limits[axis] > newIndex)
        break;
      if (axis == 0)
        return DenseElementsAttr::get(dstType, ArrayRef<T>(result));
      indices[axis] = offsets[axis];
      readIndex -= (newIndex - offsets[axis]) * srcStrides[axis];
    }
  }
  llvm_unreachable("expected return from slice impl loop");
}

ElementsAttr mlir::constantFoldSliceOffsetLimitStride(
    ElementsAttr attr, RankedTensorType outputType, ArrayRef<int64_t> offsets,
    ArrayRef<int64_t> limits, ArrayRef<int64_t> strides) {
  if (outputType.getNumElements() == 0)
    return cast<ElementsAttr>(
        DenseElementsAttr::get(outputType, ArrayRef<Attribute>{}));

  if (!isa<IntegerType, FloatType>(attr.getElementType()))
    return {};

  // Dispatch implementation based on data type.
  // If the constant is elided, then we can't do anything. Just simulate the
  // transposition. This ensures we get the same effect in tests with elided
  // weights.
  if (std::optional<DenseResourceElementsHandle> handle =
          mlir::getElidedResourceElementsAttr(attr))
    return cast<ElementsAttr>(
        DenseResourceElementsAttr::get(outputType, *handle));

  // If the constant is a splat, then we can just change the shape directly.
  if (attr.isSplat() && attr.isa<DenseElementsAttr>())
    return cast<ElementsAttr>(
        cast<DenseElementsAttr>(attr).resizeSplat(outputType));

  if (!isa<DenseElementsAttr>(attr))
    return {};

  Type elementType = attr.getElementType();
  if (auto intType = llvm::dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 1:
      return stridedSliceImpl<bool>(attr, outputType, offsets, limits, strides);
    case 8:
      return stridedSliceImpl<int8_t>(attr, outputType, offsets, limits,
                                      strides);
    case 16:
      return stridedSliceImpl<int16_t>(attr, outputType, offsets, limits,
                                       strides);
    case 32:
      return stridedSliceImpl<int32_t>(attr, outputType, offsets, limits,
                                       strides);
    case 64:
      return stridedSliceImpl<int64_t>(attr, outputType, offsets, limits,
                                       strides);
    default:
      return stridedSliceImpl<APInt>(attr, outputType, offsets, limits,
                                     strides);
    }
  }

  assert(isa<FloatType>(elementType) && "expected FloatType");
  if (elementType.isF32())
    return stridedSliceImpl<float>(attr, outputType, offsets, limits, strides);

  return stridedSliceImpl<APFloat>(attr, outputType, offsets, limits, strides);
}
