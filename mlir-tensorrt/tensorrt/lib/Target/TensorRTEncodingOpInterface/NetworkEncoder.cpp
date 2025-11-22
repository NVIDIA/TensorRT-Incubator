//===- NetworkEncoder.h -----------------------------------------*- C++ -*-===//
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
// This file contains the implementation for NvInferNetworkEncoder that
// traverses MLIR IR data-structures and emits appropriate calls to translate IR
// to a TensorRT network.
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/NetworkEncoder.h"
#include "mlir-tensorrt-common/Interfaces/TensorKindOpInterface.h"
#include "mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/TensorRTEncodingOpInterface.h"
#include "mlir-tensorrt-dialect/TensorRT/Utils/Utils.h"
#include "mlir-tensorrt-dialect/Utils/NvInferAdaptor.h"
#include "mlir-tensorrt-dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace mlir;
using namespace mlir::tensorrt;

Type tensorrt::getNvInferDataTypeAsMlirType(MLIRContext *ctx,
                                            nvinfer1::DataType t) {
  using nvinfer1::DataType;
  switch (t) {
  case DataType::kFLOAT:
    return Float32Type::get(ctx);
  case DataType::kHALF:
    return Float16Type::get(ctx);
  case DataType::kINT8:
    return IntegerType::get(ctx, 8);
  case DataType::kINT32:
    return IntegerType::get(ctx, 32);
  case DataType::kBOOL:
    return IntegerType::get(ctx, 1);
  case DataType::kUINT8:
    return IntegerType::get(ctx, 8, IntegerType::SignednessSemantics::Unsigned);
  case DataType::kFP8:
    return Float8E4M3FNType::get(ctx);
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(9, 1, 0)
  case DataType::kBF16:
    return BFloat16Type::get(ctx);
  case DataType::kINT64:
    return IntegerType::get(ctx, 64);
#endif
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  case DataType::kINT4:
    return IntegerType::get(ctx, 4);
#endif
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
  case DataType::kFP4:
    return Float4E2M1FNType::get(ctx);
#endif
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 12, 0)
  case DataType::kE8M0:
    return Float8E4M3FNType::get(
        ctx); // Using FP8 as a proxy since MLIR doesn't have E8M0
#endif
  }
  llvm_unreachable("invalid TRT -> MLIR type conversion for the given TensorRT "
                   "version. ");
}

uint32_t tensorrt::getBitMaskFromDimensionList(ArrayRef<int64_t> dimensions) {
  uint32_t axisMask = 0;
  for (int64_t axis : dimensions) {
    assert(axis >= 0 &&
           "expected elements in TensorRT_DimensionList to be non-negative");
    axisMask |= 1U << axis;
  }
  return axisMask;
}

/// Verify that the tensors have equal shape.
static LogicalResult sanityCheckTypes(nvinfer1::ITensor *trtTensor,
                                      Value mlirTensor) {
  nvinfer1::Dims dims = trtTensor->getDimensions();
  auto type = cast<RankedTensorType>(mlirTensor.getType());
  Location loc = mlirTensor.getLoc();
  if (dims.nbDims != type.getRank())
    return emitError(loc) << "ranks of MLIR Tensor is " << type.getRank()
                          << " while TRT ITensor has rank " << dims.nbDims;

  auto emitTypeDiagnostic = [&](bool isError, StringRef extraNote = "") {
    std::string str;
    llvm::raw_string_ostream ss(str);
    ss << "MLIR Tensor has type: ";
    type.print(ss);
    ss << ", TRT ITensor has type tensor<";
    llvm::interleave(llvm::ArrayRef(dims.d, dims.d + dims.nbDims), ss, "x");
    ss << "x"
       << getNvInferDataTypeAsMlirType(mlirTensor.getContext(),
                                       trtTensor->getType())
       << ">";
    ss.flush();
    InFlightDiagnostic diagnostic = isError ? emitError(loc) : emitWarning(loc);
    diagnostic << str;
    if (!extraNote.empty())
      diagnostic.attachNote() << extraNote;
    return diagnostic;
  };

  FailureOr<nvinfer1::DataType> trtTypeFromMlirTensorType =
      getNvInferDataType(loc, type.getElementType());
  if (failed(trtTypeFromMlirTensorType))
    return failure();
  if (trtTypeFromMlirTensorType.value() != trtTensor->getType())
    return emitTypeDiagnostic(/*isError=*/true);

  for (int64_t i = 0, e = type.getRank(); i < e; i++) {
    // If TensorRT cannot infer a static shape, then the TRT ITensor dimension
    // will have extent of -1. The MLIR type shape has extent unknown or static
    // -- both cases are not an error. If MLIR shape is static, that is probably
    // because we can sometimes carry or deduce shape information more specific
    // than TRT can infer at construction time.
    if (dims.d[i] == -1)
      continue;

    // Otherwise, if both dimension extents are static, then they should agree.
    if (!type.isDynamicDim(i) &&
        type.getDimSize(i) != static_cast<int64_t>(dims.d[i]))
      return emitTypeDiagnostic(/*isError=*/true);

    // The below check can fail since TensorRT will do some folding and type
    // inference while the network is being constructed. For example, this is
    // legal TensorRT MLIR:
    //
    // ```
    // %size = tensorrt.constant dense<[128, 128]> : tensor<2xi32>
    // %0 = tensorrt.slice %arg0[512, 512][%size: tensor<2xi32>][2, 2]
    //   : tensor<1024x1024xf32> to tensor<?x?xf32>
    // ```
    //
    // But TensorRT will deduce the shape of %0 is '2x2'. Since we
    // don't want to require canonicalization before exporting, just
    // emit a warning for now. This could change in the future.
    if (type.isDynamicDim(i) && dims.d[i] != -1)
      emitTypeDiagnostic(
          /*isError=*/false,
          "TensorRT does some type inference while constructing the TensorRT "
          "network; consider running canonicalization prior to TensorRT "
          "translation in order to run type inference to potentially eliminate "
          "these differences");
  }
  return success();
}

/// Perform some basic checks to verify that the network signature matches the
/// function it should be encoding.
static LogicalResult
sanityCheckFuncSignature(nvinfer1::INetworkDefinition *network,
                         FunctionOpInterface func) {
  TypeRange argumentTypes = func.getArgumentTypes();
  TypeRange resultTypes = func.getResultTypes();
  if (network->getNbInputs() != static_cast<int64_t>(argumentTypes.size()) ||
      network->getNbOutputs() != static_cast<int64_t>(resultTypes.size()))
    return failure();

  for (unsigned i = 0; i < argumentTypes.size(); i++) {
    Value arg = func.getArgument(i);
    if (failed(sanityCheckTypes(network->getInput(i), arg)))
      return failure();
  }
  // The whole encoder assumes that the func is a single-block function.
  for (unsigned i = 0; i < resultTypes.size(); i++) {
    Value returnedVal = func.front().getTerminator()->getOperand(i);
    if (failed(sanityCheckTypes(network->getOutput(i), returnedVal)))
      return failure();
  }

  return success();
}

nvinfer1::Dims tensorrt::getNvInferDims(RankedTensorType t) {
  return getNvInferDims(t.getShape());
}

/// Given an `ArrayRef<in64_t>`, return those values as an
/// `nvinfer1::Permutation`.
nvinfer1::Permutation tensorrt::getNvInferPermutation(ArrayRef<int64_t> array) {
  assert(array.size() <= nvinfer1::Dims::MAX_DIMS &&
         "permutation array exceeds max dims");
  nvinfer1::Permutation permutation;
  // TODO(cbate): we should validate against loss of precision or change the
  // attribute type that is resulting in an array of int64 values.
  llvm::copy(
      llvm::map_range(
          array, [](int64_t x) -> int32_t { return static_cast<int32_t>(x); }),
      permutation.order);
  return permutation;
}

//===----------------------------------------------------------------------===//
// NvInferNetworkEncoder
//===----------------------------------------------------------------------===//

static std::string getUniqueName(NvInferNetworkEncoder::NamesSet &names,
                                 std::string name) {
  static unsigned i = 0;
  std::string uniqueName = name;
  while (names.contains(uniqueName))
    uniqueName = name + "_" + std::to_string(i++);
  names.insert(uniqueName);
  return uniqueName;
}

/// Print a representation of the given location to the string. Since MLIR has
/// an open system of location attributes, there may be some location types that
/// we cannot handle. We do not use the location's builtin printer because it
/// could be extremely verbose.
static void getCallSiteLocs(Location loc, SmallVector<Location> &locs) {
  if (auto callLoc = dyn_cast<CallSiteLoc>(loc)) {
    getCallSiteLocs(callLoc.getCaller(), locs);
    getCallSiteLocs(callLoc.getCallee(), locs);
  } else {
    locs.push_back(loc);
  }
}

static void translateLocation(Location loc, llvm::raw_ostream &os) {
  if (auto callLoc = dyn_cast<CallSiteLoc>(loc)) {
    SmallVector<Location> locs;
    getCallSiteLocs(callLoc, locs);
    // only include the last 3 locations in the names as this should be
    // sufficient to identify the call site for an op
    for (size_t i = locs.size() > 3 ? locs.size() - 3 : 0; i < locs.size();
         i++) {
      translateLocation(locs[i], os);
      if (i < locs.size() - 1) {
        os << " -> ";
      }
    }
    return;
  }
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    // A scope of a DILocation cannot be null.
    os << fileLoc.getFilename() << ":" << fileLoc.getLine();
    return;
  }
  if (auto fileLoc = dyn_cast<FileLineColRange>(loc)) {
    os << fileLoc.getFilename() << ":" << fileLoc.getStartLine();
    return;
  }
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    ArrayRef<Location> locations = fusedLoc.getLocations();
    translateLocation(locations.front(), os);
    return;
  }
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    os << nameLoc.getName();
    return;
  }
  if (auto opaqueLoc = dyn_cast<OpaqueLoc>(loc)) {
    translateLocation(opaqueLoc.getFallbackLocation(), os);
    return;
  }
  os << "unknown location";
  return;
}

static std::string createName(NvInferNetworkEncoder::NamesSet &names,
                              Operation *sourceOp) {
  std::string name;
  {
    llvm::raw_string_ostream ss(name);
    ss << "[" << sourceOp->getName().getStringRef() << "] ";
    translateLocation(sourceOp->getLoc(), ss);
    ss.flush();
  }
  // Truncate to TRT limit.
  static constexpr size_t kLayerNameSizeLimit =
      2048 - 6; // -6 to give some space for UniqueName
  if (name.size() > kLayerNameSizeLimit)
    name = name.substr(0, kLayerNameSizeLimit);

  // TRT name does not allow nested quotations.
  name = llvm::join(llvm::split(name, "\""), "");
  return getUniqueName(names, name);
}

void NvInferNetworkEncoder::setMetadata(nvinfer1::ILayer *layer,
                                        Operation *sourceOp) {
  std::string name = createName(namesSet, sourceOp);
  layer->setName(name.c_str());
  if (auto fusedLoc = dyn_cast<FusedLoc>(sourceOp->getLoc()))
    if (auto metadataAttr =
            dyn_cast_if_present<StringAttr>(fusedLoc.getMetadata()))
      layer->setMetadata(metadataAttr.getValue().str().c_str());
}

nvinfer1::ITensor *NvInferNetworkEncoder::lookup(Value v) const {
  return valueMap.lookup(v);
}

SmallVector<nvinfer1::ITensor *>
NvInferNetworkEncoder::lookupValues(ValueRange values) {
  return llvm::to_vector(
      llvm::map_range(values, [&](Value v) { return valueMap.lookup(v); }));
}

void NvInferNetworkEncoder::map(Value from, nvinfer1::ITensor *to) {
  valueMap.insert(from, to);
}

void NvInferNetworkEncoder::map(ValueRange from,
                                ArrayRef<nvinfer1::ILayer *> to) {
  for (auto [v, l] : llvm::zip(from, to))
    valueMap.insert(v, l->getOutput(0));
}

void NvInferNetworkEncoder::map(Operation *op, nvinfer1::ILayer *layer) {
  if (!layerMap.count(op))
    layerMap[op] = {};

  layerMap[op].push_back(layer);
}

bool NvInferNetworkEncoder::isStronglyTyped() const {
  if (!usesStronglyTyped)
    return false;
  return nvinfer1::adaptor::isStronglyTypedFlagEnabled(network);
}

nvinfer1::ITensor *
NvInferNetworkEncoder::insertCastLayer(nvinfer1::ITensor *input,
                                       nvinfer1::DataType dataType,
                                       Operation *sourceOp) {
  auto castLayer = network->addCast(*input, dataType);
  setMetadata(castLayer, sourceOp);
  return castLayer->getOutput(0);
}

nvinfer1::ILayer *NvInferNetworkEncoder::addDequantizeLayer(
    nvinfer1::ITensor *input, nvinfer1::ITensor *scale,
    nvinfer1::DataType outputType, std::optional<uint32_t> axis) {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(9, 1, 0)
  nvinfer1::IDequantizeLayer *dequantizeLayer =
      network->addDequantize(*input, *scale, outputType);
  if (axis)
    dequantizeLayer->setAxis(*axis);
  return dequantizeLayer;
#else
  nvinfer1::IDequantizeLayer *dequantizeLayer =
      network->addDequantize(*input, *scale);
  nvinfer1::IIdentityLayer *identityLayer =
      network->addIdentity(*dequantizeLayer->getOutput(0));
  identityLayer->setOutputType(0, outputType);
  if (axis)
    dequantizeLayer->setAxis(*axis);
  return identityLayer;
#endif
}

nvinfer1::IFillLayer *populateFillLayerParameters(
    nvinfer1::IFillLayer *layer, const nvinfer1::Dims &staticShape,
    nvinfer1::ITensor *dynamicShape, std::optional<double> alpha,
    std::optional<double> beta, nvinfer1::ITensor *dynamicAlpha,
    nvinfer1::ITensor *dynamicBeta) {
  assert(layer != nullptr && "expected valid layer");
  if (dynamicShape != nullptr)
    layer->setInput(0, *dynamicShape);
  else
    layer->setDimensions(staticShape);

  if (alpha)
    layer->setAlpha(*alpha);
  else if (dynamicAlpha)
    layer->setInput(1, *dynamicAlpha);

  if (beta)
    layer->setBeta(*beta);
  else if (dynamicBeta)
    layer->setInput(2, *dynamicBeta);

  return layer;
}

nvinfer1::ILayer *NvInferNetworkEncoder::addFillLayer(
    nvinfer1::DataType elementType, nvinfer1::Dims staticShape,
    nvinfer1::ITensor *dynamicShape, nvinfer1::FillOperation fillOperation,
    std::optional<double> alpha, std::optional<double> beta,
    nvinfer1::ITensor *dynamicAlpha, nvinfer1::ITensor *dynamicBeta) {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(9, 1, 0)
  if (dynamicShape) {

    // Starting in TensorRT 10.5, TensorRT will give an error if we don't give a
    // fully valid static result shape, even if we are about to override it with
    // a dynamic shape.

#ifndef NDEBUG
    nvinfer1::Dims shapeDims = dynamicShape->getDimensions();
    assert(shapeDims.nbDims == 1 && shapeDims.d[0] > 0 &&
           "invalid shape tensor shape");
#endif // NDEBUG

    staticShape.nbDims = 1;
    staticShape.d[0] = 1;
  }
  nvinfer1::IFillLayer *layer =
      network->addFill(staticShape, fillOperation, elementType);
  assert(layer != nullptr && "expected valid layer");
  return populateFillLayerParameters(layer, staticShape, dynamicShape, alpha,
                                     beta, dynamicAlpha, dynamicBeta);
#else
  nvinfer1::IFillLayer *layer = network->addFill(staticShape, fillOperation);
  layer = populateFillLayerParameters(layer, staticShape, dynamicShape, alpha,
                                      beta, dynamicAlpha, dynamicBeta);
  layer->setOutputType(0, elementType);
  return layer;
#endif
}

FailureOr<nvinfer1::ILayer *> NvInferNetworkEncoder::addOpaquePlugin(
    tensorrt::OpaquePluginOp op, SmallVector<nvinfer1::ITensor *> &results) {

  FailureOr<PluginInterfaceBase *> pluginBase = failure();
  pluginBase = pluginMgr.getExternalPlugin(
      op.getLoc(), op.getPluginName(), op.getPluginVersion(),
      op.getPluginNamespace(), op.getCreatorParams(), createName(namesSet, op),
      op.getDsoPath(), op.getCreatorFunc());

  if (failed(pluginBase))
    return failure();

  SmallVector<nvinfer1::ITensor *> inputs = lookupValues(op.getInputs());
  nvinfer1::ILayer *pluginLayer;

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  if (auto plugin = llvm::dyn_cast<Plugin<nvinfer1::IPluginV3>>(*pluginBase)) {
    pluginLayer = network->addPluginV3(inputs.data(), inputs.size(), nullptr, 0,
                                       *(plugin->ptr));
  } else
#endif
  {
    auto pluginV2 =
        static_cast<Plugin<nvinfer1::IPluginV2DynamicExt> *>(*pluginBase);
    pluginLayer =
        network->addPluginV2(inputs.data(), inputs.size(), *(pluginV2->ptr));
  }

  if (!pluginLayer ||
      pluginLayer->getNbOutputs() != static_cast<int64_t>(op->getNumResults()))
    return failure();

  results.reserve(pluginLayer->getNbOutputs());
  for (unsigned i = 0, e = op->getNumResults(); i < e; i++)
    results.push_back(pluginLayer->getOutput(i));

  return pluginLayer;
}

/// Sets lower precision use flags [`usesF16`, `usesInt8`, `usesF8` and/or
/// `usesBf16`] to true if any of the op's inputs or results have tensors with
/// element type fp16/int8/fp8 respectively.
static void updateLowerPrecisionIndicators(Operation *op, bool &usesF16,
                                           bool &usesInt8, bool &usesF8,
                                           bool &usesBf16, bool &usesInt4) {
  auto update = [&](Type t) {
    Type elType = cast<RankedTensorType>(t).getElementType();
    usesF16 |= elType.isF16();
    usesInt8 |= isTensorRTInt8Type(elType);
    usesF8 |= isa<Float8E4M3FNType>(elType);
    usesBf16 |= elType.isBF16();
    usesInt4 |= elType.isInteger(4);
  };
  for (Type t : op->getOperandTypes())
    update(t);
  for (Type t : op->getResultTypes())
    update(t);
}

/// Set the range of an int8 tensor to use identity scaling (-127, 127). Note
/// that this should have no affect if QDQ nodes are present, but if QDQ nodes
/// are not present, then this helps TensorRT interpret the int8 data as we
/// would prefer.
static void setIdentityInt8DynamicRange(nvinfer1::ITensor *tensor) {
  tensor->setDynamicRange(-127.0f, 127.0f);
}

/// For a given operation, try to add that operation to `network` and populate
/// `valueMap` with its results. If `op` doesn't not represent a TensorRT
/// dialect operation, then return failure.
LogicalResult
NvInferNetworkEncoder::encodeOp(tensorrt::TensorRTEncodingOpInterface op) {
  SmallVector<nvinfer1::ITensor *> results;
  if (failed(op.encodeOp(*this, results)))
    return failure();

  for (auto [opResult, tensorResult] : llvm::zip(op->getResults(), results)) {
    if (failed(sanityCheckTypes(tensorResult, opResult)))
      return failure();
    if (!usesStronglyTyped && !hasQDQOps &&
        tensorResult->getType() == nvinfer1::DataType::kINT8)
      setIdentityInt8DynamicRange(tensorResult);
    valueMap.insert(opResult, tensorResult);
  }
  updateLowerPrecisionIndicators(op.getOperation(), usesFp16, usesInt8, usesFp8,
                                 usesBf16, usesInt4);

  return success();
}

/// For a given block, try to add all ops to `network` and populate
/// `valueMap` with its results. If `op` doesn't not represent a TensorRT
/// dialect operation, then return failure.
/// TODO: change this to non-recursive implementation.
LogicalResult NvInferNetworkEncoder::encodeBlock(Block &block) {
  for (Operation &op : block.without_terminator()) {
    auto trtOp = dyn_cast<tensorrt::TensorRTEncodingOpInterface>(op);
    if (!trtOp)
      return op.emitOpError() << "not a TensorRTEncodingOpInterface operation";

    if (failed(encodeOp(trtOp)))
      return op.emitOpError() << "failed to encode operation";
  }
  return success();
}

LogicalResult NvInferNetworkEncoder::encodeRegion(Region &region) {
  for (Block &block : region.getBlocks()) {
    if (failed(encodeBlock(block)))
      return emitError(block.front().getLoc()) << "failed to encode block";
  }
  return success();
}

static void packNonSplatInt4Tensor(ElementsAttr values,
                                   std::vector<int8_t> &result) {
  auto APIntValues = values.getValues<APInt>();
  auto iter = APIntValues.begin();
  int64_t rIdx = 0;
  while (iter != APIntValues.end()) {
    // 2 INT4 are packed in an INT8 in the little-endian format.
    // For example, {0,1} is packed as 0x10.
    int8_t packed = 0;
    uint8_t first = *reinterpret_cast<const uint8_t *>((*iter).getRawData());
    packed |= (first & 0x0F);
    iter++;
    uint8_t second =
        iter == APIntValues.end()
            ? 0
            : *reinterpret_cast<const uint8_t *>((*iter).getRawData());
    packed |= ((second & 0x0F) << 4);
    iter++;
    result[rIdx] = packed;
    rIdx++;
  }
}

static void packDenseResourceInt4Tensor(AsmResourceBlob *blob,
                                        std::vector<int8_t> &result) {
  // 2 INT4's from blob are packed in an INT8 in the little-endian format.
  // For example, {0,1} is packed as 0x10.
  // NOTE* MLIR represents INT4 as lower 4 bits of INT8.
  assert(result.size() == llvm::divideCeil(blob->getData().size(), 2) &&
         "for INT4 data, size(result) == ceil(size(blob), 2)");
  ArrayRef<char> blobData = blob->getData();
  size_t iterIdx = 0;
  for (size_t i = 0; i < result.size(); i++) {
    int8_t resultElement = 0;
    uint8_t first = blobData[iterIdx];
    resultElement |= (first & 0x0F);
    iterIdx++;
    uint8_t second = iterIdx == blobData.size() ? 0 : blobData[iterIdx];
    resultElement |= ((second & 0x0F) << 4);
    iterIdx++;
    result[i] = resultElement;
  }
}

static LogicalResult serializeSplatElements(DenseIntOrFPElementsAttr values,
                                            std::vector<int8_t> &data) {
  assert(values.isSplat() && "expected SplatElementsAttr");

  auto rtt = cast<RankedTensorType>(values.getType());
  if (rtt.getElementType().isInteger(32)) {
    std::fill_n(reinterpret_cast<int32_t *>(data.data()),
                values.getNumElements(), values.getSplatValue<int32_t>());
    return llvm::success();
  }
  if (rtt.getElementType().isInteger(64)) {
    std::fill_n(reinterpret_cast<int64_t *>(data.data()),
                values.getNumElements(), values.getSplatValue<int64_t>());
    return llvm::success();
  }
  if (rtt.getElementType().isInteger(8)) {
    std::fill_n(reinterpret_cast<int8_t *>(data.data()),
                values.getNumElements(), values.getSplatValue<int8_t>());
    return llvm::success();
  }
  if (rtt.getElementType().isF32()) {
    std::fill_n(reinterpret_cast<float *>(data.data()), values.getNumElements(),
                values.getSplatValue<float>());
    return llvm::success();
  }
  if (rtt.getElementType().isF16() || rtt.getElementType().isBF16()) {
    APInt tmp = values.getSplatValue<APFloat>().bitcastToAPInt();
    assert(tmp.getBitWidth() == 16 && "unexpected bitwidth");
    uint16_t fillValue = *reinterpret_cast<const uint16_t *>(tmp.getRawData());
    std::fill_n(reinterpret_cast<uint16_t *>(data.data()),
                values.getNumElements(), fillValue);
    return llvm::success();
  }
  if (isa<Float8E4M3FNType>(rtt.getElementType())) {
    APInt tmp = values.getSplatValue<APFloat>().bitcastToAPInt();
    assert(tmp.getBitWidth() == 8 && "unexpected bitwidth");
    uint8_t fillValue = *reinterpret_cast<const uint8_t *>(tmp.getRawData());
    std::fill_n(reinterpret_cast<uint8_t *>(data.data()),
                values.getNumElements(), fillValue);
    return llvm::success();
  }
  if (rtt.getElementType().isInteger(4)) {
    APInt tmp = values.getSplatValue<APInt>();
    assert(tmp.getBitWidth() == 4 && "expected 4 bit integer");
    uint8_t packed = 0;
    uint8_t value = *reinterpret_cast<const uint8_t *>(tmp.getRawData());
    // Pack `value` in the upper and the lower nibble
    packed |= (value & 0x0F);
    packed |= ((value & 0x0F) << 4);
    // Fill `data` vector with `packed`
    std::fill_n(reinterpret_cast<uint8_t *>(data.data()), data.size(), packed);
    return llvm::success();
  }

  return emitError(UnknownLoc::get(values.getContext()))
         << "unsupported data type to convert MLIR splat attribute to TensorRT "
            "weights!";
}

FailureOr<nvinfer1::Weights>
NvInferNetworkEncoder::getNvInferWeights(ElementsAttr values) {
  if (llvm::endianness::native == llvm::endianness::big)
    llvm_unreachable("big endian system currently not implemented");

  auto rtt = dyn_cast<RankedTensorType>(values.getType());
  if (!rtt)
    return kNullWeights;

  nvinfer1::Weights weights;
  weights.count = rtt.getNumElements();
  FailureOr<nvinfer1::DataType> tensorrtType = getNvInferDataType(
      UnknownLoc::get(values.getContext()), rtt.getElementType());
  if (failed(tensorrtType))
    return failure();
  weights.type = *tensorrtType;

  // Since Attributes are uniqued in the MLIR context, there should be no
  // duplicate attributes. If this attribute is already present in the map, then
  // we can just re-use its value.
  if (weightsMap.find(values) != weightsMap.end()) {
    weights.values = weightsMap[values].data();
    return weights;
  }

  // Pre-allocate a buffer for holding the data.
  if (rtt.getElementType().isInteger(4)) {
    // TensorRT expects INT4 data to be packed thus size of buffer will be
    // ceil(numInt4Elements/2). Ceil to handle odd elements case where 0 is
    // packed as the last element.
    weightsMap[values] =
        std::vector<int8_t>(llvm::divideCeil(rtt.getNumElements(), 2));
  } else {
    weightsMap[values] = std::vector<int8_t>(
        rtt.getNumElements() *
        llvm::divideCeil(rtt.getElementType().getIntOrFloatBitWidth(),
                         CHAR_BIT));
  }
  std::vector<int8_t> &data = weightsMap[values];
  weights.values = data.data();

  if (values.isSplat() && isa<DenseIntOrFPElementsAttr>(values)) {
    LogicalResult status = serializeSplatElements(
        cast<DenseIntOrFPElementsAttr>(values), weightsMap[values]);
    if (failed(status))
      return failure();
    return weights;
  }
  // Handle dense resources with non-elided handle
  if (auto denseResourceAttr = dyn_cast<DenseResourceElementsAttr>(values)) {
    DenseResourceElementsHandle handle = denseResourceAttr.getRawHandle();
    if (handle.getKey() != "__elided__") {
      AsmResourceBlob *blob = handle.getBlob();
      if (!blob)
        return failure();
      // Handle i4 element type specially
      if (denseResourceAttr.getElementType().isInteger(4)) {
        packDenseResourceInt4Tensor(blob, weightsMap[values]);
        return weights;
      }
      // Handle everything else
      if (blob->getData().size() != data.size())
        return failure();
      llvm::copy(blob->getData(), data.data());
      return weights;
    }
  }

  // How we serialize the weights to TensorRT's format depends on the data
  // type.
  if (mlir::getElidedResourceElementsAttr(values)) {
    // We also handle elided attributes by generating weights filled with zeros.
    std::memset(reinterpret_cast<void *>(data.data()), 0, data.size());
  } else if (rtt.getElementType().isInteger(64)) {
    llvm::copy(values.getValues<int64_t>(),
               reinterpret_cast<int64_t *>(data.data()));
  } else if (rtt.getElementType().isInteger(32)) {
    llvm::copy(values.getValues<int32_t>(),
               reinterpret_cast<int32_t *>(data.data()));
  } else if (rtt.getElementType().isInteger(8)) {
    llvm::copy(values.getValues<int8_t>(),
               reinterpret_cast<int8_t *>(data.data()));
  } else if (rtt.getElementType().isF32()) {
    llvm::copy(values.getValues<float>(),
               reinterpret_cast<float *>(data.data()));
  } else if (rtt.getElementType().isF16()) {
    auto dst = llvm::MutableArrayRef(reinterpret_cast<uint16_t *>(data.data()),
                                     rtt.getNumElements());
    for (auto [index, v] : llvm::enumerate(values.getValues<APFloat>())) {
      assert(&v.getSemantics() == &APFloat::IEEEhalf() &&
             "expected IEEE fp16 semantics");
      dst[index] = v.bitcastToAPInt().getZExtValue();
    }
  } else if (isa<Float8E4M3FNType>(rtt.getElementType())) {
    auto dst = llvm::MutableArrayRef(reinterpret_cast<uint8_t *>(data.data()),
                                     rtt.getNumElements());
    for (auto [index, v] : llvm::enumerate(values.getValues<APFloat>())) {
      assert(&v.getSemantics() == &APFloat::Float8E4M3FN() &&
             "expected Float8 f8E4M3FN semantics");
      dst[index] = v.bitcastToAPInt().getZExtValue();
    }
  } else if (rtt.getElementType().isBF16()) {
    auto dst = llvm::MutableArrayRef(reinterpret_cast<uint16_t *>(data.data()),
                                     rtt.getNumElements());
    for (auto [index, v] : llvm::enumerate(values.getValues<APFloat>())) {
      assert(&v.getSemantics() == &APFloat::BFloat() &&
             "expected bf16 semantics");
      dst[index] = v.bitcastToAPInt().getZExtValue();
    }
  } else if (rtt.getElementType().isInteger(4)) {
    packNonSplatInt4Tensor(values, weightsMap[values]);
  } else {
    llvm_unreachable(
        "unsupported data type to convert MLIR attribute to TensorRT weights!");
  }
  return weights;
}

FailureOr<nvinfer1::Weights>
NvInferNetworkEncoder::getNvInferWeights(std::optional<ElementsAttr> attr) {
  if (!attr)
    return kNullWeights;
  return this->getNvInferWeights(*attr);
}

/// Returns true if the "elementTypeOrSelf" of `t` has a corresponding TensorRT
/// enum value.
static bool isValidTensorRTInputType(Type t) {
  Type elType = mlir::getElementTypeOrSelf(t);
  return elType.isF32() || elType.isF16() || isTensorRTInt8Type(elType) ||
         elType.isInteger(32) || elType.isInteger(1) || elType.isInteger(4) ||
         elType.isUnsignedInteger(8) || isa<Float8E4M3FNType>(elType) ||
         isa<Float4E2M1FNType>(elType) || elType.isBF16() ||
         elType.isInteger(64);
}

/// Add the argument and shape information to the optimization profile.
static void setProfileDimensions(nvinfer1::IOptimizationProfile *profile,
                                 const std::string &argName,
                                 ArrayRef<int64_t> minShape,
                                 ArrayRef<int64_t> optShape,
                                 ArrayRef<int64_t> maxShape) {
  profile->setDimensions(argName.c_str(), nvinfer1::OptProfileSelector::kMIN,
                         getNvInferDims(minShape));
  profile->setDimensions(argName.c_str(), nvinfer1::OptProfileSelector::kOPT,
                         getNvInferDims(optShape));
  profile->setDimensions(argName.c_str(), nvinfer1::OptProfileSelector::kMAX,
                         getNvInferDims(maxShape));
}

using NvInferShapeValueType = std::remove_const_t<std::remove_reference_t<
    decltype(std::declval<nvinfer1::IOptimizationProfile>().getShapeValues(
        "arg0", nvinfer1::OptProfileSelector::kMIN)[0])>>;

static NvInferShapeValueType clampToNvInferShapeValueType(int64_t x) {
  return static_cast<NvInferShapeValueType>(std::max<int64_t>(
      std::min<int64_t>(x, std::numeric_limits<NvInferShapeValueType>::max()),
      std::numeric_limits<NvInferShapeValueType>::min()));
}

/// Add the argument and shape tensor bounds information to the optimization
/// profile.
static void setShapeTensorInputProfile(nvinfer1::IOptimizationProfile *profile,
                                       const std::string &argName,
                                       ArrayRef<int64_t> minShape,
                                       ArrayRef<int64_t> optShape,
                                       ArrayRef<int64_t> maxShape) {
  using nvinfer1::OptProfileSelector;
  SmallVector<OptProfileSelector> profiles{OptProfileSelector::kMIN,
                                           OptProfileSelector::kOPT,
                                           OptProfileSelector::kMAX};
  SmallVector<ArrayRef<int64_t>> shapes{minShape, optShape, maxShape};
  for (auto [kind, shape] : llvm::zip(profiles, shapes)) {
    nvinfer1::Dims dims = getNvInferDims(shape);
    SmallVector<NvInferShapeValueType> shapeValues;
    shapeValues.reserve(dims.nbDims);
    for (int32_t i = 0, e = dims.nbDims; i < e; ++i)
      shapeValues.push_back(clampToNvInferShapeValueType(dims.d[i]));
    profile->setShapeValues(argName.c_str(), kind, shapeValues.data(),
                            shapeValues.size());
  }
}

/// Add the argument and shape information to the optimization profile.
static void setProfileDimensions(nvinfer1::IOptimizationProfile *profile,
                                 const std::string &argName,
                                 ShapeProfileAttr shapeInfo) {
  setProfileDimensions(profile, argName, shapeInfo.getMin(), shapeInfo.getOpt(),
                       shapeInfo.getMax());
}

/// Add the argument and shape tensor bounds information to the optimization
/// profile.
static void setShapeTensorInputProfile(nvinfer1::IOptimizationProfile *profile,
                                       const std::string &argName,
                                       ShapeProfileAttr shapeInfo) {
  setShapeTensorInputProfile(profile, argName, shapeInfo.getMin(),
                             shapeInfo.getOpt(), shapeInfo.getMax());
}

/// Return the assumed unique terminator of the given function-like op.
static Operation *getAssumedUniqueReturnOp(FunctionOpInterface op) {
  assert(op.getFunctionBody().hasOneBlock() &&
         "only single-block function-like region supported");
  return op.getFunctionBody().front().getTerminator();
}

// Passing an argument directly to a terminator is perfectly valid. However,
// TensorRT does not allows this. Therefore, we insert an identity layer if the
// value is a function block arg. Furthermore, a single identity layer can't be
// marked as an output for multiple result indices. See Issue #591.
static LogicalResult
insertIdentityForPassthroughArgs(FunctionOpInterface func) {
  Operation *term = getAssumedUniqueReturnOp(func);
  OpBuilder b(func->getContext());
  for (auto [idx, arg] : llvm::enumerate(term->getOperands())) {
    // Check if the argument is passed directly to the terminator. If so, insert
    // an identity layer. Further
    if (isa<BlockArgument>(arg)) {
      b.setInsertionPoint(term);
      auto identityOp = b.create<IdentityOp>(func.getLoc(), arg.getType(), arg);
      term->setOperand(idx, identityOp);
    }
  }
  return success();
}

std::vector<std::string>
tensorrt::getResultTensorNames(unsigned numResults,
                               const NvInferNetworkEncoder &encoder) {
  auto range =
      llvm::map_range(llvm::seq<unsigned>(0, numResults), [](unsigned idx) {
        return llvm::formatv("result{0}", idx).str();
      });
  return std::vector<std::string>(range.begin(), range.end());
}

static LogicalResult encodeFuncTerminator(NvInferNetworkEncoder &encoder,
                                          Operation *terminator) {
  llvm::SmallPtrSet<nvinfer1::ITensor *, 4> outputsSet;
  std::vector<std::string> names =
      getResultTensorNames(terminator->getNumOperands(), encoder);
  for (auto [idx, returnedVal] : llvm::enumerate(terminator->getOperands())) {
    assert(encoder.contains(returnedVal) != 0 &&
           "all values should be mapped to tensors");
    nvinfer1::ITensor *outputTensor = encoder.lookup(returnedVal);
    FailureOr<nvinfer1::DataType> returnDataType =
        getNvInferDataType(terminator->getLoc(), returnedVal.getType());
    if (failed(returnDataType))
      return failure();
    // As a convention, we cannot have the same tensor alias multiple results.
    // Therefore, we need to insert an identity layer to differentiate.
    if (outputsSet.contains(outputTensor)) {
      if (encoder.isStronglyTyped() &&
          outputTensor->getType() != returnDataType) {
        // Add cast layer in strongly typed mode when the output dtype is same
        // as MLIR type
        nvinfer1::ICastLayer *castOutputLayer =
            encoder.getNetworkDefinition()->addCast(*outputTensor,
                                                    *returnDataType);
        outputTensor = castOutputLayer->getOutput(0);
      } else {
        // Still need an identity layer to support returning input arguments
        // directly when the output dtype is same as MLIR type
        nvinfer1::IIdentityLayer *identityOutputLayer =
            encoder.getNetworkDefinition()->addIdentity(*outputTensor);
        if (!encoder.isStronglyTyped())
          identityOutputLayer->setOutputType(0, outputTensor->getType());
        outputTensor = identityOutputLayer->getOutput(0);
      }
      // Ensure we re-map this value incase the user looks it up later.
      encoder.map(returnedVal, outputTensor);
    }
    encoder.getNetworkDefinition()->markOutput(*outputTensor);
    encoder.getNetworkDefinition()->getOutput(idx)->setType(*returnDataType);
    outputsSet.insert(outputTensor);
    outputTensor->setName(names[idx].c_str());
  }
  return success();
}

/// Returns true if the func contains a QDQ operation.
static bool hasQDQOperation(FunctionOpInterface func) {
  return func
      ->walk([](Operation *op) {
        if (isa<tensorrt::QuantizeOp, tensorrt::DequantizeOp>(op))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

LogicalResult NvInferNetworkEncoder::encodeFunc(FunctionOpInterface func) {
  int32_t idx = 0;
  hasQDQOps = hasQDQOperation(func);
  TensorMapScope regionScope(valueMap);

  if (failed(insertIdentityForPassthroughArgs(func)))
    return func->emitOpError()
           << "failed to insert passthrough identity layers";

  for (BlockArgument arg : func.getArguments()) {
    RankedTensorType argType = dyn_cast<RankedTensorType>(arg.getType());
    if (!argType)
      return func.emitOpError() << "expect all inputs to be ranked tensors";
    if (!isValidTensorRTInputType(argType))
      return func->emitError()
             << "input does not have valid TensorRT type: " << argType;

    // Add the argument as an input.
    std::string name = "arg" + std::to_string(idx++);
    nvinfer1::Dims trtShape = getNvInferDims(argType.getShape());
    FailureOr<nvinfer1::DataType> dtype =
        getNvInferDataType(func->getLoc(), argType);
    if (failed(dtype))
      return failure();
    nvinfer1::ITensor *inputTensor =
        getNetworkDefinition()->addInput(name.c_str(), *dtype, trtShape);

    // setDimensionName must be called immediately after addInput, or TensorRT
    // will not deduplicate equal dimensions, which leads to perf gaps.
    auto dimNamesAttr = func.getArgAttrOfType<DictionaryAttr>(
        arg.getArgNumber(), TensorRTDialect::getDimensionNamesArgAttrName());
    if (dimNamesAttr) {
      for (NamedAttribute namedAttr : dimNamesAttr) {
        int32_t key;
        if (namedAttr.getName().getValue().getAsInteger(10, key))
          return func->emitOpError()
                 << "dimension name key '" << namedAttr.getName()
                 << "' is not an integer";

        if (key < 0 || key >= argType.getRank())
          return func->emitOpError()
                 << "dimension name key '" << key
                 << "' is out of bounds for rank " << argType.getRank();

        StringAttr strAttr = dyn_cast<StringAttr>(namedAttr.getValue());
        if (!strAttr)
          return func->emitOpError()
                 << "dimension name value '" << namedAttr.getValue()
                 << "' is not a string";

        inputTensor->setDimensionName(key, strAttr.getValue().str().c_str());
      }
    }

    if (!usesStronglyTyped && dtype == nvinfer1::DataType::kINT8)
      setIdentityInt8DynamicRange(inputTensor);
    this->map(arg, inputTensor);
  }

  if (failed(this->encodeRegion(func.getFunctionBody())))
    return failure();

  // Mark outputs.
  Operation *term = getAssumedUniqueReturnOp(func);
  if (term->getNumOperands() == 0)
    return func->emitOpError()
           << "TensorRT engine function must have >=1 results";

  if (failed(encodeFuncTerminator(*this, term)))
    return emitError(func->getLoc(), "failed to encode function terminator");

  // It's possible for TensorRT INetworkDefinition to not pass errors that alter
  // the function signature (see #591). So do a sanity check on the
  // function/network signature to catch such errors.
  if (failed(sanityCheckFuncSignature(getNetworkDefinition(), func)))
    return emitError(func.getLoc()) << "TensorRT network signature does not "
                                       "match the function type signature";

  // Encode shape profile / shape value bounds information.
  for (BlockArgument arg : func.getArguments()) {
    RankedTensorType argType = cast<RankedTensorType>(arg.getType());
    nvinfer1::ITensor *inputTensor = this->lookup(arg);
    std::string name = inputTensor->getName();
    bool isShapeTensor = func.getArgAttr(arg.getArgNumber(),
                                         getHostTensorArgAttrName()) != nullptr;
    if (isShapeTensor && !inputTensor->isShapeTensor()) {
      isShapeTensor = false;
      emitWarning(arg.getLoc())
          << "expected argument#" << arg.getArgNumber()
          << " to be a shape tensor, but TensorRT reports that it is not a "
             "shape tensor; treating as execution tensor";
    }
    if (!isShapeTensor && inputTensor->isShapeTensor()) {
      return emitError(arg.getLoc())
             << "expected argument #" << arg.getArgNumber() << " of function '"
             << func.getName()
             << "' to be a device tensor, but TensorRT reports that it is a "
                "host tensor";
    }

    if (isShapeTensor) {
      auto shapeBounds = func.getArgAttrOfType<ShapeProfileAttr>(
          arg.getArgNumber(),
          TensorRTDialect::getShapeTensorValueBoundsArgAttrName());
      if (!shapeBounds)
        return emitError(arg.getLoc())
               << "argument#" << arg.getArgNumber()
               << " is a shape tensor, but it does not have "
                  "shape value bounds specified";
      setShapeTensorInputProfile(profile, name, shapeBounds);
      continue;
    }

    // For execution tensors with static shapes, there is nothing to do.
    if (argType.hasStaticShape())
      continue;

    FailureOr<ShapeProfileAttr> info =
        getArgumentShapeProfile(func, arg.getArgNumber());
    if (failed(info))
      return func->emitError()
             << "could not resolve shape_min/shape_opt/shape_max attributes"
                "for arg "
             << arg.getArgNumber();
    setProfileDimensions(profile, name, *info);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Tablegen'd Declarations for nvinfer<->mlir enum converters.
//===----------------------------------------------------------------------===//

namespace mlir::tensorrt {
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#define GEN_TRT_ENUM_CONVERTER_DEFS
#include "mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/EnumConverters.inc.cpp"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
} // namespace mlir::tensorrt

std::optional<nvinfer1::SliceMode>
mlir::tensorrt::convertSliceModeToNvInferEnum(SliceMode value) {
  switch (value) {
  case SliceMode::kDEFAULT:
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
    return nvinfer1::SliceMode::kSTRICT_BOUNDS;
#else
    return nvinfer1::SliceMode::kDEFAULT;
#endif
  case SliceMode::kCLAMP:
    return nvinfer1::SliceMode::kCLAMP;
  case SliceMode::kFILL:
    return nvinfer1::SliceMode::kFILL;
  case SliceMode::kREFLECT:
    return nvinfer1::SliceMode::kREFLECT;
  case SliceMode::kWRAP:
    return nvinfer1::SliceMode::kWRAP;
  }
  llvm_unreachable("unhandled SliceMode nvinfer translation");
  return std::nullopt;
}

std::optional<nvinfer1::ActivationType>
mlir::tensorrt::convertActivationTypeToNvInferEnum(ActivationType value) {
  switch (value) {
  case ActivationType::kRELU:
    return nvinfer1::ActivationType::kRELU;
  case ActivationType::kSIGMOID:
    return nvinfer1::ActivationType::kSIGMOID;
  case ActivationType::kTANH:
    return nvinfer1::ActivationType::kTANH;
  case ActivationType::kLEAKY_RELU:
    return nvinfer1::ActivationType::kLEAKY_RELU;
  case ActivationType::kELU:
    return nvinfer1::ActivationType::kELU;
  case ActivationType::kSELU:
    return nvinfer1::ActivationType::kSELU;
  case ActivationType::kSOFTSIGN:
    return nvinfer1::ActivationType::kSOFTSIGN;
  case ActivationType::kSOFTPLUS:
    return nvinfer1::ActivationType::kSOFTPLUS;
  case ActivationType::kCLIP:
    return nvinfer1::ActivationType::kCLIP;
  case ActivationType::kHARD_SIGMOID:
    return nvinfer1::ActivationType::kHARD_SIGMOID;
  case ActivationType::kSCALED_TANH:
    return nvinfer1::ActivationType::kSCALED_TANH;
  case ActivationType::kTHRESHOLDED_RELU:
    return nvinfer1::ActivationType::kTHRESHOLDED_RELU;
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_LT(10, 0, 0)
  case ActivationType::kGELU_ERF:
    return std::nullopt;
  case ActivationType::kGELU_TANH:
    return std::nullopt;
#elif MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  case ActivationType::kGELU_ERF:
    return nvinfer1::ActivationType::kGELU_ERF;
  case ActivationType::kGELU_TANH:
    return nvinfer1::ActivationType::kGELU_TANH;
#endif
  }
  llvm_unreachable("unknown NvInfer enum conversion from MLIR ActivationType");
}
