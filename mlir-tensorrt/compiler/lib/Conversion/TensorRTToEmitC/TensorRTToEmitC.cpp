//===- TensorRTToEmitC.cpp --------------------------------------*- c++ -*-===//
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
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Utils/Utils.h"
#include "mlir-tensorrt-dialect/Utils/TensorRTVersion.h"
#include "mlir-tensorrt/Conversion/Passes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORRTTOEMITCPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::tensorrt;

using ValueMap = llvm::DenseMap<Value, Value>;
using COpaqueAttr = emitc::OpaqueAttr;
using COpaqueType = emitc::OpaqueType;

//===----------------------------------------------------------------------===//
// Utility functions for creating handles to common types.
//===----------------------------------------------------------------------===//
static COpaqueType getWeightsMapType(MLIRContext *ctx) {
  return COpaqueType::get(
      ctx, "std::unordered_map<const char*, std::vector<uint8_t>>&");
}
static COpaqueType getNvInferBuilderType(MLIRContext *ctx) {
  return COpaqueType::get(ctx, "::nvinfer1::IBuilder");
}
static COpaqueType getNvInferNetworkDefinitionType(MLIRContext *ctx) {
  return COpaqueType::get(ctx, "::nvinfer1::INetworkDefinition");
}
static COpaqueType getNvInferBuilderConfigType(MLIRContext *ctx) {
  return COpaqueType::get(ctx, "::nvinfer1::IBuilderConfig");
}
static COpaqueType getNvInferOptimizationProfileType(MLIRContext *ctx) {
  return COpaqueType::get(ctx, "::nvinfer1::IOptimizationProfile");
}
static emitc::PointerType getPointerType(COpaqueType t) {
  return emitc::PointerType::get(t);
}
static emitc::PointerType getNvInferITensorPtrType(MLIRContext *ctx) {
  return getPointerType(COpaqueType::get(ctx, "::nvinfer1::ITensor"));
}
static COpaqueType getNvInferWeightsType(MLIRContext *ctx) {
  return COpaqueType::get(ctx, "::nvinfer1::Weights");
}
static COpaqueType getStdioLoggerType(MLIRContext *ctx) {
  return COpaqueType::get(
      ctx, "::std::unique_ptr<::nvinfer1::adaptor::StdioLogger>");
}
static COpaqueType getAdaptorBuilderType(MLIRContext *ctx) {
  return COpaqueType::get(ctx, "::std::unique_ptr<::nvinfer1::IBuilder>");
}
static COpaqueType getAdaptorNetworkType(MLIRContext *ctx) {
  return COpaqueType::get(ctx,
                          "::std::unique_ptr<::nvinfer1::INetworkDefinition>");
}
static COpaqueType getAdaptorBuilderConfigType(MLIRContext *ctx) {
  return COpaqueType::get(ctx, "::std::unique_ptr<::nvinfer1::IBuilderConfig>");
}
static COpaqueType getAdaptorHostMemoryType(MLIRContext *ctx) {
  return COpaqueType::get(ctx, "::std::unique_ptr<::nvinfer1::IHostMemory>");
}
static COpaqueType getAdaptorWeightsMapType(MLIRContext *ctx) {
  return COpaqueType::get(
      ctx, "std::unordered_map<const char*, std::vector<uint8_t>>");
}
static COpaqueAttr getEscapedLiteral(MLIRContext *ctx, StringRef lit) {
  return COpaqueAttr::get(ctx, ("\"" + lit + "\"").str());
}

/// Returns an nvinfer1::DataType enum value as an emitc OpaqueAttr.
static COpaqueAttr getNvInferDataTypeEnumAttr(Type elType) {
  auto getOpaqueAttr = [&](StringRef name) {
    return COpaqueAttr::get(elType.getContext(), name);
  };
  if (elType.isF32())
    return getOpaqueAttr("::nvinfer1::DataType::kFLOAT");
  if (elType.isF16())
    return getOpaqueAttr("nvinfer1::DataType::kHALF");
  if (isa<Float8E4M3FNType>(elType))
    return getOpaqueAttr("nvinfer1::DataType::kFP8");
  if (elType.isInteger(32))
    return getOpaqueAttr("nvinfer1::DataType::kINT32");
  if (elType.isInteger(8))
    return getOpaqueAttr("nvinfer1::DataType::kINT8");
  if (elType.isInteger(1))
    return getOpaqueAttr("nvinfer1::DataType::kBOOL");
  if (elType.isBF16())
    return getOpaqueAttr("nvinfer1::DataType::kBF16");
  llvm_unreachable("invalid MLIR -> TRT type conversion");
}

/// Returns an nvinfer1::Permutation value as an emitc OpaqueAttr.
/// TODO: this is currently done via string interpolation. A better way may
/// be to create an adaptor method for constructing the permutation and
/// let emitc cpp translation handle translating elements attr to string.
template <typename T, std::enable_if_t<std::is_pod<T>::value, T *> = nullptr>
static COpaqueAttr getNvInferPermutation(MLIRContext *ctx, ArrayRef<T> t) {
  std::string str;
  llvm::raw_string_ostream ss(str);
  ss << "::nvinfer1::Permutation{{";
  llvm::interleaveComma(t, ss);
  ss << "}}";
  ss.flush();
  return COpaqueAttr::get(ctx, ss.str());
}

/// Returns nvinfer1::Dims as an OpaqueAttr via string interpolation.
/// TODO: this could be done via an adaptor method, same as above.
template <typename T, std::enable_if_t<std::is_pod<T>::value, T *> = nullptr>
static COpaqueAttr getNvInferDimsOpaqueAttr(MLIRContext *ctx, ArrayRef<T> t) {
  std::string str;
  llvm::raw_string_ostream ss(str);
  ss << "::nvinfer1::Dims{" << t.size() << ", {";
  llvm::interleaveComma(t, ss, [&](int64_t x) {
    if (x == ShapedType::kDynamic) {
      ss << "-1";
      return;
    }
    ss << static_cast<int32_t>(x);
  });
  ss << "}}";
  ss.flush();
  return COpaqueAttr::get(ctx, ss.str());
}

/// Given an optional array ref of integers (for shape), either return null
/// dims or return nvinfer1::Dims attr as OpaqueAttr.
template <typename T, std::enable_if_t<std::is_pod<T>::value, T *> = nullptr>
static COpaqueAttr getNvInferDimsOpaqueAttr(MLIRContext *ctx,
                                            std::optional<ArrayRef<T>> t) {
  if (!t.has_value())
    return COpaqueAttr::get(ctx, "::nvinfer1::adaptor::OptionalDims::None()");
  return getNvInferDimsOpaqueAttr(ctx, *t);
}

namespace {
class EmitCConverter {
public:
  EmitCConverter(Value network, Value weightsMap)
      : network(network), weightsMap(weightsMap) {}

  /// Lookup the TRT ITensor* equivalent of a Value.
  Value lookup(OpBuilder &b, Value v);

  /// Add a map from a Value to a TRT ITEnsor*.
  void map(Value from, Value to) { valueMap.insert(from, to); }

  /// Check whether the value map contains `v`.
  size_t contains(Value v) { return valueMap.count(v); }

  /// Get a Weights from an elements attr.
  Value getNvInferWeights(OpBuilder &b, Location loc, ElementsAttr values);

  /// Get a Weights from an optional elements attr. If attr is not present,
  /// then return kNullWeights.
  Value getNvInferWeights(OpBuilder &b, Location loc,
                          std::optional<ElementsAttr> attr);

  /// For a given operation, try to add that operation to `network` and populate
  /// `valueMap` with its results. If `op` doesn't not represent a TensorRT
  /// dialect operation, then return failure.
  LogicalResult encodeOp(OpBuilder &b, tensorrt::TensorRTOpInterface op);

  /// For a given block, try to add all ops to `network` and populate
  /// `valueMap` with its results. If `op` doesn't not represent a TensorRT
  /// dialect operation, then return failure.
  /// TODO: change this to non-recursive implementation.
  LogicalResult encodeBlock(OpBuilder &b, Block &block);

  /// Encode a given region to a TensorRT engine.
  LogicalResult encodeRegion(OpBuilder &b, Region &region);

  /// Encode a given function to a TensorRT engine.
  LogicalResult encodeFunc(OpBuilder &b, FunctionOpInterface func);

  /// A type that maps mlir::Value typed objects to the corresponding TensorRT
  /// ITensor object.
  using TensorMap = llvm::ScopedHashTable<mlir::Value, mlir::Value>;
  using TensorMapScope = TensorMap::ScopeTy;

  TensorMap &getTensorMap() { return valueMap; }

  Value getNetworkDefinitionPtr() { return network; }

private:
  TensorMap valueMap;
  Value network;
  Value weightsMap;
};

/// A utility to more easily construct `emitc::CallOpaqueOp`s.
/// TODO: remove this when better builders are created upstream.
struct EmitCall {
  EmitCall(MLIRContext *ctx, StringRef name) : ctx(ctx), name(name) {}

  EmitCall &pushOperand(Value operand) {
    values.push_back(operand);
    args.push_back(IntegerAttr::get(IndexType::get(ctx), values.size() - 1));
    return *this;
  }

  EmitCall &pushOperand(Attribute operand) {
    args.push_back(operand);
    return *this;
  }

  template <typename T>
  EmitCall &pushNvInferDims(T in) {
    return pushOperand(getNvInferDimsOpaqueAttr(ctx, in));
  }

  /// Add the `dimensions` list to the operand list as a bitmask (of type ui32).
  EmitCall &pushDimensionListAsBitMask(ArrayRef<int64_t> dimensions) {
    uint32_t bitMask = 0;
    for (int64_t axis : dimensions) {
      assert(axis >= 0 &&
             "expected elements in TensorRT_DimensionList to be non-negative");
      bitMask |= 1U << axis;
    }
    this->pushOperand(IntegerAttr::get(
        IntegerType::get(ctx, 32, IntegerType::SignednessSemantics::Unsigned),
        bitMask));
    return *this;
  }

  template <typename T>
  EmitCall &pushOperands(ArrayRef<T> operands) {
    for (auto x : operands)
      pushOperand(x);
    return *this;
  }

  EmitCall &pushOperands(ValueRange operands) {
    for (Value x : operands)
      pushOperand(x);
    return *this;
  }

  EmitCall &pushNvInferDataType(Type t) {
    return pushOperand(getNvInferDataTypeEnumAttr(t));
  }

  template <typename T>
  EmitCall &pushNvInferPermutation(ArrayRef<T> permutation) {
    return pushOperand(getTensorRTPermutation(permutation));
  }

  EmitCall &addTemplateParam(Attribute param) {
    templateArgs.push_back(param);
    return *this;
  }

  EmitCall &setResults(TypeRange resultTypes) {
    this->types = resultTypes;
    return *this;
  }

  emitc::CallOpaqueOp build(OpBuilder &b, Location loc) {
    assert(llvm::all_of(values, [](Value v) { return v != nullptr; }));
    return b.create<emitc::CallOpaqueOp>(
        loc, types, name, ArrayAttr::get(ctx, args),
        templateArgs.empty() ? ArrayAttr() : ArrayAttr::get(ctx, templateArgs),
        values);
  }

  emitc::CallOpaqueOp build(OpBuilder &b, Location loc, EmitCConverter &e) {
    for (unsigned i = 0; i < values.size(); i++)
      values[i] = e.lookup(b, values[i]);
    assert(llvm::all_of(values, [](Value v) { return v != nullptr; }));
    return b.create<emitc::CallOpaqueOp>(
        loc, types, name, ArrayAttr::get(ctx, args),
        templateArgs.empty() ? ArrayAttr() : ArrayAttr::get(ctx, templateArgs),
        values);
  }

  MLIRContext *ctx;
  SmallVector<Type> types;
  std::string name;
  SmallVector<Value> values;
  SmallVector<Attribute> args;
  SmallVector<Attribute> templateArgs;
  EmitCConverter *converter = nullptr;
};
} // namespace

/// For a given block, try to add all ops to `network` and populate
/// `valueMap` with its results. If `op` doesn't not represent a TensorRT
/// dialect operation, then return failure.
/// TODO: change this to non-recursive implementation.
LogicalResult EmitCConverter::encodeBlock(OpBuilder &b, Block &block) {
  for (Operation &op : block.without_terminator()) {
    auto trtOp = dyn_cast<tensorrt::TensorRTOpInterface>(op);
    if (!trtOp)
      return op.emitOpError() << "not a TensorRTOpInterface operation";

    if (failed(encodeOp(b, trtOp)))
      return op.emitOpError() << "failed to encode operation";
  }
  return success();
}

LogicalResult EmitCConverter::encodeRegion(OpBuilder &b, Region &region) {
  for (Block &block : region.getBlocks()) {
    if (failed(encodeBlock(b, block)))
      return emitError(block.front().getLoc()) << "failed to encode block";
  }
  return success();
}

static bool isElided(ElementsAttr elementsAttr) {
  if (auto denseResourceAttr =
          dyn_cast<DenseResourceElementsAttr>(elementsAttr)) {
    DenseResourceElementsHandle handle = denseResourceAttr.getRawHandle();
    if (handle.getKey() == "__elided__")
      return true;
  }
  return false;
}

/// Return a default value for filling elided constants.
/// This should only be relevant in debugging and testing situations.
static Attribute getSplatAttrValueForType(OpBuilder &b, Type elType) {
  if (elType.isF32())
    return b.getF32FloatAttr(0.1f);
  if (elType.isF16())
    return b.getF16FloatAttr(0.1f);
  if (elType.isInteger(32))
    b.getI32IntegerAttr(1);
  if (elType.isInteger(8))
    b.getI8IntegerAttr(1);
  if (elType.isInteger(1))
    b.getIntegerAttr(b.getI1Type(), 1);
  llvm_unreachable("invalid MLIR -> TRT type conversion");
}

Value EmitCConverter::getNvInferWeights(OpBuilder &b, Location loc,
                                        ElementsAttr weights) {
  assert(weightsMap && "expected weightsMap to be a valid Value");
  MLIRContext *ctx = b.getContext();
  RankedTensorType type = cast<RankedTensorType>(weights.getType());
  static unsigned constCount = 0;
  std::string name = "c" + std::to_string(constCount++);

  // If the weights are elided, emit a call to fill weights with splat.
  if (isElided(weights)) {
    return EmitCall(ctx, "nvinfer1::adaptor::trtSetWeightsSplat")
        .setResults({getNvInferWeightsType(ctx)})
        .pushOperand(weightsMap)
        .pushOperand(getEscapedLiteral(ctx, name))
        .pushOperand(b.getI64IntegerAttr(type.getNumElements()))
        .pushOperand(getSplatAttrValueForType(b, type.getElementType()))
        .addTemplateParam(TypeAttr::get(type.getElementType()))
        .build(b, loc)
        .getResult(0);
  }
  return EmitCall(ctx, "nvinfer1::adaptor::trtSetWeights")
      .setResults({getNvInferWeightsType(ctx)})
      .pushOperand(weightsMap)
      .pushOperand(getEscapedLiteral(ctx, name))
      .pushOperand(weights)
      .addTemplateParam(TypeAttr::get(type.getElementType()))
      .build(b, loc)
      .getResult(0);
}

Value EmitCConverter::getNvInferWeights(OpBuilder &b, Location loc,
                                        std::optional<ElementsAttr> attr) {
  if (!attr)
    return b.create<emitc::ConstantOp>(
        loc, getNvInferWeightsType(b.getContext()),
        COpaqueAttr::get(
            b.getContext(),
            "::nvinfer1::Weights{::nvinfer1::DataType::kFLOAT,nullptr,0}"));

  return this->getNvInferWeights(b, loc, *attr);
}

Value EmitCConverter::lookup(OpBuilder &b, Value v) {
  MLIRContext *ctx = b.getContext();
  emitc::PointerType tensorPtrType = getNvInferITensorPtrType(ctx);
  COpaqueAttr nullPtrAttr = COpaqueAttr::get(ctx, "nullptr");
  // If src is null, return nullptr value. This happens if `src` was an optional
  // SSA operand.
  if (!v)
    return b.create<emitc::ConstantOp>(b.getUnknownLoc(), tensorPtrType,
                                       nullPtrAttr);
  // Pass through non-Tensor types. Otherwise, we should remap them to the
  // appropriate Values that represent ITensor pointer values in the EmitC
  // dialect.
  if (!isa<RankedTensorType>(v.getType()))
    return v;
  assert(valueMap.count(v) > 0 && "expected value to be remapped");
  return valueMap.lookup(v);
}

/// Return the name of the C++ adaptor for the TRT API for adding `op` to the
/// TRT INetworkDefinition graph.
static std::string getLayerBuilderCallee(TensorRTOpInterface op) {
  std::string name = llvm::convertToCamelFromSnakeCase(
      op.getOperation()->getName().stripDialect());
  name[0] = llvm::toUpper(name[0]);
  return "::nvinfer1::adaptor::networkAdd" + name;
}

/// Generate the `emitc::CallOpaqueOp` args for building the call for builder
/// for `op`.
static emitc::CallOpaqueOp dispatchLayerBuilderCall(OpBuilder &b,
                                                    TensorRTOpInterface op,
                                                    EmitCConverter &e) {
  Location loc = op.getLoc();
  MLIRContext *ctx = b.getContext();
  Value network = e.getNetworkDefinitionPtr();
  SmallVector<Type> resultTypes(op->getNumResults(),
                                getNvInferITensorPtrType(ctx));
  auto call = EmitCall(ctx, getLayerBuilderCallee(op))
                  .setResults(resultTypes)
                  .pushOperand(network);
  // Some ops cannot be handled generically. These operations are: any op
  // that has API parameters derived from attributes and ops that have
  // multiple optional parameters.
  if (auto constOp = dyn_cast<tensorrt::ConstantOp>(op.getOperation())) {
    Value weights = e.getNvInferWeights(b, loc, constOp.getWeights());
    return call.pushNvInferDims(constOp.getType().getShape())
        .pushOperand(weights)
        .build(b, loc, e);
  }
  if (auto convOp = dyn_cast<ConvolutionOp>(op.getOperation())) {
    Value kernelWeights = e.getNvInferWeights(b, loc, convOp.getKernelStatic());
    Value biasWeights = e.getNvInferWeights(b, loc, convOp.getBiasStatic());
    int32_t numOutputMaps = convOp.getType().getDimSize(1);
    ArrayRef<int64_t> kernelSpatialShape = convOp.getKernelSpatialShape();
    return call
        .pushOperands({
            convOp.getInput(),
            convOp.getKernel(),
            convOp.getBias(),
            kernelWeights,
            biasWeights,
        })
        .pushOperands<Attribute>(
            {b.getI32IntegerAttr(numOutputMaps),
             getNvInferDimsOpaqueAttr(ctx, kernelSpatialShape),
             getNvInferDimsOpaqueAttr(ctx, convOp.getStride()),
             getNvInferDimsOpaqueAttr(ctx, convOp.getPrePadding()),
             getNvInferDimsOpaqueAttr(ctx, convOp.getPostPadding()),
             convOp.getNumGroupsAttr(),
             getNvInferDimsOpaqueAttr(ctx, convOp.getDilation())})
        .build(b, loc, e);
  }
  if (auto shuffleOp = dyn_cast<ShuffleOp>(*op)) {
    call.pushOperands<Value>(
        {shuffleOp.getInput(), shuffleOp.getDynamicReshape()});
    call.pushOperands<Attribute>(
        {getNvInferPermutation(ctx, shuffleOp.getFirstTranspose()),
         getNvInferDimsOpaqueAttr(ctx, shuffleOp.getReshape()),
         getNvInferPermutation(ctx, shuffleOp.getSecondTranspose()),
         shuffleOp.getZeroIsPlaceholderAttr()});
    return call.build(b, loc, e);
  }
  if (auto sliceOp = dyn_cast<SliceOp>(op.getOperation())) {
    auto sliceModeStr = stringifySliceMode(sliceOp.getMode());
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
    if (sliceModeStr == "kDEFAULT")
      sliceModeStr = "kSTRICT_BOUNDS";
#endif
    auto sliceMode =
        COpaqueAttr::get(ctx, ("::nvinfer1::SliceMode::" + sliceModeStr).str());
    return call.pushOperand(sliceOp.getInput())
        .pushOperand(sliceOp.getStart())
        .pushOperand(sliceOp.getSize())
        .pushOperand(sliceOp.getStride())
        .pushOperand(sliceOp.getFill())
        .pushNvInferDims(sliceOp.getStaticStart())
        .pushNvInferDims(sliceOp.getStaticSize())
        .pushNvInferDims(sliceOp.getStaticStride())
        .pushOperand(sliceMode)
        .build(b, loc, e);
  }
  if (auto reduceOp = dyn_cast<ReduceOp>(op.getOperation())) {
    auto mode = COpaqueAttr::get(ctx, cast<tensorrt::TensorRTEnumAttrInterface>(
                                          reduceOp.getReduceOperationAttr())
                                          .getNvInferEnumValueStr());
    return call.pushOperand(reduceOp.getInput())
        .pushOperand(reduceOp.getKeepDimensionsAttr())
        .pushDimensionListAsBitMask(reduceOp.getReduceAxes())
        .pushOperand(mode)
        .build(b, loc, e);
  }
  if (auto topKOp = dyn_cast<TopKOp>(op.getOperation())) {
    auto mode = COpaqueAttr::get(ctx, cast<tensorrt::TensorRTEnumAttrInterface>(
                                          topKOp.getTopkOperationAttr())
                                          .getNvInferEnumValueStr());
    return call.pushOperand(topKOp.getInput())
        .pushOperands<Attribute>({topKOp.getKAttr()})
        .pushDimensionListAsBitMask(topKOp.getAxis())
        .pushOperand(mode)
        .build(b, loc, e);
  }
  if (auto softmaxOp = dyn_cast<SoftMaxOp>(op.getOperation())) {
    return call.pushOperand(softmaxOp.getInput())
        .pushDimensionListAsBitMask(softmaxOp.getAxis())
        .build(b, loc, e);
  }
  if (auto concatOp = dyn_cast<ConcatenationOp>(op.getOperation())) {
    return call.pushOperand(concatOp.getAxisAttr())
        .pushOperands(concatOp.getInputs())
        .build(b, loc, e);
  }
  if (auto identityOp = dyn_cast<IdentityOp>(op.getOperation())) {
    return call.pushOperand(identityOp.getInput())
        .pushNvInferDataType(identityOp.getType().getElementType())
        .build(b, loc, e);
  }

  // All other operands can be handled generically in the below manner.

  // Add all the operands to the call.
  for (unsigned i = 0; i < op->getNumOperands(); i++)
    call.pushOperand(op->getOperand(i));

  // Add all the attributes to the call. We remap attributes to the appropriate
  // compile-time constant values like nvinfer Dims/Weights/enums.
  for (NamedAttribute namedAttr : op->getAttrs()) {
    Attribute attr = namedAttr.getValue();
    if (auto dimArray = dyn_cast<DenseI64ArrayAttr>(attr)) {
      call.pushNvInferDims(dimArray.asArrayRef());
      continue;
    }
    if (auto dimArray = dyn_cast<DenseI32ArrayAttr>(attr)) {
      call.pushNvInferDims(dimArray.asArrayRef());
      continue;
    }
    if (auto typeAttr = dyn_cast<TypeAttr>(attr)) {
      call.pushNvInferDataType(typeAttr.getValue());
      continue;
    }
    if (auto elAttr = dyn_cast<ElementsAttr>(attr)) {
      call.pushOperand(e.getNvInferWeights(b, loc, elAttr));
      continue;
    }

    if (auto trtEnumAttr =
            dyn_cast<tensorrt::TensorRTEnumAttrInterface>(attr)) {
      call.pushOperand(
          COpaqueAttr::get(ctx, trtEnumAttr.getNvInferEnumValueStr()));
      continue;
    }
    call.pushOperand(attr);
  }
  return call.build(b, loc, e);
}

LogicalResult EmitCConverter::encodeOp(OpBuilder &b,
                                       tensorrt::TensorRTOpInterface op) {
  emitc::CallOpaqueOp callOp = dispatchLayerBuilderCall(b, op, *this);
  if (!callOp)
    return op.emitOpError() << "failed to create emitc network "
                               "builder call";
  assert(callOp.getNumResults() == op->getNumResults());
  for (auto [trtResult, emitcResult] :
       llvm::zip(op->getResults(), callOp->getResults())) {
    this->map(trtResult, emitcResult);
  }
  return success();
}

/// Return the assumed unique terminator of the given function-like op.
static Operation *getAssumedUniqueReturnOp(FunctionOpInterface op) {
  assert(op.getFunctionBody().hasOneBlock() &&
         "only single-block function-like region supported");
  return op.getFunctionBody().front().getTerminator();
}

// Create the new function that represents a function that uses
// `nvinfer1::Inetwork*` to construct the given network represented by
// `op`. This uses the EmitC dialect.
LogicalResult EmitCConverter::encodeFunc(OpBuilder &b,
                                         FunctionOpInterface func) {
  MLIRContext *ctx = b.getContext();
  Location loc = func.getLoc();

  TensorMapScope scope(valueMap);

  // Add the inputs to the builder.
  for (auto [idx, argType] : llvm::enumerate(func.getArgumentTypes())) {
    auto rtt = dyn_cast<RankedTensorType>(argType);
    if (!rtt)
      return failure();
    std::string argName = "input_" + std::to_string(idx);
    auto callOp = EmitCall(ctx, "::nvinfer1::adaptor::networkAddInput")
                      .setResults(getNvInferITensorPtrType(ctx))
                      .pushOperand(network)
                      .pushOperand(getEscapedLiteral(ctx, argName))
                      .pushNvInferDataType(rtt.getElementType())
                      .pushNvInferDims(rtt.getShape())
                      .build(b, loc, *this);
    this->map(func.getArgument(idx), callOp->getResult(0));
  }

  // Emit calls to build other layers.
  if (failed(this->encodeRegion(b, func.getFunctionBody())))
    return failure();

  // Mark outputs.
  Operation *term = getAssumedUniqueReturnOp(func);
  if (term->getNumOperands() == 0)
    return func->emitOpError()
           << "TensorRT engine function must have >=1 results";

  for (Value v : term->getOperands()) {
    EmitCall(ctx, "::nvinfer1::adaptor::networkMarkOutput")
        .pushOperand(network)
        .pushOperand(v)
        .build(b, loc, *this);
  }
  return success();
}

// Create the new function that represents a function that uses
// actually builds serializes an engine.
static FailureOr<func::FuncOp> createEmitCTesterOp(ModuleOp moduleOp,
                                                   func::FuncOp op) {
  MLIRContext *ctx = moduleOp->getContext();
  Location loc = op.getLoc();
  // Create a builder function
  std::string funcName = (op.getName() + "_tester").str();
  auto builderFunc = func::FuncOp::create(
      loc, funcName,
      FunctionType::get(ctx, TypeRange{}, getAdaptorHostMemoryType(ctx)));
  Block *body = builderFunc.addEntryBlock();
  OpBuilder b(body, body->begin());

  auto getRawPtr = [&](COpaqueType valueType, Value uniquePtr) -> Value {
    return EmitCall(ctx, "::nvinfer1::adaptor::getRawPointer")
        .setResults({getPointerType(valueType)})
        .pushOperand(uniquePtr)
        .build(b, loc)
        .getResult(0);
  };

  // Create a logger.
  Value logger = EmitCall(ctx, "::nvinfer1::adaptor::createStdioLogger")
                     .setResults(getStdioLoggerType(ctx))
                     .build(b, loc)
                     .getResult(0);

  // Create a builder.
  Value builder = EmitCall(ctx, "::nvinfer1::adaptor::createBuilder")
                      .setResults(getAdaptorBuilderType(ctx))
                      .pushOperand(getRawPtr(
                          COpaqueType::get(ctx, "::nvinfer1::ILogger"), logger))
                      .build(b, loc)
                      .getResult(0);

  Value optimizationProfile =
      EmitCall(ctx, "::nvinfer1::adaptor::createOptimizationProfile")
          .setResults(getPointerType(getNvInferOptimizationProfileType(ctx)))
          .pushOperands({builder})
          .build(b, loc)
          .getResult(0);

  // Create a network definition.
  Value network = EmitCall(ctx, "::nvinfer1::adaptor::createNetworkV2")
                      .pushOperand(builder)
                      .setResults({getAdaptorNetworkType(ctx)})
                      .build(b, loc)
                      .getResult(0);
  COpaqueType networkDefType = getNvInferNetworkDefinitionType(ctx);
  Value weightsMap = EmitCall(ctx, "::nvinfer1::adaptor::createWeightsMap")
                         .setResults(getAdaptorWeightsMapType(ctx))
                         .build(b, loc)
                         .getResult(0);
  // Build network.
  std::string builderName = (op.getName() + "_builder").str();
  EmitCall(ctx, builderName)
      .pushOperands({getRawPtr(networkDefType, network), weightsMap})
      .build(b, loc);

  // Set shape profiles, if required.
  // Add the inputs to the builder.
  for (auto [idx, argType] : llvm::enumerate(op.getArgumentTypes())) {
    RankedTensorType type = dyn_cast<RankedTensorType>(argType);
    if (!type)
      return failure();

    FailureOr<ShapeProfileAttr> shapeProfile =
        getArgumentShapeProfile(dyn_cast<FunctionOpInterface>(*op), idx);
    if (failed(shapeProfile))
      return op->emitError() << "could not resolve shape profile "
                                "for arg "
                             << idx << "\n";

    // Emit a call to set shape profile dimensions.
    EmitCall(ctx,
             "::nvinfer1::adaptor::setOptimizationProfileArgumentShapeBounds")
        .pushOperands({network, optimizationProfile})
        .pushOperands<Attribute>(
            {b.getI32IntegerAttr(idx),
             getNvInferDimsOpaqueAttr(ctx, shapeProfile->getMin()),
             getNvInferDimsOpaqueAttr(ctx, shapeProfile->getOpt()),
             getNvInferDimsOpaqueAttr(ctx, shapeProfile->getMax())})
        .build(b, loc);
  }

  // Create config.
  Value builderConfig =
      EmitCall(ctx, "::nvinfer1::adaptor::createBuilderConfig")
          .setResults(getAdaptorBuilderConfigType(ctx))
          .pushOperands({getRawPtr(getNvInferBuilderType(ctx), builder)})
          .build(b, loc)
          .getResult(0);

  // Add the profile.
  EmitCall(ctx, "::nvinfer1::adaptor::"
                "setBuilderConfigOptimizationProfile")
      .pushOperands({getRawPtr(getNvInferBuilderConfigType(ctx), builderConfig),
                     optimizationProfile})
      .build(b, loc);

  // Compile and serialize.
  Value hostMemory =
      EmitCall(ctx, "::nvinfer1::adaptor::buildSerializedNetwork")
          .setResults(getAdaptorHostMemoryType(ctx))
          .pushOperands({builder, network, builderConfig})
          .build(b, loc)
          .getResult(0);

  b.create<func::ReturnOp>(op.getLoc(), hostMemory);
  moduleOp.getBodyRegion().front().push_back(builderFunc);
  return builderFunc;
}

static FailureOr<func::FuncOp> createEmitCBuilderOp(ModuleOp module,
                                                    FunctionOpInterface func) {
  // Create a builder function
  Location loc = func.getLoc();
  MLIRContext *ctx = module->getContext();
  std::string funcName = (func.getName() + "_builder").str();
  Type networkDefPtrType = getPointerType(getNvInferNetworkDefinitionType(ctx));
  auto builderFunc = func::FuncOp::create(
      loc, funcName,
      FunctionType::get(ctx,
                        TypeRange{networkDefPtrType, getWeightsMapType(ctx)},
                        TypeRange{}));
  Block *body = builderFunc.addEntryBlock();
  OpBuilder builder(body, body->begin());

  EmitCConverter converter(/*network=*/body->getArgument(0),
                           /*weightsMap=*/body->getArgument(1));
  if (failed(converter.encodeFunc(builder, func)))
    return failure();

  builder.create<func::ReturnOp>(loc);
  module.getBodyRegion().front().push_back(builderFunc);

  return builderFunc;
}

/// Returns true if the func has a single block in the body region and contains
/// only TensorRT dialect ops (besides the terminator).
static bool isTensorRTFuncOp(Operation &op) {
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    if (!func.getFunctionBody().hasOneBlock())
      return false;
    return llvm::all_of(
        func.getFunctionBody().getBlocks().front().without_terminator(),
        [](Operation &op) { return isa<tensorrt::TensorRTOpInterface>(op); });
  }
  return false;
}

// Add `emitc.include` operations to the top of the module if they are not
// already present.
static void addIncludesToModule(ModuleOp op) {
  assert(!op.getBodyRegion().getBlocks().empty() &&
         "expected non-empty region");
  if (!op.getOps<emitc::IncludeOp>().empty())
    return;
  OpBuilder b(op->getContext());
  b.setInsertionPointToStart(&op.getBodyRegion().front());
  llvm::StringRef howToUse =
      R"(//===----------------------------------------------------------------------===//
// This C++ file is generated by translating the `emitc` IR produced by the mlir-tensorrt
// `TensorRTToEmitC` pass.
//
// For each `func.func` operation in the input IR (named `x` and representing a TensorRT
// network graph), two corresponding functions are generated: `x_builder` and `x_tester`.
//
// The `x_builder` function constructs the `INetworkDefinition` from the IR.
//
// The `x_tester` function:
// 1. Calls `x_builder` with pointers to `INetworkDefinition` and a map for weight lookup.
// 2. Builds a TensorRT engine from the network definition
// 3. Returns a pointer to the serialized engine
//
// Usage:
// Implement a `main` function that calls one or more `x_tester` functions
// for your specific use case. Common use cases include:
// - Testing engine construction
// - Running inference
//
// Example:
// The following is a simple example that verifies successful engine construction.
// ```
// int main() {
//   auto engine = x_tester();
//   return engine ? 0 : 1;
// }
// ```
//
// Dependencies:
// Building the generated C++ code requires:
// 1. Headers from the `mlir-tensorrt` project:
//    - `NvInferAdaptor.h`
//    - `TensorRTVersion.h`
// 2. CUDA headers and libraries
// 3. TensorRT headers and libraries
//
// Build Instructions:
// Compile the generated C++ code using:
// ```
// g++ my_code.cpp -o my_code \
//     -I/path/to/dir/containing/NvInferAdaptor.h \
//     -I/usr/local/cuda/include \
//     -I/path/to/dir/containing/TensorRTVersion.h \
//     -I/path/to/dir/containing/TensorRT/include \
//     -lnvinfer \
//     -L/path/to/dir/containing/TensorRT/lib
// ```
//===----------------------------------------------------------------------===//
  )";
  b.create<emitc::VerbatimOp>(op.getLoc(), howToUse);
  b.create<emitc::IncludeOp>(op->getLoc(), "NvInfer.h", false);
  b.create<emitc::IncludeOp>(op->getLoc(), "NvInferAdaptor.h", false);
  b.create<emitc::IncludeOp>(op->getLoc(), "cstdint", true);
  b.create<emitc::IncludeOp>(op->getLoc(), "vector", true);
}

namespace {
class ConvertTensorRTToEmitCPass
    : public mlir::impl::ConvertTensorRTToEmitCPassBase<
          ConvertTensorRTToEmitCPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    SmallVector<Operation *> funcs;
    for (auto &op : llvm::make_early_inc_range(moduleOp.getOps())) {

      // For now delete non-tensorrt operations. This will
      // occur for e.g. `onnx.EntryPoint` operations that are
      // leftover.
      if (!isTensorRTFuncOp(op)) {
        op.erase();
        continue;
      }
      funcs.push_back(&op);
    }

    for (Operation *op : funcs) {
      // Emit a function of `emitc` operations that takes an
      // INetworkDefinition pointer and constructs the
      // equivalent TensorRT network represented by `op`.
      FailureOr<func::FuncOp> builderFunc =
          createEmitCBuilderOp(moduleOp, dyn_cast<func::FuncOp>(op));
      if (failed(builderFunc)) {
        emitError(op->getLoc()) << "failed to translate function "
                                   "into emitc builder";
        return signalPassFailure();
      }

      // Emit a function of `emitc` operations that actually
      // builds the network into a TensorRT engine. It also
      // sets shape profiles and configuration. It calls the
      // `builderFunc`.
      FailureOr<func::FuncOp> testerFunc =
          createEmitCTesterOp(moduleOp, dyn_cast<func::FuncOp>(op));
      if (failed(testerFunc)) {
        emitError(op->getLoc()) << "failed to translate function "
                                   "into emitc tester";
        return signalPassFailure();
      }

      // Add includes at the top of the module if they are not
      // already present.
      addIncludesToModule(moduleOp);

      // Erase the original function as it is not compatible
      // with `mlir-to-cpp` translation. After this point,
      // everything in the module can be translated to C++.
      op->erase();
    }
  }
};
} // namespace
