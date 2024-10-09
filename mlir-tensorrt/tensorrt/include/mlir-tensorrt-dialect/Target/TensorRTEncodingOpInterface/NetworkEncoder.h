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
// This file contains declarations for NvInferNetworkEncoder that traverses
// MLIR IR data-structures and emits appropriate calls to translate IR to a
// TensorRT network.
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_TARGET_TENSORRT_TENSORRTENCODINGOPINTERFACE_NETWORKENCODER
#define MLIR_TENSORRT_TARGET_TENSORRT_TENSORRTENCODINGOPINTERFACE_NETWORKENCODER

#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/Utils/NvInferAdaptor.h"
#include "mlir-tensorrt-dialect/Utils/NvInferPluginUtils.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringSet.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "NvInfer.h"

/// A RAII unique pointer for TensorRT plugins. PluginV2 is deprecated since
/// TensorRT 10, so until we upgrade to PluginV3, keep this in between the
/// pragmas to suppress the deprecation warning.
namespace mlir::tensorrt {
using NvInferPluginPtr =
    std::unique_ptr<nvinfer1::IPluginV2, void (*)(nvinfer1::IPluginV2 *)>;
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace mlir {

class FunctionOpInterface;

namespace tensorrt {
class NvInferNetworkEncoder;
}
} // namespace mlir

namespace mlir {
namespace tensorrt {
class TensorRTEncodingOpInterface;

/// An empty nvinfer1::Dims object.
static constexpr nvinfer1::Dims kNullDims = {0, {0}};
/// An empty nvinfer1::Weights object
static constexpr nvinfer1::Weights kNullWeights =
    nvinfer1::Weights{nvinfer1::DataType::kFLOAT, nullptr, 0};

//===----------------------------------------------------------------------===//
// NvInferNetworkEncode
//===----------------------------------------------------------------------===//

class NvInferNetworkEncoder {
public:
  NvInferNetworkEncoder(nvinfer1::INetworkDefinition *network,
                        nvinfer1::IOptimizationProfile *profile,
                        TensorRTVersion version, bool usesStronglyTyped)
      : network(network), profile(profile), version(std::move(version)),
        usesStronglyTyped(usesStronglyTyped) {}

  /// Lookup the TRT ITensor* equivalent of a Value.
  nvinfer1::ITensor *lookup(Value v) const;

  /// Lookup the TRT ITensor* equivalents of a ValueRange.
  SmallVector<nvinfer1::ITensor *> lookupValues(ValueRange values);

  /// Add a map from a Value to a TRT ITEnsor*.
  void map(Value from, nvinfer1::ITensor *to);

  /// Remap values in `from` to each layer in `to` using the output at index 0
  /// for each layer.
  void map(ValueRange from, ArrayRef<nvinfer1::ILayer *> to);

  /// Check whether the value map contains `v`.
  size_t contains(Value v) { return valueMap.count(v); }

  /// Get a Weights from an elements attr.
  FailureOr<nvinfer1::Weights> getNvInferWeights(ElementsAttr values);

  /// Get a Weights from an optional elements attr. If attr is not present,
  /// then return kNullWeights.
  FailureOr<nvinfer1::Weights>
  getNvInferWeights(std::optional<ElementsAttr> attr);

  /// For a given operation, try to add that operation to `network` and populate
  /// `valueMap` with its results. If `op` doesn't not represent a TensorRT
  /// dialect operation, then return failure.
  LogicalResult encodeOp(tensorrt::TensorRTEncodingOpInterface op);

  /// For a given block, try to add all ops to `network` and populate
  /// `valueMap` with its results. If `op` doesn't not represent a TensorRT
  /// dialect operation, then return failure.
  /// TODO: change this to non-recursive implementation.
  LogicalResult encodeBlock(Block &block);

  /// Encode a given region to a TensorRT engine.
  LogicalResult encodeRegion(Region &region);

  /// Encode a given function to a TensorRT engine.
  LogicalResult encodeFunc(FunctionOpInterface func);

  nvinfer1::INetworkDefinition *getNetworkDefinition() { return network; }

  /// A type that maps mlir::Value typed objects to the corresponding TensorRT
  /// ITensor object.
  using TensorMap = llvm::ScopedHashTable<mlir::Value, nvinfer1::ITensor *>;
  using TensorMapScope = TensorMap::ScopeTy;

  /// A type that tracks constant values which are created to back weights
  /// objects
  /// and other temporary buffers.
  using WeightsMap = llvm::DenseMap<mlir::Attribute, std::vector<int8_t>>;

  using NamesSet = llvm::StringSet<>;

  TensorMap &getTensorMap() { return valueMap; }

  /// Return the set that stores all currently used names for TensorRT ILayers.
  NamesSet &getNamesSet() { return namesSet; }

  /// Set the name of the `trtLayer` to a unique string that contains the op
  /// name and location information from `sourceOp`.
  void setName(nvinfer1::ILayer *layer, Operation *sourceOp);

  // Check if network uses fp16 types.
  bool hasFp16Usage() const { return usesFp16; }

  // Check if network uses int8 types.
  bool hasInt8Usage() const { return usesInt8; }

  // Check if network uses f8 types.
  bool hasFp8Usage() const { return usesFp8; }

  // Check if network has bf16 types.
  bool hasBf16Usage() const { return usesBf16; }

  // Check if network uses int4 types.
  bool hasInt4Usage() const { return usesInt4; }

  // Check if network uses strongly typed mode.
  bool isStronglyTyped() const;

  /// Ask the encoder to keep track of the given plugin so that it will be
  /// destroyed with the encoder goes out of scope.
  void insertPlugin(NvInferPluginPtr ptr) {
    pluginReferences.push_back(std::move(ptr));
  }

  /// Insert a ICastLayer to convert the input ITensor to the given dataType.
  nvinfer1::ITensor *insertCastLayer(nvinfer1::ITensor *input,
                                     nvinfer1::DataType dataType,
                                     Operation *sourceOp);

  /// Adds IDequantizeLayer to the network. This switches between different
  /// APIs depending on the compile-time TensorRT version and whether or not
  /// the strongly-typed flags is enabled.
  nvinfer1::ILayer *addDequantizeLayer(nvinfer1::ITensor *input,
                                       nvinfer1::ITensor *scale,
                                       nvinfer1::DataType outputType,
                                       std::optional<uint32_t> axis);

  /// Adds IFillLayer to the network. This switches between different
  /// APIs depending on the compile-time TensorRT version and whether or not
  /// the strongly-typed flags is enabled.
  nvinfer1::ILayer *addFillLayer(
      nvinfer1::DataType elementType, const nvinfer1::Dims &staticShape,
      nvinfer1::ITensor *dynamicShape, nvinfer1::FillOperation fillOperation,
      std::optional<double> alpha, std::optional<double> beta,
      nvinfer1::ITensor *dynamicAlpha, nvinfer1::ITensor *dynamicBeta);

  /// Adds a TensorRT plugin.
  FailureOr<nvinfer1::ILayer *>
  addOpaquePlugin(tensorrt::OpaquePluginOp op,
                  SmallVector<nvinfer1::ITensor *> &results);

  /// Return the TensorRT plugin manager.
  PluginManager &getPluginManager() { return pluginMgr; }

private:
  nvinfer1::INetworkDefinition *network;
  nvinfer1::IOptimizationProfile *profile;
  TensorMap valueMap;
  WeightsMap weightsMap;

  // Retains references to created plugins, which must be held until network
  // build ends.
  SmallVector<NvInferPluginPtr> pluginReferences;

  /// Holds the set of strings currently assigned as names to TensorRT ILayers.
  /// This is required because we must make new names unique. The TensorRT API
  /// does not have a set object to query names.
  NamesSet namesSet;

  /// Contains version information for the TensorRT library loaded at runtime.
  TensorRTVersion version;

  /// Whether or not this encoder observed an int8 tensor type being used.
  bool usesInt8{false};

  /// Whether or not this encoder observed an f16 tensor type being used.
  bool usesFp16{false};

  /// Whether or not this encoder observed an f8 tensor type being used.
  bool usesFp8{false};

  /// Whether or not this encoder observed an bf16 tensor type being used.
  bool usesBf16{false};

  /// Whether or not this encoder observed an int4 tensor type being used.
  bool usesInt4{false};

  /// Whether or not the encoder observed strongly typed mode usage.
  bool usesStronglyTyped{false};

  /// Whether the network has QDQ nodes. This determines whether we set dynamic
  /// ranges on i8 tensors.
  bool hasQDQOps{false};

  PluginManager pluginMgr;
};

//===----------------------------------------------------------------------===//
// Other helpers for TensorRT encoding
//===----------------------------------------------------------------------===//

/// Return the names of the tensors corresponding to the results of the encoded
/// function, in the order that they are returned from the MLIR function. This
/// assumes that NetworkEncoder was used to encode the function.
std::vector<std::string>
getResultTensorNames(unsigned numResults, const NvInferNetworkEncoder &encoder);

/// Convert an `ArrayRef` of integers into an `nvinfer1::Dims` object.
template <typename T, std::enable_if_t<std::is_same_v<T, int32_t> ||
                                           std::is_same_v<T, int64_t>,
                                       T *> = nullptr>
static nvinfer1::Dims getNvInferDims(ArrayRef<T> arrayRef) {
  assert(arrayRef.size() < nvinfer1::Dims::MAX_DIMS &&
         "input array exceeds max dims");
  nvinfer1::Dims dims;
  dims.nbDims = arrayRef.size();
  llvm::copy(llvm::map_range(arrayRef,
                             [](auto x) {
                               if (static_cast<int64_t>(x) ==
                                   ShapedType::kDynamic)
                                 return -1;
                               return static_cast<int32_t>(x);
                             }),
             dims.d);
  return dims;
}

/// Given a RankedTensorType, convert the shape of the type to an
/// `nvinfer1::Dims` object.
nvinfer1::Dims getNvInferDims(RankedTensorType t);

/// Convert an optional `ArrayRef` of integers into an `nvinfer1::Dims`
/// object if it has a value, otherwise return the "null dims" object.
template <typename T>
std::optional<nvinfer1::Dims>
getOptionalNvInferDims(std::optional<ArrayRef<T>> arrayRef) {
  if (!arrayRef.has_value())
    return std::nullopt;
  return getNvInferDims(*arrayRef);
}

/// Given an `ArrayRef<in64_t>`, return those values as an
/// `nvinfer1::Permutation`.
nvinfer1::Permutation getNvInferPermutation(ArrayRef<int64_t> array);

/// Given type `t` return the corresponding TensorRT::DataType enum value of the
/// elementType of `t` (if it is a `ShapedType`) or `t` itself.
/// If there is no corresponding `nvinfer1::DataType` enum value, this function
/// will cause the program to abort. This is meant to simplify the usage API
/// below in the `buildLayer` dispatch function, so types should be
/// appropriately verified before using.
FailureOr<nvinfer1::DataType> getNvInferDataType(Location loc, Type t);
Type getNvInferDataTypeAsMlirType(MLIRContext *ctx, nvinfer1::DataType t);

/// Convert an array of dimension indices into a bit mask. This is used below in
/// the generated `NetworkEncoder.inc.cpp`.
uint32_t getBitMaskFromDimensionList(ArrayRef<int64_t> dimensions);

//===----------------------------------------------------------------------===//
// NvInfer<->mlir enum converters.
//===----------------------------------------------------------------------===//

#define GEN_TRT_ENUM_CONVERTER_DECLS
#include "mlir-tensorrt-dialect/Target/TensorRTEncodingOpInterface/EnumConverters.inc.cpp"

/// Convert SliceMode to NvInfer enum.
std::optional<nvinfer1::SliceMode>
convertSliceModeToNvInferEnum(SliceMode value);

/// Convert ActivationType to NvInfer enum.
std::optional<nvinfer1::ActivationType>
convertActivationTypeToNvInferEnum(ActivationType value);

} // namespace tensorrt
} // namespace mlir

#endif // MLIR_TENSORRT_TARGET_TENSORRT_TENSORRTENCODINGOPINTERFACE_NETWORKENCODER
