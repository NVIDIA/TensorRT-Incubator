//===- NvInferAdaptors.h ----------------------------------------*- C++ -*-===//
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
// This function contains a set of adaptors to normalize the INetworkBuilder
// API from C++. These functions are meant to be included inline and be
// distributable to users.
//
// This is currently a WIP and many methods are essentially
// convenience/debugging tools for the C++ emission.
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_MLIR_TENSORRT_DIALECT_UTILS_NVINFERADAPTOR
#define INCLUDE_MLIR_TENSORRT_DIALECT_UTILS_NVINFERADAPTOR

//===----------------------------------------------------------------------===//
// Only TensorRT/CUDA and STL headers can be included here.
//===----------------------------------------------------------------------===//
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <NvInfer.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "TensorRTVersion.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace nvinfer1 {

// In TensorRT10, several enums will be removed or renamed. We insert this
// adaptor to avoid having IFDEF everywhere.
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
using SliceMode = nvinfer1::SampleMode;
using ResizeMode = nvinfer1::InterpolationMode;
#endif

namespace adaptor {
template <typename T>
T *getRawPointer(std::unique_ptr<T> &p) {
  return p.get();
}

//===----------------------------------------------------------------------===//
// A simple adaptor to represent "optional dims" passed by value.
//===----------------------------------------------------------------------===//
struct OptionalDims {
  OptionalDims(Dims d) : valid(true), value(d) {}
  OptionalDims() : valid(false), value(Dims{0, {}}) {}
  static OptionalDims None() { return OptionalDims(); }

  operator bool() const { return valid; }

  nvinfer1::Dims operator*() const {
    assert(valid && "accessing invalid optional value");
    return value;
  }

  operator Dims() const {
    assert(valid && "accessing invalid optional value");
    return value;
  }

  bool valid;
  Dims value;
};

//===----------------------------------------------------------------------===//
// Default Stderr Logger (using stdio)
//===----------------------------------------------------------------------===//

/// A simple logger that implements TensorRT's logging interface. Errors and
/// warnings are reported through TensorRT's diagnostic system, everything else
/// is printed to stderr if the verbose flag is present.
class StdioLogger : public nvinfer1::ILogger {
public:
  StdioLogger(bool verbose) : verbose(verbose) {}

protected:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      fprintf(stderr, "%s\n", msg);
      return;
    }
    if (severity == Severity::kWARNING) {
      fprintf(stderr, "%s\n", msg);
      return;
    }
    if (!verbose)
      return;
    fprintf(stderr, "%s\n", msg);
  }
  bool verbose;
};

inline std::unique_ptr<StdioLogger> createStdioLogger() {
  return std::make_unique<StdioLogger>(true);
}

//===----------------------------------------------------------------------===//
// IBuilder Adaptor
//===----------------------------------------------------------------------===//
inline std::unique_ptr<IBuilder> createBuilder(nvinfer1::ILogger *logger) {
  return std::unique_ptr<IBuilder>(createInferBuilder(*logger));
}

//===----------------------------------------------------------------------===//
// IBuilderConfig Adaptor
//===----------------------------------------------------------------------===//
inline std::unique_ptr<IBuilderConfig> createBuilderConfig(IBuilder *builder) {
  return std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
}

inline void
setBuilderConfigOptimizationProfile(IBuilderConfig *config,
                                    nvinfer1::IOptimizationProfile *profile) {
  config->addOptimizationProfile(profile);
}

//===----------------------------------------------------------------------===//
// IOptimizationProfile Adaptor
//===----------------------------------------------------------------------===//

inline nvinfer1::IOptimizationProfile *
createOptimizationProfile(std::unique_ptr<IBuilder> &builder) {
  return builder->createOptimizationProfile();
}

inline void setOptimizationProfileArgumentShapeBounds(
    std::unique_ptr<nvinfer1::INetworkDefinition> &network,
    IOptimizationProfile *profile, int32_t idx, const Dims &minShape,
    const Dims &optShape, const Dims &maxShape) {
  assert(idx < network->getNbInputs() &&
         "tried to set profile dimensions on an invalid argument number");
  const char *name = network->getInput(idx)->getName();
  profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, minShape);
  profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, optShape);
  profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, maxShape);
}

//===----------------------------------------------------------------------===//
// Constant Management Convenience Functions
//===----------------------------------------------------------------------===//

// These functions are meant to assist with constant data management. The ABI is
// not set in stone, the is the simplest possible mechanism to unblock initial
// development on the C++ emission.

/// Right now we make the assumption that weights are stored in a map
/// of this type, but we can relax this later with
using WeightsMap = std::unordered_map<const char *, std::vector<uint8_t>>;

inline WeightsMap createWeightsMap() { return WeightsMap(); }

/// Store the weights `w` in the `weightsMap` using `name` as the key.
/// This is meant to be used for small constants that are passed as an
/// initializer list.
template <typename T>
Weights trtSetWeights(WeightsMap &weightsMap, const char *name,
                      const std::vector<T> &w) {
  weightsMap[name] = std::vector<uint8_t>(w.size() * sizeof(T));
  std::vector<uint8_t> &data = weightsMap[name];
  std::memcpy(data.data(), w.data(), w.size());
  DataType dt = DataType::kFLOAT;
  if (std::is_same<T, float>::value) {
    dt = DataType::kFLOAT;
  } else if (std::is_same<T, int32_t>::value) {
    dt = DataType::kINT32;
  }
  return Weights{dt, data.data(), static_cast<int64_t>(w.size())};
}

/// Store the weights `w` in the `weightsMap` using `name` as the key.
/// This is meant to be used for small constants that are passed as an
/// initializer list.
template <typename T>
Weights trtSetWeightsSplat(WeightsMap &weightsMap, const char *name,
                           int64_t count, T splatValue) {
  weightsMap[name] = std::vector<uint8_t>(count * sizeof(T));
  std::vector<uint8_t> &data = weightsMap[name];
  std::fill_n(reinterpret_cast<T *>(data.data()), count, splatValue);
  DataType dt = DataType::kFLOAT;
  if (std::is_same<T, float>::value) {
    dt = DataType::kFLOAT;
  } else if (std::is_same<T, int32_t>::value) {
    dt = DataType::kINT32;
  }
  return Weights{dt, data.data(), count};
}

//===----------------------------------------------------------------------===//
// INetworkDefinition Adaptor
//===----------------------------------------------------------------------===//

inline std::unique_ptr<INetworkDefinition>
createNetworkV2(std::unique_ptr<IBuilder> &builder, uint32_t flags = 0) {
  // The ExplicitBatch flag will be removed in TRT10 since implicit batch mode
  // will be removed.
#if !MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  flags |= 1U << static_cast<uint32_t>(
               nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
  return std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));
}

/// Returns true if the INetworkDefinition was created with the `STRONGLY_TYPED`
/// creation flag. This only can occur for TensorRT 9.1+.
static inline bool
isStronglyTypedFlagEnabled(const nvinfer1::INetworkDefinition *network) {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(9, 1, 0)
  return network->getFlag(
      nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#else
  return false;
#endif
}

inline std::unique_ptr<nvinfer1::IHostMemory>
buildSerializedNetwork(std::unique_ptr<IBuilder> &builder,
                       std::unique_ptr<INetworkDefinition> &network,
                       std::unique_ptr<IBuilderConfig> &config) {
  return std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
}

/// Adaptor for `addInput`.
inline ITensor *networkAddInput(INetworkDefinition *n, const char *name,
                                DataType t, Dims d) {
  return n->addInput(name, t, d);
}

/// Adaptor for `network->markOutput`.
inline void networkMarkOutput(INetworkDefinition *n,
                              nvinfer1::ITensor *tensor) {
  n->markOutput(*tensor);
}

/// Adaptor for `addConstant`.
inline ITensor *networkAddConstant(INetworkDefinition *n, Dims d, Weights w) {
  return n->addConstant(d, w)->getOutput(0);
}

/// Adaptor for `addSlice`.
inline ITensor *networkAddSlice(INetworkDefinition *n, ITensor *input,
                                ITensor *start, ITensor *size, ITensor *stride,
                                ITensor *fill, Dims startStatic,
                                Dims sizeStatic, Dims strideStatic,
                                SliceMode mode) {
  ISliceLayer *layer =
      n->addSlice(*input, startStatic, sizeStatic, strideStatic);
  layer->setMode(mode);
  if (!start) {
    layer->setStart(startStatic);
  } else {
    layer->setInput(1, *start);
  }
  if (size) {
    layer->setInput(2, *size);
  }
  if (stride) {
    layer->setInput(3, *stride);
  }
  if (fill) {
    layer->setInput(4, *fill);
  }
  return layer->getOutput(0);
}

/// Adaptor for `addElementWise`.
inline ITensor *networkAddElementWise(INetworkDefinition *n, ITensor *input1,
                                      ITensor *input2,
                                      ElementWiseOperation operationType) {
  // The enum is designed to map 1-1 so we don't need conversion.
  IElementWiseLayer *layer = n->addElementWise(*input1, *input2, operationType);
  return layer->getOutput(0);
}

/// Adaptor for `addIfConditional`
inline IIfConditional *addIfConditional(INetworkDefinition *n,
                                        ITensor *condition) {
  IIfConditional *conditional = n->addIfConditional();
  conditional->setCondition(*condition);
  return conditional;
}

/// Adaptor for `IIfConditional::addInput`.
inline ITensor *addIfConditionalInput(IIfConditional *conditional,
                                      ITensor *input) {
  IIfConditionalInputLayer *layer = conditional->addInput(*input);
  return layer->getOutput(0);
}

/// Adaptor for `IIfConditional::addOutput`.
inline ITensor *addIfConditionalOutput(IIfConditional *conditional,
                                       ITensor *trueBranch,
                                       ITensor *falseBranch) {
  IIfConditionalOutputLayer *layer =
      conditional->addOutput(*trueBranch, *falseBranch);
  return layer->getOutput(0);
}

/// Adaptor for `addShuffle` with a dynamic reshape argument.
inline ITensor *networkAddShuffle(INetworkDefinition *n, ITensor *input,
                                  ITensor *dynamicReshapeShape,
                                  Permutation transpose1,
                                  OptionalDims staticReshapeShape,
                                  Permutation transpose2,
                                  bool zeroIsPlaceholder) {
  IShuffleLayer *layer = n->addShuffle(*input);
  layer->setZeroIsPlaceholder(zeroIsPlaceholder);
  layer->setFirstTranspose(transpose1);
  if (dynamicReshapeShape != nullptr) {
    layer->setInput(1, *dynamicReshapeShape);
  } else if (staticReshapeShape) {
    layer->setReshapeDimensions(*staticReshapeShape);
  }
  layer->setSecondTranspose(transpose2);
  return layer->getOutput(0);
}

/// Adaptor for `addPooling`.
inline ITensor *networkAddPooling(INetworkDefinition *n, ITensor *input,
                                  PoolingType poolingType, Dims postPadding,
                                  Dims prePadding, Dims stride, Dims windowSize,
                                  const bool *averageCountExcludesPadding,
                                  const float *blendFactor) {
  nvinfer1::IPoolingLayer *layer = n->addPoolingNd(
      /*input=*/*input,
      /*type=*/poolingType,
      /*windowSize=*/windowSize);
  layer->setPrePadding(prePadding);
  layer->setPostPadding(postPadding);
  layer->setStrideNd(stride);
  if (averageCountExcludesPadding) {
    layer->setAverageCountExcludesPadding(*averageCountExcludesPadding);
  }
  if (blendFactor) {
    layer->setBlendFactor(*blendFactor);
  }
  return layer->getOutput(0);
}

/// Adaptor for `addPooling`.
inline ITensor *networkAddPooling(INetworkDefinition *n, ITensor *input,
                                  PoolingType poolingType, Dims postPadding,
                                  Dims prePadding, Dims stride,
                                  Dims windowSize) {
  assert(poolingType == nvinfer1::PoolingType::kMAX);
  nvinfer1::IPoolingLayer *layer = n->addPoolingNd(
      /*input=*/*input,
      /*type=*/poolingType,
      /*windowSize=*/windowSize);
  layer->setPrePadding(prePadding);
  layer->setPostPadding(postPadding);
  layer->setStrideNd(stride);
  return layer->getOutput(0);
}

/// Adaptor for `addMatrixMultiply`.
inline ITensor *networkAddMatrixMultiply(INetworkDefinition *n, ITensor *input0,
                                         ITensor *input1, MatrixOperation op0,
                                         MatrixOperation op1) {
  nvinfer1::IMatrixMultiplyLayer *layer = n->addMatrixMultiply(
      *input0, static_cast<nvinfer1::MatrixOperation>(op0), *input1,
      static_cast<nvinfer1::MatrixOperation>(op1));
  return layer->getOutput(0);
}

/// Adaptor for `addActivation`.
inline ITensor *networkAddActivation(INetworkDefinition *n, ITensor *input,
                                     nvinfer1::ActivationType type, float alpha,
                                     float beta) {
  nvinfer1::IActivationLayer *layer = n->addActivation(*input, type);
  if (alpha != 0.0f) {
    layer->setAlpha(alpha);
  }
  if (beta != 0.0f) {
    layer->setBeta(beta);
  }
  return layer->getOutput(0);
}

/// Adaptor for `addReduce`.
inline ITensor *networkAddReduce(INetworkDefinition *n, ITensor *input,
                                 bool keepdims, uint32_t reduceAxes,
                                 nvinfer1::ReduceOperation reduceOperation) {
  nvinfer1::IReduceLayer *layer =
      n->addReduce(*input, reduceOperation, reduceAxes, keepdims);
  return layer->getOutput(0);
}

/// Adaptor for `addConvolution`.
inline ITensor *networkAddConvolution(
    INetworkDefinition *n, ITensor *input, ITensor *kernel, ITensor *bias,
    nvinfer1::Weights kernelWeights, nvinfer1::Weights biasWeights,
    int32_t numOutputMaps, const nvinfer1::Dims &kernelSpatialShape,
    const nvinfer1::Dims &stride, const nvinfer1::Dims &prePadding,
    const nvinfer1::Dims &postPadding, int32_t groups,
    const OptionalDims dilation) {
  nvinfer1::IConvolutionLayer *layer = n->addConvolutionNd(
      *input, numOutputMaps, kernelSpatialShape, kernelWeights, biasWeights);
  if (kernel != nullptr)
    layer->setInput(1, *kernel);
  if (bias != nullptr)
    layer->setInput(2, *bias);
  layer->setStrideNd(stride);
  layer->setPrePadding(prePadding);
  layer->setPostPadding(postPadding);
  if (dilation)
    layer->setDilationNd(*dilation);
  layer->setNbGroups(groups);
  return layer->getOutput(0);
}

/// Adaptor for `addShape`.
inline ITensor *networkAddShape(INetworkDefinition *n, ITensor *input) {
  IShapeLayer *layer = n->addShape(*input);
  return layer->getOutput(0);
}

/// Adaptor for `addTopK`
inline std::tuple<ITensor *, ITensor *>
networkAddTopK(INetworkDefinition *n, ITensor *input, int32_t k,
               uint32_t reduceAxes, TopKOperation op) {
  ITopKLayer *layer = n->addTopK(*input, op, k, reduceAxes);
  return {layer->getOutput(0), layer->getOutput(1)};
}

/// Adaptor for `addSoftmax`
inline ITensor *networkAddSoftmax(INetworkDefinition *n, ITensor *input,
                                  uint32_t axis) {
  nvinfer1::ISoftMaxLayer *layer = n->addSoftMax(*input);
  layer->setAxes(axis);
  return layer->getOutput(0);
}

/// Adaptor for `addUnary`
inline ITensor *networkAddUnary(INetworkDefinition *n, ITensor *input,
                                nvinfer1::UnaryOperation operation) {
  nvinfer1::IUnaryLayer *layer = n->addUnary(*input, operation);
  return layer->getOutput(0);
}

/// Adaptor for `addSelect`
inline ITensor *networkAddSelect(INetworkDefinition *n, ITensor *condition,
                                 ITensor *thenBranch, ITensor *elseBranch) {
  ISelectLayer *layer = n->addSelect(*condition, *thenBranch, *elseBranch);
  return layer->getOutput(0);
}

/// Adaptor for `addConcatenation`
template <typename... Args>
inline ITensor *networkAddConcatenation(INetworkDefinition *n, int32_t axis,
                                        Args... args) {
  static_assert((std::is_same_v<Args, ITensor *> && ...),
                "All operands except axis must be ITensor*");
  std::vector<ITensor *> concatInputs{args...};
  IConcatenationLayer *layer =
      n->addConcatenation(concatInputs.data(), concatInputs.size());
  layer->setAxis(axis);
  return layer->getOutput(0);
}

/// Adaptor for `addIdentity`.
inline ITensor *networkAddIdentity(INetworkDefinition *n, ITensor *input,
                                   nvinfer1::DataType targetType) {
  IIdentityLayer *layer = n->addIdentity(*input);
  layer->setOutputType(0, targetType);
  return layer->getOutput(0);
}

/// Adaptor for `addGatherElements`
inline ITensor *networkAddGatherElements(INetworkDefinition *n, ITensor *input,
                                         ITensor *indices, int32_t axis) {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  IGatherLayer *layer =
      n->addGatherV2(*input, *indices, nvinfer1::GatherMode::kELEMENT);
  layer->setGatherAxis(axis);
#else
  IGatherLayer *layer = n->addGather(*input, *indices, axis);
  layer->setMode(nvinfer1::GatherMode::kELEMENT);
#endif
  return layer->getOutput(0);
}

/// Adaptor for `addGatherNd`
inline ITensor *networkAddGatherNd(INetworkDefinition *n, ITensor *input,
                                   ITensor *indices) {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  IGatherLayer *layer =
      n->addGatherV2(*input, *indices, nvinfer1::GatherMode::kND);
#else
  IGatherLayer *layer = n->addGather(*input, *indices, 0);
  layer->setMode(nvinfer1::GatherMode::kND);
#endif
  return layer->getOutput(0);
}

} // namespace adaptor
} // namespace nvinfer1

#endif // INCLUDE_MLIR_TENSORRT_DIALECT_UTILS_NVINFERADAPTOR
