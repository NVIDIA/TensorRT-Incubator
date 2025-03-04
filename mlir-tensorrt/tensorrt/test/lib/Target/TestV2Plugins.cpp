//===- TestPlugins.cpp  ---------------------------------------------------===//
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
/// This file contains dummy TensorRT plugins using the V2 interface for use
/// with testing the `tensorrt.opaque_plugin` translation.
///
//===----------------------------------------------------------------------===//

#include "NvInferRuntime.h"
#include "NvInferRuntimePlugin.h"
#include "PluginUtils.h"
#include "mlir-tensorrt-dialect/Utils/TensorRTVersion.h"
#include "tensorrttestplugins_export.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace {

struct PluginParams {
  int32_t i32Param;
  int16_t i16Param;
  int8_t i8Param;
  Dims shapeParam;
  const int64_t *i64DenseParam;
  const int32_t *i32DenseParam;
  const int32_t *i16DenseParam;
  const int32_t *i8DenseParam;
  const int64_t *i64SplatParam;
  const int32_t *i32SplatParam;
  const int32_t *i16SplatParam;
  const int32_t *i8SplatParam;
};

class TestV2PluginBase : public nvinfer1::IPluginV2DynamicExt {
public:
  TestV2PluginBase() = default;
  TestV2PluginBase(const TestV2PluginBase &) = default;

  TestV2PluginBase(const PluginParams &params_) : params(params_) {}

  size_t getSerializationSize() const noexcept override { return 0; }

  void serialize(void *buffer) const noexcept override {}

  char const *getPluginVersion() const noexcept override { return "0"; }

  char const *getPluginNamespace() const noexcept override {
    return _namespace.c_str();
  }

  int32_t getNbOutputs() const noexcept override { return 1; }

  void configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs,
                       DynamicPluginTensorDesc const *out,
                       int32_t nbOutputs) noexcept override {}

  int32_t enqueue(PluginTensorDesc const *inputDesc,
                  PluginTensorDesc const *outputDesc, void const *const *inputs,
                  void *const *outputs, void *workspace,
                  cudaStream_t stream) noexcept override {
    return 0;
  }

  bool supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    PluginTensorDesc const &desc = inOut[pos];
    return desc.format == PluginFormat::kLINEAR &&
           desc.type == DataType::kFLOAT;
  }

  void setPluginNamespace(char const *libNamespace) noexcept override {
    _namespace = libNamespace;
  }

  int32_t initialize() noexcept override { return 0; }
  void terminate() noexcept override {}
  void destroy() noexcept override {}
  size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const *inputs,
                          int32_t nbInputs,
                          nvinfer1::PluginTensorDesc const *outputs,
                          int32_t nbOutputs) const noexcept override {
    return 0;
  }

protected:
  std::string _namespace;
  PluginParams params;
};

class TestV2Plugin1 : public TestV2PluginBase {
public:
  TestV2Plugin1() = default;

  using TestV2PluginBase::TestV2PluginBase;

  char const *getPluginType() const noexcept override {
    return "TestV2Plugin1";
  }

  nvinfer1::DataType
  getOutputDataType(int32_t index, nvinfer1::DataType const *inputType,
                    int32_t nbInputs) const noexcept override {
    return DataType::kFLOAT;
  }

  nvinfer1::DimsExprs
  getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const *inputs,
                      int32_t nbInputs,
                      nvinfer1::IExprBuilder &exprBuilder) noexcept override {
    nvinfer1::DimsExprs outDims;
    outDims.nbDims = 1;
    outDims.d[0] = exprBuilder.constant(1);
    return outDims;
  }

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override {
    return new TestV2Plugin1(*this);
  }
};

class TestV2Plugin1Creator : public nvinfer1::IPluginCreator {
public:
  using PluginType = TestV2Plugin1;

  TestV2Plugin1Creator() {
    fields.push_back(
        PluginField{"i32_param", nullptr, PluginFieldType::kINT32});
    fields.push_back(
        PluginField{"i16_param", nullptr, PluginFieldType::kINT16});
    fields.push_back(PluginField{"i8_param", nullptr, PluginFieldType::kINT8});
    fields.push_back(
        PluginField{"shape_param", nullptr, PluginFieldType::kDIMS});
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
    fields.push_back(
        PluginField{"i64_dense_param", nullptr, PluginFieldType::kINT64});
#endif
    fields.push_back(
        PluginField{"i32_dense_param", nullptr, PluginFieldType::kINT32});
    fields.push_back(
        PluginField{"i16_dense_param", nullptr, PluginFieldType::kINT16});
    fields.push_back(
        PluginField{"i8_dense_param", nullptr, PluginFieldType::kINT8});
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
    fields.push_back(
        PluginField{"i64_splat_param", nullptr, PluginFieldType::kINT64});
#endif
    fields.push_back(
        PluginField{"i32_splat_param", nullptr, PluginFieldType::kINT32});
    fields.push_back(
        PluginField{"i16_splat_param", nullptr, PluginFieldType::kINT16});
    fields.push_back(
        PluginField{"i8_splat_param", nullptr, PluginFieldType::kINT8});
    fields.push_back(PluginField{kPLUGIN_FAILURE_TRIGGER_FIELD_NAME, nullptr,
                                 PluginFieldType::kINT32});

    collection.nbFields = fields.size();
    collection.fields = fields.data();
  }

  char const *getPluginName() const noexcept override {
    return "TestV2Plugin1";
  }

  char const *getPluginVersion() const noexcept override { return "0"; }

  nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override {
    return &collection;
  }

  nvinfer1::IPluginV2DynamicExt *
  createPlugin(const char *name,
               const nvinfer1::PluginFieldCollection *fc) noexcept override {

    if (creatorShouldFail(fc))
      return nullptr;

    PluginParams pluginParams;

    std::cout << "Created TestV2Plugin1 with " << fc->nbFields << " fields:\n";
    for (int32_t i = 0; i < fc->nbFields; ++i) {
      auto const &field = fc->fields[i];
      std::cout << "field[" << i << "] ";
      printField(std::cout, field);
      std::string attrName = field.name;

      if (attrName == "i32_param")
        pluginParams.i32Param =
            *static_cast<const decltype(pluginParams.i32Param) *>(field.data);
      else if (attrName == "i16_param")
        pluginParams.i16Param =
            *static_cast<const decltype(pluginParams.i16Param) *>(field.data);
      else if (attrName == "i8_param")
        pluginParams.i8Param =
            *static_cast<const decltype(pluginParams.i8Param) *>(field.data);
      else if (attrName == "shape_param")
        pluginParams.shapeParam =
            *static_cast<const decltype(pluginParams.shapeParam) *>(field.data);
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
      else if (attrName == "i64_dense_param")
        pluginParams.i64DenseParam =
            static_cast<const decltype(pluginParams.i64DenseParam)>(field.data);
#endif
      else if (attrName == "i32_dense_param")
        pluginParams.i32DenseParam =
            static_cast<const decltype(pluginParams.i32DenseParam)>(field.data);
      else if (attrName == "i16_dense_param")
        pluginParams.i16DenseParam =
            static_cast<const decltype(pluginParams.i16DenseParam)>(field.data);
      else if (attrName == "i8_dense_param")
        pluginParams.i8DenseParam =
            static_cast<const decltype(pluginParams.i8DenseParam)>(field.data);
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
      else if (attrName == "i64_splat_param")
        pluginParams.i64SplatParam =
            static_cast<const decltype(pluginParams.i64SplatParam)>(field.data);
#endif
      else if (attrName == "i32_splat_param")
        pluginParams.i32SplatParam =
            static_cast<const decltype(pluginParams.i32SplatParam)>(field.data);
      else if (attrName == "i16_splat_param")
        pluginParams.i16SplatParam =
            static_cast<const decltype(pluginParams.i16SplatParam)>(field.data);
      else if (attrName == "i8_splat_param")
        pluginParams.i8SplatParam =
            static_cast<const decltype(pluginParams.i8SplatParam)>(field.data);
    }

    return new PluginType(pluginParams);
  }

  void setPluginNamespace(const char *) noexcept override {}
  char const *getPluginNamespace() const noexcept override { return ""; }

  IPluginV2DynamicExt *deserializePlugin(char const *name, void const *data,
                                         size_t length) noexcept override {
    assert(false && "This should never be called!");
    return nullptr;
  }

protected:
  nvinfer1::PluginFieldCollection collection;
  std::vector<nvinfer1::PluginField> fields;
};

class TestV2InferShapePlugin : public TestV2PluginBase {
public:
  TestV2InferShapePlugin() = default;
  TestV2InferShapePlugin(const TestV2InferShapePlugin &) = default;
  TestV2InferShapePlugin(const PluginFieldCollection &data) {}

  char const *getPluginType() const noexcept override {
    return "TestV2InferShapePlugin";
  }

  nvinfer1::DataType
  getOutputDataType(int32_t index, nvinfer1::DataType const *inputType,
                    int32_t nbInputs) const noexcept override {
    return DataType::kFLOAT;
  }

  nvinfer1::DimsExprs
  getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const *inputs,
                      int32_t nbInputs,
                      nvinfer1::IExprBuilder &exprBuilder) noexcept override {

    nvinfer1::DimsExprs outDims;
    outDims.nbDims = 4;
    outDims.d[0] = exprBuilder.constant(42);
    outDims.d[1] = exprBuilder.operation(DimensionOperation::kSUM,
                                         *inputs[0].d[0], *inputs[0].d[1]);
    outDims.d[2] = exprBuilder.operation(DimensionOperation::kMAX,
                                         *inputs[0].d[1], *inputs[0].d[2]);
    outDims.d[3] =
        exprBuilder.operation(DimensionOperation::kCEIL_DIV, *inputs[0].d[3],
                              *exprBuilder.constant(3));

    return outDims;
  }

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override {
    return new TestV2InferShapePlugin(*this);
  }
};

class TestV2InferShapePluginCreator : public nvinfer1::IPluginCreator {
public:
  using PluginType = TestV2InferShapePlugin;

  TestV2InferShapePluginCreator() {}

  char const *getPluginName() const noexcept override {
    return "TestV2InferShapePlugin";
  }
  char const *getPluginVersion() const noexcept override { return "0"; }

  nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override {
    return &collection;
  }

  nvinfer1::IPluginV2DynamicExt *
  createPlugin(const char *name,
               const nvinfer1::PluginFieldCollection *fc) noexcept override {
    return new PluginType(*fc);
  }

  void setPluginNamespace(const char *) noexcept override {}
  char const *getPluginNamespace() const noexcept override { return ""; }

  IPluginV2DynamicExt *deserializePlugin(char const *name, void const *data,
                                         size_t length) noexcept override {
    assert(false && "This should never be called!");
    return nullptr;
  }

protected:
  nvinfer1::PluginFieldCollection collection{0, nullptr};
};
} // namespace

// Do the static registration of the creator.
REGISTER_TENSORRT_PLUGIN(TestV2Plugin1Creator);
REGISTER_TENSORRT_PLUGIN(TestV2InferShapePluginCreator);

extern "C" {
/// Provide an exported C-style function for creating the plugin creator.
TENSORRTTESTPLUGINS_EXPORT
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
nvinfer1::IPluginCreatorInterface *getTestV2Plugin1Creator() {
  return std::make_unique<TestV2Plugin1Creator>().release();
}
#else
nvinfer1::IPluginCreator *getTestV2Plugin1Creator() {
  return std::make_unique<TestV2Plugin1Creator>().release();
}
#endif
}
