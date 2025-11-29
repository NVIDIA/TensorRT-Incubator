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
/// This file contains dummy TensorRT plugins for use with testing the
/// `tensorrt.opaque_plugin` translation.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Utils/TensorRTVersion.h"
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)

#include "NvInferRuntime.h"
#include "NvInferRuntimePlugin.h"
#include "PluginUtils.h"
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
#include "nvinfer/trt_plugin_python.h"
#endif // MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
#include "tensorrttestplugins_export.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {
class TestPluginBase : public nvinfer1::IPluginV3,
                       public nvinfer1::IPluginV3OneCore,
                       public nvinfer1::IPluginV3OneBuild,
                       public nvinfer1::IPluginV3OneRuntime {
public:
  TestPluginBase() = default;
  TestPluginBase(TestPluginBase const &p) = default;

  char const *getPluginName() const noexcept override { return "TestPlugin1"; }

  char const *getPluginVersion() const noexcept override { return "0"; }

  char const *getPluginNamespace() const noexcept override {
    return _namespace.c_str();
  }

  int32_t getNbOutputs() const noexcept override { return 1; }

  int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in,
                          int32_t nbInputs,
                          nvinfer1::DynamicPluginTensorDesc const *out,
                          int32_t nbOutputs) noexcept override {
    return 0;
  }

  int32_t enqueue(nvinfer1::PluginTensorDesc const *inputDesc,
                  nvinfer1::PluginTensorDesc const *outputDesc,
                  void const *const *inputs, void *const *outputs,
                  void *workspace, cudaStream_t stream) noexcept override {
    return 0;
  }

  int32_t onShapeChange(nvinfer1::PluginTensorDesc const *in, int32_t nbInputs,
                        nvinfer1::PluginTensorDesc const *out,
                        int32_t nbOutputs) noexcept override {
    return 0;
  }

  nvinfer1::PluginFieldCollection const *
  getFieldsToSerialize() noexcept override {
    return &collection;
  }

  nvinfer1::IPluginV3 *
  attachToContext(nvinfer1::IPluginResourceContext *context) noexcept override {
    return clone();
  }

  bool supportsFormatCombination(int32_t pos,
                                 DynamicPluginTensorDesc const *inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    PluginTensorDesc const &desc = inOut[pos].desc;
    return desc.format == PluginFormat::kLINEAR &&
           desc.type == DataType::kFLOAT;
  }

protected:
  std::string _namespace;
  std::vector<nvinfer1::PluginField> fields;
  nvinfer1::PluginFieldCollection collection;
};

class TestPlugin1 : public TestPluginBase {
public:
  IPluginCapability *
  getCapabilityInterface(PluginCapabilityType type) noexcept override {
    if (type == PluginCapabilityType::kBUILD)
      return static_cast<IPluginV3OneBuild *>(this);

    if (type == PluginCapabilityType::kRUNTIME)
      return static_cast<IPluginV3OneRuntime *>(this);

    assert(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore *>(this);
  }

  TestPlugin1() = default;
  TestPlugin1(const PluginFieldCollection &data) {
    std::cout << "Created TestPlugin1 with " << data.nbFields << " fields:\n";
    for (unsigned i = 0, e = data.nbFields; i < e; i++) {
      PluginField field = data.fields[i];
      std::cout << "field[" << i << "] ";
      printField(std::cout, field);
    }
  }

  int32_t getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs,
                             DataType const *inputTypes,
                             int32_t nbInputs) const noexcept override {
    outputTypes[0] = DataType::kFLOAT;
    return 0;
  }

  int32_t getOutputShapes(DimsExprs const *inputs, int32_t nbInputs,
                          DimsExprs const *shapeInputs, int32_t nbShapeInputs,
                          DimsExprs *outputs, int32_t nbOutputs,
                          IExprBuilder &exprBuilder) noexcept override {
    outputs[0].nbDims = 1;
    outputs[0].d[0] = exprBuilder.constant(1);
    return 0;
  }

  IPluginV3 *clone() noexcept override { return new TestPlugin1(); }
};

/// The purpose of this plugin is purely to test the ability of the
/// `tensorrt-infer-plugin-shapes` pass to hijack the plugin machinery to build
/// MLIR scalar IR by calling the plugin's `getOutputShapes` function.
class TestInferShapePlugin : public TestPluginBase {
public:
  IPluginCapability *
  getCapabilityInterface(PluginCapabilityType type) noexcept override {
    if (type == PluginCapabilityType::kBUILD)
      return static_cast<IPluginV3OneBuild *>(this);
    if (type == PluginCapabilityType::kRUNTIME)
      return static_cast<IPluginV3OneRuntime *>(this);
    assert(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore *>(this);
  }

  TestInferShapePlugin() = default;
  TestInferShapePlugin(const PluginFieldCollection &data) {}

  int32_t getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs,
                             DataType const *inputTypes,
                             int32_t nbInputs) const noexcept override {
    outputTypes[0] = DataType::kFLOAT;
    return 0;
  }

  int32_t getOutputShapes(DimsExprs const *inputs, int32_t nbInputs,
                          DimsExprs const *shapeInputs, int32_t nbShapeInputs,
                          DimsExprs *outputs, int32_t nbOutputs,
                          IExprBuilder &exprBuilder) noexcept override {
    if (nbInputs != 1 || inputs[0].nbDims != 4)
      return 1;
    if (nbOutputs != 1)
      return 1;

    std::cout << "Input DimExprs Constant Values:" << std::endl;
    for (int32_t i = 0; i < inputs[0].nbDims; ++i)
      std::cout << "Dimension " << i
                << " value: " << inputs[0].d[i]->getConstantValue()
                << std::endl;

    outputs[0].nbDims = 4;
    outputs[0].d[0] = exprBuilder.constant(42);
    outputs[0].d[1] = exprBuilder.operation(DimensionOperation::kSUM,
                                            *inputs[0].d[0], *inputs[0].d[1]);
    outputs[0].d[2] = exprBuilder.operation(DimensionOperation::kMAX,
                                            *inputs[0].d[1], *inputs[0].d[2]);
    outputs[0].d[3] =
        exprBuilder.operation(DimensionOperation::kCEIL_DIV, *inputs[0].d[3],
                              *exprBuilder.constant(3));

    return 0;
  }

  IPluginV3 *clone() noexcept override { return new TestInferShapePlugin(); }
};

/// The fields for this plugin creator correspond to the fields given in
/// `test_opaque_plugin_field_serialization` in
//// `test/Target/TensorRT/OpaquePlugin/opaque-plugin.mlir`.
class TestPlugin1Creator : public nvinfer1::IPluginCreatorV3One {
public:
  using PluginType = TestPlugin1;

  TestPlugin1Creator() {
    fields.push_back(
        PluginField{"i64_param", nullptr, PluginFieldType::kINT64});
    fields.push_back(
        PluginField{"i32_param", nullptr, PluginFieldType::kINT32});
    fields.push_back(
        PluginField{"i16_param", nullptr, PluginFieldType::kINT16});
    fields.push_back(PluginField{"i8_param", nullptr, PluginFieldType::kINT8});
    fields.push_back(
        PluginField{"f32_param", nullptr, PluginFieldType::kFLOAT32});
    fields.push_back(
        PluginField{"f64_param", nullptr, PluginFieldType::kFLOAT64});
    fields.push_back(
        PluginField{"string_param", nullptr, PluginFieldType::kCHAR});
    fields.push_back(
        PluginField{"shape_param", nullptr, PluginFieldType::kDIMS});
    fields.push_back(
        PluginField{"shape_vec_param", nullptr, PluginFieldType::kDIMS});
    fields.push_back(
        PluginField{"f32_elements_param", nullptr, PluginFieldType::kFLOAT32});
    fields.push_back(
        PluginField{"f16_elements_param", nullptr, PluginFieldType::kFLOAT16});
    fields.push_back(PluginField{kPLUGIN_FAILURE_TRIGGER_FIELD_NAME, nullptr,
                                 PluginFieldType::kINT32});

    collection.nbFields = fields.size();
    collection.fields = fields.data();
  }

  char const *getPluginName() const noexcept override { return "TestPlugin1"; }

  char const *getPluginVersion() const noexcept override { return "0"; }

  nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override {
    return &collection;
  }

  nvinfer1::IPluginV3 *
  createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc,
               nvinfer1::TensorRTPhase phase) noexcept override {

    if (creatorShouldFail(fc))
      return nullptr;

    if (phase == nvinfer1::TensorRTPhase::kBUILD)
      return new PluginType(*fc);

    if (phase == nvinfer1::TensorRTPhase::kRUNTIME)
      return new PluginType(*fc);

    return nullptr;
  }

  char const *getPluginNamespace() const noexcept override { return ""; }

protected:
  nvinfer1::PluginFieldCollection collection;
  std::vector<nvinfer1::PluginField> fields;
};

/// Creator for the TestInferShapePlugin, there are no creation parameters.
class TestInferShapePluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
  using PluginType = TestInferShapePlugin;

  TestInferShapePluginCreator() {}

  char const *getPluginName() const noexcept override {
    return "TestInferShapePlugin";
  }
  char const *getPluginVersion() const noexcept override { return "0"; }
  nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override {
    return &collection;
  }

  nvinfer1::IPluginV3 *
  createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc,
               nvinfer1::TensorRTPhase phase) noexcept override {
    if (phase == nvinfer1::TensorRTPhase::kBUILD)
      return new PluginType(*fc);
    if (phase == nvinfer1::TensorRTPhase::kRUNTIME)
      return new PluginType(*fc);
    return nullptr;
  }

  char const *getPluginNamespace() const noexcept override { return ""; }

protected:
  nvinfer1::PluginFieldCollection collection{0, nullptr};
};

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)

class SymExpr : public ISymExpr {
public:
  SymExpr() : expr(nullptr) {}

  SymExpr(int32_t i, IExprBuilder &exprBuilder)
      : expr(exprBuilder.constant(i)) {}

  virtual ~SymExpr() {}

  void *getExpr() noexcept override {
    return const_cast<void *>(static_cast<const void *>(expr));
  }

  PluginArgType getType() const noexcept override {
    return PluginArgType::kINT;
  }

  PluginArgDataType getDataType() const noexcept override {
    return PluginArgDataType::kINT32;
  }

private:
  IDimensionExpr const *expr;
};

class TestQuickPlugin : public nvinfer1::IPluginV3,
                        public nvinfer1::IPluginV3QuickCore,
                        public nvinfer1::IPluginV3QuickAOTBuild {
public:
  TestQuickPlugin() = default;
  TestQuickPlugin(const TestQuickPlugin &) = default;

  TestQuickPlugin(const PluginFieldCollection &data) {
    std::cout << "Created TestQuickPlugin with " << data.nbFields
              << " fields:\n";
    for (unsigned i = 0, e = data.nbFields; i < e; i++) {
      PluginField field = data.fields[i];
      std::cout << "field[" << i << "] ";
      printField(std::cout, field);
    }
  }

  char const *getPluginName() const noexcept override {
    return "TestQuickPlugin";
  }
  char const *getPluginVersion() const noexcept override { return "0"; }
  char const *getPluginNamespace() const noexcept override { return ""; }
  int32_t getNbOutputs() const noexcept override { return 1; }

  int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in,
                          int32_t nbInputs,
                          nvinfer1::DynamicPluginTensorDesc const *out,
                          int32_t nbOutputs) noexcept override {
    return 0;
  }

  int32_t
  getNbSupportedFormatCombinations(DynamicPluginTensorDesc const *inOut,
                                   int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override {
    return 1;
  }

  int32_t getSupportedFormatCombinations(
      DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs,
      PluginTensorDesc *supportedCombinations,
      int32_t nbFormatCombinations) noexcept override {
    supportedCombinations[0].format = PluginFormat::kLINEAR;
    supportedCombinations[0].type = DataType::kFLOAT;
    return 0;
  }

  int32_t getLaunchParams(DimsExprs const *inputs,
                          DynamicPluginTensorDesc const *inOut,
                          int32_t nbInputs, int32_t nbOutputs,
                          IKernelLaunchParams *launchParams,
                          ISymExprs *extraArgs,
                          IExprBuilder &exprBuilder) noexcept override {
    dummyExpr = SymExpr(1, exprBuilder);
    launchParams->setGridX(&dummyExpr);
    launchParams->setGridY(&dummyExpr);
    launchParams->setGridZ(&dummyExpr);
    launchParams->setBlockX(&dummyExpr);
    launchParams->setBlockY(&dummyExpr);
    launchParams->setBlockZ(&dummyExpr);
    launchParams->setSharedMem(&dummyExpr);
    return 0;
  }

  int32_t getKernel(PluginTensorDesc const *in, int32_t nbInputs,
                    PluginTensorDesc const *out, int32_t nbOutputs,
                    const char **kernelName, char **compiledKernel,
                    int32_t *compiledKernelSize) noexcept override {
    *kernelName = "TestQuickPluginKernel";
    return 0;
  }

  IPluginCapability *
  getCapabilityInterface(PluginCapabilityType type) noexcept override {
    if (type == PluginCapabilityType::kCORE)
      return static_cast<IPluginV3QuickCore *>(this);
    if (type == PluginCapabilityType::kBUILD)
      return static_cast<IPluginV3QuickAOTBuild *>(this);
    // runtime interface is not public for AOT QDPs
    if (type == PluginCapabilityType::kRUNTIME)
      return nullptr;
    return nullptr;
  }

  // IPluginV3QuickAOTBuild implementation
  int32_t getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs,
                             DataType const *inputTypes,
                             int32_t const *inputRanks,
                             int32_t nbInputs) const noexcept override {
    outputTypes[0] = DataType::kFLOAT;
    return 0;
  }

  int32_t getOutputShapes(DimsExprs const *inputs, int32_t nbInputs,
                          DimsExprs const *shapeInputs, int32_t nbShapeInputs,
                          DimsExprs *outputs, int32_t nbOutputs,
                          IExprBuilder &exprBuilder) noexcept override {
    outputs[0].nbDims = 1;
    outputs[0].d[0] = exprBuilder.constant(1);
    return 0;
  }

  IPluginV3 *clone() noexcept override { return new TestQuickPlugin(*this); }

private:
  SymExpr dummyExpr;
};

class TestQuickPluginShape : public TestQuickPlugin {
public:
  TestQuickPluginShape() = default;
  TestQuickPluginShape(const TestQuickPluginShape &) = default;

  IPluginCapability *
  getCapabilityInterface(PluginCapabilityType type) noexcept override {
    if (type == PluginCapabilityType::kCORE)
      return static_cast<IPluginV3QuickCore *>(this);
    if (type == PluginCapabilityType::kBUILD)
      return static_cast<IPluginV3QuickAOTBuild *>(this);
    // runtime interface is not public for AOT QDPs
    if (type == PluginCapabilityType::kRUNTIME)
      return nullptr;
    return nullptr;
  }

  char const *getPluginName() const noexcept override {
    return "TestQuickPluginShape";
  }

  // IPluginV3QuickAOTBuild implementation
  int32_t getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs,
                             DataType const *inputTypes,
                             int32_t const *inputRanks,
                             int32_t nbInputs) const noexcept override {
    std::cout << "Input ranks: ";
    for (int32_t i = 0; i < nbInputs; ++i)
      std::cout << inputRanks[i] << " ";
    std::cout << std::endl;
    outputTypes[0] = DataType::kFLOAT;
    return 0;
  }

  int32_t getOutputShapes(DimsExprs const *inputs, int32_t nbInputs,
                          DimsExprs const *shapeInputs, int32_t nbShapeInputs,
                          DimsExprs *outputs, int32_t nbOutputs,
                          IExprBuilder &exprBuilder) noexcept override {
    if (nbInputs != 1 || inputs[0].nbDims != 4)
      return 1;
    if (nbOutputs != 1)
      return 1;

    std::cout << "Input DimExprs Constant Values:" << std::endl;
    for (int32_t i = 0; i < inputs[0].nbDims; ++i)
      std::cout << "Dimension " << i
                << " value: " << inputs[0].d[i]->getConstantValue()
                << std::endl;

    outputs[0].nbDims = 4;
    outputs[0].d[0] = exprBuilder.constant(42);
    outputs[0].d[1] = exprBuilder.operation(DimensionOperation::kSUM,
                                            *inputs[0].d[0], *inputs[0].d[1]);
    outputs[0].d[2] = exprBuilder.operation(DimensionOperation::kMAX,
                                            *inputs[0].d[1], *inputs[0].d[2]);
    outputs[0].d[3] =
        exprBuilder.operation(DimensionOperation::kCEIL_DIV, *inputs[0].d[3],
                              *exprBuilder.constant(3));
    return 0;
  }

  IPluginV3 *clone() noexcept override { return new TestQuickPlugin(*this); }
};

class TestQuickPluginCreator : public nvinfer1::IPluginCreatorV3Quick {
public:
  TestQuickPluginCreator() {
    fields.push_back(
        PluginField{"i64_param", nullptr, PluginFieldType::kINT64});
    fields.push_back(
        PluginField{"i32_param", nullptr, PluginFieldType::kINT32});
    fields.push_back(
        PluginField{"i16_param", nullptr, PluginFieldType::kINT16});
    fields.push_back(PluginField{"i8_param", nullptr, PluginFieldType::kINT8});
    fields.push_back(
        PluginField{"f32_param", nullptr, PluginFieldType::kFLOAT32});
    fields.push_back(
        PluginField{"f64_param", nullptr, PluginFieldType::kFLOAT64});
    fields.push_back(
        PluginField{"string_param", nullptr, PluginFieldType::kCHAR});
    fields.push_back(
        PluginField{"shape_param", nullptr, PluginFieldType::kDIMS});
    fields.push_back(
        PluginField{"shape_vec_param", nullptr, PluginFieldType::kDIMS});
    fields.push_back(
        PluginField{"f32_elements_param", nullptr, PluginFieldType::kFLOAT32});
    fields.push_back(
        PluginField{"f16_elements_param", nullptr, PluginFieldType::kFLOAT16});
    fields.push_back(PluginField{kPLUGIN_FAILURE_TRIGGER_FIELD_NAME, nullptr,
                                 PluginFieldType::kINT32});

    collection.nbFields = fields.size();
    collection.fields = fields.data();
  }

  char const *getPluginName() const noexcept override {
    return "TestQuickPlugin";
  }

  char const *getPluginVersion() const noexcept override { return "0"; }

  PluginFieldCollection const *getFieldNames() noexcept override {
    return &collection;
  }

  IPluginV3 *
  createPlugin(char const *name, char const *ns,
               PluginFieldCollection const *fc, TensorRTPhase phase,
               QuickPluginCreationRequest request) noexcept override {
    switch (request) {
    case QuickPluginCreationRequest::kSTRICT_AOT:
      std::cout << "QuickPluginCreationRequest: kSTRICT_AOT" << std::endl;
      break;
    default:
      std::cout << "QuickPluginCreationRequest: ERROR" << std::endl;
      return nullptr;
    }
    return new TestQuickPlugin(*fc);
  }

  char const *getPluginNamespace() const noexcept override { return ""; }

protected:
  nvinfer1::PluginFieldCollection collection;
  std::vector<nvinfer1::PluginField> fields;
};

class TestQuickShapePluginCreator : public nvinfer1::IPluginCreatorV3Quick {
public:
  TestQuickShapePluginCreator() {}

  char const *getPluginName() const noexcept override {
    return "TestQuickPluginShape";
  }

  char const *getPluginVersion() const noexcept override { return "0"; }

  PluginFieldCollection const *getFieldNames() noexcept override {
    return &collection;
  }

  IPluginV3 *
  createPlugin(char const *name, char const *ns,
               PluginFieldCollection const *fc, TensorRTPhase phase,
               QuickPluginCreationRequest request) noexcept override {
    return new TestQuickPluginShape();
  }

  char const *getPluginNamespace() const noexcept override { return ""; }

private:
  nvinfer1::PluginFieldCollection collection{0, nullptr};
};

#endif // MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)

} // namespace

// Do the static registration of the creator.
REGISTER_TENSORRT_PLUGIN(TestPlugin1Creator);
REGISTER_TENSORRT_PLUGIN(TestInferShapePluginCreator);
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
REGISTER_TENSORRT_PLUGIN(TestQuickPluginCreator);
REGISTER_TENSORRT_PLUGIN(TestQuickShapePluginCreator);
#endif // MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-prototypes"
#endif

extern "C" {
/// Provide an exported C-style function for creating the plugin creator.
TENSORRTTESTPLUGINS_EXPORT nvinfer1::IPluginCreatorInterface *
getTestPlugin1Creator() {
  return std::make_unique<TestPlugin1Creator>().release();
}
}

#endif
