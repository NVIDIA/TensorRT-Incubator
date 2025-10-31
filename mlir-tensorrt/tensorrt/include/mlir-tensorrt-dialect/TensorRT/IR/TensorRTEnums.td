//===- TensorRTEnums.td -----------------------------------------*- C++ -*-===//
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
// This file contains definitions for the TensorRT dialect enums.
// These enums were originally bootstraped by parsing the NvInfer.h header file.
// Some enums are currently not be used but are kept for completeness.
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTENUMS
#define MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTENUMS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.td"

//===----------------------------------------------------------------------===//
// All TensorRT enums that are derived from an `nvinfer` namespace enum
// from the C++ TensorRT API should use the below sub-class instead of
// I32EnumAttr/EnumAttr directly. The reason is because we use these class
// names to automatically generate converters that convert the `mlir::tensorrt`
// dialect enum to the `nvinfer` dialect enum using the Tablegen generator
// `mlir-tensorrt-tblgen -gen-tensorrt-enum-converter-defs`.
//
// Some enums are currently not be used but are kept for completeness.
//===----------------------------------------------------------------------===//
class TensorRT_I32EnumAttr<string name, string description, list<I32EnumAttrCase> cases> :
        I32EnumAttr<name, description, cases> {
  bit skipNvInferEnumConverterGeneration = 0;
}

class TensorRT_NonNvInferI32EnumAttr<string name, string description, list<I32EnumAttrCase> cases> :
        I32EnumAttr<name, description, cases>;

class TensorRT_EnumAttr<IntEnumAttr intEnumAttr, string mnemonic> :
        EnumAttr<TensorRT_Dialect, intEnumAttr, mnemonic, [TensorRTEnumAttrInterface]> {
  let extraClassDeclaration = [{
    /// Return a string which has the equivalent enum value in the nvinfer1 namespace.
    /// This is used in the TensorRTEnumAttrInterface and in C++ code generation.
    std::string getNvInferValueStr() {
      return "}] # "::nvinfer1::"
        # intEnumAttr.className
        # "::\" + stringify" # intEnumAttr.className # [{(this->getValue()).str();
    }
  }];

  let assemblyFormat = "`<` $value `>`";
}

def TensorRT_ActivationType : TensorRT_I32EnumAttr<
  "ActivationType", "",
  [
    I32EnumAttrCase<"kRELU", 0>,
    I32EnumAttrCase<"kSIGMOID", 1>,
    I32EnumAttrCase<"kTANH", 2>,
    I32EnumAttrCase<"kLEAKY_RELU", 3>,
    I32EnumAttrCase<"kELU", 4>,
    I32EnumAttrCase<"kSELU", 5>,
    I32EnumAttrCase<"kSOFTSIGN", 6>,
    I32EnumAttrCase<"kSOFTPLUS", 7>,
    I32EnumAttrCase<"kCLIP", 8>,
    I32EnumAttrCase<"kHARD_SIGMOID", 9>,
    I32EnumAttrCase<"kSCALED_TANH", 10>,
    I32EnumAttrCase<"kTHRESHOLDED_RELU", 11>,
    I32EnumAttrCase<"kGELU_ERF", 12>,
    I32EnumAttrCase<"kGELU_TANH", 13>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
  let skipNvInferEnumConverterGeneration = 1;
}

def TensorRT_ActivationTypeAttr : TensorRT_EnumAttr<TensorRT_ActivationType, "activation_type">{
}

def TensorRT_PaddingMode : TensorRT_I32EnumAttr<
  "PaddingMode", "",
  [
    I32EnumAttrCase<"kEXPLICIT_ROUND_DOWN", 0>,
    I32EnumAttrCase<"kEXPLICIT_ROUND_UP", 1>,
    I32EnumAttrCase<"kSAME_UPPER", 2>,
    I32EnumAttrCase<"kSAME_LOWER", 3>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_PaddingModeAttr : TensorRT_EnumAttr<TensorRT_PaddingMode, "padding_mode">{
}

def TensorRT_PoolingType : TensorRT_I32EnumAttr<
  "PoolingType", "",
  [
    I32EnumAttrCase<"kMAX", 0>,
    I32EnumAttrCase<"kAVERAGE", 1>,
    I32EnumAttrCase<"kMAX_AVERAGE_BLEND", 2>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_PoolingTypeAttr : TensorRT_EnumAttr<TensorRT_PoolingType, "pooling_type">{
}

def TensorRT_ElementWiseOperation : TensorRT_I32EnumAttr<
  "ElementWiseOperation", "",
  [
    I32EnumAttrCase<"kSUM", 0>,
    I32EnumAttrCase<"kPROD", 1>,
    I32EnumAttrCase<"kMAX", 2>,
    I32EnumAttrCase<"kMIN", 3>,
    I32EnumAttrCase<"kSUB", 4>,
    I32EnumAttrCase<"kDIV", 5>,
    I32EnumAttrCase<"kPOW", 6>,
    I32EnumAttrCase<"kFLOOR_DIV", 7>,
    I32EnumAttrCase<"kAND", 8>,
    I32EnumAttrCase<"kOR", 9>,
    I32EnumAttrCase<"kXOR", 10>,
    I32EnumAttrCase<"kEQUAL", 11>,
    I32EnumAttrCase<"kGREATER", 12>,
    I32EnumAttrCase<"kLESS", 13>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_ElementWiseOperationAttr : TensorRT_EnumAttr<TensorRT_ElementWiseOperation, "element_wise_operation">{
}

def TensorRT_GatherMode : TensorRT_I32EnumAttr<
  "GatherMode", "",
  [
    I32EnumAttrCase<"kDEFAULT", 0>,
    I32EnumAttrCase<"kELEMENT", 1>,
    I32EnumAttrCase<"kND", 2>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_GatherModeAttr : TensorRT_EnumAttr<TensorRT_GatherMode, "gather_mode">{
}

def TensorRT_UnaryOperation : TensorRT_I32EnumAttr<
  "UnaryOperation", "",
  [
    I32EnumAttrCase<"kEXP", 0>,
    I32EnumAttrCase<"kLOG", 1>,
    I32EnumAttrCase<"kSQRT", 2>,
    I32EnumAttrCase<"kRECIP", 3>,
    I32EnumAttrCase<"kABS", 4>,
    I32EnumAttrCase<"kNEG", 5>,
    I32EnumAttrCase<"kSIN", 6>,
    I32EnumAttrCase<"kCOS", 7>,
    I32EnumAttrCase<"kTAN", 8>,
    I32EnumAttrCase<"kSINH", 9>,
    I32EnumAttrCase<"kCOSH", 10>,
    I32EnumAttrCase<"kASIN", 11>,
    I32EnumAttrCase<"kACOS", 12>,
    I32EnumAttrCase<"kATAN", 13>,
    I32EnumAttrCase<"kASINH", 14>,
    I32EnumAttrCase<"kACOSH", 15>,
    I32EnumAttrCase<"kATANH", 16>,
    I32EnumAttrCase<"kCEIL", 17>,
    I32EnumAttrCase<"kFLOOR", 18>,
    I32EnumAttrCase<"kERF", 19>,
    I32EnumAttrCase<"kNOT", 20>,
    I32EnumAttrCase<"kSIGN", 21>,
    I32EnumAttrCase<"kROUND", 22>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_UnaryOperationAttr : TensorRT_EnumAttr<TensorRT_UnaryOperation, "unary_operation">{
}

def TensorRT_ReduceOperation : TensorRT_I32EnumAttr<
  "ReduceOperation", "",
  [
    I32EnumAttrCase<"kSUM", 0>,
    I32EnumAttrCase<"kPROD", 1>,
    I32EnumAttrCase<"kMAX", 2>,
    I32EnumAttrCase<"kMIN", 3>,
    I32EnumAttrCase<"kAVG", 4>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_ReduceOperationAttr : TensorRT_EnumAttr<TensorRT_ReduceOperation, "reduce_operation">{
}

def TensorRT_SliceMode : TensorRT_I32EnumAttr<
  "SliceMode", "",
  [
    I32EnumAttrCase<"kDEFAULT", 0>,
    I32EnumAttrCase<"kWRAP", 1>,
    I32EnumAttrCase<"kCLAMP", 2>,
    I32EnumAttrCase<"kFILL", 3>,
    I32EnumAttrCase<"kREFLECT", 4>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
  let skipNvInferEnumConverterGeneration = 1;
}

def TensorRT_SliceModeAttr : TensorRT_EnumAttr<TensorRT_SliceMode, "slice_mode">{
}

def TensorRT_TopKOperation : TensorRT_I32EnumAttr<
  "TopKOperation", "",
  [
    I32EnumAttrCase<"kMAX", 0>,
    I32EnumAttrCase<"kMIN", 1>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_TopKOperationAttr : TensorRT_EnumAttr<TensorRT_TopKOperation, "top_k_operation">{
}

def TensorRT_MatrixOperation : TensorRT_I32EnumAttr<
  "MatrixOperation", "",
  [
    I32EnumAttrCase<"kNONE", 0>,
    I32EnumAttrCase<"kTRANSPOSE", 1>,
    I32EnumAttrCase<"kVECTOR", 2>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_MatrixOperationAttr : TensorRT_EnumAttr<TensorRT_MatrixOperation, "matrix_operation">{
}

def TensorRT_ResizeMode : TensorRT_I32EnumAttr<
  "ResizeMode", "",
  [
    I32EnumAttrCase<"kNEAREST", 0>,
    I32EnumAttrCase<"kLINEAR", 1>,
    I32EnumAttrCase<"kCUBIC", 2>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_ResizeModeAttr : TensorRT_EnumAttr<TensorRT_ResizeMode, "resize_mode">{
}

def TensorRT_ResizeCoordinateTransformation : TensorRT_I32EnumAttr<
  "ResizeCoordinateTransformation", "",
  [
    I32EnumAttrCase<"kALIGN_CORNERS", 0>,
    I32EnumAttrCase<"kASYMMETRIC", 1>,
    I32EnumAttrCase<"kHALF_PIXEL", 2>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_ResizeCoordinateTransformationAttr : TensorRT_EnumAttr<TensorRT_ResizeCoordinateTransformation, "resize_coordinate_transformation">{
}

def TensorRT_ResizeSelector : TensorRT_I32EnumAttr<
  "ResizeSelector", "",
  [
    I32EnumAttrCase<"kFORMULA", 0>,
    I32EnumAttrCase<"kUPPER", 1>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_ResizeSelectorAttr : TensorRT_EnumAttr<TensorRT_ResizeSelector, "resize_selector">{
}

def TensorRT_ResizeRoundMode : TensorRT_I32EnumAttr<
  "ResizeRoundMode", "",
  [
    I32EnumAttrCase<"kHALF_UP", 0>,
    I32EnumAttrCase<"kHALF_DOWN", 1>,
    I32EnumAttrCase<"kFLOOR", 2>,
    I32EnumAttrCase<"kCEIL", 3>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_ResizeRoundModeAttr : TensorRT_EnumAttr<TensorRT_ResizeRoundMode, "resize_round_mode">{
}

def TensorRT_LoopOutput : TensorRT_I32EnumAttr<
  "LoopOutput", "",
  [
    I32EnumAttrCase<"kLAST_VALUE", 0>,
    I32EnumAttrCase<"kCONCATENATE", 1>,
    I32EnumAttrCase<"kREVERSE", 2>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_LoopOutputAttr : TensorRT_EnumAttr<TensorRT_LoopOutput, "loop_output">{
}

def TensorRT_TripLimit : TensorRT_I32EnumAttr<
  "TripLimit", "",
  [
    I32EnumAttrCase<"kCOUNT", 0>,
    I32EnumAttrCase<"kWHILE", 1>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_TripLimitAttr : TensorRT_EnumAttr<TensorRT_TripLimit, "trip_limit">{
}

def TensorRT_FillOperation : TensorRT_I32EnumAttr<
  "FillOperation", "",
  [
    I32EnumAttrCase<"kLINSPACE", 0>,
    I32EnumAttrCase<"kRANDOM_UNIFORM", 1>,
    I32EnumAttrCase<"kRANDOM_NORMAL", 2>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_FillOperationAttr : TensorRT_EnumAttr<TensorRT_FillOperation, "fill_operation">{
}

def TensorRT_ScatterMode : TensorRT_I32EnumAttr<
  "ScatterMode", "",
  [
    I32EnumAttrCase<"kELEMENT", 0>,
    I32EnumAttrCase<"kND", 1>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_ScatterModeAttr : TensorRT_EnumAttr<TensorRT_ScatterMode, "scatter_mode">{
}

def TensorRT_AttentionNormalizationOp : TensorRT_I32EnumAttr<
  "AttentionNormalizationOp", "",
  [
    I32EnumAttrCase<"kNONE", 0>,
    I32EnumAttrCase<"kSOFTMAX", 1>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_AttentionNormalizationOpAttr : TensorRT_EnumAttr<TensorRT_AttentionNormalizationOp, "attention_normalization_op">{
}

def TensorRT_DataType : TensorRT_I32EnumAttr<
  "DataType", "",
  [
    I32EnumAttrCase<"kFLOAT", 0>,
    I32EnumAttrCase<"kHALF", 1>,
    I32EnumAttrCase<"kINT8", 2>,
    I32EnumAttrCase<"kINT32", 3>,
    I32EnumAttrCase<"kBOOL", 4>,
    I32EnumAttrCase<"kUINT8", 5>,
    I32EnumAttrCase<"kFP8", 6>,
    I32EnumAttrCase<"kBF16", 7>,
    I32EnumAttrCase<"kINT64", 8>,
    I32EnumAttrCase<"kINT4", 9>,
    I32EnumAttrCase<"kFP4", 10>,
    I32EnumAttrCase<"kE8M0", 11>
  ]>
{
  let cppNamespace = "::mlir::tensorrt";
  let genSpecializedAttr = 0;
}

def TensorRT_DataTypeAttr : TensorRT_EnumAttr<TensorRT_DataType, "data_type">{
}

#endif // MLIR_TENSORRT_DIALECT_TENSORRT_IR_TENSORRTENUMS
