//===- PlanToExecutor.cpp -------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024 2025 CORPORATION & AFFILIATES.
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
/// Implementation of the `convert-plan-to-executor` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/PlanToLLVM/PlanToLLVM.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

using namespace mlir;

/// Populate type and op conversions for the Plan dialect. Currently we only
/// need this for converting the plan memory space attributes. We convert them
/// into numeric address space attributes for LLVM. This is different from the
/// "Executor" conversion pathway where we use the "plan-to-executor" pass to
/// completely convert the Plan memory space attributes to Executor memory space
/// attributes inplace.
void mlir::populatePlanToLLVMTypeConversions(LLVMTypeConverter &typeConverter) {
  typeConverter.addTypeAttributeConversion(
      [](MemRefType type, plan::MemorySpaceAttr attr) -> Attribute {
        auto zero = IntegerAttr::get(IntegerType::get(type.getContext(), 0), 0);
        return zero;
      });
}

namespace {
/// Implement the interface to convert Plan to LLVM.
struct PlanToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populatePlanToLLVMTypeConversions(typeConverter);
  }
};
} // namespace

void mlir::registerConvertPlanToLLVMPatternInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, plan::PlanDialect *dialect) {
    dialect->addInterfaces<PlanToLLVMDialectInterface>();
  });
}
