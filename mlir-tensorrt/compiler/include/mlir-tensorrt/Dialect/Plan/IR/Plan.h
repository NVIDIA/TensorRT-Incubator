//===- Plan.h ---------------------------------------------------*- C++ -*-===//
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
/// Plan dialect declarations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_PLAN_IR_PLAN_H
#define MLIR_TENSORRT_DIALECT_PLAN_IR_PLAN_H

#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Transforms/Clustering/Clustering.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"

//===----------------------------------------------------------------------===//
// Plan Dialect
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/PlanOpsDialect.h.inc"

template <typename Attr>
void mlir::plan::PlanDialect::addExtensionAttribute() {
  StringRef mnemonic = Attr::getMnemonic();
  attrParsingHooks.try_emplace(mnemonic, Attr::parse);
  attrPrintingHooks.try_emplace(TypeID::get<Attr>(),
                                [](Attribute attr, AsmPrinter &printer) {
                                  printer << cast<Attr>(attr).getMnemonic();
                                  cast<Attr>(attr).print(printer);
                                });
  addAttributes<Attr>();
}

namespace mlir::plan {

//===----------------------------------------------------------------------===//
// PlanDialectExtension
//===----------------------------------------------------------------------===//

/// PlanDialectExtension is the bae class for DialectExtensions to add
/// Attributes and CompilationTaskExtensions into the PlanDialect.
template <typename DerivedTy, typename... ExtraDialects>
class PlanDialectExtension
    : public DialectExtension<DerivedTy, PlanDialect, ExtraDialects...> {

  using Initializer = std::function<void(PlanDialect *)>;
  using DialectLoader = std::function<void(MLIRContext *)>;

public:
  using Base = PlanDialectExtension<DerivedTy, ExtraDialects...>;

  StringRef getName() const { return ""; }

  /// Extension constructor. The argument indicates whether to skip generated
  /// dialects when applying the extension.
  explicit PlanDialectExtension() { static_cast<DerivedTy *>(this)->init(); }

  /// Extension application hook. Actually loads the dependent dialects and
  /// registers the additional operations. Not expected to be called directly.
  void apply(MLIRContext *context, PlanDialect *planDialect,
             ExtraDialects *...) const final {
    for (const DialectLoader &loader : dialectLoaders)
      loader(context);
    for (const DialectLoader &loader : generatedDialectLoaders)
      loader(context);
    for (const Initializer &init : initializers)
      init(planDialect);
  }

  /// Hook for derived classes to inject constructor behavior.
  void init() {}

  template <typename... AttrTypes>
  void registerAttributes() {
    initializers.push_back([](PlanDialect *planDialect) {
      planDialect->addExtensionAttribute<AttrTypes...>();
    });
  }

protected:
  /// Callbacks performing extension initialization, e.g., registering ops,
  /// types and defining the additional data.
  SmallVector<Initializer> initializers;

  /// Callbacks loading the dependent dialects, i.e. the dialect needed for the
  /// extension ops.
  SmallVector<DialectLoader> dialectLoaders;

  /// Callbacks loading the generated dialects, i.e. the dialects produced when
  /// applying the transformations.
  SmallVector<DialectLoader> generatedDialectLoaders;
};
} // namespace mlir::plan

//===----------------------------------------------------------------------===//
// Plan Enums
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/PlanEnums.h.inc"

//===----------------------------------------------------------------------===//
// Plan Attributes
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/PlanAttrInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir-tensorrt/Dialect/Plan/IR/PlanAttributes.h.inc"

//===----------------------------------------------------------------------===//
// Plan Types
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "mlir-tensorrt/Dialect/Plan/IR/PlanOpsTypes.h.inc"

namespace mlir::plan {
namespace PlanOpTrait {
template <typename ConcreteType>
class PlanDialectOp
    : public ::mlir::OpTrait::TraitBase<ConcreteType, PlanDialectOp> {};

} // namespace PlanOpTrait
} // namespace mlir::plan

//===----------------------------------------------------------------------===//
// Plan Ops
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-tensorrt/Dialect/Plan/IR/PlanOps.h.inc"

#endif // MLIR_TENSORRT_DIALECT_PLAN_IR_PLAN_H
