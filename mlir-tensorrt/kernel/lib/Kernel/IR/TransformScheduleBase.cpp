//===- TransformScheduleBase.cpp ------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Definitions for the transform schedule generator base class and
/// schedule generator registration infrastructure.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/TransformScheduleBase.h"
#include "mlir-kernel/Kernel/IR/Dialect.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::kernel;

bool TransformScheduleBase::isElementTypeSupported(const mlir::DataLayout &,
                                                   Type type,
                                                   bool doesSupportComplex,
                                                   bool requiresVectorType) {
  if (requiresVectorType)
    return mlir::isValidVectorTypeElementType(type);
  if (doesSupportComplex)
    return isa<IntegerType, FloatType, ComplexType>(type);
  return isa<IntegerType, FloatType>(type);
}

bool TransformScheduleBase::allTypesAreSupported(
    const mlir::DataLayout &dataLayout, Operation *op, bool doesSupportComplex,
    bool requiresVectorType) {
  auto typeIsSupported = [&](Type t) {
    return isElementTypeSupported(dataLayout, mlir::getElementTypeOrSelf(t),
                                  doesSupportComplex, requiresVectorType);
  };
  return llvm::all_of(op->getOperandTypes(), typeIsSupported) &&
         llvm::all_of(op->getResultTypes(), typeIsSupported);
}

//===----------------------------------------------------------------------===//
// TransformScheduleRegistry
//===----------------------------------------------------------------------===//

FailureOr<const TransformScheduleBase *>
TransformScheduleRegistry::parseTransformScheduleGenerator(
    llvm::StringRef str) const {
  auto it = scheduleGeneratorNameMap.find(str);
  if (it == scheduleGeneratorNameMap.end())
    return failure();
  auto schedule = schedules.find(it->second);
  if (schedule == schedules.end())
    return failure();
  return schedule->second.get();
}

void TransformScheduleRegistry::registerTransformScheduleGenerator(
    std::unique_ptr<TransformScheduleBase> generator) {
  StringRef mneomonic = generator->getMnemonic();
  TypeID typeID = generator->getTypeID();

  if (llvm::is_contained(scheduleGeneratorNameMap, mneomonic)) {
    std::string err = llvm::formatv("TransformScheduleRegistry: "
                                    "mnemonic {0} already registered",
                                    mneomonic);
    llvm::report_fatal_error(StringRef(err));
  }

  if (llvm::is_contained(schedules, typeID)) {
    std::string err =
        llvm::formatv("TransformScheduleRegistry: "
                      "typeID for schedule {0} already registered",
                      mneomonic);
    llvm::report_fatal_error(StringRef(err));
  }

  schedules.insert(std::make_pair(typeID, std::move(generator)));
  scheduleGeneratorNameMap[mneomonic] = typeID;
}

//===----------------------------------------------------------------------===//
// Registration Helpers
//===----------------------------------------------------------------------===//

/// Register a transform schedule generator with the given mnemonic and type ID.
/// This is the internal implementation of the public interface and is not
/// intended to be used directly.
void kernel::detail::registerTransformScheduleGenerator(
    KernelDialect *dialect, std::unique_ptr<TransformScheduleBase> generator) {
  dialect->transformScheduleGeneratorRegistry
      .registerTransformScheduleGenerator(std::move(generator));
}
