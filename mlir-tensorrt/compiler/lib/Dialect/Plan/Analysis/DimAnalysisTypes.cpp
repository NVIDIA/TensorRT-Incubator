//===- DimAnalysisTypes.cpp - Dimension Analysis Types ----------*- C++ -*-===//
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
/// Implementation of dimension analysis types and expression factory.
///
//===----------------------------------------------------------------------===//

#include "mlir-tensorrt/Dialect/Plan/Analysis/DimAnalysisTypes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::plan;

//===----------------------------------------------------------------------===//
// DimVariable utilities
//===----------------------------------------------------------------------===//

std::string mlir::plan::describeDimVariable(const DimVariable &var) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::visit(
      [&os](const auto &v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, IntegerSSAValue>) {
          os << "SSA(";
          v.value.print(os);
          os << ")";
        } else if constexpr (std::is_same_v<T, TensorDim>) {
          os << "Dim(";
          v.tensor.print(os);
          os << ", " << v.dimIndex << ")";
        } else if constexpr (std::is_same_v<T, DimSymbol>) {
          os << "Symbol(" << v.symbolName << ")";
        }
      },
      var);

  return result;
}

//===----------------------------------------------------------------------===//
// DimExprFactory implementation
//===----------------------------------------------------------------------===//

DimExprRef DimExprFactory::createVariable(DimVariable var) {
  variableExprs.push_back(std::make_unique<VariableExpr>(std::move(var)));
  return variableExprs.back().get();
}

DimExprRef DimExprFactory::getOrCreateValueExpr(Value value) {
  auto it = valueToExpr.find(value);
  if (it != valueToExpr.end())
    return it->second;

  DimExprRef expr = createVariable(IntegerSSAValue{value});
  valueToExpr[value] = expr;
  return expr;
}

DimExprRef DimExprFactory::getOrCreateTensorDimExpr(Value tensor, int32_t dim) {
  auto key = std::make_pair(tensor, dim);
  auto it = tensorDimToExpr.find(key);
  if (it != tensorDimToExpr.end())
    return it->second;

  DimExprRef expr = createVariable(TensorDim{tensor, dim});
  tensorDimToExpr[key] = expr;
  return expr;
}

DimExprRef DimExprFactory::getOrCreateSymbolExpr(StringRef symbol) {
  auto it = symbolToExpr.find(symbol);
  if (it != symbolToExpr.end())
    return it->second;

  DimExprRef expr = createVariable(DimSymbol{symbol.str()});
  symbolToExpr[symbol] = expr;
  return expr;
}
