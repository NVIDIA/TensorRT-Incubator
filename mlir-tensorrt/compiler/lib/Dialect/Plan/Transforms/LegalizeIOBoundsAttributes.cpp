//===- LegalizeIOBoundsAttributes.cpp -------------------------------------===//
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
/// Implementation of the `plan-legalize-io-bounds-attributes` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::plan {
#define GEN_PASS_DEF_LEGALIZEIOBOUNDSATTRIBUTESPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

/// Convert 'tensorrt' dialect arg/result value bounds attribute into 'plan'
/// bounds attribute.
static Attribute convertValueBoundsAttr(OpBuilder &b, Type type,
                                        tensorrt::ShapeProfileAttr trtAttr) {
  MLIRContext *ctx = b.getContext();
  Type elementType = mlir::getElementTypeOrSelf(type);
  assert(elementType.isIntOrIndex() && "expected int or index element type");
  SmallVector<int64_t> boundsShape;
  if (auto shapedType = dyn_cast<ShapedType>(type))
    boundsShape = llvm::to_vector(shapedType.getShape());
  auto boundsValueType = RankedTensorType::get(boundsShape, elementType);
  auto convertI64ArrayToDenseElements = [&](ArrayRef<int64_t> i64Vals) {
    return DenseElementsAttr::get(
        boundsValueType,
        llvm::map_to_vector(i64Vals, [&](int64_t i64Val) -> Attribute {
          return b.getIntegerAttr(elementType, i64Val);
        }));
  };
  return plan::BoundsAttr::get(
      ctx, BoundsKind::Value, DenseI64ArrayAttr{}, DenseI64ArrayAttr{},
      convertI64ArrayToDenseElements(trtAttr.getMin()),
      convertI64ArrayToDenseElements(trtAttr.getMax()));
}

/// Convert an argument or result dictionary attribute using the given
/// converters.
static DictionaryAttr convertDictAttr(
    OpBuilder &b, DictionaryAttr attrs,
    ArrayRef<std::function<std::optional<NamedAttribute>(StringRef, Attribute,
                                                         Type)>>
        converters,
    Type argumentType) {
  if (!attrs)
    return b.getDictionaryAttr({});
  SmallVector<NamedAttribute> result;
  for (NamedAttribute attr : attrs) {
    for (const auto &converter : converters) {
      if (std::optional<NamedAttribute> converted =
              converter(attr.getName(), attr.getValue(), argumentType)) {
        result.push_back(*converted);
        break;
      }
    }
  }
  return b.getDictionaryAttr(result);
}

static void convertFuncArgAndResultAttrs(OpBuilder &b, func::FuncOp op) {
  SmallVector<DictionaryAttr> argAttrs;
  SmallVector<DictionaryAttr> resAttrs;
  SmallVector<
      std::function<std::optional<NamedAttribute>(StringRef, Attribute, Type)>>
      converters;

  FunctionType funcType = op.getFunctionType();

  converters.push_back([&](StringRef name, Attribute value,
                           Type argumentType) -> std::optional<NamedAttribute> {
    auto trtAttr = dyn_cast<tensorrt::ShapeProfileAttr>(value);
    if (!trtAttr)
      return std::nullopt;
    if (name == tensorrt::TensorRTDialect::getShapeProfileArgAttrName())
      return b.getNamedAttr(
          plan::PlanDialect::kShapeBoundsAttrName,
          plan::BoundsAttr::get(b.getContext(), plan::BoundsKind::Shape,
                                trtAttr.getMin(), trtAttr.getMax()));

    if (name ==
        tensorrt::TensorRTDialect::getShapeTensorValueBoundsArgAttrName())
      return b.getNamedAttr(plan::PlanDialect::kValueBoundsAttrName,
                            convertValueBoundsAttr(b, argumentType, trtAttr));
    return std::nullopt;
  });

  // Fallback: Always pass through.
  converters.push_back([&](StringRef name, Attribute value,
                           Type argumentType) -> std::optional<NamedAttribute> {
    return b.getNamedAttr(name, value);
  });

  for (unsigned argIdx = 0, e = op.getNumArguments(); argIdx < e; argIdx++) {
    Type argumentType = funcType.getInput(argIdx);
    argAttrs.push_back(convertDictAttr(b, op.getArgAttrDict(argIdx), converters,
                                       argumentType));
  }
  for (unsigned resIdx = 0, e = op.getNumResults(); resIdx < e; resIdx++) {
    Type resultType = funcType.getResult(resIdx);
    resAttrs.push_back(convertDictAttr(b, op.getResultAttrDict(resIdx),
                                       converters, resultType));
  }

  op.setAllArgAttrs(argAttrs);
  op.setAllResultAttrs(resAttrs);
}

namespace {
class LegalizeIOBoundsAttributesPass
    : public plan::impl::LegalizeIOBoundsAttributesPassBase<
          LegalizeIOBoundsAttributesPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    ModuleOp op = getOperation();
    OpBuilder b(op);
    for (auto func : op.getOps<func::FuncOp>())
      convertFuncArgAndResultAttrs(b, func);
  }
};
} // namespace
