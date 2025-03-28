//===- PlanInterfaces.cpp -- ----------------------------------------------===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Definitions for Plan op/attribute/type interfaces.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.h"
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"

#define DEBUG_TYPE "plan-interfaces"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;
using namespace mlir::plan;

#include "mlir-tensorrt/Dialect/Plan/IR/PlanAttrInterfaces.cpp.inc"
