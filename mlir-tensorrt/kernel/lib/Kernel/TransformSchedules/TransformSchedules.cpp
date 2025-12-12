//===- TransformSchedules.cpp ---------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of TransformSchedule common data structures.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/TransformSchedules/TransformSchedules.h"
#include "mlir-kernel/Kernel/IR/Enums.h"
#include "mlir-kernel/Kernel/IR/TransformScheduleBase.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::kernel;

///===----------------------------------------------------------------------===//
// Common helper methods for transform schedule generators
//===----------------------------------------------------------------------===//

template <typename MappingAttrType>
FailureOr<ArrayAttr>
getGpuDistributionMappingAttr(Location loc, ArrayRef<int64_t> tileShape) {

  MLIRContext *ctx = loc->getContext();
  SmallVector<Attribute, 4> elements;
  static constexpr std::array<gpu::MappingId, 3> dimensions = {
      gpu::MappingId::DimX, gpu::MappingId::DimY, gpu::MappingId::DimZ};

  // Only consider non-zero tile dimension sizes. The zero is placeholder for
  // "not tiled".
  auto filteredShape =
      llvm::make_filter_range(tileShape, [](int64_t x) { return x != 0; });
  const unsigned numTiledDims = tileShape.size() - llvm::count(tileShape, 0);
  if (numTiledDims > 10)
    return emitError(loc) << "cannot distribute more than 10 dimensions to GPU "
                             "blocks/threads ";

  if (numTiledDims <= dimensions.size()) {
    for (auto [dimSize, gpuDim] : llvm::zip(filteredShape, dimensions))
      elements.push_back(MappingAttrType::get(ctx, gpuDim));
    return ArrayAttr::get(ctx, elements);
  }

  for (auto [idx, dimSize] : llvm::enumerate(filteredShape))
    elements.push_back(MappingAttrType::get(
        ctx, static_cast<gpu::MappingId>(
                 static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx)));

  return ArrayAttr::get(ctx, elements);
}

/// Return an ArrayAttr containing a GPUBlockMappingAttr for each non-zero
/// element in `tileShape`.
FailureOr<ArrayAttr>
kernel::getCTADistributionMappingAttr(Location loc,
                                      ArrayRef<int64_t> tileShape) {
  if (tileShape.empty())
    return getGpuDistributionMappingAttr<gpu::GPUBlockMappingAttr>(loc, {1});
  return getGpuDistributionMappingAttr<gpu::GPUBlockMappingAttr>(loc,
                                                                 tileShape);
}

/// Return an ArrayAttr containing a GPUThreadMappingAttr for each non-zero
/// element in `tileShape`.
FailureOr<ArrayAttr>
kernel::getThreadDistributionMappingAttr(Location loc,
                                         ArrayRef<int64_t> tileShape) {
  return getGpuDistributionMappingAttr<gpu::GPUThreadMappingAttr>(loc,
                                                                  tileShape);
}

FailureOr<ArrayAttr>
kernel::getWarpDistributionMappingAttr(Location loc,
                                       ArrayRef<int64_t> tileShape) {
  return getGpuDistributionMappingAttr<gpu::GPUWarpMappingAttr>(loc, tileShape);
}

//===----------------------------------------------------------------------===//
// TransformScheduleSelector
//===----------------------------------------------------------------------===//

TransformScheduleSelector::TransformScheduleSelector(
    MLIRContext *context, const TransformScheduleOptions &options,
    ScheduleFilterFunc scheduleFilter)
    : TransformScheduleSelector(
          context->getOrLoadDialect<kernel::KernelDialect>()
              ->getTransformScheduleRegistry(),
          options, scheduleFilter) {}

TransformScheduleSelector::TransformScheduleSelector(
    const TransformScheduleRegistry &registry, TransformScheduleOptions options,
    ScheduleFilterFunc scheduleFilter)
    : registry(registry), options(options), scheduleFilter(scheduleFilter) {
  // Functor used to walk all of the operations registered in the context. This
  // is useful for patterns that get applied to multiple operations, such as
  // interface and trait based patterns.
  std::vector<RegisteredOperationName> opInfos;
  auto addToOpsWhen =
      [&](const std::unique_ptr<TransformScheduleBase> &pattern,
          function_ref<bool(RegisteredOperationName)> callbackFn) {
        if (opInfos.empty())
          opInfos = pattern->getContext()->getRegisteredOperations();
        for (RegisteredOperationName info : opInfos)
          if (callbackFn(info))
            schedules[info].push_back(pattern.get());
      };

  for (const auto &[typeId, pat] : registry.schedules) {
    if (std::optional<OperationName> rootName = pat->getRootKind()) {
      schedules[*rootName].push_back(pat.get());
      continue;
    }
    if (std::optional<TypeID> interfaceID = pat->getRootInterfaceID()) {
      addToOpsWhen(pat, [&](RegisteredOperationName info) {
        return info.hasInterface(*interfaceID);
      });
      continue;
    }
    if (std::optional<TypeID> traitID = pat->getRootTraitID()) {
      addToOpsWhen(pat, [&](RegisteredOperationName info) {
        return info.hasTrait(*traitID);
      });
      continue;
    }
  }
}

FailureOr<const TransformScheduleBase *>
TransformScheduleSelector::getScheduleGenerator(Operation *op) const {
  auto it = schedules.find(op->getName());
  if (it == schedules.end())
    return failure();

  SmallVector<const TransformScheduleBase *> schedules =
      llvm::to_vector(llvm::make_filter_range(
          it->second, [&](const TransformScheduleBase *schedule) {
            return scheduleFilter ? scheduleFilter(schedule->getMnemonic())
                                  : true;
          }));

  // First sort possible generators by benefit. If benefits are equal, then
  // sort by name to ensure that the selection is deterministic.
  llvm::sort(schedules, [](const TransformScheduleBase *lhs,
                           const TransformScheduleBase *rhs) {
    if (lhs->getBenefit() != rhs->getBenefit())
      return lhs->getBenefit() > rhs->getBenefit();
    return lhs->getMnemonic() < rhs->getMnemonic();
  });

  for (const TransformScheduleBase *schedule : schedules) {
    if (schedule->isSupported(op, options))
      return schedule;
  }

  return failure();
}
