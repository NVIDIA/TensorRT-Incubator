//===- SetGPUTarget.cpp ---------------------------------------------------===//
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
/// This file implements the SetGPUTarget pass.
///
//===----------------------------------------------------------------------===//

#include "mlir-executor/Support/DeviceInfo.h"
#include "mlir-kernel/Kernel/IR/Configuration.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir-kernel/Kernel/Transforms/Transforms.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace mlir::kernel {
#define GEN_PASS_DEF_SETGPUTARGETPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace mlir::kernel

using namespace mlir;
using namespace mlir::kernel;

static TargetDeviceSpecAttr
getGpuDeviceSpec(OpBuilder &builder, int64_t maxRegistersPerBlock,
                 int64_t maxSharedMemoryPerBlockKb) {
  return TargetDeviceSpecAttr::get(
      builder.getContext(),
      {DataLayoutEntryAttr::get(
           builder.getStringAttr("maxSharedMemoryPerBlockKb"),
           builder.getI64IntegerAttr(maxSharedMemoryPerBlockKb)),
       DataLayoutEntryAttr::get(
           builder.getStringAttr("maxRegisterPerBlock"),
           builder.getI64IntegerAttr(maxRegistersPerBlock))});
}

static TargetSystemSpecAttr
getSystemSpec(OpBuilder &builder, ArrayRef<TargetDeviceSpecAttr> deviceSpecs) {
  SmallVector<DataLayoutEntryInterface> deviceSpecAttrs;
  for (auto [idx, deviceSpec] : llvm::enumerate(deviceSpecs)) {
    deviceSpecAttrs.push_back(DataLayoutEntryAttr::get(
        builder.getStringAttr(llvm::formatv("GPU:{0}", idx)), deviceSpec));
  }
  return TargetSystemSpecAttr::get(builder.getContext(), deviceSpecAttrs);
}

static NVVM::NVVMTargetAttr getNVVMTarget(OpBuilder &b, Location loc,
                                          StringRef chip, StringRef features,
                                          TargetDeviceSpecAttr deviceSpec) {
  return NVVM::NVVMTargetAttr::getChecked(
      mlir::detail::getDefaultDiagnosticEmitFn(loc), loc->getContext(),
      /*optLevel=*/int(2),
      /*triple=*/StringRef("nvptx64-nvidia-cuda"),
      /*chip=*/chip, /*features=*/features,
      b.getDictionaryAttr(
          {b.getNamedAttr(b.getStringAttr("spec"), deviceSpec)}),
      ArrayAttr{});
}

LogicalResult kernel::setGPUTargets(gpu::GPUModuleOp op,
                                    ArrayRef<Attribute> targets) {
  OpBuilder builder(op.getContext());
  if (op.getTargetsAttr())
    return emitError(op.getLoc()) << "GPU module already has a target spec";
  op.setTargetsAttr(builder.getArrayAttr(targets));
  return success();
}

namespace {
class SetGPUTargetPass
    : public kernel::impl::SetGPUTargetPassBase<SetGPUTargetPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());
    SmallVector<gpu::GPUModuleOp> kernelModules =
        llvm::to_vector(module.getOps<gpu::GPUModuleOp>());

    SmallVector<Attribute> targets;
    SmallVector<TargetDeviceSpecAttr> deviceSpecs;

    if (inferTargetFromHost) {
      mtrt::StatusOr<SmallVector<mtrt::DeviceInfo>> devInfos_ =
          mtrt::getAllDeviceInformationFromHost();
      if (!devInfos_.isOk()) {
        emitError(module.getLoc())
            << "failed to determine host GPU information: "
            << devInfos_.getStatus().getMessage();
        return signalPassFailure();
      }

      for (const mtrt::DeviceInfo &devInfo : *devInfos_) {
        TargetDeviceSpecAttr deviceSpec =
            getGpuDeviceSpec(builder, devInfo.maxRegistersPerBlock,
                             devInfo.maxSharedMemoryPerBlockKb);
        NVVM::NVVMTargetAttr target =
            getNVVMTarget(builder, module.getLoc(),
                          "sm_" + std::to_string(devInfo.computeCapability),
                          features, deviceSpec);
        if (!target)
          return signalPassFailure();
        targets.push_back(target);
        deviceSpecs.push_back(deviceSpec);
      }
    } else {
      TargetDeviceSpecAttr deviceSpec = getGpuDeviceSpec(
          builder, this->maxRegistersPerBlock, this->maxSharedMemoryPerBlockKb);
      NVVM::NVVMTargetAttr target =
          getNVVMTarget(builder, module.getLoc(), chip, features, deviceSpec);
      if (!target)
        return signalPassFailure();
      targets.push_back(target);
      deviceSpecs.push_back(deviceSpec);
    }

    for (gpu::GPUModuleOp kernelModule : kernelModules) {
      if (failed(setGPUTargets(kernelModule, targets)))
        return signalPassFailure();
    }

    // Only populate a host system spec on the top-level module if directly
    // specified. Some compilation pipelines may not currently support DLTI
    // attributes in the final artifact.
    if (populateHostSystemSpec) {
      TargetSystemSpecAttr systemSpec = getSystemSpec(builder, deviceSpecs);
      module->setAttr(DLTIDialect::kTargetSystemDescAttrName, systemSpec);
    }
  };
};
} // namespace
