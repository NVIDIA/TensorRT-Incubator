//===- Passes.td ----------------------------------------------------------===//
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
#ifndef MLIR_KERNEL_KERNEL_TRANSFORMSCHEDULES_PASSES
#define MLIR_KERNEL_KERNEL_TRANSFORMSCHEDULES_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// InitialTransformSchedule
//===----------------------------------------------------------------------===//

def InitialTransformSchedulePass : Pass<"kernel-initial-transform-schedule",
        "::mlir::ModuleOp"> {
  let summary = "generates Transform IR for each func.func containing "
    "linalg operations";

  let description = [{
    This pass generates a Transform IR schedule for each function in the
    module containing 'linalg' operations.

    The caller can specify an attribute name through the 'func-filter' option;
    this restricts the pass to consider only 'func.func' that have an attribute
    keyed by this name (of any attribute type).

    The options prefixed by `device-` specify device properties. They may
    either be specified directly or the `infer-device-opts-from-host`
    flag can be set, which will query the local CUDA device for this
    information.
  }];

  let dependentDialects = [
    "::mlir::kernel::KernelDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::NVVM::NVVMDialect",
    "::mlir::transform::TransformDialect"
  ];
  let options = [
    ListOption<"generatorBenefit", "generator-benefit", "std::string",
      "A list of 'name:benefit' pairs to adjust generator benefits.">,
    Option<"funcFilter", "func-filter", "std::string", "",
      "Specifies on funcs with which attr to generate the transform schedules.">,
    Option<"computeCapability", "device-compute-capability", "int64_t", "80",
      "Specifies the target device compute capability (SM version)">,
    Option<"maxSharedMemoryPerBlockKb", "device-max-smem-per-block", "int64_t", "164",
      "Specifies the maximum amount of shared memory (in kilobytes) allowed per block">,
    Option<"maxRegistersPerBlock", "device-max-registers-per-block",
      "uint64_t", "65536",
      "Specifies the maximum number of 4-byte registers per thread block">,
    Option<"inferDeviceInfoFromHost", "infer-device-opts-from-host", "bool", "false",
      "Infer device information from the host system">
  ];
}

#endif // MLIR_KERNEL_KERNEL_TRANSFORMSCHEDULES_PASSES
