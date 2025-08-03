//===- TensorKindOpInterface.cpp ------------------------------------------===//
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
/// Implementation of TensorKindOpInterface.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Interfaces/TensorKindOpInterface.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

StringRef mlir::stringifyTensorKind(TensorKind kind) {
  switch (kind) {
  case TensorKind::Unknown:
    return "unknown";
  case TensorKind::Both:
    return "both";
  case TensorKind::Device:
    return "device";
  case TensorKind::Host:
    return "host";
  }
  llvm_unreachable("unknown TensorKind");
}

StringRef mlir::getHostTensorArgAttrName() { return "tensorrt.host_tensor"; }

//===----------------------------------------------------------------------===//
// TensorKindInfo
//===----------------------------------------------------------------------===//

void TensorKindInfo::print(raw_ostream &os) const {
  if (isUninitialized()) {
    os << "<<uninitialized>>";
    return;
  }
  os << stringifyTensorKind(*kind);
}

TensorKindInfo TensorKindInfo::join(const TensorKindInfo &lhs,
                                    const TensorKindInfo &rhs) {
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  if (lhs < rhs)
    return rhs;
  if (rhs < lhs)
    return lhs;
  if (rhs == lhs)
    return lhs;
  return TensorKind::Both;
}

TensorKindInfo TensorKindInfo::meet(const TensorKindInfo &lhs,
                                    const TensorKindInfo &rhs) {
  return join(lhs, rhs);
}

static constexpr int64_t kSmallTensorThresholdElements = 8;

bool detail::isHostTensorCandidate(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return tensorType.hasStaticShape() &&
           tensorType.getNumElements() <= kSmallTensorThresholdElements;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// TensorKindOpInterface & TensorKindAttrInterface Definitions
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Interfaces/TensorKindAttrInterface.cpp.inc"
#include "mlir-tensorrt-common/Interfaces/TensorKindOpInterface.cpp.inc"
