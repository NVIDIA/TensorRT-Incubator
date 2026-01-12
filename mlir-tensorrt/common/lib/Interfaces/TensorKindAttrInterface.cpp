//===- TensorKindAttrInterface.cpp ----------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Implementation of TensorKindAttrInterface.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Interfaces/TensorKindAttrInterface.h"
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

TensorKindInfo::TensorKindInfo() : kind(std::nullopt) {}

TensorKindInfo::TensorKindInfo(TensorKind initialKind) : kind(initialKind) {}

void TensorKindInfo::print(raw_ostream &os) const {
  if (isUninitialized()) {
    os << "<<uninitialized>>";
    return;
  }
  os << stringifyTensorKind(*kind);
}

ChangeResult TensorKindInfo::setKind(TensorKind kind) {
  auto newKind = kind;
  ChangeResult result = isUninitialized() || newKind != *this->kind
                            ? ChangeResult::Change
                            : ChangeResult::NoChange;
  this->kind = newKind;
  return result;
}

bool TensorKindInfo::isUninitialized() const { return !kind.has_value(); }

bool TensorKindInfo::isUnknown() const { return *kind == TensorKind::Unknown; }

TensorKind TensorKindInfo::getKind() const {
  assert(!isUninitialized());
  return *kind;
}

bool TensorKindInfo::isHostOnly() const {
  assert(!isUninitialized() && "expected initialized value");
  return getKind() == TensorKind::Host;
}

bool TensorKindInfo::isHostVisible() const {
  assert(!isUninitialized() && "expected initialized value");
  return getKind() == TensorKind::Host || getKind() == TensorKind::Both;
}

bool TensorKindInfo::isDeviceOnly() const {
  assert(!isUninitialized() && "expected initialized value");
  return getKind() == TensorKind::Device;
}

bool TensorKindInfo::isDeviceVisible() const {
  assert(!isUninitialized() && "expected initialized value");
  return getKind() == TensorKind::Device || getKind() == TensorKind::Both;
}

bool TensorKindInfo::isBothHostAndDevice() const {
  assert(!isUninitialized() && "expected initialized value");
  return getKind() == TensorKind::Both;
}

bool TensorKindInfo::operator<(const TensorKindInfo &other) const {
  // Uninitialized value is always less than other.
  if (isUninitialized())
    return true;
  // Uninitialized other can't be greater than anything.
  if (other.isUninitialized())
    return false;
  // If we are unknown, we are less than any host/device/both.
  if (isUnknown())
    return static_cast<int32_t>(getKind()) <
           static_cast<int32_t>(other.getKind());
  // If we are known, we are less than 'both'.
  if ((isHostOnly() || isDeviceOnly()) && other.isBothHostAndDevice())
    return true;
  return false;
}

bool TensorKindInfo::operator==(const TensorKindInfo &other) const {
  return other.kind == this->kind;
}

bool TensorKindInfo::operator!=(const TensorKindInfo &other) const {
  return !(*this == other);
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
  return TensorKindInfo(TensorKind::Both);
}

TensorKindInfo TensorKindInfo::meet(const TensorKindInfo &lhs,
                                    const TensorKindInfo &rhs) {
  return join(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// TensorKindOpInterface & TensorKindAttrInterface Definitions
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Interfaces/TensorKindAttrInterface.cpp.inc"
