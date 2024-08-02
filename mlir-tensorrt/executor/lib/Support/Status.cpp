//===- Status.cpp  --------------------------------------------------------===//
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
/// Definitions for StatusCode, Status, and StatusOr.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Support/Status.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>

namespace mlirtrt {
#define GEN_ENUM_DEFS
#include "mlir-executor/Support/StatusEnums.h.inc"
} // namespace mlirtrt

using namespace mlirtrt;

Status::Status(StatusCode code, std::string_view msg)
    : code(code), additionalMsg(msg.empty() ? std::nullopt
                                            : std::optional<std::string>(msg)) {
}
Status::Status(const Status &other)
    : code(other.getCode()),
      additionalMsg(
          other.getAdditionalMsg().empty()
              ? std::nullopt
              : std::optional<std::string>(other.getAdditionalMsg())) {}

Status::Status(StatusCode code, const llvm::formatv_object_base &message)
    : code(code), additionalMsg(std::string(message)) {}

Status Status::getOk() { return StatusCode(StatusCode::Success); }

bool Status::isOk() const { return code == StatusCode::Success; }

bool Status::isError() const { return code != StatusCode::Success; }

StatusCode Status::getCode() const { return code; }

std::string Status::getString() const {
  return additionalMsg ? llvm::formatv("{0}: {1}", stringifyStatusCode(code),
                                       *additionalMsg)
                             .str()
                       : std::string(stringifyStatusCode(code));
}

std::ostream &mlirtrt::operator<<(std::ostream &os, const Status &x) {
  return os << x.getString();
}
