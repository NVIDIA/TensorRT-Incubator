//===- Status.h -------------------------------------------------*- C++ -*-===//
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
/// A generic StatusCode, Status, and "StatusOr" object for use where we don't
/// have access to or don't want to use MLIR/LLVM's standard error handling
/// mechanisms.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_SUPPORT_STATUS_H
#define MLIR_TENSORRT_SUPPORT_STATUS_H

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>

namespace mlirtrt {

#define GEN_ENUM_DECLS
#include "mlir-executor/Support/StatusEnums.h.inc"

class Status {
public:
  Status() = delete;
  Status(const Status &);
  Status &operator=(const Status &);
  Status(Status &&) = default;
  Status &operator=(Status &&) = default;
  Status(StatusCode code, std::string_view additionalMsg = "");
  Status(StatusCode code, const llvm::formatv_object_base &);

  /// Return an OK status.
  static Status getOk();
  /// Returns true if the status does not indicate an error.
  bool isOk() const;
  /// Returns true if the status indicates an error.
  bool isError() const;

  /// Returns the code enum.
  StatusCode getCode() const;

  /// Returns the string representation of the code.
  std::string getString() const;

  const Status &getStatus() const { return *this; }

  /// Returns the additional payload message, if it exists.
  std::string_view getAdditionalMsg() const {
    return additionalMsg ? std::string_view(*additionalMsg) : "";
  }

private:
  StatusCode code;
  std::optional<std::string> additionalMsg{};
};

std::ostream &operator<<(std::ostream &os, const Status &x);

//===----------------------------------------------------------------------===//
// Convenience Builders for Status objects
//===----------------------------------------------------------------------===//
inline Status getOkStatus() { return Status(StatusCode::Success); }

template <typename... Args>
Status getInvalidArgStatus(const char *format, Args &&...args) {
  return Status(StatusCode::InvalidArgument,
                llvm::formatv(format, std::forward<Args>(args)...));
}

template <typename... Args>
Status getInternalErrorStatus(const char *format, Args &&...args) {
  return Status(StatusCode::InternalError,
                llvm::formatv(format, std::forward<Args>(args)...));
}

template <typename... Args>
inline Status getStatusWithMsg(StatusCode code, Args &&...strings) {
  return Status(code, llvm::join_items("", std::forward<Args>(strings)...));
}

template <typename T>
class StatusOr {
public:
  StatusOr() = delete;
  StatusOr(T &&payload)
      : status(Status::getOk()), payload(std::forward<T>(payload)) {}
  StatusOr(const StatusOr<T> &) = delete;
  StatusOr(StatusOr<T> &&) = default;
  StatusOr(Status &&status)
      : status(std::forward<Status>(status)), payload(std::nullopt) {
    assert(this->status.isError() && "expected error status");
  }
  StatusOr(const Status &status) : status(status), payload(std::nullopt) {
    assert(this->status.isError() && "expected error status");
  }

  StatusOr<T> &operator=(const Status &status) = delete;
  StatusOr<T> &operator=(Status &&status) = delete;

  template <typename S, typename std::enable_if<
                            std::is_base_of<T, S>::value>::type = nullptr>
  StatusOr(S &&payload)
      : status(Status::getOk()), payload(std::forward<S>(payload)) {}

  /// Returns true if the status does not indicate an error.
  bool isOk() const { return status.isOk() && payload; }

  /// Returns true if the status indicates an error.
  bool isError() const { return !isOk() || !payload; }

  /// Returns the status object.
  const Status &getStatus() const { return status; }

  /// Returns the string representation of the status object.
  std::string getString() const { return status.getString(); }

  /// Returns underlying payload and asserts no error.
  const T *operator->() const {
    assert(isOk() && "expected valid payload and no error");
    return &*payload;
  }

  /// Returns underlying payload and asserts no error.
  T *operator->() {
    assert(isOk() && "expected valid payload and no error");
    return &*payload;
  }

  /// Returns underlying payload and asserts no error.
  const T &operator*() const {
    assert(isOk() && "expected valid payload and no error");
    return *payload;
  }

  /// Returns underlying payload and asserts no error.
  T &operator*() {
    assert(isOk() && "expected valid payload and no error");
    return *payload;
  }

private:
  Status status;
  std::optional<T> payload;
};

template <typename T>
auto fmtRange(llvm::ArrayRef<T> r) {
  return llvm::make_range(r.begin(), r.end());
}

#define MTRT_CONCAT(x, y) _MTRT_CONCAT(x, y)
#define _MTRT_CONCAT(x, y) x##y

#define MTRT_ASSIGN_OR_RETURN(lhs, rexpr)                                      \
  MTRT_ASSIGN_OR_RETURN_(MTRT_CONCAT(_status_or_value, __COUNTER__), lhs, rexpr)

#define MTRT_ASSIGN_OR_RETURN_(statusor, lhs, rexpr)                           \
  auto statusor = (rexpr);                                                     \
  if (statusor.isError())                                                      \
    return statusor.getStatus();                                               \
  lhs = std::move(*statusor);

#define MTRT_RETURN_IF_ERROR(rexpr)                                            \
  MTRT_RETURN_IF_ERROR_(MTRT_CONCAT(_tmpStatus, __COUNTER__), rexpr)
#define MTRT_RETURN_IF_ERROR_(tmpName, rexpr)                                  \
  do {                                                                         \
    auto tmpName = (rexpr);                                                    \
    if (!tmpName.isOk())                                                       \
      return tmpName;                                                          \
  } while (false)

#define RETURN_ERROR_IF_CUDART_ERROR(x)                                        \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      std::stringstream ss;                                                    \
      ss << __FILE__ << ":" << __LINE__ << " " << err;                         \
      return ::mlirtrt::getInternalErrorStatus(                                \
          "{0}:{1} ({2}) {3}", __FILE__, __LINE__, cudaGetErrorName(err),      \
          cudaGetErrorString(err));                                            \
    }                                                                          \
  } while (false);

#define RETURN_ERROR_IF_CUDADRV_ERROR(x)                                       \
  do {                                                                         \
    CUresult err = (x);                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      std::stringstream ss;                                                    \
      ss << __FILE__ << ":" << __LINE__ << " " << err;                         \
      return ::mlirtrt::getInternalErrorStatus("{0}:{1} {2}");                 \
    }                                                                          \
  } while (false);

#define RETURN_ERROR_WITH_MSG_IF_CUDADRV_ERROR(x, msg)                         \
  do {                                                                         \
    CUresult err = (x);                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *errName;                                                     \
      cuGetErrorName(err, &errName);                                           \
      const char *errStr;                                                      \
      cuGetErrorString(err, &errStr);                                          \
      return ::mlirtrt::getInternalErrorStatus(                                \
          "{0}:{1} {2} ({3}); {4}", __FILE__, __LINE__, msg,                   \
          errName ? errName : "", errStr ? errStr : "");                       \
    }                                                                          \
  } while (false);

#define RETURN_ERROR_IF_NCCL_ERROR(x, comm)                                    \
  do {                                                                         \
    ncclResult_t err = (x);                                                    \
    if (err != ncclSuccess) {                                                  \
      std::stringstream ss;                                                    \
      return getInternalErrorStatus(                                           \
          "{0}:{1} NCCL error [msg=\"{2}\" ncclGetLastError=\"{3}\"]",         \
          __FILE__, __LINE__, ncclGetErrorString(err),                         \
          comm ? ncclGetLastError(comm) : "");                                 \
    }                                                                          \
  } while (false);

#define RETURN_STATUS_IF_ERROR(rexpr)                                          \
  do {                                                                         \
    auto err = (rexpr);                                                        \
    if (!err.isOk()) {                                                         \
      return ::mlirtrt::getInternalErrorStatus("{0}:{1} {3}", __FILE__,        \
                                               __LINE__, err.getString());     \
    }                                                                          \
  } while (false);

} // namespace mlirtrt

#endif // MLIR_TENSORRT_SUPPORT_STATUS_H
