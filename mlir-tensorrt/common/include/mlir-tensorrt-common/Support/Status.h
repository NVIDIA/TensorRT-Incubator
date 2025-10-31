//===- Status.h -------------------------------------------------*- C++ -*-===//
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
/// A generic StatusCode, Status, and "StatusOr" object for use where we don't
/// have access to or don't want to use MLIR/LLVM's standard error handling
/// mechanisms.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_SUPPORT_STATUS
#define MLIR_EXECUTOR_SUPPORT_STATUS

#include "mlir-tensorrt-common/Support/ADTExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <string_view>
#include <variant>

namespace mtrt {

#define GEN_ENUM_DECLS
#include "mlir-tensorrt-common/Support/StatusEnums.h.inc"

class [[nodiscard]] Status {
public:
  Status() = default;
  Status(StatusCode code, std::string_view msg = {})
      : code(code),
        message(msg.empty() ? llvm::formatv("{0}", stringifyStatusCode(code))
                            : std::string(msg)) {}
  Status(StatusCode code, const llvm::formatv_object_base &fmtv)
      : code(code), message(fmtv.str()) {}

  /// Return an OK status.
  static Status getOk() { return Status(); }
  /// Returns true if the status does not indicate an error.
  bool isOk() const { return code == StatusCode::Success; }
  /// Returns true if the status indicates an error.
  bool isError() const { return code != StatusCode::Success; }

  /// Returns the code enum.
  StatusCode getCode() const { return code; }

  /// Returns the string representation of the error.
  const std::string &getMessage() const { return message; }

  // For compatability with various macros.
  const Status &getStatus() const { return *this; }
  const Status &checkStatus() const { return *this; }

private:
  StatusCode code{StatusCode::Success};
  std::string message;
};

inline static std::ostream &operator<<(std::ostream &os, const Status &x) {
  return os << x.getMessage();
}
inline static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                            const Status &x) {
  return os << x.getMessage();
}

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
Status getUnimplementedStatus(const char *format, Args &&...args) {
  return Status(StatusCode::Unimplemented,
                llvm::formatv(format, std::forward<Args>(args)...));
}

template <typename... Args>
inline Status getStatusWithMsg(StatusCode code, Args &&...strings) {
  return Status(code, llvm::join_items("", std::forward<Args>(strings)...));
}

template <typename T>
class [[nodiscard]] StatusOr {
public:
  StatusOr() = delete;
  template <typename U>
  StatusOr(U &&payload) : payload(std::forward<U>(payload)) {}

  /// Returns true if the status does not indicate an error.
  bool isOk() const { return std::holds_alternative<T>(payload); }

  /// Returns true if the status indicates an error.
  bool isError() const { return std::holds_alternative<Status>(payload); }

  // clang-format off
  // &-ref qualified (variants called when StatusOr<T> is an lvalue).
  T &getValue() & { return std::get<T>(payload); }
  const T &getValue() const & { return std::get<T>(payload); }
  const Status &getStatus() const & { return std::get<Status>(payload); }
  const T &operator*() const& { return getValue(); }
  T &operator*() &{ return getValue(); }
  // &&-ref qualified (variants called when StatusOr<T> is an rvalue).
  T &&getValue() && { return std::get<T>(std::move(payload)); }
  const T &getValue() const && { return std::get<T>(std::move(payload)); }
  const Status &getStatus() const && { return std::get<Status>(std::move(payload)); }
  const T &operator*() const&& { return getValue(); }
  T &operator*() && { return getValue(); }
  // clang-format on

  T *operator->() { return &getValue(); }
  const T *operator->() const { return &getValue(); }

  /// Returns the Status if error, otherwise an OK status. Since `getStatus()`
  /// will assert when not an error, this is useful in certain macros.
  Status checkStatus() const { return isOk() ? getOkStatus() : getStatus(); }

private:
  /// Holds a tagged union of the payload and the status.
  /// Rationale: A Status object is ~40 bytes on most platforms assuming that
  /// `std::string` is ~32 bytes. The variant size is `max(sizeof(T),
  /// sizeof(Status))`, so considering that often `sizeof(T) > sizeof(Status)`,
  /// it doesn't seem necessary to optimize. If `T` were often very small, then
  /// it might make sense to replace `Status` with `unique_ptr<Status>` or
  /// `shared_ptr<Status>`, but this also introduces a heap allocation in the
  /// error path, which is failable.
  std::variant<T, Status> payload;
};

/// Logs the unhandled errors to the given output stream. Used in cases where
/// there is no reasonable way to handle the error and, besides logging it,
/// ignoring the error and proceeding is the best option.
inline static void logUnhandledErrors(Status status, llvm::raw_ostream &os) {
  if (status.isOk())
    return;
  os << "error: " << status.getMessage() << "\n";
}

/// Handles the error via llvm::report_fatal_error. Used in cases where the
/// error is fatal and the program should terminate.
inline static void cantFail(const Status &status) {
  if (!status.isOk())
    llvm::report_fatal_error(llvm::createStringError(status.getMessage()));
}

/// Handles the error via llvm::report_fatal_error. Used in cases where the
/// error is fatal and the program should terminate.
template <typename T>
void cantFail(const StatusOr<T> &statusOr) {
  return cantFail(statusOr.checkStatus());
}

#define MTRT_CONCAT(x, y) _MTRT_CONCAT(x, y)
#define _MTRT_CONCAT(x, y) x##y

#define MTRT_ASSIGN_OR_RETURN_(statusor, lhs, rexpr)                           \
  auto statusor = (rexpr);                                                     \
  if (statusor.isError())                                                      \
    return statusor.getStatus();                                               \
  lhs = std::move(*statusor);

#define MTRT_ASSIGN_OR_RETURN(lhs, rexpr)                                      \
  MTRT_ASSIGN_OR_RETURN_(MTRT_CONCAT(_status_or_value, __COUNTER__), lhs, rexpr)

#define MTRT_RETURN_IF_ERROR_(tmpName, rexpr)                                  \
  do {                                                                         \
    auto tmpName = (rexpr);                                                    \
    if (!tmpName.isOk())                                                       \
      return tmpName;                                                          \
  } while (false)

#define MTRT_RETURN_IF_ERROR(rexpr)                                            \
  MTRT_RETURN_IF_ERROR_(MTRT_CONCAT(_tmpStatus, __COUNTER__), rexpr)

#define RETURN_ERROR_IF_CUDART_ERROR(x)                                        \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      return ::mtrt::getInternalErrorStatus("{0}:{1} ({2}) {3}", __FILE__,     \
                                            __LINE__, cudaGetErrorName(err),   \
                                            cudaGetErrorString(err));          \
    }                                                                          \
  } while (false);

#define RETURN_ERROR_IF_CUDADRV_ERROR(x)                                       \
  do {                                                                         \
    CUresult err = (x);                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      return ::mtrt::getInternalErrorStatus("{0}:{1} {2}");                    \
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
      return ::mtrt::getInternalErrorStatus(                                   \
          "{0}:{1} {2} ({3}); {4}", __FILE__, __LINE__, msg,                   \
          errName ? errName : "", errStr ? errStr : "");                       \
    }                                                                          \
  } while (false);

/// Causes returning an InternalError status from the current scope if the NCCL
/// result is not ncclSuccess or ncclInProgress.
#define RETURN_ERROR_IF_NCCL_ERROR(x, comm)                                    \
  do {                                                                         \
    ncclResult_t err = (x);                                                    \
    if (err != ncclSuccess && err != ncclInProgress) {                         \
      return getInternalErrorStatus(                                           \
          "{0}:{1} NCCL error [msg=\"{2}\" ncclGetLastError=\"{3}\"]",         \
          __FILE__, __LINE__, ncclGetErrorString(err),                         \
          comm ? ncclGetLastError(comm) : "");                                 \
    }                                                                          \
  } while (false);

#define RETURN_STATUS_IF_ERROR(rexpr)                                          \
  do {                                                                         \
    Status err = (rexpr);                                                      \
    if (!err.isOk()) {                                                         \
      return ::mtrt::getInternalErrorStatus("{0}:{1} {2}", __FILE__, __LINE__, \
                                            err.getMessage());                 \
    }                                                                          \
  } while (false);

#ifndef NDEBUG
#define MTRT_CHECK(cond, msg)                                                  \
  do {                                                                         \
    if (cond) {                                                                \
      llvm::report_fatal_error(msg);                                           \
    }                                                                          \
  } while (false);
#else
#define MTRT_CHECK(cond, msg)
#endif

} // namespace mtrt

#endif // MLIR_EXECUTOR_SUPPORT_STATUS
