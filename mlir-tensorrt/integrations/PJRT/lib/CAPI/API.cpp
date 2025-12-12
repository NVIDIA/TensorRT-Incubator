//===- API.cpp ------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of PJRT API Interface -- the organization of this file
/// approximately tracks `xla/pjrt/c/pjrt_c_wrapper_impl.cc` to ease
/// maintenance. The order the list ing of functions should correspond to that
/// file.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-pjrt/CAPI/API.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/API/ExecutableFlatbuffer.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt-pjrt/Client.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include <sstream>

using namespace mtrt;
using namespace mtrt::pjrt;

template <typename ToTy, typename FromTy>
static ToTy *AsImpl(FromTy *from) {
  return reinterpret_cast<ToTy *>(from);
}

template <typename ToTy, typename FromTy>
static const ToTy *AsConstImpl(const FromTy *from) {
  return reinterpret_cast<const ToTy *>(from);
}

/// Check struct size is valid and log entry to the API function.
/// Returns an error if struct_size is smaller than expected.
#define PJRT_RETURN_IF_STRUCT_SIZE_ERROR(args_type, args)                      \
  do {                                                                         \
    PJRT_DBGF("%s", "Entered API call: " #args_type);                          \
    if (args->struct_size < args_type##_STRUCT_SIZE) {                         \
      return PJRT_Error::allocateInvalidArgumentError(                         \
          "struct size mismatch for " #args_type);                             \
    }                                                                          \
  } while (false)

/// Check struct size is valid and log entry to the API function (void version).
#define PJRT_CHECK_STRUCT_SIZE_VOID(args_type, args)                           \
  do {                                                                         \
    PJRT_DBGF("%s", "Entered API call: " #args_type);                          \
  } while (false)

/// Represents an error message with an arbitrary payload.
struct PJRT_Error {
  PJRT_Error() = delete;

  /// Constructs an error from a Status. Asserts that the status has an error
  /// state.
  PJRT_Error(Status &&status);

  /// Constructs an error from a Status. Asserts that the status has an error
  /// state.
  PJRT_Error(const Status &status);

  /// Allocate a new error and return it as a PJRT_Error*. Caller takes
  /// ownership.
  template <typename... Args>
  static PJRT_Error *allocateFromStatus(Args &&...status) {
    Status s(std::forward<Args>(status)...);
    assert(s.isError() && "expected status with an error");
    PJRT_DBGF("[error] %s", s.getMessage().c_str());
    return new PJRT_Error(std::move(s));
  }

  /// Allocate a new 'unimplemented' error and return it as a PJRT_Error*.
  /// Caller takes ownership.
  template <typename... Args>
  static PJRT_Error *allocateUnimplemented(Args &&...status) {
    return new PJRT_Error(getStatusWithMsg(StatusCode::Unimplemented,
                                           std::forward<Args>(status)...));
  }

  /// Allocate a new 'internal' error and return it as a PJRT_Error*.
  /// Caller takes ownership.
  template <typename... Args>
  static PJRT_Error *allocateInternalError(Args &&...status) {
    return new PJRT_Error(getStatusWithMsg(StatusCode::InternalError,
                                           std::forward<Args>(status)...));
  }

  /// Allocate a new 'invalid argument' error and return it as a PJRT_Error*.
  /// Caller takes ownership.
  template <typename... Args>
  static PJRT_Error *allocateInvalidArgumentError(Args &&...status) {
    return new PJRT_Error(getStatusWithMsg(StatusCode::InvalidArgument,
                                           std::forward<Args>(status)...));
  }

  /// Returns the PJRT_Error API representation of 'no error'. Currently this is
  /// a nullptr.
  static PJRT_Error *getOk() { return nullptr; }

  /// Returns the status error message.
  const std::string &getMessage() const { return status.getMessage(); }

  /// Returns the PJRT error code.
  PJRT_Error_Code getCode() const;

private:
  Status status;
};

PJRT_Error::PJRT_Error(Status &&status) : status(std::forward<Status>(status)) {
  assert(this->status.isError() && "expected status with an error");
}

PJRT_Error::PJRT_Error(const Status &status) : status(status) {
  assert(this->status.isError() && "expected status with an error");
}

PJRT_Error_Code PJRT_Error::getCode() const {
  switch (status.getCode()) {
  case StatusCode::InternalError:
    return PJRT_Error_Code_INTERNAL;
  case StatusCode::Unimplemented:
    return PJRT_Error_Code_UNIMPLEMENTED;
  case StatusCode::InvalidArgument:
    return PJRT_Error_Code_INVALID_ARGUMENT;
  case StatusCode::Unknown:
  default:
    return PJRT_Error_Code_UNKNOWN;
  }
}

//===----------------------------------------------------------------------===//
// Buffer helpers
//===----------------------------------------------------------------------===//

/// Given a PJRT element type, return a MLIR-TensorRT element type if possible.
/// Otherwise, return error.
static StatusOr<mtrt::ScalarType>
getElementTypeFromPJRTElementType(PJRT_Buffer_Type type) {
#define HANDLE_CASE(x, y)                                                      \
  case PJRT_Buffer_Type::PJRT_Buffer_Type_##x:                                 \
    return mtrt::ScalarType(mtrt::ScalarTypeCode::y);

  switch (type) {
    HANDLE_CASE(S4, i4)
    HANDLE_CASE(U4, i4)
    HANDLE_CASE(S8, i8)
    HANDLE_CASE(U8, i8)
    HANDLE_CASE(S16, i16)
    HANDLE_CASE(U16, i16)
    HANDLE_CASE(S32, i32)
    HANDLE_CASE(U32, i32)
    HANDLE_CASE(S64, i64)
    HANDLE_CASE(U64, i64)
    HANDLE_CASE(F16, f16)
    HANDLE_CASE(F32, f32)
    HANDLE_CASE(F64, f64)
    HANDLE_CASE(BF16, bf16)
    HANDLE_CASE(PRED, i1)
    HANDLE_CASE(C64, complex32)
    HANDLE_CASE(C128, complex64)
    HANDLE_CASE(F4E2M1FN, f4e2m1fn)
    HANDLE_CASE(F8E4M3FN, f8e4m3fn)
  case PJRT_Buffer_Type_S2:
  case PJRT_Buffer_Type_U2:
  case PJRT_Buffer_Type_INVALID:
  case PJRT_Buffer_Type_F8E5M2:
  case PJRT_Buffer_Type_F8E4M3B11FNUZ:
  case PJRT_Buffer_Type_F8E5M2FNUZ:
  case PJRT_Buffer_Type_F8E4M3FNUZ:
  case PJRT_Buffer_Type_TOKEN:
  case PJRT_Buffer_Type_F8E8M0FNU:
  case PJRT_Buffer_Type_F8E3M4:
  case PJRT_Buffer_Type_F8E4M3:
    return mtrt::getInvalidArgStatus("unimplemented conversion from "
                                     "PJRT_Buffer_Type ({0}) to ScalarDataType",
                                     static_cast<int>(type));
  }
#undef HANDLE_CASE
}

/// Given a MLIR-TensorRT element type, return a PJRT element type if possible.
/// Otherwise, return error.
static StatusOr<PJRT_Buffer_Type>
getPjrtBufferTypeFromElementType(mtrt::ScalarType type) {
#define HANDLE_CASE(x, y)                                                      \
  case mtrt::ScalarTypeCode::x:                                                \
    return PJRT_Buffer_Type::PJRT_Buffer_Type_##y;
  switch (type.getCode()) {
    HANDLE_CASE(i1, PRED)
    HANDLE_CASE(i4, S4)
    HANDLE_CASE(i8, S8)
    HANDLE_CASE(ui8, U8)
    HANDLE_CASE(i16, S16)
    HANDLE_CASE(i32, S32)
    HANDLE_CASE(i64, S64)
    HANDLE_CASE(f32, F32)
    HANDLE_CASE(f64, F64)
    HANDLE_CASE(f16, F16)
    HANDLE_CASE(bf16, BF16)
    HANDLE_CASE(complex32, C64)
    HANDLE_CASE(complex64, C128)
    HANDLE_CASE(f8e4m3fn, F8E4M3FN)
    HANDLE_CASE(f4e2m1fn, F4E2M1FN)
  case mtrt::ScalarTypeCode::unknown:
    return getStatusWithMsg(StatusCode::InternalError,
                            "unimplemented conversion from MLIR-TRT "
                            "ScalarDataType ({0}) to PJRT_Buffer_Type",
                            mtrt::flat::EnumNameScalarTypeCode(type));
  }
#undef HANDLE_CASE
}

//===----------------------------------------------------------------------===//
// Error Helpers
//===----------------------------------------------------------------------===//

#define PJRT_RETURN_ERROR_IF_CUDART_ERROR(e)                                   \
  do {                                                                         \
    if (e != cudaSuccess) {                                                    \
      PJRT_DBGF("CUDA Runtime error with name: %s", cudaGetErrorName(e));      \
      return PJRT_Error::allocateInternalError(                                \
          "CUDA Runtime error -- see logs");                                   \
    }                                                                          \
  } while (false)

//===----------------------------------------------------------------------===//
// PJRT API Implementation - All functions in namespace to avoid conflicts
// with typedef'd function types in pjrt_c_api.h
//===----------------------------------------------------------------------===//
namespace mtrt::pjrt {

//===----------------------------------------------------------------------===//
// PJRT Error API Implementation
//===----------------------------------------------------------------------===//

void PJRT_Error_Destroy(PJRT_Error_Destroy_Args *args) {
  PJRT_CHECK_STRUCT_SIZE_VOID(PJRT_Error_Destroy_Args, args);
  delete args->error;
}

void PJRT_Error_Message(PJRT_Error_Message_Args *args) {
  PJRT_CHECK_STRUCT_SIZE_VOID(PJRT_Error_Message_Args, args);
  assert(args && args->error && "expected valid error with error message");
  const PJRT_Error *error = args->error;
  args->message = error->getMessage().c_str();
  args->message_size = error->getMessage().size();
}

PJRT_Error *PJRT_Error_GetCode(PJRT_Error_GetCode_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Error_GetCode_Args, args);
  args->code = args->error->getCode();
  return PJRT_Error::getOk();
}

//===----------------------------------------------------------------------===//
// PJRT Plugin API Implementation
//===----------------------------------------------------------------------===//

PJRT_Error *PJRT_Plugin_Initialize(PJRT_Plugin_Initialize_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Plugin_Initialize_Args, args);
  return PJRT_Error::getOk();
}

/// Returns an array of plugin attributes which are key-value pairs. One
/// example attribute is the minimum supported StableHLO version.
PJRT_Error *PJRT_Plugin_Attributes(PJRT_Plugin_Attributes_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Plugin_Attributes_Args, args);
  args->num_attributes = 0;
  args->attributes = nullptr;
  return PJRT_Error::getOk();
}

//===----------------------------------------------------------------------===//
// PJRT Event API Implementation
//===----------------------------------------------------------------------===//

// PJRT Doc: Frees `args->event`. `event` can be `nullptr`.
PJRT_Error *PJRT_Event_Destroy(PJRT_Event_Destroy_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Event_Destroy_Args, args);
  assert(args->event != nullptr);
  std::unique_ptr<Event> event(AsImpl<Event>(args->event));
  if (event)
    Event::releaseWhenReady(std::move(event));
  return PJRT_Error::getOk();
}

// PJRT Doc: Returns true if this PJRT_Event has completed, including if an
// error has occurred.
PJRT_Error *PJRT_Event_IsReady(PJRT_Event_IsReady_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Event_IsReady_Args, args);
  Event *event = AsImpl<Event>(args->event);
  if (!event)
    return PJRT_Error::getOk();
  args->is_ready = event->checkIsReady();
  return PJRT_Error::getOk();
}

// PJRT Doc:
// Should only be called if PJRT_Event_IsReady returns true.
// Returns `nullptr` if there is no error.
// The returned error should be freed with `PJRT_Error_Destroy`.
//
// If `PJRT_Event_Await` has been called, this will return a pointer to an
// identical error status as that call, as will subsequent calls to
// `PJRT_Event_Error`. However, each of these `PJRT_Error *` pointers are
// independent of `PJRT_Error *`s returned by other function calls, so they
// must each be freed separately using `PJRT_Error_Destroy`.
PJRT_Error *PJRT_Event_Error(PJRT_Event_Error_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Event_Error_Args, args);
  Status s = AsImpl<Event>(args->event)->getStatus();
  return s.isOk() ? PJRT_Error::getOk() : PJRT_Error::allocateFromStatus(s);
}

// PJRT Doc:
// Blocks the calling thread until `event` is ready, then returns the error
// status (with `nullptr` indicating no error). The returned status should be
// freed with `PJRT_Error_Destroy`.
PJRT_Error *PJRT_Event_Await(PJRT_Event_Await_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Event_Await_Args, args);
  return PJRT_Error::allocateUnimplemented("PJRT_Event_Await is unimplemented");
}

// PJRT Doc:
// Registers `callback` to be called once `event` is ready, with `event`'s
// error status and a pointer to an object of the caller's choice as
// arguments.
PJRT_Error *PJRT_Event_OnReady(PJRT_Event_OnReady_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Event_OnReady_Args, args);
  Event *event = AsImpl<Event>(args->event);
  PJRT_Event_OnReadyCallback callback = args->callback;
  event->addReadyCallback(
      [callback](Status status, void *userData) {
        callback(status.isOk() ? nullptr
                               : PJRT_Error::allocateFromStatus(status),
                 userData);
      },
      args->user_arg);
  return PJRT_Error::getOk();
}

//===----------------------------------------------------------------------===//
// PJRT Buffer API Implementation
//===----------------------------------------------------------------------===//

PJRT_Error *PJRT_Buffer_Destroy(PJRT_Buffer_Destroy_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_Destroy_Args, args);
  DeviceBufferDescriptor *buffer = AsImpl<DeviceBufferDescriptor>(args->buffer);
  if (buffer == nullptr)
    return PJRT_Error::getOk();
  delete buffer;
  return PJRT_Error::getOk();
}

/// PJRT Doc: Returns the type of the array elements of a buffer.
PJRT_Error *PJRT_Buffer_ElementType(PJRT_Buffer_ElementType_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_ElementType_Args, args);
  const DeviceBufferDescriptor *buffer =
      AsImpl<DeviceBufferDescriptor>(args->buffer);
  StatusOr<PJRT_Buffer_Type> elType = getPjrtBufferTypeFromElementType(
      buffer->getType().getElementType().getCode());
  if (!elType.isOk())
    return PJRT_Error::allocateFromStatus(elType.getStatus());
  args->type = *elType;
  return PJRT_Error::getOk();
}

/// PJRT Doc: Returns the array shape of `buffer`, i.e. the size of each
/// dimension.
PJRT_Error *PJRT_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_Dimensions_Args, args);
  const DeviceBufferDescriptor *buffer =
      AsImpl<DeviceBufferDescriptor>(args->buffer);
  args->dims = buffer->getType().getShape().data();
  args->num_dims = buffer->getType().getShape().size();
  return PJRT_Error::getOk();
}

/// PJRT Doc: Returns the unpadded array shape of `buffer`. This usually is
/// equivalent to PJRT_Buffer_Dimensions, but for implementations that support
/// dynamically-sized dimensions via padding to a fixed size, any dynamic
/// dimensions may have a smaller unpadded size than the padded size reported
/// by PJRT_Buffer_Dimensions. ("Dynamic" dimensions are those whose length is
/// only known at runtime, vs. "static" dimensions whose size is fixed at
/// compile time.)
PJRT_Error *
PJRT_Buffer_UnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_UnpaddedDimensions_Args, args);
  DeviceBufferDescriptor *buffer = AsImpl<DeviceBufferDescriptor>(args->buffer);
  args->unpadded_dims = buffer->getType().getShape().data();
  args->num_dims = buffer->getType().getShape().size();
  return PJRT_Error::getOk();
}

/// PJRT Doc: Returns the indices of dynamically-sized dimensions, or an empty
/// list if all dimensions are static. ("Dynamic" dimensions are those whose
/// length is only known at runtime, vs. "static" dimensions whose size is
/// fixed at compile time.)
/// NOTE: Currently we do not support dynamic dimensions.
PJRT_Error *PJRT_Buffer_DynamicDimensionIndices(
    PJRT_Buffer_DynamicDimensionIndices_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_DynamicDimensionIndices_Args,
                                   args);
  args->num_dynamic_dims = 0;
  args->dynamic_dim_indices = nullptr;
  return PJRT_Error::allocateUnimplemented("Buffer_DynamicDimensionIndices");
}

/// PJRT Doc: Returns the memory layout of the data in this buffer.
PJRT_Error *
PJRT_Buffer_GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_GetMemoryLayout_Args, args);
  DeviceBufferDescriptor *buffer = AsImpl<DeviceBufferDescriptor>(args->buffer);
  // We need to cast away const because the minor-to-major ordering performs
  // lazy construction and therefore is non-const.
  BufferType &type = const_cast<BufferType &>(buffer->getType());
  args->layout.type =
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled;
  args->layout.tiled.minor_to_major_size = type.getRank();
  args->layout.tiled.minor_to_major = buffer->getMinorToMajorOrdering().data();
  args->layout.tiled.minor_to_major_size =
      buffer->getMinorToMajorOrdering().size();
  args->layout.tiled.num_tiles = 0;
  return PJRT_Error::getOk();
}

// PJRT Doc:
// Gets the number of bytes of the buffer storage on the device
PJRT_Error *
PJRT_Buffer_OnDeviceSizeInBytes(PJRT_Buffer_OnDeviceSizeInBytes_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_OnDeviceSizeInBytes_Args, args);
  DeviceBufferDescriptor *buffer = AsImpl<DeviceBufferDescriptor>(args->buffer);
  const BufferType &type = buffer->getType();
  args->on_device_size_in_bytes = type.getFootprintSizeInBytes();
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Buffer_Device(PJRT_Buffer_Device_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_Device_Args, args);
  const DeviceBufferDescriptor *buffer =
      AsImpl<DeviceBufferDescriptor>(args->buffer);
  args->device = reinterpret_cast<PJRT_Device *>(&buffer->getDevice());
  return PJRT_Error::getOk();
}

/// Returns the memory space associated with the buffer.
PJRT_Error *PJRT_Buffer_Memory(PJRT_Buffer_Memory_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_Memory_Args, args);
  args->memory = reinterpret_cast<PJRT_Memory *>(
      AsImpl<const DeviceBufferDescriptor>(args->buffer)->getMemorySpace());
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Buffer_Delete(PJRT_Buffer_Delete_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_Delete_Args, args);
  DeviceBufferDescriptor *buffer = AsImpl<DeviceBufferDescriptor>(args->buffer);
  Status status = buffer->freeBufferAsync();
  if (!status.isOk())
    return PJRT_Error::allocateFromStatus(status);
  return PJRT_Error::getOk();
}

// PJRT Doc:
// True if and only if PJRT_Buffer_Delete has previously been called.
PJRT_Error *PJRT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_IsDeleted_Args, args);
  DeviceBufferDescriptor *buffer = AsImpl<DeviceBufferDescriptor>(args->buffer);
  args->is_deleted = buffer->isScheduledForDeletion();
  return PJRT_Error::getOk();
}

// PJRT Doc:
// Copies the buffer to device `dst_device`. Caller is responsible for freeing
// returned `dst_buffer` with PJRT_Buffer_Destroy. Returns an error if the
// buffer is already on `dst_device`.
PJRT_Error *PJRT_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_CopyToDevice_Args, args);
  DeviceBufferDescriptor *buffer = AsImpl<DeviceBufferDescriptor>(args->buffer);
  PjRtDevice *dstDevice = AsImpl<PjRtDevice>(args->dst_device);

  std::unique_ptr<Event> copyDoneEvent = nullptr;
  StatusOr<std::unique_ptr<DeviceBufferDescriptor>> dstBuffer =
      dstDevice->getClient()->copyDeviceBufferToOtherDevice(*dstDevice, *buffer,
                                                            copyDoneEvent);
  if (!dstBuffer.isOk())
    return PJRT_Error::allocateFromStatus(dstBuffer.getStatus());

  // This API does not return an event.
  Status waitStatus = copyDoneEvent->waitForReady();
  if (!waitStatus.isOk())
    return PJRT_Error::allocateFromStatus(waitStatus);
  Event::releaseWhenReady(std::move(copyDoneEvent));

  // Set outputs
  args->dst_buffer = reinterpret_cast<PJRT_Buffer *>((*dstBuffer).release());
  return PJRT_Error::getOk();
}

// PJRT Doc:
// Asynchronously copies the buffer's value into a preallocated host buffer.
PJRT_Error *PJRT_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_ToHostBuffer_Args, args);
  // We put "nullptr" here to signal that the host buffer should have the same
  // layout has the device buffer.
  args->host_layout = nullptr;

  DeviceBufferDescriptor *buffer = AsImpl<DeviceBufferDescriptor>(args->src);
  BufferType t = buffer->getType();
  // A null destination pointer means we just return the size required.
  // Right now, we only support canonical row-major layouts, so all buffers
  // are fully backed (i.e. no padding) on host and device.
  if (args->dst == nullptr) {
    args->dst_size = t.getFootprintSizeInBytes();
    PJRT_DBGF("returning required buffer size of %lu bytes", args->dst_size);
    return PJRT_Error::getOk();
  }
  if (args->dst_size < t.getFootprintSizeInBytes()) {
    std::stringstream ss;
    ss << "host buffer has " << args->dst_size << " bytes < "
       << t.getFootprintSizeInBytes() << " required for buffer of type "
       << t.toString();
    PJRT_DBGF("%s", ss.str().c_str());
    return PJRT_Error::allocateInvalidArgumentError(ss.str());
  }

  // Copy to the host buffer.
  std::unique_ptr<Event> copyDoneEvent{nullptr};
  Status s = buffer->getDevice().getClient()->copyDeviceBufferToHost(
      buffer->getDevice(), *buffer, args->dst, t.getFootprintSizeInBytes(),
      &copyDoneEvent);
  if (!s.isOk())
    return PJRT_Error::allocateFromStatus(s.getStatus());

  args->event = reinterpret_cast<PJRT_Event *>(copyDoneEvent.release());

  return PJRT_Error::getOk();
}

// PJRT Doc:
// Whether this buffer is on CPU and thus allows for certain optimizations.
PJRT_Error *PJRT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_IsOnCpu_Args, args);
  args->is_on_cpu = false;
  return PJRT_Error::getOk();
}

// PJRT Doc:
// Returns an event that is triggered when either of the following happens:
// * the data in the PJRT_Buffer becomes ready, or
// * an error has occurred.
//
// TODO(b/241967811): change these weird semantics
// If the buffer has been deleted or donated, the returned event will
// immediately indicate an error. However, if PJRT_Buffer_ReadyEvent() is
// called on the buffer before PJRT_Buffer_Delete() is, the returned event
// will not transition to an error state after PJRT_Buffer_Delete() is called.
PJRT_Error *PJRT_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_ReadyEvent_Args, args);
  args->event =
      reinterpret_cast<PJRT_Event *>(Event::createReadyEvent().release());
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Buffer_UnsafePointer(PJRT_Buffer_UnsafePointer_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_UnsafePointer_Args, args);
  DeviceBufferDescriptor *buffer = AsImpl<DeviceBufferDescriptor>(args->buffer);
  args->buffer_pointer = reinterpret_cast<uintptr_t>(buffer->getVoidPtr());
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Buffer_IncreaseExternalReferenceCount(
    PJRT_Buffer_IncreaseExternalReferenceCount_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_Buffer_IncreaseExternalReferenceCount_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "Buffer_IncreaseExternalReferenceCount");
}

PJRT_Error *PJRT_Buffer_DecreaseExternalReferenceCount(
    PJRT_Buffer_DecreaseExternalReferenceCount_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_Buffer_DecreaseExternalReferenceCount_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "Buffer_DecreaseExternalReferenceCount");
}

PJRT_Error *PJRT_Buffer_OpaqueDeviceMemoryDataPointer(
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "Buffer_OpaqueDeviceMemoryDataPointer");
}

PJRT_Error *PJRT_Buffer_CopyToMemory(PJRT_Buffer_CopyToMemory_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_CopyToMemory_Args, args);
  const Memory *destinationMemory = AsImpl<const Memory>(args->dst_memory);
  DeviceBufferDescriptor *src = AsImpl<DeviceBufferDescriptor>(args->buffer);

  if (src->getMemorySpace() == destinationMemory)
    return PJRT_Error::allocateInvalidArgumentError(
        "buffer is already in the destination memory space");

  if (destinationMemory->getDevices().size() != 1)
    return PJRT_Error::allocateInternalError(
        "the destination memory is associated with more than one device");

  // For our use case, this is basically the same as Buffer_CopyToDevice.
  std::unique_ptr<Event> copyDoneEvent = nullptr;
  StatusOr<std::unique_ptr<DeviceBufferDescriptor>> dstBuffer =
      destinationMemory->getClient()->copyDeviceBufferToOtherDevice(
          *destinationMemory->getDevices().front(), *src, copyDoneEvent);
  if (!dstBuffer.isOk())
    return PJRT_Error::allocateFromStatus(dstBuffer.getStatus());

  Status waitStatus = copyDoneEvent->waitForReady();
  if (!waitStatus.isOk())
    return PJRT_Error::allocateFromStatus(waitStatus);
  Event::releaseWhenReady(std::move(copyDoneEvent));

  // Set outputs
  args->dst_buffer = reinterpret_cast<PJRT_Buffer *>((*dstBuffer).release());
  return PJRT_Error::getOk();
}

//===----------------------------------------------------------------------===//
// PJRT Executable API Implementation
//===----------------------------------------------------------------------===//

// Frees `executable` and deletes the underlying runtime object as if
// `PJRT_Executable_Delete` were called. `executable` can be nullptr.
PJRT_Error *PJRT_Executable_Destroy(PJRT_Executable_Destroy_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_Destroy_Args, args);
  if (args->executable == nullptr)
    return PJRT_Error::getOk();
  delete AsImpl<PJRTExecutable>(args->executable);
  return PJRT_Error::getOk();
}

// PJRT Doc:
// Returns a string that identifies the executable.
PJRT_Error *PJRT_Executable_Name(PJRT_Executable_Name_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_Name_Args, args);
  static constexpr std::string_view view = "tensorrt_engine";
  args->executable_name = view.data();
  args->executable_name_size = view.size();
  return PJRT_Error::getOk();
}

PJRT_Error *
PJRT_Executable_NumReplicas(PJRT_Executable_NumReplicas_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_NumReplicas_Args, args);
  PJRTExecutable *exe = AsImpl<PJRTExecutable>(args->executable);
  assert(exe != nullptr && "expected valid executable");
  assert(exe->getProcessorGridShape().size() == 2 &&
         "expected rank-2 processor grid shape");
  args->num_replicas = exe->getProcessorGridShape()[0];
  return PJRT_Error::getOk();
}

PJRT_Error *
PJRT_Executable_NumPartitions(PJRT_Executable_NumPartitions_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_NumPartitions_Args, args);
  PJRTExecutable *exe = AsImpl<PJRTExecutable>(args->executable);
  assert(exe != nullptr && "expected valid executable");
  assert(exe->getProcessorGridShape().size() == 2 &&
         "expected rank-2 processor grid shape");
  args->num_partitions = exe->getProcessorGridShape()[1];
  return PJRT_Error::getOk();
}

// Gets the number of outputs per device produced by `executable`.
PJRT_Error *PJRT_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_NumOutputs_Args, args);
  PJRTExecutable *exe = AsImpl<PJRTExecutable>(args->executable);
  assert(exe != nullptr && "expected valid executable pointer");
  const mtrt::FunctionSignatureView &sig = exe->getEntrypointSignature();
  args->num_outputs = sig.getNumOutputArgs();
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Executable_SizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_Executable_SizeOfGeneratedCodeInBytes_Args, args);
  PJRTExecutable *exe = AsImpl<PJRTExecutable>(args->executable);
  assert(exe != nullptr && "expected valid executable");
  args->size_in_bytes = exe->getCode().size();
  return PJRT_Error::getOk();
}

PJRT_Error *
PJRT_Executable_GetCostAnalysis(PJRT_Executable_GetCostAnalysis_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_GetCostAnalysis_Args, args);
  return PJRT_Error::allocateUnimplemented("Executable_GetCostAnalysis");
}

/// PJRT Doc: Returns a list of memory kind strings for outputs.
PJRT_Error *PJRT_Executable_OutputMemoryKinds(
    PJRT_Executable_OutputMemoryKinds_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_OutputMemoryKinds_Args,
                                   args);
  return PJRT_Error::allocateUnimplemented("Executable_OutputMemoryKinds");
}

// PJRT Doc:
// Retrieves the optimized program for a given PJRT_Executable (SPMD).
/// NOTE: This is used to get output sharding.
PJRT_Error *
PJRT_Executable_OptimizedProgram(PJRT_Executable_OptimizedProgram_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_OptimizedProgram_Args, args);
  return PJRT_Error::allocateUnimplemented("Executable_OptimizedProgram");
}

PJRT_Error *PJRT_Executable_Serialize(PJRT_Executable_Serialize_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_Serialize_Args, args);
  return PJRT_Error::allocateUnimplemented("Executable_Serialize");
}

PJRT_Error *PJRT_Executable_OutputElementTypes(
    PJRT_Executable_OutputElementTypes_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_OutputElementTypes_Args,
                                   args);
  return PJRT_Error::allocateUnimplemented("Executable_OutputElementTypes");
}

PJRT_Error *
PJRT_Executable_OutputDimensions(PJRT_Executable_OutputDimensions_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_OutputDimensions_Args, args);
  return PJRT_Error::allocateUnimplemented("Executable_OutputDimensions");
}

PJRT_Error *
PJRT_Executable_Fingerprint(PJRT_Executable_Fingerprint_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_Fingerprint_Args, args);
  return PJRT_Error::allocateUnimplemented("Executable_Fingerprint");
}

PJRT_Error *PJRT_Executable_GetCompiledMemoryStats(
    PJRT_Executable_GetCompiledMemoryStats_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_GetCompiledMemoryStats_Args,
                                   args);
  return PJRT_Error::allocateUnimplemented("Executable_GetCompiledMemoryStats");
}

//===----------------------------------------------------------------------===//
// PJRT LoadedExecutable API Implementation
//===----------------------------------------------------------------------===//

// Frees `executable`. `executable` can be nullptr.
PJRT_Error *
PJRT_LoadedExecutable_Destroy(PJRT_LoadedExecutable_Destroy_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_LoadedExecutable_Destroy_Args, args);
  PJRTLoadedExecutable *exe = AsImpl<PJRTLoadedExecutable>(args->executable);
  if (exe == nullptr)
    return PJRT_Error::getOk();
  delete exe;
  return PJRT_Error::getOk();
}

// Constructs a PJRT_Executable from a PJRT_LoadedExecutable. The returned
// executable should be freed by the caller with PJRT_Executable_Destroy.
PJRT_Error *PJRT_LoadedExecutable_GetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_LoadedExecutable_GetExecutable_Args,
                                   args);
  PJRTLoadedExecutable *loadedExe =
      AsImpl<PJRTLoadedExecutable>(args->loaded_executable);
  assert(loadedExe != nullptr && "expected valid executable");
  // Make a copy of the executable.
  std::unique_ptr<PJRTExecutable> exe = loadedExe->getExecutableCopy();
  // Release it to the caller.
  args->executable = reinterpret_cast<PJRT_Executable *>(exe.release());
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_LoadedExecutable_AddressableDevices_Args, args);
  PJRTLoadedExecutable *loadedExe =
      AsImpl<PJRTLoadedExecutable>(args->executable);
  assert(loadedExe && "expected valid executable");

  args->addressable_devices =
      reinterpret_cast<PJRT_Device **>(loadedExe->getDevices().data());
  args->num_addressable_devices = loadedExe->getDevices().size();
  return PJRT_Error::getOk();
}

// Frees `executable` and deletes the underlying runtime object as if
// `PJRT_LoadedExecutable_Delete` were called. `executable` can be nullptr.
PJRT_Error *
PJRT_LoadedExecutable_Delete(PJRT_LoadedExecutable_Delete_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_LoadedExecutable_Delete_Args, args);
  PJRTLoadedExecutable *exe = AsImpl<PJRTLoadedExecutable>(args->executable);
  if (exe == nullptr)
    return PJRT_Error::getOk();
  delete exe;
  return PJRT_Error::getOk();
}

// True if and only if PJRT_Executable_Delete has previously been called.
PJRT_Error *
PJRT_LoadedExecutable_IsDeleted(PJRT_LoadedExecutable_IsDeleted_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_LoadedExecutable_IsDeleted_Args, args);
  args->is_deleted = false;
  return PJRT_Error::getOk();
}

// PJRT Doc:
// Executes on devices addressable by the client.
// See the PJRT struct `PJRT_Executable_Execute_Args` in
// `xla/pjrt/c/pjrt_c_api.h` for more information.
//
// NOTE: the results for each engine are allocated by us. they will be freed
// when the caller calls PJRT_Buffer_Destroy on the returned
// `args->output_list` buffers.
PJRT_Error *
PJRT_LoadedExecutable_Execute(PJRT_LoadedExecutable_Execute_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_LoadedExecutable_Execute_Args, args);
  PJRTLoadedExecutable *loadedExecutable =
      AsImpl<PJRTLoadedExecutable>(args->executable);
  assert(loadedExecutable && "expected valid executable");
  // This method supports two options:
  // 1) execution of a single executable across multiple devices
  // 2) execution of a single executable on a specific device (the
  //    `execute_device`).
  // Opt 2 is currently unsupported.
  if (args->execute_device)
    return PJRT_Error::allocateUnimplemented(
        "LoadedExecutable_Execute with execute_device");

  const std::vector<PjRtDevice *> devices = loadedExecutable->getDevices();
  if (devices.size() != args->num_devices)
    return PJRT_Error::allocateInternalError(
        "expected number of devices to match");

  Status s = loadedExecutable->execute(
      reinterpret_cast<DeviceBufferDescriptor *const *const *>(
          args->argument_lists),
      args->num_args,
      reinterpret_cast<DeviceBufferDescriptor **const *>(args->output_lists),
      reinterpret_cast<Event **>(args->device_complete_events));
  if (!s.isOk())
    return PJRT_Error::allocateFromStatus(s);
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Executable_DeserializeAndLoad(
    PJRT_Executable_DeserializeAndLoad_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Executable_DeserializeAndLoad_Args,
                                   args);
  return PJRT_Error::allocateUnimplemented("Executable_DeserializeAndLoad");
}

/// PJRT Doc:
/// A unique fingerprint for `executable`. Two executables that were produced
/// by compiling with identical inputs (same program, compile options,
/// compiler version, etc.) should have the same fingerprint. May not be
/// implemented by all platforms.
PJRT_Error *PJRT_LoadedExecutable_Fingerprint(
    PJRT_LoadedExecutable_Fingerprint_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_LoadedExecutable_Fingerprint_Args,
                                   args);
  return PJRT_Error::allocateUnimplemented("LoadedExecutable_Fingerprint");
}

//===----------------------------------------------------------------------===//
// PJRT Client API Implementation
//===----------------------------------------------------------------------===//

// Creates and initializes a new PJRT_Client and returns in `client`.
PJRT_Error *PJRT_Client_Create(PJRT_Client_Create_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_Create_Args, args);
  StatusOr<std::unique_ptr<Client>> client = Client::create();
  if (!client.isOk())
    return PJRT_Error::allocateFromStatus(client.getStatus());
  args->client = reinterpret_cast<PJRT_Client *>(client->release());
  return PJRT_Error::getOk();
}

// Shuts down and frees `client`. `client` can be nullptr.
PJRT_Error *PJRT_Client_Destroy(PJRT_Client_Destroy_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_Destroy_Args, args);
  Client *client = AsImpl<Client>(args->client);
  if (!client)
    return PJRT_Error::getOk();
  delete client;
  return PJRT_Error::getOk();
}

// Return the process index of this client. Always 0 in single-process
// settings.
PJRT_Error *PJRT_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_ProcessIndex_Args, args);
  args->process_index = 0;
  return PJRT_Error::getOk();
}

// Returns a string that identifies the platform (e.g. "cpu", "gpu", "tpu").
PJRT_Error *PJRT_Client_PlatformName(PJRT_Client_PlatformName_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_PlatformName_Args, args);
  static constexpr std::string_view view = "mlir_tensorrt";
  args->platform_name = view.data();
  args->platform_name_size = view.size();
  return PJRT_Error::getOk();
}

// PJRT Doc:
// Returns a string containing human-readable, platform-specific version info
// (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
PJRT_Error *
PJRT_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_PlatformVersion_Args, args);
  static constexpr std::string_view view = "mlir-tensorrt-v0.0.1";
  args->platform_version = view.data();
  args->platform_version_size = view.size();
  return PJRT_Error::getOk();
}

PJRT_Error *
PJRT_Client_TopologyDescription(PJRT_Client_TopologyDescription_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_TopologyDescription_Args, args);
  return PJRT_Error::allocateUnimplemented("Client_TopologyDescription");
}

// Returns a list of all devices visible to the runtime, including addressable
// and non-addressable devices.
PJRT_Error *PJRT_Client_Devices(PJRT_Client_Devices_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_Devices_Args, args);
  Client *client = AsImpl<Client>(args->client);
  if (!client)
    return PJRT_Error::allocateInvalidArgumentError("expected non-null client");
  PJRT_Device *const *devices =
      reinterpret_cast<PJRT_Device *const *>(client->getDevices().data());
  args->devices = const_cast<PJRT_Device **>(devices);
  args->num_devices = client->getDevices().size();
  return PJRT_Error::getOk();
}

// Returns a list of devices that are addressable from the client.
// Addressable devices are those that the client can issue commands to.
// All devices are addressable in a single-process environment.
PJRT_Error *
PJRT_Client_AddressableDevices(PJRT_Client_AddressableDevices_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_AddressableDevices_Args, args);
  Client *client = AsImpl<Client>(args->client);
  if (!client)
    return PJRT_Error::allocateInvalidArgumentError("expected non-null client");
  PJRT_Device *const *devices =
      reinterpret_cast<PJRT_Device *const *>(client->getDevices().data());
  args->addressable_devices = const_cast<PJRT_Device **>(devices);
  args->num_addressable_devices = client->getDevices().size();
  return PJRT_Error::getOk();
}

// Returns a PJRT_Device* with the specified ID as returned by PJRT_Device_Id.
PJRT_Error *PJRT_Client_LookupDevice(PJRT_Client_LookupDevice_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_LookupDevice_Args, args);
  Client *client = AsImpl<Client>(args->client);
  if (!client)
    return PJRT_Error::allocateInvalidArgumentError("expected non-null client");
  size_t id = args->id;
  if (id >= client->getDevices().size())
    return PJRT_Error::allocateInternalError("device index out of range");
  args->device = reinterpret_cast<PJRT_Device *>(client->getDevices()[id]);
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_LookupAddressableDevice_Args,
                                   args);
  Client *client = AsImpl<Client>(args->client);
  if (!client)
    return PJRT_Error::allocateInvalidArgumentError("expected non-null client");
  size_t id = args->local_hardware_id;
  if (id >= client->getDevices().size())
    return PJRT_Error::allocateInternalError("device index out of range");
  args->addressable_device =
      reinterpret_cast<PJRT_Device *>(client->getDevices()[id]);
  return PJRT_Error::getOk();
}

/// Returns a list of memories that are addressable from the client.
/// Addressable memories are those that the client can directly transfer data
/// to and from. All memories are addressable in a single-process environment.
PJRT_Error *
PJRT_Client_AddressableMemories(PJRT_Client_AddressableMemories_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_AddressableMemories_Args, args);
  Client *client = AsImpl<Client>(args->client);
  args->addressable_memories = reinterpret_cast<PJRT_Memory *const *>(
      client->getAddressableMemories().data());
  args->num_addressable_memories = client->getAddressableMemories().size();
  return PJRT_Error::getOk();
}

// PJRT Doc:
// Compiles a program in specified format (such as MLIR or HLO) with given
// `options`.
PJRT_Error *PJRT_Client_Compile(PJRT_Client_Compile_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_Compile_Args, args);
  Client *client = AsImpl<Client>(args->client);
  if (!client)
    return PJRT_Error::allocateInvalidArgumentError("expected non-null client");

  // TODO: implement MLIR compilation pipeline
  std::string_view format(args->program->format, args->program->format_size);
  if (format != "mlir")
    return PJRT_Error::allocateUnimplemented(
        "only MLIR programs are supported");

  xla::CompileOptionsProto compileOptionsProto;
  if (!compileOptionsProto.ParseFromString(
          std::string(args->compile_options, args->compile_options_size)))
    return PJRT_Error::allocateInvalidArgumentError(
        "failed to parse compile options");

  StatusOr<std::unique_ptr<PJRTLoadedExecutable>> loadedExecutable =
      client->compileAndLoadMlirProgram(
          llvm::StringRef(args->program->code, args->program->code_size),
          compileOptionsProto);
  if (!loadedExecutable.isOk())
    return PJRT_Error::allocateFromStatus(loadedExecutable.getStatus());
  args->executable =
      reinterpret_cast<PJRT_LoadedExecutable *>(loadedExecutable->release());
  return PJRT_Error::getOk();
}

// PJRT is missing documentation for this method.
PJRT_Error *PJRT_Client_DefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_DefaultDeviceAssignment_Args,
                                   args);
  for (size_t i = 0; i < args->default_assignment_size; ++i)
    args->default_assignment[i] = 0;
  return PJRT_Error::getOk();
}

// Asynchronously copies a buffer stored on host to device memory.
// The `args->host_buffer_semantics` tell us whether this host buffer might
// change outside of this call. If so, we need to copy it synchronously. If we
// don't want to copy to the **device** synchronously, then we must make a
// **host-side copy** of it to use as a source of the  the asynchronous
// transfer.
PJRT_Error *
PJRT_Client_BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_BufferFromHostBuffer_Args, args);
  Client *client = AsImpl<Client>(args->client);

  if (!args->memory) {
    return PJRT_Error::allocateUnimplemented(
        "BufferFromHostBuffer requires 'memory' argument");
  }
  const auto *memory = AsImpl<const Memory>(args->memory);
  if (memory->getDevices().size() != 1)
    return PJRT_Error::allocateInternalError(
        "BufferFromHostBuffer only knows how to copy to memories associated "
        "with a unique device");

  PjRtDevice *device = memory->getDevices().front();

  // Check whether the caller specified a layout for the device buffer.
  if (args->device_layout)
    return PJRT_Error::allocateUnimplemented(
        "unhandled specification of device layout");

  StatusOr<mtrt::ScalarType> elementType =
      getElementTypeFromPJRTElementType(args->type);
  if (!elementType.isOk())
    return PJRT_Error::allocateUnimplemented("unsupported element type");

  auto hostBufferType = BufferType::createWithByteStrides(
      *elementType,
      std::vector<int64_t>(args->dims, args->dims + args->num_dims),
      std::vector<int64_t>(args->byte_strides,
                           args->byte_strides + args->num_dims),
      mtrt::PointerType::host, /*offset=*/0);

  StatusOr<std::unique_ptr<mtrt::MemRefValue>> hostBuffer =
      client->getRuntimeClient()->createExternalMemRef(
          hostBufferType, reinterpret_cast<uintptr_t>(args->data),
          /*device=*/nullptr, /*assertCanonicalStrides=*/false,
          /*destroyCallback=*/nullptr);
  if (!hostBuffer.isOk())
    return PJRT_Error::allocateFromStatus(hostBuffer.getStatus());

  BufferDescriptor sourceBuffer(std::move(*hostBuffer));

  // Compute the device buffer layout. It is assumed that on the device,
  // the strides are such that dimensions are ordered major-to-minor. If this
  // is not the case, then the implementation will implement the transpose.
  auto deviceBufferType = BufferType::createWithCanonicalLayout(
      *elementType, hostBufferType.getShape(), mtrt::PointerType::device);

  std::unique_ptr<Event> hostDoneEvent = nullptr;
  StatusOr<std::unique_ptr<DeviceBufferDescriptor>> deviceBuffer =
      client->getDeviceBufferFromHostBuffer(
          sourceBuffer, args->host_buffer_semantics, *device, deviceBufferType,
          &hostDoneEvent);
  if (!deviceBuffer.isOk())
    return PJRT_Error::allocateFromStatus(deviceBuffer.getStatus());

  // Set outputs
  args->buffer = reinterpret_cast<PJRT_Buffer *>((*deviceBuffer).release());
  args->done_with_host_buffer =
      reinterpret_cast<PJRT_Event *>(hostDoneEvent.release());

  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Client_CreateViewOfDeviceBuffer(
    PJRT_Client_CreateViewOfDeviceBuffer_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_CreateViewOfDeviceBuffer_Args,
                                   args);
  return PJRT_Error::allocateUnimplemented("Client_CreateViewOfDeviceBuffer");
}

//===----------------------------------------------------------------------===//
// PJRT Device Description API Implementation
//===----------------------------------------------------------------------===//

// The ID of this device. IDs are unique among devices of this type
// (e.g. CPUs, GPUs). On multi-host platforms, this will be unique across all
// hosts' devices.
PJRT_Error *PJRT_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_DeviceDescription_Id_Args, args);
  PjRtDeviceDescription *device =
      AsImpl<PjRtDeviceDescription>(args->device_description);
  args->id = device->getUniqueID();
  return PJRT_Error::getOk();
}

// The index of the process that this device belongs to, i.e. is addressable
// from. This is not always identical to PJRT_Client_ProcessIndex in a
// multi-process setting, where each client can see devices from all
// processes, but only a subset of them are addressable and have the same
// process_index as the client.
PJRT_Error *PJRT_DeviceDescription_ProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_DeviceDescription_ProcessIndex_Args,
                                   args);
  PjRtDeviceDescription *device =
      AsImpl<PjRtDeviceDescription>(args->device_description);
  args->process_index = device->getHostID();
  return PJRT_Error::getOk();
}

// Returns an array of device specific attributes with attribute name, value
// and value type.
PJRT_Error *PJRT_DeviceDescription_Attributes(
    PJRT_DeviceDescription_Attributes_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_DeviceDescription_Attributes_Args,
                                   args);
  args->num_attributes = 0;
  args->attributes = nullptr;
  return PJRT_Error::getOk();
}

// A vendor-dependent string that uniquely identifies the kind of device,
// e.g., "Tesla V100-SXM2-16GB".
PJRT_Error *
PJRT_DeviceDescription_Kind(PJRT_DeviceDescription_Kind_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_DeviceDescription_Kind_Args, args);
  PjRtDeviceDescription *device =
      AsImpl<PjRtDeviceDescription>(args->device_description);
  mtrt::DeviceKind kind = device->getKind();
  llvm::StringRef view = mtrt::getDeviceKindString(kind);
  args->device_kind = view.data();
  args->device_kind_size = view.size();
  return PJRT_Error::getOk();
}

// Debug string suitable for logging when errors occur. Should be verbose
// enough to describe the current device unambiguously.
PJRT_Error *PJRT_DeviceDescription_DebugString(
    PJRT_DeviceDescription_DebugString_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_DeviceDescription_DebugString_Args,
                                   args);
  PjRtDeviceDescription *device =
      AsImpl<PjRtDeviceDescription>(args->device_description);
  llvm::StringRef view = device->getString(/*verbose=*/true);
  args->debug_string = view.data();
  args->debug_string_size = view.size();
  return PJRT_Error::getOk();
}

// Debug string suitable for reading by end users, should be reasonably terse,
// for example: "CpuDevice(id=0)".
PJRT_Error *
PJRT_DeviceDescription_ToString(PJRT_DeviceDescription_ToString_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_DeviceDescription_ToString_Args, args);
  PjRtDeviceDescription *device =
      AsImpl<PjRtDeviceDescription>(args->device_description);
  llvm::StringRef view = device->getString(/*verbose=*/false);
  args->to_string = view.data();
  args->to_string_size = view.size();
  return PJRT_Error::getOk();
}

//===----------------------------------------------------------------------===//
// PJRT Device API Implementation
//===----------------------------------------------------------------------===//

PJRT_Error *PJRT_Device_GetDescription(PJRT_Device_GetDescription_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Device_GetDescription_Args, args);
  PjRtDevice *device = AsImpl<PjRtDevice>(args->device);
  args->device_description = reinterpret_cast<PJRT_DeviceDescription *>(
      const_cast<mtrt::DeviceDescription *>(
          &device->getMTRTDevice()->getDescription()));
  return PJRT_Error::getOk();
}

// Whether client can issue command to this device.
PJRT_Error *PJRT_Device_IsAddressable(PJRT_Device_IsAddressable_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Device_IsAddressable_Args, args);
  args->is_addressable = true;
  return PJRT_Error::getOk();
}

// Opaque hardware ID, e.g., the CUDA device number. In general, not
// guaranteed to be dense, and -1 if undefined.
PJRT_Error *
PJRT_Device_LocalHardwareId(PJRT_Device_LocalHardwareId_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Device_LocalHardwareId_Args, args);
  args->local_hardware_id = 0;
  return PJRT_Error::getOk();
}

PJRT_Error *
PJRT_Device_AddressableMemories(PJRT_Device_AddressableMemories_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Device_AddressableMemories_Args, args);
  const auto *device = AsImpl<const PjRtDevice>(args->device);
  args->memories =
      reinterpret_cast<PJRT_Memory *const *>(device->getMemories().data());
  args->num_memories = device->getMemories().size();
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Device_DefaultMemory_Args, args);
  auto *device = AsImpl<const PjRtDevice>(args->device);
  args->memory = reinterpret_cast<PJRT_Memory *>(device->getMemories().front());
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Device_MemoryStats(PJRT_Device_MemoryStats_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Device_MemoryStats_Args, args);
  return PJRT_Error::allocateUnimplemented("Device_MemoryStats");
}

//===----------------------------------------------------------------------===//
// PJRT Memory API Implementation
//===----------------------------------------------------------------------===//

PJRT_Error *PJRT_Memory_Id(PJRT_Memory_Id_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Memory_Id_Args, args);
  const auto *memory = AsImpl<Memory>(args->memory);
  args->id = memory->getId();
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Memory_Kind(PJRT_Memory_Kind_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Memory_Kind_Args, args);
  const auto *memory = AsImpl<Memory>(args->memory);
  args->kind = memory->getKind().data();
  args->kind_size = memory->getKind().size();
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Memory_Kind_Id_Args, args);
  const auto *memory = AsImpl<Memory>(args->memory);
  args->kind_id = memory->getKindId();
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Memory_DebugString(PJRT_Memory_DebugString_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Memory_DebugString_Args, args);
  const auto *memory = AsImpl<Memory>(args->memory);
  args->debug_string = memory->getDebugString().data();
  args->debug_string_size = memory->getDebugString().size();
  return PJRT_Error::getOk();
}

PJRT_Error *PJRT_Memory_ToString(PJRT_Memory_ToString_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Memory_ToString_Args, args);
  const auto *memory = AsImpl<Memory>(args->memory);
  args->to_string = memory->getDebugString().data();
  args->to_string_size = memory->getDebugString().size();
  return PJRT_Error::getOk();
}

PJRT_Error *
PJRT_Memory_AddressableByDevices(PJRT_Memory_AddressableByDevices_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Memory_AddressableByDevices_Args, args);
  const auto *memory = AsImpl<Memory>(args->memory);
  args->num_devices = memory->getDevices().size();
  args->devices =
      reinterpret_cast<PJRT_Device *const *>(memory->getDevices().data());
  return PJRT_Error::getOk();
}

//===----------------------------------------------------------------------===//
// PJRT ExecuteContext API Implementation
//===----------------------------------------------------------------------===//

PJRT_Error *PJRT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_ExecuteContext_Create_Args, args);
  return PJRT_Error::allocateUnimplemented("ExecuteContext_Create");
}

PJRT_Error *
PJRT_ExecuteContext_Destroy(PJRT_ExecuteContext_Destroy_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_ExecuteContext_Destroy_Args, args);
  return PJRT_Error::allocateUnimplemented("ExecuteContext_Destroy");
}

//===----------------------------------------------------------------------===//
// PJRT CopyToDeviceStream API Implementation
//===----------------------------------------------------------------------===//

PJRT_Error *
PJRT_CopyToDeviceStream_Destroy(PJRT_CopyToDeviceStream_Destroy_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_CopyToDeviceStream_Destroy_Args, args);
  return PJRT_Error::allocateUnimplemented("CopyToDeviceStream_Destroy");
}

PJRT_Error *
PJRT_CopyToDeviceStream_AddChunk(PJRT_CopyToDeviceStream_AddChunk_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_CopyToDeviceStream_AddChunk_Args, args);
  return PJRT_Error::allocateUnimplemented("CopyToDeviceStream_AddChunk");
}

PJRT_Error *PJRT_CopyToDeviceStream_TotalBytes(
    PJRT_CopyToDeviceStream_TotalBytes_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_CopyToDeviceStream_TotalBytes_Args,
                                   args);
  return PJRT_Error::allocateUnimplemented("CopyToDeviceStream_TotalBytes");
}

PJRT_Error *PJRT_CopyToDeviceStream_GranuleSize(
    PJRT_CopyToDeviceStream_GranuleSize_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_CopyToDeviceStream_GranuleSize_Args,
                                   args);
  return PJRT_Error::allocateUnimplemented("CopyToDeviceStream_GranuleSize");
}

PJRT_Error *PJRT_CopyToDeviceStream_CurrentBytes(
    PJRT_CopyToDeviceStream_CurrentBytes_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_CopyToDeviceStream_CurrentBytes_Args,
                                   args);
  return PJRT_Error::allocateUnimplemented("CopyToDeviceStream_CurrentBytes");
}

//===----------------------------------------------------------------------===//
// PJRT TopologyDescription API Implementation
//===----------------------------------------------------------------------===//

PJRT_Error *
PJRT_TopologyDescription_Create(PJRT_TopologyDescription_Create_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_TopologyDescription_Create_Args, args);
  return PJRT_Error::allocateUnimplemented("TopologyDescription_Create");
}

PJRT_Error *
PJRT_TopologyDescription_Destroy(PJRT_TopologyDescription_Destroy_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_TopologyDescription_Destroy_Args, args);
  return PJRT_Error::allocateUnimplemented("TopologyDescription_Destroy");
}

PJRT_Error *PJRT_TopologyDescription_PlatformName(
    PJRT_TopologyDescription_PlatformName_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_TopologyDescription_PlatformName_Args,
                                   args);
  return PJRT_Error::allocateUnimplemented("TopologyDescription_PlatformName");
}

PJRT_Error *PJRT_TopologyDescription_PlatformVersion(
    PJRT_TopologyDescription_PlatformVersion_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_TopologyDescription_PlatformVersion_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "TopologyDescription_PlatformVersion");
}

PJRT_Error *PJRT_TopologyDescription_GetDeviceDescriptions(
    PJRT_TopologyDescription_GetDeviceDescriptions_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_TopologyDescription_GetDeviceDescriptions_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "TopologyDescription_GetDeviceDescriptions");
}

PJRT_Error *PJRT_TopologyDescription_Serialize(
    PJRT_TopologyDescription_Serialize_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_TopologyDescription_Serialize_Args,
                                   args);
  return PJRT_Error::allocateUnimplemented("TopologyDescription_Serialize");
}

PJRT_Error *PJRT_TopologyDescription_Attributes(
    PJRT_TopologyDescription_Attributes_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_TopologyDescription_Attributes_Args,
                                   args);
  return PJRT_Error::allocateUnimplemented("TopologyDescription_Attributes");
}

//===----------------------------------------------------------------------===//
// PJRT Compile API Implementation
//===----------------------------------------------------------------------===//

PJRT_Error *PJRT_Compile(PJRT_Compile_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Compile_Args, args);
  return PJRT_Error::allocateUnimplemented("Compile");
}

//===----------------------------------------------------------------------===//
// PJRT AsyncHostToDeviceTransferManager API Implementation
//===----------------------------------------------------------------------===//

PJRT_Error *PJRT_AsyncHostToDeviceTransferManager_Destroy(
    PJRT_AsyncHostToDeviceTransferManager_Destroy_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_AsyncHostToDeviceTransferManager_Destroy_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "AsyncHostToDeviceTransferManager_Destroy");
}

PJRT_Error *PJRT_AsyncHostToDeviceTransferManager_TransferData(
    PJRT_AsyncHostToDeviceTransferManager_TransferData_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_AsyncHostToDeviceTransferManager_TransferData_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "AsyncHostToDeviceTransferManager_TransferData");
}

PJRT_Error *PJRT_Client_CreateBuffersForAsyncHostToDevice(
    PJRT_Client_CreateBuffersForAsyncHostToDevice_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_Client_CreateBuffersForAsyncHostToDevice_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "Client_CreateBuffersForAsyncHostToDevice");
}

PJRT_Error *PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer(
    PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "AsyncHostToDeviceTransferManager_RetrieveBuffer");
}

PJRT_Error *PJRT_AsyncHostToDeviceTransferManager_Device(
    PJRT_AsyncHostToDeviceTransferManager_Device_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_AsyncHostToDeviceTransferManager_Device_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "AsyncHostToDeviceTransferManager_Device");
}

PJRT_Error *PJRT_AsyncHostToDeviceTransferManager_BufferCount(
    PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "AsyncHostToDeviceTransferManager_BufferCount");
}

PJRT_Error *PJRT_AsyncHostToDeviceTransferManager_BufferSize(
    PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "AsyncHostToDeviceTransferManager_BufferSize");
}

PJRT_Error *PJRT_AsyncHostToDeviceTransferManager_SetBufferError(
    PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "AsyncHostToDeviceTransferManager_SetBufferError");
}

PJRT_Error *PJRT_AsyncHostToDeviceTransferManager_AddMetadata(
    PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(
      PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args, args);
  return PJRT_Error::allocateUnimplemented(
      "AsyncHostToDeviceTransferManager_AddMetadata");
}

PJRT_Error *PJRT_Client_DmaMap(PJRT_Client_DmaMap_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_DmaMap_Args, args);
  return PJRT_Error::allocateUnimplemented("Client_DmaMap");
}

PJRT_Error *PJRT_Client_DmaUnmap(PJRT_Client_DmaUnmap_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Client_DmaUnmap_Args, args);
  return PJRT_Error::allocateUnimplemented("Client_DmaUnmap");
}

PJRT_Error *PJRT_Buffer_CopyRawToHost(PJRT_Buffer_CopyRawToHost_Args *args) {
  PJRT_RETURN_IF_STRUCT_SIZE_ERROR(PJRT_Buffer_CopyRawToHost_Args, args);
  return PJRT_Error::allocateUnimplemented("Buffer_CopyRawToHost");
}

} // namespace mtrt::pjrt

//===----------------------------------------------------------------------===//
// GetPjrtApi - Main API Entry Point
//===----------------------------------------------------------------------===//

const PJRT_Api *GetPjrtApi() {
  PJRT_DBGF("%s", "GetPjrtApi");
  static PJRT_Api api = {
      /*struct_size=*/PJRT_Api_STRUCT_SIZE,
      /*extension_start=*/nullptr,

      /*pjrt_api_version=*/
      PJRT_Api_Version{/*struct_size=*/PJRT_Api_Version_STRUCT_SIZE,
                       /*extension_start=*/nullptr,
                       /*major_version=*/PJRT_API_MAJOR,
                       /*minor_version=*/PJRT_API_MINOR},

      /*PJRT_Error_Destroy=*/mtrt::pjrt::PJRT_Error_Destroy,
      /*PJRT_Error_Message=*/mtrt::pjrt::PJRT_Error_Message,
      /*PJRT_Error_GetCode=*/mtrt::pjrt::PJRT_Error_GetCode,

      /*PJRT_Plugin_Initialize=*/mtrt::pjrt::PJRT_Plugin_Initialize,
      /*PJRT_Plugin_Attributes=*/mtrt::pjrt::PJRT_Plugin_Attributes,

      /*PJRT_Event_Destroy=*/mtrt::pjrt::PJRT_Event_Destroy,
      /*PJRT_Event_IsReady=*/mtrt::pjrt::PJRT_Event_IsReady,
      /*PJRT_Event_Error=*/mtrt::pjrt::PJRT_Event_Error,
      /*PJRT_Event_Await=*/mtrt::pjrt::PJRT_Event_Await,
      /*PJRT_Event_OnReady=*/mtrt::pjrt::PJRT_Event_OnReady,

      /*PJRT_Client_Create=*/mtrt::pjrt::PJRT_Client_Create,
      /*PJRT_Client_Destroy=*/mtrt::pjrt::PJRT_Client_Destroy,
      /*PJRT_Client_PlatformName=*/mtrt::pjrt::PJRT_Client_PlatformName,
      /*PJRT_Client_ProcessIndex=*/mtrt::pjrt::PJRT_Client_ProcessIndex,
      /*PJRT_Client_PlatformVersion=*/mtrt::pjrt::PJRT_Client_PlatformVersion,
      /*PJRT_Client_Devices=*/mtrt::pjrt::PJRT_Client_Devices,
      /*PJRT_Client_AddressableDevices=*/
      mtrt::pjrt::PJRT_Client_AddressableDevices,
      /*PJRT_Client_LookupDevice=*/mtrt::pjrt::PJRT_Client_LookupDevice,
      /*PJRT_Client_LookupAddressableDevice=*/
      mtrt::pjrt::PJRT_Client_LookupAddressableDevice,
      /*PJRT_Client_AddressableMemories=*/
      mtrt::pjrt::PJRT_Client_AddressableMemories,
      /*PJRT_Client_Compile=*/mtrt::pjrt::PJRT_Client_Compile,
      /*PJRT_Client_DefaultDeviceAssignment=*/
      mtrt::pjrt::PJRT_Client_DefaultDeviceAssignment,
      /*PJRT_Client_BufferFromHostBuffer=*/
      mtrt::pjrt::PJRT_Client_BufferFromHostBuffer,

      /*PJRT_DeviceDescription_Id=*/mtrt::pjrt::PJRT_DeviceDescription_Id,
      /*PJRT_DeviceDescription_ProcessIndex=*/
      mtrt::pjrt::PJRT_DeviceDescription_ProcessIndex,
      /*PJRT_DeviceDescription_Attributes=*/
      mtrt::pjrt::PJRT_DeviceDescription_Attributes,
      /*PJRT_DeviceDescription_Kind=*/mtrt::pjrt::PJRT_DeviceDescription_Kind,
      /*PJRT_DeviceDescription_DebugString=*/
      mtrt::pjrt::PJRT_DeviceDescription_DebugString,
      /*PJRT_DeviceDescription_ToString=*/
      mtrt::pjrt::PJRT_DeviceDescription_ToString,

      /*PJRT_Device_GetDescription=*/mtrt::pjrt::PJRT_Device_GetDescription,
      /*PJRT_Device_IsAddressable=*/mtrt::pjrt::PJRT_Device_IsAddressable,
      /*PJRT_Device_LocalHardwareId=*/mtrt::pjrt::PJRT_Device_LocalHardwareId,
      /*PJRT_Device_AddressableMemories=*/
      mtrt::pjrt::PJRT_Device_AddressableMemories,
      /*PJRT_Device_DefaultMemory=*/mtrt::pjrt::PJRT_Device_DefaultMemory,
      /*PJRT_Device_MemoryStats=*/mtrt::pjrt::PJRT_Device_MemoryStats,

      /*PJRT_Memory_Id=*/mtrt::pjrt::PJRT_Memory_Id,
      /*PJRT_Memory_Kind=*/mtrt::pjrt::PJRT_Memory_Kind,
      /*PJRT_Memory_DebugString=*/mtrt::pjrt::PJRT_Memory_DebugString,
      /*PJRT_Memory_ToString=*/mtrt::pjrt::PJRT_Memory_ToString,
      /*PJRT_Memory_AddressableByDevices=*/
      mtrt::pjrt::PJRT_Memory_AddressableByDevices,

      /*PJRT_Executable_Destroy=*/mtrt::pjrt::PJRT_Executable_Destroy,
      /*PJRT_Executable_Name=*/mtrt::pjrt::PJRT_Executable_Name,
      /*PJRT_Executable_NumReplicas=*/mtrt::pjrt::PJRT_Executable_NumReplicas,
      /*PJRT_Executable_NumPartitions=*/
      mtrt::pjrt::PJRT_Executable_NumPartitions,
      /*PJRT_Executable_NumOutputs=*/mtrt::pjrt::PJRT_Executable_NumOutputs,
      /*PJRT_Executable_SizeOfGeneratedCodeInBytes=*/
      mtrt::pjrt::PJRT_Executable_SizeOfGeneratedCodeInBytes,
      /*PJRT_Executable_GetCostAnalysis=*/
      mtrt::pjrt::PJRT_Executable_GetCostAnalysis,
      /*PJRT_Executable_OutputMemoryKinds=*/
      mtrt::pjrt::PJRT_Executable_OutputMemoryKinds,
      /*PJRT_Executable_OptimizedProgram=*/
      mtrt::pjrt::PJRT_Executable_OptimizedProgram,
      /*PJRT_Executable_Serialize=*/mtrt::pjrt::PJRT_Executable_Serialize,

      /*PJRT_LoadedExecutable_Destroy=*/
      mtrt::pjrt::PJRT_LoadedExecutable_Destroy,
      /*PJRT_LoadedExecutable_GetExecutable=*/
      mtrt::pjrt::PJRT_LoadedExecutable_GetExecutable,
      /*PJRT_LoadedExecutable_AddressableDevices=*/
      mtrt::pjrt::PJRT_LoadedExecutable_AddressableDevices,
      /*PJRT_LoadedExecutable_Delete=*/mtrt::pjrt::PJRT_LoadedExecutable_Delete,
      /*PJRT_LoadedExecutable_IsDeleted=*/
      mtrt::pjrt::PJRT_LoadedExecutable_IsDeleted,
      /*PJRT_LoadedExecutable_Execute=*/
      mtrt::pjrt::PJRT_LoadedExecutable_Execute,
      /*PJRT_Executable_DeserializeAndLoad=*/
      mtrt::pjrt::PJRT_Executable_DeserializeAndLoad,
      /*PJRT_LoadedExecutable_Fingerprint=*/
      mtrt::pjrt::PJRT_LoadedExecutable_Fingerprint,

      /*PJRT_Buffer_Destroy=*/mtrt::pjrt::PJRT_Buffer_Destroy,
      /*PJRT_Buffer_ElementType=*/mtrt::pjrt::PJRT_Buffer_ElementType,
      /*PJRT_Buffer_Dimensions=*/mtrt::pjrt::PJRT_Buffer_Dimensions,
      /*PJRT_Buffer_UnpaddedDimensions=*/
      mtrt::pjrt::PJRT_Buffer_UnpaddedDimensions,
      /*PJRT_Buffer_DynamicDimensionIndices=*/
      mtrt::pjrt::PJRT_Buffer_DynamicDimensionIndices,
      /*PJRT_Buffer_GetMemoryLayout=*/mtrt::pjrt::PJRT_Buffer_GetMemoryLayout,
      /*PJRT_Buffer_OnDeviceSizeInBytes=*/
      mtrt::pjrt::PJRT_Buffer_OnDeviceSizeInBytes,
      /*PJRT_Buffer_Device=*/mtrt::pjrt::PJRT_Buffer_Device,
      /*PJRT_Buffer_Memory=*/mtrt::pjrt::PJRT_Buffer_Memory,
      /*PJRT_Buffer_Delete=*/mtrt::pjrt::PJRT_Buffer_Delete,
      /*PJRT_Buffer_IsDeleted=*/mtrt::pjrt::PJRT_Buffer_IsDeleted,
      /*PJRT_Buffer_CopyToDevice=*/mtrt::pjrt::PJRT_Buffer_CopyToDevice,
      /*PJRT_Buffer_ToHostBuffer=*/mtrt::pjrt::PJRT_Buffer_ToHostBuffer,
      /*PJRT_Buffer_IsOnCpu=*/mtrt::pjrt::PJRT_Buffer_IsOnCpu,
      /*PJRT_Buffer_ReadyEvent=*/mtrt::pjrt::PJRT_Buffer_ReadyEvent,
      /*PJRT_Buffer_UnsafePointer=*/mtrt::pjrt::PJRT_Buffer_UnsafePointer,
      /*PJRT_Buffer_IncreaseExternalReferenceCount=*/
      mtrt::pjrt::PJRT_Buffer_IncreaseExternalReferenceCount,
      /*PJRT_Buffer_DecreaseExternalReferenceCount=*/
      mtrt::pjrt::PJRT_Buffer_DecreaseExternalReferenceCount,
      /*PJRT_Buffer_OpaqueDeviceMemoryDataPointer=*/
      mtrt::pjrt::PJRT_Buffer_OpaqueDeviceMemoryDataPointer,

      /*PJRT_CopyToDeviceStream_Destroy=*/
      mtrt::pjrt::PJRT_CopyToDeviceStream_Destroy,
      /*PJRT_CopyToDeviceStream_AddChunk=*/
      mtrt::pjrt::PJRT_CopyToDeviceStream_AddChunk,
      /*PJRT_CopyToDeviceStream_TotalBytes=*/
      mtrt::pjrt::PJRT_CopyToDeviceStream_TotalBytes,
      /*PJRT_CopyToDeviceStream_GranuleSize=*/
      mtrt::pjrt::PJRT_CopyToDeviceStream_GranuleSize,
      /*PJRT_CopyToDeviceStream_CurrentBytes=*/
      mtrt::pjrt::PJRT_CopyToDeviceStream_CurrentBytes,

      /*PJRT_TopologyDescription_Create=*/
      mtrt::pjrt::PJRT_TopologyDescription_Create,
      /*PJRT_TopologyDescription_Destroy=*/
      mtrt::pjrt::PJRT_TopologyDescription_Destroy,
      /*PJRT_TopologyDescription_PlatformName=*/
      mtrt::pjrt::PJRT_TopologyDescription_PlatformName,
      /*PJRT_TopologyDescription_PlatformVersion=*/
      mtrt::pjrt::PJRT_TopologyDescription_PlatformVersion,
      /*PJRT_TopologyDescription_GetDeviceDescriptions=*/
      mtrt::pjrt::PJRT_TopologyDescription_GetDeviceDescriptions,
      /*PJRT_TopologyDescription_Serialize=*/
      mtrt::pjrt::PJRT_TopologyDescription_Serialize,
      /*PJRT_TopologyDescription_Attributes=*/
      mtrt::pjrt::PJRT_TopologyDescription_Attributes,

      /*PJRT_Compile=*/mtrt::pjrt::PJRT_Compile,

      // Always add new fields to the end of the struct. Move fields below to
      // their corresponding places after each major version bump.

      /*PJRT_Executable_OutputElementTypes=*/
      mtrt::pjrt::PJRT_Executable_OutputElementTypes,
      /*PJRT_Executable_OutputDimensions=*/
      mtrt::pjrt::PJRT_Executable_OutputDimensions,

      /*PJRT_Buffer_CopyToMemory=*/mtrt::pjrt::PJRT_Buffer_CopyToMemory,

      /*PJRT_Client_CreateViewOfDeviceBuffer=*/
      mtrt::pjrt::PJRT_Client_CreateViewOfDeviceBuffer,

      /*PJRT_Executable_Fingerprint=*/mtrt::pjrt::PJRT_Executable_Fingerprint,

      /*PJRT_Client_TopologyDescription=*/
      mtrt::pjrt::PJRT_Client_TopologyDescription,

      /*PJRT_Executable_GetCompiledMemoryStats=*/
      mtrt::pjrt::PJRT_Executable_GetCompiledMemoryStats,

      /*PJRT_Memory_Kind_Id=*/mtrt::pjrt::PJRT_Memory_Kind_Id,

      /*PJRT_ExecuteContext_Create=*/mtrt::pjrt::PJRT_ExecuteContext_Create,
      /*PJRT_ExecuteContext_Destroy=*/mtrt::pjrt::PJRT_ExecuteContext_Destroy,
      /*PJRT_Buffer_CopyRawToHost=*/mtrt::pjrt::PJRT_Buffer_CopyRawToHost,
      /*PJRT_AsyncHostToDeviceTransferManager_Destroy=*/
      mtrt::pjrt::PJRT_AsyncHostToDeviceTransferManager_Destroy,
      /*PJRT_AsyncHostToDeviceTransferManager_TransferData=*/
      mtrt::pjrt::PJRT_AsyncHostToDeviceTransferManager_TransferData,
      /*PJRT_Client_CreateBuffersForAsyncHostToDevice=*/
      mtrt::pjrt::PJRT_Client_CreateBuffersForAsyncHostToDevice,
      /*PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer=*/
      mtrt::pjrt::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer,
      /*PJRT_AsyncHostToDeviceTransferManager_Device=*/
      mtrt::pjrt::PJRT_AsyncHostToDeviceTransferManager_Device,
      /*PJRT_AsyncHostToDeviceTransferManager_BufferCount=*/
      mtrt::pjrt::PJRT_AsyncHostToDeviceTransferManager_BufferCount,
      /*PJRT_AsyncHostToDeviceTransferManager_BufferSize=*/
      mtrt::pjrt::PJRT_AsyncHostToDeviceTransferManager_BufferSize,
      /*PJRT_AsyncHostToDeviceTransferManager_SetBufferError=*/
      mtrt::pjrt::PJRT_AsyncHostToDeviceTransferManager_SetBufferError,
      /*PJRT_AsyncHostToDeviceTransferManager_AddMetadata=*/
      mtrt::pjrt::PJRT_AsyncHostToDeviceTransferManager_AddMetadata,
      /*PJRT_Client_DmaMap=*/mtrt::pjrt::PJRT_Client_DmaMap,
      /*PJRT_Client_DmaUnmap=*/mtrt::pjrt::PJRT_Client_DmaUnmap,
  };
  return &api;
}
