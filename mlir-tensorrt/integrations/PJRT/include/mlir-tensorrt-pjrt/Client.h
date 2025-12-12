//===- Client.h -----------------------------------------------------------===//
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
#ifndef INCLUDE_MLIR_TENSORRT_PJRT_CLIENT
#define INCLUDE_MLIR_TENSORRT_PJRT_CLIENT

#include "mlir-executor/Runtime/API/API.h"
#include "mlir-tensorrt/Compiler/Client.h"
#include "llvm/Support/ThreadPool.h"

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgcc-compat"
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#pragma GCC diagnostic ignored "-Wc++98-compat-extra-semi"
#endif
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/compile_options.pb.h"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#define PJRT_DBGF(fmt, ...)                                                    \
  DEBUG_WITH_TYPE("pjrt", fprintf(stderr, "%s:%d " fmt "\n", __FILE__,         \
                                  __LINE__, __VA_ARGS__))

namespace mlir {
class MLIRContext;
}

namespace mtrt::compiler {
class CompilerClient;
}

namespace mtrt::pjrt {

/// Register the usual MLIR options that can be parsed from MLIR_TRT_FLAGS.
void registerPJRTCompilerCLOptions();

class PJRTExecutable;

//===----------------------------------------------------------------------===//
// Compiler
//===----------------------------------------------------------------------===//

/// A simple opaque interface to the MLIR-TensorRT compiler that returns an
/// opaque executable representing the compiled artifact.
class Compiler {
public:
  static StatusOr<std::unique_ptr<Compiler>>
  create(llvm::ThreadPoolInterface &threadPool);

  ~Compiler() = default;

  /// Given an MLIR program pointed to by `data` (in either MLIR bytecode or
  /// MLIR textual form), compiles the program into a Executable and returns the
  /// Executable pointer.
  StatusOr<std::unique_ptr<PJRTExecutable>>
  compileMlirModule(std::string_view mlirIr,
                    const xla::CompileOptionsProto &compileOptions);

  mlir::MLIRContext *getContext() { return context.get(); }

private:
  explicit Compiler(std::unique_ptr<mlir::MLIRContext> context,
                    std::unique_ptr<mtrt::compiler::CompilerClient> client)
      : context(std::move(context)), client(std::move(client)) {}

  std::unique_ptr<mlir::MLIRContext> context;
  std::unique_ptr<mtrt::compiler::CompilerClient> client;

  /// Mutex to protect the compileMlirModule method. The PJRT user may compile
  /// programs concurrently, but currently we only use a single MLIRContext.
  mutable std::mutex compileMutex;
};

class PjRtDevice;
class Client;

//===----------------------------------------------------------------------===//
// MemorySpace
//===----------------------------------------------------------------------===//

/// Describes different types of memory. Currently there is only GPU memory (one
/// distinct memory per GPU) but in the future there could be Unified or other
/// memory types.
enum class MemoryKind {
  DeviceGlobal = 0,
};

/// A Memory is a descriptor that describes a memory that is accessible by one
/// or more Devices. The corresponding PJRT API object is `PJRT_Memory`.
class Memory {
public:
  /// Constructs a DeviceGlobal memory associated with one device.
  explicit Memory(PjRtDevice *device, Client *client);

  /// Returns the memory space identifier.
  MemoryKind getSpace() const { return space; }

  /// Returns an integer ID corresponding to the memory space identifier value.
  int64_t getId() const { return id; }

  /// Returns a string describing the space.
  std::string_view getDebugString() const;

  static constexpr std::string_view kDeviceKind = "device";
  static constexpr int kDeviceKindId = 0;

  std::string_view getKind() const { return kDeviceKind; }
  int getKindId() const { return kDeviceKindId; }

  /// Returns the devices that can access this memory.
  const std::vector<PjRtDevice *> &getDevices() const { return devices; }

  Client *getClient() const { return client; }

private:
  int32_t id;

  /// The memory kind of memory.
  MemoryKind space{MemoryKind::DeviceGlobal};

  /// The devices that can access this memory.
  std::vector<PjRtDevice *> devices;

  /// A cached string describing the memory.
  std::string debugString;

  std::string kindString;

  Client *client;
};

//===----------------------------------------------------------------------===//
// Device
//===----------------------------------------------------------------------===//

/// A device represents a CUDA device. Each device has a cuda device number
/// (ordinal), at least one CUDA stream, and a device allocator associated with
/// it.
class PjRtDevice {
public:
  /// Construct a Device and associated Memory.
  static StatusOr<
      std::pair<std::unique_ptr<PjRtDevice>, std::unique_ptr<Memory>>>
  make_device_and_memory(Client *client, int32_t cudaDeviceNumber);

private:
  /// Private constructor used during setup.
  explicit PjRtDevice(Client *client, mtrt::Device *mtrtDevice)
      : client(client), mtrtDevice(mtrtDevice) {
    assert(mtrtDevice->getStream() && "expected valid stream");
  }

public:
  ~PjRtDevice() = default;

  const std::vector<Memory *> &getMemories() const {
    return addressableMemories;
  }

  Memory *getDefaultMemory() const {
    assert(!addressableMemories.empty());
    return addressableMemories.front();
  }

  Client *getClient() const { return client; }

  mtrt::Device *getMTRTDevice() const { return mtrtDevice; }

  int32_t getCudaDeviceNumber() const { return mtrtDevice->getDeviceNumber(); }

  Ref<Stream> getStream() const { return mtrtDevice->getStream(); }

private:
  /// Ref to the client.
  Client *client;

  /// Ref to the MTRT device (owned by the RuntimeClient).
  mtrt::Device *mtrtDevice;

  /// Memories addressable by the device.
  std::vector<Memory *> addressableMemories;
};

std::ostream &operator<<(std::ostream &os, const BufferType &t);

//===----------------------------------------------------------------------===//
// BufferDescriptor
//===----------------------------------------------------------------------===//

class BufferDescriptor {
public:
  /// Create a runtime buffer view from a type and a memory pointer.
  BufferDescriptor(std::unique_ptr<mtrt::MemRefValue> memRefValue);

  const BufferType &getType() const { return memRefValue->getType(); }

  void *getVoidPtr() const { return memRefValue->getVoidPtr(); }

  /// Returns true if the buffer is empty based on the type byte count or if the
  /// memory is null.
  bool isEmpty() const;

  llvm::ArrayRef<int64_t> getMinorToMajorOrdering() const {
    return minorToMajorOrdering;
  }

  const std::unique_ptr<mtrt::MemRefValue> &getMemRefValue() const {
    return memRefValue;
  }
  std::unique_ptr<mtrt::MemRefValue> &getMemRefValue() { return memRefValue; }

protected:
  std::unique_ptr<mtrt::MemRefValue> memRefValue;
  std::vector<int64_t> minorToMajorOrdering;
};

//===----------------------------------------------------------------------===//
// DeviceBufferDescriptor
//===----------------------------------------------------------------------===//

/// DeviceBufferDescriptor represents a device buffer.
class DeviceBufferDescriptor : public BufferDescriptor {
public:
  DeviceBufferDescriptor() = delete;
  /// Create a runtime buffer from a type, a memory pointer, and a device
  /// allocator. The `memory` must already be allocated. The allocator is
  /// passed so that it may be used for deallocating the memory when the
  /// DeviceBufferDescriptor is destroyed.
  DeviceBufferDescriptor(std::unique_ptr<mtrt::MemRefValue> memRefValue,
                         PjRtDevice *device, Memory *memorySpace)
      : BufferDescriptor(std::move(memRefValue)), device(device),
        memorySpace(memorySpace), scheduledForDeletion(false) {}

  ~DeviceBufferDescriptor();

  PjRtDevice &getDevice() const {
    assert(device != nullptr && "expected valid device");
    return *device;
  }

  /// Free the memory using `cudaStreamAsync` on the stream associated with the
  /// device. This sets the `scheduledForDeletion` flag to true.
  Status freeBufferAsync();

  /// Returns whether `freeBufferAsync` was previously called.
  bool isScheduledForDeletion() const;

  Memory *getMemorySpace() const { return memorySpace; }

private:
  PjRtDevice *device;
  Memory *memorySpace;
  bool scheduledForDeletion;
};

class Client;

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

/// An Executable encapsulates the result of compiling a StableHLO or ONNX
/// module using mlir-tensorrt.
class PJRTExecutable : public mtrt::Executable {
public:
  explicit PJRTExecutable(mtrt::Executable exe)
      : mtrt::Executable(std::move(exe)) {
    assert(this->getProcessorGridShape().size() == 2 &&
           "expected rank-2 process grid shape");
  }

  std::unique_ptr<PJRTExecutable> getPJRTExecutableCopy() const {
    std::unique_ptr<mtrt::Executable> copy = this->getCopy();
    return std::make_unique<PJRTExecutable>(std::move(*copy));
  }

  /// Return the signature of the function described by this executable.
  mtrt::FunctionSignatureView getEntrypointSignature() const;
  mtrt::FunctionView getEntrypointFunctionInfo() const;
};

//===----------------------------------------------------------------------===//
// PJRTRunnable
//===----------------------------------------------------------------------===//

/// A `PJRTRunnable` is just a packaged rt::LoadedExecutable and a device, and
/// result buffers. It is specific to a particular device.
struct PJRTRunnable {
  PJRTRunnable(std::unique_ptr<mtrt::RuntimeSession> state, PjRtDevice *device);

public:
  PjRtDevice *getDevice() { return device; }

  std::vector<DeviceBufferDescriptor *> getResultBuffers() {
    return resultBuffers;
  }

  mtrt::RuntimeSession &getRuntimeSession() { return *state; }

private:
  std::unique_ptr<mtrt::RuntimeSession> state;
  /// A non-owned reference to the device this runnable is associated with.
  PjRtDevice *device;
  /// The list of device buffers that will hold the results of this execution.
  /// These are not owned by the Runnable as they may persist beyond the
  /// lifetime of the engine.
  std::vector<DeviceBufferDescriptor *> resultBuffers;
};

//===----------------------------------------------------------------------===//
// PJRTLoadedExecutable
//===----------------------------------------------------------------------===//

/// A PJRTLoadedExecutable is comprised of a PJRTExecutable plus
/// a set of devices and one PJRTRunnable per device.
class PJRTLoadedExecutable {
private:
  PJRTLoadedExecutable(Client *client,
                       std::unique_ptr<PJRTExecutable> executable,
                       llvm::ThreadPoolInterface &threadPool,
                       const std::vector<PjRtDevice *> &devices,
                       std::vector<std::unique_ptr<PJRTRunnable>> runnables);

  // LoadedExecutable cannot be copied or default constructed.
  PJRTLoadedExecutable() = delete;
  PJRTLoadedExecutable(const PJRTLoadedExecutable &) = delete;
  PJRTLoadedExecutable operator=(const PJRTLoadedExecutable &) = delete;

public:
  static StatusOr<std::unique_ptr<PJRTLoadedExecutable>>
  create(Client *client, std::unique_ptr<PJRTExecutable> executable,
         llvm::ThreadPoolInterface &threadPool,
         const std::vector<PjRtDevice *> &devices);

  std::vector<std::unique_ptr<PJRTRunnable>> &getRunnables() {
    return runnables;
  }

  std::vector<PjRtDevice *> &getDevices() { return devices; }

  StatusOr<std::vector<std::vector<std::unique_ptr<DeviceBufferDescriptor>>>>
  allocateResultBuffers() const;

  struct ExecutionArgs {
    std::vector<std::vector<mtrt::RuntimeValue *>> argumentBuffers;
    std::vector<std::vector<mtrt::RuntimeValue *>> resultBuffers;
  };

  /// Execute the executable on the given devices.
  Status execute(ExecutionArgs &args);

  /// Execute the executable using the provided arguments.
  Status execute(DeviceBufferDescriptor *const *const *argument_lists,
                 unsigned num_args,
                 DeviceBufferDescriptor **const *output_lists,
                 Event **device_complete_events);

  /// Returns a copy of the executable.
  std::unique_ptr<PJRTExecutable> getExecutableCopy() const;

  mtrt::ExecutableView getExecutable() const { return executable->getView(); }

  Client *getClient() const { return client; }

private:
  Client *client;

  std::unique_ptr<PJRTExecutable> executable;

  /// The thread pool used by the loaded executable during execution. This is
  /// used if the number of devices > 1.
  llvm::ThreadPoolInterface &threadPool;

  /// Reference to devices of size `num_replicas*num_partitions`.
  std::vector<PjRtDevice *> devices;

  /// Reference to runnables, one per device.
  std::vector<std::unique_ptr<PJRTRunnable>> runnables;
};

//===----------------------------------------------------------------------===//
// Client
//===----------------------------------------------------------------------===//

class Client {
public:
  /// Client constructor will find all available CUDA devices.
  static StatusOr<std::unique_ptr<Client>> create();

  ~Client();

  /// Return a list of devices addressable by this client.
  const std::vector<PjRtDevice *> &getDevices() const;

  /// Return a list of memories addressable by this client.
  const std::vector<Memory *> &getAddressableMemories() const;

  /// Compile a MLIR program into a PJRTExecutable.
  StatusOr<std::unique_ptr<PJRTExecutable>>
  compileMlirProgram(std::string_view mlirIr,
                     const xla::CompileOptionsProto &compileOptions);

  /// Compile a MLIR program into a PJRTLoadedExecutable.
  StatusOr<std::unique_ptr<PJRTLoadedExecutable>>
  compileAndLoadMlirProgram(llvm::StringRef mlirIr,
                            const xla::CompileOptionsProto &compileOptions);

  /// Allocate a device buffer on `device` with enough memory to hold
  /// `sourceBuffer` and perform the copy synchronously. Returns the device
  /// buffer. Places a trivially signaled event in `doneWithHostBuffferEvent`.
  /// Currently `semantics` is ignored.
  /// TODO: implement async version
  /// TODO: handle semantics
  StatusOr<std::unique_ptr<DeviceBufferDescriptor>>
  getDeviceBufferFromHostBuffer(const BufferDescriptor &sourceBuffer,
                                PJRT_HostBufferSemantics semantics,
                                PjRtDevice &device,
                                const BufferType &deviceBufferType,
                                std::unique_ptr<Event> *hostDoneEvent);

  /// Copy the device buffer `sourceBuffer` to the host memory at `dest` using
  /// the stream associated with `device`. Places an event in `copyDoneEvent`
  /// that will signal when the copy is complete.
  Status copyDeviceBufferToHost(PjRtDevice &device,
                                const BufferDescriptor &sourceBuffer,
                                void *dest, size_t sizeBytes,
                                std::unique_ptr<Event> *copyDoneEvent);

  /// Copy the device buffer `sourceBuffer` to another device `dstDevice`.
  /// Returns a newly allocated device buffer on the destination device. Places
  /// an event in `copyDoneEvent` that will signal when the copy is complete.
  StatusOr<std::unique_ptr<DeviceBufferDescriptor>>
  copyDeviceBufferToOtherDevice(PjRtDevice &dstDevice,
                                const DeviceBufferDescriptor &sourceBuffer,
                                std::unique_ptr<Event> &copyDoneEvent);

  Ref<mtrt::RuntimeClient> getRuntimeClient() const { return runtimeClient; }

private:
  Client(Ref<mtrt::RuntimeClient> runtimeClient,
         std::unique_ptr<llvm::StdThreadPool> threadPool,
         std::unique_ptr<Compiler> compiler, unsigned numDevices);

  Ref<mtrt::RuntimeClient> runtimeClient;

  /// A thread pool for async execution of tasks.
  std::unique_ptr<llvm::StdThreadPool> threadPool;

  std::unique_ptr<Compiler> compiler;

  /// List of owned visible CUDA devices.
  std::vector<std::unique_ptr<PjRtDevice>> devices;

  /// A cached view of the ownedDevices.
  std::vector<PjRtDevice *> devicesView;

  /// The addressable memories of the client. This consists of a space for each
  /// device.
  std::vector<std::unique_ptr<Memory>> memories;

  /// A cached view of the `memories`.
  std::vector<Memory *> memoriesView;
};

using PjRtDeviceDescription = mtrt::DeviceDescription;

} // namespace mtrt::pjrt

#endif // INCLUDE_MLIR_TENSORRT_PJRT_CLIENT
