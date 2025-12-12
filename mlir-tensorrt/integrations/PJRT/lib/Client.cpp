//===- Client.cpp ---------------------------------------------------------===//
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
#include "mlir-tensorrt-pjrt/Client.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaRuntime.h"
#include "mlir-executor/Runtime/Support/CUDAHelpers.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt/Features.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/compile_options.pb.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <numeric>

using namespace mtrt;
using namespace mtrt::pjrt;
using ScopedGuard = std::lock_guard<std::mutex>;

//===----------------------------------------------------------------------===//
// BufferDescriptor
//===----------------------------------------------------------------------===//

BufferDescriptor::BufferDescriptor(
    std::unique_ptr<mtrt::MemRefValue> memRefValue)
    : memRefValue(std::move(memRefValue)) {
  auto seq = llvm::seq<int64_t>(
      0, this->memRefValue->getType().getLayout().getStrides().size());
  minorToMajorOrdering = std::vector<int64_t>(seq.begin(), seq.end());
  llvm::sort(minorToMajorOrdering, [&](int64_t lhs, int64_t rhs) {
    return this->memRefValue->getType().getLayout().getStrides()[lhs] <
           this->memRefValue->getType().getLayout().getStrides()[rhs];
  });
}

bool BufferDescriptor::isEmpty() const {
  return memRefValue->getType().getFootprintSizeInBytes() == 0 ||
         memRefValue->getVoidPtr() == nullptr;
}

//===----------------------------------------------------------------------===//
// Runtime Buffer
//===----------------------------------------------------------------------===//

DeviceBufferDescriptor::~DeviceBufferDescriptor() {}

Status DeviceBufferDescriptor::freeBufferAsync() {
  if (scheduledForDeletion)
    return mtrt::getOkStatus();
  scheduledForDeletion = true;
  memRefValue.reset();
  return mtrt::getOkStatus();
}

bool DeviceBufferDescriptor::isScheduledForDeletion() const {
  return scheduledForDeletion;
}

//===----------------------------------------------------------------------===//
// MemorySpace
//===----------------------------------------------------------------------===//

static std::atomic<int> MemoryIdCounter = 0;

Memory::Memory(PjRtDevice *device, Client *client)
    : id(MemoryIdCounter++), space(MemoryKind::DeviceGlobal), devices({device}),
      client(client) {
  debugString =
      llvm::formatv("MemorySpace<global, id={0}, device={1}>", id,
                    device->getMTRTDevice()->getDescription().getString())
          .str();
}

std::string_view Memory::getDebugString() const { return debugString; }

//===----------------------------------------------------------------------===//
// PjRtDevice
//===----------------------------------------------------------------------===//

/// Construct a Device and associated Memory.
StatusOr<std::pair<std::unique_ptr<PjRtDevice>, std::unique_ptr<Memory>>>
PjRtDevice::make_device_and_memory(Client *client, int32_t cudaDeviceNumber) {
  llvm::ArrayRef<std::unique_ptr<mtrt::Device>> devices =
      client->getRuntimeClient()->getDevices();
  auto it = llvm::find_if(
      devices, [cudaDeviceNumber](const std::unique_ptr<mtrt::Device> &device) {
        return device->getDeviceNumber() == mtrt::HardwareId(cudaDeviceNumber);
      });
  if (it == devices.end())
    return getStatusWithMsg(StatusCode::InternalError,
                            "device index out of range");
  mtrt::Device *mtrtDevice = it->get();
  auto device = std::unique_ptr<PjRtDevice>(new PjRtDevice(client, mtrtDevice));
  auto memory = std::make_unique<Memory>(device.get(), client);
  device->addressableMemories.push_back(memory.get());
  return std::make_pair(std::move(device), std::move(memory));
}

//===----------------------------------------------------------------------===//
// Client Flag Parsing
//===----------------------------------------------------------------------===//

/// Parse the MLIR_TRT_DEBUG environment variable as a comma-separated list of
/// strings. These should be strings like "pjrt" or "runtime" to debug the PJRT
/// and MLIR-TRT runtime components respectively. You can also put specific LLVM
/// pass strings there for debugging the compilation step.
static Status parseDebugFlags() {
  static llvm::once_flag runOnce{};
  Status result = Status::getOk();
  llvm::call_once(runOnce, [&]() {
    std::vector<const char *> argv = {
        "mlir-tensorrt-pjrt", "--mlir-elide-elementsattrs-if-larger=32"};
    mtrt::pjrt::registerPJRTCompilerCLOptions();

    std::string error;
    llvm::raw_string_ostream ss(error);
    if (!llvm::cl::ParseCommandLineOptions(argv.size(), argv.data(),
                                           "MLIR-TRT flags", &ss,
                                           "MLIR_TRT_FLAGS", false)) {
      ss.flush();
      result = mtrt::getInternalErrorStatus(
          "Failed to parse MLIR_TRT_FLAGS options: {0}", error.c_str());
    }
  });
  return result;
}

//===----------------------------------------------------------------------===//
// Runnable
//===----------------------------------------------------------------------===//

mtrt::pjrt::PJRTRunnable::PJRTRunnable(
    std::unique_ptr<mtrt::RuntimeSession> state, PjRtDevice *device)
    : state(std::move(state)), device(device) {}

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

static mtrt::FunctionView getMainFunction(const mtrt::Executable &exe) {
  // Locate the main function.
  llvm::SmallVector<mtrt::FunctionView> funcInfos = exe.getFunctions();
  auto it = std::find_if(
      funcInfos.begin(), funcInfos.end(),
      [](mtrt::FunctionView func) { return func.getName() == "main"; });
  assert(it != funcInfos.end() &&
         "expected executable to contain a main function");
  return *it;
}

mtrt::FunctionSignatureView PJRTExecutable::getEntrypointSignature() const {
  return getMainFunction(*this).getSignature();
}

mtrt::FunctionView PJRTExecutable::getEntrypointFunctionInfo() const {
  return getMainFunction(*this);
}

/// Allocate a set of buffers for the resutls of a givn runnable.
/// The resutls are returned as a vector of pointers. They should be passed back
/// to the caller in the PJRT interface, and the caller should take ownership.
static StatusOr<std::vector<std::unique_ptr<DeviceBufferDescriptor>>>
allocateBuffersForRunnableResults(PJRTRunnable &runnable) {
  std::vector<std::unique_ptr<DeviceBufferDescriptor>> buffers;
  StatusOr<mtrt::FunctionView> mainFunction =
      runnable.getRuntimeSession().getExecutable().getFunction("main");
  if (!mainFunction.isOk())
    return getStatusWithMsg(StatusCode::InternalError,
                            "could not get `main` function from executable");
  mtrt::FunctionView engine = *mainFunction;
  mtrt::FunctionSignatureView sig = engine.getSignature();
  mtrt::Ref<mtrt::RuntimeClient> runtimeClient =
      runnable.getDevice()->getClient()->getRuntimeClient();
  PjRtDevice &device = *runnable.getDevice();
  mtrt::Device &mtrtDevice = *device.getMTRTDevice();
  for (int64_t outIdx = 0, e = sig.getNumResults(); outIdx < e; outIdx++) {
    assert(sig.getResult(outIdx).isa<mtrt::MemRefTypeView>() &&
           "expected memref type");
    mtrt::MemRefTypeView memrefType =
        sig.getResult(outIdx).get<mtrt::MemRefTypeView>();
    mtrt::BufferType bufferType = BufferType::getFromSerializedType(memrefType);
    MTRT_ASSIGN_OR_RETURN(std::unique_ptr<mtrt::MemRefValue> deviceBuffer,
                          runtimeClient->allocateMemRef(
                              bufferType, &mtrtDevice, mtrtDevice.getStream()));
    buffers.push_back(std::make_unique<DeviceBufferDescriptor>(
        std::move(deviceBuffer), &device, device.getDefaultMemory()));
    runnable.getResultBuffers().push_back(buffers.back().get());
  }
  return buffers;
}

//===----------------------------------------------------------------------===//
// LoadedExecutable
//===----------------------------------------------------------------------===//

StatusOr<std::unique_ptr<PJRTLoadedExecutable>>
mtrt::pjrt::PJRTLoadedExecutable::create(
    Client *client, std::unique_ptr<PJRTExecutable> executable,
    llvm::ThreadPoolInterface &threadPool,
    const std::vector<PjRtDevice *> &devices) {
  std::vector<std::unique_ptr<PJRTRunnable>> runnables;
  std::vector<std::vector<std::unique_ptr<DeviceBufferDescriptor>>>
      resultBuffers;

  int numDevices = devices.size();
  MTRT_ASSIGN_OR_RETURN(StatusOr<std::string> ncclUuid,
                        mtrt::getCommunicatorUniqueId());

  std::vector<std::string> runtimeFeaturesToEnable = {"core", "cuda"};
  IF_MLIR_TRT_TARGET_TENSORRT(
      { runtimeFeaturesToEnable.push_back("tensorrt"); });
  IF_MLIR_TRT_ENABLE_NCCL({
    if (numDevices > 1)
      runtimeFeaturesToEnable.push_back("nccl");
  });

  // For each device, setup the required data.
  for (PjRtDevice *device : devices) {
    MTRT_ASSIGN_OR_RETURN(std::unique_ptr<mtrt::DeviceGuard> deviceGuard,
                          device->getMTRTDevice()->createDeviceGuard());
    PJRT_DBGF("%s", "Loading executable:");
    DEBUG_WITH_TYPE("pjrt", mtrt::print(llvm::errs(), executable);
                    std::cerr << std::endl;);

    mtrt::RuntimeSessionOptions options(
        numDevices, device->getCudaDeviceNumber(), *ncclUuid);

    options.enableFeatures(runtimeFeaturesToEnable);
    MTRT_ASSIGN_OR_RETURN(
        std::unique_ptr<mtrt::RuntimeSession> state,
        mtrt::LuaRuntimeSession::create(client->getRuntimeClient(), options,
                                        executable->getView()));

    auto runnable = std::make_unique<PJRTRunnable>(std::move(state), device);
    runnables.push_back(std::move(runnable));
  }

  PJRT_DBGF("%s", "Returning new loaded executable");
  return std::unique_ptr<PJRTLoadedExecutable>(
      new PJRTLoadedExecutable(client, std::move(executable), threadPool,
                               devices, std::move(runnables)));
}

PJRTLoadedExecutable::PJRTLoadedExecutable(
    Client *client, std::unique_ptr<PJRTExecutable> executable,
    llvm::ThreadPoolInterface &threadPool,
    const std::vector<PjRtDevice *> &devices,
    std::vector<std::unique_ptr<PJRTRunnable>> runnables)
    : client(client), executable(std::move(executable)), threadPool(threadPool),
      devices(devices), runnables(std::move(runnables)) {}

static Status
executeExecutable(PJRTRunnable *runnable,
                  llvm::ArrayRef<mtrt::RuntimeValue *> argPtrs,
                  llvm::ArrayRef<mtrt::RuntimeValue *> resultPtrs) {
  int deviceIdx = runnable->getDevice()->getCudaDeviceNumber();
  PJRT_DBGF("Executing on device %d", deviceIdx);
  MTRT_RETURN_IF_ERROR(mtrt::setCurrentCUDADevice(deviceIdx));
  StatusOr<mtrt::FunctionView> mainFunction =
      runnable->getRuntimeSession().getExecutable().getFunction("main");
  if (!mainFunction.isOk())
    return getStatusWithMsg(StatusCode::InternalError,
                            "could not get `main` function from executable");
  MTRT_RETURN_IF_ERROR(runnable->getRuntimeSession().setStream(
      runnable->getDevice()->getStream()));
  MTRT_ASSIGN_OR_RETURN(
      llvm::SmallVector<std::unique_ptr<mtrt::RuntimeValue>> resultValues,
      runnable->getRuntimeSession().executeFunction((*mainFunction).getName(),
                                                    argPtrs, resultPtrs));
  return getOkStatus();
}

Status PJRTLoadedExecutable::execute(ExecutionArgs &args) {
  PJRT_DBGF("executing: %s", std::string(executable->getName()).c_str());
  std::vector<std::thread> threads;
  int numDevices = getRunnables().size();

  if (numDevices == 1)
    return executeExecutable(runnables.front().get(), args.argumentBuffers[0],
                             args.resultBuffers[0]);

  // Launch a thread for each device in the > 1 runnable (>1 device) case.
  llvm::ThreadPoolTaskGroup tasksGroup(threadPool);
  std::atomic<bool> processingFailed(false);
  Status result = getOkStatus();
  for (const auto &[idx, runnable] : llvm::enumerate(getRunnables())) {
    tasksGroup.async([&, idx = idx, runnable = runnable.get()]() {
      if (processingFailed)
        return;
      Status status = executeExecutable(runnable, args.argumentBuffers[idx],
                                        args.resultBuffers[idx]);
      if (!status.isOk()) {
        bool beforeIsFailed = processingFailed.exchange(true);
        if (!beforeIsFailed)
          std::swap(result, status);
      }
    });
  }
  tasksGroup.wait();
  return result;
}

std::unique_ptr<PJRTExecutable>
PJRTLoadedExecutable::getExecutableCopy() const {
  return executable->getPJRTExecutableCopy();
}

StatusOr<std::vector<std::vector<std::unique_ptr<DeviceBufferDescriptor>>>>
PJRTLoadedExecutable::allocateResultBuffers() const {
  // For each device, setup the required data.
  std::vector<std::vector<std::unique_ptr<DeviceBufferDescriptor>>>
      resultBuffers;
  for (const std::unique_ptr<PJRTRunnable> &runnable : runnables) {
    StatusOr<std::vector<std::unique_ptr<DeviceBufferDescriptor>>>
        resultBuffersForDevice = allocateBuffersForRunnableResults(*runnable);
    if (!resultBuffersForDevice.isOk())
      return resultBuffersForDevice.getStatus();
    resultBuffers.emplace_back(std::move(*resultBuffersForDevice));
  }
  return resultBuffers;
}

Status PJRTLoadedExecutable::execute(
    DeviceBufferDescriptor *const *const *argument_lists, unsigned num_args,
    DeviceBufferDescriptor **const *output_lists,
    Event **device_complete_events) {
  auto *loadedExecutable = this;

  // Get the signature.
  StatusOr<mtrt::FunctionView> mainFunction =
      loadedExecutable->getExecutable().getFunction("main");
  assert(mainFunction.isOk() && "main function not present in the executable");
  mtrt::FunctionSignatureView sig = (*mainFunction).getSignature();

  // Vector of execution args.
  PJRTLoadedExecutable::ExecutionArgs executionArgs;

  // Lazily allocate space for result buffers.
  StatusOr<std::vector<std::vector<std::unique_ptr<DeviceBufferDescriptor>>>>
      resultBuffers = loadedExecutable->allocateResultBuffers();
  if (!resultBuffers.isOk())
    return resultBuffers.getStatus();

  // Setup the execution parameters for each device.
  for (unsigned deviceIdx = 0, e = devices.size(); deviceIdx < e; ++deviceIdx) {
    assert(devices[deviceIdx] != nullptr && "expected a valid device");

    executionArgs.argumentBuffers.emplace_back();
    executionArgs.resultBuffers.emplace_back();
    std::vector<mtrt::RuntimeValue *> &deviceArgPtrs =
        executionArgs.argumentBuffers.back();
    std::vector<mtrt::RuntimeValue *> &deviceResultPtrs =
        executionArgs.resultBuffers.back();

    const DeviceBufferDescriptor *const *argsForDevice =
        argument_lists[deviceIdx];
    for (unsigned argIdx = 0; argIdx < num_args; argIdx++) {
      const DeviceBufferDescriptor *buffer = argsForDevice[argIdx];
      PJRT_DBGF("Pushing Arg %u = %lu ", argIdx,
                reinterpret_cast<uintptr_t>(buffer->getVoidPtr()));
      assert(buffer != nullptr && "expected valid buffer");
      assert((buffer->getVoidPtr() != nullptr ||
              buffer->getType().getFootprintSizeInBytes() == 0) &&
             "expected valid device memory pointer");
      deviceArgPtrs.push_back(buffer->getMemRefValue().get());
    }

    DeviceBufferDescriptor **resultsForDevice = output_lists[deviceIdx];
    for (unsigned resultIdx = 0; resultIdx < sig.getNumOutputArgs();
         resultIdx++) {
      assert(resultIdx < (*resultBuffers)[deviceIdx].size());
      DeviceBufferDescriptor *buffer =
          (*resultBuffers)[deviceIdx][resultIdx].release();
      PJRT_DBGF("Pushing ResultArg %u = %lu ", resultIdx,
                reinterpret_cast<uintptr_t>(buffer->getVoidPtr()));

      // Hand over ownership of `buffer` to the caller.
      resultsForDevice[resultIdx] = buffer;
      assert(buffer != nullptr && "expected valid result buffer ");
      assert((buffer->getType().getFootprintSizeInBytes() == 0 ||
              buffer->getVoidPtr() != nullptr) &&
             "expected valid result device memory pointer");
      deviceResultPtrs.push_back(buffer->getMemRefValue().get());
    }
  }

  PJRT_DBGF("%s", "Executing");
  Status executeStatus = loadedExecutable->execute(executionArgs);
  if (!executeStatus.isOk())
    return executeStatus;

  for (unsigned deviceIdx = 0, e = devices.size(); deviceIdx < e; deviceIdx++) {
    PJRT_DBGF("synchronizing on device %u", deviceIdx);

    MTRT_RETURN_IF_ERROR(mtrt::synchronizeCUDAStream(
        reinterpret_cast<uintptr_t>(loadedExecutable->getDevices()[deviceIdx]
                                        ->getStream()
                                        ->getCUDAHandle())));
    // If `device_complete_events` isn't nullptr,
    // `device_complete_events` needs to be the same length as
    // `output_lists` (i.e. of length `num_devices`), and each `PJRT_Event`
    // will become ready once the corresponding device execution is
    // complete. If Execute returns an error, then `device_complete_events`
    // will not be populated. The caller is responsible for calling
    // PJRT_Event_Destroy on the returned PJRT_Event*s.
    if (device_complete_events)
      // TODO: right now we just say we are trivially done
      device_complete_events[deviceIdx] = Event::createReadyEvent().release();
  }
  return getOkStatus();
}

//===----------------------------------------------------------------------===//
// Client
//===----------------------------------------------------------------------===//

StatusOr<std::unique_ptr<Client>> Client::create() {

  RETURN_STATUS_IF_ERROR(parseDebugFlags());

  // Make sure to initialize the CUDA runtime before creating the thread pool.
  MTRT_RETURN_IF_ERROR(mtrt::warmupCUDA());

  auto threadPool = std::make_unique<llvm::StdThreadPool>();

  MTRT_ASSIGN_OR_RETURN(mtrt::Ref<mtrt::RuntimeClient> runtimeClient,
                        mtrt::RuntimeClient::create());

  // Create thread pool. This is used for both MLIRContext's thread pool as
  // well as for other async tasks (never at the same time).
  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<Compiler> compiler,
                        Compiler::create(*threadPool));

  // Find the number of devices. In single-process mode, the "addressable
  // devices" is equivalent to any devices the process can view, but this is
  // not true in multi-process mode.
  MTRT_ASSIGN_OR_RETURN(int32_t numDevices, mtrt::getCUDADeviceCount());

  return std::unique_ptr<Client>(new Client(std::move(runtimeClient),
                                            std::move(threadPool),
                                            std::move(compiler), numDevices));
}

Client::Client(Ref<mtrt::RuntimeClient> runtimeClient,
               std::unique_ptr<llvm::StdThreadPool> threadPool,
               std::unique_ptr<Compiler> compiler, unsigned numDevices)
    : runtimeClient(std::move(runtimeClient)),
      threadPool(std::move(threadPool)), compiler(std::move(compiler)) {

  // Setup device objects. Create a view of the device pointers.
  for (unsigned i = 0; i < numDevices; ++i) {
    auto deviceAndMemory = PjRtDevice::make_device_and_memory(this, i);
    mtrt::cantFail(deviceAndMemory);
    memories.push_back(std::move(deviceAndMemory->second));
    memoriesView.push_back(memories.back().get());
    devices.push_back(std::move(deviceAndMemory->first));
    devicesView.push_back(devices.back().get());
  }
}

Client::~Client() {}

const std::vector<PjRtDevice *> &Client::getDevices() const {
  return devicesView;
}

StatusOr<std::unique_ptr<PJRTExecutable>>
Client::compileMlirProgram(std::string_view mlirIr,
                           const xla::CompileOptionsProto &compileOptions) {
  return compiler->compileMlirModule(mlirIr, compileOptions);
}

/// Select the devices from the available client devices according to the
/// compile options.
static std::optional<std::vector<PjRtDevice *>>
getExecutableDevices(const xla::CompileOptionsProto &compileOptions,
                     llvm::ArrayRef<PjRtDevice *> devices) {

  if (!compileOptions.has_executable_build_options())
    return {};

  if (compileOptions.executable_build_options().device_ordinal() >= 0) {
    PJRT_DBGF("deviceId = %ld",
              compileOptions.executable_build_options().device_ordinal());
    return std::vector<PjRtDevice *>{
        devices[compileOptions.executable_build_options().device_ordinal()]};
  }

  if (!compileOptions.executable_build_options().has_device_assignment())
    return {};

  const xla::DeviceAssignmentProto &deviceAssignment =
      compileOptions.executable_build_options().device_assignment();

  if (deviceAssignment.replica_count() != 1)
    return {};

  std::vector<PjRtDevice *> executableDevices;
  for (const xla::DeviceAssignmentProto::ComputationDevice &deviceList :
       deviceAssignment.computation_devices()) {
    for (int64_t deviceId : deviceList.replica_device_ids()) {
      PJRT_DBGF("deviceId = %ld", deviceId);
      if (static_cast<size_t>(deviceId) >= devices.size())
        return {};
      executableDevices.push_back(devices[deviceId]);
    }
  }
  if (executableDevices.empty())
    return {};

  return executableDevices;
}

/// Compile a MLIR program into a PJRTLoadedExecutable.
StatusOr<std::unique_ptr<PJRTLoadedExecutable>>
Client::compileAndLoadMlirProgram(
    llvm::StringRef mlirIr, const xla::CompileOptionsProto &compileOptions) {

  PJRT_DBGF("compile options = %s", compileOptions.DebugString().c_str());

  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<PJRTExecutable> executable,
                        compiler->compileMlirModule(mlirIr, compileOptions));

  // Limit devices to only those needed for this executable.
  llvm::ArrayRef<uint32_t> processGrid = executable->getProcessorGridShape();
  unsigned numProcesses = std::accumulate(
      processGrid.begin(), processGrid.end(), 1, std::multiplies<>());

  if (numProcesses > getDevices().size())
    return mtrt::getInternalErrorStatus(
        "expected total process grid volume ({0}) <= number of "
        "addressable devices ({1})",
        numProcesses, getDevices().size());

  std::optional<std::vector<PjRtDevice *>> devices =
      getExecutableDevices(compileOptions, getDevices());
  if (!devices)
    devices = getDevices();

  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<PJRTLoadedExecutable> loadedExecutable,
                        PJRTLoadedExecutable::create(this,
                                                     std::move(executable),
                                                     *threadPool, *devices));
  return loadedExecutable;
}

const std::vector<Memory *> &Client::getAddressableMemories() const {
  return memoriesView;
}

StatusOr<std::unique_ptr<DeviceBufferDescriptor>>
Client::getDeviceBufferFromHostBuffer(
    const BufferDescriptor &sourceBuffer, PJRT_HostBufferSemantics semantics,
    PjRtDevice &device, const BufferType &deviceBufferType,
    std::unique_ptr<Event> *doneWithHostBufferEvent) {
  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<mtrt::MemRefValue> deviceBuffer,
                        getRuntimeClient()->copyToDevice(
                            *sourceBuffer.getMemRefValue(),
                            *device.getMTRTDevice(), device.getStream(),
                            doneWithHostBufferEvent));
  return std::make_unique<DeviceBufferDescriptor>(
      std::move(deviceBuffer), &device, device.getDefaultMemory());
}

Status Client::copyDeviceBufferToHost(PjRtDevice &device,
                                      const BufferDescriptor &sourceBuffer,
                                      void *dest, size_t sizeBytes,
                                      std::unique_ptr<Event> *copyDoneEvent) {
  PJRT_DBGF("Copying %lu bytes from CUDA device #%d ptr = %p (%lu bytes) to "
            "host address %p",
            sizeBytes, device.getCudaDeviceNumber(), sourceBuffer.getVoidPtr(),
            sourceBuffer.getType().getFootprintSizeInBytes(), dest);
  MTRT_RETURN_IF_ERROR(
      mtrt::setCurrentCUDADevice(device.getCudaDeviceNumber()));
  MTRT_RETURN_IF_ERROR(mtrt::copyCUDADeviceToHostAsync(
      dest, sourceBuffer.getVoidPtr(), sizeBytes,
      reinterpret_cast<uintptr_t>(device.getStream()->getCUDAHandle())));

  StatusOr<std::unique_ptr<Event>> event = Event::create(device.getStream());
  if (!event.isOk())
    return event.getStatus();

  *copyDoneEvent = std::move(*event);

  return getOkStatus();
}

StatusOr<std::unique_ptr<DeviceBufferDescriptor>>
Client::copyDeviceBufferToOtherDevice(
    PjRtDevice &dstDevice, const DeviceBufferDescriptor &sourceBuffer,
    std::unique_ptr<Event> &copyDoneEvent) {

  std::unique_ptr<DeviceBufferDescriptor> dstBuffer;

  Ref<Stream> destStream = dstDevice.getStream();
  Ref<Stream> sourceStream = sourceBuffer.getDevice().getStream();

  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<mtrt::MemRefValue> dstMemRef,
                        getRuntimeClient()->copyDeviceBufferToOtherDevice(
                            *sourceBuffer.getMemRefValue(),
                            *dstDevice.getMTRTDevice(), copyDoneEvent));
  return std::make_unique<DeviceBufferDescriptor>(
      std::move(dstMemRef), &dstDevice, dstDevice.getDefaultMemory());
}
