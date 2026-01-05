//===- EmitCppSupportFiles.cpp --------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Emits EmitC support files (StandaloneCPP runtime, CMake, test driver) as
/// `executor.file_artifact` operations so they can be serialized into
/// `-artifacts-dir`.
//===----------------------------------------------------------------------===//

#include "mlir-tensorrt/Conversion/HostToEmitC/EmbeddedStandaloneCPP.h"
#include "mlir-tensorrt/Conversion/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir-executor/Executor/IR/Executor.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Path.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_EMITCPPSUPPORTFILESPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

namespace {

/// Create a DenseResourceElementsAttr containing the bytes of `contents`.
static ElementsAttr createI8ElementsAttrFromString(Location loc,
                                                   llvm::StringRef name,
                                                   llvm::StringRef contents) {
  MLIRContext *ctx = loc.getContext();
  auto i8 = IntegerType::get(ctx, 8);
  auto type =
      RankedTensorType::get({static_cast<int64_t>(contents.size())}, i8);
  auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign(
      llvm::ArrayRef<char>(contents.data(), contents.size()));
  auto resAttr = DenseResourceElementsAttr::get(type, name, std::move(blob));
  return ElementsAttr(resAttr);
}

/// Emit an `executor.file_artifact` op containing the UTF-8 bytes of a file.
static void emitTextFileAsArtifact(OpBuilder &b, Location loc,
                                   llvm::StringRef relPath,
                                   llvm::StringRef contents) {
  ElementsAttr data =
      createI8ElementsAttrFromString(loc, /*name=*/relPath, contents);
  b.create<executor::FileArtifactOp>(loc, relPath, data);
}

/// Returns true if the module contains an `emitc.include` for the given header.
static bool hasEmitCInclude(ModuleOp module, llvm::StringRef header) {
  for (auto include : module.getOps<emitc::IncludeOp>()) {
    if (include.getInclude() == header)
      return true;
  }
  return false;
}

/// Find the first top-level EmitC class op (Program wrapper), if present.
static emitc::ClassOp findFirstEmitCClass(ModuleOp module) {
  for (auto cls : module.getOps<emitc::ClassOp>())
    return cls;
  return {};
}

/// Collect names of methods on the Program class that require a TensorRT
/// runtime handle. These are currently emitted as
/// `*_initialize(nvinfer1::IRuntime*)`.
static llvm::SmallVector<std::string>
collectTRTRuntimeInitMethods(emitc::ClassOp cls) {
  llvm::SmallVector<std::string> result;
  if (!cls)
    return result;

  auto isTRTRuntimePtrType = [](Type t) -> bool {
    auto ptr = dyn_cast<emitc::PointerType>(t);
    if (!ptr)
      return false;
    auto opaque = dyn_cast<emitc::OpaqueType>(ptr.getPointee());
    return opaque && opaque.getValue() == "nvinfer1::IRuntime";
  };

  for (auto fn : cls.getOps<emitc::FuncOp>()) {
    if (!fn.getName().ends_with("_initialize"))
      continue;
    FunctionType ft = fn.getFunctionType();
    if (ft.getNumInputs() != 1)
      continue;
    if (!isTRTRuntimePtrType(ft.getInput(0)))
      continue;
    result.push_back(fn.getName().str());
  }

  llvm::sort(result);
  return result;
}

/// Return the best-effort entrypoint method name to invoke for testing.
static std::string inferEntrypointName(emitc::ClassOp cls,
                                       llvm::StringRef requestedEntrypoint) {
  if (!cls)
    return "main";

  auto hasMethod = [&](llvm::StringRef name) -> bool {
    for (auto fn : cls.getOps<emitc::FuncOp>())
      if (fn.getName() == name)
        return true;
    return false;
  };

  if (hasMethod("main"))
    return "main";
  if (!requestedEntrypoint.empty() && hasMethod(requestedEntrypoint))
    return requestedEntrypoint.str();
  // Fall back to the first method that looks like an entrypoint (no args).
  for (auto fn : cls.getOps<emitc::FuncOp>()) {
    FunctionType ft = fn.getFunctionType();
    if (ft.getNumInputs() == 0)
      return fn.getName().str();
  }
  return "main";
}

struct EmitCppSupportFilesPass
    : public mlir::impl::EmitCppSupportFilesPassBase<EmitCppSupportFilesPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Location loc = module.getLoc();

    // If nothing requested, do nothing.
    if (!emitRuntimeFiles && !emitCMakeFile && !emitTestDriver)
      return;

    // Infer runtime requirements from EmitC includes inserted by
    // `convert-host-to-emitc`.
    const bool needsCudaRuntime = hasEmitCInclude(module, "MTRTRuntimeCuda.h");
    const bool needsTensorRTRuntime =
        hasEmitCInclude(module, "MTRTRuntimeTensorRT.h");

    OpBuilder b(module.getContext());
    b.setInsertionPointToStart(module.getBody());

    // Place support files under a stable subdirectory to avoid collisions with
    // other artifacts.
    const std::string supportDir = supportSubdir;
    const std::string runtimeDir = (llvm::Twine(supportDir) + "/runtime").str();

    auto emitRuntimeFile = [&](llvm::StringRef fileName) {
      llvm::StringRef contents =
          mtrt::compiler::emitc_support::getEmbeddedStandaloneCPPFileContents(
              fileName);
      if (contents.empty()) {
        emitError(loc) << "missing embedded StandaloneCPP file: " << fileName;
        return failure();
      }
      std::string relPath = (llvm::Twine(runtimeDir) + "/" + fileName).str();
      emitTextFileAsArtifact(b, loc, relPath, contents);
      return success();
    };

    if (emitRuntimeFiles) {
      // Always emit the minimal core runtime.
      if (failed(emitRuntimeFile("MTRTRuntimeStatus.h")) ||
          failed(emitRuntimeFile("MTRTRuntimeStatus.cpp")) ||
          failed(emitRuntimeFile("MTRTRuntimeCore.h")) ||
          failed(emitRuntimeFile("MTRTRuntimeCore.cpp")))
        return signalPassFailure();

      // Emit CUDA/TensorRT runtime only if required by the module.
      if (needsCudaRuntime) {
        if (failed(emitRuntimeFile("MTRTRuntimeCuda.h")) ||
            failed(emitRuntimeFile("MTRTRuntimeCuda.cpp")))
          return signalPassFailure();
      }
      if (needsTensorRTRuntime) {
        if (failed(emitRuntimeFile("MTRTRuntimeTensorRT.h")) ||
            failed(emitRuntimeFile("MTRTRuntimeTensorRT.cpp")))
          return signalPassFailure();
      }
    }

    // Emit a simple test driver, when possible.
    if (emitTestDriver) {
      emitc::ClassOp programClass = findFirstEmitCClass(module);
      const bool hasProgramClass = static_cast<bool>(programClass);

      const std::string driverPath =
          (llvm::Twine(supportDir) + "/emitc_test_driver.cpp").str();

      std::string outputInclude =
          outputFile.empty() ? std::string("output.cpp")
                             : llvm::sys::path::filename(outputFile).str();

      std::string driver;
      llvm::raw_string_ostream os(driver);
      os << "#include \"MTRTRuntimeCore.h\"\n";
      os << "#include \"MTRTRuntimeStatus.h\"\n";
      if (needsTensorRTRuntime)
        os << "#include <NvInferRuntime.h>\n";
      os << "#include \"" << outputInclude << "\"\n";
      os << "#include <cstdint>\n";
      os << "#include <cstdio>\n\n";

      if (!hasProgramClass) {
        os << "int main() {\n";
        os << "  // No EmitC Program class found; generated driver is a "
              "stub.\n";
        os << "  return 0;\n";
        os << "}\n";
        os.flush();
        emitTextFileAsArtifact(b, loc, driverPath, driver);
      } else {
        const std::string className =
            programClass.getNameAttr().getValue().str();
        const std::string entrypointName =
            inferEntrypointName(programClass, entrypoint);
        const auto trtInitMethods = collectTRTRuntimeInitMethods(programClass);

        if (needsTensorRTRuntime) {
          os << "/// A simple logger that implements TensorRT's logging "
                "interface.\n";
          os << "class StdioLogger : public nvinfer1::ILogger {\n";
          os << "protected:\n";
          os << "  void log(Severity severity, const char *msg) noexcept "
                "override {\n";
          os << "    (void)severity;\n";
          os << "    std::fprintf(stderr, \"%s\\n\", msg);\n";
          os << "  }\n";
          os << "};\n\n";
        }

        os << "int main() {\n";
        if (needsTensorRTRuntime) {
          os << "  StdioLogger logger;\n";
          os << "  nvinfer1::IRuntime *runtime = "
                "nvinfer1::createInferRuntime(logger);\n";
        }
        os << "  " << className << " program;\n";
        os << "  int32_t rc = 0;\n";
        if (needsTensorRTRuntime) {
          for (const auto &m : trtInitMethods) {
            os << "  rc = program." << m << "(runtime);\n";
            os << "  mtrt::abort_on_error(rc);\n";
          }
        }
        os << "  rc = program.initialize();\n";
        os << "  mtrt::abort_on_error(rc);\n";
        os << "  program." << entrypointName << "();\n";
        os << "  rc = program.destroy();\n";
        os << "  mtrt::abort_on_error(rc);\n";
        os << "  return 0;\n";
        os << "}\n";
        os.flush();
        emitTextFileAsArtifact(b, loc, driverPath, driver);
      }
    }

    if (emitCMakeFile) {
      // Emit a minimal example CMakeLists. Supports CUDA and TensorRT when
      // the corresponding paths are provided via cmake defines.
      constexpr llvm::StringRef cmakePath = "CMakeLists.txt";
      std::string cmake;
      llvm::raw_string_ostream os(cmake);
      os << "cmake_minimum_required(VERSION 3.18)\n";
      os << "project(mlir_tensorrt_emitc_example CXX)\n\n";
      os << "set(CMAKE_CXX_STANDARD 17)\n";
      os << "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n\n";

      // Path to the generated EmitC output (header or .cpp).
      llvm::SmallString<128> emitcOutputPath("${CMAKE_CURRENT_LIST_DIR}");
      llvm::sys::path::append(emitcOutputPath, outputFile);
      os << "set(MTRT_EMITC_OUTPUT \"" << emitcOutputPath << "\")\n\n";

      // CUDA support - look for CUDAToolkit when needed.
      if (needsCudaRuntime) {
        os << "# CUDA support\n";
        os << "find_package(CUDAToolkit REQUIRED)\n\n";
      }

      // TensorRT support - use user-provided paths.
      if (needsTensorRTRuntime) {
        os << "# TensorRT support (set TENSORRT_ROOT or provide paths)\n";
        os << "if(NOT TENSORRT_INCLUDE_DIR)\n";
        os << "  if(DEFINED ENV{TENSORRT_ROOT})\n";
        os << "    set(TENSORRT_INCLUDE_DIR \"$ENV{TENSORRT_ROOT}/include\")\n";
        os << "  elseif(TENSORRT_ROOT)\n";
        os << "    set(TENSORRT_INCLUDE_DIR \"${TENSORRT_ROOT}/include\")\n";
        os << "  else()\n";
        os << "    message(FATAL_ERROR \"TENSORRT_INCLUDE_DIR or TENSORRT_ROOT "
              "must be set\")\n";
        os << "  endif()\n";
        os << "endif()\n";
        os << "if(NOT TENSORRT_LIB_DIR)\n";
        os << "  if(DEFINED ENV{TENSORRT_ROOT})\n";
        os << "    set(TENSORRT_LIB_DIR \"$ENV{TENSORRT_ROOT}/lib\")\n";
        os << "  elseif(TENSORRT_ROOT)\n";
        os << "    set(TENSORRT_LIB_DIR \"${TENSORRT_ROOT}/lib\")\n";
        os << "  else()\n";
        os << "    message(FATAL_ERROR \"TENSORRT_LIB_DIR or TENSORRT_ROOT "
              "must be set\")\n";
        os << "  endif()\n";
        os << "endif()\n\n";
      }

      os << "add_executable(emitc_test\n";
      os << "  ${CMAKE_CURRENT_LIST_DIR}/" << supportDir
         << "/emitc_test_driver.cpp\n";
      os << "  ${CMAKE_CURRENT_LIST_DIR}/" << supportDir
         << "/runtime/MTRTRuntimeStatus.cpp\n";
      os << "  ${CMAKE_CURRENT_LIST_DIR}/" << supportDir
         << "/runtime/MTRTRuntimeCore.cpp\n";
      if (needsCudaRuntime)
        os << "  ${CMAKE_CURRENT_LIST_DIR}/" << supportDir
           << "/runtime/MTRTRuntimeCuda.cpp\n";
      if (needsTensorRTRuntime)
        os << "  ${CMAKE_CURRENT_LIST_DIR}/" << supportDir
           << "/runtime/MTRTRuntimeTensorRT.cpp\n";
      os << ")\n\n";

      os << "target_include_directories(emitc_test PRIVATE\n";
      os << "  ${CMAKE_CURRENT_LIST_DIR}/" << supportDir << "/runtime\n";

      // The output file path is relative to the artifacts directory.
      llvm::StringRef parentPath = llvm::sys::path::parent_path(outputFile);
      if (parentPath.empty())
        os << "  ${CMAKE_CURRENT_LIST_DIR}\n";
      else
        os << "  ${CMAKE_CURRENT_LIST_DIR}/" << parentPath << "\n";
      os << ")\n\n";

      // Link libraries.
      if (needsCudaRuntime || needsTensorRTRuntime) {
        os << "target_link_libraries(emitc_test PRIVATE\n";
        if (needsCudaRuntime) {
          os << "  CUDA::cudart\n";
          os << "  CUDA::cuda_driver\n";
        }
        if (needsTensorRTRuntime)
          os << "  nvinfer\n";
        os << ")\n\n";
      }

      // Include directories for CUDA and TensorRT.
      if (needsCudaRuntime || needsTensorRTRuntime) {
        os << "target_include_directories(emitc_test PRIVATE\n";
        if (needsCudaRuntime)
          os << "  ${CUDAToolkit_INCLUDE_DIRS}\n";
        if (needsTensorRTRuntime)
          os << "  ${TENSORRT_INCLUDE_DIR}\n";
        os << ")\n\n";
      }

      // Link directories for TensorRT.
      if (needsTensorRTRuntime) {
        os << "target_link_directories(emitc_test PRIVATE\n";
        os << "  ${TENSORRT_LIB_DIR}\n";
        os << ")\n";
      }

      os.flush();
      emitTextFileAsArtifact(b, loc, cmakePath, cmake);
    }
  }
};

} // namespace
