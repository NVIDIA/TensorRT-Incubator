# This file contains CMake configuration for use when building an LLVM distribution
# for MLIR-TensorRT development.

# CMake Global Configuration
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# LLVM Build Configuration
set(LLVM_BUILD_LLVM_DYLIB OFF CACHE BOOL "")
set(LLVM_BUILD_TESTS ON CACHE BOOL "")
set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
set(LLVM_BUILD_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_GTEST ON CACHE BOOL "")
set(LLVM_CCACHE_BUILD ON CACHE BOOL "")
set(LLVM_ENABLE_LIBCXX OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "")
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
set(LLVM_ENABLE_RTTI OFF CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(LLVM_ENABLE_UNWIND_TABLES OFF CACHE BOOL "")
set(LLVM_ENABLE_Z3_SOLVER OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB ON CACHE BOOL "")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
set(MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "")
set(MLIR_INSTALL_PYTHON_PACKAGES ON CACHE BOOL "")
set(LLVM_FORCE_ENABLE_STATS OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_GO_TESTS OFF CACHE BOOL "")
set(LLVM_INSTALL_TOOLCHAIN_ONLY OFF CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_LINK_LLVM_DYLIB OFF CACHE BOOL "")


# we will build separate llvm distribution in x86 and aarch64, this is faster
# the llvm distribution built for x86 won't run in aarch64 anyway
#set(LLVM_TARGETS_TO_BUILD "X86;AArch64;NVPTX" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "host;NVPTX" CACHE STRING "")

set(MLIR_LINK_LLVM_DYLIB OFF CACHE BOOL "")
set(MLIR_LINK_MLIR_DYLIB OFF CACHE BOOL "")
set(LLVM_APPEND_VC_REV OFF CACHE BOOL "")


set(LLVM_TOOLCHAIN_TOOLS
  mlir-lsp-server
  mlir-opt
  mlir-pdll
  mlir-runner
  mlir-tblgen
  mlir-translate
  mlir-runner
  CACHE STRING "")

set(LLVM_TOOLCHAIN_UTILITIES
    FileCheck
    count
    not
    CACHE STRING "")

set(LLVM_DISTRIBUTION_COMPONENTS
      ${LLVM_TOOLCHAIN_TOOLS}
      ${LLVM_TOOLCHAIN_UTILITIES}
      cmake-exports
      llvm-headers
      llvm-libraries
      mlir-cmake-exports
      mlir-headers
      mlir-libraries
      mlir-python-sources
      MLIRPythonModules
    CACHE STRING "")
