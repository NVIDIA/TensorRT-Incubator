cmake_minimum_required(VERSION 3.24)

# Bootstrap CPM
set(CMAKE_POLICY_DEFAULT_CMP0168 NEW)
set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/.cache.cpm" CACHE STRING "")
set(CPM_DONT_UPDATE_MODULE_PATH ON CACHE BOOL "" FORCE)

include(build_tools/cmake/CPM.cmake)
include(build_tools/cmake/PackageUtils.cmake)
include(build_tools/cmake/Dependencies.cmake)

#-------------------------------------------------------------------------------------
# Declare the LLVM dependency.
#-------------------------------------------------------------------------------------
set(MTRT_BUILD_LLVM_FROM_SOURCE ON)
if("${MLIR_TRT_USE_LLVM}" STREQUAL "prebuilt")
  set(MTRT_BUILD_LLVM_FROM_SOURCE OFF)
endif()

set(MLIR_TRT_LLVM_COMMIT "f137c3d592e96330e450a8fd63ef7e8877fc1908")

set(mlir_patch_dir "${CMAKE_CURRENT_LIST_DIR}/build_tools/patches/mlir")

if(NOT MTRT_BUILD_LLVM_FROM_SOURCE)
  message(WARNING "Using 'find_package' to locate pre-built LLVM. Please set MLIR_DIR to the directory containing MLIRConfig.cmake")
else()

  nv_register_package(
    NAME LLVM
    URL "https://github.com/llvm/llvm-project/archive/${MLIR_TRT_LLVM_COMMIT}.zip"
    EXCLUDE_FROM_ALL TRUE
    SOURCE_SUBDIR "llvm"
    OPTIONS
      "LLVM_ENABLE_PROJECTS mlir"
      "MLIR_ENABLE_BINDINGS_PYTHON ON"
      "LLVM_TARGETS_TO_BUILD host"
      "LLVM_ENABLE_ASSERTIONS ON"
      # Never append VCS revision information
      "LLVM_APPEND_VC_REV OFF"
      # Don't mixup LLVM targets with our project's installation/packaging.
      "LLVM_INSTALL_TOOLCHAIN_ONLY ON"
    PATCHES
      "${mlir_patch_dir}/0005-mlir-memref-Fix-memref.global-overly-constrained-ver.patch"
      "${mlir_patch_dir}/0006-mlir-emitc-Fix-two-EmitC-bugs.patch"
      "${mlir_patch_dir}/0009-mlir-Support-FileLineColRange-in-LLVM-debug-translat.patch"
      "${mlir_patch_dir}/0011-MLIR-Fix-bufferization-interface-for-tensor-reshape.patch"
    # Set the CPM cache key to the Git hash for easy navigation.
    PRE_ADD_HOOK [[
      list(APPEND _vap_UNPARSED_ARGUMENTS
        CUSTOM_CACHE_KEY "archive-${MLIR_TRT_LLVM_COMMIT}")
    ]]
    POST_ADD_HOOK [[
      find_path(LLVM_DIR NAMES LLVMConfig.cmake REQUIRED
        HINTS "${LLVM_BINARY_DIR}/lib/cmake/llvm")
      find_path(MLIR_DIR NAMES MLIRConfig.cmake REQUIRED
        HINTS "${CMAKE_BINARY_DIR}/lib/cmake/mlir")
      find_path(MLIR_CMAKE_DIR
        NAMES AddMLIR.cmake
        HINTS
          "${LLVM_SOURCE_DIR}/../mlir/cmake/modules"
          "${LLVM_SOURCE_DIR}/mlir/cmake/modules"
        REQUIRED
      )
      list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

      set(MLIR_MAIN_SRC_DIR "${LLVM_SOURCE_DIR}/mlir" CACHE STRING "" FORCE)

      if(TARGET MLIRPythonExtension.Core)
        get_property(mlir_core_pybind_capi_embed
          TARGET MLIRPythonExtension.Core
          PROPERTY mlir_python_EMBED_CAPI_LINK_LIBS)
        list(FIND mlir_core_pybind_capi_embed MLIRCAPITransforms item_index)
        if(item_index EQUAL -1)
          set_property(TARGET MLIRPythonExtension.Core
            APPEND PROPERTY mlir_python_EMBED_CAPI_LINK_LIBS MLIRCAPITransforms)
        endif()
      endif()
    ]]
  )
endif()

#-------------------------------------------------------------------------------------
# Flatbuffers
#-------------------------------------------------------------------------------------

# Downstream targets should depend on `FlatBuffers::FlatBuffers` and flatbuffer
# schema compilation custom commands should use `flatc` in their command.
nv_register_package(
  NAME Flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG v25.2.10
  EXCLUDE_FROM_ALL TRUE
  OPTIONS
    "FLATBUFFERS_BUILD_TESTS OFF"
    "FLATBUFFERS_INSTALL ON"
    "CMAKE_CXX_FLAGS -Wno-suggest-override"
)

#-------------------------------------------------------------------------------------
# Stablehlo
#-------------------------------------------------------------------------------------
set(stablehlo_patch_dir "${CMAKE_SOURCE_DIR}/build_tools/patches/stablehlo")

nv_register_package(
  NAME Stablehlo
  VERSION 1.9.3
  GIT_TAG 4bf77d23bd9150782a70d85fda9c12a2dec5328c
  GIT_REPOSITORY "https://github.com/openxla/stablehlo.git"
  PATCHES
    "${stablehlo_patch_dir}/0001-Fix-a-couple-missing-checks-for-static-shapes-in-sta.patch"
    "${stablehlo_patch_dir}/0002-cmake-Update-usage-of-HandleLLVMOptions-and-LLVM_DEF.patch"
    "${stablehlo_patch_dir}/0004-Fix-ZeroExtent-condition-in-simplification-pattern.patch"
    "${stablehlo_patch_dir}/0006-Remove-explicit-use-of-LLVMSupport.patch"
    "${stablehlo_patch_dir}/0007-Fix-circular-dependence-between-StablehloPasses-and-.patch"
  OPTIONS
    "STABLEHLO_ENABLE_BINDINGS_PYTHON ON"
    "STABLEHLO_BUILD_EMBEDDED ON"
  POST_ADD_HOOK [[
    # Mimic what a StablehloConfig.cmake file would do.
    set(STABLEHLO_INCLUDE_DIRS
      ${Stablehlo_SOURCE_DIR}
      ${Stablehlo_BINARY_DIR})
  ]]
)

#-------------------------------------------------------------------------------------
# MLIRTensorRTCommon
#
# MLIRTensorRTCommon is a sub-project that contains components used across the
# other sub-projects like MLIRExecutor and MLIRTensorRTDialect.
#-------------------------------------------------------------------------------------

nv_register_package(
  NAME MLIRTensorRTCommon
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/common"
)

# -----------------------------------------------------------------------------
# NVTX
# -----------------------------------------------------------------------------

nv_register_package(
  NAME NVTX
  GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
  GIT_TAG v3.1.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR c
  EXCLUDE_FROM_ALL TRUE
  DOWNLOAD_ONLY TRUE
  POST_ADD_HOOK [[
    if(NOT TARGET nvtx3-cpp)
      add_library(nvtx3-cpp INTERFACE IMPORTED)
      target_include_directories(nvtx3-cpp INTERFACE
        "$<BUILD_INTERFACE:${NVTX_SOURCE_DIR}/c/include>")
      # Ignore some warnings due to NVTX3 code style.
      target_compile_options(nvtx3-cpp INTERFACE
        -Wno-missing-braces)
    endif()
  ]]
)

#-------------------------------------------------------------------------------------
# MLIR-Executor
#
# We build MLIR-Executor as an independent sub-project
#-------------------------------------------------------------------------------------
nv_register_package(
  NAME MLIRExecutor
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/executor"
  POST_ADD_HOOK [[
    # Mimic what would be in MLIRExecutorConfig.cmake
    set(MLIR_EXECUTOR_INCLUDE_DIRS
      "${MLIRExecutor_SOURCE_DIR}/include"
      "${MLIRExecutor_BINARY_DIR}/include")
  ]]
)

#-------------------------------------------------------------------------------------
# MLIRTensorRTDialect
#
# We build MLIR-TensorRT-Dialect as an independent sub-project
#-------------------------------------------------------------------------------------
nv_register_package(
  NAME MLIRTensorRTDialect
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/tensorrt"
  POST_ADD_HOOK [[
    # Mimic what would be in TensorRTDialectConfig.cmake
    set(MLIR_TENSORRT_DIALECT_INCLUDE_DIRS
      "${MLIRTensorRTDialect_SOURCE_DIR}/include"
      "${MLIRTensorRTDialect_BINARY_DIR}/include")
  ]]
)

#-------------------------------------------------------------------------------------
# Torch-MLIR
#-------------------------------------------------------------------------------------
set(MLIR_TRT_TORCH_MLIR_COMMIT "9f2ba5abaa85cefd95cc85579fafd0c53c1101e8")
nv_register_package(
  NAME torch_mlir
  URL "https://github.com/llvm/torch-mlir/archive/${MLIR_TRT_TORCH_MLIR_COMMIT}.zip"
  # We need to specify an existing directory that is not actually a submodule
  # since GIT_SUBMODULES does not except the empty string due to
  # https://gitlab.kitware.com/cmake/cmake/-/issues/24578
  GIT_SUBMODULES "docs"
  DOWNLOAD_ONLY TRUE

  POST_ADD_HOOK [[
    add_subdirectory(
        ${CMAKE_SOURCE_DIR}/third_party/torch-mlir-cmake
        ${CMAKE_BINARY_DIR}/_deps/torch_mlir-build
        EXCLUDE_FROM_ALL
    )
  ]]
)

#-------------------------------------------------------------------------------------
# Dependency Provider Main Logic
#-------------------------------------------------------------------------------------

# Both FIND_PACKAGE and FETCHCONTENT_MAKEAVAILABLE_SERIAL methods provide
# the package or dependency name as the first method-specific argument.
macro(mtrt_provide_dependency method dep_name)
  cmake_parse_arguments(_pargs "" "" "COMPONENTS" ${ARGN})

  # We handle finding TensorrT as a special case since it has
  # dedicated find/download logic to allow locating a previously
  # installed version.
  if("${dep_name}" MATCHES "TensorRT")
    find_tensorrt(
      INSTALL_DIR "${MLIR_TRT_TENSORRT_DIR}"
      DOWNLOAD_VERSION "${MLIR_TRT_DOWNLOAD_TENSORRT_VERSION}"
      MIN_VERSION ""
    )
    set("${dep_name}_FOUND" TRUE)
  endif()

  if("${dep_name}" MATCHES
     "^(MLIRExecutor|MLIRTensorRTDialect|Stablehlo|torch_mlir|NVTX|MLIRTensorRTCommon)$")
    nv_add_package("${dep_name}")
    set("${dep_name}_FOUND" TRUE)
  endif()

  if("${dep_name}" MATCHES
     "^(Flatbuffers)$")
    nv_add_package("${dep_name}")
    set("${dep_name}_FOUND" TRUE)
  endif()

  # If we invoke 'find_package(MLIR)' prior to 'find_package(LLVM)',
  # search for LLVM first, since the LLVM package actually provides
  # the MLIR config when building from source.
  if(MTRT_BUILD_LLVM_FROM_SOURCE)
    if(("${dep_name}" MATCHES "MLIR") AND (NOT MLIR_FOUND))
      nv_add_package(LLVM)
      find_package(LLVM CONFIG REQUIRED BYPASS_PROVIDER)
      find_package(MLIR ${ARGN} BYPASS_PROVIDER)
    endif()

    # Handle LLVM. We want to invoke find_package after
    # adding it.
    if("${dep_name}" MATCHES "^(LLVM)$")
      nv_add_package(${dep_name})
      find_package(LLVM ${ARGN} BYPASS_PROVIDER)
    endif()
  endif()

endmacro()

cmake_language(
  SET_DEPENDENCY_PROVIDER mtrt_provide_dependency
  SUPPORTED_METHODS
    FIND_PACKAGE
)