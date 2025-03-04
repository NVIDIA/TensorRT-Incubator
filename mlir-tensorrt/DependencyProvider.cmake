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

set(MLIR_TRT_LLVM_COMMIT "6c64c8a6f3f77c30745c751d4163ff6bf2fc323b")

if(NOT MTRT_BUILD_LLVM_FROM_SOURCE)
  message(WARNING "Using 'find_package' to locate pre-built LLVM. Please set MLIR_DIR to the directory containing MLIRConfig.cmake")
else()
  set(patch_dir "${CMAKE_CURRENT_LIST_DIR}/build_tools/patches/mlir")
  nv_register_package(
    NAME LLVM
    VERSION 0.0.20241126
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
      "${patch_dir}/000_fix_bufferization_tensor_encoding_memory_spaces.patch"
      "${patch_dir}/001-mlir-Add-a-null-pointer-check-in-symbol-lookup-11516.patch"
      "${patch_dir}/0003-mlir-EmitC-memref-to-emitc-insert-conversion_casts-1.patch"
      "${patch_dir}/0004-NFC-mlir-emitc-fix-misspelling-in-description-of-emi.patch"
      "${patch_dir}/0005-emitc-func-Set-default-dialect-to-emitc-116297.patch"
      "${patch_dir}/0006-MLIR-EmitC-arith-to-emitc-Fix-lowering-of-fptoui-118.patch"
      "${patch_dir}/0007-mlir-emitc-Add-support-for-C-API-python-binding-to-E.patch"
      "${patch_dir}/0008-mlir-emitc-DCE-unimplemented-decls-121253.patch"
      "${patch_dir}/0009-Re-introduce-Type-Conversion-on-EmitC-121476.patch"
      "${patch_dir}/0010-mlir-emitc-Fix-invalid-syntax-in-example-of-emitc.re.patch"
      "${patch_dir}/0011-mlir-emitc-Don-t-emit-extra-semicolon-after-bracket-.patch"
      "${patch_dir}/0012-mlir-emitc-Expose-emitc-dialect-types-119645.patch"
      "${patch_dir}/0013-mlir-emitc-Support-convert-arith.extf-and-arith.trun.patch"
      "${patch_dir}/0014-EmitC-Allow-arrays-of-size-zero-123292.patch"
      "${patch_dir}/0015-mlir-EmitC-Add-MathToEmitC-pass-for-math-function-lo.patch"
      "${patch_dir}/0016-mlir-emitc-Set-default-dialect-to-emitc-in-ops-with-.patch"
      "${patch_dir}/0017-mlir-emitc-Fix-two-EmitC-bugs.patch"
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
nv_register_package(
  NAME Stablehlo
  VERSION 1.8.0
  GIT_TAG 6e403b1aa6a71f5eaa09cc720e4ad42f692745e6
  GIT_REPOSITORY "https://github.com/openxla/stablehlo.git"
  PATCHES
    "${CMAKE_CURRENT_LIST_DIR}/build_tools/patches/stablehlo/0001-transforms-Fix-simplification-patterns-for-stablehlo.patch"
    "${CMAKE_CURRENT_LIST_DIR}/build_tools/patches/stablehlo/0002-Fix-a-couple-missing-checks-for-static-shapes-in-sta.patch"
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
nv_register_package(
  NAME torch_mlir
  GIT_REPOSITORY https://github.com/llvm/torch-mlir.git
  GIT_TAG 30c519369ed7eabad0282d0f874500a9b41fcbbd
  PATCHES "${CMAKE_CURRENT_LIST_DIR}/build_tools/patches/torch_mlir/torch_mlir.patch"
  EXCLUDE_FROM_ALL TRUE
  # We need to specify an existing directory that is not actually a submodule
  # since GIT_SUBMODULES does not except the empty string due to
  # https://gitlab.kitware.com/cmake/cmake/-/issues/24578
  GIT_SUBMODULES "docs"
  OPTIONS
    "TORCH_MLIR_OUT_OF_TREE_BUILD ON"
    "TORCH_MLIR_ENABLE_STABLEHLO ON"
    "TORCH_MLIR_EXTERNAL_STABLEHLO_DIR find_package"
    "MLIR_DIR ${CMAKE_BINARY_DIR}/lib/cmake/mlir"
    "LLVM_DIR ${LLVM_BINARY_DIR}/lib/cmake/llvm"
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
     "^(MLIRExecutor|MLIRTensorRTDialect|Stablehlo|torch_mlir)$")
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