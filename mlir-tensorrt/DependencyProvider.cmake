cmake_minimum_required(VERSION 3.24)

# Bootstrap CPM and package registration/download utilities.
set(CMAKE_POLICY_DEFAULT_CMP0168 NEW)
set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_LIST_DIR}/.cache.cpm" CACHE STRING "")
set(CPM_DONT_UPDATE_MODULE_PATH ON CACHE BOOL "" FORCE)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake")
include(MTRTCPM)
include(MTRTPackageUtils)

# We often need to reference files relative to the top-level MLIR-TensorRT
# directory. Using `CMAKE_CURRENT_LIST_DIR` is not correct inside
# `(PRE|POST)_ADD_HOOK` code, use this variable instead.
set(MTRT_TOP_LEVEL_DIR "${CMAKE_CURRENT_LIST_DIR}")

#-------------------------------------------------------------------------------------
# DLPack
#-------------------------------------------------------------------------------------

nv_register_package(
  NAME DLPack
  GIT_REPOSITORY https://github.com/dmlc/dlpack.git
  GIT_TAG 93c8f2a3c774b84af6f652b1992c48164fae60fc
  DOWNLOAD_ONLY TRUE
  GIT_SHALLOW TRUE
  POST_ADD_HOOK [[
    if(NOT TARGET DLPackHeaderOnly)
      add_library(DLPackHeaderOnly INTERFACE IMPORTED)
      target_include_directories(DLPackHeaderOnly INTERFACE
        $<BUILD_INTERFACE:${DLPack_SOURCE_DIR}/include>)
      add_library(DLPack::Headers ALIAS DLPackHeaderOnly)
    endif()
  ]]
)

#-------------------------------------------------------------------------------------
# TVM-FFI
#-------------------------------------------------------------------------------------

function(mtrt_find_tvm_ffi)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import tvm_ffi.libinfo as l; print(';'.join([l.find_include_path(),l.find_libtvm_ffi()]))"
    OUTPUT_VARIABLE TVM_FFI_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TVM_FFI_RESULT
    ERROR_VARIABLE TVM_FFI_ERROR
  )

  if(NOT TVM_FFI_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find TVM-FFI. Ensure TVM-FFI is installed in your Python environment (try running 'uv sync')"
     "Logs: ${TVM_FFI_ERROR} ${TVM_FFI_PATHS}")

  endif()

  list(GET TVM_FFI_PATHS 0 TVM_FFI_INCLUDE_PATH)
  list(GET TVM_FFI_PATHS 1 TVM_FFI_LIBRARY_PATH)

  # Create imported interface target for headers
  add_library(tvm_ffi_header INTERFACE IMPORTED)
  target_include_directories(tvm_ffi_header INTERFACE
   "$<BUILD_INTERFACE:${TVM_FFI_INCLUDE_PATH}>")

  # Create imported library for libtvm_ffi.so
  add_library(tvm_ffi_shared SHARED IMPORTED)
  set_target_properties(tvm_ffi_shared PROPERTIES
    IMPORTED_LOCATION "${TVM_FFI_LIBRARY_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_FFI_INCLUDE_PATH}"
  )
endfunction()

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
  GIT_TAG v25.9.23
  EXCLUDE_FROM_ALL TRUE
  OPTIONS
    "FLATBUFFERS_BUILD_TESTS OFF"
    "FLATBUFFERS_INSTALL OFF"
  PRE_ADD_HOOK [[
    nv_pkg_append_cxx_flags(-Wno-suggest-override)
    nv_pkg_append_cxx_flags(-Wno-covered-switch-default)
    nv_pkg_append_cxx_flags(-Wno-c++98-compat-extra-semi)
  ]]
)

#-------------------------------------------------------------------------------------
# Stablehlo
#-------------------------------------------------------------------------------------
set(stablehlo_patch_dir "${CMAKE_SOURCE_DIR}/build_tools/patches/stablehlo")
set(MLIR_TRT_STABLEHLO_COMMIT "4bf77d23bd9150782a70d85fda9c12a2dec5328c")


nv_register_package(
  NAME Stablehlo
  URL "https://github.com/openxla/stablehlo/archive/${MLIR_TRT_STABLEHLO_COMMIT}.zip"
  EXCLUDE_FROM_ALL TRUE
  OPTIONS
    "STABLEHLO_ENABLE_BINDINGS_PYTHON ON"
    "STABLEHLO_BUILD_EMBEDDED ON"
  PATCHES
   "${stablehlo_patch_dir}/0001-cmake-Update-usage-of-HandleLLVMOptions-and-LLVM_DEF.patch"
   "${stablehlo_patch_dir}/0002-Fix-ZeroExtent-condition-in-simplification-pattern.patch"
   "${stablehlo_patch_dir}/0003-Remove-explicit-use-of-LLVMSupport.patch"
   "${stablehlo_patch_dir}/0004-Fix-circular-dependence-between-StablehloPasses-and-.patch"
   "${stablehlo_patch_dir}/0005-Modernize-the-stablehlo-convert-to-signless-pass.patch"
   "${stablehlo_patch_dir}/0006-Add-additional-support-for-stablehlo-chlo-ops-to-lin.patch"
   "${stablehlo_patch_dir}/0007-Fix-stablehlo-convert-to-signless-pass-to-handle-sta.patch"

  POST_ADD_HOOK [[
    # Mimic what a StablehloConfig.cmake file would do.
    set(STABLEHLO_INCLUDE_DIRS
      ${Stablehlo_SOURCE_DIR}
      ${Stablehlo_BINARY_DIR})
  ]]
)

#-------------------------------------------------------------------------------------
# Lua
#-------------------------------------------------------------------------------------

macro(lua_set_target_copts target)
    target_compile_options(${target} PRIVATE
      # These come from the Makefile that ships with lua.
      -std=c99 -Wfatal-errors -Wextra -Wshadow
      -Wsign-compare -Wundef -Wwrite-strings -Wredundant-decls
      -Wdisabled-optimization -Wdouble-promotion -Wmissing-declarations
      "$<$<CXX_COMPILER_ID:GNU>:-Wno-pedantic>"
      # We enable -Wall by default globally. Suppress
      # some warnings that will appear in Lua C code.
      "$<$<CXX_COMPILER_ID:Clang>:-Wno-gnu-label-as-value>"
      -Wno-implicit-fallthrough -Wno-cast-qual)
    # TODO: fix these if platform is not linux
    target_compile_definitions(${target} PRIVATE
      LUA_USE_LINUX
      LUA_USE_READLINE
    )
    target_include_directories(${target} PUBLIC
      "$<BUILD_INTERFACE:${Lua_SOURCE_DIR}/src>"
    )
    set_target_properties(${target}
      PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${Lua_BINARY_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${Lua_BINARY_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${Lua_BINARY_DIR}"
    )
  endmacro()

nv_register_package(
  NAME Lua
  URL https://www.lua.org/ftp/lua-5.4.4.tar.gz
  EXCLUDE_FROM_ALL TRUE
  DOWNLOAD_ONLY TRUE
  POST_ADD_HOOK [[
    FILE(GLOB lua_sources ${Lua_SOURCE_DIR}/src/*.c)
    # Remove lua.c (standalone lua interpreter) and onelua.c (a combination of all files).
    list(REMOVE_ITEM lua_sources
      "${Lua_SOURCE_DIR}/src/lua.c"
      "${Lua_SOURCE_DIR}/src/luac.c"
      "${Lua_SOURCE_DIR}/src/onelua.c")
    # Main lua library
    add_library(lua-core EXCLUDE_FROM_ALL ${lua_sources})
    lua_set_target_copts(lua-core)
    target_link_libraries(lua-core PUBLIC dl m)
    mtrt_add_install(lua-core UMBRELLA mtrt-dependencies)
    # Other libs should link `lua::core`.
    add_library(lua::core ALIAS lua-core)
    # Allow building main lua interpreter for whatever reason.
    add_executable(lua-interpreter EXCLUDE_FROM_ALL "${Lua_SOURCE_DIR}/src/lua.c")
    add_executable(luac EXCLUDE_FROM_ALL
      "${Lua_SOURCE_DIR}/src/luac.c"
      ${lua_sources})
    lua_set_target_copts(lua-interpreter)
    lua_set_target_copts(luac)
    # Note that this requires `libreadline-dev` package, we we don't install by
    # default in the devcontainer.
    target_link_libraries(lua-interpreter PRIVATE lua::core m dl readline)
    target_link_libraries(luac PRIVATE lua::core m dl)
    set_target_properties(lua-interpreter PROPERTIES
        OUTPUT_NAME "lua"
    )
  ]]
)

#-------------------------------------------------------------------------------------
# Sol2
#-------------------------------------------------------------------------------------

nv_register_package(
  NAME Sol2
  URL https://github.com/ThePhD/sol2/archive/refs/tags/v3.5.0.tar.gz
  EXCLUDE_FROM_ALL TRUE
  OPTIONS
    "SOL2_ENABLE_INSTALL ON"
    "SOL2_SINGLE OFF"
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

  get_property(NV_CPM_PACKAGES GLOBAL PROPERTY NV_CPM_PACKAGES)

  # We handle finding TensorrT as a special case since it has
  # dedicated find/download logic to allow locating a previously
  # installed version.
  if("${dep_name}" MATCHES "TensorRT")
    list(APPEND mycomp_provider_args ${method} ${dep_name})
    find_tensorrt(
      INSTALL_DIR "${MLIR_TRT_TENSORRT_DIR}"
      DOWNLOAD_VERSION "${MLIR_TRT_DOWNLOAD_TENSORRT_VERSION}"
      MIN_VERSION ""
    )
    list(POP_BACK mycomp_provider_args dep_name method)
    set("${dep_name}_FOUND" TRUE)
  elseif("${dep_name}" STREQUAL "TVMFFI")
    mtrt_find_tvm_ffi()
    set("${dep_name}_FOUND" TRUE)
  # Handle LLVM.
  elseif(MTRT_BUILD_LLVM_FROM_SOURCE AND
         "${dep_name}" MATCHES "^(LLVM)$")
    list(APPEND mycomp_provider_args ${method} ${dep_name})
    nv_add_package(${dep_name})
    list(POP_BACK mycomp_provider_args dep_name method)
    find_package(LLVM ${ARGN} BYPASS_PROVIDER)
  # For all other packages, the logic is the same.
  elseif("${dep_name}" IN_LIST NV_CPM_PACKAGES)
    list(APPEND mycomp_provider_args ${method} ${dep_name})
    nv_add_package("${dep_name}")
    list(POP_BACK mycomp_provider_args dep_name method)
    set("${dep_name}_FOUND" TRUE)
  endif()

endmacro()

cmake_language(
  SET_DEPENDENCY_PROVIDER mtrt_provide_dependency
  SUPPORTED_METHODS
    FIND_PACKAGE
)
