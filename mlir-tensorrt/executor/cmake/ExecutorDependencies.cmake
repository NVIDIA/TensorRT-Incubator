include(CMakeParseArguments)

#-------------------------------------------------------------------------------------
# Wrapper around CPMAddPackage This functions exactly like CPMAddPackage.
# However, if the GROUP_CICD_TOKEN environment variable is set, then it will
# rewrite ssh:// URIs to https and use the token as auth.
#-------------------------------------------------------------------------------------
macro(mlir_executor_add_package)
  CPMAddPackage(
    ${ARGN}
  )
endmacro()

#-------------------------------------------------------------------------------------
# Downloads FlatBuffers release and adds it to the build. Downstream targets
# should depend on `FlatBuffers::FlatBuffers` and flatbuffer schema compilation
# custom commands should use `flatc` in their command.
#-------------------------------------------------------------------------------------
function(mlir_executor_add_flatbuffers)
  mlir_executor_add_package(
    NAME flatbuffers
    URL https://github.com/google/flatbuffers/archive/refs/tags/v23.5.26.tar.gz
    URL_HASH MD5=2ef00eaaa86ab5e9ad5eafe09c2e7b60
    EXCLUDE_FROM_ALL TRUE
    OPTIONS
      "FLATBUFFERS_BUILD_TESTS OFF"
      "FLATBUFFERS_INSTALL ON"
      "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Wno-covered-switch-default -Wno-suggest-override"
  )
endfunction()

#-------------------------------------------------------------------------------------
# Downlaods the Lua (5.4) source and sets up targets. For depending on these
# targets, use `lua::core` for lua library. Also declares target
# `lua-interpreter` for the plain `lua` executable, but this is excluded from
# the 'all' target.
#-------------------------------------------------------------------------------------
function(mlir_executor_add_lua)
  if(TARGET lua-core)
    return()
  endif()
  CPMAddPackage(
    NAME lua
    URL https://www.lua.org/ftp/lua-5.4.4.tar.gz
    URL_HASH MD5=bd8ce7069ff99a400efd14cf339a727b
    DOWNLOAD_ONLY
  )

  macro(lua_set_target_copts target)
    target_compile_options(${target} PRIVATE
      # These come from the Makefile that ships with lua.
      -std=c99
      -Wfatal-errors
      -Wextra
      -Wshadow
      -Wsign-compare
      -Wundef
      -Wwrite-strings
      -Wredundant-decls
      -Wdisabled-optimization
      -Wdouble-promotion
      -Wmissing-declarations
      # We enable -Wall by default globally. Suppress
      # some warnings that will appear in Lua C code.
      -Wno-gnu-label-as-value
      -Wno-implicit-fallthrough
      -Wno-cast-qual)
    # TODO: fix these if platform is not linux
    target_compile_definitions(${target} PRIVATE
      LUA_USE_LINUX
      LUA_USE_READLINE
    )
    target_include_directories(${target} PUBLIC
      "$<BUILD_INTERFACE:${lua_SOURCE_DIR}/src>"
    )
    set_target_properties(${target}
      PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${lua_BINARY_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${lua_BINARY_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${lua_BINARY_DIR}"
    )
  endmacro()

  FILE(GLOB lua_sources ${lua_SOURCE_DIR}/src/*.c)
  # Remove lua.c (standalone lua interpreter) and onelua.c (a combination of all files).
  list(REMOVE_ITEM lua_sources "${lua_SOURCE_DIR}/src/lua.c" "${lua_SOURCE_DIR}/src/onelua.c")
  # Main lua library
  add_library(lua-core STATIC EXCLUDE_FROM_ALL ${lua_sources})
  lua_set_target_copts(lua-core)
  target_link_libraries(lua-core PUBLIC dl)
  # Other libs should link `lua::core`.
  add_library(lua::core ALIAS lua-core)
  # Allow building main lua interpreter for whatever reason.
  add_executable(lua-interpreter EXCLUDE_FROM_ALL "${lua_SOURCE_DIR}/src/lua.c")
  lua_set_target_copts(lua-interpreter)
  # Note that this requires `libreadline-dev` package, we we don't install by
  # default in the devcontainer.
  target_link_libraries(lua-interpreter PRIVATE lua::core m dl readline)
  set_target_properties(lua-interpreter PROPERTIES
      OUTPUT_NAME "lua"
  )
  add_mlir_library_install(lua-core)
endfunction()

#-------------------------------------------------------------------------------------
# Downlaods the Sol2 source and sets up targets. For depending on Sol2, use
# `sol2::sol2`.
#-------------------------------------------------------------------------------------
macro(mlir_executor_add_sol2)
  CPMAddPackage(
    NAME sol2
    URL "https://github.com/ThePhD/sol2/archive/eab1430ccdbf61a0d61d11bf86b4975838dcfb9a.zip"
    URL_HASH MD5=c15a2db4cf1f859154bc75542ef8bff5
    EXCLUDE_FROM_ALL TRUE
    OPTIONS
      "SOL2_ENABLE_INSTALL ON"
      "SOL2_SINGLE OFF"
  )
  target_compile_definitions(sol2 INTERFACE SOL_ALL_SAFETIES_ON=1)
endmacro()

# -----------------------------------------------------------------------------
# Downloads NVTX from Github and adds it to the build
# -----------------------------------------------------------------------------
function(mlir_executor_add_nvtx)
  CPMAddPackage(
    NAME nvtx
    GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
    GIT_TAG v3.1.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR c
    EXCLUDE_FROM_ALL TRUE
    DOWNLOAD_ONLY TRUE
  )
  add_library(nvtx3-cpp INTERFACE IMPORTED)
  target_include_directories(nvtx3-cpp INTERFACE
    "$<BUILD_INTERFACE:${nvtx_SOURCE_DIR}/c/include>")
  # Ignore some warnings due to NVTX3 code style.
  target_compile_options(nvtx3-cpp INTERFACE
    -Wno-missing-braces)
endfunction()


# -----------------------------------------------------------------------------
# Finds the NCCL headers and creates an interface target `NCCL`.
# -----------------------------------------------------------------------------
function(mlir_executor_find_nccl)
  find_path(nccl_header
    NAMES nccl.h
    REQUIRED)
  # libnccl_static.a for static
  find_library(NcclLibPath NAMES nccl
    REQUIRED
  )

  message(STATUS "Found NCCL headers in ${nccl_header}")
  message(STATUS "Found NCCL libs in ${NcclLibPath}")

  if(NOT TARGET NCCL)
    add_library(NCCL SHARED IMPORTED)
    set_property(TARGET NCCL PROPERTY IMPORTED_LOCATION
      "${NcclLibPath}")
    target_include_directories(NCCL INTERFACE
        $<BUILD_INTERFACE:${nccl_header}>)
  endif()
endfunction()

# -----------------------------------------------------------------------------
# Find `libnvptxcompiler_static.a`. Then the `.ctor` sections in the library
# need to be patched to `.init_array` sections. Otherwise, this causes a
# segfault when linking with LLD since global initialization of certain objects
# in the library will not occur when the executable is launched.
# The patch step should be compatible with all linkers, so we copy the library
# and do the section re-naming unconditionally.
# -----------------------------------------------------------------------------
function(mlir_executor_find_and_patch_libnvptxcompiler target_name)
  find_library(NvPtxCompilerLibPath NAMES nvptxcompiler_static
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATHS ${CUDAToolkit_LIBRARY_DIR}
    REQUIRED
  )

  # Run the patch step.
  file(COPY "${NvPtxCompilerLibPath}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
  set(dstPath "${CMAKE_CURRENT_BINARY_DIR}/libnvptxcompiler_static.a")
  execute_process(
    COMMAND objcopy --rename-section .ctors=.init_array --rename-section .dtors=.fini_array "${dstPath}"
    COMMAND_ERROR_IS_FATAL ANY
  )

  # Create the imported target.
  add_library(${target_name} UNKNOWN IMPORTED)
  target_link_libraries(${target_name} INTERFACE CUDA::cuda_driver Threads::Threads)
  set_property(TARGET ${target_name} PROPERTY IMPORTED_LOCATION "${dstPath}")
  target_include_directories(${target_name} SYSTEM INTERFACE
    "${CUDAToolkit_INCLUDE_DIRS}")
endfunction()
