#-------------------------------------------------------------------------------------
# Generates enum definitions using executor-tblgen
#
# Parameters:
#   targetName - Target name (required)
#   inputFileName - Input .td file (required)
#   outputFileName - Output file name (required)
#   GEN_C - Generate C definitions instead of C++ (optional)
#
# Usage:
#   add_mlir_executor_enum_gen(MyEnumGen input.td output.h.inc GEN_C)
#-------------------------------------------------------------------------------------
function(add_mlir_executor_enum_gen targetName inputFileName outputFileName)
  # Validate required arguments
  if(NOT targetName)
    message(FATAL_ERROR "add_mlir_executor_enum_gen: targetName parameter is required")
  endif()
  if(NOT inputFileName)
    message(FATAL_ERROR "add_mlir_executor_enum_gen: inputFileName parameter is required")
  endif()
  if(NOT outputFileName)
    message(FATAL_ERROR "add_mlir_executor_enum_gen: outputFileName parameter is required")
  endif()

  cmake_parse_arguments(ARG "GEN_C" "" "" ${ARGN})

  set(command "--gen-custom-enum-defs")
  if(ARG_GEN_C)
    set(command "--gen-custom-enum-c-defs")
  endif()

  list(TRANSFORM MLIR_INCLUDE_DIRS PREPEND "-I" OUTPUT_VARIABLE _mlir_includes)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}"
    COMMAND executor-tblgen ${command}
      "${CMAKE_CURRENT_LIST_DIR}/${inputFileName}"
      ${_mlir_includes}
      -o "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}"
    DEPENDS "${inputFileName}" executor-tblgen
    COMMENT "Generating ${outputFileName} from ${inputFileName}"
  )
  add_custom_target(${targetName} DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/${outputFileName}")
  add_dependencies(mtrt-headers ${targetName})
endfunction()

#-------------------------------------------------------------------------------------
# Generates FlatBuffer schema headers using flatc compiler
#
# Parameters:
#   target - Target name (required)
#   SRC - Source .fbs file (required)
#
# Usage:
#   add_mlir_executor_flatbuffer_schema(MySchemaGen SRC schema.fbs)
#
# Note: Generates ${srcFileName}Flatbuffer.h in the build directory
#-------------------------------------------------------------------------------------
function(add_mlir_executor_flatbuffer_schema target)
  # Validate required arguments
  if(NOT target)
    message(FATAL_ERROR "add_mlir_executor_flatbuffer_schema: target parameter is required")
  endif()

  cmake_parse_arguments(ARG "" "SRC" "" ${ARGN})

  if(NOT ARG_SRC)
    message(FATAL_ERROR "add_mlir_executor_flatbuffer_schema: SRC parameter is required")
  endif()

  get_filename_component(srcFileName "${ARG_SRC}" NAME_WE)
  set(generatedFileName "${srcFileName}Flatbuffer.h")
  add_custom_command(
    OUTPUT "${generatedFileName}"
    COMMAND flatc --cpp --cpp-std c++17 -o ${CMAKE_CURRENT_BINARY_DIR}
            --filename-suffix Flatbuffer "${ARG_SRC}"
            --gen-object-api
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRC}" flatc
    COMMENT "Generating FlatBuffer schema ${generatedFileName} from ${ARG_SRC}"
  )
  add_custom_target(${target}
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${generatedFileName}"
  )
  add_dependencies(mtrt-headers ${target})
endfunction()


#-------------------------------------------------------------------------------------
# Creates a MLIR-Executor library with proper setup and installation
#
# Parameters:
#   name - Target name (required)
#   All other arguments are passed to add_mlir_library
#
# Usage:
#   add_mlir_executor_library(MyLibrary SOURCES file1.cpp file2.cpp)
#
# Note: This uses mtrt_add_project_library if available for consistency
#-------------------------------------------------------------------------------------
function(add_mlir_executor_library name)
  mtrt_add_project_library(${name}
    PROJECT_NAME MLIRExecutor
    ${ARGN}
  )
endfunction()

#-------------------------------------------------------------------------------------
# Creates a MLIR-Executor runtime library with proper setup and installation
#
# Parameters:
#   name - Target name (required)
#   All other arguments are passed to add_mlir_library
#
# Usage:
#   add_mlir_executor_runtime_library(MyRuntimeLib SOURCES file1.cpp file2.cpp)
#-------------------------------------------------------------------------------------
function(add_mlir_executor_runtime_library name)
  # Validate required arguments
  if(NOT name)
    message(FATAL_ERROR "add_mlir_executor_runtime_library: name parameter is required")
  endif()

  mtrt_add_project_library(${name}
    PROJECT_NAME MLIRExecutor
    ${ARGN}
  )
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
  if(TARGET ${target_name})
    return()
  endif()

  find_library(NvPtxCompilerLibPath NAMES nvptxcompiler_static
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATHS ${CUDAToolkit_LIBRARY_DIR}
    REQUIRED
  )

  set(DEST_PATH "${CMAKE_CURRENT_BINARY_DIR}/libnvptxcompiler_static.a")
  find_program(OBJCOPY_EXE NAMES objcopy llvm-objcopy REQUIRED)
  add_custom_command(
    OUTPUT "${DEST_PATH}"
    COMMAND "${OBJCOPY_EXE}" --rename-section .ctors=.init_array --rename-section .dtors=.fini_array "${NvPtxCompilerLibPath}" "${DEST_PATH}"
    DEPENDS "${NvPtxCompilerLibPath}"
    COMMENT "Patching ${NvPtxCompilerLibPath} → ${DEST_PATH}"
  )

  # Create the imported target.
  add_custom_target(mtrt_nvptxcompiler_patch DEPENDS "${DEST_PATH}")
  add_library(${target_name} UNKNOWN IMPORTED GLOBAL)
  target_link_libraries(${target_name} INTERFACE CUDA::cuda_driver Threads::Threads)
  set_property(TARGET ${target_name} PROPERTY IMPORTED_LOCATION "${DEST_PATH}")
  target_include_directories(${target_name} SYSTEM INTERFACE
    "${CUDAToolkit_INCLUDE_DIRS}")
  add_dependencies(${target_name} mtrt_nvptxcompiler_patch)
endfunction()
