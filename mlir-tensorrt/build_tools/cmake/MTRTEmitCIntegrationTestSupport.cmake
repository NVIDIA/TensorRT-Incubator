
# Configure a list of C++ compilation flags used for EmitC integration tests
# that need the CUDA Toolkit (Linux-only currently).
function(configure_emitc_integration_test_support out_var)
  if(MLIR_TRT_ENABLE_CUDA AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(NOT CUDAToolkit_INCLUDE_DIRS OR NOT CUDAToolkit_LIBRARY_DIR)
        message(FATAL_ERROR "CUDAToolkit_INCLUDE_DIRS and CUDAToolkit_LIBRARY_DIR must be set")
    endif()
    list(TRANSFORM CUDAToolkit_INCLUDE_DIRS
        PREPEND "-I" OUTPUT_VARIABLE EMITC_CUDA_CXX_FLAGS)  
    list(APPEND EMITC_CUDA_CXX_FLAGS
      "-L${CUDAToolkit_LIBRARY_DIR}" "-L${CUDAToolkit_LIBRARY_DIR}/stubs" "-lcudart" "-lcuda")
    list(JOIN EMITC_CUDA_CXX_FLAGS " " EMITC_CUDA_CXX_FLAGS)
    set("${out_var}" "${EMITC_CUDA_CXX_FLAGS}" PARENT_SCOPE)
  else()
    set("${out_var}" "" PARENT_SCOPE)
  endif()
endfunction()
