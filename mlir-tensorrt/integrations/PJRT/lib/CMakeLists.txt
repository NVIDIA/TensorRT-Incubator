# Currently we don't link the MTRT dylib for PJRT, so just get all the
# MLIR-TRT libraries to link against.
mtrt_get_project_targets(Kernel OUT_VAR MLIR_KERNEL_LIBS)
mtrt_get_project_targets(MLIRTensorRT OUT_VAR MLIR_TENSORRT_LIBS)
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE DIALECT OUT_VAR MLIR_TENSORRT_DIALECT_LIBS)
mtrt_get_project_targets(MLIRTensorRTInternal OUT_VAR MLIR_TENSORRT_INTERNAL_LIBS)

function(add_protobuf_gen_command)
  set(proto_gen_output )
  set(proto_gen_input )
  foreach(proto IN LISTS ARGN)
    list(APPEND proto_gen_output ${CMAKE_CURRENT_BINARY_DIR}/${proto}.pb.cc)
    list(APPEND proto_gen_output ${CMAKE_CURRENT_BINARY_DIR}/${proto}.pb.h)
    list(APPEND proto_gen_input ${MLIR_TRT_XLA_SOURCE_DIR}/${proto}.proto)
  endforeach()
  if(NOT TARGET protobuf::protoc)
    message(FATAL_ERROR "protobuf::protoc not found")
  endif()
  add_custom_command(
    OUTPUT
      ${proto_gen_output}
    COMMAND protobuf::protoc
      -I${MLIR_TRT_XLA_SOURCE_DIR}
      -I${Protobuf_SOURCE_DIR}/src
      --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
      --experimental_allow_proto3_optional
      ${proto_gen_input}
  )

  mtrt_add_project_library(
    XLAProtobufLibs
    PROJECT_NAME MTRTPJRT
    LIBRARY_TYPE LIBS
    ${proto_gen_output}

    PARTIAL_SOURCES_INTENDED
    DISABLE_INSTALL

    LINK_LIBS PUBLIC
    protobuf::libprotobuf
  )
  target_compile_options(obj.XLAProtobufLibs PUBLIC
    -w
  )
  target_include_directories(XLAProtobufLibs PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
  )
endfunction()

include_directories(${MLIR_TRT_XLA_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_protobuf_gen_command(
  xla/autotune_results
  xla/autotuning
  xla/xla
  xla/xla_data
  xla/stream_executor/device_description
  xla/pjrt/compile_options
  xla/tsl/protobuf/dnn
  xla/stream_executor/cuda/cuda_compute_capability
  xla/service/hlo
  xla/service/metrics
)

mtrt_add_project_library(MLIRTensorRTPJRTImplementation
  PROJECT_NAME MTRTPJRT
  LIBRARY_TYPE LIBS
  Client.cpp
  Compiler.cpp

  DISABLE_INSTALL

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  ${MLIR_KERNEL_LIBS}
  ${MLIR_TENSORRT_DIALECT_LIBS}
  ${MLIR_TENSORRT_INTERNAL_LIBS}
  ${MLIR_TENSORRT_LIBS}
  PJRT_CApiHeaders
  XLAProtobufLibs
)

add_subdirectory(CAPI)
