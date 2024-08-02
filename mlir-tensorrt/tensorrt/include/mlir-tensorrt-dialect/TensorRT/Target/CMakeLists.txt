add_tensorrt_encoding_def_gen(MLIRTensorRTTensorRTEncodingIncGen
  ${CMAKE_CURRENT_LIST_DIR}/../IR/TensorRTOps.td
  TensorRTEncodingImpl.inc.cpp
)
