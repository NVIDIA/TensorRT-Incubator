install(
  FILES
  MTRTRuntimeCore.cpp
  MTRTRuntimeCuda.cpp
  MTRTRuntimeTensorRT.cpp
  MTRTRuntimeCore.h
  MTRTRuntimeCuda.h
  MTRTRuntimeTensorRT.h

  DESTINATION src/mtrt-runtime

  COMPONENT
  MTRT_CPP_Runtime
)
