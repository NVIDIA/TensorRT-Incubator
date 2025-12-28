install(
  FILES
  MTRTRuntime.cpp
  MTRTRuntimeTensorRT.cpp
  MTRTRuntime.h
  MTRTRuntimeCore.h
  MTRTRuntimeCuda.h
  MTRTRuntimeTensorRT.h

  DESTINATION src/mtrt-runtime

  COMPONENT
  MTRT_CPP_Runtime
)