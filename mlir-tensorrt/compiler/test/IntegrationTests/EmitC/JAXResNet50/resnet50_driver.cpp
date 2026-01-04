#include "MTRTRuntimeCore.h"
#include "MTRTRuntimeCuda.h"
#include "MTRTRuntimeTensorRT.h"
#include <NvInferRuntime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

extern CUstream resnet50_cuda_stream;
int32_t
resnet50_tensorrt_cluster_engine_data_initialize(nvinfer1::IRuntime *v1);
void resnet50_tensorrt_cluster_engine_data_destroy();
void resnet50_forward(mtrt::RankedMemRef<4> v1, mtrt::RankedMemRef<1> v2);

/// A simple logger that implements TensorRT's logging interface.
class StdioLogger : public nvinfer1::ILogger {
protected:
  void log(Severity severity, const char *msg) noexcept override {
    fprintf(stderr, "%s\n", msg);
  }
};

int main() {
  // Create TensorRT runtime handle.
  StdioLogger logger;
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);

  // Call the generated module's initialize function and set the stream.
  int32_t initRc = resnet50_tensorrt_cluster_engine_data_initialize(runtime);
  if (initRc != 0) {
    std::cerr << mtrt::get_last_error_message() << "\n";
    return initRc;
  }
  resnet50_cuda_stream = cudaStreamDefault;

  // Allocate some test data.
  size_t size = 16 * 3 * 224 * 224 * sizeof(float);
  void *imageBuffer = nullptr;
  int32_t rc =
      mtrt::cuda_alloc(cudaStreamDefault, size, false, false, &imageBuffer);
  if (rc != 0) {
    std::cerr << mtrt::get_last_error_message() << "\n";
    return rc;
  }
  void *outputBuffer = nullptr;
  rc = mtrt::cuda_alloc(cudaStreamDefault, 16 * sizeof(int32_t), false, true,
                        &outputBuffer);
  if (rc != 0) {
    std::cerr << mtrt::get_last_error_message() << "\n";
    return rc;
  }

  // Invoke the generated inference entrypoint.
  resnet50_forward(
      mtrt::make_memref_descriptor<4>(imageBuffer, imageBuffer, 0, 16, 3, 224,
                                      224, 224 * 224 * 3, 224 * 224, 224, 1),
      mtrt::make_memref_descriptor<1>(outputBuffer, outputBuffer, 0, 16, 1));
  cudaStreamSynchronize(resnet50_cuda_stream);

  for (int64_t i = 0; i < 16; i++) {
    std::cerr << "output." << i << " = " << ((int32_t *)outputBuffer)[i]
              << std::endl;
  }

  // Invoke cleanup methods.
  resnet50_tensorrt_cluster_engine_data_destroy();
  ::delete runtime;

  (void)mtrt::cuda_free(cudaStreamDefault, imageBuffer, false, false);
  (void)mtrt::cuda_free(cudaStreamDefault, outputBuffer, false, true);

  return 0;
}
