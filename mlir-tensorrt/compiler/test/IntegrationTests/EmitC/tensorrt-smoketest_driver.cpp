#include "MTRTRuntimeCore.h"
#include "MTRTRuntimeCuda.h"
#include "MTRTRuntimeStatus.h"
#include "MTRTRuntimeTensorRT.h"
#include "add.h"
#include <NvInferRuntime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

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

  // Setup the program.
  smoketest_tensorrtProgram program;
  int32_t rc =
      program.smoketest_tensorrt_tensorrt_cluster_0_engine_data_initialize(
          runtime);
  mtrt::abort_on_error(rc);
  rc = program.smoketest_tensorrt_tensorrt_cluster_engine_data_initialize(
      runtime);
  mtrt::abort_on_error(rc);
  rc = program.initialize();
  mtrt::abort_on_error(rc);
  program.main();

  // Teardown.
  rc = program.destroy();
  mtrt::abort_on_error(rc);

  return 0;
}
