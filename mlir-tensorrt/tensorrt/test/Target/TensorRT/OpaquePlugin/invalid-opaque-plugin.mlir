// RUN: tensorrt-opt -allow-unregistered-dialect -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 -split-input-file --verify-diagnostics %s

// Ensure that when the plugin creator returns a nullptr, we fail gracefully.
// expected-error @below {{failed to translate function 'test_opaque_plugin_creation_failure' to a TensorRT engine}}
func.func @test_opaque_plugin_creation_failure(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  // expected-error @below {{failed to create plugin}}
  // expected-error @below {{'tensorrt.opaque_plugin' op failed to encode operation}}
  // expected-error @below {{failed to encode block}}
  %0 = tensorrt.opaque_plugin {
    dso_path = "libTensorRTTestPlugins.so",
    plugin_name = "TestPlugin1",
    plugin_version = "0",
    plugin_namespace = "",
    creator_params = {
        trigger_failure = 1 : i32
    }} (%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}
