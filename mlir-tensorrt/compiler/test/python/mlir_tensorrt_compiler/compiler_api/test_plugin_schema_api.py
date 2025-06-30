# REQUIRES: tensorrt-version-ge-10.0
# RUN: %PYTHON %s 2>&1 | FileCheck %s
import ctypes

import mlir_tensorrt.compiler.api as api

ctypes.CDLL("TensorRTTestPlugins.so")

print("Querying Plugin Field Schema")
schema = api.get_tensorrt_plugin_field_schema(
    "TestPlugin1", "0", "", "TensorRTTestPlugins.so"
)
print(schema)
print(
    "\n".join(
        map(
            str,
            sorted(
                {key: (val.type, val.length) for key, val in schema.items()}.items()
            ),
        )
    )
)
# CHECK-LABEL: Querying Plugin Field Schema
# CHECK: ('f16_elements_param', (<PluginFieldType.FLOAT16: 0>, 0))
# CHECK: ('f32_elements_param', (<PluginFieldType.FLOAT32: 1>, 0))
# CHECK: ('f32_param', (<PluginFieldType.FLOAT32: 1>, 0))
# CHECK: ('f64_param', (<PluginFieldType.FLOAT64: 2>, 0))
# CHECK: ('i16_param', (<PluginFieldType.INT16: 4>, 0))
# CHECK: ('i32_param', (<PluginFieldType.INT32: 5>, 0))
# CHECK: ('i64_param', (<PluginFieldType.INT64: 10>, 0))
# CHECK: ('i8_param', (<PluginFieldType.INT8: 3>, 0))
# CHECK: ('shape_param', (<PluginFieldType.DIMS: 7>, 0))
# CHECK: ('shape_vec_param', (<PluginFieldType.DIMS: 7>, 0))
# CHECK: ('string_param', (<PluginFieldType.CHAR: 6>, 0))
