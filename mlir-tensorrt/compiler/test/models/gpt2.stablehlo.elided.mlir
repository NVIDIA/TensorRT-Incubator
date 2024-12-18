module @gpt_bs1 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x7xi32> {jax.arg_info = "inputs['attention_mask']", mhlo.sharding = "{replicated}"}, %arg1: tensor<1x7xi32> {jax.arg_info = "inputs['input_ids']", mhlo.sharding = "{replicated}"}) -> (tensor<1x20xi32> {jax.result_info = ""}) {
    %0 = stablehlo.constant dense_resource<__elided__> : tensor<50257x768xf16>
    %1 = stablehlo.constant dense_resource<__elided__> : tensor<1024x768xf16>
    %2 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %3 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %4 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %5 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %6 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %7 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %8 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %9 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %10 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %11 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %12 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %13 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %14 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %15 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %16 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %17 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %18 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %19 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %20 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %21 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %22 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %23 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %24 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %25 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %26 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %27 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %28 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %29 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %30 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %31 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %32 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %33 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %34 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %35 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %36 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %37 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %38 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %39 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %40 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %41 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %42 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %43 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %44 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %45 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %46 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %47 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %48 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %49 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %50 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %51 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %52 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %53 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %54 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %55 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %56 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %57 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %58 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %59 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %60 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %61 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %62 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %63 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %64 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %65 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %66 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %67 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %68 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %69 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %70 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %71 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %72 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %73 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %74 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %75 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %76 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %77 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %78 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %79 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %80 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %81 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %82 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %83 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %84 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %85 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %86 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %87 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %88 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %89 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %90 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %91 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %92 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %93 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %94 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %95 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %96 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %97 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %98 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %99 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %100 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %101 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %102 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %103 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %104 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %105 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %106 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %107 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %108 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %109 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %110 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %111 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %112 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %113 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %114 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %115 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %116 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %117 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %118 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %119 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %120 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %121 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %122 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %123 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %124 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %125 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %126 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %127 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %128 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %129 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %130 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %131 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %132 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %133 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %134 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %135 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %136 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %137 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %138 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %139 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %140 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %141 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %142 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %143 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %144 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %145 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %146 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %147 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %148 = stablehlo.constant dense<50256> : tensor<i32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [] : (tensor<i32>) -> tensor<1x20xi32>
    %150 = stablehlo.constant dense<0> : tensor<i32>
    %151 = stablehlo.constant dense<0> : tensor<i32>
    %152 = stablehlo.dynamic_update_slice %149, %arg1, %150, %151 : (tensor<1x20xi32>, tensor<1x7xi32>, tensor<i32>, tensor<i32>) -> tensor<1x20xi32>
    %153 = stablehlo.constant dense<false> : tensor<i1>
    %154 = stablehlo.broadcast_in_dim %153, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %156 = stablehlo.broadcast_in_dim %155, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %158 = stablehlo.broadcast_in_dim %157, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %159 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %160 = stablehlo.broadcast_in_dim %159, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %161 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %162 = stablehlo.broadcast_in_dim %161, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %165 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %166 = stablehlo.broadcast_in_dim %165, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %167 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %168 = stablehlo.broadcast_in_dim %167, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %169 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %170 = stablehlo.broadcast_in_dim %169, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %173 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %174 = stablehlo.broadcast_in_dim %173, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %175 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %176 = stablehlo.broadcast_in_dim %175, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %177 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %178 = stablehlo.broadcast_in_dim %177, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %180 = stablehlo.broadcast_in_dim %179, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %181 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %182 = stablehlo.broadcast_in_dim %181, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %184 = stablehlo.broadcast_in_dim %183, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %185 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %186 = stablehlo.broadcast_in_dim %185, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %187 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %189 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %190 = stablehlo.broadcast_in_dim %189, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %191 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %192 = stablehlo.broadcast_in_dim %191, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %193 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %195 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %196 = stablehlo.broadcast_in_dim %195, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %197 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %200 = stablehlo.broadcast_in_dim %199, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %202 = stablehlo.broadcast_in_dim %201, dims = [] : (tensor<f32>) -> tensor<1x20x12x64xf32>
    %203 = stablehlo.constant dense<1> : tensor<i32>
    %204 = stablehlo.broadcast_in_dim %203, dims = [] : (tensor<i32>) -> tensor<1x20xi32>
    %205 = call @_cumulative_reduction(%arg0) : (tensor<1x7xi32>) -> tensor<1x7xi32>
    %206 = stablehlo.constant dense<1> : tensor<i32>
    %207 = stablehlo.broadcast_in_dim %206, dims = [] : (tensor<i32>) -> tensor<1x7xi32>
    %208 = stablehlo.subtract %205, %207 : tensor<1x7xi32>
    %209 = stablehlo.constant dense<0> : tensor<i32>
    %210 = stablehlo.constant dense<0> : tensor<i32>
    %211 = stablehlo.dynamic_update_slice %204, %arg0, %209, %210 : (tensor<1x20xi32>, tensor<1x7xi32>, tensor<i32>, tensor<i32>) -> tensor<1x20xi32>
    %212 = stablehlo.convert %0 : (tensor<50257x768xf16>) -> tensor<50257x768xf32>
    %213 = call @_take(%212, %arg1) : (tensor<50257x768xf32>, tensor<1x7xi32>) -> tensor<1x7x768xf32>
    %214 = stablehlo.convert %1 : (tensor<1024x768xf16>) -> tensor<1024x768xf32>
    %215 = call @_take_0(%214, %208) : (tensor<1024x768xf32>, tensor<1x7xi32>) -> tensor<1x7x768xf32>
    %216 = stablehlo.add %213, %215 : tensor<1x7x768xf32>
    %217 = stablehlo.multiply %216, %216 : tensor<1x7x768xf32>
    %218 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %219 = stablehlo.reduce(%217 init: %218) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %220 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %221 = stablehlo.broadcast_in_dim %220, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %222 = stablehlo.divide %219, %221 : tensor<1x7xf32>
    %223 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %224 = stablehlo.reduce(%216 init: %223) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %225 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %226 = stablehlo.broadcast_in_dim %225, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %227 = stablehlo.divide %224, %226 : tensor<1x7xf32>
    %228 = stablehlo.multiply %227, %227 : tensor<1x7xf32>
    %229 = stablehlo.subtract %222, %228 : tensor<1x7xf32>
    %230 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %231 = stablehlo.broadcast_in_dim %230, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %232 = stablehlo.maximum %231, %229 : tensor<1x7xf32>
    %233 = stablehlo.broadcast_in_dim %227, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %234 = stablehlo.broadcast_in_dim %232, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %235 = stablehlo.broadcast_in_dim %233, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %236 = stablehlo.subtract %216, %235 : tensor<1x7x768xf32>
    %237 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %238 = stablehlo.broadcast_in_dim %237, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %239 = stablehlo.add %234, %238 : tensor<1x7x1xf32>
    %240 = stablehlo.rsqrt %239 : tensor<1x7x1xf32>
    %241 = stablehlo.reshape %2 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %242 = stablehlo.convert %241 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %243 = stablehlo.broadcast_in_dim %240, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %244 = stablehlo.broadcast_in_dim %242, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %245 = stablehlo.multiply %243, %244 : tensor<1x7x768xf32>
    %246 = stablehlo.multiply %236, %245 : tensor<1x7x768xf32>
    %247 = stablehlo.reshape %3 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %248 = stablehlo.convert %247 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %249 = stablehlo.broadcast_in_dim %248, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %250 = stablehlo.add %246, %249 : tensor<1x7x768xf32>
    %251 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %252 = stablehlo.broadcast_in_dim %251, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %253 = stablehlo.broadcast_in_dim %252, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %254 = stablehlo.broadcast_in_dim %252, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %255 = stablehlo.broadcast_in_dim %253, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %256 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %257 = stablehlo.compare  GE, %255, %256,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %258 = stablehlo.broadcast_in_dim %257, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %259 = stablehlo.transpose %4, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %260 = stablehlo.convert %259 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %261 = stablehlo.dot_general %250, %260, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x2304xf32>) -> tensor<1x7x2304xf32>
    %262 = stablehlo.convert %5 : (tensor<2304xf16>) -> tensor<2304xf32>
    %263 = stablehlo.broadcast_in_dim %262, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %264 = stablehlo.broadcast_in_dim %263, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<1x7x2304xf32>
    %265 = stablehlo.add %261, %264 : tensor<1x7x2304xf32>
    %266 = stablehlo.slice %265 [0:1, 0:7, 0:768] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %267 = stablehlo.slice %265 [0:1, 0:7, 768:1536] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %268 = stablehlo.slice %265 [0:1, 0:7, 1536:2304] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %269 = stablehlo.reshape %266 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %270 = stablehlo.reshape %267 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %271 = stablehlo.reshape %268 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %272 = stablehlo.constant dense<0> : tensor<i32>
    %273 = stablehlo.constant dense<0> : tensor<i32>
    %274 = stablehlo.compare  LT, %272, %273,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %275 = stablehlo.constant dense<0> : tensor<i32>
    %276 = stablehlo.constant dense<1024> : tensor<i32>
    %277 = stablehlo.add %275, %276 : tensor<i32>
    %278 = stablehlo.constant dense<0> : tensor<i32>
    %279 = stablehlo.select %274, %277, %278 : tensor<i1>, tensor<i32>
    %280 = stablehlo.constant dense<0> : tensor<i32>
    %281 = stablehlo.constant dense<0> : tensor<i32>
    %282 = stablehlo.constant dense<0> : tensor<i32>
    %283 = stablehlo.dynamic_slice %258, %280, %281, %279, %282, sizes = [1, 1, 7, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x7x20xi1>
    %284 = stablehlo.broadcast_in_dim %211, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %285 = stablehlo.reshape %284 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x7x20xi32>
    %287 = stablehlo.constant dense<0> : tensor<i32>
    %288 = stablehlo.broadcast_in_dim %287, dims = [] : (tensor<i32>) -> tensor<1x1x7x20xi32>
    %289 = stablehlo.compare  NE, %286, %288,  SIGNED : (tensor<1x1x7x20xi32>, tensor<1x1x7x20xi32>) -> tensor<1x1x7x20xi1>
    %290 = stablehlo.and %289, %283 : tensor<1x1x7x20xi1>
    %291 = stablehlo.convert %290 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %292 = stablehlo.constant dense<0> : tensor<i32>
    %293 = stablehlo.constant dense<0> : tensor<i32>
    %294 = stablehlo.compare  LT, %292, %293,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %295 = stablehlo.constant dense<0> : tensor<i32>
    %296 = stablehlo.constant dense<20> : tensor<i32>
    %297 = stablehlo.add %295, %296 : tensor<i32>
    %298 = stablehlo.constant dense<0> : tensor<i32>
    %299 = stablehlo.select %294, %297, %298 : tensor<i1>, tensor<i32>
    %300 = stablehlo.constant dense<0> : tensor<i32>
    %301 = stablehlo.constant dense<0> : tensor<i32>
    %302 = stablehlo.constant dense<0> : tensor<i32>
    %303 = stablehlo.dynamic_update_slice %156, %270, %300, %299, %301, %302 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %304 = stablehlo.constant dense<0> : tensor<i32>
    %305 = stablehlo.constant dense<0> : tensor<i32>
    %306 = stablehlo.compare  LT, %304, %305,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %307 = stablehlo.constant dense<0> : tensor<i32>
    %308 = stablehlo.constant dense<20> : tensor<i32>
    %309 = stablehlo.add %307, %308 : tensor<i32>
    %310 = stablehlo.constant dense<0> : tensor<i32>
    %311 = stablehlo.select %306, %309, %310 : tensor<i1>, tensor<i32>
    %312 = stablehlo.constant dense<0> : tensor<i32>
    %313 = stablehlo.constant dense<0> : tensor<i32>
    %314 = stablehlo.constant dense<0> : tensor<i32>
    %315 = stablehlo.dynamic_update_slice %158, %271, %312, %311, %313, %314 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %316 = stablehlo.constant dense<0> : tensor<i32>
    %317 = stablehlo.constant dense<7> : tensor<i32>
    %318 = stablehlo.add %316, %317 : tensor<i32>
    %319 = stablehlo.iota dim = 0 : tensor<20xi32>
    %320 = stablehlo.constant dense<0> : tensor<i32>
    %321 = stablehlo.constant dense<7> : tensor<i32>
    %322 = stablehlo.add %320, %321 : tensor<i32>
    %323 = stablehlo.broadcast_in_dim %322, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %324 = stablehlo.compare  LT, %319, %323,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %325 = stablehlo.broadcast_in_dim %324, dims = [3] : (tensor<20xi1>) -> tensor<1x1x7x20xi1>
    %326 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %327 = stablehlo.broadcast_in_dim %326, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %328 = stablehlo.compare  NE, %291, %327,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %329 = stablehlo.and %325, %328 : tensor<1x1x7x20xi1>
    %330 = stablehlo.convert %329 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %331 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %332 = stablehlo.broadcast_in_dim %331, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %333 = stablehlo.compare  GT, %330, %332,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %334 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %335 = stablehlo.broadcast_in_dim %334, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %336 = stablehlo.convert %335 : tensor<1x1x7x20xf32>
    %337 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %338 = stablehlo.broadcast_in_dim %337, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %339 = stablehlo.select %333, %336, %338 : tensor<1x1x7x20xi1>, tensor<1x1x7x20xf32>
    %340 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %341 = stablehlo.sqrt %340 : tensor<f32>
    %342 = stablehlo.convert %341 : tensor<f32>
    %343 = stablehlo.broadcast_in_dim %342, dims = [] : (tensor<f32>) -> tensor<1x7x12x64xf32>
    %344 = stablehlo.divide %269, %343 : tensor<1x7x12x64xf32>
    %345 = stablehlo.dot_general %344, %303, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x7x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x7x20xf32>
    %346 = stablehlo.broadcast_in_dim %339, dims = [0, 1, 2, 3] : (tensor<1x1x7x20xf32>) -> tensor<1x12x7x20xf32>
    %347 = stablehlo.add %345, %346 : tensor<1x12x7x20xf32>
    %348 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %349 = stablehlo.reduce(%347 init: %348) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %350 = stablehlo.broadcast_in_dim %349, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %351 = stablehlo.broadcast_in_dim %350, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %352 = stablehlo.subtract %347, %351 : tensor<1x12x7x20xf32>
    %353 = stablehlo.exponential %352 : tensor<1x12x7x20xf32>
    %354 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %355 = stablehlo.reduce(%353 init: %354) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %356 = stablehlo.broadcast_in_dim %355, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %357 = stablehlo.broadcast_in_dim %356, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %358 = stablehlo.divide %353, %357 : tensor<1x12x7x20xf32>
    %359 = stablehlo.dot_general %315, %358, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x7x20xf32>) -> tensor<1x12x64x7xf32>
    %360 = stablehlo.transpose %359, dims = [0, 3, 1, 2] : (tensor<1x12x64x7xf32>) -> tensor<1x7x12x64xf32>
    %361 = stablehlo.reshape %360 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %362 = stablehlo.transpose %6, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %363 = stablehlo.convert %362 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %364 = stablehlo.dot_general %361, %363, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x768xf32>) -> tensor<1x7x768xf32>
    %365 = stablehlo.convert %7 : (tensor<768xf16>) -> tensor<768xf32>
    %366 = stablehlo.broadcast_in_dim %365, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %367 = stablehlo.broadcast_in_dim %366, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %368 = stablehlo.add %364, %367 : tensor<1x7x768xf32>
    %369 = stablehlo.add %368, %216 : tensor<1x7x768xf32>
    %370 = stablehlo.multiply %369, %369 : tensor<1x7x768xf32>
    %371 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %372 = stablehlo.reduce(%370 init: %371) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %373 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %374 = stablehlo.broadcast_in_dim %373, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %375 = stablehlo.divide %372, %374 : tensor<1x7xf32>
    %376 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %377 = stablehlo.reduce(%369 init: %376) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %378 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %379 = stablehlo.broadcast_in_dim %378, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %380 = stablehlo.divide %377, %379 : tensor<1x7xf32>
    %381 = stablehlo.multiply %380, %380 : tensor<1x7xf32>
    %382 = stablehlo.subtract %375, %381 : tensor<1x7xf32>
    %383 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %384 = stablehlo.broadcast_in_dim %383, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %385 = stablehlo.maximum %384, %382 : tensor<1x7xf32>
    %386 = stablehlo.broadcast_in_dim %380, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %387 = stablehlo.broadcast_in_dim %385, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %388 = stablehlo.broadcast_in_dim %386, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %389 = stablehlo.subtract %369, %388 : tensor<1x7x768xf32>
    %390 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %391 = stablehlo.broadcast_in_dim %390, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %392 = stablehlo.add %387, %391 : tensor<1x7x1xf32>
    %393 = stablehlo.rsqrt %392 : tensor<1x7x1xf32>
    %394 = stablehlo.reshape %8 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %395 = stablehlo.convert %394 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %396 = stablehlo.broadcast_in_dim %393, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %397 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %398 = stablehlo.multiply %396, %397 : tensor<1x7x768xf32>
    %399 = stablehlo.multiply %389, %398 : tensor<1x7x768xf32>
    %400 = stablehlo.reshape %9 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %401 = stablehlo.convert %400 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %403 = stablehlo.add %399, %402 : tensor<1x7x768xf32>
    %404 = stablehlo.transpose %10, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %405 = stablehlo.convert %404 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %406 = stablehlo.dot_general %403, %405, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x3072xf32>) -> tensor<1x7x3072xf32>
    %407 = stablehlo.convert %11 : (tensor<3072xf16>) -> tensor<3072xf32>
    %408 = stablehlo.broadcast_in_dim %407, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<1x7x3072xf32>
    %410 = stablehlo.add %406, %409 : tensor<1x7x3072xf32>
    %411 = stablehlo.multiply %410, %410 : tensor<1x7x3072xf32>
    %412 = stablehlo.multiply %410, %411 : tensor<1x7x3072xf32>
    %413 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %414 = stablehlo.broadcast_in_dim %413, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %415 = stablehlo.multiply %414, %412 : tensor<1x7x3072xf32>
    %416 = stablehlo.add %410, %415 : tensor<1x7x3072xf32>
    %417 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %418 = stablehlo.broadcast_in_dim %417, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %419 = stablehlo.multiply %418, %416 : tensor<1x7x3072xf32>
    %420 = stablehlo.tanh %419 : tensor<1x7x3072xf32>
    %421 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %422 = stablehlo.broadcast_in_dim %421, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %423 = stablehlo.add %422, %420 : tensor<1x7x3072xf32>
    %424 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %425 = stablehlo.broadcast_in_dim %424, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %426 = stablehlo.multiply %425, %423 : tensor<1x7x3072xf32>
    %427 = stablehlo.multiply %410, %426 : tensor<1x7x3072xf32>
    %428 = stablehlo.transpose %12, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %429 = stablehlo.convert %428 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %430 = stablehlo.dot_general %427, %429, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x3072xf32>, tensor<3072x768xf32>) -> tensor<1x7x768xf32>
    %431 = stablehlo.convert %13 : (tensor<768xf16>) -> tensor<768xf32>
    %432 = stablehlo.broadcast_in_dim %431, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %433 = stablehlo.broadcast_in_dim %432, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %434 = stablehlo.add %430, %433 : tensor<1x7x768xf32>
    %435 = stablehlo.add %369, %434 : tensor<1x7x768xf32>
    %436 = stablehlo.multiply %435, %435 : tensor<1x7x768xf32>
    %437 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %438 = stablehlo.reduce(%436 init: %437) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %439 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %440 = stablehlo.broadcast_in_dim %439, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %441 = stablehlo.divide %438, %440 : tensor<1x7xf32>
    %442 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %443 = stablehlo.reduce(%435 init: %442) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %444 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %445 = stablehlo.broadcast_in_dim %444, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %446 = stablehlo.divide %443, %445 : tensor<1x7xf32>
    %447 = stablehlo.multiply %446, %446 : tensor<1x7xf32>
    %448 = stablehlo.subtract %441, %447 : tensor<1x7xf32>
    %449 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %450 = stablehlo.broadcast_in_dim %449, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %451 = stablehlo.maximum %450, %448 : tensor<1x7xf32>
    %452 = stablehlo.broadcast_in_dim %446, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %453 = stablehlo.broadcast_in_dim %451, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %454 = stablehlo.broadcast_in_dim %452, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %455 = stablehlo.subtract %435, %454 : tensor<1x7x768xf32>
    %456 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %457 = stablehlo.broadcast_in_dim %456, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %458 = stablehlo.add %453, %457 : tensor<1x7x1xf32>
    %459 = stablehlo.rsqrt %458 : tensor<1x7x1xf32>
    %460 = stablehlo.reshape %14 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %461 = stablehlo.convert %460 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %462 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %463 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %464 = stablehlo.multiply %462, %463 : tensor<1x7x768xf32>
    %465 = stablehlo.multiply %455, %464 : tensor<1x7x768xf32>
    %466 = stablehlo.reshape %15 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %467 = stablehlo.convert %466 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %468 = stablehlo.broadcast_in_dim %467, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %469 = stablehlo.add %465, %468 : tensor<1x7x768xf32>
    %470 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %472 = stablehlo.broadcast_in_dim %471, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %473 = stablehlo.broadcast_in_dim %471, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %474 = stablehlo.broadcast_in_dim %472, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %475 = stablehlo.broadcast_in_dim %473, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %476 = stablehlo.compare  GE, %474, %475,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %477 = stablehlo.broadcast_in_dim %476, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %478 = stablehlo.transpose %16, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %479 = stablehlo.convert %478 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %480 = stablehlo.dot_general %469, %479, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x2304xf32>) -> tensor<1x7x2304xf32>
    %481 = stablehlo.convert %17 : (tensor<2304xf16>) -> tensor<2304xf32>
    %482 = stablehlo.broadcast_in_dim %481, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %483 = stablehlo.broadcast_in_dim %482, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<1x7x2304xf32>
    %484 = stablehlo.add %480, %483 : tensor<1x7x2304xf32>
    %485 = stablehlo.slice %484 [0:1, 0:7, 0:768] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %486 = stablehlo.slice %484 [0:1, 0:7, 768:1536] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %487 = stablehlo.slice %484 [0:1, 0:7, 1536:2304] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %488 = stablehlo.reshape %485 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %489 = stablehlo.reshape %486 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %490 = stablehlo.reshape %487 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %491 = stablehlo.constant dense<0> : tensor<i32>
    %492 = stablehlo.constant dense<0> : tensor<i32>
    %493 = stablehlo.compare  LT, %491, %492,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %494 = stablehlo.constant dense<0> : tensor<i32>
    %495 = stablehlo.constant dense<1024> : tensor<i32>
    %496 = stablehlo.add %494, %495 : tensor<i32>
    %497 = stablehlo.constant dense<0> : tensor<i32>
    %498 = stablehlo.select %493, %496, %497 : tensor<i1>, tensor<i32>
    %499 = stablehlo.constant dense<0> : tensor<i32>
    %500 = stablehlo.constant dense<0> : tensor<i32>
    %501 = stablehlo.constant dense<0> : tensor<i32>
    %502 = stablehlo.dynamic_slice %477, %499, %500, %498, %501, sizes = [1, 1, 7, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x7x20xi1>
    %503 = stablehlo.broadcast_in_dim %211, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %504 = stablehlo.reshape %503 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %505 = stablehlo.broadcast_in_dim %504, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x7x20xi32>
    %506 = stablehlo.constant dense<0> : tensor<i32>
    %507 = stablehlo.broadcast_in_dim %506, dims = [] : (tensor<i32>) -> tensor<1x1x7x20xi32>
    %508 = stablehlo.compare  NE, %505, %507,  SIGNED : (tensor<1x1x7x20xi32>, tensor<1x1x7x20xi32>) -> tensor<1x1x7x20xi1>
    %509 = stablehlo.and %508, %502 : tensor<1x1x7x20xi1>
    %510 = stablehlo.convert %509 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %511 = stablehlo.constant dense<0> : tensor<i32>
    %512 = stablehlo.constant dense<0> : tensor<i32>
    %513 = stablehlo.compare  LT, %511, %512,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %514 = stablehlo.constant dense<0> : tensor<i32>
    %515 = stablehlo.constant dense<20> : tensor<i32>
    %516 = stablehlo.add %514, %515 : tensor<i32>
    %517 = stablehlo.constant dense<0> : tensor<i32>
    %518 = stablehlo.select %513, %516, %517 : tensor<i1>, tensor<i32>
    %519 = stablehlo.constant dense<0> : tensor<i32>
    %520 = stablehlo.constant dense<0> : tensor<i32>
    %521 = stablehlo.constant dense<0> : tensor<i32>
    %522 = stablehlo.dynamic_update_slice %160, %489, %519, %518, %520, %521 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %523 = stablehlo.constant dense<0> : tensor<i32>
    %524 = stablehlo.constant dense<0> : tensor<i32>
    %525 = stablehlo.compare  LT, %523, %524,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %526 = stablehlo.constant dense<0> : tensor<i32>
    %527 = stablehlo.constant dense<20> : tensor<i32>
    %528 = stablehlo.add %526, %527 : tensor<i32>
    %529 = stablehlo.constant dense<0> : tensor<i32>
    %530 = stablehlo.select %525, %528, %529 : tensor<i1>, tensor<i32>
    %531 = stablehlo.constant dense<0> : tensor<i32>
    %532 = stablehlo.constant dense<0> : tensor<i32>
    %533 = stablehlo.constant dense<0> : tensor<i32>
    %534 = stablehlo.dynamic_update_slice %162, %490, %531, %530, %532, %533 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %535 = stablehlo.constant dense<0> : tensor<i32>
    %536 = stablehlo.constant dense<7> : tensor<i32>
    %537 = stablehlo.add %535, %536 : tensor<i32>
    %538 = stablehlo.iota dim = 0 : tensor<20xi32>
    %539 = stablehlo.constant dense<0> : tensor<i32>
    %540 = stablehlo.constant dense<7> : tensor<i32>
    %541 = stablehlo.add %539, %540 : tensor<i32>
    %542 = stablehlo.broadcast_in_dim %541, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %543 = stablehlo.compare  LT, %538, %542,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %544 = stablehlo.broadcast_in_dim %543, dims = [3] : (tensor<20xi1>) -> tensor<1x1x7x20xi1>
    %545 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %546 = stablehlo.broadcast_in_dim %545, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %547 = stablehlo.compare  NE, %510, %546,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %548 = stablehlo.and %544, %547 : tensor<1x1x7x20xi1>
    %549 = stablehlo.convert %548 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %550 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %551 = stablehlo.broadcast_in_dim %550, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %552 = stablehlo.compare  GT, %549, %551,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %553 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %554 = stablehlo.broadcast_in_dim %553, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %555 = stablehlo.convert %554 : tensor<1x1x7x20xf32>
    %556 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %557 = stablehlo.broadcast_in_dim %556, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %558 = stablehlo.select %552, %555, %557 : tensor<1x1x7x20xi1>, tensor<1x1x7x20xf32>
    %559 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %560 = stablehlo.sqrt %559 : tensor<f32>
    %561 = stablehlo.convert %560 : tensor<f32>
    %562 = stablehlo.broadcast_in_dim %561, dims = [] : (tensor<f32>) -> tensor<1x7x12x64xf32>
    %563 = stablehlo.divide %488, %562 : tensor<1x7x12x64xf32>
    %564 = stablehlo.dot_general %563, %522, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x7x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x7x20xf32>
    %565 = stablehlo.broadcast_in_dim %558, dims = [0, 1, 2, 3] : (tensor<1x1x7x20xf32>) -> tensor<1x12x7x20xf32>
    %566 = stablehlo.add %564, %565 : tensor<1x12x7x20xf32>
    %567 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %568 = stablehlo.reduce(%566 init: %567) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %569 = stablehlo.broadcast_in_dim %568, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %570 = stablehlo.broadcast_in_dim %569, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %571 = stablehlo.subtract %566, %570 : tensor<1x12x7x20xf32>
    %572 = stablehlo.exponential %571 : tensor<1x12x7x20xf32>
    %573 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %574 = stablehlo.reduce(%572 init: %573) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %575 = stablehlo.broadcast_in_dim %574, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %576 = stablehlo.broadcast_in_dim %575, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %577 = stablehlo.divide %572, %576 : tensor<1x12x7x20xf32>
    %578 = stablehlo.dot_general %534, %577, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x7x20xf32>) -> tensor<1x12x64x7xf32>
    %579 = stablehlo.transpose %578, dims = [0, 3, 1, 2] : (tensor<1x12x64x7xf32>) -> tensor<1x7x12x64xf32>
    %580 = stablehlo.reshape %579 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %581 = stablehlo.transpose %18, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %582 = stablehlo.convert %581 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %583 = stablehlo.dot_general %580, %582, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x768xf32>) -> tensor<1x7x768xf32>
    %584 = stablehlo.convert %19 : (tensor<768xf16>) -> tensor<768xf32>
    %585 = stablehlo.broadcast_in_dim %584, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %586 = stablehlo.broadcast_in_dim %585, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %587 = stablehlo.add %583, %586 : tensor<1x7x768xf32>
    %588 = stablehlo.add %587, %435 : tensor<1x7x768xf32>
    %589 = stablehlo.multiply %588, %588 : tensor<1x7x768xf32>
    %590 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %591 = stablehlo.reduce(%589 init: %590) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %592 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %593 = stablehlo.broadcast_in_dim %592, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %594 = stablehlo.divide %591, %593 : tensor<1x7xf32>
    %595 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %596 = stablehlo.reduce(%588 init: %595) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %597 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %598 = stablehlo.broadcast_in_dim %597, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %599 = stablehlo.divide %596, %598 : tensor<1x7xf32>
    %600 = stablehlo.multiply %599, %599 : tensor<1x7xf32>
    %601 = stablehlo.subtract %594, %600 : tensor<1x7xf32>
    %602 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %603 = stablehlo.broadcast_in_dim %602, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %604 = stablehlo.maximum %603, %601 : tensor<1x7xf32>
    %605 = stablehlo.broadcast_in_dim %599, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %606 = stablehlo.broadcast_in_dim %604, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %607 = stablehlo.broadcast_in_dim %605, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %608 = stablehlo.subtract %588, %607 : tensor<1x7x768xf32>
    %609 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %610 = stablehlo.broadcast_in_dim %609, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %611 = stablehlo.add %606, %610 : tensor<1x7x1xf32>
    %612 = stablehlo.rsqrt %611 : tensor<1x7x1xf32>
    %613 = stablehlo.reshape %20 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %614 = stablehlo.convert %613 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %615 = stablehlo.broadcast_in_dim %612, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %616 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %617 = stablehlo.multiply %615, %616 : tensor<1x7x768xf32>
    %618 = stablehlo.multiply %608, %617 : tensor<1x7x768xf32>
    %619 = stablehlo.reshape %21 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %620 = stablehlo.convert %619 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %621 = stablehlo.broadcast_in_dim %620, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %622 = stablehlo.add %618, %621 : tensor<1x7x768xf32>
    %623 = stablehlo.transpose %22, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %624 = stablehlo.convert %623 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %625 = stablehlo.dot_general %622, %624, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x3072xf32>) -> tensor<1x7x3072xf32>
    %626 = stablehlo.convert %23 : (tensor<3072xf16>) -> tensor<3072xf32>
    %627 = stablehlo.broadcast_in_dim %626, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %628 = stablehlo.broadcast_in_dim %627, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<1x7x3072xf32>
    %629 = stablehlo.add %625, %628 : tensor<1x7x3072xf32>
    %630 = stablehlo.multiply %629, %629 : tensor<1x7x3072xf32>
    %631 = stablehlo.multiply %629, %630 : tensor<1x7x3072xf32>
    %632 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %633 = stablehlo.broadcast_in_dim %632, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %634 = stablehlo.multiply %633, %631 : tensor<1x7x3072xf32>
    %635 = stablehlo.add %629, %634 : tensor<1x7x3072xf32>
    %636 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %637 = stablehlo.broadcast_in_dim %636, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %638 = stablehlo.multiply %637, %635 : tensor<1x7x3072xf32>
    %639 = stablehlo.tanh %638 : tensor<1x7x3072xf32>
    %640 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %641 = stablehlo.broadcast_in_dim %640, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %642 = stablehlo.add %641, %639 : tensor<1x7x3072xf32>
    %643 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %644 = stablehlo.broadcast_in_dim %643, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %645 = stablehlo.multiply %644, %642 : tensor<1x7x3072xf32>
    %646 = stablehlo.multiply %629, %645 : tensor<1x7x3072xf32>
    %647 = stablehlo.transpose %24, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %648 = stablehlo.convert %647 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %649 = stablehlo.dot_general %646, %648, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x3072xf32>, tensor<3072x768xf32>) -> tensor<1x7x768xf32>
    %650 = stablehlo.convert %25 : (tensor<768xf16>) -> tensor<768xf32>
    %651 = stablehlo.broadcast_in_dim %650, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %652 = stablehlo.broadcast_in_dim %651, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %653 = stablehlo.add %649, %652 : tensor<1x7x768xf32>
    %654 = stablehlo.add %588, %653 : tensor<1x7x768xf32>
    %655 = stablehlo.multiply %654, %654 : tensor<1x7x768xf32>
    %656 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %657 = stablehlo.reduce(%655 init: %656) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %658 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %659 = stablehlo.broadcast_in_dim %658, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %660 = stablehlo.divide %657, %659 : tensor<1x7xf32>
    %661 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %662 = stablehlo.reduce(%654 init: %661) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %663 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %664 = stablehlo.broadcast_in_dim %663, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %665 = stablehlo.divide %662, %664 : tensor<1x7xf32>
    %666 = stablehlo.multiply %665, %665 : tensor<1x7xf32>
    %667 = stablehlo.subtract %660, %666 : tensor<1x7xf32>
    %668 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %669 = stablehlo.broadcast_in_dim %668, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %670 = stablehlo.maximum %669, %667 : tensor<1x7xf32>
    %671 = stablehlo.broadcast_in_dim %665, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %672 = stablehlo.broadcast_in_dim %670, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %673 = stablehlo.broadcast_in_dim %671, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %674 = stablehlo.subtract %654, %673 : tensor<1x7x768xf32>
    %675 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %676 = stablehlo.broadcast_in_dim %675, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %677 = stablehlo.add %672, %676 : tensor<1x7x1xf32>
    %678 = stablehlo.rsqrt %677 : tensor<1x7x1xf32>
    %679 = stablehlo.reshape %26 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %680 = stablehlo.convert %679 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %681 = stablehlo.broadcast_in_dim %678, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %682 = stablehlo.broadcast_in_dim %680, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %683 = stablehlo.multiply %681, %682 : tensor<1x7x768xf32>
    %684 = stablehlo.multiply %674, %683 : tensor<1x7x768xf32>
    %685 = stablehlo.reshape %27 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %686 = stablehlo.convert %685 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %687 = stablehlo.broadcast_in_dim %686, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %688 = stablehlo.add %684, %687 : tensor<1x7x768xf32>
    %689 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %690 = stablehlo.broadcast_in_dim %689, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %691 = stablehlo.broadcast_in_dim %690, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %692 = stablehlo.broadcast_in_dim %690, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %693 = stablehlo.broadcast_in_dim %691, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %694 = stablehlo.broadcast_in_dim %692, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %695 = stablehlo.compare  GE, %693, %694,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %696 = stablehlo.broadcast_in_dim %695, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %697 = stablehlo.transpose %28, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %698 = stablehlo.convert %697 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %699 = stablehlo.dot_general %688, %698, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x2304xf32>) -> tensor<1x7x2304xf32>
    %700 = stablehlo.convert %29 : (tensor<2304xf16>) -> tensor<2304xf32>
    %701 = stablehlo.broadcast_in_dim %700, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %702 = stablehlo.broadcast_in_dim %701, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<1x7x2304xf32>
    %703 = stablehlo.add %699, %702 : tensor<1x7x2304xf32>
    %704 = stablehlo.slice %703 [0:1, 0:7, 0:768] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %705 = stablehlo.slice %703 [0:1, 0:7, 768:1536] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %706 = stablehlo.slice %703 [0:1, 0:7, 1536:2304] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %707 = stablehlo.reshape %704 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %708 = stablehlo.reshape %705 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %709 = stablehlo.reshape %706 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %710 = stablehlo.constant dense<0> : tensor<i32>
    %711 = stablehlo.constant dense<0> : tensor<i32>
    %712 = stablehlo.compare  LT, %710, %711,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %713 = stablehlo.constant dense<0> : tensor<i32>
    %714 = stablehlo.constant dense<1024> : tensor<i32>
    %715 = stablehlo.add %713, %714 : tensor<i32>
    %716 = stablehlo.constant dense<0> : tensor<i32>
    %717 = stablehlo.select %712, %715, %716 : tensor<i1>, tensor<i32>
    %718 = stablehlo.constant dense<0> : tensor<i32>
    %719 = stablehlo.constant dense<0> : tensor<i32>
    %720 = stablehlo.constant dense<0> : tensor<i32>
    %721 = stablehlo.dynamic_slice %696, %718, %719, %717, %720, sizes = [1, 1, 7, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x7x20xi1>
    %722 = stablehlo.broadcast_in_dim %211, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %723 = stablehlo.reshape %722 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %724 = stablehlo.broadcast_in_dim %723, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x7x20xi32>
    %725 = stablehlo.constant dense<0> : tensor<i32>
    %726 = stablehlo.broadcast_in_dim %725, dims = [] : (tensor<i32>) -> tensor<1x1x7x20xi32>
    %727 = stablehlo.compare  NE, %724, %726,  SIGNED : (tensor<1x1x7x20xi32>, tensor<1x1x7x20xi32>) -> tensor<1x1x7x20xi1>
    %728 = stablehlo.and %727, %721 : tensor<1x1x7x20xi1>
    %729 = stablehlo.convert %728 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %730 = stablehlo.constant dense<0> : tensor<i32>
    %731 = stablehlo.constant dense<0> : tensor<i32>
    %732 = stablehlo.compare  LT, %730, %731,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %733 = stablehlo.constant dense<0> : tensor<i32>
    %734 = stablehlo.constant dense<20> : tensor<i32>
    %735 = stablehlo.add %733, %734 : tensor<i32>
    %736 = stablehlo.constant dense<0> : tensor<i32>
    %737 = stablehlo.select %732, %735, %736 : tensor<i1>, tensor<i32>
    %738 = stablehlo.constant dense<0> : tensor<i32>
    %739 = stablehlo.constant dense<0> : tensor<i32>
    %740 = stablehlo.constant dense<0> : tensor<i32>
    %741 = stablehlo.dynamic_update_slice %164, %708, %738, %737, %739, %740 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %742 = stablehlo.constant dense<0> : tensor<i32>
    %743 = stablehlo.constant dense<0> : tensor<i32>
    %744 = stablehlo.compare  LT, %742, %743,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %745 = stablehlo.constant dense<0> : tensor<i32>
    %746 = stablehlo.constant dense<20> : tensor<i32>
    %747 = stablehlo.add %745, %746 : tensor<i32>
    %748 = stablehlo.constant dense<0> : tensor<i32>
    %749 = stablehlo.select %744, %747, %748 : tensor<i1>, tensor<i32>
    %750 = stablehlo.constant dense<0> : tensor<i32>
    %751 = stablehlo.constant dense<0> : tensor<i32>
    %752 = stablehlo.constant dense<0> : tensor<i32>
    %753 = stablehlo.dynamic_update_slice %166, %709, %750, %749, %751, %752 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %754 = stablehlo.constant dense<0> : tensor<i32>
    %755 = stablehlo.constant dense<7> : tensor<i32>
    %756 = stablehlo.add %754, %755 : tensor<i32>
    %757 = stablehlo.iota dim = 0 : tensor<20xi32>
    %758 = stablehlo.constant dense<0> : tensor<i32>
    %759 = stablehlo.constant dense<7> : tensor<i32>
    %760 = stablehlo.add %758, %759 : tensor<i32>
    %761 = stablehlo.broadcast_in_dim %760, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %762 = stablehlo.compare  LT, %757, %761,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %763 = stablehlo.broadcast_in_dim %762, dims = [3] : (tensor<20xi1>) -> tensor<1x1x7x20xi1>
    %764 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %765 = stablehlo.broadcast_in_dim %764, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %766 = stablehlo.compare  NE, %729, %765,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %767 = stablehlo.and %763, %766 : tensor<1x1x7x20xi1>
    %768 = stablehlo.convert %767 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %769 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %770 = stablehlo.broadcast_in_dim %769, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %771 = stablehlo.compare  GT, %768, %770,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %772 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %773 = stablehlo.broadcast_in_dim %772, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %774 = stablehlo.convert %773 : tensor<1x1x7x20xf32>
    %775 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %776 = stablehlo.broadcast_in_dim %775, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %777 = stablehlo.select %771, %774, %776 : tensor<1x1x7x20xi1>, tensor<1x1x7x20xf32>
    %778 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %779 = stablehlo.sqrt %778 : tensor<f32>
    %780 = stablehlo.convert %779 : tensor<f32>
    %781 = stablehlo.broadcast_in_dim %780, dims = [] : (tensor<f32>) -> tensor<1x7x12x64xf32>
    %782 = stablehlo.divide %707, %781 : tensor<1x7x12x64xf32>
    %783 = stablehlo.dot_general %782, %741, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x7x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x7x20xf32>
    %784 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2, 3] : (tensor<1x1x7x20xf32>) -> tensor<1x12x7x20xf32>
    %785 = stablehlo.add %783, %784 : tensor<1x12x7x20xf32>
    %786 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %787 = stablehlo.reduce(%785 init: %786) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %788 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %789 = stablehlo.broadcast_in_dim %788, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %790 = stablehlo.subtract %785, %789 : tensor<1x12x7x20xf32>
    %791 = stablehlo.exponential %790 : tensor<1x12x7x20xf32>
    %792 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %793 = stablehlo.reduce(%791 init: %792) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %794 = stablehlo.broadcast_in_dim %793, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %795 = stablehlo.broadcast_in_dim %794, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %796 = stablehlo.divide %791, %795 : tensor<1x12x7x20xf32>
    %797 = stablehlo.dot_general %753, %796, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x7x20xf32>) -> tensor<1x12x64x7xf32>
    %798 = stablehlo.transpose %797, dims = [0, 3, 1, 2] : (tensor<1x12x64x7xf32>) -> tensor<1x7x12x64xf32>
    %799 = stablehlo.reshape %798 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %800 = stablehlo.transpose %30, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %801 = stablehlo.convert %800 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %802 = stablehlo.dot_general %799, %801, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x768xf32>) -> tensor<1x7x768xf32>
    %803 = stablehlo.convert %31 : (tensor<768xf16>) -> tensor<768xf32>
    %804 = stablehlo.broadcast_in_dim %803, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %805 = stablehlo.broadcast_in_dim %804, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %806 = stablehlo.add %802, %805 : tensor<1x7x768xf32>
    %807 = stablehlo.add %806, %654 : tensor<1x7x768xf32>
    %808 = stablehlo.multiply %807, %807 : tensor<1x7x768xf32>
    %809 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %810 = stablehlo.reduce(%808 init: %809) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %811 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %812 = stablehlo.broadcast_in_dim %811, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %813 = stablehlo.divide %810, %812 : tensor<1x7xf32>
    %814 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %815 = stablehlo.reduce(%807 init: %814) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %816 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %817 = stablehlo.broadcast_in_dim %816, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %818 = stablehlo.divide %815, %817 : tensor<1x7xf32>
    %819 = stablehlo.multiply %818, %818 : tensor<1x7xf32>
    %820 = stablehlo.subtract %813, %819 : tensor<1x7xf32>
    %821 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %822 = stablehlo.broadcast_in_dim %821, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %823 = stablehlo.maximum %822, %820 : tensor<1x7xf32>
    %824 = stablehlo.broadcast_in_dim %818, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %825 = stablehlo.broadcast_in_dim %823, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %826 = stablehlo.broadcast_in_dim %824, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %827 = stablehlo.subtract %807, %826 : tensor<1x7x768xf32>
    %828 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %829 = stablehlo.broadcast_in_dim %828, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %830 = stablehlo.add %825, %829 : tensor<1x7x1xf32>
    %831 = stablehlo.rsqrt %830 : tensor<1x7x1xf32>
    %832 = stablehlo.reshape %32 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %833 = stablehlo.convert %832 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %834 = stablehlo.broadcast_in_dim %831, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %835 = stablehlo.broadcast_in_dim %833, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %836 = stablehlo.multiply %834, %835 : tensor<1x7x768xf32>
    %837 = stablehlo.multiply %827, %836 : tensor<1x7x768xf32>
    %838 = stablehlo.reshape %33 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %839 = stablehlo.convert %838 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %840 = stablehlo.broadcast_in_dim %839, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %841 = stablehlo.add %837, %840 : tensor<1x7x768xf32>
    %842 = stablehlo.transpose %34, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %843 = stablehlo.convert %842 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %844 = stablehlo.dot_general %841, %843, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x3072xf32>) -> tensor<1x7x3072xf32>
    %845 = stablehlo.convert %35 : (tensor<3072xf16>) -> tensor<3072xf32>
    %846 = stablehlo.broadcast_in_dim %845, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %847 = stablehlo.broadcast_in_dim %846, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<1x7x3072xf32>
    %848 = stablehlo.add %844, %847 : tensor<1x7x3072xf32>
    %849 = stablehlo.multiply %848, %848 : tensor<1x7x3072xf32>
    %850 = stablehlo.multiply %848, %849 : tensor<1x7x3072xf32>
    %851 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %852 = stablehlo.broadcast_in_dim %851, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %853 = stablehlo.multiply %852, %850 : tensor<1x7x3072xf32>
    %854 = stablehlo.add %848, %853 : tensor<1x7x3072xf32>
    %855 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %856 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %857 = stablehlo.multiply %856, %854 : tensor<1x7x3072xf32>
    %858 = stablehlo.tanh %857 : tensor<1x7x3072xf32>
    %859 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %860 = stablehlo.broadcast_in_dim %859, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %861 = stablehlo.add %860, %858 : tensor<1x7x3072xf32>
    %862 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %863 = stablehlo.broadcast_in_dim %862, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %864 = stablehlo.multiply %863, %861 : tensor<1x7x3072xf32>
    %865 = stablehlo.multiply %848, %864 : tensor<1x7x3072xf32>
    %866 = stablehlo.transpose %36, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %867 = stablehlo.convert %866 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %868 = stablehlo.dot_general %865, %867, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x3072xf32>, tensor<3072x768xf32>) -> tensor<1x7x768xf32>
    %869 = stablehlo.convert %37 : (tensor<768xf16>) -> tensor<768xf32>
    %870 = stablehlo.broadcast_in_dim %869, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %871 = stablehlo.broadcast_in_dim %870, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %872 = stablehlo.add %868, %871 : tensor<1x7x768xf32>
    %873 = stablehlo.add %807, %872 : tensor<1x7x768xf32>
    %874 = stablehlo.multiply %873, %873 : tensor<1x7x768xf32>
    %875 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %876 = stablehlo.reduce(%874 init: %875) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %877 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %878 = stablehlo.broadcast_in_dim %877, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %879 = stablehlo.divide %876, %878 : tensor<1x7xf32>
    %880 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %881 = stablehlo.reduce(%873 init: %880) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %882 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %883 = stablehlo.broadcast_in_dim %882, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %884 = stablehlo.divide %881, %883 : tensor<1x7xf32>
    %885 = stablehlo.multiply %884, %884 : tensor<1x7xf32>
    %886 = stablehlo.subtract %879, %885 : tensor<1x7xf32>
    %887 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %888 = stablehlo.broadcast_in_dim %887, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %889 = stablehlo.maximum %888, %886 : tensor<1x7xf32>
    %890 = stablehlo.broadcast_in_dim %884, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %891 = stablehlo.broadcast_in_dim %889, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %892 = stablehlo.broadcast_in_dim %890, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %893 = stablehlo.subtract %873, %892 : tensor<1x7x768xf32>
    %894 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %895 = stablehlo.broadcast_in_dim %894, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %896 = stablehlo.add %891, %895 : tensor<1x7x1xf32>
    %897 = stablehlo.rsqrt %896 : tensor<1x7x1xf32>
    %898 = stablehlo.reshape %38 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %899 = stablehlo.convert %898 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %900 = stablehlo.broadcast_in_dim %897, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %901 = stablehlo.broadcast_in_dim %899, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %902 = stablehlo.multiply %900, %901 : tensor<1x7x768xf32>
    %903 = stablehlo.multiply %893, %902 : tensor<1x7x768xf32>
    %904 = stablehlo.reshape %39 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %905 = stablehlo.convert %904 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %906 = stablehlo.broadcast_in_dim %905, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %907 = stablehlo.add %903, %906 : tensor<1x7x768xf32>
    %908 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %909 = stablehlo.broadcast_in_dim %908, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %910 = stablehlo.broadcast_in_dim %909, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %911 = stablehlo.broadcast_in_dim %909, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %912 = stablehlo.broadcast_in_dim %910, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %913 = stablehlo.broadcast_in_dim %911, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %914 = stablehlo.compare  GE, %912, %913,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %915 = stablehlo.broadcast_in_dim %914, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %916 = stablehlo.transpose %40, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %917 = stablehlo.convert %916 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %918 = stablehlo.dot_general %907, %917, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x2304xf32>) -> tensor<1x7x2304xf32>
    %919 = stablehlo.convert %41 : (tensor<2304xf16>) -> tensor<2304xf32>
    %920 = stablehlo.broadcast_in_dim %919, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %921 = stablehlo.broadcast_in_dim %920, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<1x7x2304xf32>
    %922 = stablehlo.add %918, %921 : tensor<1x7x2304xf32>
    %923 = stablehlo.slice %922 [0:1, 0:7, 0:768] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %924 = stablehlo.slice %922 [0:1, 0:7, 768:1536] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %925 = stablehlo.slice %922 [0:1, 0:7, 1536:2304] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %926 = stablehlo.reshape %923 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %927 = stablehlo.reshape %924 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %928 = stablehlo.reshape %925 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %929 = stablehlo.constant dense<0> : tensor<i32>
    %930 = stablehlo.constant dense<0> : tensor<i32>
    %931 = stablehlo.compare  LT, %929, %930,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %932 = stablehlo.constant dense<0> : tensor<i32>
    %933 = stablehlo.constant dense<1024> : tensor<i32>
    %934 = stablehlo.add %932, %933 : tensor<i32>
    %935 = stablehlo.constant dense<0> : tensor<i32>
    %936 = stablehlo.select %931, %934, %935 : tensor<i1>, tensor<i32>
    %937 = stablehlo.constant dense<0> : tensor<i32>
    %938 = stablehlo.constant dense<0> : tensor<i32>
    %939 = stablehlo.constant dense<0> : tensor<i32>
    %940 = stablehlo.dynamic_slice %915, %937, %938, %936, %939, sizes = [1, 1, 7, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x7x20xi1>
    %941 = stablehlo.broadcast_in_dim %211, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %942 = stablehlo.reshape %941 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %943 = stablehlo.broadcast_in_dim %942, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x7x20xi32>
    %944 = stablehlo.constant dense<0> : tensor<i32>
    %945 = stablehlo.broadcast_in_dim %944, dims = [] : (tensor<i32>) -> tensor<1x1x7x20xi32>
    %946 = stablehlo.compare  NE, %943, %945,  SIGNED : (tensor<1x1x7x20xi32>, tensor<1x1x7x20xi32>) -> tensor<1x1x7x20xi1>
    %947 = stablehlo.and %946, %940 : tensor<1x1x7x20xi1>
    %948 = stablehlo.convert %947 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %949 = stablehlo.constant dense<0> : tensor<i32>
    %950 = stablehlo.constant dense<0> : tensor<i32>
    %951 = stablehlo.compare  LT, %949, %950,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %952 = stablehlo.constant dense<0> : tensor<i32>
    %953 = stablehlo.constant dense<20> : tensor<i32>
    %954 = stablehlo.add %952, %953 : tensor<i32>
    %955 = stablehlo.constant dense<0> : tensor<i32>
    %956 = stablehlo.select %951, %954, %955 : tensor<i1>, tensor<i32>
    %957 = stablehlo.constant dense<0> : tensor<i32>
    %958 = stablehlo.constant dense<0> : tensor<i32>
    %959 = stablehlo.constant dense<0> : tensor<i32>
    %960 = stablehlo.dynamic_update_slice %168, %927, %957, %956, %958, %959 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %961 = stablehlo.constant dense<0> : tensor<i32>
    %962 = stablehlo.constant dense<0> : tensor<i32>
    %963 = stablehlo.compare  LT, %961, %962,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %964 = stablehlo.constant dense<0> : tensor<i32>
    %965 = stablehlo.constant dense<20> : tensor<i32>
    %966 = stablehlo.add %964, %965 : tensor<i32>
    %967 = stablehlo.constant dense<0> : tensor<i32>
    %968 = stablehlo.select %963, %966, %967 : tensor<i1>, tensor<i32>
    %969 = stablehlo.constant dense<0> : tensor<i32>
    %970 = stablehlo.constant dense<0> : tensor<i32>
    %971 = stablehlo.constant dense<0> : tensor<i32>
    %972 = stablehlo.dynamic_update_slice %170, %928, %969, %968, %970, %971 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %973 = stablehlo.constant dense<0> : tensor<i32>
    %974 = stablehlo.constant dense<7> : tensor<i32>
    %975 = stablehlo.add %973, %974 : tensor<i32>
    %976 = stablehlo.iota dim = 0 : tensor<20xi32>
    %977 = stablehlo.constant dense<0> : tensor<i32>
    %978 = stablehlo.constant dense<7> : tensor<i32>
    %979 = stablehlo.add %977, %978 : tensor<i32>
    %980 = stablehlo.broadcast_in_dim %979, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %981 = stablehlo.compare  LT, %976, %980,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %982 = stablehlo.broadcast_in_dim %981, dims = [3] : (tensor<20xi1>) -> tensor<1x1x7x20xi1>
    %983 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %984 = stablehlo.broadcast_in_dim %983, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %985 = stablehlo.compare  NE, %948, %984,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %986 = stablehlo.and %982, %985 : tensor<1x1x7x20xi1>
    %987 = stablehlo.convert %986 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %988 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %989 = stablehlo.broadcast_in_dim %988, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %990 = stablehlo.compare  GT, %987, %989,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %991 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %992 = stablehlo.broadcast_in_dim %991, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %993 = stablehlo.convert %992 : tensor<1x1x7x20xf32>
    %994 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %995 = stablehlo.broadcast_in_dim %994, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %996 = stablehlo.select %990, %993, %995 : tensor<1x1x7x20xi1>, tensor<1x1x7x20xf32>
    %997 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %998 = stablehlo.sqrt %997 : tensor<f32>
    %999 = stablehlo.convert %998 : tensor<f32>
    %1000 = stablehlo.broadcast_in_dim %999, dims = [] : (tensor<f32>) -> tensor<1x7x12x64xf32>
    %1001 = stablehlo.divide %926, %1000 : tensor<1x7x12x64xf32>
    %1002 = stablehlo.dot_general %1001, %960, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x7x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x7x20xf32>
    %1003 = stablehlo.broadcast_in_dim %996, dims = [0, 1, 2, 3] : (tensor<1x1x7x20xf32>) -> tensor<1x12x7x20xf32>
    %1004 = stablehlo.add %1002, %1003 : tensor<1x12x7x20xf32>
    %1005 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1006 = stablehlo.reduce(%1004 init: %1005) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1007 = stablehlo.broadcast_in_dim %1006, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1008 = stablehlo.broadcast_in_dim %1007, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %1009 = stablehlo.subtract %1004, %1008 : tensor<1x12x7x20xf32>
    %1010 = stablehlo.exponential %1009 : tensor<1x12x7x20xf32>
    %1011 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1012 = stablehlo.reduce(%1010 init: %1011) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1013 = stablehlo.broadcast_in_dim %1012, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1014 = stablehlo.broadcast_in_dim %1013, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %1015 = stablehlo.divide %1010, %1014 : tensor<1x12x7x20xf32>
    %1016 = stablehlo.dot_general %972, %1015, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x7x20xf32>) -> tensor<1x12x64x7xf32>
    %1017 = stablehlo.transpose %1016, dims = [0, 3, 1, 2] : (tensor<1x12x64x7xf32>) -> tensor<1x7x12x64xf32>
    %1018 = stablehlo.reshape %1017 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %1019 = stablehlo.transpose %42, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %1020 = stablehlo.convert %1019 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1021 = stablehlo.dot_general %1018, %1020, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x768xf32>) -> tensor<1x7x768xf32>
    %1022 = stablehlo.convert %43 : (tensor<768xf16>) -> tensor<768xf32>
    %1023 = stablehlo.broadcast_in_dim %1022, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1024 = stablehlo.broadcast_in_dim %1023, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1025 = stablehlo.add %1021, %1024 : tensor<1x7x768xf32>
    %1026 = stablehlo.add %1025, %873 : tensor<1x7x768xf32>
    %1027 = stablehlo.multiply %1026, %1026 : tensor<1x7x768xf32>
    %1028 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1029 = stablehlo.reduce(%1027 init: %1028) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1030 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1031 = stablehlo.broadcast_in_dim %1030, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1032 = stablehlo.divide %1029, %1031 : tensor<1x7xf32>
    %1033 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1034 = stablehlo.reduce(%1026 init: %1033) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1035 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1036 = stablehlo.broadcast_in_dim %1035, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1037 = stablehlo.divide %1034, %1036 : tensor<1x7xf32>
    %1038 = stablehlo.multiply %1037, %1037 : tensor<1x7xf32>
    %1039 = stablehlo.subtract %1032, %1038 : tensor<1x7xf32>
    %1040 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1041 = stablehlo.broadcast_in_dim %1040, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1042 = stablehlo.maximum %1041, %1039 : tensor<1x7xf32>
    %1043 = stablehlo.broadcast_in_dim %1037, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1044 = stablehlo.broadcast_in_dim %1042, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1045 = stablehlo.broadcast_in_dim %1043, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1046 = stablehlo.subtract %1026, %1045 : tensor<1x7x768xf32>
    %1047 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1048 = stablehlo.broadcast_in_dim %1047, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1049 = stablehlo.add %1044, %1048 : tensor<1x7x1xf32>
    %1050 = stablehlo.rsqrt %1049 : tensor<1x7x1xf32>
    %1051 = stablehlo.reshape %44 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1052 = stablehlo.convert %1051 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1053 = stablehlo.broadcast_in_dim %1050, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1054 = stablehlo.broadcast_in_dim %1052, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1055 = stablehlo.multiply %1053, %1054 : tensor<1x7x768xf32>
    %1056 = stablehlo.multiply %1046, %1055 : tensor<1x7x768xf32>
    %1057 = stablehlo.reshape %45 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1058 = stablehlo.convert %1057 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1059 = stablehlo.broadcast_in_dim %1058, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1060 = stablehlo.add %1056, %1059 : tensor<1x7x768xf32>
    %1061 = stablehlo.transpose %46, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %1062 = stablehlo.convert %1061 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1063 = stablehlo.dot_general %1060, %1062, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x3072xf32>) -> tensor<1x7x3072xf32>
    %1064 = stablehlo.convert %47 : (tensor<3072xf16>) -> tensor<3072xf32>
    %1065 = stablehlo.broadcast_in_dim %1064, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %1066 = stablehlo.broadcast_in_dim %1065, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<1x7x3072xf32>
    %1067 = stablehlo.add %1063, %1066 : tensor<1x7x3072xf32>
    %1068 = stablehlo.multiply %1067, %1067 : tensor<1x7x3072xf32>
    %1069 = stablehlo.multiply %1067, %1068 : tensor<1x7x3072xf32>
    %1070 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %1071 = stablehlo.broadcast_in_dim %1070, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1072 = stablehlo.multiply %1071, %1069 : tensor<1x7x3072xf32>
    %1073 = stablehlo.add %1067, %1072 : tensor<1x7x3072xf32>
    %1074 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %1075 = stablehlo.broadcast_in_dim %1074, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1076 = stablehlo.multiply %1075, %1073 : tensor<1x7x3072xf32>
    %1077 = stablehlo.tanh %1076 : tensor<1x7x3072xf32>
    %1078 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1079 = stablehlo.broadcast_in_dim %1078, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1080 = stablehlo.add %1079, %1077 : tensor<1x7x3072xf32>
    %1081 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %1082 = stablehlo.broadcast_in_dim %1081, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1083 = stablehlo.multiply %1082, %1080 : tensor<1x7x3072xf32>
    %1084 = stablehlo.multiply %1067, %1083 : tensor<1x7x3072xf32>
    %1085 = stablehlo.transpose %48, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %1086 = stablehlo.convert %1085 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1087 = stablehlo.dot_general %1084, %1086, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x3072xf32>, tensor<3072x768xf32>) -> tensor<1x7x768xf32>
    %1088 = stablehlo.convert %49 : (tensor<768xf16>) -> tensor<768xf32>
    %1089 = stablehlo.broadcast_in_dim %1088, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1090 = stablehlo.broadcast_in_dim %1089, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1091 = stablehlo.add %1087, %1090 : tensor<1x7x768xf32>
    %1092 = stablehlo.add %1026, %1091 : tensor<1x7x768xf32>
    %1093 = stablehlo.multiply %1092, %1092 : tensor<1x7x768xf32>
    %1094 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1095 = stablehlo.reduce(%1093 init: %1094) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1096 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1097 = stablehlo.broadcast_in_dim %1096, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1098 = stablehlo.divide %1095, %1097 : tensor<1x7xf32>
    %1099 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1100 = stablehlo.reduce(%1092 init: %1099) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1101 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1102 = stablehlo.broadcast_in_dim %1101, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1103 = stablehlo.divide %1100, %1102 : tensor<1x7xf32>
    %1104 = stablehlo.multiply %1103, %1103 : tensor<1x7xf32>
    %1105 = stablehlo.subtract %1098, %1104 : tensor<1x7xf32>
    %1106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1107 = stablehlo.broadcast_in_dim %1106, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1108 = stablehlo.maximum %1107, %1105 : tensor<1x7xf32>
    %1109 = stablehlo.broadcast_in_dim %1103, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1110 = stablehlo.broadcast_in_dim %1108, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1111 = stablehlo.broadcast_in_dim %1109, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1112 = stablehlo.subtract %1092, %1111 : tensor<1x7x768xf32>
    %1113 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1114 = stablehlo.broadcast_in_dim %1113, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1115 = stablehlo.add %1110, %1114 : tensor<1x7x1xf32>
    %1116 = stablehlo.rsqrt %1115 : tensor<1x7x1xf32>
    %1117 = stablehlo.reshape %50 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1118 = stablehlo.convert %1117 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1119 = stablehlo.broadcast_in_dim %1116, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1120 = stablehlo.broadcast_in_dim %1118, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1121 = stablehlo.multiply %1119, %1120 : tensor<1x7x768xf32>
    %1122 = stablehlo.multiply %1112, %1121 : tensor<1x7x768xf32>
    %1123 = stablehlo.reshape %51 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1124 = stablehlo.convert %1123 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1125 = stablehlo.broadcast_in_dim %1124, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1126 = stablehlo.add %1122, %1125 : tensor<1x7x768xf32>
    %1127 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %1128 = stablehlo.broadcast_in_dim %1127, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %1129 = stablehlo.broadcast_in_dim %1128, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %1130 = stablehlo.broadcast_in_dim %1128, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %1131 = stablehlo.broadcast_in_dim %1129, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %1132 = stablehlo.broadcast_in_dim %1130, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %1133 = stablehlo.compare  GE, %1131, %1132,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %1134 = stablehlo.broadcast_in_dim %1133, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %1135 = stablehlo.transpose %52, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %1136 = stablehlo.convert %1135 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %1137 = stablehlo.dot_general %1126, %1136, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x2304xf32>) -> tensor<1x7x2304xf32>
    %1138 = stablehlo.convert %53 : (tensor<2304xf16>) -> tensor<2304xf32>
    %1139 = stablehlo.broadcast_in_dim %1138, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %1140 = stablehlo.broadcast_in_dim %1139, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<1x7x2304xf32>
    %1141 = stablehlo.add %1137, %1140 : tensor<1x7x2304xf32>
    %1142 = stablehlo.slice %1141 [0:1, 0:7, 0:768] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %1143 = stablehlo.slice %1141 [0:1, 0:7, 768:1536] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %1144 = stablehlo.slice %1141 [0:1, 0:7, 1536:2304] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %1145 = stablehlo.reshape %1142 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1146 = stablehlo.reshape %1143 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1147 = stablehlo.reshape %1144 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1148 = stablehlo.constant dense<0> : tensor<i32>
    %1149 = stablehlo.constant dense<0> : tensor<i32>
    %1150 = stablehlo.compare  LT, %1148, %1149,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1151 = stablehlo.constant dense<0> : tensor<i32>
    %1152 = stablehlo.constant dense<1024> : tensor<i32>
    %1153 = stablehlo.add %1151, %1152 : tensor<i32>
    %1154 = stablehlo.constant dense<0> : tensor<i32>
    %1155 = stablehlo.select %1150, %1153, %1154 : tensor<i1>, tensor<i32>
    %1156 = stablehlo.constant dense<0> : tensor<i32>
    %1157 = stablehlo.constant dense<0> : tensor<i32>
    %1158 = stablehlo.constant dense<0> : tensor<i32>
    %1159 = stablehlo.dynamic_slice %1134, %1156, %1157, %1155, %1158, sizes = [1, 1, 7, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x7x20xi1>
    %1160 = stablehlo.broadcast_in_dim %211, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %1161 = stablehlo.reshape %1160 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %1162 = stablehlo.broadcast_in_dim %1161, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x7x20xi32>
    %1163 = stablehlo.constant dense<0> : tensor<i32>
    %1164 = stablehlo.broadcast_in_dim %1163, dims = [] : (tensor<i32>) -> tensor<1x1x7x20xi32>
    %1165 = stablehlo.compare  NE, %1162, %1164,  SIGNED : (tensor<1x1x7x20xi32>, tensor<1x1x7x20xi32>) -> tensor<1x1x7x20xi1>
    %1166 = stablehlo.and %1165, %1159 : tensor<1x1x7x20xi1>
    %1167 = stablehlo.convert %1166 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %1168 = stablehlo.constant dense<0> : tensor<i32>
    %1169 = stablehlo.constant dense<0> : tensor<i32>
    %1170 = stablehlo.compare  LT, %1168, %1169,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1171 = stablehlo.constant dense<0> : tensor<i32>
    %1172 = stablehlo.constant dense<20> : tensor<i32>
    %1173 = stablehlo.add %1171, %1172 : tensor<i32>
    %1174 = stablehlo.constant dense<0> : tensor<i32>
    %1175 = stablehlo.select %1170, %1173, %1174 : tensor<i1>, tensor<i32>
    %1176 = stablehlo.constant dense<0> : tensor<i32>
    %1177 = stablehlo.constant dense<0> : tensor<i32>
    %1178 = stablehlo.constant dense<0> : tensor<i32>
    %1179 = stablehlo.dynamic_update_slice %172, %1146, %1176, %1175, %1177, %1178 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %1180 = stablehlo.constant dense<0> : tensor<i32>
    %1181 = stablehlo.constant dense<0> : tensor<i32>
    %1182 = stablehlo.compare  LT, %1180, %1181,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1183 = stablehlo.constant dense<0> : tensor<i32>
    %1184 = stablehlo.constant dense<20> : tensor<i32>
    %1185 = stablehlo.add %1183, %1184 : tensor<i32>
    %1186 = stablehlo.constant dense<0> : tensor<i32>
    %1187 = stablehlo.select %1182, %1185, %1186 : tensor<i1>, tensor<i32>
    %1188 = stablehlo.constant dense<0> : tensor<i32>
    %1189 = stablehlo.constant dense<0> : tensor<i32>
    %1190 = stablehlo.constant dense<0> : tensor<i32>
    %1191 = stablehlo.dynamic_update_slice %174, %1147, %1188, %1187, %1189, %1190 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %1192 = stablehlo.constant dense<0> : tensor<i32>
    %1193 = stablehlo.constant dense<7> : tensor<i32>
    %1194 = stablehlo.add %1192, %1193 : tensor<i32>
    %1195 = stablehlo.iota dim = 0 : tensor<20xi32>
    %1196 = stablehlo.constant dense<0> : tensor<i32>
    %1197 = stablehlo.constant dense<7> : tensor<i32>
    %1198 = stablehlo.add %1196, %1197 : tensor<i32>
    %1199 = stablehlo.broadcast_in_dim %1198, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %1200 = stablehlo.compare  LT, %1195, %1199,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %1201 = stablehlo.broadcast_in_dim %1200, dims = [3] : (tensor<20xi1>) -> tensor<1x1x7x20xi1>
    %1202 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1203 = stablehlo.broadcast_in_dim %1202, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1204 = stablehlo.compare  NE, %1167, %1203,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %1205 = stablehlo.and %1201, %1204 : tensor<1x1x7x20xi1>
    %1206 = stablehlo.convert %1205 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %1207 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1208 = stablehlo.broadcast_in_dim %1207, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1209 = stablehlo.compare  GT, %1206, %1208,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %1210 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1211 = stablehlo.broadcast_in_dim %1210, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1212 = stablehlo.convert %1211 : tensor<1x1x7x20xf32>
    %1213 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %1214 = stablehlo.broadcast_in_dim %1213, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1215 = stablehlo.select %1209, %1212, %1214 : tensor<1x1x7x20xi1>, tensor<1x1x7x20xf32>
    %1216 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %1217 = stablehlo.sqrt %1216 : tensor<f32>
    %1218 = stablehlo.convert %1217 : tensor<f32>
    %1219 = stablehlo.broadcast_in_dim %1218, dims = [] : (tensor<f32>) -> tensor<1x7x12x64xf32>
    %1220 = stablehlo.divide %1145, %1219 : tensor<1x7x12x64xf32>
    %1221 = stablehlo.dot_general %1220, %1179, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x7x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x7x20xf32>
    %1222 = stablehlo.broadcast_in_dim %1215, dims = [0, 1, 2, 3] : (tensor<1x1x7x20xf32>) -> tensor<1x12x7x20xf32>
    %1223 = stablehlo.add %1221, %1222 : tensor<1x12x7x20xf32>
    %1224 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1225 = stablehlo.reduce(%1223 init: %1224) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1226 = stablehlo.broadcast_in_dim %1225, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1227 = stablehlo.broadcast_in_dim %1226, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %1228 = stablehlo.subtract %1223, %1227 : tensor<1x12x7x20xf32>
    %1229 = stablehlo.exponential %1228 : tensor<1x12x7x20xf32>
    %1230 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1231 = stablehlo.reduce(%1229 init: %1230) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1232 = stablehlo.broadcast_in_dim %1231, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1233 = stablehlo.broadcast_in_dim %1232, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %1234 = stablehlo.divide %1229, %1233 : tensor<1x12x7x20xf32>
    %1235 = stablehlo.dot_general %1191, %1234, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x7x20xf32>) -> tensor<1x12x64x7xf32>
    %1236 = stablehlo.transpose %1235, dims = [0, 3, 1, 2] : (tensor<1x12x64x7xf32>) -> tensor<1x7x12x64xf32>
    %1237 = stablehlo.reshape %1236 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %1238 = stablehlo.transpose %54, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %1239 = stablehlo.convert %1238 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1240 = stablehlo.dot_general %1237, %1239, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x768xf32>) -> tensor<1x7x768xf32>
    %1241 = stablehlo.convert %55 : (tensor<768xf16>) -> tensor<768xf32>
    %1242 = stablehlo.broadcast_in_dim %1241, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1243 = stablehlo.broadcast_in_dim %1242, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1244 = stablehlo.add %1240, %1243 : tensor<1x7x768xf32>
    %1245 = stablehlo.add %1244, %1092 : tensor<1x7x768xf32>
    %1246 = stablehlo.multiply %1245, %1245 : tensor<1x7x768xf32>
    %1247 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1248 = stablehlo.reduce(%1246 init: %1247) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1249 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1250 = stablehlo.broadcast_in_dim %1249, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1251 = stablehlo.divide %1248, %1250 : tensor<1x7xf32>
    %1252 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1253 = stablehlo.reduce(%1245 init: %1252) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1254 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1255 = stablehlo.broadcast_in_dim %1254, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1256 = stablehlo.divide %1253, %1255 : tensor<1x7xf32>
    %1257 = stablehlo.multiply %1256, %1256 : tensor<1x7xf32>
    %1258 = stablehlo.subtract %1251, %1257 : tensor<1x7xf32>
    %1259 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1260 = stablehlo.broadcast_in_dim %1259, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1261 = stablehlo.maximum %1260, %1258 : tensor<1x7xf32>
    %1262 = stablehlo.broadcast_in_dim %1256, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1263 = stablehlo.broadcast_in_dim %1261, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1264 = stablehlo.broadcast_in_dim %1262, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1265 = stablehlo.subtract %1245, %1264 : tensor<1x7x768xf32>
    %1266 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1267 = stablehlo.broadcast_in_dim %1266, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1268 = stablehlo.add %1263, %1267 : tensor<1x7x1xf32>
    %1269 = stablehlo.rsqrt %1268 : tensor<1x7x1xf32>
    %1270 = stablehlo.reshape %56 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1271 = stablehlo.convert %1270 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1272 = stablehlo.broadcast_in_dim %1269, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1273 = stablehlo.broadcast_in_dim %1271, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1274 = stablehlo.multiply %1272, %1273 : tensor<1x7x768xf32>
    %1275 = stablehlo.multiply %1265, %1274 : tensor<1x7x768xf32>
    %1276 = stablehlo.reshape %57 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1277 = stablehlo.convert %1276 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1278 = stablehlo.broadcast_in_dim %1277, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1279 = stablehlo.add %1275, %1278 : tensor<1x7x768xf32>
    %1280 = stablehlo.transpose %58, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %1281 = stablehlo.convert %1280 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1282 = stablehlo.dot_general %1279, %1281, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x3072xf32>) -> tensor<1x7x3072xf32>
    %1283 = stablehlo.convert %59 : (tensor<3072xf16>) -> tensor<3072xf32>
    %1284 = stablehlo.broadcast_in_dim %1283, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %1285 = stablehlo.broadcast_in_dim %1284, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<1x7x3072xf32>
    %1286 = stablehlo.add %1282, %1285 : tensor<1x7x3072xf32>
    %1287 = stablehlo.multiply %1286, %1286 : tensor<1x7x3072xf32>
    %1288 = stablehlo.multiply %1286, %1287 : tensor<1x7x3072xf32>
    %1289 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %1290 = stablehlo.broadcast_in_dim %1289, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1291 = stablehlo.multiply %1290, %1288 : tensor<1x7x3072xf32>
    %1292 = stablehlo.add %1286, %1291 : tensor<1x7x3072xf32>
    %1293 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %1294 = stablehlo.broadcast_in_dim %1293, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1295 = stablehlo.multiply %1294, %1292 : tensor<1x7x3072xf32>
    %1296 = stablehlo.tanh %1295 : tensor<1x7x3072xf32>
    %1297 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1298 = stablehlo.broadcast_in_dim %1297, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1299 = stablehlo.add %1298, %1296 : tensor<1x7x3072xf32>
    %1300 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %1301 = stablehlo.broadcast_in_dim %1300, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1302 = stablehlo.multiply %1301, %1299 : tensor<1x7x3072xf32>
    %1303 = stablehlo.multiply %1286, %1302 : tensor<1x7x3072xf32>
    %1304 = stablehlo.transpose %60, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %1305 = stablehlo.convert %1304 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1306 = stablehlo.dot_general %1303, %1305, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x3072xf32>, tensor<3072x768xf32>) -> tensor<1x7x768xf32>
    %1307 = stablehlo.convert %61 : (tensor<768xf16>) -> tensor<768xf32>
    %1308 = stablehlo.broadcast_in_dim %1307, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1309 = stablehlo.broadcast_in_dim %1308, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1310 = stablehlo.add %1306, %1309 : tensor<1x7x768xf32>
    %1311 = stablehlo.add %1245, %1310 : tensor<1x7x768xf32>
    %1312 = stablehlo.multiply %1311, %1311 : tensor<1x7x768xf32>
    %1313 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1314 = stablehlo.reduce(%1312 init: %1313) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1315 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1316 = stablehlo.broadcast_in_dim %1315, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1317 = stablehlo.divide %1314, %1316 : tensor<1x7xf32>
    %1318 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1319 = stablehlo.reduce(%1311 init: %1318) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1320 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1321 = stablehlo.broadcast_in_dim %1320, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1322 = stablehlo.divide %1319, %1321 : tensor<1x7xf32>
    %1323 = stablehlo.multiply %1322, %1322 : tensor<1x7xf32>
    %1324 = stablehlo.subtract %1317, %1323 : tensor<1x7xf32>
    %1325 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1326 = stablehlo.broadcast_in_dim %1325, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1327 = stablehlo.maximum %1326, %1324 : tensor<1x7xf32>
    %1328 = stablehlo.broadcast_in_dim %1322, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1329 = stablehlo.broadcast_in_dim %1327, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1330 = stablehlo.broadcast_in_dim %1328, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1331 = stablehlo.subtract %1311, %1330 : tensor<1x7x768xf32>
    %1332 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1333 = stablehlo.broadcast_in_dim %1332, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1334 = stablehlo.add %1329, %1333 : tensor<1x7x1xf32>
    %1335 = stablehlo.rsqrt %1334 : tensor<1x7x1xf32>
    %1336 = stablehlo.reshape %62 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1337 = stablehlo.convert %1336 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1338 = stablehlo.broadcast_in_dim %1335, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1339 = stablehlo.broadcast_in_dim %1337, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1340 = stablehlo.multiply %1338, %1339 : tensor<1x7x768xf32>
    %1341 = stablehlo.multiply %1331, %1340 : tensor<1x7x768xf32>
    %1342 = stablehlo.reshape %63 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1343 = stablehlo.convert %1342 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1344 = stablehlo.broadcast_in_dim %1343, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1345 = stablehlo.add %1341, %1344 : tensor<1x7x768xf32>
    %1346 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %1347 = stablehlo.broadcast_in_dim %1346, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %1348 = stablehlo.broadcast_in_dim %1347, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %1349 = stablehlo.broadcast_in_dim %1347, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %1350 = stablehlo.broadcast_in_dim %1348, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %1351 = stablehlo.broadcast_in_dim %1349, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %1352 = stablehlo.compare  GE, %1350, %1351,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %1353 = stablehlo.broadcast_in_dim %1352, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %1354 = stablehlo.transpose %64, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %1355 = stablehlo.convert %1354 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %1356 = stablehlo.dot_general %1345, %1355, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x2304xf32>) -> tensor<1x7x2304xf32>
    %1357 = stablehlo.convert %65 : (tensor<2304xf16>) -> tensor<2304xf32>
    %1358 = stablehlo.broadcast_in_dim %1357, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %1359 = stablehlo.broadcast_in_dim %1358, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<1x7x2304xf32>
    %1360 = stablehlo.add %1356, %1359 : tensor<1x7x2304xf32>
    %1361 = stablehlo.slice %1360 [0:1, 0:7, 0:768] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %1362 = stablehlo.slice %1360 [0:1, 0:7, 768:1536] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %1363 = stablehlo.slice %1360 [0:1, 0:7, 1536:2304] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %1364 = stablehlo.reshape %1361 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1365 = stablehlo.reshape %1362 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1366 = stablehlo.reshape %1363 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1367 = stablehlo.constant dense<0> : tensor<i32>
    %1368 = stablehlo.constant dense<0> : tensor<i32>
    %1369 = stablehlo.compare  LT, %1367, %1368,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1370 = stablehlo.constant dense<0> : tensor<i32>
    %1371 = stablehlo.constant dense<1024> : tensor<i32>
    %1372 = stablehlo.add %1370, %1371 : tensor<i32>
    %1373 = stablehlo.constant dense<0> : tensor<i32>
    %1374 = stablehlo.select %1369, %1372, %1373 : tensor<i1>, tensor<i32>
    %1375 = stablehlo.constant dense<0> : tensor<i32>
    %1376 = stablehlo.constant dense<0> : tensor<i32>
    %1377 = stablehlo.constant dense<0> : tensor<i32>
    %1378 = stablehlo.dynamic_slice %1353, %1375, %1376, %1374, %1377, sizes = [1, 1, 7, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x7x20xi1>
    %1379 = stablehlo.broadcast_in_dim %211, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %1380 = stablehlo.reshape %1379 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %1381 = stablehlo.broadcast_in_dim %1380, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x7x20xi32>
    %1382 = stablehlo.constant dense<0> : tensor<i32>
    %1383 = stablehlo.broadcast_in_dim %1382, dims = [] : (tensor<i32>) -> tensor<1x1x7x20xi32>
    %1384 = stablehlo.compare  NE, %1381, %1383,  SIGNED : (tensor<1x1x7x20xi32>, tensor<1x1x7x20xi32>) -> tensor<1x1x7x20xi1>
    %1385 = stablehlo.and %1384, %1378 : tensor<1x1x7x20xi1>
    %1386 = stablehlo.convert %1385 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %1387 = stablehlo.constant dense<0> : tensor<i32>
    %1388 = stablehlo.constant dense<0> : tensor<i32>
    %1389 = stablehlo.compare  LT, %1387, %1388,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1390 = stablehlo.constant dense<0> : tensor<i32>
    %1391 = stablehlo.constant dense<20> : tensor<i32>
    %1392 = stablehlo.add %1390, %1391 : tensor<i32>
    %1393 = stablehlo.constant dense<0> : tensor<i32>
    %1394 = stablehlo.select %1389, %1392, %1393 : tensor<i1>, tensor<i32>
    %1395 = stablehlo.constant dense<0> : tensor<i32>
    %1396 = stablehlo.constant dense<0> : tensor<i32>
    %1397 = stablehlo.constant dense<0> : tensor<i32>
    %1398 = stablehlo.dynamic_update_slice %176, %1365, %1395, %1394, %1396, %1397 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %1399 = stablehlo.constant dense<0> : tensor<i32>
    %1400 = stablehlo.constant dense<0> : tensor<i32>
    %1401 = stablehlo.compare  LT, %1399, %1400,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1402 = stablehlo.constant dense<0> : tensor<i32>
    %1403 = stablehlo.constant dense<20> : tensor<i32>
    %1404 = stablehlo.add %1402, %1403 : tensor<i32>
    %1405 = stablehlo.constant dense<0> : tensor<i32>
    %1406 = stablehlo.select %1401, %1404, %1405 : tensor<i1>, tensor<i32>
    %1407 = stablehlo.constant dense<0> : tensor<i32>
    %1408 = stablehlo.constant dense<0> : tensor<i32>
    %1409 = stablehlo.constant dense<0> : tensor<i32>
    %1410 = stablehlo.dynamic_update_slice %178, %1366, %1407, %1406, %1408, %1409 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %1411 = stablehlo.constant dense<0> : tensor<i32>
    %1412 = stablehlo.constant dense<7> : tensor<i32>
    %1413 = stablehlo.add %1411, %1412 : tensor<i32>
    %1414 = stablehlo.iota dim = 0 : tensor<20xi32>
    %1415 = stablehlo.constant dense<0> : tensor<i32>
    %1416 = stablehlo.constant dense<7> : tensor<i32>
    %1417 = stablehlo.add %1415, %1416 : tensor<i32>
    %1418 = stablehlo.broadcast_in_dim %1417, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %1419 = stablehlo.compare  LT, %1414, %1418,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %1420 = stablehlo.broadcast_in_dim %1419, dims = [3] : (tensor<20xi1>) -> tensor<1x1x7x20xi1>
    %1421 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1422 = stablehlo.broadcast_in_dim %1421, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1423 = stablehlo.compare  NE, %1386, %1422,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %1424 = stablehlo.and %1420, %1423 : tensor<1x1x7x20xi1>
    %1425 = stablehlo.convert %1424 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %1426 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1427 = stablehlo.broadcast_in_dim %1426, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1428 = stablehlo.compare  GT, %1425, %1427,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %1429 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1430 = stablehlo.broadcast_in_dim %1429, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1431 = stablehlo.convert %1430 : tensor<1x1x7x20xf32>
    %1432 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %1433 = stablehlo.broadcast_in_dim %1432, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1434 = stablehlo.select %1428, %1431, %1433 : tensor<1x1x7x20xi1>, tensor<1x1x7x20xf32>
    %1435 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %1436 = stablehlo.sqrt %1435 : tensor<f32>
    %1437 = stablehlo.convert %1436 : tensor<f32>
    %1438 = stablehlo.broadcast_in_dim %1437, dims = [] : (tensor<f32>) -> tensor<1x7x12x64xf32>
    %1439 = stablehlo.divide %1364, %1438 : tensor<1x7x12x64xf32>
    %1440 = stablehlo.dot_general %1439, %1398, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x7x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x7x20xf32>
    %1441 = stablehlo.broadcast_in_dim %1434, dims = [0, 1, 2, 3] : (tensor<1x1x7x20xf32>) -> tensor<1x12x7x20xf32>
    %1442 = stablehlo.add %1440, %1441 : tensor<1x12x7x20xf32>
    %1443 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1444 = stablehlo.reduce(%1442 init: %1443) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1445 = stablehlo.broadcast_in_dim %1444, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1446 = stablehlo.broadcast_in_dim %1445, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %1447 = stablehlo.subtract %1442, %1446 : tensor<1x12x7x20xf32>
    %1448 = stablehlo.exponential %1447 : tensor<1x12x7x20xf32>
    %1449 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1450 = stablehlo.reduce(%1448 init: %1449) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1451 = stablehlo.broadcast_in_dim %1450, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1452 = stablehlo.broadcast_in_dim %1451, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %1453 = stablehlo.divide %1448, %1452 : tensor<1x12x7x20xf32>
    %1454 = stablehlo.dot_general %1410, %1453, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x7x20xf32>) -> tensor<1x12x64x7xf32>
    %1455 = stablehlo.transpose %1454, dims = [0, 3, 1, 2] : (tensor<1x12x64x7xf32>) -> tensor<1x7x12x64xf32>
    %1456 = stablehlo.reshape %1455 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %1457 = stablehlo.transpose %66, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %1458 = stablehlo.convert %1457 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1459 = stablehlo.dot_general %1456, %1458, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x768xf32>) -> tensor<1x7x768xf32>
    %1460 = stablehlo.convert %67 : (tensor<768xf16>) -> tensor<768xf32>
    %1461 = stablehlo.broadcast_in_dim %1460, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1462 = stablehlo.broadcast_in_dim %1461, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1463 = stablehlo.add %1459, %1462 : tensor<1x7x768xf32>
    %1464 = stablehlo.add %1463, %1311 : tensor<1x7x768xf32>
    %1465 = stablehlo.multiply %1464, %1464 : tensor<1x7x768xf32>
    %1466 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1467 = stablehlo.reduce(%1465 init: %1466) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1468 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1469 = stablehlo.broadcast_in_dim %1468, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1470 = stablehlo.divide %1467, %1469 : tensor<1x7xf32>
    %1471 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1472 = stablehlo.reduce(%1464 init: %1471) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1473 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1474 = stablehlo.broadcast_in_dim %1473, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1475 = stablehlo.divide %1472, %1474 : tensor<1x7xf32>
    %1476 = stablehlo.multiply %1475, %1475 : tensor<1x7xf32>
    %1477 = stablehlo.subtract %1470, %1476 : tensor<1x7xf32>
    %1478 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1479 = stablehlo.broadcast_in_dim %1478, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1480 = stablehlo.maximum %1479, %1477 : tensor<1x7xf32>
    %1481 = stablehlo.broadcast_in_dim %1475, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1482 = stablehlo.broadcast_in_dim %1480, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1483 = stablehlo.broadcast_in_dim %1481, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1484 = stablehlo.subtract %1464, %1483 : tensor<1x7x768xf32>
    %1485 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1486 = stablehlo.broadcast_in_dim %1485, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1487 = stablehlo.add %1482, %1486 : tensor<1x7x1xf32>
    %1488 = stablehlo.rsqrt %1487 : tensor<1x7x1xf32>
    %1489 = stablehlo.reshape %68 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1490 = stablehlo.convert %1489 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1491 = stablehlo.broadcast_in_dim %1488, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1492 = stablehlo.broadcast_in_dim %1490, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1493 = stablehlo.multiply %1491, %1492 : tensor<1x7x768xf32>
    %1494 = stablehlo.multiply %1484, %1493 : tensor<1x7x768xf32>
    %1495 = stablehlo.reshape %69 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1496 = stablehlo.convert %1495 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1497 = stablehlo.broadcast_in_dim %1496, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1498 = stablehlo.add %1494, %1497 : tensor<1x7x768xf32>
    %1499 = stablehlo.transpose %70, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %1500 = stablehlo.convert %1499 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1501 = stablehlo.dot_general %1498, %1500, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x3072xf32>) -> tensor<1x7x3072xf32>
    %1502 = stablehlo.convert %71 : (tensor<3072xf16>) -> tensor<3072xf32>
    %1503 = stablehlo.broadcast_in_dim %1502, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %1504 = stablehlo.broadcast_in_dim %1503, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<1x7x3072xf32>
    %1505 = stablehlo.add %1501, %1504 : tensor<1x7x3072xf32>
    %1506 = stablehlo.multiply %1505, %1505 : tensor<1x7x3072xf32>
    %1507 = stablehlo.multiply %1505, %1506 : tensor<1x7x3072xf32>
    %1508 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %1509 = stablehlo.broadcast_in_dim %1508, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1510 = stablehlo.multiply %1509, %1507 : tensor<1x7x3072xf32>
    %1511 = stablehlo.add %1505, %1510 : tensor<1x7x3072xf32>
    %1512 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %1513 = stablehlo.broadcast_in_dim %1512, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1514 = stablehlo.multiply %1513, %1511 : tensor<1x7x3072xf32>
    %1515 = stablehlo.tanh %1514 : tensor<1x7x3072xf32>
    %1516 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1517 = stablehlo.broadcast_in_dim %1516, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1518 = stablehlo.add %1517, %1515 : tensor<1x7x3072xf32>
    %1519 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %1520 = stablehlo.broadcast_in_dim %1519, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1521 = stablehlo.multiply %1520, %1518 : tensor<1x7x3072xf32>
    %1522 = stablehlo.multiply %1505, %1521 : tensor<1x7x3072xf32>
    %1523 = stablehlo.transpose %72, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %1524 = stablehlo.convert %1523 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1525 = stablehlo.dot_general %1522, %1524, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x3072xf32>, tensor<3072x768xf32>) -> tensor<1x7x768xf32>
    %1526 = stablehlo.convert %73 : (tensor<768xf16>) -> tensor<768xf32>
    %1527 = stablehlo.broadcast_in_dim %1526, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1528 = stablehlo.broadcast_in_dim %1527, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1529 = stablehlo.add %1525, %1528 : tensor<1x7x768xf32>
    %1530 = stablehlo.add %1464, %1529 : tensor<1x7x768xf32>
    %1531 = stablehlo.multiply %1530, %1530 : tensor<1x7x768xf32>
    %1532 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1533 = stablehlo.reduce(%1531 init: %1532) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1534 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1535 = stablehlo.broadcast_in_dim %1534, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1536 = stablehlo.divide %1533, %1535 : tensor<1x7xf32>
    %1537 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1538 = stablehlo.reduce(%1530 init: %1537) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1539 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1540 = stablehlo.broadcast_in_dim %1539, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1541 = stablehlo.divide %1538, %1540 : tensor<1x7xf32>
    %1542 = stablehlo.multiply %1541, %1541 : tensor<1x7xf32>
    %1543 = stablehlo.subtract %1536, %1542 : tensor<1x7xf32>
    %1544 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1545 = stablehlo.broadcast_in_dim %1544, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1546 = stablehlo.maximum %1545, %1543 : tensor<1x7xf32>
    %1547 = stablehlo.broadcast_in_dim %1541, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1548 = stablehlo.broadcast_in_dim %1546, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1549 = stablehlo.broadcast_in_dim %1547, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1550 = stablehlo.subtract %1530, %1549 : tensor<1x7x768xf32>
    %1551 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1552 = stablehlo.broadcast_in_dim %1551, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1553 = stablehlo.add %1548, %1552 : tensor<1x7x1xf32>
    %1554 = stablehlo.rsqrt %1553 : tensor<1x7x1xf32>
    %1555 = stablehlo.reshape %74 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1556 = stablehlo.convert %1555 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1557 = stablehlo.broadcast_in_dim %1554, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1558 = stablehlo.broadcast_in_dim %1556, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1559 = stablehlo.multiply %1557, %1558 : tensor<1x7x768xf32>
    %1560 = stablehlo.multiply %1550, %1559 : tensor<1x7x768xf32>
    %1561 = stablehlo.reshape %75 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1562 = stablehlo.convert %1561 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1563 = stablehlo.broadcast_in_dim %1562, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1564 = stablehlo.add %1560, %1563 : tensor<1x7x768xf32>
    %1565 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %1566 = stablehlo.broadcast_in_dim %1565, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %1567 = stablehlo.broadcast_in_dim %1566, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %1568 = stablehlo.broadcast_in_dim %1566, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %1569 = stablehlo.broadcast_in_dim %1567, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %1570 = stablehlo.broadcast_in_dim %1568, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %1571 = stablehlo.compare  GE, %1569, %1570,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %1572 = stablehlo.broadcast_in_dim %1571, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %1573 = stablehlo.transpose %76, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %1574 = stablehlo.convert %1573 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %1575 = stablehlo.dot_general %1564, %1574, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x2304xf32>) -> tensor<1x7x2304xf32>
    %1576 = stablehlo.convert %77 : (tensor<2304xf16>) -> tensor<2304xf32>
    %1577 = stablehlo.broadcast_in_dim %1576, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %1578 = stablehlo.broadcast_in_dim %1577, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<1x7x2304xf32>
    %1579 = stablehlo.add %1575, %1578 : tensor<1x7x2304xf32>
    %1580 = stablehlo.slice %1579 [0:1, 0:7, 0:768] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %1581 = stablehlo.slice %1579 [0:1, 0:7, 768:1536] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %1582 = stablehlo.slice %1579 [0:1, 0:7, 1536:2304] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %1583 = stablehlo.reshape %1580 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1584 = stablehlo.reshape %1581 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1585 = stablehlo.reshape %1582 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1586 = stablehlo.constant dense<0> : tensor<i32>
    %1587 = stablehlo.constant dense<0> : tensor<i32>
    %1588 = stablehlo.compare  LT, %1586, %1587,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1589 = stablehlo.constant dense<0> : tensor<i32>
    %1590 = stablehlo.constant dense<1024> : tensor<i32>
    %1591 = stablehlo.add %1589, %1590 : tensor<i32>
    %1592 = stablehlo.constant dense<0> : tensor<i32>
    %1593 = stablehlo.select %1588, %1591, %1592 : tensor<i1>, tensor<i32>
    %1594 = stablehlo.constant dense<0> : tensor<i32>
    %1595 = stablehlo.constant dense<0> : tensor<i32>
    %1596 = stablehlo.constant dense<0> : tensor<i32>
    %1597 = stablehlo.dynamic_slice %1572, %1594, %1595, %1593, %1596, sizes = [1, 1, 7, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x7x20xi1>
    %1598 = stablehlo.broadcast_in_dim %211, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %1599 = stablehlo.reshape %1598 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %1600 = stablehlo.broadcast_in_dim %1599, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x7x20xi32>
    %1601 = stablehlo.constant dense<0> : tensor<i32>
    %1602 = stablehlo.broadcast_in_dim %1601, dims = [] : (tensor<i32>) -> tensor<1x1x7x20xi32>
    %1603 = stablehlo.compare  NE, %1600, %1602,  SIGNED : (tensor<1x1x7x20xi32>, tensor<1x1x7x20xi32>) -> tensor<1x1x7x20xi1>
    %1604 = stablehlo.and %1603, %1597 : tensor<1x1x7x20xi1>
    %1605 = stablehlo.convert %1604 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %1606 = stablehlo.constant dense<0> : tensor<i32>
    %1607 = stablehlo.constant dense<0> : tensor<i32>
    %1608 = stablehlo.compare  LT, %1606, %1607,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1609 = stablehlo.constant dense<0> : tensor<i32>
    %1610 = stablehlo.constant dense<20> : tensor<i32>
    %1611 = stablehlo.add %1609, %1610 : tensor<i32>
    %1612 = stablehlo.constant dense<0> : tensor<i32>
    %1613 = stablehlo.select %1608, %1611, %1612 : tensor<i1>, tensor<i32>
    %1614 = stablehlo.constant dense<0> : tensor<i32>
    %1615 = stablehlo.constant dense<0> : tensor<i32>
    %1616 = stablehlo.constant dense<0> : tensor<i32>
    %1617 = stablehlo.dynamic_update_slice %180, %1584, %1614, %1613, %1615, %1616 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %1618 = stablehlo.constant dense<0> : tensor<i32>
    %1619 = stablehlo.constant dense<0> : tensor<i32>
    %1620 = stablehlo.compare  LT, %1618, %1619,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1621 = stablehlo.constant dense<0> : tensor<i32>
    %1622 = stablehlo.constant dense<20> : tensor<i32>
    %1623 = stablehlo.add %1621, %1622 : tensor<i32>
    %1624 = stablehlo.constant dense<0> : tensor<i32>
    %1625 = stablehlo.select %1620, %1623, %1624 : tensor<i1>, tensor<i32>
    %1626 = stablehlo.constant dense<0> : tensor<i32>
    %1627 = stablehlo.constant dense<0> : tensor<i32>
    %1628 = stablehlo.constant dense<0> : tensor<i32>
    %1629 = stablehlo.dynamic_update_slice %182, %1585, %1626, %1625, %1627, %1628 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %1630 = stablehlo.constant dense<0> : tensor<i32>
    %1631 = stablehlo.constant dense<7> : tensor<i32>
    %1632 = stablehlo.add %1630, %1631 : tensor<i32>
    %1633 = stablehlo.iota dim = 0 : tensor<20xi32>
    %1634 = stablehlo.constant dense<0> : tensor<i32>
    %1635 = stablehlo.constant dense<7> : tensor<i32>
    %1636 = stablehlo.add %1634, %1635 : tensor<i32>
    %1637 = stablehlo.broadcast_in_dim %1636, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %1638 = stablehlo.compare  LT, %1633, %1637,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %1639 = stablehlo.broadcast_in_dim %1638, dims = [3] : (tensor<20xi1>) -> tensor<1x1x7x20xi1>
    %1640 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1641 = stablehlo.broadcast_in_dim %1640, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1642 = stablehlo.compare  NE, %1605, %1641,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %1643 = stablehlo.and %1639, %1642 : tensor<1x1x7x20xi1>
    %1644 = stablehlo.convert %1643 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %1645 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1646 = stablehlo.broadcast_in_dim %1645, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1647 = stablehlo.compare  GT, %1644, %1646,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %1648 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1649 = stablehlo.broadcast_in_dim %1648, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1650 = stablehlo.convert %1649 : tensor<1x1x7x20xf32>
    %1651 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %1652 = stablehlo.broadcast_in_dim %1651, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1653 = stablehlo.select %1647, %1650, %1652 : tensor<1x1x7x20xi1>, tensor<1x1x7x20xf32>
    %1654 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %1655 = stablehlo.sqrt %1654 : tensor<f32>
    %1656 = stablehlo.convert %1655 : tensor<f32>
    %1657 = stablehlo.broadcast_in_dim %1656, dims = [] : (tensor<f32>) -> tensor<1x7x12x64xf32>
    %1658 = stablehlo.divide %1583, %1657 : tensor<1x7x12x64xf32>
    %1659 = stablehlo.dot_general %1658, %1617, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x7x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x7x20xf32>
    %1660 = stablehlo.broadcast_in_dim %1653, dims = [0, 1, 2, 3] : (tensor<1x1x7x20xf32>) -> tensor<1x12x7x20xf32>
    %1661 = stablehlo.add %1659, %1660 : tensor<1x12x7x20xf32>
    %1662 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1663 = stablehlo.reduce(%1661 init: %1662) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1664 = stablehlo.broadcast_in_dim %1663, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1665 = stablehlo.broadcast_in_dim %1664, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %1666 = stablehlo.subtract %1661, %1665 : tensor<1x12x7x20xf32>
    %1667 = stablehlo.exponential %1666 : tensor<1x12x7x20xf32>
    %1668 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1669 = stablehlo.reduce(%1667 init: %1668) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1670 = stablehlo.broadcast_in_dim %1669, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1671 = stablehlo.broadcast_in_dim %1670, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %1672 = stablehlo.divide %1667, %1671 : tensor<1x12x7x20xf32>
    %1673 = stablehlo.dot_general %1629, %1672, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x7x20xf32>) -> tensor<1x12x64x7xf32>
    %1674 = stablehlo.transpose %1673, dims = [0, 3, 1, 2] : (tensor<1x12x64x7xf32>) -> tensor<1x7x12x64xf32>
    %1675 = stablehlo.reshape %1674 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %1676 = stablehlo.transpose %78, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %1677 = stablehlo.convert %1676 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1678 = stablehlo.dot_general %1675, %1677, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x768xf32>) -> tensor<1x7x768xf32>
    %1679 = stablehlo.convert %79 : (tensor<768xf16>) -> tensor<768xf32>
    %1680 = stablehlo.broadcast_in_dim %1679, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1681 = stablehlo.broadcast_in_dim %1680, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1682 = stablehlo.add %1678, %1681 : tensor<1x7x768xf32>
    %1683 = stablehlo.add %1682, %1530 : tensor<1x7x768xf32>
    %1684 = stablehlo.multiply %1683, %1683 : tensor<1x7x768xf32>
    %1685 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1686 = stablehlo.reduce(%1684 init: %1685) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1687 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1688 = stablehlo.broadcast_in_dim %1687, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1689 = stablehlo.divide %1686, %1688 : tensor<1x7xf32>
    %1690 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1691 = stablehlo.reduce(%1683 init: %1690) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1692 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1693 = stablehlo.broadcast_in_dim %1692, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1694 = stablehlo.divide %1691, %1693 : tensor<1x7xf32>
    %1695 = stablehlo.multiply %1694, %1694 : tensor<1x7xf32>
    %1696 = stablehlo.subtract %1689, %1695 : tensor<1x7xf32>
    %1697 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1698 = stablehlo.broadcast_in_dim %1697, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1699 = stablehlo.maximum %1698, %1696 : tensor<1x7xf32>
    %1700 = stablehlo.broadcast_in_dim %1694, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1701 = stablehlo.broadcast_in_dim %1699, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1702 = stablehlo.broadcast_in_dim %1700, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1703 = stablehlo.subtract %1683, %1702 : tensor<1x7x768xf32>
    %1704 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1705 = stablehlo.broadcast_in_dim %1704, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1706 = stablehlo.add %1701, %1705 : tensor<1x7x1xf32>
    %1707 = stablehlo.rsqrt %1706 : tensor<1x7x1xf32>
    %1708 = stablehlo.reshape %80 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1709 = stablehlo.convert %1708 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1710 = stablehlo.broadcast_in_dim %1707, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1711 = stablehlo.broadcast_in_dim %1709, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1712 = stablehlo.multiply %1710, %1711 : tensor<1x7x768xf32>
    %1713 = stablehlo.multiply %1703, %1712 : tensor<1x7x768xf32>
    %1714 = stablehlo.reshape %81 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1715 = stablehlo.convert %1714 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1716 = stablehlo.broadcast_in_dim %1715, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1717 = stablehlo.add %1713, %1716 : tensor<1x7x768xf32>
    %1718 = stablehlo.transpose %82, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %1719 = stablehlo.convert %1718 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1720 = stablehlo.dot_general %1717, %1719, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x3072xf32>) -> tensor<1x7x3072xf32>
    %1721 = stablehlo.convert %83 : (tensor<3072xf16>) -> tensor<3072xf32>
    %1722 = stablehlo.broadcast_in_dim %1721, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %1723 = stablehlo.broadcast_in_dim %1722, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<1x7x3072xf32>
    %1724 = stablehlo.add %1720, %1723 : tensor<1x7x3072xf32>
    %1725 = stablehlo.multiply %1724, %1724 : tensor<1x7x3072xf32>
    %1726 = stablehlo.multiply %1724, %1725 : tensor<1x7x3072xf32>
    %1727 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %1728 = stablehlo.broadcast_in_dim %1727, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1729 = stablehlo.multiply %1728, %1726 : tensor<1x7x3072xf32>
    %1730 = stablehlo.add %1724, %1729 : tensor<1x7x3072xf32>
    %1731 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %1732 = stablehlo.broadcast_in_dim %1731, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1733 = stablehlo.multiply %1732, %1730 : tensor<1x7x3072xf32>
    %1734 = stablehlo.tanh %1733 : tensor<1x7x3072xf32>
    %1735 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1736 = stablehlo.broadcast_in_dim %1735, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1737 = stablehlo.add %1736, %1734 : tensor<1x7x3072xf32>
    %1738 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %1739 = stablehlo.broadcast_in_dim %1738, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1740 = stablehlo.multiply %1739, %1737 : tensor<1x7x3072xf32>
    %1741 = stablehlo.multiply %1724, %1740 : tensor<1x7x3072xf32>
    %1742 = stablehlo.transpose %84, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %1743 = stablehlo.convert %1742 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1744 = stablehlo.dot_general %1741, %1743, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x3072xf32>, tensor<3072x768xf32>) -> tensor<1x7x768xf32>
    %1745 = stablehlo.convert %85 : (tensor<768xf16>) -> tensor<768xf32>
    %1746 = stablehlo.broadcast_in_dim %1745, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1747 = stablehlo.broadcast_in_dim %1746, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1748 = stablehlo.add %1744, %1747 : tensor<1x7x768xf32>
    %1749 = stablehlo.add %1683, %1748 : tensor<1x7x768xf32>
    %1750 = stablehlo.multiply %1749, %1749 : tensor<1x7x768xf32>
    %1751 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1752 = stablehlo.reduce(%1750 init: %1751) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1753 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1754 = stablehlo.broadcast_in_dim %1753, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1755 = stablehlo.divide %1752, %1754 : tensor<1x7xf32>
    %1756 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1757 = stablehlo.reduce(%1749 init: %1756) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1758 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1759 = stablehlo.broadcast_in_dim %1758, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1760 = stablehlo.divide %1757, %1759 : tensor<1x7xf32>
    %1761 = stablehlo.multiply %1760, %1760 : tensor<1x7xf32>
    %1762 = stablehlo.subtract %1755, %1761 : tensor<1x7xf32>
    %1763 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1764 = stablehlo.broadcast_in_dim %1763, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1765 = stablehlo.maximum %1764, %1762 : tensor<1x7xf32>
    %1766 = stablehlo.broadcast_in_dim %1760, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1767 = stablehlo.broadcast_in_dim %1765, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1768 = stablehlo.broadcast_in_dim %1766, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1769 = stablehlo.subtract %1749, %1768 : tensor<1x7x768xf32>
    %1770 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1771 = stablehlo.broadcast_in_dim %1770, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1772 = stablehlo.add %1767, %1771 : tensor<1x7x1xf32>
    %1773 = stablehlo.rsqrt %1772 : tensor<1x7x1xf32>
    %1774 = stablehlo.reshape %86 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1775 = stablehlo.convert %1774 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1776 = stablehlo.broadcast_in_dim %1773, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1777 = stablehlo.broadcast_in_dim %1775, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1778 = stablehlo.multiply %1776, %1777 : tensor<1x7x768xf32>
    %1779 = stablehlo.multiply %1769, %1778 : tensor<1x7x768xf32>
    %1780 = stablehlo.reshape %87 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1781 = stablehlo.convert %1780 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1782 = stablehlo.broadcast_in_dim %1781, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1783 = stablehlo.add %1779, %1782 : tensor<1x7x768xf32>
    %1784 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %1785 = stablehlo.broadcast_in_dim %1784, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %1786 = stablehlo.broadcast_in_dim %1785, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %1787 = stablehlo.broadcast_in_dim %1785, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %1788 = stablehlo.broadcast_in_dim %1786, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %1789 = stablehlo.broadcast_in_dim %1787, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %1790 = stablehlo.compare  GE, %1788, %1789,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %1791 = stablehlo.broadcast_in_dim %1790, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %1792 = stablehlo.transpose %88, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %1793 = stablehlo.convert %1792 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %1794 = stablehlo.dot_general %1783, %1793, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x2304xf32>) -> tensor<1x7x2304xf32>
    %1795 = stablehlo.convert %89 : (tensor<2304xf16>) -> tensor<2304xf32>
    %1796 = stablehlo.broadcast_in_dim %1795, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %1797 = stablehlo.broadcast_in_dim %1796, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<1x7x2304xf32>
    %1798 = stablehlo.add %1794, %1797 : tensor<1x7x2304xf32>
    %1799 = stablehlo.slice %1798 [0:1, 0:7, 0:768] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %1800 = stablehlo.slice %1798 [0:1, 0:7, 768:1536] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %1801 = stablehlo.slice %1798 [0:1, 0:7, 1536:2304] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %1802 = stablehlo.reshape %1799 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1803 = stablehlo.reshape %1800 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1804 = stablehlo.reshape %1801 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1805 = stablehlo.constant dense<0> : tensor<i32>
    %1806 = stablehlo.constant dense<0> : tensor<i32>
    %1807 = stablehlo.compare  LT, %1805, %1806,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1808 = stablehlo.constant dense<0> : tensor<i32>
    %1809 = stablehlo.constant dense<1024> : tensor<i32>
    %1810 = stablehlo.add %1808, %1809 : tensor<i32>
    %1811 = stablehlo.constant dense<0> : tensor<i32>
    %1812 = stablehlo.select %1807, %1810, %1811 : tensor<i1>, tensor<i32>
    %1813 = stablehlo.constant dense<0> : tensor<i32>
    %1814 = stablehlo.constant dense<0> : tensor<i32>
    %1815 = stablehlo.constant dense<0> : tensor<i32>
    %1816 = stablehlo.dynamic_slice %1791, %1813, %1814, %1812, %1815, sizes = [1, 1, 7, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x7x20xi1>
    %1817 = stablehlo.broadcast_in_dim %211, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %1818 = stablehlo.reshape %1817 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %1819 = stablehlo.broadcast_in_dim %1818, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x7x20xi32>
    %1820 = stablehlo.constant dense<0> : tensor<i32>
    %1821 = stablehlo.broadcast_in_dim %1820, dims = [] : (tensor<i32>) -> tensor<1x1x7x20xi32>
    %1822 = stablehlo.compare  NE, %1819, %1821,  SIGNED : (tensor<1x1x7x20xi32>, tensor<1x1x7x20xi32>) -> tensor<1x1x7x20xi1>
    %1823 = stablehlo.and %1822, %1816 : tensor<1x1x7x20xi1>
    %1824 = stablehlo.convert %1823 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %1825 = stablehlo.constant dense<0> : tensor<i32>
    %1826 = stablehlo.constant dense<0> : tensor<i32>
    %1827 = stablehlo.compare  LT, %1825, %1826,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1828 = stablehlo.constant dense<0> : tensor<i32>
    %1829 = stablehlo.constant dense<20> : tensor<i32>
    %1830 = stablehlo.add %1828, %1829 : tensor<i32>
    %1831 = stablehlo.constant dense<0> : tensor<i32>
    %1832 = stablehlo.select %1827, %1830, %1831 : tensor<i1>, tensor<i32>
    %1833 = stablehlo.constant dense<0> : tensor<i32>
    %1834 = stablehlo.constant dense<0> : tensor<i32>
    %1835 = stablehlo.constant dense<0> : tensor<i32>
    %1836 = stablehlo.dynamic_update_slice %184, %1803, %1833, %1832, %1834, %1835 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %1837 = stablehlo.constant dense<0> : tensor<i32>
    %1838 = stablehlo.constant dense<0> : tensor<i32>
    %1839 = stablehlo.compare  LT, %1837, %1838,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1840 = stablehlo.constant dense<0> : tensor<i32>
    %1841 = stablehlo.constant dense<20> : tensor<i32>
    %1842 = stablehlo.add %1840, %1841 : tensor<i32>
    %1843 = stablehlo.constant dense<0> : tensor<i32>
    %1844 = stablehlo.select %1839, %1842, %1843 : tensor<i1>, tensor<i32>
    %1845 = stablehlo.constant dense<0> : tensor<i32>
    %1846 = stablehlo.constant dense<0> : tensor<i32>
    %1847 = stablehlo.constant dense<0> : tensor<i32>
    %1848 = stablehlo.dynamic_update_slice %186, %1804, %1845, %1844, %1846, %1847 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %1849 = stablehlo.constant dense<0> : tensor<i32>
    %1850 = stablehlo.constant dense<7> : tensor<i32>
    %1851 = stablehlo.add %1849, %1850 : tensor<i32>
    %1852 = stablehlo.iota dim = 0 : tensor<20xi32>
    %1853 = stablehlo.constant dense<0> : tensor<i32>
    %1854 = stablehlo.constant dense<7> : tensor<i32>
    %1855 = stablehlo.add %1853, %1854 : tensor<i32>
    %1856 = stablehlo.broadcast_in_dim %1855, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %1857 = stablehlo.compare  LT, %1852, %1856,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %1858 = stablehlo.broadcast_in_dim %1857, dims = [3] : (tensor<20xi1>) -> tensor<1x1x7x20xi1>
    %1859 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1860 = stablehlo.broadcast_in_dim %1859, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1861 = stablehlo.compare  NE, %1824, %1860,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %1862 = stablehlo.and %1858, %1861 : tensor<1x1x7x20xi1>
    %1863 = stablehlo.convert %1862 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %1864 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1865 = stablehlo.broadcast_in_dim %1864, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1866 = stablehlo.compare  GT, %1863, %1865,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %1867 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1868 = stablehlo.broadcast_in_dim %1867, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1869 = stablehlo.convert %1868 : tensor<1x1x7x20xf32>
    %1870 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %1871 = stablehlo.broadcast_in_dim %1870, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %1872 = stablehlo.select %1866, %1869, %1871 : tensor<1x1x7x20xi1>, tensor<1x1x7x20xf32>
    %1873 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %1874 = stablehlo.sqrt %1873 : tensor<f32>
    %1875 = stablehlo.convert %1874 : tensor<f32>
    %1876 = stablehlo.broadcast_in_dim %1875, dims = [] : (tensor<f32>) -> tensor<1x7x12x64xf32>
    %1877 = stablehlo.divide %1802, %1876 : tensor<1x7x12x64xf32>
    %1878 = stablehlo.dot_general %1877, %1836, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x7x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x7x20xf32>
    %1879 = stablehlo.broadcast_in_dim %1872, dims = [0, 1, 2, 3] : (tensor<1x1x7x20xf32>) -> tensor<1x12x7x20xf32>
    %1880 = stablehlo.add %1878, %1879 : tensor<1x12x7x20xf32>
    %1881 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1882 = stablehlo.reduce(%1880 init: %1881) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1883 = stablehlo.broadcast_in_dim %1882, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1884 = stablehlo.broadcast_in_dim %1883, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %1885 = stablehlo.subtract %1880, %1884 : tensor<1x12x7x20xf32>
    %1886 = stablehlo.exponential %1885 : tensor<1x12x7x20xf32>
    %1887 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1888 = stablehlo.reduce(%1886 init: %1887) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1889 = stablehlo.broadcast_in_dim %1888, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1890 = stablehlo.broadcast_in_dim %1889, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %1891 = stablehlo.divide %1886, %1890 : tensor<1x12x7x20xf32>
    %1892 = stablehlo.dot_general %1848, %1891, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x7x20xf32>) -> tensor<1x12x64x7xf32>
    %1893 = stablehlo.transpose %1892, dims = [0, 3, 1, 2] : (tensor<1x12x64x7xf32>) -> tensor<1x7x12x64xf32>
    %1894 = stablehlo.reshape %1893 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %1895 = stablehlo.transpose %90, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %1896 = stablehlo.convert %1895 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1897 = stablehlo.dot_general %1894, %1896, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x768xf32>) -> tensor<1x7x768xf32>
    %1898 = stablehlo.convert %91 : (tensor<768xf16>) -> tensor<768xf32>
    %1899 = stablehlo.broadcast_in_dim %1898, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1900 = stablehlo.broadcast_in_dim %1899, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1901 = stablehlo.add %1897, %1900 : tensor<1x7x768xf32>
    %1902 = stablehlo.add %1901, %1749 : tensor<1x7x768xf32>
    %1903 = stablehlo.multiply %1902, %1902 : tensor<1x7x768xf32>
    %1904 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1905 = stablehlo.reduce(%1903 init: %1904) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1906 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1907 = stablehlo.broadcast_in_dim %1906, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1908 = stablehlo.divide %1905, %1907 : tensor<1x7xf32>
    %1909 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1910 = stablehlo.reduce(%1902 init: %1909) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1911 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1912 = stablehlo.broadcast_in_dim %1911, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1913 = stablehlo.divide %1910, %1912 : tensor<1x7xf32>
    %1914 = stablehlo.multiply %1913, %1913 : tensor<1x7xf32>
    %1915 = stablehlo.subtract %1908, %1914 : tensor<1x7xf32>
    %1916 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1917 = stablehlo.broadcast_in_dim %1916, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1918 = stablehlo.maximum %1917, %1915 : tensor<1x7xf32>
    %1919 = stablehlo.broadcast_in_dim %1913, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1920 = stablehlo.broadcast_in_dim %1918, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1921 = stablehlo.broadcast_in_dim %1919, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1922 = stablehlo.subtract %1902, %1921 : tensor<1x7x768xf32>
    %1923 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1924 = stablehlo.broadcast_in_dim %1923, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1925 = stablehlo.add %1920, %1924 : tensor<1x7x1xf32>
    %1926 = stablehlo.rsqrt %1925 : tensor<1x7x1xf32>
    %1927 = stablehlo.reshape %92 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1928 = stablehlo.convert %1927 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1929 = stablehlo.broadcast_in_dim %1926, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1930 = stablehlo.broadcast_in_dim %1928, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1931 = stablehlo.multiply %1929, %1930 : tensor<1x7x768xf32>
    %1932 = stablehlo.multiply %1922, %1931 : tensor<1x7x768xf32>
    %1933 = stablehlo.reshape %93 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1934 = stablehlo.convert %1933 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1935 = stablehlo.broadcast_in_dim %1934, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1936 = stablehlo.add %1932, %1935 : tensor<1x7x768xf32>
    %1937 = stablehlo.transpose %94, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %1938 = stablehlo.convert %1937 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1939 = stablehlo.dot_general %1936, %1938, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x3072xf32>) -> tensor<1x7x3072xf32>
    %1940 = stablehlo.convert %95 : (tensor<3072xf16>) -> tensor<3072xf32>
    %1941 = stablehlo.broadcast_in_dim %1940, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %1942 = stablehlo.broadcast_in_dim %1941, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<1x7x3072xf32>
    %1943 = stablehlo.add %1939, %1942 : tensor<1x7x3072xf32>
    %1944 = stablehlo.multiply %1943, %1943 : tensor<1x7x3072xf32>
    %1945 = stablehlo.multiply %1943, %1944 : tensor<1x7x3072xf32>
    %1946 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %1947 = stablehlo.broadcast_in_dim %1946, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1948 = stablehlo.multiply %1947, %1945 : tensor<1x7x3072xf32>
    %1949 = stablehlo.add %1943, %1948 : tensor<1x7x3072xf32>
    %1950 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %1951 = stablehlo.broadcast_in_dim %1950, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1952 = stablehlo.multiply %1951, %1949 : tensor<1x7x3072xf32>
    %1953 = stablehlo.tanh %1952 : tensor<1x7x3072xf32>
    %1954 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1955 = stablehlo.broadcast_in_dim %1954, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1956 = stablehlo.add %1955, %1953 : tensor<1x7x3072xf32>
    %1957 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %1958 = stablehlo.broadcast_in_dim %1957, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1959 = stablehlo.multiply %1958, %1956 : tensor<1x7x3072xf32>
    %1960 = stablehlo.multiply %1943, %1959 : tensor<1x7x3072xf32>
    %1961 = stablehlo.transpose %96, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %1962 = stablehlo.convert %1961 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1963 = stablehlo.dot_general %1960, %1962, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x3072xf32>, tensor<3072x768xf32>) -> tensor<1x7x768xf32>
    %1964 = stablehlo.convert %97 : (tensor<768xf16>) -> tensor<768xf32>
    %1965 = stablehlo.broadcast_in_dim %1964, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1966 = stablehlo.broadcast_in_dim %1965, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1967 = stablehlo.add %1963, %1966 : tensor<1x7x768xf32>
    %1968 = stablehlo.add %1902, %1967 : tensor<1x7x768xf32>
    %1969 = stablehlo.multiply %1968, %1968 : tensor<1x7x768xf32>
    %1970 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1971 = stablehlo.reduce(%1969 init: %1970) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1972 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1973 = stablehlo.broadcast_in_dim %1972, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1974 = stablehlo.divide %1971, %1973 : tensor<1x7xf32>
    %1975 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1976 = stablehlo.reduce(%1968 init: %1975) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1977 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1978 = stablehlo.broadcast_in_dim %1977, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1979 = stablehlo.divide %1976, %1978 : tensor<1x7xf32>
    %1980 = stablehlo.multiply %1979, %1979 : tensor<1x7xf32>
    %1981 = stablehlo.subtract %1974, %1980 : tensor<1x7xf32>
    %1982 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1983 = stablehlo.broadcast_in_dim %1982, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %1984 = stablehlo.maximum %1983, %1981 : tensor<1x7xf32>
    %1985 = stablehlo.broadcast_in_dim %1979, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1986 = stablehlo.broadcast_in_dim %1984, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1987 = stablehlo.broadcast_in_dim %1985, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1988 = stablehlo.subtract %1968, %1987 : tensor<1x7x768xf32>
    %1989 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1990 = stablehlo.broadcast_in_dim %1989, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1991 = stablehlo.add %1986, %1990 : tensor<1x7x1xf32>
    %1992 = stablehlo.rsqrt %1991 : tensor<1x7x1xf32>
    %1993 = stablehlo.reshape %98 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1994 = stablehlo.convert %1993 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1995 = stablehlo.broadcast_in_dim %1992, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1996 = stablehlo.broadcast_in_dim %1994, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1997 = stablehlo.multiply %1995, %1996 : tensor<1x7x768xf32>
    %1998 = stablehlo.multiply %1988, %1997 : tensor<1x7x768xf32>
    %1999 = stablehlo.reshape %99 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2000 = stablehlo.convert %1999 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2001 = stablehlo.broadcast_in_dim %2000, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2002 = stablehlo.add %1998, %2001 : tensor<1x7x768xf32>
    %2003 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %2004 = stablehlo.broadcast_in_dim %2003, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %2005 = stablehlo.broadcast_in_dim %2004, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %2006 = stablehlo.broadcast_in_dim %2004, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %2007 = stablehlo.broadcast_in_dim %2005, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %2008 = stablehlo.broadcast_in_dim %2006, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %2009 = stablehlo.compare  GE, %2007, %2008,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %2010 = stablehlo.broadcast_in_dim %2009, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %2011 = stablehlo.transpose %100, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %2012 = stablehlo.convert %2011 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %2013 = stablehlo.dot_general %2002, %2012, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x2304xf32>) -> tensor<1x7x2304xf32>
    %2014 = stablehlo.convert %101 : (tensor<2304xf16>) -> tensor<2304xf32>
    %2015 = stablehlo.broadcast_in_dim %2014, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %2016 = stablehlo.broadcast_in_dim %2015, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<1x7x2304xf32>
    %2017 = stablehlo.add %2013, %2016 : tensor<1x7x2304xf32>
    %2018 = stablehlo.slice %2017 [0:1, 0:7, 0:768] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %2019 = stablehlo.slice %2017 [0:1, 0:7, 768:1536] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %2020 = stablehlo.slice %2017 [0:1, 0:7, 1536:2304] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %2021 = stablehlo.reshape %2018 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2022 = stablehlo.reshape %2019 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2023 = stablehlo.reshape %2020 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2024 = stablehlo.constant dense<0> : tensor<i32>
    %2025 = stablehlo.constant dense<0> : tensor<i32>
    %2026 = stablehlo.compare  LT, %2024, %2025,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2027 = stablehlo.constant dense<0> : tensor<i32>
    %2028 = stablehlo.constant dense<1024> : tensor<i32>
    %2029 = stablehlo.add %2027, %2028 : tensor<i32>
    %2030 = stablehlo.constant dense<0> : tensor<i32>
    %2031 = stablehlo.select %2026, %2029, %2030 : tensor<i1>, tensor<i32>
    %2032 = stablehlo.constant dense<0> : tensor<i32>
    %2033 = stablehlo.constant dense<0> : tensor<i32>
    %2034 = stablehlo.constant dense<0> : tensor<i32>
    %2035 = stablehlo.dynamic_slice %2010, %2032, %2033, %2031, %2034, sizes = [1, 1, 7, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x7x20xi1>
    %2036 = stablehlo.broadcast_in_dim %211, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %2037 = stablehlo.reshape %2036 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %2038 = stablehlo.broadcast_in_dim %2037, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x7x20xi32>
    %2039 = stablehlo.constant dense<0> : tensor<i32>
    %2040 = stablehlo.broadcast_in_dim %2039, dims = [] : (tensor<i32>) -> tensor<1x1x7x20xi32>
    %2041 = stablehlo.compare  NE, %2038, %2040,  SIGNED : (tensor<1x1x7x20xi32>, tensor<1x1x7x20xi32>) -> tensor<1x1x7x20xi1>
    %2042 = stablehlo.and %2041, %2035 : tensor<1x1x7x20xi1>
    %2043 = stablehlo.convert %2042 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %2044 = stablehlo.constant dense<0> : tensor<i32>
    %2045 = stablehlo.constant dense<0> : tensor<i32>
    %2046 = stablehlo.compare  LT, %2044, %2045,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2047 = stablehlo.constant dense<0> : tensor<i32>
    %2048 = stablehlo.constant dense<20> : tensor<i32>
    %2049 = stablehlo.add %2047, %2048 : tensor<i32>
    %2050 = stablehlo.constant dense<0> : tensor<i32>
    %2051 = stablehlo.select %2046, %2049, %2050 : tensor<i1>, tensor<i32>
    %2052 = stablehlo.constant dense<0> : tensor<i32>
    %2053 = stablehlo.constant dense<0> : tensor<i32>
    %2054 = stablehlo.constant dense<0> : tensor<i32>
    %2055 = stablehlo.dynamic_update_slice %188, %2022, %2052, %2051, %2053, %2054 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %2056 = stablehlo.constant dense<0> : tensor<i32>
    %2057 = stablehlo.constant dense<0> : tensor<i32>
    %2058 = stablehlo.compare  LT, %2056, %2057,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2059 = stablehlo.constant dense<0> : tensor<i32>
    %2060 = stablehlo.constant dense<20> : tensor<i32>
    %2061 = stablehlo.add %2059, %2060 : tensor<i32>
    %2062 = stablehlo.constant dense<0> : tensor<i32>
    %2063 = stablehlo.select %2058, %2061, %2062 : tensor<i1>, tensor<i32>
    %2064 = stablehlo.constant dense<0> : tensor<i32>
    %2065 = stablehlo.constant dense<0> : tensor<i32>
    %2066 = stablehlo.constant dense<0> : tensor<i32>
    %2067 = stablehlo.dynamic_update_slice %190, %2023, %2064, %2063, %2065, %2066 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %2068 = stablehlo.constant dense<0> : tensor<i32>
    %2069 = stablehlo.constant dense<7> : tensor<i32>
    %2070 = stablehlo.add %2068, %2069 : tensor<i32>
    %2071 = stablehlo.iota dim = 0 : tensor<20xi32>
    %2072 = stablehlo.constant dense<0> : tensor<i32>
    %2073 = stablehlo.constant dense<7> : tensor<i32>
    %2074 = stablehlo.add %2072, %2073 : tensor<i32>
    %2075 = stablehlo.broadcast_in_dim %2074, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %2076 = stablehlo.compare  LT, %2071, %2075,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %2077 = stablehlo.broadcast_in_dim %2076, dims = [3] : (tensor<20xi1>) -> tensor<1x1x7x20xi1>
    %2078 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2079 = stablehlo.broadcast_in_dim %2078, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2080 = stablehlo.compare  NE, %2043, %2079,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %2081 = stablehlo.and %2077, %2080 : tensor<1x1x7x20xi1>
    %2082 = stablehlo.convert %2081 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %2083 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2084 = stablehlo.broadcast_in_dim %2083, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2085 = stablehlo.compare  GT, %2082, %2084,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %2086 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2087 = stablehlo.broadcast_in_dim %2086, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2088 = stablehlo.convert %2087 : tensor<1x1x7x20xf32>
    %2089 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %2090 = stablehlo.broadcast_in_dim %2089, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2091 = stablehlo.select %2085, %2088, %2090 : tensor<1x1x7x20xi1>, tensor<1x1x7x20xf32>
    %2092 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %2093 = stablehlo.sqrt %2092 : tensor<f32>
    %2094 = stablehlo.convert %2093 : tensor<f32>
    %2095 = stablehlo.broadcast_in_dim %2094, dims = [] : (tensor<f32>) -> tensor<1x7x12x64xf32>
    %2096 = stablehlo.divide %2021, %2095 : tensor<1x7x12x64xf32>
    %2097 = stablehlo.dot_general %2096, %2055, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x7x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x7x20xf32>
    %2098 = stablehlo.broadcast_in_dim %2091, dims = [0, 1, 2, 3] : (tensor<1x1x7x20xf32>) -> tensor<1x12x7x20xf32>
    %2099 = stablehlo.add %2097, %2098 : tensor<1x12x7x20xf32>
    %2100 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2101 = stablehlo.reduce(%2099 init: %2100) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2102 = stablehlo.broadcast_in_dim %2101, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2103 = stablehlo.broadcast_in_dim %2102, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %2104 = stablehlo.subtract %2099, %2103 : tensor<1x12x7x20xf32>
    %2105 = stablehlo.exponential %2104 : tensor<1x12x7x20xf32>
    %2106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2107 = stablehlo.reduce(%2105 init: %2106) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2108 = stablehlo.broadcast_in_dim %2107, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2109 = stablehlo.broadcast_in_dim %2108, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %2110 = stablehlo.divide %2105, %2109 : tensor<1x12x7x20xf32>
    %2111 = stablehlo.dot_general %2067, %2110, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x7x20xf32>) -> tensor<1x12x64x7xf32>
    %2112 = stablehlo.transpose %2111, dims = [0, 3, 1, 2] : (tensor<1x12x64x7xf32>) -> tensor<1x7x12x64xf32>
    %2113 = stablehlo.reshape %2112 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %2114 = stablehlo.transpose %102, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %2115 = stablehlo.convert %2114 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2116 = stablehlo.dot_general %2113, %2115, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x768xf32>) -> tensor<1x7x768xf32>
    %2117 = stablehlo.convert %103 : (tensor<768xf16>) -> tensor<768xf32>
    %2118 = stablehlo.broadcast_in_dim %2117, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2119 = stablehlo.broadcast_in_dim %2118, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2120 = stablehlo.add %2116, %2119 : tensor<1x7x768xf32>
    %2121 = stablehlo.add %2120, %1968 : tensor<1x7x768xf32>
    %2122 = stablehlo.multiply %2121, %2121 : tensor<1x7x768xf32>
    %2123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2124 = stablehlo.reduce(%2122 init: %2123) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2125 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2126 = stablehlo.broadcast_in_dim %2125, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2127 = stablehlo.divide %2124, %2126 : tensor<1x7xf32>
    %2128 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2129 = stablehlo.reduce(%2121 init: %2128) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2130 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2131 = stablehlo.broadcast_in_dim %2130, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2132 = stablehlo.divide %2129, %2131 : tensor<1x7xf32>
    %2133 = stablehlo.multiply %2132, %2132 : tensor<1x7xf32>
    %2134 = stablehlo.subtract %2127, %2133 : tensor<1x7xf32>
    %2135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2136 = stablehlo.broadcast_in_dim %2135, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2137 = stablehlo.maximum %2136, %2134 : tensor<1x7xf32>
    %2138 = stablehlo.broadcast_in_dim %2132, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2139 = stablehlo.broadcast_in_dim %2137, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2140 = stablehlo.broadcast_in_dim %2138, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2141 = stablehlo.subtract %2121, %2140 : tensor<1x7x768xf32>
    %2142 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2143 = stablehlo.broadcast_in_dim %2142, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2144 = stablehlo.add %2139, %2143 : tensor<1x7x1xf32>
    %2145 = stablehlo.rsqrt %2144 : tensor<1x7x1xf32>
    %2146 = stablehlo.reshape %104 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2147 = stablehlo.convert %2146 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2148 = stablehlo.broadcast_in_dim %2145, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2149 = stablehlo.broadcast_in_dim %2147, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2150 = stablehlo.multiply %2148, %2149 : tensor<1x7x768xf32>
    %2151 = stablehlo.multiply %2141, %2150 : tensor<1x7x768xf32>
    %2152 = stablehlo.reshape %105 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2153 = stablehlo.convert %2152 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2154 = stablehlo.broadcast_in_dim %2153, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2155 = stablehlo.add %2151, %2154 : tensor<1x7x768xf32>
    %2156 = stablehlo.transpose %106, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %2157 = stablehlo.convert %2156 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %2158 = stablehlo.dot_general %2155, %2157, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x3072xf32>) -> tensor<1x7x3072xf32>
    %2159 = stablehlo.convert %107 : (tensor<3072xf16>) -> tensor<3072xf32>
    %2160 = stablehlo.broadcast_in_dim %2159, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %2161 = stablehlo.broadcast_in_dim %2160, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<1x7x3072xf32>
    %2162 = stablehlo.add %2158, %2161 : tensor<1x7x3072xf32>
    %2163 = stablehlo.multiply %2162, %2162 : tensor<1x7x3072xf32>
    %2164 = stablehlo.multiply %2162, %2163 : tensor<1x7x3072xf32>
    %2165 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %2166 = stablehlo.broadcast_in_dim %2165, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2167 = stablehlo.multiply %2166, %2164 : tensor<1x7x3072xf32>
    %2168 = stablehlo.add %2162, %2167 : tensor<1x7x3072xf32>
    %2169 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %2170 = stablehlo.broadcast_in_dim %2169, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2171 = stablehlo.multiply %2170, %2168 : tensor<1x7x3072xf32>
    %2172 = stablehlo.tanh %2171 : tensor<1x7x3072xf32>
    %2173 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2174 = stablehlo.broadcast_in_dim %2173, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2175 = stablehlo.add %2174, %2172 : tensor<1x7x3072xf32>
    %2176 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %2177 = stablehlo.broadcast_in_dim %2176, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2178 = stablehlo.multiply %2177, %2175 : tensor<1x7x3072xf32>
    %2179 = stablehlo.multiply %2162, %2178 : tensor<1x7x3072xf32>
    %2180 = stablehlo.transpose %108, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %2181 = stablehlo.convert %2180 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %2182 = stablehlo.dot_general %2179, %2181, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x3072xf32>, tensor<3072x768xf32>) -> tensor<1x7x768xf32>
    %2183 = stablehlo.convert %109 : (tensor<768xf16>) -> tensor<768xf32>
    %2184 = stablehlo.broadcast_in_dim %2183, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2185 = stablehlo.broadcast_in_dim %2184, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2186 = stablehlo.add %2182, %2185 : tensor<1x7x768xf32>
    %2187 = stablehlo.add %2121, %2186 : tensor<1x7x768xf32>
    %2188 = stablehlo.multiply %2187, %2187 : tensor<1x7x768xf32>
    %2189 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2190 = stablehlo.reduce(%2188 init: %2189) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2191 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2192 = stablehlo.broadcast_in_dim %2191, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2193 = stablehlo.divide %2190, %2192 : tensor<1x7xf32>
    %2194 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2195 = stablehlo.reduce(%2187 init: %2194) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2196 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2197 = stablehlo.broadcast_in_dim %2196, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2198 = stablehlo.divide %2195, %2197 : tensor<1x7xf32>
    %2199 = stablehlo.multiply %2198, %2198 : tensor<1x7xf32>
    %2200 = stablehlo.subtract %2193, %2199 : tensor<1x7xf32>
    %2201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2202 = stablehlo.broadcast_in_dim %2201, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2203 = stablehlo.maximum %2202, %2200 : tensor<1x7xf32>
    %2204 = stablehlo.broadcast_in_dim %2198, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2205 = stablehlo.broadcast_in_dim %2203, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2206 = stablehlo.broadcast_in_dim %2204, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2207 = stablehlo.subtract %2187, %2206 : tensor<1x7x768xf32>
    %2208 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2209 = stablehlo.broadcast_in_dim %2208, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2210 = stablehlo.add %2205, %2209 : tensor<1x7x1xf32>
    %2211 = stablehlo.rsqrt %2210 : tensor<1x7x1xf32>
    %2212 = stablehlo.reshape %110 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2213 = stablehlo.convert %2212 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2214 = stablehlo.broadcast_in_dim %2211, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2215 = stablehlo.broadcast_in_dim %2213, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2216 = stablehlo.multiply %2214, %2215 : tensor<1x7x768xf32>
    %2217 = stablehlo.multiply %2207, %2216 : tensor<1x7x768xf32>
    %2218 = stablehlo.reshape %111 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2219 = stablehlo.convert %2218 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2220 = stablehlo.broadcast_in_dim %2219, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2221 = stablehlo.add %2217, %2220 : tensor<1x7x768xf32>
    %2222 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %2223 = stablehlo.broadcast_in_dim %2222, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %2224 = stablehlo.broadcast_in_dim %2223, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %2225 = stablehlo.broadcast_in_dim %2223, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %2226 = stablehlo.broadcast_in_dim %2224, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %2227 = stablehlo.broadcast_in_dim %2225, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %2228 = stablehlo.compare  GE, %2226, %2227,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %2229 = stablehlo.broadcast_in_dim %2228, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %2230 = stablehlo.transpose %112, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %2231 = stablehlo.convert %2230 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %2232 = stablehlo.dot_general %2221, %2231, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x2304xf32>) -> tensor<1x7x2304xf32>
    %2233 = stablehlo.convert %113 : (tensor<2304xf16>) -> tensor<2304xf32>
    %2234 = stablehlo.broadcast_in_dim %2233, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %2235 = stablehlo.broadcast_in_dim %2234, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<1x7x2304xf32>
    %2236 = stablehlo.add %2232, %2235 : tensor<1x7x2304xf32>
    %2237 = stablehlo.slice %2236 [0:1, 0:7, 0:768] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %2238 = stablehlo.slice %2236 [0:1, 0:7, 768:1536] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %2239 = stablehlo.slice %2236 [0:1, 0:7, 1536:2304] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %2240 = stablehlo.reshape %2237 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2241 = stablehlo.reshape %2238 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2242 = stablehlo.reshape %2239 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2243 = stablehlo.constant dense<0> : tensor<i32>
    %2244 = stablehlo.constant dense<0> : tensor<i32>
    %2245 = stablehlo.compare  LT, %2243, %2244,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2246 = stablehlo.constant dense<0> : tensor<i32>
    %2247 = stablehlo.constant dense<1024> : tensor<i32>
    %2248 = stablehlo.add %2246, %2247 : tensor<i32>
    %2249 = stablehlo.constant dense<0> : tensor<i32>
    %2250 = stablehlo.select %2245, %2248, %2249 : tensor<i1>, tensor<i32>
    %2251 = stablehlo.constant dense<0> : tensor<i32>
    %2252 = stablehlo.constant dense<0> : tensor<i32>
    %2253 = stablehlo.constant dense<0> : tensor<i32>
    %2254 = stablehlo.dynamic_slice %2229, %2251, %2252, %2250, %2253, sizes = [1, 1, 7, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x7x20xi1>
    %2255 = stablehlo.broadcast_in_dim %211, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %2256 = stablehlo.reshape %2255 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %2257 = stablehlo.broadcast_in_dim %2256, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x7x20xi32>
    %2258 = stablehlo.constant dense<0> : tensor<i32>
    %2259 = stablehlo.broadcast_in_dim %2258, dims = [] : (tensor<i32>) -> tensor<1x1x7x20xi32>
    %2260 = stablehlo.compare  NE, %2257, %2259,  SIGNED : (tensor<1x1x7x20xi32>, tensor<1x1x7x20xi32>) -> tensor<1x1x7x20xi1>
    %2261 = stablehlo.and %2260, %2254 : tensor<1x1x7x20xi1>
    %2262 = stablehlo.convert %2261 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %2263 = stablehlo.constant dense<0> : tensor<i32>
    %2264 = stablehlo.constant dense<0> : tensor<i32>
    %2265 = stablehlo.compare  LT, %2263, %2264,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2266 = stablehlo.constant dense<0> : tensor<i32>
    %2267 = stablehlo.constant dense<20> : tensor<i32>
    %2268 = stablehlo.add %2266, %2267 : tensor<i32>
    %2269 = stablehlo.constant dense<0> : tensor<i32>
    %2270 = stablehlo.select %2265, %2268, %2269 : tensor<i1>, tensor<i32>
    %2271 = stablehlo.constant dense<0> : tensor<i32>
    %2272 = stablehlo.constant dense<0> : tensor<i32>
    %2273 = stablehlo.constant dense<0> : tensor<i32>
    %2274 = stablehlo.dynamic_update_slice %192, %2241, %2271, %2270, %2272, %2273 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %2275 = stablehlo.constant dense<0> : tensor<i32>
    %2276 = stablehlo.constant dense<0> : tensor<i32>
    %2277 = stablehlo.compare  LT, %2275, %2276,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2278 = stablehlo.constant dense<0> : tensor<i32>
    %2279 = stablehlo.constant dense<20> : tensor<i32>
    %2280 = stablehlo.add %2278, %2279 : tensor<i32>
    %2281 = stablehlo.constant dense<0> : tensor<i32>
    %2282 = stablehlo.select %2277, %2280, %2281 : tensor<i1>, tensor<i32>
    %2283 = stablehlo.constant dense<0> : tensor<i32>
    %2284 = stablehlo.constant dense<0> : tensor<i32>
    %2285 = stablehlo.constant dense<0> : tensor<i32>
    %2286 = stablehlo.dynamic_update_slice %194, %2242, %2283, %2282, %2284, %2285 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %2287 = stablehlo.constant dense<0> : tensor<i32>
    %2288 = stablehlo.constant dense<7> : tensor<i32>
    %2289 = stablehlo.add %2287, %2288 : tensor<i32>
    %2290 = stablehlo.iota dim = 0 : tensor<20xi32>
    %2291 = stablehlo.constant dense<0> : tensor<i32>
    %2292 = stablehlo.constant dense<7> : tensor<i32>
    %2293 = stablehlo.add %2291, %2292 : tensor<i32>
    %2294 = stablehlo.broadcast_in_dim %2293, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %2295 = stablehlo.compare  LT, %2290, %2294,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %2296 = stablehlo.broadcast_in_dim %2295, dims = [3] : (tensor<20xi1>) -> tensor<1x1x7x20xi1>
    %2297 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2298 = stablehlo.broadcast_in_dim %2297, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2299 = stablehlo.compare  NE, %2262, %2298,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %2300 = stablehlo.and %2296, %2299 : tensor<1x1x7x20xi1>
    %2301 = stablehlo.convert %2300 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %2302 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2303 = stablehlo.broadcast_in_dim %2302, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2304 = stablehlo.compare  GT, %2301, %2303,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %2305 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2306 = stablehlo.broadcast_in_dim %2305, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2307 = stablehlo.convert %2306 : tensor<1x1x7x20xf32>
    %2308 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %2309 = stablehlo.broadcast_in_dim %2308, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2310 = stablehlo.select %2304, %2307, %2309 : tensor<1x1x7x20xi1>, tensor<1x1x7x20xf32>
    %2311 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %2312 = stablehlo.sqrt %2311 : tensor<f32>
    %2313 = stablehlo.convert %2312 : tensor<f32>
    %2314 = stablehlo.broadcast_in_dim %2313, dims = [] : (tensor<f32>) -> tensor<1x7x12x64xf32>
    %2315 = stablehlo.divide %2240, %2314 : tensor<1x7x12x64xf32>
    %2316 = stablehlo.dot_general %2315, %2274, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x7x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x7x20xf32>
    %2317 = stablehlo.broadcast_in_dim %2310, dims = [0, 1, 2, 3] : (tensor<1x1x7x20xf32>) -> tensor<1x12x7x20xf32>
    %2318 = stablehlo.add %2316, %2317 : tensor<1x12x7x20xf32>
    %2319 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2320 = stablehlo.reduce(%2318 init: %2319) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2321 = stablehlo.broadcast_in_dim %2320, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2322 = stablehlo.broadcast_in_dim %2321, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %2323 = stablehlo.subtract %2318, %2322 : tensor<1x12x7x20xf32>
    %2324 = stablehlo.exponential %2323 : tensor<1x12x7x20xf32>
    %2325 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2326 = stablehlo.reduce(%2324 init: %2325) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2327 = stablehlo.broadcast_in_dim %2326, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2328 = stablehlo.broadcast_in_dim %2327, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %2329 = stablehlo.divide %2324, %2328 : tensor<1x12x7x20xf32>
    %2330 = stablehlo.dot_general %2286, %2329, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x7x20xf32>) -> tensor<1x12x64x7xf32>
    %2331 = stablehlo.transpose %2330, dims = [0, 3, 1, 2] : (tensor<1x12x64x7xf32>) -> tensor<1x7x12x64xf32>
    %2332 = stablehlo.reshape %2331 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %2333 = stablehlo.transpose %114, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %2334 = stablehlo.convert %2333 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2335 = stablehlo.dot_general %2332, %2334, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x768xf32>) -> tensor<1x7x768xf32>
    %2336 = stablehlo.convert %115 : (tensor<768xf16>) -> tensor<768xf32>
    %2337 = stablehlo.broadcast_in_dim %2336, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2338 = stablehlo.broadcast_in_dim %2337, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2339 = stablehlo.add %2335, %2338 : tensor<1x7x768xf32>
    %2340 = stablehlo.add %2339, %2187 : tensor<1x7x768xf32>
    %2341 = stablehlo.multiply %2340, %2340 : tensor<1x7x768xf32>
    %2342 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2343 = stablehlo.reduce(%2341 init: %2342) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2344 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2345 = stablehlo.broadcast_in_dim %2344, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2346 = stablehlo.divide %2343, %2345 : tensor<1x7xf32>
    %2347 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2348 = stablehlo.reduce(%2340 init: %2347) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2349 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2350 = stablehlo.broadcast_in_dim %2349, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2351 = stablehlo.divide %2348, %2350 : tensor<1x7xf32>
    %2352 = stablehlo.multiply %2351, %2351 : tensor<1x7xf32>
    %2353 = stablehlo.subtract %2346, %2352 : tensor<1x7xf32>
    %2354 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2355 = stablehlo.broadcast_in_dim %2354, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2356 = stablehlo.maximum %2355, %2353 : tensor<1x7xf32>
    %2357 = stablehlo.broadcast_in_dim %2351, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2358 = stablehlo.broadcast_in_dim %2356, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2359 = stablehlo.broadcast_in_dim %2357, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2360 = stablehlo.subtract %2340, %2359 : tensor<1x7x768xf32>
    %2361 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2362 = stablehlo.broadcast_in_dim %2361, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2363 = stablehlo.add %2358, %2362 : tensor<1x7x1xf32>
    %2364 = stablehlo.rsqrt %2363 : tensor<1x7x1xf32>
    %2365 = stablehlo.reshape %116 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2366 = stablehlo.convert %2365 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2367 = stablehlo.broadcast_in_dim %2364, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2368 = stablehlo.broadcast_in_dim %2366, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2369 = stablehlo.multiply %2367, %2368 : tensor<1x7x768xf32>
    %2370 = stablehlo.multiply %2360, %2369 : tensor<1x7x768xf32>
    %2371 = stablehlo.reshape %117 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2372 = stablehlo.convert %2371 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2373 = stablehlo.broadcast_in_dim %2372, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2374 = stablehlo.add %2370, %2373 : tensor<1x7x768xf32>
    %2375 = stablehlo.transpose %118, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %2376 = stablehlo.convert %2375 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %2377 = stablehlo.dot_general %2374, %2376, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x3072xf32>) -> tensor<1x7x3072xf32>
    %2378 = stablehlo.convert %119 : (tensor<3072xf16>) -> tensor<3072xf32>
    %2379 = stablehlo.broadcast_in_dim %2378, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %2380 = stablehlo.broadcast_in_dim %2379, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<1x7x3072xf32>
    %2381 = stablehlo.add %2377, %2380 : tensor<1x7x3072xf32>
    %2382 = stablehlo.multiply %2381, %2381 : tensor<1x7x3072xf32>
    %2383 = stablehlo.multiply %2381, %2382 : tensor<1x7x3072xf32>
    %2384 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %2385 = stablehlo.broadcast_in_dim %2384, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2386 = stablehlo.multiply %2385, %2383 : tensor<1x7x3072xf32>
    %2387 = stablehlo.add %2381, %2386 : tensor<1x7x3072xf32>
    %2388 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %2389 = stablehlo.broadcast_in_dim %2388, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2390 = stablehlo.multiply %2389, %2387 : tensor<1x7x3072xf32>
    %2391 = stablehlo.tanh %2390 : tensor<1x7x3072xf32>
    %2392 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2393 = stablehlo.broadcast_in_dim %2392, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2394 = stablehlo.add %2393, %2391 : tensor<1x7x3072xf32>
    %2395 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %2396 = stablehlo.broadcast_in_dim %2395, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2397 = stablehlo.multiply %2396, %2394 : tensor<1x7x3072xf32>
    %2398 = stablehlo.multiply %2381, %2397 : tensor<1x7x3072xf32>
    %2399 = stablehlo.transpose %120, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %2400 = stablehlo.convert %2399 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %2401 = stablehlo.dot_general %2398, %2400, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x3072xf32>, tensor<3072x768xf32>) -> tensor<1x7x768xf32>
    %2402 = stablehlo.convert %121 : (tensor<768xf16>) -> tensor<768xf32>
    %2403 = stablehlo.broadcast_in_dim %2402, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2404 = stablehlo.broadcast_in_dim %2403, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2405 = stablehlo.add %2401, %2404 : tensor<1x7x768xf32>
    %2406 = stablehlo.add %2340, %2405 : tensor<1x7x768xf32>
    %2407 = stablehlo.multiply %2406, %2406 : tensor<1x7x768xf32>
    %2408 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2409 = stablehlo.reduce(%2407 init: %2408) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2410 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2411 = stablehlo.broadcast_in_dim %2410, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2412 = stablehlo.divide %2409, %2411 : tensor<1x7xf32>
    %2413 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2414 = stablehlo.reduce(%2406 init: %2413) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2415 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2416 = stablehlo.broadcast_in_dim %2415, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2417 = stablehlo.divide %2414, %2416 : tensor<1x7xf32>
    %2418 = stablehlo.multiply %2417, %2417 : tensor<1x7xf32>
    %2419 = stablehlo.subtract %2412, %2418 : tensor<1x7xf32>
    %2420 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2421 = stablehlo.broadcast_in_dim %2420, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2422 = stablehlo.maximum %2421, %2419 : tensor<1x7xf32>
    %2423 = stablehlo.broadcast_in_dim %2417, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2424 = stablehlo.broadcast_in_dim %2422, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2425 = stablehlo.broadcast_in_dim %2423, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2426 = stablehlo.subtract %2406, %2425 : tensor<1x7x768xf32>
    %2427 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2428 = stablehlo.broadcast_in_dim %2427, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2429 = stablehlo.add %2424, %2428 : tensor<1x7x1xf32>
    %2430 = stablehlo.rsqrt %2429 : tensor<1x7x1xf32>
    %2431 = stablehlo.reshape %122 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2432 = stablehlo.convert %2431 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2433 = stablehlo.broadcast_in_dim %2430, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2434 = stablehlo.broadcast_in_dim %2432, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2435 = stablehlo.multiply %2433, %2434 : tensor<1x7x768xf32>
    %2436 = stablehlo.multiply %2426, %2435 : tensor<1x7x768xf32>
    %2437 = stablehlo.reshape %123 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2438 = stablehlo.convert %2437 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2439 = stablehlo.broadcast_in_dim %2438, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2440 = stablehlo.add %2436, %2439 : tensor<1x7x768xf32>
    %2441 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %2442 = stablehlo.broadcast_in_dim %2441, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %2443 = stablehlo.broadcast_in_dim %2442, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %2444 = stablehlo.broadcast_in_dim %2442, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %2445 = stablehlo.broadcast_in_dim %2443, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %2446 = stablehlo.broadcast_in_dim %2444, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %2447 = stablehlo.compare  GE, %2445, %2446,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %2448 = stablehlo.broadcast_in_dim %2447, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %2449 = stablehlo.transpose %124, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %2450 = stablehlo.convert %2449 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %2451 = stablehlo.dot_general %2440, %2450, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x2304xf32>) -> tensor<1x7x2304xf32>
    %2452 = stablehlo.convert %125 : (tensor<2304xf16>) -> tensor<2304xf32>
    %2453 = stablehlo.broadcast_in_dim %2452, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %2454 = stablehlo.broadcast_in_dim %2453, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<1x7x2304xf32>
    %2455 = stablehlo.add %2451, %2454 : tensor<1x7x2304xf32>
    %2456 = stablehlo.slice %2455 [0:1, 0:7, 0:768] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %2457 = stablehlo.slice %2455 [0:1, 0:7, 768:1536] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %2458 = stablehlo.slice %2455 [0:1, 0:7, 1536:2304] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %2459 = stablehlo.reshape %2456 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2460 = stablehlo.reshape %2457 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2461 = stablehlo.reshape %2458 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2462 = stablehlo.constant dense<0> : tensor<i32>
    %2463 = stablehlo.constant dense<0> : tensor<i32>
    %2464 = stablehlo.compare  LT, %2462, %2463,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2465 = stablehlo.constant dense<0> : tensor<i32>
    %2466 = stablehlo.constant dense<1024> : tensor<i32>
    %2467 = stablehlo.add %2465, %2466 : tensor<i32>
    %2468 = stablehlo.constant dense<0> : tensor<i32>
    %2469 = stablehlo.select %2464, %2467, %2468 : tensor<i1>, tensor<i32>
    %2470 = stablehlo.constant dense<0> : tensor<i32>
    %2471 = stablehlo.constant dense<0> : tensor<i32>
    %2472 = stablehlo.constant dense<0> : tensor<i32>
    %2473 = stablehlo.dynamic_slice %2448, %2470, %2471, %2469, %2472, sizes = [1, 1, 7, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x7x20xi1>
    %2474 = stablehlo.broadcast_in_dim %211, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %2475 = stablehlo.reshape %2474 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %2476 = stablehlo.broadcast_in_dim %2475, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x7x20xi32>
    %2477 = stablehlo.constant dense<0> : tensor<i32>
    %2478 = stablehlo.broadcast_in_dim %2477, dims = [] : (tensor<i32>) -> tensor<1x1x7x20xi32>
    %2479 = stablehlo.compare  NE, %2476, %2478,  SIGNED : (tensor<1x1x7x20xi32>, tensor<1x1x7x20xi32>) -> tensor<1x1x7x20xi1>
    %2480 = stablehlo.and %2479, %2473 : tensor<1x1x7x20xi1>
    %2481 = stablehlo.convert %2480 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %2482 = stablehlo.constant dense<0> : tensor<i32>
    %2483 = stablehlo.constant dense<0> : tensor<i32>
    %2484 = stablehlo.compare  LT, %2482, %2483,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2485 = stablehlo.constant dense<0> : tensor<i32>
    %2486 = stablehlo.constant dense<20> : tensor<i32>
    %2487 = stablehlo.add %2485, %2486 : tensor<i32>
    %2488 = stablehlo.constant dense<0> : tensor<i32>
    %2489 = stablehlo.select %2484, %2487, %2488 : tensor<i1>, tensor<i32>
    %2490 = stablehlo.constant dense<0> : tensor<i32>
    %2491 = stablehlo.constant dense<0> : tensor<i32>
    %2492 = stablehlo.constant dense<0> : tensor<i32>
    %2493 = stablehlo.dynamic_update_slice %196, %2460, %2490, %2489, %2491, %2492 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %2494 = stablehlo.constant dense<0> : tensor<i32>
    %2495 = stablehlo.constant dense<0> : tensor<i32>
    %2496 = stablehlo.compare  LT, %2494, %2495,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2497 = stablehlo.constant dense<0> : tensor<i32>
    %2498 = stablehlo.constant dense<20> : tensor<i32>
    %2499 = stablehlo.add %2497, %2498 : tensor<i32>
    %2500 = stablehlo.constant dense<0> : tensor<i32>
    %2501 = stablehlo.select %2496, %2499, %2500 : tensor<i1>, tensor<i32>
    %2502 = stablehlo.constant dense<0> : tensor<i32>
    %2503 = stablehlo.constant dense<0> : tensor<i32>
    %2504 = stablehlo.constant dense<0> : tensor<i32>
    %2505 = stablehlo.dynamic_update_slice %198, %2461, %2502, %2501, %2503, %2504 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %2506 = stablehlo.constant dense<0> : tensor<i32>
    %2507 = stablehlo.constant dense<7> : tensor<i32>
    %2508 = stablehlo.add %2506, %2507 : tensor<i32>
    %2509 = stablehlo.iota dim = 0 : tensor<20xi32>
    %2510 = stablehlo.constant dense<0> : tensor<i32>
    %2511 = stablehlo.constant dense<7> : tensor<i32>
    %2512 = stablehlo.add %2510, %2511 : tensor<i32>
    %2513 = stablehlo.broadcast_in_dim %2512, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %2514 = stablehlo.compare  LT, %2509, %2513,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %2515 = stablehlo.broadcast_in_dim %2514, dims = [3] : (tensor<20xi1>) -> tensor<1x1x7x20xi1>
    %2516 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2517 = stablehlo.broadcast_in_dim %2516, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2518 = stablehlo.compare  NE, %2481, %2517,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %2519 = stablehlo.and %2515, %2518 : tensor<1x1x7x20xi1>
    %2520 = stablehlo.convert %2519 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %2521 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2522 = stablehlo.broadcast_in_dim %2521, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2523 = stablehlo.compare  GT, %2520, %2522,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %2524 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2525 = stablehlo.broadcast_in_dim %2524, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2526 = stablehlo.convert %2525 : tensor<1x1x7x20xf32>
    %2527 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %2528 = stablehlo.broadcast_in_dim %2527, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2529 = stablehlo.select %2523, %2526, %2528 : tensor<1x1x7x20xi1>, tensor<1x1x7x20xf32>
    %2530 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %2531 = stablehlo.sqrt %2530 : tensor<f32>
    %2532 = stablehlo.convert %2531 : tensor<f32>
    %2533 = stablehlo.broadcast_in_dim %2532, dims = [] : (tensor<f32>) -> tensor<1x7x12x64xf32>
    %2534 = stablehlo.divide %2459, %2533 : tensor<1x7x12x64xf32>
    %2535 = stablehlo.dot_general %2534, %2493, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x7x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x7x20xf32>
    %2536 = stablehlo.broadcast_in_dim %2529, dims = [0, 1, 2, 3] : (tensor<1x1x7x20xf32>) -> tensor<1x12x7x20xf32>
    %2537 = stablehlo.add %2535, %2536 : tensor<1x12x7x20xf32>
    %2538 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2539 = stablehlo.reduce(%2537 init: %2538) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2540 = stablehlo.broadcast_in_dim %2539, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2541 = stablehlo.broadcast_in_dim %2540, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %2542 = stablehlo.subtract %2537, %2541 : tensor<1x12x7x20xf32>
    %2543 = stablehlo.exponential %2542 : tensor<1x12x7x20xf32>
    %2544 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2545 = stablehlo.reduce(%2543 init: %2544) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2546 = stablehlo.broadcast_in_dim %2545, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2547 = stablehlo.broadcast_in_dim %2546, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %2548 = stablehlo.divide %2543, %2547 : tensor<1x12x7x20xf32>
    %2549 = stablehlo.dot_general %2505, %2548, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x7x20xf32>) -> tensor<1x12x64x7xf32>
    %2550 = stablehlo.transpose %2549, dims = [0, 3, 1, 2] : (tensor<1x12x64x7xf32>) -> tensor<1x7x12x64xf32>
    %2551 = stablehlo.reshape %2550 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %2552 = stablehlo.transpose %126, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %2553 = stablehlo.convert %2552 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2554 = stablehlo.dot_general %2551, %2553, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x768xf32>) -> tensor<1x7x768xf32>
    %2555 = stablehlo.convert %127 : (tensor<768xf16>) -> tensor<768xf32>
    %2556 = stablehlo.broadcast_in_dim %2555, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2557 = stablehlo.broadcast_in_dim %2556, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2558 = stablehlo.add %2554, %2557 : tensor<1x7x768xf32>
    %2559 = stablehlo.add %2558, %2406 : tensor<1x7x768xf32>
    %2560 = stablehlo.multiply %2559, %2559 : tensor<1x7x768xf32>
    %2561 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2562 = stablehlo.reduce(%2560 init: %2561) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2563 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2564 = stablehlo.broadcast_in_dim %2563, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2565 = stablehlo.divide %2562, %2564 : tensor<1x7xf32>
    %2566 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2567 = stablehlo.reduce(%2559 init: %2566) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2568 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2569 = stablehlo.broadcast_in_dim %2568, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2570 = stablehlo.divide %2567, %2569 : tensor<1x7xf32>
    %2571 = stablehlo.multiply %2570, %2570 : tensor<1x7xf32>
    %2572 = stablehlo.subtract %2565, %2571 : tensor<1x7xf32>
    %2573 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2574 = stablehlo.broadcast_in_dim %2573, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2575 = stablehlo.maximum %2574, %2572 : tensor<1x7xf32>
    %2576 = stablehlo.broadcast_in_dim %2570, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2577 = stablehlo.broadcast_in_dim %2575, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2578 = stablehlo.broadcast_in_dim %2576, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2579 = stablehlo.subtract %2559, %2578 : tensor<1x7x768xf32>
    %2580 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2581 = stablehlo.broadcast_in_dim %2580, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2582 = stablehlo.add %2577, %2581 : tensor<1x7x1xf32>
    %2583 = stablehlo.rsqrt %2582 : tensor<1x7x1xf32>
    %2584 = stablehlo.reshape %128 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2585 = stablehlo.convert %2584 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2586 = stablehlo.broadcast_in_dim %2583, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2587 = stablehlo.broadcast_in_dim %2585, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2588 = stablehlo.multiply %2586, %2587 : tensor<1x7x768xf32>
    %2589 = stablehlo.multiply %2579, %2588 : tensor<1x7x768xf32>
    %2590 = stablehlo.reshape %129 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2591 = stablehlo.convert %2590 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2592 = stablehlo.broadcast_in_dim %2591, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2593 = stablehlo.add %2589, %2592 : tensor<1x7x768xf32>
    %2594 = stablehlo.transpose %130, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %2595 = stablehlo.convert %2594 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %2596 = stablehlo.dot_general %2593, %2595, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x3072xf32>) -> tensor<1x7x3072xf32>
    %2597 = stablehlo.convert %131 : (tensor<3072xf16>) -> tensor<3072xf32>
    %2598 = stablehlo.broadcast_in_dim %2597, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %2599 = stablehlo.broadcast_in_dim %2598, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<1x7x3072xf32>
    %2600 = stablehlo.add %2596, %2599 : tensor<1x7x3072xf32>
    %2601 = stablehlo.multiply %2600, %2600 : tensor<1x7x3072xf32>
    %2602 = stablehlo.multiply %2600, %2601 : tensor<1x7x3072xf32>
    %2603 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %2604 = stablehlo.broadcast_in_dim %2603, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2605 = stablehlo.multiply %2604, %2602 : tensor<1x7x3072xf32>
    %2606 = stablehlo.add %2600, %2605 : tensor<1x7x3072xf32>
    %2607 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %2608 = stablehlo.broadcast_in_dim %2607, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2609 = stablehlo.multiply %2608, %2606 : tensor<1x7x3072xf32>
    %2610 = stablehlo.tanh %2609 : tensor<1x7x3072xf32>
    %2611 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2612 = stablehlo.broadcast_in_dim %2611, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2613 = stablehlo.add %2612, %2610 : tensor<1x7x3072xf32>
    %2614 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %2615 = stablehlo.broadcast_in_dim %2614, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2616 = stablehlo.multiply %2615, %2613 : tensor<1x7x3072xf32>
    %2617 = stablehlo.multiply %2600, %2616 : tensor<1x7x3072xf32>
    %2618 = stablehlo.transpose %132, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %2619 = stablehlo.convert %2618 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %2620 = stablehlo.dot_general %2617, %2619, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x3072xf32>, tensor<3072x768xf32>) -> tensor<1x7x768xf32>
    %2621 = stablehlo.convert %133 : (tensor<768xf16>) -> tensor<768xf32>
    %2622 = stablehlo.broadcast_in_dim %2621, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2623 = stablehlo.broadcast_in_dim %2622, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2624 = stablehlo.add %2620, %2623 : tensor<1x7x768xf32>
    %2625 = stablehlo.add %2559, %2624 : tensor<1x7x768xf32>
    %2626 = stablehlo.multiply %2625, %2625 : tensor<1x7x768xf32>
    %2627 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2628 = stablehlo.reduce(%2626 init: %2627) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2629 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2630 = stablehlo.broadcast_in_dim %2629, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2631 = stablehlo.divide %2628, %2630 : tensor<1x7xf32>
    %2632 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2633 = stablehlo.reduce(%2625 init: %2632) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2634 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2635 = stablehlo.broadcast_in_dim %2634, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2636 = stablehlo.divide %2633, %2635 : tensor<1x7xf32>
    %2637 = stablehlo.multiply %2636, %2636 : tensor<1x7xf32>
    %2638 = stablehlo.subtract %2631, %2637 : tensor<1x7xf32>
    %2639 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2640 = stablehlo.broadcast_in_dim %2639, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2641 = stablehlo.maximum %2640, %2638 : tensor<1x7xf32>
    %2642 = stablehlo.broadcast_in_dim %2636, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2643 = stablehlo.broadcast_in_dim %2641, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2644 = stablehlo.broadcast_in_dim %2642, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2645 = stablehlo.subtract %2625, %2644 : tensor<1x7x768xf32>
    %2646 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2647 = stablehlo.broadcast_in_dim %2646, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2648 = stablehlo.add %2643, %2647 : tensor<1x7x1xf32>
    %2649 = stablehlo.rsqrt %2648 : tensor<1x7x1xf32>
    %2650 = stablehlo.reshape %134 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2651 = stablehlo.convert %2650 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2652 = stablehlo.broadcast_in_dim %2649, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2653 = stablehlo.broadcast_in_dim %2651, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2654 = stablehlo.multiply %2652, %2653 : tensor<1x7x768xf32>
    %2655 = stablehlo.multiply %2645, %2654 : tensor<1x7x768xf32>
    %2656 = stablehlo.reshape %135 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2657 = stablehlo.convert %2656 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2658 = stablehlo.broadcast_in_dim %2657, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2659 = stablehlo.add %2655, %2658 : tensor<1x7x768xf32>
    %2660 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %2661 = stablehlo.broadcast_in_dim %2660, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %2662 = stablehlo.broadcast_in_dim %2661, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %2663 = stablehlo.broadcast_in_dim %2661, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %2664 = stablehlo.broadcast_in_dim %2662, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %2665 = stablehlo.broadcast_in_dim %2663, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %2666 = stablehlo.compare  GE, %2664, %2665,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %2667 = stablehlo.broadcast_in_dim %2666, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %2668 = stablehlo.transpose %136, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %2669 = stablehlo.convert %2668 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %2670 = stablehlo.dot_general %2659, %2669, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x2304xf32>) -> tensor<1x7x2304xf32>
    %2671 = stablehlo.convert %137 : (tensor<2304xf16>) -> tensor<2304xf32>
    %2672 = stablehlo.broadcast_in_dim %2671, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %2673 = stablehlo.broadcast_in_dim %2672, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<1x7x2304xf32>
    %2674 = stablehlo.add %2670, %2673 : tensor<1x7x2304xf32>
    %2675 = stablehlo.slice %2674 [0:1, 0:7, 0:768] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %2676 = stablehlo.slice %2674 [0:1, 0:7, 768:1536] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %2677 = stablehlo.slice %2674 [0:1, 0:7, 1536:2304] : (tensor<1x7x2304xf32>) -> tensor<1x7x768xf32>
    %2678 = stablehlo.reshape %2675 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2679 = stablehlo.reshape %2676 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2680 = stablehlo.reshape %2677 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2681 = stablehlo.constant dense<0> : tensor<i32>
    %2682 = stablehlo.constant dense<0> : tensor<i32>
    %2683 = stablehlo.compare  LT, %2681, %2682,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2684 = stablehlo.constant dense<0> : tensor<i32>
    %2685 = stablehlo.constant dense<1024> : tensor<i32>
    %2686 = stablehlo.add %2684, %2685 : tensor<i32>
    %2687 = stablehlo.constant dense<0> : tensor<i32>
    %2688 = stablehlo.select %2683, %2686, %2687 : tensor<i1>, tensor<i32>
    %2689 = stablehlo.constant dense<0> : tensor<i32>
    %2690 = stablehlo.constant dense<0> : tensor<i32>
    %2691 = stablehlo.constant dense<0> : tensor<i32>
    %2692 = stablehlo.dynamic_slice %2667, %2689, %2690, %2688, %2691, sizes = [1, 1, 7, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x7x20xi1>
    %2693 = stablehlo.broadcast_in_dim %211, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %2694 = stablehlo.reshape %2693 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %2695 = stablehlo.broadcast_in_dim %2694, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x7x20xi32>
    %2696 = stablehlo.constant dense<0> : tensor<i32>
    %2697 = stablehlo.broadcast_in_dim %2696, dims = [] : (tensor<i32>) -> tensor<1x1x7x20xi32>
    %2698 = stablehlo.compare  NE, %2695, %2697,  SIGNED : (tensor<1x1x7x20xi32>, tensor<1x1x7x20xi32>) -> tensor<1x1x7x20xi1>
    %2699 = stablehlo.and %2698, %2692 : tensor<1x1x7x20xi1>
    %2700 = stablehlo.convert %2699 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %2701 = stablehlo.constant dense<0> : tensor<i32>
    %2702 = stablehlo.constant dense<0> : tensor<i32>
    %2703 = stablehlo.compare  LT, %2701, %2702,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2704 = stablehlo.constant dense<0> : tensor<i32>
    %2705 = stablehlo.constant dense<20> : tensor<i32>
    %2706 = stablehlo.add %2704, %2705 : tensor<i32>
    %2707 = stablehlo.constant dense<0> : tensor<i32>
    %2708 = stablehlo.select %2703, %2706, %2707 : tensor<i1>, tensor<i32>
    %2709 = stablehlo.constant dense<0> : tensor<i32>
    %2710 = stablehlo.constant dense<0> : tensor<i32>
    %2711 = stablehlo.constant dense<0> : tensor<i32>
    %2712 = stablehlo.dynamic_update_slice %200, %2679, %2709, %2708, %2710, %2711 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %2713 = stablehlo.constant dense<0> : tensor<i32>
    %2714 = stablehlo.constant dense<0> : tensor<i32>
    %2715 = stablehlo.compare  LT, %2713, %2714,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2716 = stablehlo.constant dense<0> : tensor<i32>
    %2717 = stablehlo.constant dense<20> : tensor<i32>
    %2718 = stablehlo.add %2716, %2717 : tensor<i32>
    %2719 = stablehlo.constant dense<0> : tensor<i32>
    %2720 = stablehlo.select %2715, %2718, %2719 : tensor<i1>, tensor<i32>
    %2721 = stablehlo.constant dense<0> : tensor<i32>
    %2722 = stablehlo.constant dense<0> : tensor<i32>
    %2723 = stablehlo.constant dense<0> : tensor<i32>
    %2724 = stablehlo.dynamic_update_slice %202, %2680, %2721, %2720, %2722, %2723 : (tensor<1x20x12x64xf32>, tensor<1x7x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %2725 = stablehlo.constant dense<0> : tensor<i32>
    %2726 = stablehlo.constant dense<7> : tensor<i32>
    %2727 = stablehlo.add %2725, %2726 : tensor<i32>
    %2728 = stablehlo.iota dim = 0 : tensor<20xi32>
    %2729 = stablehlo.constant dense<0> : tensor<i32>
    %2730 = stablehlo.constant dense<7> : tensor<i32>
    %2731 = stablehlo.add %2729, %2730 : tensor<i32>
    %2732 = stablehlo.broadcast_in_dim %2731, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %2733 = stablehlo.compare  LT, %2728, %2732,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %2734 = stablehlo.broadcast_in_dim %2733, dims = [3] : (tensor<20xi1>) -> tensor<1x1x7x20xi1>
    %2735 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2736 = stablehlo.broadcast_in_dim %2735, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2737 = stablehlo.compare  NE, %2700, %2736,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %2738 = stablehlo.and %2734, %2737 : tensor<1x1x7x20xi1>
    %2739 = stablehlo.convert %2738 : (tensor<1x1x7x20xi1>) -> tensor<1x1x7x20xf32>
    %2740 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2741 = stablehlo.broadcast_in_dim %2740, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2742 = stablehlo.compare  GT, %2739, %2741,  FLOAT : (tensor<1x1x7x20xf32>, tensor<1x1x7x20xf32>) -> tensor<1x1x7x20xi1>
    %2743 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2744 = stablehlo.broadcast_in_dim %2743, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2745 = stablehlo.convert %2744 : tensor<1x1x7x20xf32>
    %2746 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %2747 = stablehlo.broadcast_in_dim %2746, dims = [] : (tensor<f32>) -> tensor<1x1x7x20xf32>
    %2748 = stablehlo.select %2742, %2745, %2747 : tensor<1x1x7x20xi1>, tensor<1x1x7x20xf32>
    %2749 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %2750 = stablehlo.sqrt %2749 : tensor<f32>
    %2751 = stablehlo.convert %2750 : tensor<f32>
    %2752 = stablehlo.broadcast_in_dim %2751, dims = [] : (tensor<f32>) -> tensor<1x7x12x64xf32>
    %2753 = stablehlo.divide %2678, %2752 : tensor<1x7x12x64xf32>
    %2754 = stablehlo.dot_general %2753, %2712, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x7x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x7x20xf32>
    %2755 = stablehlo.broadcast_in_dim %2748, dims = [0, 1, 2, 3] : (tensor<1x1x7x20xf32>) -> tensor<1x12x7x20xf32>
    %2756 = stablehlo.add %2754, %2755 : tensor<1x12x7x20xf32>
    %2757 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2758 = stablehlo.reduce(%2756 init: %2757) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2759 = stablehlo.broadcast_in_dim %2758, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2760 = stablehlo.broadcast_in_dim %2759, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %2761 = stablehlo.subtract %2756, %2760 : tensor<1x12x7x20xf32>
    %2762 = stablehlo.exponential %2761 : tensor<1x12x7x20xf32>
    %2763 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2764 = stablehlo.reduce(%2762 init: %2763) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x20xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2765 = stablehlo.broadcast_in_dim %2764, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2766 = stablehlo.broadcast_in_dim %2765, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x20xf32>
    %2767 = stablehlo.divide %2762, %2766 : tensor<1x12x7x20xf32>
    %2768 = stablehlo.dot_general %2724, %2767, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x7x20xf32>) -> tensor<1x12x64x7xf32>
    %2769 = stablehlo.transpose %2768, dims = [0, 3, 1, 2] : (tensor<1x12x64x7xf32>) -> tensor<1x7x12x64xf32>
    %2770 = stablehlo.reshape %2769 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %2771 = stablehlo.transpose %138, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %2772 = stablehlo.convert %2771 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2773 = stablehlo.dot_general %2770, %2772, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x768xf32>) -> tensor<1x7x768xf32>
    %2774 = stablehlo.convert %139 : (tensor<768xf16>) -> tensor<768xf32>
    %2775 = stablehlo.broadcast_in_dim %2774, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2776 = stablehlo.broadcast_in_dim %2775, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2777 = stablehlo.add %2773, %2776 : tensor<1x7x768xf32>
    %2778 = stablehlo.add %2777, %2625 : tensor<1x7x768xf32>
    %2779 = stablehlo.multiply %2778, %2778 : tensor<1x7x768xf32>
    %2780 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2781 = stablehlo.reduce(%2779 init: %2780) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2782 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2783 = stablehlo.broadcast_in_dim %2782, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2784 = stablehlo.divide %2781, %2783 : tensor<1x7xf32>
    %2785 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2786 = stablehlo.reduce(%2778 init: %2785) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2787 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2788 = stablehlo.broadcast_in_dim %2787, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2789 = stablehlo.divide %2786, %2788 : tensor<1x7xf32>
    %2790 = stablehlo.multiply %2789, %2789 : tensor<1x7xf32>
    %2791 = stablehlo.subtract %2784, %2790 : tensor<1x7xf32>
    %2792 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2793 = stablehlo.broadcast_in_dim %2792, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2794 = stablehlo.maximum %2793, %2791 : tensor<1x7xf32>
    %2795 = stablehlo.broadcast_in_dim %2789, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2796 = stablehlo.broadcast_in_dim %2794, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2797 = stablehlo.broadcast_in_dim %2795, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2798 = stablehlo.subtract %2778, %2797 : tensor<1x7x768xf32>
    %2799 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2800 = stablehlo.broadcast_in_dim %2799, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2801 = stablehlo.add %2796, %2800 : tensor<1x7x1xf32>
    %2802 = stablehlo.rsqrt %2801 : tensor<1x7x1xf32>
    %2803 = stablehlo.reshape %140 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2804 = stablehlo.convert %2803 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2805 = stablehlo.broadcast_in_dim %2802, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2806 = stablehlo.broadcast_in_dim %2804, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2807 = stablehlo.multiply %2805, %2806 : tensor<1x7x768xf32>
    %2808 = stablehlo.multiply %2798, %2807 : tensor<1x7x768xf32>
    %2809 = stablehlo.reshape %141 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2810 = stablehlo.convert %2809 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2811 = stablehlo.broadcast_in_dim %2810, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2812 = stablehlo.add %2808, %2811 : tensor<1x7x768xf32>
    %2813 = stablehlo.transpose %142, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %2814 = stablehlo.convert %2813 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %2815 = stablehlo.dot_general %2812, %2814, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x3072xf32>) -> tensor<1x7x3072xf32>
    %2816 = stablehlo.convert %143 : (tensor<3072xf16>) -> tensor<3072xf32>
    %2817 = stablehlo.broadcast_in_dim %2816, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %2818 = stablehlo.broadcast_in_dim %2817, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<1x7x3072xf32>
    %2819 = stablehlo.add %2815, %2818 : tensor<1x7x3072xf32>
    %2820 = stablehlo.multiply %2819, %2819 : tensor<1x7x3072xf32>
    %2821 = stablehlo.multiply %2819, %2820 : tensor<1x7x3072xf32>
    %2822 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %2823 = stablehlo.broadcast_in_dim %2822, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2824 = stablehlo.multiply %2823, %2821 : tensor<1x7x3072xf32>
    %2825 = stablehlo.add %2819, %2824 : tensor<1x7x3072xf32>
    %2826 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %2827 = stablehlo.broadcast_in_dim %2826, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2828 = stablehlo.multiply %2827, %2825 : tensor<1x7x3072xf32>
    %2829 = stablehlo.tanh %2828 : tensor<1x7x3072xf32>
    %2830 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2831 = stablehlo.broadcast_in_dim %2830, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2832 = stablehlo.add %2831, %2829 : tensor<1x7x3072xf32>
    %2833 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %2834 = stablehlo.broadcast_in_dim %2833, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2835 = stablehlo.multiply %2834, %2832 : tensor<1x7x3072xf32>
    %2836 = stablehlo.multiply %2819, %2835 : tensor<1x7x3072xf32>
    %2837 = stablehlo.transpose %144, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %2838 = stablehlo.convert %2837 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %2839 = stablehlo.dot_general %2836, %2838, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x3072xf32>, tensor<3072x768xf32>) -> tensor<1x7x768xf32>
    %2840 = stablehlo.convert %145 : (tensor<768xf16>) -> tensor<768xf32>
    %2841 = stablehlo.broadcast_in_dim %2840, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2842 = stablehlo.broadcast_in_dim %2841, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2843 = stablehlo.add %2839, %2842 : tensor<1x7x768xf32>
    %2844 = stablehlo.add %2778, %2843 : tensor<1x7x768xf32>
    %2845 = stablehlo.multiply %2844, %2844 : tensor<1x7x768xf32>
    %2846 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2847 = stablehlo.reduce(%2845 init: %2846) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2848 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2849 = stablehlo.broadcast_in_dim %2848, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2850 = stablehlo.divide %2847, %2849 : tensor<1x7xf32>
    %2851 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2852 = stablehlo.reduce(%2844 init: %2851) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2853 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2854 = stablehlo.broadcast_in_dim %2853, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2855 = stablehlo.divide %2852, %2854 : tensor<1x7xf32>
    %2856 = stablehlo.multiply %2855, %2855 : tensor<1x7xf32>
    %2857 = stablehlo.subtract %2850, %2856 : tensor<1x7xf32>
    %2858 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2859 = stablehlo.broadcast_in_dim %2858, dims = [] : (tensor<f32>) -> tensor<1x7xf32>
    %2860 = stablehlo.maximum %2859, %2857 : tensor<1x7xf32>
    %2861 = stablehlo.broadcast_in_dim %2855, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2862 = stablehlo.broadcast_in_dim %2860, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2863 = stablehlo.broadcast_in_dim %2861, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2864 = stablehlo.subtract %2844, %2863 : tensor<1x7x768xf32>
    %2865 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2866 = stablehlo.broadcast_in_dim %2865, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2867 = stablehlo.add %2862, %2866 : tensor<1x7x1xf32>
    %2868 = stablehlo.rsqrt %2867 : tensor<1x7x1xf32>
    %2869 = stablehlo.reshape %146 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2870 = stablehlo.convert %2869 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2871 = stablehlo.broadcast_in_dim %2868, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2872 = stablehlo.broadcast_in_dim %2870, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2873 = stablehlo.multiply %2871, %2872 : tensor<1x7x768xf32>
    %2874 = stablehlo.multiply %2864, %2873 : tensor<1x7x768xf32>
    %2875 = stablehlo.reshape %147 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2876 = stablehlo.convert %2875 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2877 = stablehlo.broadcast_in_dim %2876, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2878 = stablehlo.add %2874, %2877 : tensor<1x7x768xf32>
    %2879 = stablehlo.transpose %0, dims = [1, 0] : (tensor<50257x768xf16>) -> tensor<768x50257xf16>
    %2880 = stablehlo.convert %2879 : (tensor<768x50257xf16>) -> tensor<768x50257xf32>
    %2881 = stablehlo.dot_general %2878, %2880, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x7x768xf32>, tensor<768x50257xf32>) -> tensor<1x7x50257xf32>
    %2882 = stablehlo.constant dense<0> : tensor<i32>
    %2883 = stablehlo.constant dense<0> : tensor<i32>
    %2884 = stablehlo.compare  LT, %2882, %2883,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2885 = stablehlo.constant dense<0> : tensor<i32>
    %2886 = stablehlo.constant dense<1> : tensor<i32>
    %2887 = stablehlo.add %2885, %2886 : tensor<i32>
    %2888 = stablehlo.constant dense<0> : tensor<i32>
    %2889 = stablehlo.select %2884, %2887, %2888 : tensor<i1>, tensor<i32>
    %2890 = stablehlo.constant dense<-1> : tensor<i32>
    %2891 = stablehlo.constant dense<0> : tensor<i32>
    %2892 = stablehlo.compare  LT, %2890, %2891,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2893 = stablehlo.constant dense<-1> : tensor<i32>
    %2894 = stablehlo.constant dense<7> : tensor<i32>
    %2895 = stablehlo.add %2893, %2894 : tensor<i32>
    %2896 = stablehlo.constant dense<-1> : tensor<i32>
    %2897 = stablehlo.select %2892, %2895, %2896 : tensor<i1>, tensor<i32>
    %2898 = stablehlo.constant dense<0> : tensor<i32>
    %2899 = stablehlo.constant dense<0> : tensor<i32>
    %2900 = stablehlo.compare  LT, %2898, %2899,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2901 = stablehlo.constant dense<0> : tensor<i32>
    %2902 = stablehlo.constant dense<50257> : tensor<i32>
    %2903 = stablehlo.add %2901, %2902 : tensor<i32>
    %2904 = stablehlo.constant dense<0> : tensor<i32>
    %2905 = stablehlo.select %2900, %2903, %2904 : tensor<i1>, tensor<i32>
    %2906 = stablehlo.dynamic_slice %2881, %2889, %2897, %2905, sizes = [1, 1, 50257] : (tensor<1x7x50257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x50257xf32>
    %2907 = stablehlo.reshape %2906 : (tensor<1x1x50257xf32>) -> tensor<1x50257xf32>
    %2908 = stablehlo.constant dense<7> : tensor<i32>
    %2909 = stablehlo.constant dense<0> : tensor<i32>
    %2910 = stablehlo.subtract %2908, %2909 : tensor<i32>
    %2911 = stablehlo.constant dense<0> : tensor<i32>
    %2912 = stablehlo.constant dense<1> : tensor<i32>
    %2913 = call @clip(%2910, %2911, %2912) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2914 = stablehlo.constant dense<1> : tensor<i32>
    %2915 = stablehlo.subtract %2914, %2913 : tensor<i32>
    %2916 = stablehlo.constant dense<50256> : tensor<i32>
    %2917 = stablehlo.broadcast_in_dim %2916, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2918 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2919 = stablehlo.broadcast_in_dim %2918, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %2920 = "stablehlo.scatter"(%2907, %2917, %2919) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      stablehlo.return %arg3 : tensor<f32>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50257xf32>, tensor<1xi32>, tensor<1xf32>) -> tensor<1x50257xf32>
    %2921 = call @_where_2(%2915, %2920, %2907) : (tensor<i32>, tensor<1x50257xf32>, tensor<1x50257xf32>) -> tensor<1x50257xf32>
    %2922 = call @argmax(%2921) : (tensor<1x50257xf32>) -> tensor<1xi32>
    %2923 = stablehlo.not %154 : tensor<1xi1>
    %2924 = stablehlo.convert %2923 : (tensor<1xi1>) -> tensor<1xi32>
    %2925 = stablehlo.multiply %2922, %2924 : tensor<1xi32>
    %2926 = stablehlo.convert %154 : (tensor<1xi1>) -> tensor<1xi32>
    %2927 = stablehlo.constant dense<50256> : tensor<i32>
    %2928 = stablehlo.broadcast_in_dim %2927, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2929 = stablehlo.multiply %2928, %2926 : tensor<1xi32>
    %2930 = stablehlo.add %2925, %2929 : tensor<1xi32>
    %2931 = stablehlo.constant dense<50256> : tensor<i32>
    %2932 = stablehlo.broadcast_in_dim %2931, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2933 = stablehlo.compare  EQ, %2930, %2932,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %2934 = stablehlo.or %154, %2933 : tensor<1xi1>
    %2935 = stablehlo.broadcast_in_dim %2930, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %2936 = stablehlo.constant dense<7> : tensor<i32>
    %2937 = stablehlo.constant dense<0> : tensor<i32>
    %2938 = stablehlo.compare  LT, %2936, %2937,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2939 = stablehlo.constant dense<7> : tensor<i32>
    %2940 = stablehlo.constant dense<20> : tensor<i32>
    %2941 = stablehlo.add %2939, %2940 : tensor<i32>
    %2942 = stablehlo.constant dense<7> : tensor<i32>
    %2943 = stablehlo.select %2938, %2941, %2942 : tensor<i1>, tensor<i32>
    %2944 = stablehlo.constant dense<0> : tensor<i32>
    %2945 = stablehlo.dynamic_update_slice %152, %2935, %2944, %2943 : (tensor<1x20xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x20xi32>
    %2946 = stablehlo.constant dense<0> : tensor<i32>
    %2947 = stablehlo.constant dense<0> : tensor<i32>
    %2948 = stablehlo.compare  LT, %2946, %2947,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2949 = stablehlo.constant dense<0> : tensor<i32>
    %2950 = stablehlo.constant dense<1> : tensor<i32>
    %2951 = stablehlo.add %2949, %2950 : tensor<i32>
    %2952 = stablehlo.constant dense<0> : tensor<i32>
    %2953 = stablehlo.select %2948, %2951, %2952 : tensor<i1>, tensor<i32>
    %2954 = stablehlo.constant dense<6> : tensor<i32>
    %2955 = stablehlo.constant dense<0> : tensor<i32>
    %2956 = stablehlo.compare  LT, %2954, %2955,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2957 = stablehlo.constant dense<6> : tensor<i32>
    %2958 = stablehlo.constant dense<7> : tensor<i32>
    %2959 = stablehlo.add %2957, %2958 : tensor<i32>
    %2960 = stablehlo.constant dense<6> : tensor<i32>
    %2961 = stablehlo.select %2956, %2959, %2960 : tensor<i1>, tensor<i32>
    %2962 = stablehlo.dynamic_slice %208, %2953, %2961, sizes = [1, 1] : (tensor<1x7xi32>, tensor<i32>, tensor<i32>) -> tensor<1x1xi32>
    %2963 = stablehlo.constant dense<1> : tensor<i32>
    %2964 = stablehlo.broadcast_in_dim %2963, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %2965 = stablehlo.add %2962, %2964 : tensor<1x1xi32>
    %2966 = stablehlo.constant dense<7> : tensor<i32>
    %2967 = stablehlo.constant dense<1> : tensor<i32>
    %2968 = stablehlo.add %2966, %2967 : tensor<i32>
    %2969 = stablehlo.constant dense<50256> : tensor<i32>
    %2970 = stablehlo.constant dense<50256> : tensor<i32>
    %2971:192 = stablehlo.while(%iterArg = %0, %iterArg_0 = %1, %iterArg_1 = %2, %iterArg_2 = %3, %iterArg_3 = %4, %iterArg_4 = %5, %iterArg_5 = %6, %iterArg_6 = %7, %iterArg_7 = %8, %iterArg_8 = %9, %iterArg_9 = %10, %iterArg_10 = %11, %iterArg_11 = %12, %iterArg_12 = %13, %iterArg_13 = %14, %iterArg_14 = %15, %iterArg_15 = %16, %iterArg_16 = %17, %iterArg_17 = %18, %iterArg_18 = %19, %iterArg_19 = %20, %iterArg_20 = %21, %iterArg_21 = %22, %iterArg_22 = %23, %iterArg_23 = %24, %iterArg_24 = %25, %iterArg_25 = %26, %iterArg_26 = %27, %iterArg_27 = %28, %iterArg_28 = %29, %iterArg_29 = %30, %iterArg_30 = %31, %iterArg_31 = %32, %iterArg_32 = %33, %iterArg_33 = %34, %iterArg_34 = %35, %iterArg_35 = %36, %iterArg_36 = %37, %iterArg_37 = %38, %iterArg_38 = %39, %iterArg_39 = %40, %iterArg_40 = %41, %iterArg_41 = %42, %iterArg_42 = %43, %iterArg_43 = %44, %iterArg_44 = %45, %iterArg_45 = %46, %iterArg_46 = %47, %iterArg_47 = %48, %iterArg_48 = %49, %iterArg_49 = %50, %iterArg_50 = %51, %iterArg_51 = %52, %iterArg_52 = %53, %iterArg_53 = %54, %iterArg_54 = %55, %iterArg_55 = %56, %iterArg_56 = %57, %iterArg_57 = %58, %iterArg_58 = %59, %iterArg_59 = %60, %iterArg_60 = %61, %iterArg_61 = %62, %iterArg_62 = %63, %iterArg_63 = %64, %iterArg_64 = %65, %iterArg_65 = %66, %iterArg_66 = %67, %iterArg_67 = %68, %iterArg_68 = %69, %iterArg_69 = %70, %iterArg_70 = %71, %iterArg_71 = %72, %iterArg_72 = %73, %iterArg_73 = %74, %iterArg_74 = %75, %iterArg_75 = %76, %iterArg_76 = %77, %iterArg_77 = %78, %iterArg_78 = %79, %iterArg_79 = %80, %iterArg_80 = %81, %iterArg_81 = %82, %iterArg_82 = %83, %iterArg_83 = %84, %iterArg_84 = %85, %iterArg_85 = %86, %iterArg_86 = %87, %iterArg_87 = %88, %iterArg_88 = %89, %iterArg_89 = %90, %iterArg_90 = %91, %iterArg_91 = %92, %iterArg_92 = %93, %iterArg_93 = %94, %iterArg_94 = %95, %iterArg_95 = %96, %iterArg_96 = %97, %iterArg_97 = %98, %iterArg_98 = %99, %iterArg_99 = %100, %iterArg_100 = %101, %iterArg_101 = %102, %iterArg_102 = %103, %iterArg_103 = %104, %iterArg_104 = %105, %iterArg_105 = %106, %iterArg_106 = %107, %iterArg_107 = %108, %iterArg_108 = %109, %iterArg_109 = %110, %iterArg_110 = %111, %iterArg_111 = %112, %iterArg_112 = %113, %iterArg_113 = %114, %iterArg_114 = %115, %iterArg_115 = %116, %iterArg_116 = %117, %iterArg_117 = %118, %iterArg_118 = %119, %iterArg_119 = %120, %iterArg_120 = %121, %iterArg_121 = %122, %iterArg_122 = %123, %iterArg_123 = %124, %iterArg_124 = %125, %iterArg_125 = %126, %iterArg_126 = %127, %iterArg_127 = %128, %iterArg_128 = %129, %iterArg_129 = %130, %iterArg_130 = %131, %iterArg_131 = %132, %iterArg_132 = %133, %iterArg_133 = %134, %iterArg_134 = %135, %iterArg_135 = %136, %iterArg_136 = %137, %iterArg_137 = %138, %iterArg_138 = %139, %iterArg_139 = %140, %iterArg_140 = %141, %iterArg_141 = %142, %iterArg_142 = %143, %iterArg_143 = %144, %iterArg_144 = %145, %iterArg_145 = %146, %iterArg_146 = %147, %iterArg_147 = %2969, %iterArg_148 = %2970, %iterArg_149 = %2968, %iterArg_150 = %2945, %iterArg_151 = %2935, %iterArg_152 = %2934, %iterArg_153 = %211, %iterArg_154 = %318, %iterArg_155 = %303, %iterArg_156 = %315, %iterArg_157 = %537, %iterArg_158 = %522, %iterArg_159 = %534, %iterArg_160 = %2508, %iterArg_161 = %2493, %iterArg_162 = %2505, %iterArg_163 = %2727, %iterArg_164 = %2712, %iterArg_165 = %2724, %iterArg_166 = %756, %iterArg_167 = %741, %iterArg_168 = %753, %iterArg_169 = %975, %iterArg_170 = %960, %iterArg_171 = %972, %iterArg_172 = %1194, %iterArg_173 = %1179, %iterArg_174 = %1191, %iterArg_175 = %1413, %iterArg_176 = %1398, %iterArg_177 = %1410, %iterArg_178 = %1632, %iterArg_179 = %1617, %iterArg_180 = %1629, %iterArg_181 = %1851, %iterArg_182 = %1836, %iterArg_183 = %1848, %iterArg_184 = %2070, %iterArg_185 = %2055, %iterArg_186 = %2067, %iterArg_187 = %2289, %iterArg_188 = %2274, %iterArg_189 = %2286, %iterArg_190 = %2965) : tensor<50257x768xf16>, tensor<1024x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x20xi32>, tensor<1x1xi32>, tensor<1xi1>, tensor<1x20xi32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<1x1xi32>
     cond {
      %2972 = stablehlo.constant dense<20> : tensor<i32>
      %2973 = stablehlo.compare  EQ, %iterArg_149, %2972,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2974 = stablehlo.constant dense<true> : tensor<i1>
      %2975 = stablehlo.reduce(%iterArg_152 init: %2974) applies stablehlo.and across dimensions = [0] : (tensor<1xi1>, tensor<i1>) -> tensor<i1>
      %2976 = stablehlo.constant dense<false> : tensor<i1>
      %2977 = stablehlo.broadcast_in_dim %2976, dims = [] : (tensor<i1>) -> tensor<i1>
      %2978 = stablehlo.compare  NE, %2973, %2977,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %2979 = stablehlo.or %2978, %2975 : tensor<i1>
      %2980 = stablehlo.not %2979 : tensor<i1>
      stablehlo.return %2980 : tensor<i1>
    } do {
      %2972 = stablehlo.convert %iterArg : (tensor<50257x768xf16>) -> tensor<50257x768xf32>
      %2973 = func.call @_take_3(%2972, %iterArg_151) : (tensor<50257x768xf32>, tensor<1x1xi32>) -> tensor<1x1x768xf32>
      %2974 = stablehlo.convert %iterArg_0 : (tensor<1024x768xf16>) -> tensor<1024x768xf32>
      %2975 = func.call @_take_5(%2974, %iterArg_190) : (tensor<1024x768xf32>, tensor<1x1xi32>) -> tensor<1x1x768xf32>
      %2976 = stablehlo.add %2973, %2975 : tensor<1x1x768xf32>
      %2977 = stablehlo.multiply %2976, %2976 : tensor<1x1x768xf32>
      %2978 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2979 = stablehlo.reduce(%2977 init: %2978) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %2980 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %2981 = stablehlo.broadcast_in_dim %2980, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %2982 = stablehlo.divide %2979, %2981 : tensor<1x1xf32>
      %2983 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2984 = stablehlo.reduce(%2976 init: %2983) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %2985 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %2986 = stablehlo.broadcast_in_dim %2985, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %2987 = stablehlo.divide %2984, %2986 : tensor<1x1xf32>
      %2988 = stablehlo.multiply %2987, %2987 : tensor<1x1xf32>
      %2989 = stablehlo.subtract %2982, %2988 : tensor<1x1xf32>
      %2990 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2991 = stablehlo.broadcast_in_dim %2990, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %2992 = stablehlo.maximum %2991, %2989 : tensor<1x1xf32>
      %2993 = stablehlo.broadcast_in_dim %2987, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %2994 = stablehlo.broadcast_in_dim %2992, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %2995 = stablehlo.broadcast_in_dim %2993, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %2996 = stablehlo.subtract %2976, %2995 : tensor<1x1x768xf32>
      %2997 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2998 = stablehlo.broadcast_in_dim %2997, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %2999 = stablehlo.add %2994, %2998 : tensor<1x1x1xf32>
      %3000 = stablehlo.rsqrt %2999 : tensor<1x1x1xf32>
      %3001 = stablehlo.reshape %iterArg_1 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3002 = stablehlo.convert %3001 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3003 = stablehlo.broadcast_in_dim %3000, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3004 = stablehlo.multiply %3003, %3002 : tensor<1x1x768xf32>
      %3005 = stablehlo.multiply %2996, %3004 : tensor<1x1x768xf32>
      %3006 = stablehlo.reshape %iterArg_2 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3007 = stablehlo.convert %3006 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3008 = stablehlo.add %3005, %3007 : tensor<1x1x768xf32>
      %3009 = stablehlo.constant dense<true> : tensor<i1>
      %3010 = stablehlo.broadcast_in_dim %3009, dims = [] : (tensor<i1>) -> tensor<1x1024xi1>
      %3011 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %3012 = stablehlo.broadcast_in_dim %3011, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %3013 = stablehlo.broadcast_in_dim %3012, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %3014 = stablehlo.broadcast_in_dim %3012, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %3015 = stablehlo.broadcast_in_dim %3013, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %3016 = stablehlo.broadcast_in_dim %3014, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %3017 = stablehlo.compare  GE, %3015, %3016,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %3018 = stablehlo.broadcast_in_dim %3017, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %3019 = stablehlo.transpose %iterArg_3, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %3020 = stablehlo.convert %3019 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %3021 = stablehlo.dot_general %3008, %3020, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32>
      %3022 = stablehlo.convert %iterArg_4 : (tensor<2304xf16>) -> tensor<2304xf32>
      %3023 = stablehlo.broadcast_in_dim %3022, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %3024 = stablehlo.add %3021, %3023 : tensor<1x1x2304xf32>
      %3025 = stablehlo.slice %3024 [0:1, 0:1, 0:768] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3026 = stablehlo.slice %3024 [0:1, 0:1, 768:1536] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3027 = stablehlo.slice %3024 [0:1, 0:1, 1536:2304] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3028 = stablehlo.reshape %3025 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3029 = stablehlo.reshape %3026 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3030 = stablehlo.reshape %3027 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3031 = stablehlo.constant dense<0> : tensor<i32>
      %3032 = stablehlo.compare  LT, %iterArg_154, %3031,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3033 = stablehlo.constant dense<1024> : tensor<i32>
      %3034 = stablehlo.add %iterArg_154, %3033 : tensor<i32>
      %3035 = stablehlo.select %3032, %3034, %iterArg_154 : tensor<i1>, tensor<i32>
      %3036 = stablehlo.constant dense<0> : tensor<i32>
      %3037 = stablehlo.constant dense<0> : tensor<i32>
      %3038 = stablehlo.constant dense<0> : tensor<i32>
      %3039 = stablehlo.dynamic_slice %3018, %3036, %3037, %3035, %3038, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %3040 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %3041 = stablehlo.constant dense<0> : tensor<i32>
      %3042 = stablehlo.broadcast_in_dim %3041, dims = [] : (tensor<i32>) -> tensor<1x1x1x20xi32>
      %3043 = stablehlo.compare  NE, %3040, %3042,  SIGNED : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %3044 = stablehlo.and %3043, %3039 : tensor<1x1x1x20xi1>
      %3045 = stablehlo.convert %3044 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %3046 = stablehlo.constant dense<0> : tensor<i32>
      %3047 = stablehlo.compare  LT, %iterArg_154, %3046,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3048 = stablehlo.constant dense<20> : tensor<i32>
      %3049 = stablehlo.add %iterArg_154, %3048 : tensor<i32>
      %3050 = stablehlo.select %3047, %3049, %iterArg_154 : tensor<i1>, tensor<i32>
      %3051 = stablehlo.constant dense<0> : tensor<i32>
      %3052 = stablehlo.constant dense<0> : tensor<i32>
      %3053 = stablehlo.constant dense<0> : tensor<i32>
      %3054 = stablehlo.dynamic_update_slice %iterArg_155, %3029, %3051, %3050, %3052, %3053 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %3055 = stablehlo.constant dense<0> : tensor<i32>
      %3056 = stablehlo.compare  LT, %iterArg_154, %3055,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3057 = stablehlo.constant dense<20> : tensor<i32>
      %3058 = stablehlo.add %iterArg_154, %3057 : tensor<i32>
      %3059 = stablehlo.select %3056, %3058, %iterArg_154 : tensor<i1>, tensor<i32>
      %3060 = stablehlo.constant dense<0> : tensor<i32>
      %3061 = stablehlo.constant dense<0> : tensor<i32>
      %3062 = stablehlo.constant dense<0> : tensor<i32>
      %3063 = stablehlo.dynamic_update_slice %iterArg_156, %3030, %3060, %3059, %3061, %3062 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %3064 = stablehlo.constant dense<1> : tensor<i32>
      %3065 = stablehlo.add %iterArg_154, %3064 : tensor<i32>
      %3066 = stablehlo.iota dim = 0 : tensor<20xi32>
      %3067 = stablehlo.constant dense<1> : tensor<i32>
      %3068 = stablehlo.add %iterArg_154, %3067 : tensor<i32>
      %3069 = stablehlo.broadcast_in_dim %3068, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %3070 = stablehlo.compare  LT, %3066, %3069,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %3071 = stablehlo.broadcast_in_dim %3070, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %3072 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3073 = stablehlo.broadcast_in_dim %3072, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3074 = stablehlo.compare  NE, %3045, %3073,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %3075 = stablehlo.and %3071, %3074 : tensor<1x1x1x20xi1>
      %3076 = stablehlo.convert %3075 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %3077 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3078 = stablehlo.broadcast_in_dim %3077, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3079 = stablehlo.compare  GT, %3076, %3078,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %3080 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3081 = stablehlo.broadcast_in_dim %3080, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3082 = stablehlo.convert %3081 : tensor<1x1x1x20xf32>
      %3083 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %3084 = stablehlo.broadcast_in_dim %3083, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3085 = stablehlo.select %3079, %3082, %3084 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %3086 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %3087 = stablehlo.sqrt %3086 : tensor<f32>
      %3088 = stablehlo.convert %3087 : tensor<f32>
      %3089 = stablehlo.broadcast_in_dim %3088, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %3090 = stablehlo.divide %3028, %3089 : tensor<1x1x12x64xf32>
      %3091 = stablehlo.dot_general %3090, %3054, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %3092 = stablehlo.broadcast_in_dim %3085, dims = [0, 1, 2, 3] : (tensor<1x1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %3093 = stablehlo.add %3091, %3092 : tensor<1x12x1x20xf32>
      %3094 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %3095 = stablehlo.reduce(%3093 init: %3094) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %3096 = stablehlo.broadcast_in_dim %3095, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %3097 = stablehlo.broadcast_in_dim %3096, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %3098 = stablehlo.subtract %3093, %3097 : tensor<1x12x1x20xf32>
      %3099 = stablehlo.exponential %3098 : tensor<1x12x1x20xf32>
      %3100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3101 = stablehlo.reduce(%3099 init: %3100) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %3102 = stablehlo.broadcast_in_dim %3101, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %3103 = stablehlo.broadcast_in_dim %3102, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %3104 = stablehlo.divide %3099, %3103 : tensor<1x12x1x20xf32>
      %3105 = stablehlo.dot_general %3063, %3104, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %3106 = stablehlo.transpose %3105, dims = [0, 3, 1, 2] : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %3107 = stablehlo.reshape %3106 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %3108 = stablehlo.transpose %iterArg_5, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %3109 = stablehlo.convert %3108 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %3110 = stablehlo.dot_general %3107, %3109, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %3111 = stablehlo.convert %iterArg_6 : (tensor<768xf16>) -> tensor<768xf32>
      %3112 = stablehlo.broadcast_in_dim %3111, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3113 = stablehlo.add %3110, %3112 : tensor<1x1x768xf32>
      %3114 = stablehlo.add %3113, %2976 : tensor<1x1x768xf32>
      %3115 = stablehlo.multiply %3114, %3114 : tensor<1x1x768xf32>
      %3116 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3117 = stablehlo.reduce(%3115 init: %3116) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3118 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3119 = stablehlo.broadcast_in_dim %3118, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3120 = stablehlo.divide %3117, %3119 : tensor<1x1xf32>
      %3121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3122 = stablehlo.reduce(%3114 init: %3121) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3123 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3124 = stablehlo.broadcast_in_dim %3123, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3125 = stablehlo.divide %3122, %3124 : tensor<1x1xf32>
      %3126 = stablehlo.multiply %3125, %3125 : tensor<1x1xf32>
      %3127 = stablehlo.subtract %3120, %3126 : tensor<1x1xf32>
      %3128 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3129 = stablehlo.broadcast_in_dim %3128, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3130 = stablehlo.maximum %3129, %3127 : tensor<1x1xf32>
      %3131 = stablehlo.broadcast_in_dim %3125, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3132 = stablehlo.broadcast_in_dim %3130, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3133 = stablehlo.broadcast_in_dim %3131, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3134 = stablehlo.subtract %3114, %3133 : tensor<1x1x768xf32>
      %3135 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3136 = stablehlo.broadcast_in_dim %3135, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %3137 = stablehlo.add %3132, %3136 : tensor<1x1x1xf32>
      %3138 = stablehlo.rsqrt %3137 : tensor<1x1x1xf32>
      %3139 = stablehlo.reshape %iterArg_7 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3140 = stablehlo.convert %3139 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3141 = stablehlo.broadcast_in_dim %3138, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3142 = stablehlo.multiply %3141, %3140 : tensor<1x1x768xf32>
      %3143 = stablehlo.multiply %3134, %3142 : tensor<1x1x768xf32>
      %3144 = stablehlo.reshape %iterArg_8 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3145 = stablehlo.convert %3144 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3146 = stablehlo.add %3143, %3145 : tensor<1x1x768xf32>
      %3147 = stablehlo.transpose %iterArg_9, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %3148 = stablehlo.convert %3147 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %3149 = stablehlo.dot_general %3146, %3148, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %3150 = stablehlo.convert %iterArg_10 : (tensor<3072xf16>) -> tensor<3072xf32>
      %3151 = stablehlo.broadcast_in_dim %3150, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %3152 = stablehlo.add %3149, %3151 : tensor<1x1x3072xf32>
      %3153 = stablehlo.multiply %3152, %3152 : tensor<1x1x3072xf32>
      %3154 = stablehlo.multiply %3152, %3153 : tensor<1x1x3072xf32>
      %3155 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
      %3156 = stablehlo.broadcast_in_dim %3155, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3157 = stablehlo.multiply %3156, %3154 : tensor<1x1x3072xf32>
      %3158 = stablehlo.add %3152, %3157 : tensor<1x1x3072xf32>
      %3159 = stablehlo.constant dense<0.797884583> : tensor<f32>
      %3160 = stablehlo.broadcast_in_dim %3159, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3161 = stablehlo.multiply %3160, %3158 : tensor<1x1x3072xf32>
      %3162 = stablehlo.tanh %3161 : tensor<1x1x3072xf32>
      %3163 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %3164 = stablehlo.broadcast_in_dim %3163, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3165 = stablehlo.add %3164, %3162 : tensor<1x1x3072xf32>
      %3166 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %3167 = stablehlo.broadcast_in_dim %3166, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3168 = stablehlo.multiply %3167, %3165 : tensor<1x1x3072xf32>
      %3169 = stablehlo.multiply %3152, %3168 : tensor<1x1x3072xf32>
      %3170 = stablehlo.transpose %iterArg_11, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %3171 = stablehlo.convert %3170 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %3172 = stablehlo.dot_general %3169, %3171, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %3173 = stablehlo.convert %iterArg_12 : (tensor<768xf16>) -> tensor<768xf32>
      %3174 = stablehlo.broadcast_in_dim %3173, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3175 = stablehlo.add %3172, %3174 : tensor<1x1x768xf32>
      %3176 = stablehlo.add %3114, %3175 : tensor<1x1x768xf32>
      %3177 = stablehlo.multiply %3176, %3176 : tensor<1x1x768xf32>
      %3178 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3179 = stablehlo.reduce(%3177 init: %3178) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3180 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3181 = stablehlo.broadcast_in_dim %3180, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3182 = stablehlo.divide %3179, %3181 : tensor<1x1xf32>
      %3183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3184 = stablehlo.reduce(%3176 init: %3183) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3185 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3186 = stablehlo.broadcast_in_dim %3185, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3187 = stablehlo.divide %3184, %3186 : tensor<1x1xf32>
      %3188 = stablehlo.multiply %3187, %3187 : tensor<1x1xf32>
      %3189 = stablehlo.subtract %3182, %3188 : tensor<1x1xf32>
      %3190 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3191 = stablehlo.broadcast_in_dim %3190, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3192 = stablehlo.maximum %3191, %3189 : tensor<1x1xf32>
      %3193 = stablehlo.broadcast_in_dim %3187, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3194 = stablehlo.broadcast_in_dim %3192, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3195 = stablehlo.broadcast_in_dim %3193, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3196 = stablehlo.subtract %3176, %3195 : tensor<1x1x768xf32>
      %3197 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3198 = stablehlo.broadcast_in_dim %3197, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %3199 = stablehlo.add %3194, %3198 : tensor<1x1x1xf32>
      %3200 = stablehlo.rsqrt %3199 : tensor<1x1x1xf32>
      %3201 = stablehlo.reshape %iterArg_13 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3202 = stablehlo.convert %3201 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3203 = stablehlo.broadcast_in_dim %3200, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3204 = stablehlo.multiply %3203, %3202 : tensor<1x1x768xf32>
      %3205 = stablehlo.multiply %3196, %3204 : tensor<1x1x768xf32>
      %3206 = stablehlo.reshape %iterArg_14 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3207 = stablehlo.convert %3206 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3208 = stablehlo.add %3205, %3207 : tensor<1x1x768xf32>
      %3209 = stablehlo.constant dense<true> : tensor<i1>
      %3210 = stablehlo.broadcast_in_dim %3209, dims = [] : (tensor<i1>) -> tensor<1x1024xi1>
      %3211 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %3212 = stablehlo.broadcast_in_dim %3211, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %3213 = stablehlo.broadcast_in_dim %3212, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %3214 = stablehlo.broadcast_in_dim %3212, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %3215 = stablehlo.broadcast_in_dim %3213, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %3216 = stablehlo.broadcast_in_dim %3214, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %3217 = stablehlo.compare  GE, %3215, %3216,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %3218 = stablehlo.broadcast_in_dim %3217, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %3219 = stablehlo.transpose %iterArg_15, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %3220 = stablehlo.convert %3219 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %3221 = stablehlo.dot_general %3208, %3220, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32>
      %3222 = stablehlo.convert %iterArg_16 : (tensor<2304xf16>) -> tensor<2304xf32>
      %3223 = stablehlo.broadcast_in_dim %3222, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %3224 = stablehlo.add %3221, %3223 : tensor<1x1x2304xf32>
      %3225 = stablehlo.slice %3224 [0:1, 0:1, 0:768] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3226 = stablehlo.slice %3224 [0:1, 0:1, 768:1536] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3227 = stablehlo.slice %3224 [0:1, 0:1, 1536:2304] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3228 = stablehlo.reshape %3225 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3229 = stablehlo.reshape %3226 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3230 = stablehlo.reshape %3227 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3231 = stablehlo.constant dense<0> : tensor<i32>
      %3232 = stablehlo.compare  LT, %iterArg_157, %3231,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3233 = stablehlo.constant dense<1024> : tensor<i32>
      %3234 = stablehlo.add %iterArg_157, %3233 : tensor<i32>
      %3235 = stablehlo.select %3232, %3234, %iterArg_157 : tensor<i1>, tensor<i32>
      %3236 = stablehlo.constant dense<0> : tensor<i32>
      %3237 = stablehlo.constant dense<0> : tensor<i32>
      %3238 = stablehlo.constant dense<0> : tensor<i32>
      %3239 = stablehlo.dynamic_slice %3218, %3236, %3237, %3235, %3238, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %3240 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %3241 = stablehlo.constant dense<0> : tensor<i32>
      %3242 = stablehlo.broadcast_in_dim %3241, dims = [] : (tensor<i32>) -> tensor<1x1x1x20xi32>
      %3243 = stablehlo.compare  NE, %3240, %3242,  SIGNED : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %3244 = stablehlo.and %3243, %3239 : tensor<1x1x1x20xi1>
      %3245 = stablehlo.convert %3244 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %3246 = stablehlo.constant dense<0> : tensor<i32>
      %3247 = stablehlo.compare  LT, %iterArg_157, %3246,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3248 = stablehlo.constant dense<20> : tensor<i32>
      %3249 = stablehlo.add %iterArg_157, %3248 : tensor<i32>
      %3250 = stablehlo.select %3247, %3249, %iterArg_157 : tensor<i1>, tensor<i32>
      %3251 = stablehlo.constant dense<0> : tensor<i32>
      %3252 = stablehlo.constant dense<0> : tensor<i32>
      %3253 = stablehlo.constant dense<0> : tensor<i32>
      %3254 = stablehlo.dynamic_update_slice %iterArg_158, %3229, %3251, %3250, %3252, %3253 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %3255 = stablehlo.constant dense<0> : tensor<i32>
      %3256 = stablehlo.compare  LT, %iterArg_157, %3255,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3257 = stablehlo.constant dense<20> : tensor<i32>
      %3258 = stablehlo.add %iterArg_157, %3257 : tensor<i32>
      %3259 = stablehlo.select %3256, %3258, %iterArg_157 : tensor<i1>, tensor<i32>
      %3260 = stablehlo.constant dense<0> : tensor<i32>
      %3261 = stablehlo.constant dense<0> : tensor<i32>
      %3262 = stablehlo.constant dense<0> : tensor<i32>
      %3263 = stablehlo.dynamic_update_slice %iterArg_159, %3230, %3260, %3259, %3261, %3262 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %3264 = stablehlo.constant dense<1> : tensor<i32>
      %3265 = stablehlo.add %iterArg_157, %3264 : tensor<i32>
      %3266 = stablehlo.iota dim = 0 : tensor<20xi32>
      %3267 = stablehlo.constant dense<1> : tensor<i32>
      %3268 = stablehlo.add %iterArg_157, %3267 : tensor<i32>
      %3269 = stablehlo.broadcast_in_dim %3268, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %3270 = stablehlo.compare  LT, %3266, %3269,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %3271 = stablehlo.broadcast_in_dim %3270, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %3272 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3273 = stablehlo.broadcast_in_dim %3272, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3274 = stablehlo.compare  NE, %3245, %3273,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %3275 = stablehlo.and %3271, %3274 : tensor<1x1x1x20xi1>
      %3276 = stablehlo.convert %3275 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %3277 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3278 = stablehlo.broadcast_in_dim %3277, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3279 = stablehlo.compare  GT, %3276, %3278,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %3280 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3281 = stablehlo.broadcast_in_dim %3280, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3282 = stablehlo.convert %3281 : tensor<1x1x1x20xf32>
      %3283 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %3284 = stablehlo.broadcast_in_dim %3283, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3285 = stablehlo.select %3279, %3282, %3284 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %3286 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %3287 = stablehlo.sqrt %3286 : tensor<f32>
      %3288 = stablehlo.convert %3287 : tensor<f32>
      %3289 = stablehlo.broadcast_in_dim %3288, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %3290 = stablehlo.divide %3228, %3289 : tensor<1x1x12x64xf32>
      %3291 = stablehlo.dot_general %3290, %3254, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %3292 = stablehlo.broadcast_in_dim %3285, dims = [0, 1, 2, 3] : (tensor<1x1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %3293 = stablehlo.add %3291, %3292 : tensor<1x12x1x20xf32>
      %3294 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %3295 = stablehlo.reduce(%3293 init: %3294) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %3296 = stablehlo.broadcast_in_dim %3295, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %3297 = stablehlo.broadcast_in_dim %3296, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %3298 = stablehlo.subtract %3293, %3297 : tensor<1x12x1x20xf32>
      %3299 = stablehlo.exponential %3298 : tensor<1x12x1x20xf32>
      %3300 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3301 = stablehlo.reduce(%3299 init: %3300) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %3302 = stablehlo.broadcast_in_dim %3301, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %3303 = stablehlo.broadcast_in_dim %3302, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %3304 = stablehlo.divide %3299, %3303 : tensor<1x12x1x20xf32>
      %3305 = stablehlo.dot_general %3263, %3304, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %3306 = stablehlo.transpose %3305, dims = [0, 3, 1, 2] : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %3307 = stablehlo.reshape %3306 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %3308 = stablehlo.transpose %iterArg_17, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %3309 = stablehlo.convert %3308 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %3310 = stablehlo.dot_general %3307, %3309, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %3311 = stablehlo.convert %iterArg_18 : (tensor<768xf16>) -> tensor<768xf32>
      %3312 = stablehlo.broadcast_in_dim %3311, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3313 = stablehlo.add %3310, %3312 : tensor<1x1x768xf32>
      %3314 = stablehlo.add %3313, %3176 : tensor<1x1x768xf32>
      %3315 = stablehlo.multiply %3314, %3314 : tensor<1x1x768xf32>
      %3316 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3317 = stablehlo.reduce(%3315 init: %3316) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3318 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3319 = stablehlo.broadcast_in_dim %3318, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3320 = stablehlo.divide %3317, %3319 : tensor<1x1xf32>
      %3321 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3322 = stablehlo.reduce(%3314 init: %3321) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3323 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3324 = stablehlo.broadcast_in_dim %3323, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3325 = stablehlo.divide %3322, %3324 : tensor<1x1xf32>
      %3326 = stablehlo.multiply %3325, %3325 : tensor<1x1xf32>
      %3327 = stablehlo.subtract %3320, %3326 : tensor<1x1xf32>
      %3328 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3329 = stablehlo.broadcast_in_dim %3328, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3330 = stablehlo.maximum %3329, %3327 : tensor<1x1xf32>
      %3331 = stablehlo.broadcast_in_dim %3325, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3332 = stablehlo.broadcast_in_dim %3330, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3333 = stablehlo.broadcast_in_dim %3331, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3334 = stablehlo.subtract %3314, %3333 : tensor<1x1x768xf32>
      %3335 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3336 = stablehlo.broadcast_in_dim %3335, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %3337 = stablehlo.add %3332, %3336 : tensor<1x1x1xf32>
      %3338 = stablehlo.rsqrt %3337 : tensor<1x1x1xf32>
      %3339 = stablehlo.reshape %iterArg_19 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3340 = stablehlo.convert %3339 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3341 = stablehlo.broadcast_in_dim %3338, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3342 = stablehlo.multiply %3341, %3340 : tensor<1x1x768xf32>
      %3343 = stablehlo.multiply %3334, %3342 : tensor<1x1x768xf32>
      %3344 = stablehlo.reshape %iterArg_20 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3345 = stablehlo.convert %3344 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3346 = stablehlo.add %3343, %3345 : tensor<1x1x768xf32>
      %3347 = stablehlo.transpose %iterArg_21, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %3348 = stablehlo.convert %3347 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %3349 = stablehlo.dot_general %3346, %3348, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %3350 = stablehlo.convert %iterArg_22 : (tensor<3072xf16>) -> tensor<3072xf32>
      %3351 = stablehlo.broadcast_in_dim %3350, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %3352 = stablehlo.add %3349, %3351 : tensor<1x1x3072xf32>
      %3353 = stablehlo.multiply %3352, %3352 : tensor<1x1x3072xf32>
      %3354 = stablehlo.multiply %3352, %3353 : tensor<1x1x3072xf32>
      %3355 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
      %3356 = stablehlo.broadcast_in_dim %3355, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3357 = stablehlo.multiply %3356, %3354 : tensor<1x1x3072xf32>
      %3358 = stablehlo.add %3352, %3357 : tensor<1x1x3072xf32>
      %3359 = stablehlo.constant dense<0.797884583> : tensor<f32>
      %3360 = stablehlo.broadcast_in_dim %3359, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3361 = stablehlo.multiply %3360, %3358 : tensor<1x1x3072xf32>
      %3362 = stablehlo.tanh %3361 : tensor<1x1x3072xf32>
      %3363 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %3364 = stablehlo.broadcast_in_dim %3363, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3365 = stablehlo.add %3364, %3362 : tensor<1x1x3072xf32>
      %3366 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %3367 = stablehlo.broadcast_in_dim %3366, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3368 = stablehlo.multiply %3367, %3365 : tensor<1x1x3072xf32>
      %3369 = stablehlo.multiply %3352, %3368 : tensor<1x1x3072xf32>
      %3370 = stablehlo.transpose %iterArg_23, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %3371 = stablehlo.convert %3370 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %3372 = stablehlo.dot_general %3369, %3371, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %3373 = stablehlo.convert %iterArg_24 : (tensor<768xf16>) -> tensor<768xf32>
      %3374 = stablehlo.broadcast_in_dim %3373, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3375 = stablehlo.add %3372, %3374 : tensor<1x1x768xf32>
      %3376 = stablehlo.add %3314, %3375 : tensor<1x1x768xf32>
      %3377 = stablehlo.multiply %3376, %3376 : tensor<1x1x768xf32>
      %3378 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3379 = stablehlo.reduce(%3377 init: %3378) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3380 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3381 = stablehlo.broadcast_in_dim %3380, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3382 = stablehlo.divide %3379, %3381 : tensor<1x1xf32>
      %3383 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3384 = stablehlo.reduce(%3376 init: %3383) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3385 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3386 = stablehlo.broadcast_in_dim %3385, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3387 = stablehlo.divide %3384, %3386 : tensor<1x1xf32>
      %3388 = stablehlo.multiply %3387, %3387 : tensor<1x1xf32>
      %3389 = stablehlo.subtract %3382, %3388 : tensor<1x1xf32>
      %3390 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3391 = stablehlo.broadcast_in_dim %3390, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3392 = stablehlo.maximum %3391, %3389 : tensor<1x1xf32>
      %3393 = stablehlo.broadcast_in_dim %3387, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3394 = stablehlo.broadcast_in_dim %3392, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3395 = stablehlo.broadcast_in_dim %3393, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3396 = stablehlo.subtract %3376, %3395 : tensor<1x1x768xf32>
      %3397 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3398 = stablehlo.broadcast_in_dim %3397, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %3399 = stablehlo.add %3394, %3398 : tensor<1x1x1xf32>
      %3400 = stablehlo.rsqrt %3399 : tensor<1x1x1xf32>
      %3401 = stablehlo.reshape %iterArg_25 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3402 = stablehlo.convert %3401 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3403 = stablehlo.broadcast_in_dim %3400, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3404 = stablehlo.multiply %3403, %3402 : tensor<1x1x768xf32>
      %3405 = stablehlo.multiply %3396, %3404 : tensor<1x1x768xf32>
      %3406 = stablehlo.reshape %iterArg_26 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3407 = stablehlo.convert %3406 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3408 = stablehlo.add %3405, %3407 : tensor<1x1x768xf32>
      %3409 = stablehlo.constant dense<true> : tensor<i1>
      %3410 = stablehlo.broadcast_in_dim %3409, dims = [] : (tensor<i1>) -> tensor<1x1024xi1>
      %3411 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %3412 = stablehlo.broadcast_in_dim %3411, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %3413 = stablehlo.broadcast_in_dim %3412, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %3414 = stablehlo.broadcast_in_dim %3412, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %3415 = stablehlo.broadcast_in_dim %3413, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %3416 = stablehlo.broadcast_in_dim %3414, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %3417 = stablehlo.compare  GE, %3415, %3416,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %3418 = stablehlo.broadcast_in_dim %3417, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %3419 = stablehlo.transpose %iterArg_27, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %3420 = stablehlo.convert %3419 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %3421 = stablehlo.dot_general %3408, %3420, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32>
      %3422 = stablehlo.convert %iterArg_28 : (tensor<2304xf16>) -> tensor<2304xf32>
      %3423 = stablehlo.broadcast_in_dim %3422, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %3424 = stablehlo.add %3421, %3423 : tensor<1x1x2304xf32>
      %3425 = stablehlo.slice %3424 [0:1, 0:1, 0:768] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3426 = stablehlo.slice %3424 [0:1, 0:1, 768:1536] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3427 = stablehlo.slice %3424 [0:1, 0:1, 1536:2304] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3428 = stablehlo.reshape %3425 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3429 = stablehlo.reshape %3426 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3430 = stablehlo.reshape %3427 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3431 = stablehlo.constant dense<0> : tensor<i32>
      %3432 = stablehlo.compare  LT, %iterArg_166, %3431,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3433 = stablehlo.constant dense<1024> : tensor<i32>
      %3434 = stablehlo.add %iterArg_166, %3433 : tensor<i32>
      %3435 = stablehlo.select %3432, %3434, %iterArg_166 : tensor<i1>, tensor<i32>
      %3436 = stablehlo.constant dense<0> : tensor<i32>
      %3437 = stablehlo.constant dense<0> : tensor<i32>
      %3438 = stablehlo.constant dense<0> : tensor<i32>
      %3439 = stablehlo.dynamic_slice %3418, %3436, %3437, %3435, %3438, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %3440 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %3441 = stablehlo.constant dense<0> : tensor<i32>
      %3442 = stablehlo.broadcast_in_dim %3441, dims = [] : (tensor<i32>) -> tensor<1x1x1x20xi32>
      %3443 = stablehlo.compare  NE, %3440, %3442,  SIGNED : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %3444 = stablehlo.and %3443, %3439 : tensor<1x1x1x20xi1>
      %3445 = stablehlo.convert %3444 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %3446 = stablehlo.constant dense<0> : tensor<i32>
      %3447 = stablehlo.compare  LT, %iterArg_166, %3446,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3448 = stablehlo.constant dense<20> : tensor<i32>
      %3449 = stablehlo.add %iterArg_166, %3448 : tensor<i32>
      %3450 = stablehlo.select %3447, %3449, %iterArg_166 : tensor<i1>, tensor<i32>
      %3451 = stablehlo.constant dense<0> : tensor<i32>
      %3452 = stablehlo.constant dense<0> : tensor<i32>
      %3453 = stablehlo.constant dense<0> : tensor<i32>
      %3454 = stablehlo.dynamic_update_slice %iterArg_167, %3429, %3451, %3450, %3452, %3453 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %3455 = stablehlo.constant dense<0> : tensor<i32>
      %3456 = stablehlo.compare  LT, %iterArg_166, %3455,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3457 = stablehlo.constant dense<20> : tensor<i32>
      %3458 = stablehlo.add %iterArg_166, %3457 : tensor<i32>
      %3459 = stablehlo.select %3456, %3458, %iterArg_166 : tensor<i1>, tensor<i32>
      %3460 = stablehlo.constant dense<0> : tensor<i32>
      %3461 = stablehlo.constant dense<0> : tensor<i32>
      %3462 = stablehlo.constant dense<0> : tensor<i32>
      %3463 = stablehlo.dynamic_update_slice %iterArg_168, %3430, %3460, %3459, %3461, %3462 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %3464 = stablehlo.constant dense<1> : tensor<i32>
      %3465 = stablehlo.add %iterArg_166, %3464 : tensor<i32>
      %3466 = stablehlo.iota dim = 0 : tensor<20xi32>
      %3467 = stablehlo.constant dense<1> : tensor<i32>
      %3468 = stablehlo.add %iterArg_166, %3467 : tensor<i32>
      %3469 = stablehlo.broadcast_in_dim %3468, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %3470 = stablehlo.compare  LT, %3466, %3469,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %3471 = stablehlo.broadcast_in_dim %3470, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %3472 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3473 = stablehlo.broadcast_in_dim %3472, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3474 = stablehlo.compare  NE, %3445, %3473,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %3475 = stablehlo.and %3471, %3474 : tensor<1x1x1x20xi1>
      %3476 = stablehlo.convert %3475 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %3477 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3478 = stablehlo.broadcast_in_dim %3477, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3479 = stablehlo.compare  GT, %3476, %3478,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %3480 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3481 = stablehlo.broadcast_in_dim %3480, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3482 = stablehlo.convert %3481 : tensor<1x1x1x20xf32>
      %3483 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %3484 = stablehlo.broadcast_in_dim %3483, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3485 = stablehlo.select %3479, %3482, %3484 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %3486 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %3487 = stablehlo.sqrt %3486 : tensor<f32>
      %3488 = stablehlo.convert %3487 : tensor<f32>
      %3489 = stablehlo.broadcast_in_dim %3488, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %3490 = stablehlo.divide %3428, %3489 : tensor<1x1x12x64xf32>
      %3491 = stablehlo.dot_general %3490, %3454, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %3492 = stablehlo.broadcast_in_dim %3485, dims = [0, 1, 2, 3] : (tensor<1x1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %3493 = stablehlo.add %3491, %3492 : tensor<1x12x1x20xf32>
      %3494 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %3495 = stablehlo.reduce(%3493 init: %3494) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %3496 = stablehlo.broadcast_in_dim %3495, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %3497 = stablehlo.broadcast_in_dim %3496, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %3498 = stablehlo.subtract %3493, %3497 : tensor<1x12x1x20xf32>
      %3499 = stablehlo.exponential %3498 : tensor<1x12x1x20xf32>
      %3500 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3501 = stablehlo.reduce(%3499 init: %3500) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %3502 = stablehlo.broadcast_in_dim %3501, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %3503 = stablehlo.broadcast_in_dim %3502, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %3504 = stablehlo.divide %3499, %3503 : tensor<1x12x1x20xf32>
      %3505 = stablehlo.dot_general %3463, %3504, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %3506 = stablehlo.transpose %3505, dims = [0, 3, 1, 2] : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %3507 = stablehlo.reshape %3506 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %3508 = stablehlo.transpose %iterArg_29, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %3509 = stablehlo.convert %3508 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %3510 = stablehlo.dot_general %3507, %3509, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %3511 = stablehlo.convert %iterArg_30 : (tensor<768xf16>) -> tensor<768xf32>
      %3512 = stablehlo.broadcast_in_dim %3511, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3513 = stablehlo.add %3510, %3512 : tensor<1x1x768xf32>
      %3514 = stablehlo.add %3513, %3376 : tensor<1x1x768xf32>
      %3515 = stablehlo.multiply %3514, %3514 : tensor<1x1x768xf32>
      %3516 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3517 = stablehlo.reduce(%3515 init: %3516) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3518 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3519 = stablehlo.broadcast_in_dim %3518, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3520 = stablehlo.divide %3517, %3519 : tensor<1x1xf32>
      %3521 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3522 = stablehlo.reduce(%3514 init: %3521) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3523 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3524 = stablehlo.broadcast_in_dim %3523, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3525 = stablehlo.divide %3522, %3524 : tensor<1x1xf32>
      %3526 = stablehlo.multiply %3525, %3525 : tensor<1x1xf32>
      %3527 = stablehlo.subtract %3520, %3526 : tensor<1x1xf32>
      %3528 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3529 = stablehlo.broadcast_in_dim %3528, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3530 = stablehlo.maximum %3529, %3527 : tensor<1x1xf32>
      %3531 = stablehlo.broadcast_in_dim %3525, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3532 = stablehlo.broadcast_in_dim %3530, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3533 = stablehlo.broadcast_in_dim %3531, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3534 = stablehlo.subtract %3514, %3533 : tensor<1x1x768xf32>
      %3535 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3536 = stablehlo.broadcast_in_dim %3535, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %3537 = stablehlo.add %3532, %3536 : tensor<1x1x1xf32>
      %3538 = stablehlo.rsqrt %3537 : tensor<1x1x1xf32>
      %3539 = stablehlo.reshape %iterArg_31 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3540 = stablehlo.convert %3539 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3541 = stablehlo.broadcast_in_dim %3538, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3542 = stablehlo.multiply %3541, %3540 : tensor<1x1x768xf32>
      %3543 = stablehlo.multiply %3534, %3542 : tensor<1x1x768xf32>
      %3544 = stablehlo.reshape %iterArg_32 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3545 = stablehlo.convert %3544 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3546 = stablehlo.add %3543, %3545 : tensor<1x1x768xf32>
      %3547 = stablehlo.transpose %iterArg_33, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %3548 = stablehlo.convert %3547 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %3549 = stablehlo.dot_general %3546, %3548, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %3550 = stablehlo.convert %iterArg_34 : (tensor<3072xf16>) -> tensor<3072xf32>
      %3551 = stablehlo.broadcast_in_dim %3550, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %3552 = stablehlo.add %3549, %3551 : tensor<1x1x3072xf32>
      %3553 = stablehlo.multiply %3552, %3552 : tensor<1x1x3072xf32>
      %3554 = stablehlo.multiply %3552, %3553 : tensor<1x1x3072xf32>
      %3555 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
      %3556 = stablehlo.broadcast_in_dim %3555, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3557 = stablehlo.multiply %3556, %3554 : tensor<1x1x3072xf32>
      %3558 = stablehlo.add %3552, %3557 : tensor<1x1x3072xf32>
      %3559 = stablehlo.constant dense<0.797884583> : tensor<f32>
      %3560 = stablehlo.broadcast_in_dim %3559, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3561 = stablehlo.multiply %3560, %3558 : tensor<1x1x3072xf32>
      %3562 = stablehlo.tanh %3561 : tensor<1x1x3072xf32>
      %3563 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %3564 = stablehlo.broadcast_in_dim %3563, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3565 = stablehlo.add %3564, %3562 : tensor<1x1x3072xf32>
      %3566 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %3567 = stablehlo.broadcast_in_dim %3566, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3568 = stablehlo.multiply %3567, %3565 : tensor<1x1x3072xf32>
      %3569 = stablehlo.multiply %3552, %3568 : tensor<1x1x3072xf32>
      %3570 = stablehlo.transpose %iterArg_35, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %3571 = stablehlo.convert %3570 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %3572 = stablehlo.dot_general %3569, %3571, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %3573 = stablehlo.convert %iterArg_36 : (tensor<768xf16>) -> tensor<768xf32>
      %3574 = stablehlo.broadcast_in_dim %3573, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3575 = stablehlo.add %3572, %3574 : tensor<1x1x768xf32>
      %3576 = stablehlo.add %3514, %3575 : tensor<1x1x768xf32>
      %3577 = stablehlo.multiply %3576, %3576 : tensor<1x1x768xf32>
      %3578 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3579 = stablehlo.reduce(%3577 init: %3578) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3580 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3581 = stablehlo.broadcast_in_dim %3580, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3582 = stablehlo.divide %3579, %3581 : tensor<1x1xf32>
      %3583 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3584 = stablehlo.reduce(%3576 init: %3583) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3585 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3586 = stablehlo.broadcast_in_dim %3585, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3587 = stablehlo.divide %3584, %3586 : tensor<1x1xf32>
      %3588 = stablehlo.multiply %3587, %3587 : tensor<1x1xf32>
      %3589 = stablehlo.subtract %3582, %3588 : tensor<1x1xf32>
      %3590 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3591 = stablehlo.broadcast_in_dim %3590, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3592 = stablehlo.maximum %3591, %3589 : tensor<1x1xf32>
      %3593 = stablehlo.broadcast_in_dim %3587, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3594 = stablehlo.broadcast_in_dim %3592, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3595 = stablehlo.broadcast_in_dim %3593, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3596 = stablehlo.subtract %3576, %3595 : tensor<1x1x768xf32>
      %3597 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3598 = stablehlo.broadcast_in_dim %3597, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %3599 = stablehlo.add %3594, %3598 : tensor<1x1x1xf32>
      %3600 = stablehlo.rsqrt %3599 : tensor<1x1x1xf32>
      %3601 = stablehlo.reshape %iterArg_37 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3602 = stablehlo.convert %3601 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3603 = stablehlo.broadcast_in_dim %3600, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3604 = stablehlo.multiply %3603, %3602 : tensor<1x1x768xf32>
      %3605 = stablehlo.multiply %3596, %3604 : tensor<1x1x768xf32>
      %3606 = stablehlo.reshape %iterArg_38 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3607 = stablehlo.convert %3606 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3608 = stablehlo.add %3605, %3607 : tensor<1x1x768xf32>
      %3609 = stablehlo.constant dense<true> : tensor<i1>
      %3610 = stablehlo.broadcast_in_dim %3609, dims = [] : (tensor<i1>) -> tensor<1x1024xi1>
      %3611 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %3612 = stablehlo.broadcast_in_dim %3611, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %3613 = stablehlo.broadcast_in_dim %3612, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %3614 = stablehlo.broadcast_in_dim %3612, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %3615 = stablehlo.broadcast_in_dim %3613, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %3616 = stablehlo.broadcast_in_dim %3614, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %3617 = stablehlo.compare  GE, %3615, %3616,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %3618 = stablehlo.broadcast_in_dim %3617, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %3619 = stablehlo.transpose %iterArg_39, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %3620 = stablehlo.convert %3619 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %3621 = stablehlo.dot_general %3608, %3620, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32>
      %3622 = stablehlo.convert %iterArg_40 : (tensor<2304xf16>) -> tensor<2304xf32>
      %3623 = stablehlo.broadcast_in_dim %3622, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %3624 = stablehlo.add %3621, %3623 : tensor<1x1x2304xf32>
      %3625 = stablehlo.slice %3624 [0:1, 0:1, 0:768] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3626 = stablehlo.slice %3624 [0:1, 0:1, 768:1536] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3627 = stablehlo.slice %3624 [0:1, 0:1, 1536:2304] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3628 = stablehlo.reshape %3625 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3629 = stablehlo.reshape %3626 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3630 = stablehlo.reshape %3627 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3631 = stablehlo.constant dense<0> : tensor<i32>
      %3632 = stablehlo.compare  LT, %iterArg_169, %3631,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3633 = stablehlo.constant dense<1024> : tensor<i32>
      %3634 = stablehlo.add %iterArg_169, %3633 : tensor<i32>
      %3635 = stablehlo.select %3632, %3634, %iterArg_169 : tensor<i1>, tensor<i32>
      %3636 = stablehlo.constant dense<0> : tensor<i32>
      %3637 = stablehlo.constant dense<0> : tensor<i32>
      %3638 = stablehlo.constant dense<0> : tensor<i32>
      %3639 = stablehlo.dynamic_slice %3618, %3636, %3637, %3635, %3638, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %3640 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %3641 = stablehlo.constant dense<0> : tensor<i32>
      %3642 = stablehlo.broadcast_in_dim %3641, dims = [] : (tensor<i32>) -> tensor<1x1x1x20xi32>
      %3643 = stablehlo.compare  NE, %3640, %3642,  SIGNED : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %3644 = stablehlo.and %3643, %3639 : tensor<1x1x1x20xi1>
      %3645 = stablehlo.convert %3644 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %3646 = stablehlo.constant dense<0> : tensor<i32>
      %3647 = stablehlo.compare  LT, %iterArg_169, %3646,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3648 = stablehlo.constant dense<20> : tensor<i32>
      %3649 = stablehlo.add %iterArg_169, %3648 : tensor<i32>
      %3650 = stablehlo.select %3647, %3649, %iterArg_169 : tensor<i1>, tensor<i32>
      %3651 = stablehlo.constant dense<0> : tensor<i32>
      %3652 = stablehlo.constant dense<0> : tensor<i32>
      %3653 = stablehlo.constant dense<0> : tensor<i32>
      %3654 = stablehlo.dynamic_update_slice %iterArg_170, %3629, %3651, %3650, %3652, %3653 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %3655 = stablehlo.constant dense<0> : tensor<i32>
      %3656 = stablehlo.compare  LT, %iterArg_169, %3655,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3657 = stablehlo.constant dense<20> : tensor<i32>
      %3658 = stablehlo.add %iterArg_169, %3657 : tensor<i32>
      %3659 = stablehlo.select %3656, %3658, %iterArg_169 : tensor<i1>, tensor<i32>
      %3660 = stablehlo.constant dense<0> : tensor<i32>
      %3661 = stablehlo.constant dense<0> : tensor<i32>
      %3662 = stablehlo.constant dense<0> : tensor<i32>
      %3663 = stablehlo.dynamic_update_slice %iterArg_171, %3630, %3660, %3659, %3661, %3662 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %3664 = stablehlo.constant dense<1> : tensor<i32>
      %3665 = stablehlo.add %iterArg_169, %3664 : tensor<i32>
      %3666 = stablehlo.iota dim = 0 : tensor<20xi32>
      %3667 = stablehlo.constant dense<1> : tensor<i32>
      %3668 = stablehlo.add %iterArg_169, %3667 : tensor<i32>
      %3669 = stablehlo.broadcast_in_dim %3668, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %3670 = stablehlo.compare  LT, %3666, %3669,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %3671 = stablehlo.broadcast_in_dim %3670, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %3672 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3673 = stablehlo.broadcast_in_dim %3672, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3674 = stablehlo.compare  NE, %3645, %3673,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %3675 = stablehlo.and %3671, %3674 : tensor<1x1x1x20xi1>
      %3676 = stablehlo.convert %3675 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %3677 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3678 = stablehlo.broadcast_in_dim %3677, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3679 = stablehlo.compare  GT, %3676, %3678,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %3680 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3681 = stablehlo.broadcast_in_dim %3680, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3682 = stablehlo.convert %3681 : tensor<1x1x1x20xf32>
      %3683 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %3684 = stablehlo.broadcast_in_dim %3683, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3685 = stablehlo.select %3679, %3682, %3684 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %3686 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %3687 = stablehlo.sqrt %3686 : tensor<f32>
      %3688 = stablehlo.convert %3687 : tensor<f32>
      %3689 = stablehlo.broadcast_in_dim %3688, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %3690 = stablehlo.divide %3628, %3689 : tensor<1x1x12x64xf32>
      %3691 = stablehlo.dot_general %3690, %3654, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %3692 = stablehlo.broadcast_in_dim %3685, dims = [0, 1, 2, 3] : (tensor<1x1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %3693 = stablehlo.add %3691, %3692 : tensor<1x12x1x20xf32>
      %3694 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %3695 = stablehlo.reduce(%3693 init: %3694) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %3696 = stablehlo.broadcast_in_dim %3695, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %3697 = stablehlo.broadcast_in_dim %3696, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %3698 = stablehlo.subtract %3693, %3697 : tensor<1x12x1x20xf32>
      %3699 = stablehlo.exponential %3698 : tensor<1x12x1x20xf32>
      %3700 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3701 = stablehlo.reduce(%3699 init: %3700) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %3702 = stablehlo.broadcast_in_dim %3701, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %3703 = stablehlo.broadcast_in_dim %3702, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %3704 = stablehlo.divide %3699, %3703 : tensor<1x12x1x20xf32>
      %3705 = stablehlo.dot_general %3663, %3704, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %3706 = stablehlo.transpose %3705, dims = [0, 3, 1, 2] : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %3707 = stablehlo.reshape %3706 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %3708 = stablehlo.transpose %iterArg_41, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %3709 = stablehlo.convert %3708 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %3710 = stablehlo.dot_general %3707, %3709, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %3711 = stablehlo.convert %iterArg_42 : (tensor<768xf16>) -> tensor<768xf32>
      %3712 = stablehlo.broadcast_in_dim %3711, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3713 = stablehlo.add %3710, %3712 : tensor<1x1x768xf32>
      %3714 = stablehlo.add %3713, %3576 : tensor<1x1x768xf32>
      %3715 = stablehlo.multiply %3714, %3714 : tensor<1x1x768xf32>
      %3716 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3717 = stablehlo.reduce(%3715 init: %3716) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3718 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3719 = stablehlo.broadcast_in_dim %3718, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3720 = stablehlo.divide %3717, %3719 : tensor<1x1xf32>
      %3721 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3722 = stablehlo.reduce(%3714 init: %3721) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3723 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3724 = stablehlo.broadcast_in_dim %3723, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3725 = stablehlo.divide %3722, %3724 : tensor<1x1xf32>
      %3726 = stablehlo.multiply %3725, %3725 : tensor<1x1xf32>
      %3727 = stablehlo.subtract %3720, %3726 : tensor<1x1xf32>
      %3728 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3729 = stablehlo.broadcast_in_dim %3728, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3730 = stablehlo.maximum %3729, %3727 : tensor<1x1xf32>
      %3731 = stablehlo.broadcast_in_dim %3725, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3732 = stablehlo.broadcast_in_dim %3730, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3733 = stablehlo.broadcast_in_dim %3731, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3734 = stablehlo.subtract %3714, %3733 : tensor<1x1x768xf32>
      %3735 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3736 = stablehlo.broadcast_in_dim %3735, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %3737 = stablehlo.add %3732, %3736 : tensor<1x1x1xf32>
      %3738 = stablehlo.rsqrt %3737 : tensor<1x1x1xf32>
      %3739 = stablehlo.reshape %iterArg_43 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3740 = stablehlo.convert %3739 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3741 = stablehlo.broadcast_in_dim %3738, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3742 = stablehlo.multiply %3741, %3740 : tensor<1x1x768xf32>
      %3743 = stablehlo.multiply %3734, %3742 : tensor<1x1x768xf32>
      %3744 = stablehlo.reshape %iterArg_44 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3745 = stablehlo.convert %3744 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3746 = stablehlo.add %3743, %3745 : tensor<1x1x768xf32>
      %3747 = stablehlo.transpose %iterArg_45, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %3748 = stablehlo.convert %3747 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %3749 = stablehlo.dot_general %3746, %3748, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %3750 = stablehlo.convert %iterArg_46 : (tensor<3072xf16>) -> tensor<3072xf32>
      %3751 = stablehlo.broadcast_in_dim %3750, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %3752 = stablehlo.add %3749, %3751 : tensor<1x1x3072xf32>
      %3753 = stablehlo.multiply %3752, %3752 : tensor<1x1x3072xf32>
      %3754 = stablehlo.multiply %3752, %3753 : tensor<1x1x3072xf32>
      %3755 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
      %3756 = stablehlo.broadcast_in_dim %3755, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3757 = stablehlo.multiply %3756, %3754 : tensor<1x1x3072xf32>
      %3758 = stablehlo.add %3752, %3757 : tensor<1x1x3072xf32>
      %3759 = stablehlo.constant dense<0.797884583> : tensor<f32>
      %3760 = stablehlo.broadcast_in_dim %3759, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3761 = stablehlo.multiply %3760, %3758 : tensor<1x1x3072xf32>
      %3762 = stablehlo.tanh %3761 : tensor<1x1x3072xf32>
      %3763 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %3764 = stablehlo.broadcast_in_dim %3763, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3765 = stablehlo.add %3764, %3762 : tensor<1x1x3072xf32>
      %3766 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %3767 = stablehlo.broadcast_in_dim %3766, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3768 = stablehlo.multiply %3767, %3765 : tensor<1x1x3072xf32>
      %3769 = stablehlo.multiply %3752, %3768 : tensor<1x1x3072xf32>
      %3770 = stablehlo.transpose %iterArg_47, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %3771 = stablehlo.convert %3770 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %3772 = stablehlo.dot_general %3769, %3771, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %3773 = stablehlo.convert %iterArg_48 : (tensor<768xf16>) -> tensor<768xf32>
      %3774 = stablehlo.broadcast_in_dim %3773, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3775 = stablehlo.add %3772, %3774 : tensor<1x1x768xf32>
      %3776 = stablehlo.add %3714, %3775 : tensor<1x1x768xf32>
      %3777 = stablehlo.multiply %3776, %3776 : tensor<1x1x768xf32>
      %3778 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3779 = stablehlo.reduce(%3777 init: %3778) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3780 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3781 = stablehlo.broadcast_in_dim %3780, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3782 = stablehlo.divide %3779, %3781 : tensor<1x1xf32>
      %3783 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3784 = stablehlo.reduce(%3776 init: %3783) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3785 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3786 = stablehlo.broadcast_in_dim %3785, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3787 = stablehlo.divide %3784, %3786 : tensor<1x1xf32>
      %3788 = stablehlo.multiply %3787, %3787 : tensor<1x1xf32>
      %3789 = stablehlo.subtract %3782, %3788 : tensor<1x1xf32>
      %3790 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3791 = stablehlo.broadcast_in_dim %3790, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3792 = stablehlo.maximum %3791, %3789 : tensor<1x1xf32>
      %3793 = stablehlo.broadcast_in_dim %3787, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3794 = stablehlo.broadcast_in_dim %3792, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3795 = stablehlo.broadcast_in_dim %3793, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3796 = stablehlo.subtract %3776, %3795 : tensor<1x1x768xf32>
      %3797 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3798 = stablehlo.broadcast_in_dim %3797, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %3799 = stablehlo.add %3794, %3798 : tensor<1x1x1xf32>
      %3800 = stablehlo.rsqrt %3799 : tensor<1x1x1xf32>
      %3801 = stablehlo.reshape %iterArg_49 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3802 = stablehlo.convert %3801 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3803 = stablehlo.broadcast_in_dim %3800, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3804 = stablehlo.multiply %3803, %3802 : tensor<1x1x768xf32>
      %3805 = stablehlo.multiply %3796, %3804 : tensor<1x1x768xf32>
      %3806 = stablehlo.reshape %iterArg_50 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3807 = stablehlo.convert %3806 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3808 = stablehlo.add %3805, %3807 : tensor<1x1x768xf32>
      %3809 = stablehlo.constant dense<true> : tensor<i1>
      %3810 = stablehlo.broadcast_in_dim %3809, dims = [] : (tensor<i1>) -> tensor<1x1024xi1>
      %3811 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %3812 = stablehlo.broadcast_in_dim %3811, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %3813 = stablehlo.broadcast_in_dim %3812, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %3814 = stablehlo.broadcast_in_dim %3812, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %3815 = stablehlo.broadcast_in_dim %3813, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %3816 = stablehlo.broadcast_in_dim %3814, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %3817 = stablehlo.compare  GE, %3815, %3816,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %3818 = stablehlo.broadcast_in_dim %3817, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %3819 = stablehlo.transpose %iterArg_51, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %3820 = stablehlo.convert %3819 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %3821 = stablehlo.dot_general %3808, %3820, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32>
      %3822 = stablehlo.convert %iterArg_52 : (tensor<2304xf16>) -> tensor<2304xf32>
      %3823 = stablehlo.broadcast_in_dim %3822, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %3824 = stablehlo.add %3821, %3823 : tensor<1x1x2304xf32>
      %3825 = stablehlo.slice %3824 [0:1, 0:1, 0:768] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3826 = stablehlo.slice %3824 [0:1, 0:1, 768:1536] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3827 = stablehlo.slice %3824 [0:1, 0:1, 1536:2304] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %3828 = stablehlo.reshape %3825 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3829 = stablehlo.reshape %3826 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3830 = stablehlo.reshape %3827 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %3831 = stablehlo.constant dense<0> : tensor<i32>
      %3832 = stablehlo.compare  LT, %iterArg_172, %3831,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3833 = stablehlo.constant dense<1024> : tensor<i32>
      %3834 = stablehlo.add %iterArg_172, %3833 : tensor<i32>
      %3835 = stablehlo.select %3832, %3834, %iterArg_172 : tensor<i1>, tensor<i32>
      %3836 = stablehlo.constant dense<0> : tensor<i32>
      %3837 = stablehlo.constant dense<0> : tensor<i32>
      %3838 = stablehlo.constant dense<0> : tensor<i32>
      %3839 = stablehlo.dynamic_slice %3818, %3836, %3837, %3835, %3838, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %3840 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %3841 = stablehlo.constant dense<0> : tensor<i32>
      %3842 = stablehlo.broadcast_in_dim %3841, dims = [] : (tensor<i32>) -> tensor<1x1x1x20xi32>
      %3843 = stablehlo.compare  NE, %3840, %3842,  SIGNED : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %3844 = stablehlo.and %3843, %3839 : tensor<1x1x1x20xi1>
      %3845 = stablehlo.convert %3844 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %3846 = stablehlo.constant dense<0> : tensor<i32>
      %3847 = stablehlo.compare  LT, %iterArg_172, %3846,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3848 = stablehlo.constant dense<20> : tensor<i32>
      %3849 = stablehlo.add %iterArg_172, %3848 : tensor<i32>
      %3850 = stablehlo.select %3847, %3849, %iterArg_172 : tensor<i1>, tensor<i32>
      %3851 = stablehlo.constant dense<0> : tensor<i32>
      %3852 = stablehlo.constant dense<0> : tensor<i32>
      %3853 = stablehlo.constant dense<0> : tensor<i32>
      %3854 = stablehlo.dynamic_update_slice %iterArg_173, %3829, %3851, %3850, %3852, %3853 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %3855 = stablehlo.constant dense<0> : tensor<i32>
      %3856 = stablehlo.compare  LT, %iterArg_172, %3855,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3857 = stablehlo.constant dense<20> : tensor<i32>
      %3858 = stablehlo.add %iterArg_172, %3857 : tensor<i32>
      %3859 = stablehlo.select %3856, %3858, %iterArg_172 : tensor<i1>, tensor<i32>
      %3860 = stablehlo.constant dense<0> : tensor<i32>
      %3861 = stablehlo.constant dense<0> : tensor<i32>
      %3862 = stablehlo.constant dense<0> : tensor<i32>
      %3863 = stablehlo.dynamic_update_slice %iterArg_174, %3830, %3860, %3859, %3861, %3862 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %3864 = stablehlo.constant dense<1> : tensor<i32>
      %3865 = stablehlo.add %iterArg_172, %3864 : tensor<i32>
      %3866 = stablehlo.iota dim = 0 : tensor<20xi32>
      %3867 = stablehlo.constant dense<1> : tensor<i32>
      %3868 = stablehlo.add %iterArg_172, %3867 : tensor<i32>
      %3869 = stablehlo.broadcast_in_dim %3868, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %3870 = stablehlo.compare  LT, %3866, %3869,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %3871 = stablehlo.broadcast_in_dim %3870, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %3872 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3873 = stablehlo.broadcast_in_dim %3872, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3874 = stablehlo.compare  NE, %3845, %3873,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %3875 = stablehlo.and %3871, %3874 : tensor<1x1x1x20xi1>
      %3876 = stablehlo.convert %3875 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %3877 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3878 = stablehlo.broadcast_in_dim %3877, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3879 = stablehlo.compare  GT, %3876, %3878,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %3880 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3881 = stablehlo.broadcast_in_dim %3880, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3882 = stablehlo.convert %3881 : tensor<1x1x1x20xf32>
      %3883 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %3884 = stablehlo.broadcast_in_dim %3883, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %3885 = stablehlo.select %3879, %3882, %3884 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %3886 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %3887 = stablehlo.sqrt %3886 : tensor<f32>
      %3888 = stablehlo.convert %3887 : tensor<f32>
      %3889 = stablehlo.broadcast_in_dim %3888, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %3890 = stablehlo.divide %3828, %3889 : tensor<1x1x12x64xf32>
      %3891 = stablehlo.dot_general %3890, %3854, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %3892 = stablehlo.broadcast_in_dim %3885, dims = [0, 1, 2, 3] : (tensor<1x1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %3893 = stablehlo.add %3891, %3892 : tensor<1x12x1x20xf32>
      %3894 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %3895 = stablehlo.reduce(%3893 init: %3894) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %3896 = stablehlo.broadcast_in_dim %3895, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %3897 = stablehlo.broadcast_in_dim %3896, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %3898 = stablehlo.subtract %3893, %3897 : tensor<1x12x1x20xf32>
      %3899 = stablehlo.exponential %3898 : tensor<1x12x1x20xf32>
      %3900 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3901 = stablehlo.reduce(%3899 init: %3900) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %3902 = stablehlo.broadcast_in_dim %3901, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %3903 = stablehlo.broadcast_in_dim %3902, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %3904 = stablehlo.divide %3899, %3903 : tensor<1x12x1x20xf32>
      %3905 = stablehlo.dot_general %3863, %3904, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %3906 = stablehlo.transpose %3905, dims = [0, 3, 1, 2] : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %3907 = stablehlo.reshape %3906 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %3908 = stablehlo.transpose %iterArg_53, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %3909 = stablehlo.convert %3908 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %3910 = stablehlo.dot_general %3907, %3909, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %3911 = stablehlo.convert %iterArg_54 : (tensor<768xf16>) -> tensor<768xf32>
      %3912 = stablehlo.broadcast_in_dim %3911, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3913 = stablehlo.add %3910, %3912 : tensor<1x1x768xf32>
      %3914 = stablehlo.add %3913, %3776 : tensor<1x1x768xf32>
      %3915 = stablehlo.multiply %3914, %3914 : tensor<1x1x768xf32>
      %3916 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3917 = stablehlo.reduce(%3915 init: %3916) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3918 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3919 = stablehlo.broadcast_in_dim %3918, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3920 = stablehlo.divide %3917, %3919 : tensor<1x1xf32>
      %3921 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3922 = stablehlo.reduce(%3914 init: %3921) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3923 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3924 = stablehlo.broadcast_in_dim %3923, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3925 = stablehlo.divide %3922, %3924 : tensor<1x1xf32>
      %3926 = stablehlo.multiply %3925, %3925 : tensor<1x1xf32>
      %3927 = stablehlo.subtract %3920, %3926 : tensor<1x1xf32>
      %3928 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3929 = stablehlo.broadcast_in_dim %3928, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3930 = stablehlo.maximum %3929, %3927 : tensor<1x1xf32>
      %3931 = stablehlo.broadcast_in_dim %3925, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3932 = stablehlo.broadcast_in_dim %3930, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3933 = stablehlo.broadcast_in_dim %3931, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3934 = stablehlo.subtract %3914, %3933 : tensor<1x1x768xf32>
      %3935 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3936 = stablehlo.broadcast_in_dim %3935, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %3937 = stablehlo.add %3932, %3936 : tensor<1x1x1xf32>
      %3938 = stablehlo.rsqrt %3937 : tensor<1x1x1xf32>
      %3939 = stablehlo.reshape %iterArg_55 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3940 = stablehlo.convert %3939 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3941 = stablehlo.broadcast_in_dim %3938, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3942 = stablehlo.multiply %3941, %3940 : tensor<1x1x768xf32>
      %3943 = stablehlo.multiply %3934, %3942 : tensor<1x1x768xf32>
      %3944 = stablehlo.reshape %iterArg_56 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3945 = stablehlo.convert %3944 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3946 = stablehlo.add %3943, %3945 : tensor<1x1x768xf32>
      %3947 = stablehlo.transpose %iterArg_57, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %3948 = stablehlo.convert %3947 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %3949 = stablehlo.dot_general %3946, %3948, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %3950 = stablehlo.convert %iterArg_58 : (tensor<3072xf16>) -> tensor<3072xf32>
      %3951 = stablehlo.broadcast_in_dim %3950, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %3952 = stablehlo.add %3949, %3951 : tensor<1x1x3072xf32>
      %3953 = stablehlo.multiply %3952, %3952 : tensor<1x1x3072xf32>
      %3954 = stablehlo.multiply %3952, %3953 : tensor<1x1x3072xf32>
      %3955 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
      %3956 = stablehlo.broadcast_in_dim %3955, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3957 = stablehlo.multiply %3956, %3954 : tensor<1x1x3072xf32>
      %3958 = stablehlo.add %3952, %3957 : tensor<1x1x3072xf32>
      %3959 = stablehlo.constant dense<0.797884583> : tensor<f32>
      %3960 = stablehlo.broadcast_in_dim %3959, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3961 = stablehlo.multiply %3960, %3958 : tensor<1x1x3072xf32>
      %3962 = stablehlo.tanh %3961 : tensor<1x1x3072xf32>
      %3963 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %3964 = stablehlo.broadcast_in_dim %3963, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3965 = stablehlo.add %3964, %3962 : tensor<1x1x3072xf32>
      %3966 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %3967 = stablehlo.broadcast_in_dim %3966, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %3968 = stablehlo.multiply %3967, %3965 : tensor<1x1x3072xf32>
      %3969 = stablehlo.multiply %3952, %3968 : tensor<1x1x3072xf32>
      %3970 = stablehlo.transpose %iterArg_59, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %3971 = stablehlo.convert %3970 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %3972 = stablehlo.dot_general %3969, %3971, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %3973 = stablehlo.convert %iterArg_60 : (tensor<768xf16>) -> tensor<768xf32>
      %3974 = stablehlo.broadcast_in_dim %3973, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3975 = stablehlo.add %3972, %3974 : tensor<1x1x768xf32>
      %3976 = stablehlo.add %3914, %3975 : tensor<1x1x768xf32>
      %3977 = stablehlo.multiply %3976, %3976 : tensor<1x1x768xf32>
      %3978 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3979 = stablehlo.reduce(%3977 init: %3978) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3980 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3981 = stablehlo.broadcast_in_dim %3980, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3982 = stablehlo.divide %3979, %3981 : tensor<1x1xf32>
      %3983 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3984 = stablehlo.reduce(%3976 init: %3983) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %3985 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %3986 = stablehlo.broadcast_in_dim %3985, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3987 = stablehlo.divide %3984, %3986 : tensor<1x1xf32>
      %3988 = stablehlo.multiply %3987, %3987 : tensor<1x1xf32>
      %3989 = stablehlo.subtract %3982, %3988 : tensor<1x1xf32>
      %3990 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3991 = stablehlo.broadcast_in_dim %3990, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %3992 = stablehlo.maximum %3991, %3989 : tensor<1x1xf32>
      %3993 = stablehlo.broadcast_in_dim %3987, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3994 = stablehlo.broadcast_in_dim %3992, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %3995 = stablehlo.broadcast_in_dim %3993, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %3996 = stablehlo.subtract %3976, %3995 : tensor<1x1x768xf32>
      %3997 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3998 = stablehlo.broadcast_in_dim %3997, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %3999 = stablehlo.add %3994, %3998 : tensor<1x1x1xf32>
      %4000 = stablehlo.rsqrt %3999 : tensor<1x1x1xf32>
      %4001 = stablehlo.reshape %iterArg_61 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4002 = stablehlo.convert %4001 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4003 = stablehlo.broadcast_in_dim %4000, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4004 = stablehlo.multiply %4003, %4002 : tensor<1x1x768xf32>
      %4005 = stablehlo.multiply %3996, %4004 : tensor<1x1x768xf32>
      %4006 = stablehlo.reshape %iterArg_62 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4007 = stablehlo.convert %4006 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4008 = stablehlo.add %4005, %4007 : tensor<1x1x768xf32>
      %4009 = stablehlo.constant dense<true> : tensor<i1>
      %4010 = stablehlo.broadcast_in_dim %4009, dims = [] : (tensor<i1>) -> tensor<1x1024xi1>
      %4011 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %4012 = stablehlo.broadcast_in_dim %4011, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %4013 = stablehlo.broadcast_in_dim %4012, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %4014 = stablehlo.broadcast_in_dim %4012, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %4015 = stablehlo.broadcast_in_dim %4013, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %4016 = stablehlo.broadcast_in_dim %4014, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %4017 = stablehlo.compare  GE, %4015, %4016,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %4018 = stablehlo.broadcast_in_dim %4017, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %4019 = stablehlo.transpose %iterArg_63, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %4020 = stablehlo.convert %4019 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %4021 = stablehlo.dot_general %4008, %4020, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32>
      %4022 = stablehlo.convert %iterArg_64 : (tensor<2304xf16>) -> tensor<2304xf32>
      %4023 = stablehlo.broadcast_in_dim %4022, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %4024 = stablehlo.add %4021, %4023 : tensor<1x1x2304xf32>
      %4025 = stablehlo.slice %4024 [0:1, 0:1, 0:768] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4026 = stablehlo.slice %4024 [0:1, 0:1, 768:1536] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4027 = stablehlo.slice %4024 [0:1, 0:1, 1536:2304] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4028 = stablehlo.reshape %4025 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4029 = stablehlo.reshape %4026 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4030 = stablehlo.reshape %4027 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4031 = stablehlo.constant dense<0> : tensor<i32>
      %4032 = stablehlo.compare  LT, %iterArg_175, %4031,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4033 = stablehlo.constant dense<1024> : tensor<i32>
      %4034 = stablehlo.add %iterArg_175, %4033 : tensor<i32>
      %4035 = stablehlo.select %4032, %4034, %iterArg_175 : tensor<i1>, tensor<i32>
      %4036 = stablehlo.constant dense<0> : tensor<i32>
      %4037 = stablehlo.constant dense<0> : tensor<i32>
      %4038 = stablehlo.constant dense<0> : tensor<i32>
      %4039 = stablehlo.dynamic_slice %4018, %4036, %4037, %4035, %4038, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %4040 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %4041 = stablehlo.constant dense<0> : tensor<i32>
      %4042 = stablehlo.broadcast_in_dim %4041, dims = [] : (tensor<i32>) -> tensor<1x1x1x20xi32>
      %4043 = stablehlo.compare  NE, %4040, %4042,  SIGNED : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %4044 = stablehlo.and %4043, %4039 : tensor<1x1x1x20xi1>
      %4045 = stablehlo.convert %4044 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %4046 = stablehlo.constant dense<0> : tensor<i32>
      %4047 = stablehlo.compare  LT, %iterArg_175, %4046,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4048 = stablehlo.constant dense<20> : tensor<i32>
      %4049 = stablehlo.add %iterArg_175, %4048 : tensor<i32>
      %4050 = stablehlo.select %4047, %4049, %iterArg_175 : tensor<i1>, tensor<i32>
      %4051 = stablehlo.constant dense<0> : tensor<i32>
      %4052 = stablehlo.constant dense<0> : tensor<i32>
      %4053 = stablehlo.constant dense<0> : tensor<i32>
      %4054 = stablehlo.dynamic_update_slice %iterArg_176, %4029, %4051, %4050, %4052, %4053 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %4055 = stablehlo.constant dense<0> : tensor<i32>
      %4056 = stablehlo.compare  LT, %iterArg_175, %4055,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4057 = stablehlo.constant dense<20> : tensor<i32>
      %4058 = stablehlo.add %iterArg_175, %4057 : tensor<i32>
      %4059 = stablehlo.select %4056, %4058, %iterArg_175 : tensor<i1>, tensor<i32>
      %4060 = stablehlo.constant dense<0> : tensor<i32>
      %4061 = stablehlo.constant dense<0> : tensor<i32>
      %4062 = stablehlo.constant dense<0> : tensor<i32>
      %4063 = stablehlo.dynamic_update_slice %iterArg_177, %4030, %4060, %4059, %4061, %4062 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %4064 = stablehlo.constant dense<1> : tensor<i32>
      %4065 = stablehlo.add %iterArg_175, %4064 : tensor<i32>
      %4066 = stablehlo.iota dim = 0 : tensor<20xi32>
      %4067 = stablehlo.constant dense<1> : tensor<i32>
      %4068 = stablehlo.add %iterArg_175, %4067 : tensor<i32>
      %4069 = stablehlo.broadcast_in_dim %4068, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %4070 = stablehlo.compare  LT, %4066, %4069,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %4071 = stablehlo.broadcast_in_dim %4070, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %4072 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4073 = stablehlo.broadcast_in_dim %4072, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4074 = stablehlo.compare  NE, %4045, %4073,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %4075 = stablehlo.and %4071, %4074 : tensor<1x1x1x20xi1>
      %4076 = stablehlo.convert %4075 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %4077 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4078 = stablehlo.broadcast_in_dim %4077, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4079 = stablehlo.compare  GT, %4076, %4078,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %4080 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4081 = stablehlo.broadcast_in_dim %4080, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4082 = stablehlo.convert %4081 : tensor<1x1x1x20xf32>
      %4083 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %4084 = stablehlo.broadcast_in_dim %4083, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4085 = stablehlo.select %4079, %4082, %4084 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %4086 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %4087 = stablehlo.sqrt %4086 : tensor<f32>
      %4088 = stablehlo.convert %4087 : tensor<f32>
      %4089 = stablehlo.broadcast_in_dim %4088, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %4090 = stablehlo.divide %4028, %4089 : tensor<1x1x12x64xf32>
      %4091 = stablehlo.dot_general %4090, %4054, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %4092 = stablehlo.broadcast_in_dim %4085, dims = [0, 1, 2, 3] : (tensor<1x1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %4093 = stablehlo.add %4091, %4092 : tensor<1x12x1x20xf32>
      %4094 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %4095 = stablehlo.reduce(%4093 init: %4094) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %4096 = stablehlo.broadcast_in_dim %4095, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %4097 = stablehlo.broadcast_in_dim %4096, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %4098 = stablehlo.subtract %4093, %4097 : tensor<1x12x1x20xf32>
      %4099 = stablehlo.exponential %4098 : tensor<1x12x1x20xf32>
      %4100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4101 = stablehlo.reduce(%4099 init: %4100) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %4102 = stablehlo.broadcast_in_dim %4101, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %4103 = stablehlo.broadcast_in_dim %4102, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %4104 = stablehlo.divide %4099, %4103 : tensor<1x12x1x20xf32>
      %4105 = stablehlo.dot_general %4063, %4104, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %4106 = stablehlo.transpose %4105, dims = [0, 3, 1, 2] : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %4107 = stablehlo.reshape %4106 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %4108 = stablehlo.transpose %iterArg_65, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %4109 = stablehlo.convert %4108 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %4110 = stablehlo.dot_general %4107, %4109, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %4111 = stablehlo.convert %iterArg_66 : (tensor<768xf16>) -> tensor<768xf32>
      %4112 = stablehlo.broadcast_in_dim %4111, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4113 = stablehlo.add %4110, %4112 : tensor<1x1x768xf32>
      %4114 = stablehlo.add %4113, %3976 : tensor<1x1x768xf32>
      %4115 = stablehlo.multiply %4114, %4114 : tensor<1x1x768xf32>
      %4116 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4117 = stablehlo.reduce(%4115 init: %4116) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4118 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4119 = stablehlo.broadcast_in_dim %4118, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4120 = stablehlo.divide %4117, %4119 : tensor<1x1xf32>
      %4121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4122 = stablehlo.reduce(%4114 init: %4121) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4123 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4124 = stablehlo.broadcast_in_dim %4123, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4125 = stablehlo.divide %4122, %4124 : tensor<1x1xf32>
      %4126 = stablehlo.multiply %4125, %4125 : tensor<1x1xf32>
      %4127 = stablehlo.subtract %4120, %4126 : tensor<1x1xf32>
      %4128 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4129 = stablehlo.broadcast_in_dim %4128, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4130 = stablehlo.maximum %4129, %4127 : tensor<1x1xf32>
      %4131 = stablehlo.broadcast_in_dim %4125, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4132 = stablehlo.broadcast_in_dim %4130, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4133 = stablehlo.broadcast_in_dim %4131, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4134 = stablehlo.subtract %4114, %4133 : tensor<1x1x768xf32>
      %4135 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %4136 = stablehlo.broadcast_in_dim %4135, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %4137 = stablehlo.add %4132, %4136 : tensor<1x1x1xf32>
      %4138 = stablehlo.rsqrt %4137 : tensor<1x1x1xf32>
      %4139 = stablehlo.reshape %iterArg_67 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4140 = stablehlo.convert %4139 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4141 = stablehlo.broadcast_in_dim %4138, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4142 = stablehlo.multiply %4141, %4140 : tensor<1x1x768xf32>
      %4143 = stablehlo.multiply %4134, %4142 : tensor<1x1x768xf32>
      %4144 = stablehlo.reshape %iterArg_68 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4145 = stablehlo.convert %4144 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4146 = stablehlo.add %4143, %4145 : tensor<1x1x768xf32>
      %4147 = stablehlo.transpose %iterArg_69, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %4148 = stablehlo.convert %4147 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %4149 = stablehlo.dot_general %4146, %4148, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %4150 = stablehlo.convert %iterArg_70 : (tensor<3072xf16>) -> tensor<3072xf32>
      %4151 = stablehlo.broadcast_in_dim %4150, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %4152 = stablehlo.add %4149, %4151 : tensor<1x1x3072xf32>
      %4153 = stablehlo.multiply %4152, %4152 : tensor<1x1x3072xf32>
      %4154 = stablehlo.multiply %4152, %4153 : tensor<1x1x3072xf32>
      %4155 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
      %4156 = stablehlo.broadcast_in_dim %4155, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4157 = stablehlo.multiply %4156, %4154 : tensor<1x1x3072xf32>
      %4158 = stablehlo.add %4152, %4157 : tensor<1x1x3072xf32>
      %4159 = stablehlo.constant dense<0.797884583> : tensor<f32>
      %4160 = stablehlo.broadcast_in_dim %4159, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4161 = stablehlo.multiply %4160, %4158 : tensor<1x1x3072xf32>
      %4162 = stablehlo.tanh %4161 : tensor<1x1x3072xf32>
      %4163 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %4164 = stablehlo.broadcast_in_dim %4163, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4165 = stablehlo.add %4164, %4162 : tensor<1x1x3072xf32>
      %4166 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %4167 = stablehlo.broadcast_in_dim %4166, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4168 = stablehlo.multiply %4167, %4165 : tensor<1x1x3072xf32>
      %4169 = stablehlo.multiply %4152, %4168 : tensor<1x1x3072xf32>
      %4170 = stablehlo.transpose %iterArg_71, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %4171 = stablehlo.convert %4170 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %4172 = stablehlo.dot_general %4169, %4171, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %4173 = stablehlo.convert %iterArg_72 : (tensor<768xf16>) -> tensor<768xf32>
      %4174 = stablehlo.broadcast_in_dim %4173, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4175 = stablehlo.add %4172, %4174 : tensor<1x1x768xf32>
      %4176 = stablehlo.add %4114, %4175 : tensor<1x1x768xf32>
      %4177 = stablehlo.multiply %4176, %4176 : tensor<1x1x768xf32>
      %4178 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4179 = stablehlo.reduce(%4177 init: %4178) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4180 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4181 = stablehlo.broadcast_in_dim %4180, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4182 = stablehlo.divide %4179, %4181 : tensor<1x1xf32>
      %4183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4184 = stablehlo.reduce(%4176 init: %4183) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4185 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4186 = stablehlo.broadcast_in_dim %4185, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4187 = stablehlo.divide %4184, %4186 : tensor<1x1xf32>
      %4188 = stablehlo.multiply %4187, %4187 : tensor<1x1xf32>
      %4189 = stablehlo.subtract %4182, %4188 : tensor<1x1xf32>
      %4190 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4191 = stablehlo.broadcast_in_dim %4190, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4192 = stablehlo.maximum %4191, %4189 : tensor<1x1xf32>
      %4193 = stablehlo.broadcast_in_dim %4187, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4194 = stablehlo.broadcast_in_dim %4192, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4195 = stablehlo.broadcast_in_dim %4193, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4196 = stablehlo.subtract %4176, %4195 : tensor<1x1x768xf32>
      %4197 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %4198 = stablehlo.broadcast_in_dim %4197, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %4199 = stablehlo.add %4194, %4198 : tensor<1x1x1xf32>
      %4200 = stablehlo.rsqrt %4199 : tensor<1x1x1xf32>
      %4201 = stablehlo.reshape %iterArg_73 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4202 = stablehlo.convert %4201 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4203 = stablehlo.broadcast_in_dim %4200, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4204 = stablehlo.multiply %4203, %4202 : tensor<1x1x768xf32>
      %4205 = stablehlo.multiply %4196, %4204 : tensor<1x1x768xf32>
      %4206 = stablehlo.reshape %iterArg_74 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4207 = stablehlo.convert %4206 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4208 = stablehlo.add %4205, %4207 : tensor<1x1x768xf32>
      %4209 = stablehlo.constant dense<true> : tensor<i1>
      %4210 = stablehlo.broadcast_in_dim %4209, dims = [] : (tensor<i1>) -> tensor<1x1024xi1>
      %4211 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %4212 = stablehlo.broadcast_in_dim %4211, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %4213 = stablehlo.broadcast_in_dim %4212, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %4214 = stablehlo.broadcast_in_dim %4212, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %4215 = stablehlo.broadcast_in_dim %4213, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %4216 = stablehlo.broadcast_in_dim %4214, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %4217 = stablehlo.compare  GE, %4215, %4216,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %4218 = stablehlo.broadcast_in_dim %4217, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %4219 = stablehlo.transpose %iterArg_75, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %4220 = stablehlo.convert %4219 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %4221 = stablehlo.dot_general %4208, %4220, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32>
      %4222 = stablehlo.convert %iterArg_76 : (tensor<2304xf16>) -> tensor<2304xf32>
      %4223 = stablehlo.broadcast_in_dim %4222, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %4224 = stablehlo.add %4221, %4223 : tensor<1x1x2304xf32>
      %4225 = stablehlo.slice %4224 [0:1, 0:1, 0:768] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4226 = stablehlo.slice %4224 [0:1, 0:1, 768:1536] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4227 = stablehlo.slice %4224 [0:1, 0:1, 1536:2304] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4228 = stablehlo.reshape %4225 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4229 = stablehlo.reshape %4226 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4230 = stablehlo.reshape %4227 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4231 = stablehlo.constant dense<0> : tensor<i32>
      %4232 = stablehlo.compare  LT, %iterArg_178, %4231,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4233 = stablehlo.constant dense<1024> : tensor<i32>
      %4234 = stablehlo.add %iterArg_178, %4233 : tensor<i32>
      %4235 = stablehlo.select %4232, %4234, %iterArg_178 : tensor<i1>, tensor<i32>
      %4236 = stablehlo.constant dense<0> : tensor<i32>
      %4237 = stablehlo.constant dense<0> : tensor<i32>
      %4238 = stablehlo.constant dense<0> : tensor<i32>
      %4239 = stablehlo.dynamic_slice %4218, %4236, %4237, %4235, %4238, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %4240 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %4241 = stablehlo.constant dense<0> : tensor<i32>
      %4242 = stablehlo.broadcast_in_dim %4241, dims = [] : (tensor<i32>) -> tensor<1x1x1x20xi32>
      %4243 = stablehlo.compare  NE, %4240, %4242,  SIGNED : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %4244 = stablehlo.and %4243, %4239 : tensor<1x1x1x20xi1>
      %4245 = stablehlo.convert %4244 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %4246 = stablehlo.constant dense<0> : tensor<i32>
      %4247 = stablehlo.compare  LT, %iterArg_178, %4246,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4248 = stablehlo.constant dense<20> : tensor<i32>
      %4249 = stablehlo.add %iterArg_178, %4248 : tensor<i32>
      %4250 = stablehlo.select %4247, %4249, %iterArg_178 : tensor<i1>, tensor<i32>
      %4251 = stablehlo.constant dense<0> : tensor<i32>
      %4252 = stablehlo.constant dense<0> : tensor<i32>
      %4253 = stablehlo.constant dense<0> : tensor<i32>
      %4254 = stablehlo.dynamic_update_slice %iterArg_179, %4229, %4251, %4250, %4252, %4253 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %4255 = stablehlo.constant dense<0> : tensor<i32>
      %4256 = stablehlo.compare  LT, %iterArg_178, %4255,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4257 = stablehlo.constant dense<20> : tensor<i32>
      %4258 = stablehlo.add %iterArg_178, %4257 : tensor<i32>
      %4259 = stablehlo.select %4256, %4258, %iterArg_178 : tensor<i1>, tensor<i32>
      %4260 = stablehlo.constant dense<0> : tensor<i32>
      %4261 = stablehlo.constant dense<0> : tensor<i32>
      %4262 = stablehlo.constant dense<0> : tensor<i32>
      %4263 = stablehlo.dynamic_update_slice %iterArg_180, %4230, %4260, %4259, %4261, %4262 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %4264 = stablehlo.constant dense<1> : tensor<i32>
      %4265 = stablehlo.add %iterArg_178, %4264 : tensor<i32>
      %4266 = stablehlo.iota dim = 0 : tensor<20xi32>
      %4267 = stablehlo.constant dense<1> : tensor<i32>
      %4268 = stablehlo.add %iterArg_178, %4267 : tensor<i32>
      %4269 = stablehlo.broadcast_in_dim %4268, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %4270 = stablehlo.compare  LT, %4266, %4269,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %4271 = stablehlo.broadcast_in_dim %4270, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %4272 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4273 = stablehlo.broadcast_in_dim %4272, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4274 = stablehlo.compare  NE, %4245, %4273,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %4275 = stablehlo.and %4271, %4274 : tensor<1x1x1x20xi1>
      %4276 = stablehlo.convert %4275 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %4277 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4278 = stablehlo.broadcast_in_dim %4277, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4279 = stablehlo.compare  GT, %4276, %4278,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %4280 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4281 = stablehlo.broadcast_in_dim %4280, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4282 = stablehlo.convert %4281 : tensor<1x1x1x20xf32>
      %4283 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %4284 = stablehlo.broadcast_in_dim %4283, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4285 = stablehlo.select %4279, %4282, %4284 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %4286 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %4287 = stablehlo.sqrt %4286 : tensor<f32>
      %4288 = stablehlo.convert %4287 : tensor<f32>
      %4289 = stablehlo.broadcast_in_dim %4288, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %4290 = stablehlo.divide %4228, %4289 : tensor<1x1x12x64xf32>
      %4291 = stablehlo.dot_general %4290, %4254, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %4292 = stablehlo.broadcast_in_dim %4285, dims = [0, 1, 2, 3] : (tensor<1x1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %4293 = stablehlo.add %4291, %4292 : tensor<1x12x1x20xf32>
      %4294 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %4295 = stablehlo.reduce(%4293 init: %4294) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %4296 = stablehlo.broadcast_in_dim %4295, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %4297 = stablehlo.broadcast_in_dim %4296, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %4298 = stablehlo.subtract %4293, %4297 : tensor<1x12x1x20xf32>
      %4299 = stablehlo.exponential %4298 : tensor<1x12x1x20xf32>
      %4300 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4301 = stablehlo.reduce(%4299 init: %4300) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %4302 = stablehlo.broadcast_in_dim %4301, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %4303 = stablehlo.broadcast_in_dim %4302, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %4304 = stablehlo.divide %4299, %4303 : tensor<1x12x1x20xf32>
      %4305 = stablehlo.dot_general %4263, %4304, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %4306 = stablehlo.transpose %4305, dims = [0, 3, 1, 2] : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %4307 = stablehlo.reshape %4306 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %4308 = stablehlo.transpose %iterArg_77, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %4309 = stablehlo.convert %4308 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %4310 = stablehlo.dot_general %4307, %4309, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %4311 = stablehlo.convert %iterArg_78 : (tensor<768xf16>) -> tensor<768xf32>
      %4312 = stablehlo.broadcast_in_dim %4311, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4313 = stablehlo.add %4310, %4312 : tensor<1x1x768xf32>
      %4314 = stablehlo.add %4313, %4176 : tensor<1x1x768xf32>
      %4315 = stablehlo.multiply %4314, %4314 : tensor<1x1x768xf32>
      %4316 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4317 = stablehlo.reduce(%4315 init: %4316) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4318 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4319 = stablehlo.broadcast_in_dim %4318, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4320 = stablehlo.divide %4317, %4319 : tensor<1x1xf32>
      %4321 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4322 = stablehlo.reduce(%4314 init: %4321) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4323 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4324 = stablehlo.broadcast_in_dim %4323, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4325 = stablehlo.divide %4322, %4324 : tensor<1x1xf32>
      %4326 = stablehlo.multiply %4325, %4325 : tensor<1x1xf32>
      %4327 = stablehlo.subtract %4320, %4326 : tensor<1x1xf32>
      %4328 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4329 = stablehlo.broadcast_in_dim %4328, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4330 = stablehlo.maximum %4329, %4327 : tensor<1x1xf32>
      %4331 = stablehlo.broadcast_in_dim %4325, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4332 = stablehlo.broadcast_in_dim %4330, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4333 = stablehlo.broadcast_in_dim %4331, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4334 = stablehlo.subtract %4314, %4333 : tensor<1x1x768xf32>
      %4335 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %4336 = stablehlo.broadcast_in_dim %4335, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %4337 = stablehlo.add %4332, %4336 : tensor<1x1x1xf32>
      %4338 = stablehlo.rsqrt %4337 : tensor<1x1x1xf32>
      %4339 = stablehlo.reshape %iterArg_79 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4340 = stablehlo.convert %4339 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4341 = stablehlo.broadcast_in_dim %4338, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4342 = stablehlo.multiply %4341, %4340 : tensor<1x1x768xf32>
      %4343 = stablehlo.multiply %4334, %4342 : tensor<1x1x768xf32>
      %4344 = stablehlo.reshape %iterArg_80 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4345 = stablehlo.convert %4344 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4346 = stablehlo.add %4343, %4345 : tensor<1x1x768xf32>
      %4347 = stablehlo.transpose %iterArg_81, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %4348 = stablehlo.convert %4347 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %4349 = stablehlo.dot_general %4346, %4348, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %4350 = stablehlo.convert %iterArg_82 : (tensor<3072xf16>) -> tensor<3072xf32>
      %4351 = stablehlo.broadcast_in_dim %4350, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %4352 = stablehlo.add %4349, %4351 : tensor<1x1x3072xf32>
      %4353 = stablehlo.multiply %4352, %4352 : tensor<1x1x3072xf32>
      %4354 = stablehlo.multiply %4352, %4353 : tensor<1x1x3072xf32>
      %4355 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
      %4356 = stablehlo.broadcast_in_dim %4355, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4357 = stablehlo.multiply %4356, %4354 : tensor<1x1x3072xf32>
      %4358 = stablehlo.add %4352, %4357 : tensor<1x1x3072xf32>
      %4359 = stablehlo.constant dense<0.797884583> : tensor<f32>
      %4360 = stablehlo.broadcast_in_dim %4359, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4361 = stablehlo.multiply %4360, %4358 : tensor<1x1x3072xf32>
      %4362 = stablehlo.tanh %4361 : tensor<1x1x3072xf32>
      %4363 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %4364 = stablehlo.broadcast_in_dim %4363, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4365 = stablehlo.add %4364, %4362 : tensor<1x1x3072xf32>
      %4366 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %4367 = stablehlo.broadcast_in_dim %4366, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4368 = stablehlo.multiply %4367, %4365 : tensor<1x1x3072xf32>
      %4369 = stablehlo.multiply %4352, %4368 : tensor<1x1x3072xf32>
      %4370 = stablehlo.transpose %iterArg_83, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %4371 = stablehlo.convert %4370 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %4372 = stablehlo.dot_general %4369, %4371, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %4373 = stablehlo.convert %iterArg_84 : (tensor<768xf16>) -> tensor<768xf32>
      %4374 = stablehlo.broadcast_in_dim %4373, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4375 = stablehlo.add %4372, %4374 : tensor<1x1x768xf32>
      %4376 = stablehlo.add %4314, %4375 : tensor<1x1x768xf32>
      %4377 = stablehlo.multiply %4376, %4376 : tensor<1x1x768xf32>
      %4378 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4379 = stablehlo.reduce(%4377 init: %4378) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4380 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4381 = stablehlo.broadcast_in_dim %4380, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4382 = stablehlo.divide %4379, %4381 : tensor<1x1xf32>
      %4383 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4384 = stablehlo.reduce(%4376 init: %4383) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4385 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4386 = stablehlo.broadcast_in_dim %4385, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4387 = stablehlo.divide %4384, %4386 : tensor<1x1xf32>
      %4388 = stablehlo.multiply %4387, %4387 : tensor<1x1xf32>
      %4389 = stablehlo.subtract %4382, %4388 : tensor<1x1xf32>
      %4390 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4391 = stablehlo.broadcast_in_dim %4390, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4392 = stablehlo.maximum %4391, %4389 : tensor<1x1xf32>
      %4393 = stablehlo.broadcast_in_dim %4387, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4394 = stablehlo.broadcast_in_dim %4392, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4395 = stablehlo.broadcast_in_dim %4393, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4396 = stablehlo.subtract %4376, %4395 : tensor<1x1x768xf32>
      %4397 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %4398 = stablehlo.broadcast_in_dim %4397, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %4399 = stablehlo.add %4394, %4398 : tensor<1x1x1xf32>
      %4400 = stablehlo.rsqrt %4399 : tensor<1x1x1xf32>
      %4401 = stablehlo.reshape %iterArg_85 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4402 = stablehlo.convert %4401 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4403 = stablehlo.broadcast_in_dim %4400, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4404 = stablehlo.multiply %4403, %4402 : tensor<1x1x768xf32>
      %4405 = stablehlo.multiply %4396, %4404 : tensor<1x1x768xf32>
      %4406 = stablehlo.reshape %iterArg_86 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4407 = stablehlo.convert %4406 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4408 = stablehlo.add %4405, %4407 : tensor<1x1x768xf32>
      %4409 = stablehlo.constant dense<true> : tensor<i1>
      %4410 = stablehlo.broadcast_in_dim %4409, dims = [] : (tensor<i1>) -> tensor<1x1024xi1>
      %4411 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %4412 = stablehlo.broadcast_in_dim %4411, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %4413 = stablehlo.broadcast_in_dim %4412, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %4414 = stablehlo.broadcast_in_dim %4412, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %4415 = stablehlo.broadcast_in_dim %4413, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %4416 = stablehlo.broadcast_in_dim %4414, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %4417 = stablehlo.compare  GE, %4415, %4416,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %4418 = stablehlo.broadcast_in_dim %4417, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %4419 = stablehlo.transpose %iterArg_87, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %4420 = stablehlo.convert %4419 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %4421 = stablehlo.dot_general %4408, %4420, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32>
      %4422 = stablehlo.convert %iterArg_88 : (tensor<2304xf16>) -> tensor<2304xf32>
      %4423 = stablehlo.broadcast_in_dim %4422, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %4424 = stablehlo.add %4421, %4423 : tensor<1x1x2304xf32>
      %4425 = stablehlo.slice %4424 [0:1, 0:1, 0:768] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4426 = stablehlo.slice %4424 [0:1, 0:1, 768:1536] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4427 = stablehlo.slice %4424 [0:1, 0:1, 1536:2304] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4428 = stablehlo.reshape %4425 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4429 = stablehlo.reshape %4426 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4430 = stablehlo.reshape %4427 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4431 = stablehlo.constant dense<0> : tensor<i32>
      %4432 = stablehlo.compare  LT, %iterArg_181, %4431,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4433 = stablehlo.constant dense<1024> : tensor<i32>
      %4434 = stablehlo.add %iterArg_181, %4433 : tensor<i32>
      %4435 = stablehlo.select %4432, %4434, %iterArg_181 : tensor<i1>, tensor<i32>
      %4436 = stablehlo.constant dense<0> : tensor<i32>
      %4437 = stablehlo.constant dense<0> : tensor<i32>
      %4438 = stablehlo.constant dense<0> : tensor<i32>
      %4439 = stablehlo.dynamic_slice %4418, %4436, %4437, %4435, %4438, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %4440 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %4441 = stablehlo.constant dense<0> : tensor<i32>
      %4442 = stablehlo.broadcast_in_dim %4441, dims = [] : (tensor<i32>) -> tensor<1x1x1x20xi32>
      %4443 = stablehlo.compare  NE, %4440, %4442,  SIGNED : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %4444 = stablehlo.and %4443, %4439 : tensor<1x1x1x20xi1>
      %4445 = stablehlo.convert %4444 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %4446 = stablehlo.constant dense<0> : tensor<i32>
      %4447 = stablehlo.compare  LT, %iterArg_181, %4446,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4448 = stablehlo.constant dense<20> : tensor<i32>
      %4449 = stablehlo.add %iterArg_181, %4448 : tensor<i32>
      %4450 = stablehlo.select %4447, %4449, %iterArg_181 : tensor<i1>, tensor<i32>
      %4451 = stablehlo.constant dense<0> : tensor<i32>
      %4452 = stablehlo.constant dense<0> : tensor<i32>
      %4453 = stablehlo.constant dense<0> : tensor<i32>
      %4454 = stablehlo.dynamic_update_slice %iterArg_182, %4429, %4451, %4450, %4452, %4453 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %4455 = stablehlo.constant dense<0> : tensor<i32>
      %4456 = stablehlo.compare  LT, %iterArg_181, %4455,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4457 = stablehlo.constant dense<20> : tensor<i32>
      %4458 = stablehlo.add %iterArg_181, %4457 : tensor<i32>
      %4459 = stablehlo.select %4456, %4458, %iterArg_181 : tensor<i1>, tensor<i32>
      %4460 = stablehlo.constant dense<0> : tensor<i32>
      %4461 = stablehlo.constant dense<0> : tensor<i32>
      %4462 = stablehlo.constant dense<0> : tensor<i32>
      %4463 = stablehlo.dynamic_update_slice %iterArg_183, %4430, %4460, %4459, %4461, %4462 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %4464 = stablehlo.constant dense<1> : tensor<i32>
      %4465 = stablehlo.add %iterArg_181, %4464 : tensor<i32>
      %4466 = stablehlo.iota dim = 0 : tensor<20xi32>
      %4467 = stablehlo.constant dense<1> : tensor<i32>
      %4468 = stablehlo.add %iterArg_181, %4467 : tensor<i32>
      %4469 = stablehlo.broadcast_in_dim %4468, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %4470 = stablehlo.compare  LT, %4466, %4469,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %4471 = stablehlo.broadcast_in_dim %4470, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %4472 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4473 = stablehlo.broadcast_in_dim %4472, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4474 = stablehlo.compare  NE, %4445, %4473,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %4475 = stablehlo.and %4471, %4474 : tensor<1x1x1x20xi1>
      %4476 = stablehlo.convert %4475 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %4477 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4478 = stablehlo.broadcast_in_dim %4477, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4479 = stablehlo.compare  GT, %4476, %4478,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %4480 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4481 = stablehlo.broadcast_in_dim %4480, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4482 = stablehlo.convert %4481 : tensor<1x1x1x20xf32>
      %4483 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %4484 = stablehlo.broadcast_in_dim %4483, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4485 = stablehlo.select %4479, %4482, %4484 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %4486 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %4487 = stablehlo.sqrt %4486 : tensor<f32>
      %4488 = stablehlo.convert %4487 : tensor<f32>
      %4489 = stablehlo.broadcast_in_dim %4488, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %4490 = stablehlo.divide %4428, %4489 : tensor<1x1x12x64xf32>
      %4491 = stablehlo.dot_general %4490, %4454, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %4492 = stablehlo.broadcast_in_dim %4485, dims = [0, 1, 2, 3] : (tensor<1x1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %4493 = stablehlo.add %4491, %4492 : tensor<1x12x1x20xf32>
      %4494 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %4495 = stablehlo.reduce(%4493 init: %4494) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %4496 = stablehlo.broadcast_in_dim %4495, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %4497 = stablehlo.broadcast_in_dim %4496, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %4498 = stablehlo.subtract %4493, %4497 : tensor<1x12x1x20xf32>
      %4499 = stablehlo.exponential %4498 : tensor<1x12x1x20xf32>
      %4500 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4501 = stablehlo.reduce(%4499 init: %4500) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %4502 = stablehlo.broadcast_in_dim %4501, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %4503 = stablehlo.broadcast_in_dim %4502, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %4504 = stablehlo.divide %4499, %4503 : tensor<1x12x1x20xf32>
      %4505 = stablehlo.dot_general %4463, %4504, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %4506 = stablehlo.transpose %4505, dims = [0, 3, 1, 2] : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %4507 = stablehlo.reshape %4506 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %4508 = stablehlo.transpose %iterArg_89, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %4509 = stablehlo.convert %4508 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %4510 = stablehlo.dot_general %4507, %4509, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %4511 = stablehlo.convert %iterArg_90 : (tensor<768xf16>) -> tensor<768xf32>
      %4512 = stablehlo.broadcast_in_dim %4511, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4513 = stablehlo.add %4510, %4512 : tensor<1x1x768xf32>
      %4514 = stablehlo.add %4513, %4376 : tensor<1x1x768xf32>
      %4515 = stablehlo.multiply %4514, %4514 : tensor<1x1x768xf32>
      %4516 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4517 = stablehlo.reduce(%4515 init: %4516) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4518 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4519 = stablehlo.broadcast_in_dim %4518, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4520 = stablehlo.divide %4517, %4519 : tensor<1x1xf32>
      %4521 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4522 = stablehlo.reduce(%4514 init: %4521) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4523 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4524 = stablehlo.broadcast_in_dim %4523, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4525 = stablehlo.divide %4522, %4524 : tensor<1x1xf32>
      %4526 = stablehlo.multiply %4525, %4525 : tensor<1x1xf32>
      %4527 = stablehlo.subtract %4520, %4526 : tensor<1x1xf32>
      %4528 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4529 = stablehlo.broadcast_in_dim %4528, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4530 = stablehlo.maximum %4529, %4527 : tensor<1x1xf32>
      %4531 = stablehlo.broadcast_in_dim %4525, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4532 = stablehlo.broadcast_in_dim %4530, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4533 = stablehlo.broadcast_in_dim %4531, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4534 = stablehlo.subtract %4514, %4533 : tensor<1x1x768xf32>
      %4535 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %4536 = stablehlo.broadcast_in_dim %4535, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %4537 = stablehlo.add %4532, %4536 : tensor<1x1x1xf32>
      %4538 = stablehlo.rsqrt %4537 : tensor<1x1x1xf32>
      %4539 = stablehlo.reshape %iterArg_91 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4540 = stablehlo.convert %4539 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4541 = stablehlo.broadcast_in_dim %4538, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4542 = stablehlo.multiply %4541, %4540 : tensor<1x1x768xf32>
      %4543 = stablehlo.multiply %4534, %4542 : tensor<1x1x768xf32>
      %4544 = stablehlo.reshape %iterArg_92 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4545 = stablehlo.convert %4544 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4546 = stablehlo.add %4543, %4545 : tensor<1x1x768xf32>
      %4547 = stablehlo.transpose %iterArg_93, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %4548 = stablehlo.convert %4547 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %4549 = stablehlo.dot_general %4546, %4548, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %4550 = stablehlo.convert %iterArg_94 : (tensor<3072xf16>) -> tensor<3072xf32>
      %4551 = stablehlo.broadcast_in_dim %4550, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %4552 = stablehlo.add %4549, %4551 : tensor<1x1x3072xf32>
      %4553 = stablehlo.multiply %4552, %4552 : tensor<1x1x3072xf32>
      %4554 = stablehlo.multiply %4552, %4553 : tensor<1x1x3072xf32>
      %4555 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
      %4556 = stablehlo.broadcast_in_dim %4555, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4557 = stablehlo.multiply %4556, %4554 : tensor<1x1x3072xf32>
      %4558 = stablehlo.add %4552, %4557 : tensor<1x1x3072xf32>
      %4559 = stablehlo.constant dense<0.797884583> : tensor<f32>
      %4560 = stablehlo.broadcast_in_dim %4559, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4561 = stablehlo.multiply %4560, %4558 : tensor<1x1x3072xf32>
      %4562 = stablehlo.tanh %4561 : tensor<1x1x3072xf32>
      %4563 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %4564 = stablehlo.broadcast_in_dim %4563, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4565 = stablehlo.add %4564, %4562 : tensor<1x1x3072xf32>
      %4566 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %4567 = stablehlo.broadcast_in_dim %4566, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4568 = stablehlo.multiply %4567, %4565 : tensor<1x1x3072xf32>
      %4569 = stablehlo.multiply %4552, %4568 : tensor<1x1x3072xf32>
      %4570 = stablehlo.transpose %iterArg_95, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %4571 = stablehlo.convert %4570 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %4572 = stablehlo.dot_general %4569, %4571, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %4573 = stablehlo.convert %iterArg_96 : (tensor<768xf16>) -> tensor<768xf32>
      %4574 = stablehlo.broadcast_in_dim %4573, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4575 = stablehlo.add %4572, %4574 : tensor<1x1x768xf32>
      %4576 = stablehlo.add %4514, %4575 : tensor<1x1x768xf32>
      %4577 = stablehlo.multiply %4576, %4576 : tensor<1x1x768xf32>
      %4578 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4579 = stablehlo.reduce(%4577 init: %4578) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4580 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4581 = stablehlo.broadcast_in_dim %4580, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4582 = stablehlo.divide %4579, %4581 : tensor<1x1xf32>
      %4583 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4584 = stablehlo.reduce(%4576 init: %4583) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4585 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4586 = stablehlo.broadcast_in_dim %4585, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4587 = stablehlo.divide %4584, %4586 : tensor<1x1xf32>
      %4588 = stablehlo.multiply %4587, %4587 : tensor<1x1xf32>
      %4589 = stablehlo.subtract %4582, %4588 : tensor<1x1xf32>
      %4590 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4591 = stablehlo.broadcast_in_dim %4590, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4592 = stablehlo.maximum %4591, %4589 : tensor<1x1xf32>
      %4593 = stablehlo.broadcast_in_dim %4587, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4594 = stablehlo.broadcast_in_dim %4592, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4595 = stablehlo.broadcast_in_dim %4593, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4596 = stablehlo.subtract %4576, %4595 : tensor<1x1x768xf32>
      %4597 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %4598 = stablehlo.broadcast_in_dim %4597, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %4599 = stablehlo.add %4594, %4598 : tensor<1x1x1xf32>
      %4600 = stablehlo.rsqrt %4599 : tensor<1x1x1xf32>
      %4601 = stablehlo.reshape %iterArg_97 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4602 = stablehlo.convert %4601 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4603 = stablehlo.broadcast_in_dim %4600, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4604 = stablehlo.multiply %4603, %4602 : tensor<1x1x768xf32>
      %4605 = stablehlo.multiply %4596, %4604 : tensor<1x1x768xf32>
      %4606 = stablehlo.reshape %iterArg_98 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4607 = stablehlo.convert %4606 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4608 = stablehlo.add %4605, %4607 : tensor<1x1x768xf32>
      %4609 = stablehlo.constant dense<true> : tensor<i1>
      %4610 = stablehlo.broadcast_in_dim %4609, dims = [] : (tensor<i1>) -> tensor<1x1024xi1>
      %4611 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %4612 = stablehlo.broadcast_in_dim %4611, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %4613 = stablehlo.broadcast_in_dim %4612, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %4614 = stablehlo.broadcast_in_dim %4612, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %4615 = stablehlo.broadcast_in_dim %4613, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %4616 = stablehlo.broadcast_in_dim %4614, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %4617 = stablehlo.compare  GE, %4615, %4616,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %4618 = stablehlo.broadcast_in_dim %4617, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %4619 = stablehlo.transpose %iterArg_99, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %4620 = stablehlo.convert %4619 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %4621 = stablehlo.dot_general %4608, %4620, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32>
      %4622 = stablehlo.convert %iterArg_100 : (tensor<2304xf16>) -> tensor<2304xf32>
      %4623 = stablehlo.broadcast_in_dim %4622, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %4624 = stablehlo.add %4621, %4623 : tensor<1x1x2304xf32>
      %4625 = stablehlo.slice %4624 [0:1, 0:1, 0:768] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4626 = stablehlo.slice %4624 [0:1, 0:1, 768:1536] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4627 = stablehlo.slice %4624 [0:1, 0:1, 1536:2304] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4628 = stablehlo.reshape %4625 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4629 = stablehlo.reshape %4626 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4630 = stablehlo.reshape %4627 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4631 = stablehlo.constant dense<0> : tensor<i32>
      %4632 = stablehlo.compare  LT, %iterArg_184, %4631,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4633 = stablehlo.constant dense<1024> : tensor<i32>
      %4634 = stablehlo.add %iterArg_184, %4633 : tensor<i32>
      %4635 = stablehlo.select %4632, %4634, %iterArg_184 : tensor<i1>, tensor<i32>
      %4636 = stablehlo.constant dense<0> : tensor<i32>
      %4637 = stablehlo.constant dense<0> : tensor<i32>
      %4638 = stablehlo.constant dense<0> : tensor<i32>
      %4639 = stablehlo.dynamic_slice %4618, %4636, %4637, %4635, %4638, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %4640 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %4641 = stablehlo.constant dense<0> : tensor<i32>
      %4642 = stablehlo.broadcast_in_dim %4641, dims = [] : (tensor<i32>) -> tensor<1x1x1x20xi32>
      %4643 = stablehlo.compare  NE, %4640, %4642,  SIGNED : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %4644 = stablehlo.and %4643, %4639 : tensor<1x1x1x20xi1>
      %4645 = stablehlo.convert %4644 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %4646 = stablehlo.constant dense<0> : tensor<i32>
      %4647 = stablehlo.compare  LT, %iterArg_184, %4646,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4648 = stablehlo.constant dense<20> : tensor<i32>
      %4649 = stablehlo.add %iterArg_184, %4648 : tensor<i32>
      %4650 = stablehlo.select %4647, %4649, %iterArg_184 : tensor<i1>, tensor<i32>
      %4651 = stablehlo.constant dense<0> : tensor<i32>
      %4652 = stablehlo.constant dense<0> : tensor<i32>
      %4653 = stablehlo.constant dense<0> : tensor<i32>
      %4654 = stablehlo.dynamic_update_slice %iterArg_185, %4629, %4651, %4650, %4652, %4653 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %4655 = stablehlo.constant dense<0> : tensor<i32>
      %4656 = stablehlo.compare  LT, %iterArg_184, %4655,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4657 = stablehlo.constant dense<20> : tensor<i32>
      %4658 = stablehlo.add %iterArg_184, %4657 : tensor<i32>
      %4659 = stablehlo.select %4656, %4658, %iterArg_184 : tensor<i1>, tensor<i32>
      %4660 = stablehlo.constant dense<0> : tensor<i32>
      %4661 = stablehlo.constant dense<0> : tensor<i32>
      %4662 = stablehlo.constant dense<0> : tensor<i32>
      %4663 = stablehlo.dynamic_update_slice %iterArg_186, %4630, %4660, %4659, %4661, %4662 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %4664 = stablehlo.constant dense<1> : tensor<i32>
      %4665 = stablehlo.add %iterArg_184, %4664 : tensor<i32>
      %4666 = stablehlo.iota dim = 0 : tensor<20xi32>
      %4667 = stablehlo.constant dense<1> : tensor<i32>
      %4668 = stablehlo.add %iterArg_184, %4667 : tensor<i32>
      %4669 = stablehlo.broadcast_in_dim %4668, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %4670 = stablehlo.compare  LT, %4666, %4669,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %4671 = stablehlo.broadcast_in_dim %4670, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %4672 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4673 = stablehlo.broadcast_in_dim %4672, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4674 = stablehlo.compare  NE, %4645, %4673,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %4675 = stablehlo.and %4671, %4674 : tensor<1x1x1x20xi1>
      %4676 = stablehlo.convert %4675 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %4677 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4678 = stablehlo.broadcast_in_dim %4677, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4679 = stablehlo.compare  GT, %4676, %4678,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %4680 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4681 = stablehlo.broadcast_in_dim %4680, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4682 = stablehlo.convert %4681 : tensor<1x1x1x20xf32>
      %4683 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %4684 = stablehlo.broadcast_in_dim %4683, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4685 = stablehlo.select %4679, %4682, %4684 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %4686 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %4687 = stablehlo.sqrt %4686 : tensor<f32>
      %4688 = stablehlo.convert %4687 : tensor<f32>
      %4689 = stablehlo.broadcast_in_dim %4688, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %4690 = stablehlo.divide %4628, %4689 : tensor<1x1x12x64xf32>
      %4691 = stablehlo.dot_general %4690, %4654, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %4692 = stablehlo.broadcast_in_dim %4685, dims = [0, 1, 2, 3] : (tensor<1x1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %4693 = stablehlo.add %4691, %4692 : tensor<1x12x1x20xf32>
      %4694 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %4695 = stablehlo.reduce(%4693 init: %4694) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %4696 = stablehlo.broadcast_in_dim %4695, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %4697 = stablehlo.broadcast_in_dim %4696, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %4698 = stablehlo.subtract %4693, %4697 : tensor<1x12x1x20xf32>
      %4699 = stablehlo.exponential %4698 : tensor<1x12x1x20xf32>
      %4700 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4701 = stablehlo.reduce(%4699 init: %4700) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %4702 = stablehlo.broadcast_in_dim %4701, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %4703 = stablehlo.broadcast_in_dim %4702, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %4704 = stablehlo.divide %4699, %4703 : tensor<1x12x1x20xf32>
      %4705 = stablehlo.dot_general %4663, %4704, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %4706 = stablehlo.transpose %4705, dims = [0, 3, 1, 2] : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %4707 = stablehlo.reshape %4706 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %4708 = stablehlo.transpose %iterArg_101, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %4709 = stablehlo.convert %4708 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %4710 = stablehlo.dot_general %4707, %4709, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %4711 = stablehlo.convert %iterArg_102 : (tensor<768xf16>) -> tensor<768xf32>
      %4712 = stablehlo.broadcast_in_dim %4711, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4713 = stablehlo.add %4710, %4712 : tensor<1x1x768xf32>
      %4714 = stablehlo.add %4713, %4576 : tensor<1x1x768xf32>
      %4715 = stablehlo.multiply %4714, %4714 : tensor<1x1x768xf32>
      %4716 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4717 = stablehlo.reduce(%4715 init: %4716) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4718 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4719 = stablehlo.broadcast_in_dim %4718, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4720 = stablehlo.divide %4717, %4719 : tensor<1x1xf32>
      %4721 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4722 = stablehlo.reduce(%4714 init: %4721) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4723 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4724 = stablehlo.broadcast_in_dim %4723, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4725 = stablehlo.divide %4722, %4724 : tensor<1x1xf32>
      %4726 = stablehlo.multiply %4725, %4725 : tensor<1x1xf32>
      %4727 = stablehlo.subtract %4720, %4726 : tensor<1x1xf32>
      %4728 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4729 = stablehlo.broadcast_in_dim %4728, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4730 = stablehlo.maximum %4729, %4727 : tensor<1x1xf32>
      %4731 = stablehlo.broadcast_in_dim %4725, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4732 = stablehlo.broadcast_in_dim %4730, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4733 = stablehlo.broadcast_in_dim %4731, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4734 = stablehlo.subtract %4714, %4733 : tensor<1x1x768xf32>
      %4735 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %4736 = stablehlo.broadcast_in_dim %4735, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %4737 = stablehlo.add %4732, %4736 : tensor<1x1x1xf32>
      %4738 = stablehlo.rsqrt %4737 : tensor<1x1x1xf32>
      %4739 = stablehlo.reshape %iterArg_103 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4740 = stablehlo.convert %4739 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4741 = stablehlo.broadcast_in_dim %4738, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4742 = stablehlo.multiply %4741, %4740 : tensor<1x1x768xf32>
      %4743 = stablehlo.multiply %4734, %4742 : tensor<1x1x768xf32>
      %4744 = stablehlo.reshape %iterArg_104 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4745 = stablehlo.convert %4744 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4746 = stablehlo.add %4743, %4745 : tensor<1x1x768xf32>
      %4747 = stablehlo.transpose %iterArg_105, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %4748 = stablehlo.convert %4747 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %4749 = stablehlo.dot_general %4746, %4748, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %4750 = stablehlo.convert %iterArg_106 : (tensor<3072xf16>) -> tensor<3072xf32>
      %4751 = stablehlo.broadcast_in_dim %4750, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %4752 = stablehlo.add %4749, %4751 : tensor<1x1x3072xf32>
      %4753 = stablehlo.multiply %4752, %4752 : tensor<1x1x3072xf32>
      %4754 = stablehlo.multiply %4752, %4753 : tensor<1x1x3072xf32>
      %4755 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
      %4756 = stablehlo.broadcast_in_dim %4755, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4757 = stablehlo.multiply %4756, %4754 : tensor<1x1x3072xf32>
      %4758 = stablehlo.add %4752, %4757 : tensor<1x1x3072xf32>
      %4759 = stablehlo.constant dense<0.797884583> : tensor<f32>
      %4760 = stablehlo.broadcast_in_dim %4759, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4761 = stablehlo.multiply %4760, %4758 : tensor<1x1x3072xf32>
      %4762 = stablehlo.tanh %4761 : tensor<1x1x3072xf32>
      %4763 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %4764 = stablehlo.broadcast_in_dim %4763, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4765 = stablehlo.add %4764, %4762 : tensor<1x1x3072xf32>
      %4766 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %4767 = stablehlo.broadcast_in_dim %4766, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4768 = stablehlo.multiply %4767, %4765 : tensor<1x1x3072xf32>
      %4769 = stablehlo.multiply %4752, %4768 : tensor<1x1x3072xf32>
      %4770 = stablehlo.transpose %iterArg_107, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %4771 = stablehlo.convert %4770 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %4772 = stablehlo.dot_general %4769, %4771, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %4773 = stablehlo.convert %iterArg_108 : (tensor<768xf16>) -> tensor<768xf32>
      %4774 = stablehlo.broadcast_in_dim %4773, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4775 = stablehlo.add %4772, %4774 : tensor<1x1x768xf32>
      %4776 = stablehlo.add %4714, %4775 : tensor<1x1x768xf32>
      %4777 = stablehlo.multiply %4776, %4776 : tensor<1x1x768xf32>
      %4778 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4779 = stablehlo.reduce(%4777 init: %4778) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4780 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4781 = stablehlo.broadcast_in_dim %4780, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4782 = stablehlo.divide %4779, %4781 : tensor<1x1xf32>
      %4783 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4784 = stablehlo.reduce(%4776 init: %4783) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4785 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4786 = stablehlo.broadcast_in_dim %4785, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4787 = stablehlo.divide %4784, %4786 : tensor<1x1xf32>
      %4788 = stablehlo.multiply %4787, %4787 : tensor<1x1xf32>
      %4789 = stablehlo.subtract %4782, %4788 : tensor<1x1xf32>
      %4790 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4791 = stablehlo.broadcast_in_dim %4790, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4792 = stablehlo.maximum %4791, %4789 : tensor<1x1xf32>
      %4793 = stablehlo.broadcast_in_dim %4787, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4794 = stablehlo.broadcast_in_dim %4792, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4795 = stablehlo.broadcast_in_dim %4793, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4796 = stablehlo.subtract %4776, %4795 : tensor<1x1x768xf32>
      %4797 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %4798 = stablehlo.broadcast_in_dim %4797, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %4799 = stablehlo.add %4794, %4798 : tensor<1x1x1xf32>
      %4800 = stablehlo.rsqrt %4799 : tensor<1x1x1xf32>
      %4801 = stablehlo.reshape %iterArg_109 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4802 = stablehlo.convert %4801 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4803 = stablehlo.broadcast_in_dim %4800, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4804 = stablehlo.multiply %4803, %4802 : tensor<1x1x768xf32>
      %4805 = stablehlo.multiply %4796, %4804 : tensor<1x1x768xf32>
      %4806 = stablehlo.reshape %iterArg_110 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4807 = stablehlo.convert %4806 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4808 = stablehlo.add %4805, %4807 : tensor<1x1x768xf32>
      %4809 = stablehlo.constant dense<true> : tensor<i1>
      %4810 = stablehlo.broadcast_in_dim %4809, dims = [] : (tensor<i1>) -> tensor<1x1024xi1>
      %4811 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %4812 = stablehlo.broadcast_in_dim %4811, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %4813 = stablehlo.broadcast_in_dim %4812, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %4814 = stablehlo.broadcast_in_dim %4812, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %4815 = stablehlo.broadcast_in_dim %4813, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %4816 = stablehlo.broadcast_in_dim %4814, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %4817 = stablehlo.compare  GE, %4815, %4816,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %4818 = stablehlo.broadcast_in_dim %4817, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %4819 = stablehlo.transpose %iterArg_111, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %4820 = stablehlo.convert %4819 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %4821 = stablehlo.dot_general %4808, %4820, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32>
      %4822 = stablehlo.convert %iterArg_112 : (tensor<2304xf16>) -> tensor<2304xf32>
      %4823 = stablehlo.broadcast_in_dim %4822, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %4824 = stablehlo.add %4821, %4823 : tensor<1x1x2304xf32>
      %4825 = stablehlo.slice %4824 [0:1, 0:1, 0:768] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4826 = stablehlo.slice %4824 [0:1, 0:1, 768:1536] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4827 = stablehlo.slice %4824 [0:1, 0:1, 1536:2304] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %4828 = stablehlo.reshape %4825 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4829 = stablehlo.reshape %4826 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4830 = stablehlo.reshape %4827 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %4831 = stablehlo.constant dense<0> : tensor<i32>
      %4832 = stablehlo.compare  LT, %iterArg_187, %4831,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4833 = stablehlo.constant dense<1024> : tensor<i32>
      %4834 = stablehlo.add %iterArg_187, %4833 : tensor<i32>
      %4835 = stablehlo.select %4832, %4834, %iterArg_187 : tensor<i1>, tensor<i32>
      %4836 = stablehlo.constant dense<0> : tensor<i32>
      %4837 = stablehlo.constant dense<0> : tensor<i32>
      %4838 = stablehlo.constant dense<0> : tensor<i32>
      %4839 = stablehlo.dynamic_slice %4818, %4836, %4837, %4835, %4838, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %4840 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %4841 = stablehlo.constant dense<0> : tensor<i32>
      %4842 = stablehlo.broadcast_in_dim %4841, dims = [] : (tensor<i32>) -> tensor<1x1x1x20xi32>
      %4843 = stablehlo.compare  NE, %4840, %4842,  SIGNED : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %4844 = stablehlo.and %4843, %4839 : tensor<1x1x1x20xi1>
      %4845 = stablehlo.convert %4844 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %4846 = stablehlo.constant dense<0> : tensor<i32>
      %4847 = stablehlo.compare  LT, %iterArg_187, %4846,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4848 = stablehlo.constant dense<20> : tensor<i32>
      %4849 = stablehlo.add %iterArg_187, %4848 : tensor<i32>
      %4850 = stablehlo.select %4847, %4849, %iterArg_187 : tensor<i1>, tensor<i32>
      %4851 = stablehlo.constant dense<0> : tensor<i32>
      %4852 = stablehlo.constant dense<0> : tensor<i32>
      %4853 = stablehlo.constant dense<0> : tensor<i32>
      %4854 = stablehlo.dynamic_update_slice %iterArg_188, %4829, %4851, %4850, %4852, %4853 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %4855 = stablehlo.constant dense<0> : tensor<i32>
      %4856 = stablehlo.compare  LT, %iterArg_187, %4855,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4857 = stablehlo.constant dense<20> : tensor<i32>
      %4858 = stablehlo.add %iterArg_187, %4857 : tensor<i32>
      %4859 = stablehlo.select %4856, %4858, %iterArg_187 : tensor<i1>, tensor<i32>
      %4860 = stablehlo.constant dense<0> : tensor<i32>
      %4861 = stablehlo.constant dense<0> : tensor<i32>
      %4862 = stablehlo.constant dense<0> : tensor<i32>
      %4863 = stablehlo.dynamic_update_slice %iterArg_189, %4830, %4860, %4859, %4861, %4862 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %4864 = stablehlo.constant dense<1> : tensor<i32>
      %4865 = stablehlo.add %iterArg_187, %4864 : tensor<i32>
      %4866 = stablehlo.iota dim = 0 : tensor<20xi32>
      %4867 = stablehlo.constant dense<1> : tensor<i32>
      %4868 = stablehlo.add %iterArg_187, %4867 : tensor<i32>
      %4869 = stablehlo.broadcast_in_dim %4868, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %4870 = stablehlo.compare  LT, %4866, %4869,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %4871 = stablehlo.broadcast_in_dim %4870, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %4872 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4873 = stablehlo.broadcast_in_dim %4872, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4874 = stablehlo.compare  NE, %4845, %4873,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %4875 = stablehlo.and %4871, %4874 : tensor<1x1x1x20xi1>
      %4876 = stablehlo.convert %4875 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %4877 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4878 = stablehlo.broadcast_in_dim %4877, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4879 = stablehlo.compare  GT, %4876, %4878,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %4880 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4881 = stablehlo.broadcast_in_dim %4880, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4882 = stablehlo.convert %4881 : tensor<1x1x1x20xf32>
      %4883 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %4884 = stablehlo.broadcast_in_dim %4883, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %4885 = stablehlo.select %4879, %4882, %4884 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %4886 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %4887 = stablehlo.sqrt %4886 : tensor<f32>
      %4888 = stablehlo.convert %4887 : tensor<f32>
      %4889 = stablehlo.broadcast_in_dim %4888, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %4890 = stablehlo.divide %4828, %4889 : tensor<1x1x12x64xf32>
      %4891 = stablehlo.dot_general %4890, %4854, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %4892 = stablehlo.broadcast_in_dim %4885, dims = [0, 1, 2, 3] : (tensor<1x1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %4893 = stablehlo.add %4891, %4892 : tensor<1x12x1x20xf32>
      %4894 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %4895 = stablehlo.reduce(%4893 init: %4894) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %4896 = stablehlo.broadcast_in_dim %4895, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %4897 = stablehlo.broadcast_in_dim %4896, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %4898 = stablehlo.subtract %4893, %4897 : tensor<1x12x1x20xf32>
      %4899 = stablehlo.exponential %4898 : tensor<1x12x1x20xf32>
      %4900 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4901 = stablehlo.reduce(%4899 init: %4900) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %4902 = stablehlo.broadcast_in_dim %4901, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %4903 = stablehlo.broadcast_in_dim %4902, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %4904 = stablehlo.divide %4899, %4903 : tensor<1x12x1x20xf32>
      %4905 = stablehlo.dot_general %4863, %4904, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %4906 = stablehlo.transpose %4905, dims = [0, 3, 1, 2] : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %4907 = stablehlo.reshape %4906 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %4908 = stablehlo.transpose %iterArg_113, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %4909 = stablehlo.convert %4908 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %4910 = stablehlo.dot_general %4907, %4909, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %4911 = stablehlo.convert %iterArg_114 : (tensor<768xf16>) -> tensor<768xf32>
      %4912 = stablehlo.broadcast_in_dim %4911, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4913 = stablehlo.add %4910, %4912 : tensor<1x1x768xf32>
      %4914 = stablehlo.add %4913, %4776 : tensor<1x1x768xf32>
      %4915 = stablehlo.multiply %4914, %4914 : tensor<1x1x768xf32>
      %4916 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4917 = stablehlo.reduce(%4915 init: %4916) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4918 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4919 = stablehlo.broadcast_in_dim %4918, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4920 = stablehlo.divide %4917, %4919 : tensor<1x1xf32>
      %4921 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4922 = stablehlo.reduce(%4914 init: %4921) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4923 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4924 = stablehlo.broadcast_in_dim %4923, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4925 = stablehlo.divide %4922, %4924 : tensor<1x1xf32>
      %4926 = stablehlo.multiply %4925, %4925 : tensor<1x1xf32>
      %4927 = stablehlo.subtract %4920, %4926 : tensor<1x1xf32>
      %4928 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4929 = stablehlo.broadcast_in_dim %4928, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4930 = stablehlo.maximum %4929, %4927 : tensor<1x1xf32>
      %4931 = stablehlo.broadcast_in_dim %4925, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4932 = stablehlo.broadcast_in_dim %4930, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4933 = stablehlo.broadcast_in_dim %4931, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4934 = stablehlo.subtract %4914, %4933 : tensor<1x1x768xf32>
      %4935 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %4936 = stablehlo.broadcast_in_dim %4935, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %4937 = stablehlo.add %4932, %4936 : tensor<1x1x1xf32>
      %4938 = stablehlo.rsqrt %4937 : tensor<1x1x1xf32>
      %4939 = stablehlo.reshape %iterArg_115 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4940 = stablehlo.convert %4939 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4941 = stablehlo.broadcast_in_dim %4938, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4942 = stablehlo.multiply %4941, %4940 : tensor<1x1x768xf32>
      %4943 = stablehlo.multiply %4934, %4942 : tensor<1x1x768xf32>
      %4944 = stablehlo.reshape %iterArg_116 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4945 = stablehlo.convert %4944 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4946 = stablehlo.add %4943, %4945 : tensor<1x1x768xf32>
      %4947 = stablehlo.transpose %iterArg_117, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %4948 = stablehlo.convert %4947 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %4949 = stablehlo.dot_general %4946, %4948, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %4950 = stablehlo.convert %iterArg_118 : (tensor<3072xf16>) -> tensor<3072xf32>
      %4951 = stablehlo.broadcast_in_dim %4950, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %4952 = stablehlo.add %4949, %4951 : tensor<1x1x3072xf32>
      %4953 = stablehlo.multiply %4952, %4952 : tensor<1x1x3072xf32>
      %4954 = stablehlo.multiply %4952, %4953 : tensor<1x1x3072xf32>
      %4955 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
      %4956 = stablehlo.broadcast_in_dim %4955, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4957 = stablehlo.multiply %4956, %4954 : tensor<1x1x3072xf32>
      %4958 = stablehlo.add %4952, %4957 : tensor<1x1x3072xf32>
      %4959 = stablehlo.constant dense<0.797884583> : tensor<f32>
      %4960 = stablehlo.broadcast_in_dim %4959, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4961 = stablehlo.multiply %4960, %4958 : tensor<1x1x3072xf32>
      %4962 = stablehlo.tanh %4961 : tensor<1x1x3072xf32>
      %4963 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %4964 = stablehlo.broadcast_in_dim %4963, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4965 = stablehlo.add %4964, %4962 : tensor<1x1x3072xf32>
      %4966 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %4967 = stablehlo.broadcast_in_dim %4966, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %4968 = stablehlo.multiply %4967, %4965 : tensor<1x1x3072xf32>
      %4969 = stablehlo.multiply %4952, %4968 : tensor<1x1x3072xf32>
      %4970 = stablehlo.transpose %iterArg_119, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %4971 = stablehlo.convert %4970 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %4972 = stablehlo.dot_general %4969, %4971, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %4973 = stablehlo.convert %iterArg_120 : (tensor<768xf16>) -> tensor<768xf32>
      %4974 = stablehlo.broadcast_in_dim %4973, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4975 = stablehlo.add %4972, %4974 : tensor<1x1x768xf32>
      %4976 = stablehlo.add %4914, %4975 : tensor<1x1x768xf32>
      %4977 = stablehlo.multiply %4976, %4976 : tensor<1x1x768xf32>
      %4978 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4979 = stablehlo.reduce(%4977 init: %4978) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4980 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4981 = stablehlo.broadcast_in_dim %4980, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4982 = stablehlo.divide %4979, %4981 : tensor<1x1xf32>
      %4983 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4984 = stablehlo.reduce(%4976 init: %4983) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %4985 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %4986 = stablehlo.broadcast_in_dim %4985, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4987 = stablehlo.divide %4984, %4986 : tensor<1x1xf32>
      %4988 = stablehlo.multiply %4987, %4987 : tensor<1x1xf32>
      %4989 = stablehlo.subtract %4982, %4988 : tensor<1x1xf32>
      %4990 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4991 = stablehlo.broadcast_in_dim %4990, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %4992 = stablehlo.maximum %4991, %4989 : tensor<1x1xf32>
      %4993 = stablehlo.broadcast_in_dim %4987, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4994 = stablehlo.broadcast_in_dim %4992, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %4995 = stablehlo.broadcast_in_dim %4993, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %4996 = stablehlo.subtract %4976, %4995 : tensor<1x1x768xf32>
      %4997 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %4998 = stablehlo.broadcast_in_dim %4997, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %4999 = stablehlo.add %4994, %4998 : tensor<1x1x1xf32>
      %5000 = stablehlo.rsqrt %4999 : tensor<1x1x1xf32>
      %5001 = stablehlo.reshape %iterArg_121 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %5002 = stablehlo.convert %5001 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %5003 = stablehlo.broadcast_in_dim %5000, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %5004 = stablehlo.multiply %5003, %5002 : tensor<1x1x768xf32>
      %5005 = stablehlo.multiply %4996, %5004 : tensor<1x1x768xf32>
      %5006 = stablehlo.reshape %iterArg_122 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %5007 = stablehlo.convert %5006 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %5008 = stablehlo.add %5005, %5007 : tensor<1x1x768xf32>
      %5009 = stablehlo.constant dense<true> : tensor<i1>
      %5010 = stablehlo.broadcast_in_dim %5009, dims = [] : (tensor<i1>) -> tensor<1x1024xi1>
      %5011 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %5012 = stablehlo.broadcast_in_dim %5011, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %5013 = stablehlo.broadcast_in_dim %5012, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %5014 = stablehlo.broadcast_in_dim %5012, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %5015 = stablehlo.broadcast_in_dim %5013, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %5016 = stablehlo.broadcast_in_dim %5014, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %5017 = stablehlo.compare  GE, %5015, %5016,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %5018 = stablehlo.broadcast_in_dim %5017, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %5019 = stablehlo.transpose %iterArg_123, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %5020 = stablehlo.convert %5019 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %5021 = stablehlo.dot_general %5008, %5020, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32>
      %5022 = stablehlo.convert %iterArg_124 : (tensor<2304xf16>) -> tensor<2304xf32>
      %5023 = stablehlo.broadcast_in_dim %5022, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %5024 = stablehlo.add %5021, %5023 : tensor<1x1x2304xf32>
      %5025 = stablehlo.slice %5024 [0:1, 0:1, 0:768] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %5026 = stablehlo.slice %5024 [0:1, 0:1, 768:1536] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %5027 = stablehlo.slice %5024 [0:1, 0:1, 1536:2304] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %5028 = stablehlo.reshape %5025 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %5029 = stablehlo.reshape %5026 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %5030 = stablehlo.reshape %5027 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %5031 = stablehlo.constant dense<0> : tensor<i32>
      %5032 = stablehlo.compare  LT, %iterArg_160, %5031,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %5033 = stablehlo.constant dense<1024> : tensor<i32>
      %5034 = stablehlo.add %iterArg_160, %5033 : tensor<i32>
      %5035 = stablehlo.select %5032, %5034, %iterArg_160 : tensor<i1>, tensor<i32>
      %5036 = stablehlo.constant dense<0> : tensor<i32>
      %5037 = stablehlo.constant dense<0> : tensor<i32>
      %5038 = stablehlo.constant dense<0> : tensor<i32>
      %5039 = stablehlo.dynamic_slice %5018, %5036, %5037, %5035, %5038, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %5040 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %5041 = stablehlo.constant dense<0> : tensor<i32>
      %5042 = stablehlo.broadcast_in_dim %5041, dims = [] : (tensor<i32>) -> tensor<1x1x1x20xi32>
      %5043 = stablehlo.compare  NE, %5040, %5042,  SIGNED : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %5044 = stablehlo.and %5043, %5039 : tensor<1x1x1x20xi1>
      %5045 = stablehlo.convert %5044 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %5046 = stablehlo.constant dense<0> : tensor<i32>
      %5047 = stablehlo.compare  LT, %iterArg_160, %5046,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %5048 = stablehlo.constant dense<20> : tensor<i32>
      %5049 = stablehlo.add %iterArg_160, %5048 : tensor<i32>
      %5050 = stablehlo.select %5047, %5049, %iterArg_160 : tensor<i1>, tensor<i32>
      %5051 = stablehlo.constant dense<0> : tensor<i32>
      %5052 = stablehlo.constant dense<0> : tensor<i32>
      %5053 = stablehlo.constant dense<0> : tensor<i32>
      %5054 = stablehlo.dynamic_update_slice %iterArg_161, %5029, %5051, %5050, %5052, %5053 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %5055 = stablehlo.constant dense<0> : tensor<i32>
      %5056 = stablehlo.compare  LT, %iterArg_160, %5055,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %5057 = stablehlo.constant dense<20> : tensor<i32>
      %5058 = stablehlo.add %iterArg_160, %5057 : tensor<i32>
      %5059 = stablehlo.select %5056, %5058, %iterArg_160 : tensor<i1>, tensor<i32>
      %5060 = stablehlo.constant dense<0> : tensor<i32>
      %5061 = stablehlo.constant dense<0> : tensor<i32>
      %5062 = stablehlo.constant dense<0> : tensor<i32>
      %5063 = stablehlo.dynamic_update_slice %iterArg_162, %5030, %5060, %5059, %5061, %5062 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %5064 = stablehlo.constant dense<1> : tensor<i32>
      %5065 = stablehlo.add %iterArg_160, %5064 : tensor<i32>
      %5066 = stablehlo.iota dim = 0 : tensor<20xi32>
      %5067 = stablehlo.constant dense<1> : tensor<i32>
      %5068 = stablehlo.add %iterArg_160, %5067 : tensor<i32>
      %5069 = stablehlo.broadcast_in_dim %5068, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %5070 = stablehlo.compare  LT, %5066, %5069,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %5071 = stablehlo.broadcast_in_dim %5070, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %5072 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5073 = stablehlo.broadcast_in_dim %5072, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %5074 = stablehlo.compare  NE, %5045, %5073,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %5075 = stablehlo.and %5071, %5074 : tensor<1x1x1x20xi1>
      %5076 = stablehlo.convert %5075 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %5077 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5078 = stablehlo.broadcast_in_dim %5077, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %5079 = stablehlo.compare  GT, %5076, %5078,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %5080 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5081 = stablehlo.broadcast_in_dim %5080, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %5082 = stablehlo.convert %5081 : tensor<1x1x1x20xf32>
      %5083 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %5084 = stablehlo.broadcast_in_dim %5083, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %5085 = stablehlo.select %5079, %5082, %5084 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %5086 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %5087 = stablehlo.sqrt %5086 : tensor<f32>
      %5088 = stablehlo.convert %5087 : tensor<f32>
      %5089 = stablehlo.broadcast_in_dim %5088, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %5090 = stablehlo.divide %5028, %5089 : tensor<1x1x12x64xf32>
      %5091 = stablehlo.dot_general %5090, %5054, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %5092 = stablehlo.broadcast_in_dim %5085, dims = [0, 1, 2, 3] : (tensor<1x1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %5093 = stablehlo.add %5091, %5092 : tensor<1x12x1x20xf32>
      %5094 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %5095 = stablehlo.reduce(%5093 init: %5094) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %5096 = stablehlo.broadcast_in_dim %5095, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %5097 = stablehlo.broadcast_in_dim %5096, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %5098 = stablehlo.subtract %5093, %5097 : tensor<1x12x1x20xf32>
      %5099 = stablehlo.exponential %5098 : tensor<1x12x1x20xf32>
      %5100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5101 = stablehlo.reduce(%5099 init: %5100) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %5102 = stablehlo.broadcast_in_dim %5101, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %5103 = stablehlo.broadcast_in_dim %5102, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %5104 = stablehlo.divide %5099, %5103 : tensor<1x12x1x20xf32>
      %5105 = stablehlo.dot_general %5063, %5104, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %5106 = stablehlo.transpose %5105, dims = [0, 3, 1, 2] : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %5107 = stablehlo.reshape %5106 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %5108 = stablehlo.transpose %iterArg_125, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %5109 = stablehlo.convert %5108 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %5110 = stablehlo.dot_general %5107, %5109, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %5111 = stablehlo.convert %iterArg_126 : (tensor<768xf16>) -> tensor<768xf32>
      %5112 = stablehlo.broadcast_in_dim %5111, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %5113 = stablehlo.add %5110, %5112 : tensor<1x1x768xf32>
      %5114 = stablehlo.add %5113, %4976 : tensor<1x1x768xf32>
      %5115 = stablehlo.multiply %5114, %5114 : tensor<1x1x768xf32>
      %5116 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5117 = stablehlo.reduce(%5115 init: %5116) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %5118 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %5119 = stablehlo.broadcast_in_dim %5118, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %5120 = stablehlo.divide %5117, %5119 : tensor<1x1xf32>
      %5121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5122 = stablehlo.reduce(%5114 init: %5121) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %5123 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %5124 = stablehlo.broadcast_in_dim %5123, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %5125 = stablehlo.divide %5122, %5124 : tensor<1x1xf32>
      %5126 = stablehlo.multiply %5125, %5125 : tensor<1x1xf32>
      %5127 = stablehlo.subtract %5120, %5126 : tensor<1x1xf32>
      %5128 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5129 = stablehlo.broadcast_in_dim %5128, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %5130 = stablehlo.maximum %5129, %5127 : tensor<1x1xf32>
      %5131 = stablehlo.broadcast_in_dim %5125, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %5132 = stablehlo.broadcast_in_dim %5130, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %5133 = stablehlo.broadcast_in_dim %5131, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %5134 = stablehlo.subtract %5114, %5133 : tensor<1x1x768xf32>
      %5135 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %5136 = stablehlo.broadcast_in_dim %5135, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %5137 = stablehlo.add %5132, %5136 : tensor<1x1x1xf32>
      %5138 = stablehlo.rsqrt %5137 : tensor<1x1x1xf32>
      %5139 = stablehlo.reshape %iterArg_127 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %5140 = stablehlo.convert %5139 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %5141 = stablehlo.broadcast_in_dim %5138, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %5142 = stablehlo.multiply %5141, %5140 : tensor<1x1x768xf32>
      %5143 = stablehlo.multiply %5134, %5142 : tensor<1x1x768xf32>
      %5144 = stablehlo.reshape %iterArg_128 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %5145 = stablehlo.convert %5144 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %5146 = stablehlo.add %5143, %5145 : tensor<1x1x768xf32>
      %5147 = stablehlo.transpose %iterArg_129, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %5148 = stablehlo.convert %5147 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %5149 = stablehlo.dot_general %5146, %5148, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %5150 = stablehlo.convert %iterArg_130 : (tensor<3072xf16>) -> tensor<3072xf32>
      %5151 = stablehlo.broadcast_in_dim %5150, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %5152 = stablehlo.add %5149, %5151 : tensor<1x1x3072xf32>
      %5153 = stablehlo.multiply %5152, %5152 : tensor<1x1x3072xf32>
      %5154 = stablehlo.multiply %5152, %5153 : tensor<1x1x3072xf32>
      %5155 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
      %5156 = stablehlo.broadcast_in_dim %5155, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %5157 = stablehlo.multiply %5156, %5154 : tensor<1x1x3072xf32>
      %5158 = stablehlo.add %5152, %5157 : tensor<1x1x3072xf32>
      %5159 = stablehlo.constant dense<0.797884583> : tensor<f32>
      %5160 = stablehlo.broadcast_in_dim %5159, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %5161 = stablehlo.multiply %5160, %5158 : tensor<1x1x3072xf32>
      %5162 = stablehlo.tanh %5161 : tensor<1x1x3072xf32>
      %5163 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %5164 = stablehlo.broadcast_in_dim %5163, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %5165 = stablehlo.add %5164, %5162 : tensor<1x1x3072xf32>
      %5166 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %5167 = stablehlo.broadcast_in_dim %5166, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %5168 = stablehlo.multiply %5167, %5165 : tensor<1x1x3072xf32>
      %5169 = stablehlo.multiply %5152, %5168 : tensor<1x1x3072xf32>
      %5170 = stablehlo.transpose %iterArg_131, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %5171 = stablehlo.convert %5170 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %5172 = stablehlo.dot_general %5169, %5171, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %5173 = stablehlo.convert %iterArg_132 : (tensor<768xf16>) -> tensor<768xf32>
      %5174 = stablehlo.broadcast_in_dim %5173, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %5175 = stablehlo.add %5172, %5174 : tensor<1x1x768xf32>
      %5176 = stablehlo.add %5114, %5175 : tensor<1x1x768xf32>
      %5177 = stablehlo.multiply %5176, %5176 : tensor<1x1x768xf32>
      %5178 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5179 = stablehlo.reduce(%5177 init: %5178) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %5180 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %5181 = stablehlo.broadcast_in_dim %5180, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %5182 = stablehlo.divide %5179, %5181 : tensor<1x1xf32>
      %5183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5184 = stablehlo.reduce(%5176 init: %5183) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %5185 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %5186 = stablehlo.broadcast_in_dim %5185, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %5187 = stablehlo.divide %5184, %5186 : tensor<1x1xf32>
      %5188 = stablehlo.multiply %5187, %5187 : tensor<1x1xf32>
      %5189 = stablehlo.subtract %5182, %5188 : tensor<1x1xf32>
      %5190 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5191 = stablehlo.broadcast_in_dim %5190, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %5192 = stablehlo.maximum %5191, %5189 : tensor<1x1xf32>
      %5193 = stablehlo.broadcast_in_dim %5187, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %5194 = stablehlo.broadcast_in_dim %5192, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %5195 = stablehlo.broadcast_in_dim %5193, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %5196 = stablehlo.subtract %5176, %5195 : tensor<1x1x768xf32>
      %5197 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %5198 = stablehlo.broadcast_in_dim %5197, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %5199 = stablehlo.add %5194, %5198 : tensor<1x1x1xf32>
      %5200 = stablehlo.rsqrt %5199 : tensor<1x1x1xf32>
      %5201 = stablehlo.reshape %iterArg_133 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %5202 = stablehlo.convert %5201 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %5203 = stablehlo.broadcast_in_dim %5200, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %5204 = stablehlo.multiply %5203, %5202 : tensor<1x1x768xf32>
      %5205 = stablehlo.multiply %5196, %5204 : tensor<1x1x768xf32>
      %5206 = stablehlo.reshape %iterArg_134 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %5207 = stablehlo.convert %5206 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %5208 = stablehlo.add %5205, %5207 : tensor<1x1x768xf32>
      %5209 = stablehlo.constant dense<true> : tensor<i1>
      %5210 = stablehlo.broadcast_in_dim %5209, dims = [] : (tensor<i1>) -> tensor<1x1024xi1>
      %5211 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %5212 = stablehlo.broadcast_in_dim %5211, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %5213 = stablehlo.broadcast_in_dim %5212, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %5214 = stablehlo.broadcast_in_dim %5212, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %5215 = stablehlo.broadcast_in_dim %5213, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %5216 = stablehlo.broadcast_in_dim %5214, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %5217 = stablehlo.compare  GE, %5215, %5216,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %5218 = stablehlo.broadcast_in_dim %5217, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %5219 = stablehlo.transpose %iterArg_135, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %5220 = stablehlo.convert %5219 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %5221 = stablehlo.dot_general %5208, %5220, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32>
      %5222 = stablehlo.convert %iterArg_136 : (tensor<2304xf16>) -> tensor<2304xf32>
      %5223 = stablehlo.broadcast_in_dim %5222, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %5224 = stablehlo.add %5221, %5223 : tensor<1x1x2304xf32>
      %5225 = stablehlo.slice %5224 [0:1, 0:1, 0:768] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %5226 = stablehlo.slice %5224 [0:1, 0:1, 768:1536] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %5227 = stablehlo.slice %5224 [0:1, 0:1, 1536:2304] : (tensor<1x1x2304xf32>) -> tensor<1x1x768xf32>
      %5228 = stablehlo.reshape %5225 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %5229 = stablehlo.reshape %5226 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %5230 = stablehlo.reshape %5227 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %5231 = stablehlo.constant dense<0> : tensor<i32>
      %5232 = stablehlo.compare  LT, %iterArg_163, %5231,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %5233 = stablehlo.constant dense<1024> : tensor<i32>
      %5234 = stablehlo.add %iterArg_163, %5233 : tensor<i32>
      %5235 = stablehlo.select %5232, %5234, %iterArg_163 : tensor<i1>, tensor<i32>
      %5236 = stablehlo.constant dense<0> : tensor<i32>
      %5237 = stablehlo.constant dense<0> : tensor<i32>
      %5238 = stablehlo.constant dense<0> : tensor<i32>
      %5239 = stablehlo.dynamic_slice %5218, %5236, %5237, %5235, %5238, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %5240 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %5241 = stablehlo.constant dense<0> : tensor<i32>
      %5242 = stablehlo.broadcast_in_dim %5241, dims = [] : (tensor<i32>) -> tensor<1x1x1x20xi32>
      %5243 = stablehlo.compare  NE, %5240, %5242,  SIGNED : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %5244 = stablehlo.and %5243, %5239 : tensor<1x1x1x20xi1>
      %5245 = stablehlo.convert %5244 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %5246 = stablehlo.constant dense<0> : tensor<i32>
      %5247 = stablehlo.compare  LT, %iterArg_163, %5246,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %5248 = stablehlo.constant dense<20> : tensor<i32>
      %5249 = stablehlo.add %iterArg_163, %5248 : tensor<i32>
      %5250 = stablehlo.select %5247, %5249, %iterArg_163 : tensor<i1>, tensor<i32>
      %5251 = stablehlo.constant dense<0> : tensor<i32>
      %5252 = stablehlo.constant dense<0> : tensor<i32>
      %5253 = stablehlo.constant dense<0> : tensor<i32>
      %5254 = stablehlo.dynamic_update_slice %iterArg_164, %5229, %5251, %5250, %5252, %5253 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %5255 = stablehlo.constant dense<0> : tensor<i32>
      %5256 = stablehlo.compare  LT, %iterArg_163, %5255,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %5257 = stablehlo.constant dense<20> : tensor<i32>
      %5258 = stablehlo.add %iterArg_163, %5257 : tensor<i32>
      %5259 = stablehlo.select %5256, %5258, %iterArg_163 : tensor<i1>, tensor<i32>
      %5260 = stablehlo.constant dense<0> : tensor<i32>
      %5261 = stablehlo.constant dense<0> : tensor<i32>
      %5262 = stablehlo.constant dense<0> : tensor<i32>
      %5263 = stablehlo.dynamic_update_slice %iterArg_165, %5230, %5260, %5259, %5261, %5262 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %5264 = stablehlo.constant dense<1> : tensor<i32>
      %5265 = stablehlo.add %iterArg_163, %5264 : tensor<i32>
      %5266 = stablehlo.iota dim = 0 : tensor<20xi32>
      %5267 = stablehlo.constant dense<1> : tensor<i32>
      %5268 = stablehlo.add %iterArg_163, %5267 : tensor<i32>
      %5269 = stablehlo.broadcast_in_dim %5268, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %5270 = stablehlo.compare  LT, %5266, %5269,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %5271 = stablehlo.broadcast_in_dim %5270, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %5272 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5273 = stablehlo.broadcast_in_dim %5272, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %5274 = stablehlo.compare  NE, %5245, %5273,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %5275 = stablehlo.and %5271, %5274 : tensor<1x1x1x20xi1>
      %5276 = stablehlo.convert %5275 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %5277 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5278 = stablehlo.broadcast_in_dim %5277, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %5279 = stablehlo.compare  GT, %5276, %5278,  FLOAT : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %5280 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5281 = stablehlo.broadcast_in_dim %5280, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %5282 = stablehlo.convert %5281 : tensor<1x1x1x20xf32>
      %5283 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %5284 = stablehlo.broadcast_in_dim %5283, dims = [] : (tensor<f32>) -> tensor<1x1x1x20xf32>
      %5285 = stablehlo.select %5279, %5282, %5284 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %5286 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %5287 = stablehlo.sqrt %5286 : tensor<f32>
      %5288 = stablehlo.convert %5287 : tensor<f32>
      %5289 = stablehlo.broadcast_in_dim %5288, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %5290 = stablehlo.divide %5228, %5289 : tensor<1x1x12x64xf32>
      %5291 = stablehlo.dot_general %5290, %5254, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %5292 = stablehlo.broadcast_in_dim %5285, dims = [0, 1, 2, 3] : (tensor<1x1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %5293 = stablehlo.add %5291, %5292 : tensor<1x12x1x20xf32>
      %5294 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %5295 = stablehlo.reduce(%5293 init: %5294) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %5296 = stablehlo.broadcast_in_dim %5295, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %5297 = stablehlo.broadcast_in_dim %5296, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %5298 = stablehlo.subtract %5293, %5297 : tensor<1x12x1x20xf32>
      %5299 = stablehlo.exponential %5298 : tensor<1x12x1x20xf32>
      %5300 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5301 = stablehlo.reduce(%5299 init: %5300) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %5302 = stablehlo.broadcast_in_dim %5301, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %5303 = stablehlo.broadcast_in_dim %5302, dims = [0, 1, 2, 3] : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x20xf32>
      %5304 = stablehlo.divide %5299, %5303 : tensor<1x12x1x20xf32>
      %5305 = stablehlo.dot_general %5263, %5304, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %5306 = stablehlo.transpose %5305, dims = [0, 3, 1, 2] : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %5307 = stablehlo.reshape %5306 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %5308 = stablehlo.transpose %iterArg_137, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %5309 = stablehlo.convert %5308 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %5310 = stablehlo.dot_general %5307, %5309, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %5311 = stablehlo.convert %iterArg_138 : (tensor<768xf16>) -> tensor<768xf32>
      %5312 = stablehlo.broadcast_in_dim %5311, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %5313 = stablehlo.add %5310, %5312 : tensor<1x1x768xf32>
      %5314 = stablehlo.add %5313, %5176 : tensor<1x1x768xf32>
      %5315 = stablehlo.multiply %5314, %5314 : tensor<1x1x768xf32>
      %5316 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5317 = stablehlo.reduce(%5315 init: %5316) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %5318 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %5319 = stablehlo.broadcast_in_dim %5318, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %5320 = stablehlo.divide %5317, %5319 : tensor<1x1xf32>
      %5321 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5322 = stablehlo.reduce(%5314 init: %5321) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %5323 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %5324 = stablehlo.broadcast_in_dim %5323, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %5325 = stablehlo.divide %5322, %5324 : tensor<1x1xf32>
      %5326 = stablehlo.multiply %5325, %5325 : tensor<1x1xf32>
      %5327 = stablehlo.subtract %5320, %5326 : tensor<1x1xf32>
      %5328 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5329 = stablehlo.broadcast_in_dim %5328, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %5330 = stablehlo.maximum %5329, %5327 : tensor<1x1xf32>
      %5331 = stablehlo.broadcast_in_dim %5325, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %5332 = stablehlo.broadcast_in_dim %5330, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %5333 = stablehlo.broadcast_in_dim %5331, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %5334 = stablehlo.subtract %5314, %5333 : tensor<1x1x768xf32>
      %5335 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %5336 = stablehlo.broadcast_in_dim %5335, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %5337 = stablehlo.add %5332, %5336 : tensor<1x1x1xf32>
      %5338 = stablehlo.rsqrt %5337 : tensor<1x1x1xf32>
      %5339 = stablehlo.reshape %iterArg_139 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %5340 = stablehlo.convert %5339 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %5341 = stablehlo.broadcast_in_dim %5338, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %5342 = stablehlo.multiply %5341, %5340 : tensor<1x1x768xf32>
      %5343 = stablehlo.multiply %5334, %5342 : tensor<1x1x768xf32>
      %5344 = stablehlo.reshape %iterArg_140 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %5345 = stablehlo.convert %5344 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %5346 = stablehlo.add %5343, %5345 : tensor<1x1x768xf32>
      %5347 = stablehlo.transpose %iterArg_141, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %5348 = stablehlo.convert %5347 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %5349 = stablehlo.dot_general %5346, %5348, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %5350 = stablehlo.convert %iterArg_142 : (tensor<3072xf16>) -> tensor<3072xf32>
      %5351 = stablehlo.broadcast_in_dim %5350, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %5352 = stablehlo.add %5349, %5351 : tensor<1x1x3072xf32>
      %5353 = stablehlo.multiply %5352, %5352 : tensor<1x1x3072xf32>
      %5354 = stablehlo.multiply %5352, %5353 : tensor<1x1x3072xf32>
      %5355 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
      %5356 = stablehlo.broadcast_in_dim %5355, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %5357 = stablehlo.multiply %5356, %5354 : tensor<1x1x3072xf32>
      %5358 = stablehlo.add %5352, %5357 : tensor<1x1x3072xf32>
      %5359 = stablehlo.constant dense<0.797884583> : tensor<f32>
      %5360 = stablehlo.broadcast_in_dim %5359, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %5361 = stablehlo.multiply %5360, %5358 : tensor<1x1x3072xf32>
      %5362 = stablehlo.tanh %5361 : tensor<1x1x3072xf32>
      %5363 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %5364 = stablehlo.broadcast_in_dim %5363, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %5365 = stablehlo.add %5364, %5362 : tensor<1x1x3072xf32>
      %5366 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %5367 = stablehlo.broadcast_in_dim %5366, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %5368 = stablehlo.multiply %5367, %5365 : tensor<1x1x3072xf32>
      %5369 = stablehlo.multiply %5352, %5368 : tensor<1x1x3072xf32>
      %5370 = stablehlo.transpose %iterArg_143, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %5371 = stablehlo.convert %5370 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %5372 = stablehlo.dot_general %5369, %5371, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %5373 = stablehlo.convert %iterArg_144 : (tensor<768xf16>) -> tensor<768xf32>
      %5374 = stablehlo.broadcast_in_dim %5373, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %5375 = stablehlo.add %5372, %5374 : tensor<1x1x768xf32>
      %5376 = stablehlo.add %5314, %5375 : tensor<1x1x768xf32>
      %5377 = stablehlo.multiply %5376, %5376 : tensor<1x1x768xf32>
      %5378 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5379 = stablehlo.reduce(%5377 init: %5378) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %5380 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %5381 = stablehlo.broadcast_in_dim %5380, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %5382 = stablehlo.divide %5379, %5381 : tensor<1x1xf32>
      %5383 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5384 = stablehlo.reduce(%5376 init: %5383) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %5385 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
      %5386 = stablehlo.broadcast_in_dim %5385, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %5387 = stablehlo.divide %5384, %5386 : tensor<1x1xf32>
      %5388 = stablehlo.multiply %5387, %5387 : tensor<1x1xf32>
      %5389 = stablehlo.subtract %5382, %5388 : tensor<1x1xf32>
      %5390 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5391 = stablehlo.broadcast_in_dim %5390, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
      %5392 = stablehlo.maximum %5391, %5389 : tensor<1x1xf32>
      %5393 = stablehlo.broadcast_in_dim %5387, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %5394 = stablehlo.broadcast_in_dim %5392, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %5395 = stablehlo.broadcast_in_dim %5393, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %5396 = stablehlo.subtract %5376, %5395 : tensor<1x1x768xf32>
      %5397 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %5398 = stablehlo.broadcast_in_dim %5397, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %5399 = stablehlo.add %5394, %5398 : tensor<1x1x1xf32>
      %5400 = stablehlo.rsqrt %5399 : tensor<1x1x1xf32>
      %5401 = stablehlo.reshape %iterArg_145 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %5402 = stablehlo.convert %5401 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %5403 = stablehlo.broadcast_in_dim %5400, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x768xf32>
      %5404 = stablehlo.multiply %5403, %5402 : tensor<1x1x768xf32>
      %5405 = stablehlo.multiply %5396, %5404 : tensor<1x1x768xf32>
      %5406 = stablehlo.reshape %iterArg_146 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %5407 = stablehlo.convert %5406 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %5408 = stablehlo.add %5405, %5407 : tensor<1x1x768xf32>
      %5409 = stablehlo.transpose %iterArg, dims = [1, 0] : (tensor<50257x768xf16>) -> tensor<768x50257xf16>
      %5410 = stablehlo.convert %5409 : (tensor<768x50257xf16>) -> tensor<768x50257xf32>
      %5411 = stablehlo.dot_general %5408, %5410, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x50257xf32>) -> tensor<1x1x50257xf32>
      %5412 = stablehlo.constant dense<0> : tensor<i32>
      %5413 = stablehlo.constant dense<0> : tensor<i32>
      %5414 = stablehlo.compare  LT, %5412, %5413,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %5415 = stablehlo.constant dense<0> : tensor<i32>
      %5416 = stablehlo.constant dense<1> : tensor<i32>
      %5417 = stablehlo.add %5415, %5416 : tensor<i32>
      %5418 = stablehlo.constant dense<0> : tensor<i32>
      %5419 = stablehlo.select %5414, %5417, %5418 : tensor<i1>, tensor<i32>
      %5420 = stablehlo.constant dense<-1> : tensor<i32>
      %5421 = stablehlo.constant dense<0> : tensor<i32>
      %5422 = stablehlo.compare  LT, %5420, %5421,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %5423 = stablehlo.constant dense<-1> : tensor<i32>
      %5424 = stablehlo.constant dense<1> : tensor<i32>
      %5425 = stablehlo.add %5423, %5424 : tensor<i32>
      %5426 = stablehlo.constant dense<-1> : tensor<i32>
      %5427 = stablehlo.select %5422, %5425, %5426 : tensor<i1>, tensor<i32>
      %5428 = stablehlo.constant dense<0> : tensor<i32>
      %5429 = stablehlo.constant dense<0> : tensor<i32>
      %5430 = stablehlo.compare  LT, %5428, %5429,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %5431 = stablehlo.constant dense<0> : tensor<i32>
      %5432 = stablehlo.constant dense<50257> : tensor<i32>
      %5433 = stablehlo.add %5431, %5432 : tensor<i32>
      %5434 = stablehlo.constant dense<0> : tensor<i32>
      %5435 = stablehlo.select %5430, %5433, %5434 : tensor<i1>, tensor<i32>
      %5436 = stablehlo.dynamic_slice %5411, %5419, %5427, %5435, sizes = [1, 1, 50257] : (tensor<1x1x50257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x50257xf32>
      %5437 = stablehlo.reshape %5436 : (tensor<1x1x50257xf32>) -> tensor<1x50257xf32>
      %5438 = stablehlo.constant dense<0> : tensor<i32>
      %5439 = stablehlo.subtract %iterArg_149, %5438 : tensor<i32>
      %5440 = stablehlo.constant dense<0> : tensor<i32>
      %5441 = stablehlo.constant dense<1> : tensor<i32>
      %5442 = func.call @clip_7(%5439, %5440, %5441) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %5443 = stablehlo.constant dense<1> : tensor<i32>
      %5444 = stablehlo.subtract %5443, %5442 : tensor<i32>
      %5445 = stablehlo.constant dense<50256> : tensor<i32>
      %5446 = stablehlo.broadcast_in_dim %5445, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %5447 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %5448 = stablehlo.broadcast_in_dim %5447, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %5449 = "stablehlo.scatter"(%5437, %5446, %5448) ({
      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        stablehlo.return %arg3 : tensor<f32>
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50257xf32>, tensor<1xi32>, tensor<1xf32>) -> tensor<1x50257xf32>
      %5450 = func.call @_where_8(%5444, %5449, %5437) : (tensor<i32>, tensor<1x50257xf32>, tensor<1x50257xf32>) -> tensor<1x50257xf32>
      %5451 = func.call @argmax(%5450) : (tensor<1x50257xf32>) -> tensor<1xi32>
      %5452 = stablehlo.not %iterArg_152 : tensor<1xi1>
      %5453 = stablehlo.convert %5452 : (tensor<1xi1>) -> tensor<1xi32>
      %5454 = stablehlo.multiply %5451, %5453 : tensor<1xi32>
      %5455 = stablehlo.convert %iterArg_152 : (tensor<1xi1>) -> tensor<1xi32>
      %5456 = stablehlo.broadcast_in_dim %iterArg_147, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %5457 = stablehlo.multiply %5456, %5455 : tensor<1xi32>
      %5458 = stablehlo.add %5454, %5457 : tensor<1xi32>
      %5459 = stablehlo.broadcast_in_dim %iterArg_148, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %5460 = stablehlo.compare  EQ, %5458, %5459,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      %5461 = stablehlo.or %iterArg_152, %5460 : tensor<1xi1>
      %5462 = stablehlo.broadcast_in_dim %5458, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
      %5463 = stablehlo.constant dense<0> : tensor<i32>
      %5464 = stablehlo.compare  LT, %iterArg_149, %5463,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %5465 = stablehlo.convert %iterArg_149 : tensor<i32>
      %5466 = stablehlo.constant dense<20> : tensor<i32>
      %5467 = stablehlo.add %5465, %5466 : tensor<i32>
      %5468 = stablehlo.select %5464, %5467, %iterArg_149 : tensor<i1>, tensor<i32>
      %5469 = stablehlo.constant dense<0> : tensor<i32>
      %5470 = stablehlo.dynamic_update_slice %iterArg_150, %5462, %5469, %5468 : (tensor<1x20xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x20xi32>
      %5471 = stablehlo.constant dense<1> : tensor<i32>
      %5472 = stablehlo.broadcast_in_dim %5471, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
      %5473 = stablehlo.add %iterArg_190, %5472 : tensor<1x1xi32>
      %5474 = stablehlo.constant dense<1> : tensor<i32>
      %5475 = stablehlo.add %iterArg_149, %5474 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_0, %iterArg_1, %iterArg_2, %iterArg_3, %iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %iterArg_13, %iterArg_14, %iterArg_15, %iterArg_16, %iterArg_17, %iterArg_18, %iterArg_19, %iterArg_20, %iterArg_21, %iterArg_22, %iterArg_23, %iterArg_24, %iterArg_25, %iterArg_26, %iterArg_27, %iterArg_28, %iterArg_29, %iterArg_30, %iterArg_31, %iterArg_32, %iterArg_33, %iterArg_34, %iterArg_35, %iterArg_36, %iterArg_37, %iterArg_38, %iterArg_39, %iterArg_40, %iterArg_41, %iterArg_42, %iterArg_43, %iterArg_44, %iterArg_45, %iterArg_46, %iterArg_47, %iterArg_48, %iterArg_49, %iterArg_50, %iterArg_51, %iterArg_52, %iterArg_53, %iterArg_54, %iterArg_55, %iterArg_56, %iterArg_57, %iterArg_58, %iterArg_59, %iterArg_60, %iterArg_61, %iterArg_62, %iterArg_63, %iterArg_64, %iterArg_65, %iterArg_66, %iterArg_67, %iterArg_68, %iterArg_69, %iterArg_70, %iterArg_71, %iterArg_72, %iterArg_73, %iterArg_74, %iterArg_75, %iterArg_76, %iterArg_77, %iterArg_78, %iterArg_79, %iterArg_80, %iterArg_81, %iterArg_82, %iterArg_83, %iterArg_84, %iterArg_85, %iterArg_86, %iterArg_87, %iterArg_88, %iterArg_89, %iterArg_90, %iterArg_91, %iterArg_92, %iterArg_93, %iterArg_94, %iterArg_95, %iterArg_96, %iterArg_97, %iterArg_98, %iterArg_99, %iterArg_100, %iterArg_101, %iterArg_102, %iterArg_103, %iterArg_104, %iterArg_105, %iterArg_106, %iterArg_107, %iterArg_108, %iterArg_109, %iterArg_110, %iterArg_111, %iterArg_112, %iterArg_113, %iterArg_114, %iterArg_115, %iterArg_116, %iterArg_117, %iterArg_118, %iterArg_119, %iterArg_120, %iterArg_121, %iterArg_122, %iterArg_123, %iterArg_124, %iterArg_125, %iterArg_126, %iterArg_127, %iterArg_128, %iterArg_129, %iterArg_130, %iterArg_131, %iterArg_132, %iterArg_133, %iterArg_134, %iterArg_135, %iterArg_136, %iterArg_137, %iterArg_138, %iterArg_139, %iterArg_140, %iterArg_141, %iterArg_142, %iterArg_143, %iterArg_144, %iterArg_145, %iterArg_146, %iterArg_147, %iterArg_148, %5475, %5470, %5462, %5461, %iterArg_153, %3065, %3054, %3063, %3265, %3254, %3263, %5065, %5054, %5063, %5265, %5254, %5263, %3465, %3454, %3463, %3665, %3654, %3663, %3865, %3854, %3863, %4065, %4054, %4063, %4265, %4254, %4263, %4465, %4454, %4463, %4665, %4654, %4663, %4865, %4854, %4863, %5473 : tensor<50257x768xf16>, tensor<1024x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x20xi32>, tensor<1x1xi32>, tensor<1xi1>, tensor<1x20xi32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<1x1xi32>
    }
    return %2971#151 : tensor<1x20xi32>
  }
  func.func private @_cumulative_reduction(%arg0: tensor<1x7xi32>) -> tensor<1x7xi32> {
    %0 = call @cumsum(%arg0) : (tensor<1x7xi32>) -> tensor<1x7xi32>
    return %0 : tensor<1x7xi32>
  }
  func.func private @cumsum(%arg0: tensor<1x7xi32>) -> tensor<1x7xi32> {
    %0 = stablehlo.slice %arg0 [0:1, 0:6:2] : (tensor<1x7xi32>) -> tensor<1x3xi32>
    %1 = stablehlo.slice %arg0 [0:1, 1:7:2] : (tensor<1x7xi32>) -> tensor<1x3xi32>
    %2 = stablehlo.add %0, %1 : tensor<1x3xi32>
    %3 = stablehlo.slice %2 [0:1, 0:2:2] : (tensor<1x3xi32>) -> tensor<1x1xi32>
    %4 = stablehlo.slice %2 [0:1, 1:3:2] : (tensor<1x3xi32>) -> tensor<1x1xi32>
    %5 = stablehlo.add %3, %4 : tensor<1x1xi32>
    %6 = stablehlo.slice %2 [0:1, 2:3:2] : (tensor<1x3xi32>) -> tensor<1x1xi32>
    %7 = stablehlo.add %5, %6 : tensor<1x1xi32>
    %8 = stablehlo.slice %2 [0:1, 0:1] : (tensor<1x3xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.concatenate %8, %7, dim = 1 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
    %10 = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.pad %9, %10, low = [0, 0], high = [0, 0], interior = [0, 1] : (tensor<1x2xi32>, tensor<i32>) -> tensor<1x3xi32>
    %12 = stablehlo.constant dense<0> : tensor<i32>
    %13 = stablehlo.pad %5, %12, low = [0, 1], high = [0, 1], interior = [0, 1] : (tensor<1x1xi32>, tensor<i32>) -> tensor<1x3xi32>
    %14 = stablehlo.add %11, %13 : tensor<1x3xi32>
    %15 = stablehlo.slice %arg0 [0:1, 2:7:2] : (tensor<1x7xi32>) -> tensor<1x3xi32>
    %16 = stablehlo.add %14, %15 : tensor<1x3xi32>
    %17 = stablehlo.slice %arg0 [0:1, 0:1] : (tensor<1x7xi32>) -> tensor<1x1xi32>
    %18 = stablehlo.concatenate %17, %16, dim = 1 : (tensor<1x1xi32>, tensor<1x3xi32>) -> tensor<1x4xi32>
    %19 = stablehlo.constant dense<0> : tensor<i32>
    %20 = stablehlo.pad %18, %19, low = [0, 0], high = [0, 0], interior = [0, 1] : (tensor<1x4xi32>, tensor<i32>) -> tensor<1x7xi32>
    %21 = stablehlo.constant dense<0> : tensor<i32>
    %22 = stablehlo.pad %14, %21, low = [0, 1], high = [0, 1], interior = [0, 1] : (tensor<1x3xi32>, tensor<i32>) -> tensor<1x7xi32>
    %23 = stablehlo.add %20, %22 : tensor<1x7xi32>
    return %23 : tensor<1x7xi32>
  }
  func.func private @_take(%arg0: tensor<50257x768xf32>, %arg1: tensor<1x7xi32>) -> tensor<1x7x768xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1x7xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<1x7xi32>, tensor<1x7xi32>) -> tensor<1x7xi1>
    %3 = stablehlo.constant dense<50257> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<1x7xi32>
    %5 = stablehlo.add %arg1, %4 : tensor<1x7xi32>
    %6 = call @_where(%2, %5, %arg1) : (tensor<1x7xi1>, tensor<1x7xi32>, tensor<1x7xi32>) -> tensor<1x7xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x7xi32>) -> tensor<1x7x1xi32>
    %8 = stablehlo.constant dense<0> : tensor<1xi32>
    %9 = stablehlo.constant dense<0> : tensor<1xi32>
    %10 = stablehlo.constant dense<50257> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.constant dense<768> : tensor<i32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %14 = stablehlo.concatenate %11, %13, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %15 = stablehlo.constant dense<0> : tensor<i32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.compare  LT, %8, %16,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %18 = stablehlo.constant dense<2> : tensor<i32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %20 = stablehlo.add %8, %19 : tensor<1xi32>
    %21 = stablehlo.select %17, %20, %8 : tensor<1xi1>, tensor<1xi32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %23 = "stablehlo.gather"(%14, %22) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %24 = stablehlo.constant dense<1> : tensor<i32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %26 = stablehlo.constant dense<768> : tensor<i32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %28 = stablehlo.concatenate %25, %27, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %29 = stablehlo.constant dense<0> : tensor<i32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %31 = stablehlo.compare  LT, %9, %30,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %32 = stablehlo.constant dense<2> : tensor<i32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %34 = stablehlo.add %9, %33 : tensor<1xi32>
    %35 = stablehlo.select %31, %34, %9 : tensor<1xi1>, tensor<1xi32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %37 = "stablehlo.gather"(%28, %36) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %38 = stablehlo.subtract %23, %37 : tensor<1xi32>
    %39 = stablehlo.constant dense<0> : tensor<i32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1x7x1xi32>
    %41 = stablehlo.compare  GE, %7, %40,  SIGNED : (tensor<1x7x1xi32>, tensor<1x7x1xi32>) -> tensor<1x7x1xi1>
    %42 = stablehlo.broadcast_in_dim %38, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x7x1xi32>
    %44 = stablehlo.compare  LE, %7, %43,  SIGNED : (tensor<1x7x1xi32>, tensor<1x7x1xi32>) -> tensor<1x7x1xi1>
    %45 = stablehlo.and %41, %44 : tensor<1x7x1xi1>
    %46 = stablehlo.constant dense<true> : tensor<i1>
    %47 = stablehlo.reduce(%45 init: %46) applies stablehlo.and across dimensions = [2] : (tensor<1x7x1xi1>, tensor<i1>) -> tensor<1x7xi1>
    %48 = "stablehlo.gather"(%arg0, %7) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>} : (tensor<50257x768xf32>, tensor<1x7x1xi32>) -> tensor<1x7x768xf32>
    %49 = stablehlo.broadcast_in_dim %47, dims = [0, 1] : (tensor<1x7xi1>) -> tensor<1x7x768xi1>
    %50 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %52 = stablehlo.select %49, %48, %51 : tensor<1x7x768xi1>, tensor<1x7x768xf32>
    return %52 : tensor<1x7x768xf32>
  }
  func.func private @_where(%arg0: tensor<1x7xi1>, %arg1: tensor<1x7xi32>, %arg2: tensor<1x7xi32>) -> tensor<1x7xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x7xi1>, tensor<1x7xi32>
    return %0 : tensor<1x7xi32>
  }
  func.func private @_take_0(%arg0: tensor<1024x768xf32>, %arg1: tensor<1x7xi32>) -> tensor<1x7x768xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1x7xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<1x7xi32>, tensor<1x7xi32>) -> tensor<1x7xi1>
    %3 = stablehlo.constant dense<1024> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<1x7xi32>
    %5 = stablehlo.add %arg1, %4 : tensor<1x7xi32>
    %6 = call @_where_1(%2, %5, %arg1) : (tensor<1x7xi1>, tensor<1x7xi32>, tensor<1x7xi32>) -> tensor<1x7xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x7xi32>) -> tensor<1x7x1xi32>
    %8 = stablehlo.constant dense<0> : tensor<1xi32>
    %9 = stablehlo.constant dense<0> : tensor<1xi32>
    %10 = stablehlo.constant dense<1024> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.constant dense<768> : tensor<i32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %14 = stablehlo.concatenate %11, %13, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %15 = stablehlo.constant dense<0> : tensor<i32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.compare  LT, %8, %16,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %18 = stablehlo.constant dense<2> : tensor<i32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %20 = stablehlo.add %8, %19 : tensor<1xi32>
    %21 = stablehlo.select %17, %20, %8 : tensor<1xi1>, tensor<1xi32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %23 = "stablehlo.gather"(%14, %22) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %24 = stablehlo.constant dense<1> : tensor<i32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %26 = stablehlo.constant dense<768> : tensor<i32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %28 = stablehlo.concatenate %25, %27, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %29 = stablehlo.constant dense<0> : tensor<i32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %31 = stablehlo.compare  LT, %9, %30,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %32 = stablehlo.constant dense<2> : tensor<i32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %34 = stablehlo.add %9, %33 : tensor<1xi32>
    %35 = stablehlo.select %31, %34, %9 : tensor<1xi1>, tensor<1xi32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %37 = "stablehlo.gather"(%28, %36) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %38 = stablehlo.subtract %23, %37 : tensor<1xi32>
    %39 = stablehlo.constant dense<0> : tensor<i32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1x7x1xi32>
    %41 = stablehlo.compare  GE, %7, %40,  SIGNED : (tensor<1x7x1xi32>, tensor<1x7x1xi32>) -> tensor<1x7x1xi1>
    %42 = stablehlo.broadcast_in_dim %38, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x7x1xi32>
    %44 = stablehlo.compare  LE, %7, %43,  SIGNED : (tensor<1x7x1xi32>, tensor<1x7x1xi32>) -> tensor<1x7x1xi1>
    %45 = stablehlo.and %41, %44 : tensor<1x7x1xi1>
    %46 = stablehlo.constant dense<true> : tensor<i1>
    %47 = stablehlo.reduce(%45 init: %46) applies stablehlo.and across dimensions = [2] : (tensor<1x7x1xi1>, tensor<i1>) -> tensor<1x7xi1>
    %48 = "stablehlo.gather"(%arg0, %7) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>} : (tensor<1024x768xf32>, tensor<1x7x1xi32>) -> tensor<1x7x768xf32>
    %49 = stablehlo.broadcast_in_dim %47, dims = [0, 1] : (tensor<1x7xi1>) -> tensor<1x7x768xi1>
    %50 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %52 = stablehlo.select %49, %48, %51 : tensor<1x7x768xi1>, tensor<1x7x768xf32>
    return %52 : tensor<1x7x768xf32>
  }
  func.func private @_where_1(%arg0: tensor<1x7xi1>, %arg1: tensor<1x7xi32>, %arg2: tensor<1x7xi32>) -> tensor<1x7xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x7xi1>, tensor<1x7xi32>
    return %0 : tensor<1x7xi32>
  }
  func.func private @clip(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<i32>
    %1 = stablehlo.minimum %arg2, %0 : tensor<i32>
    return %1 : tensor<i32>
  }
  func.func private @_where_2(%arg0: tensor<i32>, %arg1: tensor<1x50257xf32>, %arg2: tensor<1x50257xf32>) -> tensor<1x50257xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.compare  NE, %arg0, %0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i1>) -> tensor<1x50257xi1>
    %3 = stablehlo.select %2, %arg1, %arg2 : tensor<1x50257xi1>, tensor<1x50257xf32>
    return %3 : tensor<1x50257xf32>
  }
  func.func private @argmax(%arg0: tensor<1x50257xf32>) -> tensor<1xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<1x50257xi32>
    %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [1] : (tensor<1x50257xf32>, tensor<1x50257xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %4 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %5 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.or %4, %5 : tensor<i1>
      %7 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %8 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = stablehlo.and %7, %8 : tensor<i1>
      %10 = stablehlo.or %6, %9 : tensor<i1>
      %11 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %12 = stablehlo.select %10, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %11, %12 : tensor<f32>, tensor<i32>
    }
    return %3#1 : tensor<1xi32>
  }
  func.func private @_take_3(%arg0: tensor<50257x768xf32>, %arg1: tensor<1x1xi32>) -> tensor<1x1x768xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %3 = stablehlo.constant dense<50257> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %5 = stablehlo.add %arg1, %4 : tensor<1x1xi32>
    %6 = call @_where_4(%2, %5, %arg1) : (tensor<1x1xi1>, tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
    %8 = stablehlo.constant dense<0> : tensor<1xi32>
    %9 = stablehlo.constant dense<0> : tensor<1xi32>
    %10 = stablehlo.constant dense<50257> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.constant dense<768> : tensor<i32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %14 = stablehlo.concatenate %11, %13, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %15 = stablehlo.constant dense<0> : tensor<i32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.compare  LT, %8, %16,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %18 = stablehlo.constant dense<2> : tensor<i32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %20 = stablehlo.add %8, %19 : tensor<1xi32>
    %21 = stablehlo.select %17, %20, %8 : tensor<1xi1>, tensor<1xi32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %23 = "stablehlo.gather"(%14, %22) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %24 = stablehlo.constant dense<1> : tensor<i32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %26 = stablehlo.constant dense<768> : tensor<i32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %28 = stablehlo.concatenate %25, %27, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %29 = stablehlo.constant dense<0> : tensor<i32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %31 = stablehlo.compare  LT, %9, %30,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %32 = stablehlo.constant dense<2> : tensor<i32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %34 = stablehlo.add %9, %33 : tensor<1xi32>
    %35 = stablehlo.select %31, %34, %9 : tensor<1xi1>, tensor<1xi32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %37 = "stablehlo.gather"(%28, %36) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %38 = stablehlo.subtract %23, %37 : tensor<1xi32>
    %39 = stablehlo.constant dense<0> : tensor<i32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
    %41 = stablehlo.compare  GE, %7, %40,  SIGNED : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi1>
    %42 = stablehlo.broadcast_in_dim %38, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %43 = stablehlo.compare  LE, %7, %42,  SIGNED : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi1>
    %44 = stablehlo.and %41, %43 : tensor<1x1x1xi1>
    %45 = stablehlo.constant dense<true> : tensor<i1>
    %46 = stablehlo.reduce(%44 init: %45) applies stablehlo.and across dimensions = [2] : (tensor<1x1x1xi1>, tensor<i1>) -> tensor<1x1xi1>
    %47 = "stablehlo.gather"(%arg0, %7) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>} : (tensor<50257x768xf32>, tensor<1x1x1xi32>) -> tensor<1x1x768xf32>
    %48 = stablehlo.broadcast_in_dim %46, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<1x1x768xi1>
    %49 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %49, dims = [] : (tensor<f32>) -> tensor<1x1x768xf32>
    %51 = stablehlo.select %48, %47, %50 : tensor<1x1x768xi1>, tensor<1x1x768xf32>
    return %51 : tensor<1x1x768xf32>
  }
  func.func private @_where_4(%arg0: tensor<1x1xi1>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x1xi32>) -> tensor<1x1xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x1xi1>, tensor<1x1xi32>
    return %0 : tensor<1x1xi32>
  }
  func.func private @_take_5(%arg0: tensor<1024x768xf32>, %arg1: tensor<1x1xi32>) -> tensor<1x1x768xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %3 = stablehlo.constant dense<1024> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %5 = stablehlo.add %arg1, %4 : tensor<1x1xi32>
    %6 = call @_where_6(%2, %5, %arg1) : (tensor<1x1xi1>, tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
    %8 = stablehlo.constant dense<0> : tensor<1xi32>
    %9 = stablehlo.constant dense<0> : tensor<1xi32>
    %10 = stablehlo.constant dense<1024> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.constant dense<768> : tensor<i32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %14 = stablehlo.concatenate %11, %13, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %15 = stablehlo.constant dense<0> : tensor<i32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.compare  LT, %8, %16,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %18 = stablehlo.constant dense<2> : tensor<i32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %20 = stablehlo.add %8, %19 : tensor<1xi32>
    %21 = stablehlo.select %17, %20, %8 : tensor<1xi1>, tensor<1xi32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %23 = "stablehlo.gather"(%14, %22) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %24 = stablehlo.constant dense<1> : tensor<i32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %26 = stablehlo.constant dense<768> : tensor<i32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %28 = stablehlo.concatenate %25, %27, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %29 = stablehlo.constant dense<0> : tensor<i32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %31 = stablehlo.compare  LT, %9, %30,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %32 = stablehlo.constant dense<2> : tensor<i32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %34 = stablehlo.add %9, %33 : tensor<1xi32>
    %35 = stablehlo.select %31, %34, %9 : tensor<1xi1>, tensor<1xi32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %37 = "stablehlo.gather"(%28, %36) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %38 = stablehlo.subtract %23, %37 : tensor<1xi32>
    %39 = stablehlo.constant dense<0> : tensor<i32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
    %41 = stablehlo.compare  GE, %7, %40,  SIGNED : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi1>
    %42 = stablehlo.broadcast_in_dim %38, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %43 = stablehlo.compare  LE, %7, %42,  SIGNED : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi1>
    %44 = stablehlo.and %41, %43 : tensor<1x1x1xi1>
    %45 = stablehlo.constant dense<true> : tensor<i1>
    %46 = stablehlo.reduce(%44 init: %45) applies stablehlo.and across dimensions = [2] : (tensor<1x1x1xi1>, tensor<i1>) -> tensor<1x1xi1>
    %47 = "stablehlo.gather"(%arg0, %7) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>} : (tensor<1024x768xf32>, tensor<1x1x1xi32>) -> tensor<1x1x768xf32>
    %48 = stablehlo.broadcast_in_dim %46, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<1x1x768xi1>
    %49 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %49, dims = [] : (tensor<f32>) -> tensor<1x1x768xf32>
    %51 = stablehlo.select %48, %47, %50 : tensor<1x1x768xi1>, tensor<1x1x768xf32>
    return %51 : tensor<1x1x768xf32>
  }
  func.func private @_where_6(%arg0: tensor<1x1xi1>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x1xi32>) -> tensor<1x1xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x1xi1>, tensor<1x1xi32>
    return %0 : tensor<1x1xi32>
  }
  func.func private @clip_7(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<i32>
    %1 = stablehlo.minimum %arg2, %0 : tensor<i32>
    return %1 : tensor<i32>
  }
  func.func private @_where_8(%arg0: tensor<i32>, %arg1: tensor<1x50257xf32>, %arg2: tensor<1x50257xf32>) -> tensor<1x50257xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.compare  NE, %arg0, %0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i1>) -> tensor<1x50257xi1>
    %3 = stablehlo.select %2, %arg1, %arg2 : tensor<1x50257xi1>, tensor<1x50257xf32>
    return %3 : tensor<1x50257xf32>
  }
}
