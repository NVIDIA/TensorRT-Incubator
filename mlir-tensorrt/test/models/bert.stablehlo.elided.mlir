module @jit__unnamed_wrapped_function_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<32x8xi32> {mhlo.layout_mode = "default"}) -> (tensor<32x8x768xf16> {mhlo.layout_mode = "default"}, tensor<32x768xf16> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.constant dense_resource<__elided__> : tensor<30522x768xf32>
    %1 = stablehlo.constant dense_resource<__elided__> : tensor<512x768xf32>
    %2 = stablehlo.constant dense_resource<__elided__> : tensor<2x768xf32>
    %3 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %4 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %5 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %6 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %7 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %8 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %9 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %10 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %11 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %12 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %13 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %14 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %15 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %16 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf32>
    %17 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %18 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %19 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %20 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %21 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %22 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %23 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %24 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %25 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %26 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %27 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %28 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %29 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %30 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %31 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %32 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf32>
    %33 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %34 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %35 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %36 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %37 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %38 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %39 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %40 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %41 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %42 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %43 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %44 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %45 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %46 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %47 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %48 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf32>
    %49 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %50 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %51 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %52 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %53 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %54 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %55 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %56 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %57 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %58 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %59 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %60 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %61 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %62 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %63 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %64 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf32>
    %65 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %66 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %67 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %68 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %69 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %70 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %71 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %72 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %73 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %74 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %75 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %76 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %77 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %78 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %79 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %80 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf32>
    %81 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %82 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %83 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %84 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %85 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %86 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %87 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %88 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %89 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %90 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %91 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %92 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %93 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %94 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %95 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %96 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf32>
    %97 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %98 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %99 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %100 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %101 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %102 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %103 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %104 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %105 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %106 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %107 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %108 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %109 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %110 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %111 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %112 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf32>
    %113 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %114 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %115 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %116 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %117 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %118 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %119 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %120 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %121 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %122 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %123 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %124 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %125 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %126 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %127 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %128 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf32>
    %129 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %130 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %131 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %132 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %133 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %134 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %135 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %136 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %137 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %138 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %139 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %140 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %141 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %142 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %143 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %144 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf32>
    %145 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %146 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %147 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %148 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %149 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %150 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %151 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %152 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %153 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %154 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %155 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %156 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %157 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %158 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %159 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %160 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf32>
    %161 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %162 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %163 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %164 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %165 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %166 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %167 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %168 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %169 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %170 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %171 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %172 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %173 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %174 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %175 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %176 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf32>
    %177 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %178 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %179 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %180 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %181 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %182 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %183 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %184 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %185 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %186 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %187 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %188 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %189 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %190 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %191 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %192 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf32>
    %193 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %194 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %195 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %196 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %197 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %198 = stablehlo.constant dense<0.000000e+00> : tensor<768xf32>
    %199 = stablehlo.constant dense<0> : tensor<i32>
    %200 = stablehlo.broadcast_in_dim %199, dims = [] : (tensor<i32>) -> tensor<32x8xi32>
    %201 = stablehlo.iota dim = 0 : tensor<8xi32>
    %202 = stablehlo.broadcast_in_dim %201, dims = [1] : (tensor<8xi32>) -> tensor<32x8xi32>
    %203 = stablehlo.constant dense<1> : tensor<i32>
    %204 = stablehlo.broadcast_in_dim %203, dims = [] : (tensor<i32>) -> tensor<32x8xi32>
    %205 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [] : (tensor<f32>) -> tensor<12x12xf32>
    %207 = stablehlo.convert %206 : (tensor<12x12xf32>) -> tensor<12x12xi32>
    %208 = stablehlo.convert %0 : (tensor<30522x768xf32>) -> tensor<30522x768xf16>
    %209 = call @_take(%208, %arg0) : (tensor<30522x768xf16>, tensor<32x8xi32>) -> tensor<32x8x768xf16>
    %210 = stablehlo.convert %1 : (tensor<512x768xf32>) -> tensor<512x768xf16>
    %211 = call @_take_0(%210, %202) : (tensor<512x768xf16>, tensor<32x8xi32>) -> tensor<32x8x768xf16>
    %212 = stablehlo.convert %2 : (tensor<2x768xf32>) -> tensor<2x768xf16>
    %213 = call @_take_1(%212, %200) : (tensor<2x768xf16>, tensor<32x8xi32>) -> tensor<32x8x768xf16>
    %214 = stablehlo.add %209, %213 : tensor<32x8x768xf16>
    %215 = stablehlo.add %214, %211 : tensor<32x8x768xf16>
    %216 = stablehlo.convert %215 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %217 = stablehlo.multiply %216, %216 : tensor<32x8x768xf32>
    %218 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %219 = stablehlo.reduce(%216 init: %218) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %220 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %221 = stablehlo.broadcast_in_dim %220, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %222 = stablehlo.divide %219, %221 : tensor<32x8xf32>
    %223 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %224 = stablehlo.reduce(%217 init: %223) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %225 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %226 = stablehlo.broadcast_in_dim %225, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %227 = stablehlo.divide %224, %226 : tensor<32x8xf32>
    %228 = stablehlo.multiply %222, %222 : tensor<32x8xf32>
    %229 = stablehlo.subtract %227, %228 : tensor<32x8xf32>
    %230 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %231 = stablehlo.broadcast_in_dim %230, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %232 = stablehlo.maximum %231, %229 : tensor<32x8xf32>
    %233 = stablehlo.broadcast_in_dim %222, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %234 = stablehlo.broadcast_in_dim %232, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %235 = stablehlo.convert %215 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %236 = stablehlo.broadcast_in_dim %233, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %237 = stablehlo.subtract %235, %236 : tensor<32x8x768xf32>
    %238 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %240 = stablehlo.add %234, %239 : tensor<32x8x1xf32>
    %241 = stablehlo.rsqrt %240 : tensor<32x8x1xf32>
    %242 = stablehlo.reshape %3 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %243 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %244 = stablehlo.broadcast_in_dim %242, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %245 = stablehlo.multiply %243, %244 : tensor<32x8x768xf32>
    %246 = stablehlo.multiply %237, %245 : tensor<32x8x768xf32>
    %247 = stablehlo.reshape %4 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %248 = stablehlo.broadcast_in_dim %247, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %249 = stablehlo.add %246, %248 : tensor<32x8x768xf32>
    %250 = stablehlo.convert %249 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %251 = stablehlo.slice %207 [0:1, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
    %252 = stablehlo.reshape %251 : (tensor<1x12xi32>) -> tensor<12xi32>
    %253 = stablehlo.convert %5 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %254 = stablehlo.convert %6 : (tensor<768xf32>) -> tensor<768xf16>
    %255 = stablehlo.convert %250 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %256 = stablehlo.convert %253 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %257 = stablehlo.dot_general %255, %256, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %258 = stablehlo.reshape %254 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %259 = stablehlo.broadcast_in_dim %258, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %260 = stablehlo.add %257, %259 : tensor<32x8x768xf16>
    %261 = stablehlo.convert %7 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %262 = stablehlo.convert %8 : (tensor<768xf32>) -> tensor<768xf16>
    %263 = stablehlo.convert %250 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %264 = stablehlo.convert %261 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %265 = stablehlo.dot_general %263, %264, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %266 = stablehlo.reshape %262 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %267 = stablehlo.broadcast_in_dim %266, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %268 = stablehlo.add %265, %267 : tensor<32x8x768xf16>
    %269 = stablehlo.convert %9 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %270 = stablehlo.convert %10 : (tensor<768xf32>) -> tensor<768xf16>
    %271 = stablehlo.convert %250 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %272 = stablehlo.convert %269 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %273 = stablehlo.dot_general %271, %272, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %274 = stablehlo.reshape %270 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %276 = stablehlo.add %273, %275 : tensor<32x8x768xf16>
    %277 = stablehlo.reshape %260 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %278 = stablehlo.reshape %268 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %279 = stablehlo.reshape %276 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %280 = stablehlo.broadcast_in_dim %204, dims = [0, 3] : (tensor<32x8xi32>) -> tensor<32x1x1x8xi32>
    %281 = stablehlo.constant dense<0> : tensor<i32>
    %282 = stablehlo.broadcast_in_dim %281, dims = [] : (tensor<i32>) -> tensor<32x1x1x8xi32>
    %283 = stablehlo.compare  GT, %280, %282,  SIGNED : (tensor<32x1x1x8xi32>, tensor<32x1x1x8xi32>) -> tensor<32x1x1x8xi1>
    %284 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %285 = stablehlo.broadcast_in_dim %284, dims = [] : (tensor<f32>) -> tensor<32x1x1x8xf32>
    %286 = stablehlo.convert %285 : (tensor<32x1x1x8xf32>) -> tensor<32x1x1x8xf16>
    %287 = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
    %288 = stablehlo.broadcast_in_dim %287, dims = [] : (tensor<f16>) -> tensor<32x1x1x8xf16>
    %289 = stablehlo.select %283, %286, %288 : tensor<32x1x1x8xi1>, tensor<32x1x1x8xf16>
    %290 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %291 = stablehlo.sqrt %290 : tensor<f32>
    %292 = stablehlo.convert %291 : (tensor<f32>) -> tensor<f16>
    %293 = stablehlo.broadcast_in_dim %292, dims = [] : (tensor<f16>) -> tensor<32x8x12x64xf16>
    %294 = stablehlo.divide %277, %293 : tensor<32x8x12x64xf16>
    %295 = stablehlo.convert %294 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %296 = stablehlo.convert %278 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %297 = stablehlo.dot_general %295, %296, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<32x8x12x64xf32>, tensor<32x8x12x64xf32>) -> tensor<32x12x8x8xf16>
    %298 = stablehlo.broadcast_in_dim %289, dims = [0, 1, 2, 3] : (tensor<32x1x1x8xf16>) -> tensor<32x12x8x8xf16>
    %299 = stablehlo.add %297, %298 : tensor<32x12x8x8xf16>
    %300 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %301 = stablehlo.reduce(%299 init: %300) applies stablehlo.maximum across dimensions = [3] : (tensor<32x12x8x8xf16>, tensor<f16>) -> tensor<32x12x8xf16>
    %302 = stablehlo.broadcast_in_dim %301, dims = [0, 1, 2] : (tensor<32x12x8xf16>) -> tensor<32x12x8x1xf16>
    %303 = stablehlo.broadcast_in_dim %302, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %304 = stablehlo.subtract %299, %303 : tensor<32x12x8x8xf16>
    %305 = stablehlo.exponential %304 : tensor<32x12x8x8xf16>
    %306 = stablehlo.convert %305 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %307 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %308 = stablehlo.reduce(%306 init: %307) applies stablehlo.add across dimensions = [3] : (tensor<32x12x8x8xf32>, tensor<f32>) -> tensor<32x12x8xf32>
    %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1, 2] : (tensor<32x12x8xf32>) -> tensor<32x12x8x1xf32>
    %310 = stablehlo.convert %309 : (tensor<32x12x8x1xf32>) -> tensor<32x12x8x1xf16>
    %311 = stablehlo.broadcast_in_dim %310, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %312 = stablehlo.divide %305, %311 : tensor<32x12x8x8xf16>
    %313 = stablehlo.convert %252 : (tensor<12xi32>) -> tensor<12xf16>
    %314 = stablehlo.convert %312 : tensor<32x12x8x8xf16>
    %315 = stablehlo.convert %313 : (tensor<12xf16>) -> tensor<12xf32>
    %316 = stablehlo.convert %314 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %317 = stablehlo.dot_general %315, %316, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<12xf32>, tensor<32x12x8x8xf32>) -> tensor<12x32x8x8xf16>
    %318 = stablehlo.transpose %317, dims = [1, 0, 2, 3] : (tensor<12x32x8x8xf16>) -> tensor<32x12x8x8xf16>
    %319 = stablehlo.convert %279 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %320 = stablehlo.convert %318 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %321 = stablehlo.dot_general %319, %320, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<32x8x12x64xf32>, tensor<32x12x8x8xf32>) -> tensor<32x12x64x8xf16>
    %322 = stablehlo.transpose %321, dims = [0, 3, 1, 2] : (tensor<32x12x64x8xf16>) -> tensor<32x8x12x64xf16>
    %323 = stablehlo.reshape %322 : (tensor<32x8x12x64xf16>) -> tensor<32x8x768xf16>
    %324 = stablehlo.convert %11 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %325 = stablehlo.convert %12 : (tensor<768xf32>) -> tensor<768xf16>
    %326 = stablehlo.convert %323 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %327 = stablehlo.convert %324 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %328 = stablehlo.dot_general %326, %327, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %329 = stablehlo.reshape %325 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %330 = stablehlo.broadcast_in_dim %329, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %331 = stablehlo.add %328, %330 : tensor<32x8x768xf16>
    %332 = stablehlo.add %331, %250 : tensor<32x8x768xf16>
    %333 = stablehlo.convert %332 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %334 = stablehlo.multiply %333, %333 : tensor<32x8x768xf32>
    %335 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %336 = stablehlo.reduce(%333 init: %335) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %337 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %338 = stablehlo.broadcast_in_dim %337, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %339 = stablehlo.divide %336, %338 : tensor<32x8xf32>
    %340 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %341 = stablehlo.reduce(%334 init: %340) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %342 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %343 = stablehlo.broadcast_in_dim %342, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %344 = stablehlo.divide %341, %343 : tensor<32x8xf32>
    %345 = stablehlo.multiply %339, %339 : tensor<32x8xf32>
    %346 = stablehlo.subtract %344, %345 : tensor<32x8xf32>
    %347 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %348 = stablehlo.broadcast_in_dim %347, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %349 = stablehlo.maximum %348, %346 : tensor<32x8xf32>
    %350 = stablehlo.broadcast_in_dim %339, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %351 = stablehlo.broadcast_in_dim %349, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %352 = stablehlo.convert %332 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %353 = stablehlo.broadcast_in_dim %350, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %354 = stablehlo.subtract %352, %353 : tensor<32x8x768xf32>
    %355 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %356 = stablehlo.broadcast_in_dim %355, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %357 = stablehlo.add %351, %356 : tensor<32x8x1xf32>
    %358 = stablehlo.rsqrt %357 : tensor<32x8x1xf32>
    %359 = stablehlo.reshape %13 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %360 = stablehlo.broadcast_in_dim %358, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %361 = stablehlo.broadcast_in_dim %359, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %362 = stablehlo.multiply %360, %361 : tensor<32x8x768xf32>
    %363 = stablehlo.multiply %354, %362 : tensor<32x8x768xf32>
    %364 = stablehlo.reshape %14 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %366 = stablehlo.add %363, %365 : tensor<32x8x768xf32>
    %367 = stablehlo.convert %366 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %368 = stablehlo.convert %15 : (tensor<768x3072xf32>) -> tensor<768x3072xf16>
    %369 = stablehlo.convert %16 : (tensor<3072xf32>) -> tensor<3072xf16>
    %370 = stablehlo.convert %367 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %371 = stablehlo.convert %368 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %372 = stablehlo.dot_general %370, %371, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x3072xf32>) -> tensor<32x8x3072xf16>
    %373 = stablehlo.reshape %369 : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %374 = stablehlo.broadcast_in_dim %373, dims = [0, 1, 2] : (tensor<1x1x3072xf16>) -> tensor<32x8x3072xf16>
    %375 = stablehlo.add %372, %374 : tensor<32x8x3072xf16>
    %376 = stablehlo.constant dense<1.414060e+00> : tensor<f16>
    %377 = stablehlo.broadcast_in_dim %376, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %378 = stablehlo.divide %375, %377 : tensor<32x8x3072xf16>
    %379 = stablehlo.custom_call @mhlo.erf(%378) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf16>
    %380 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %381 = stablehlo.broadcast_in_dim %380, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %382 = stablehlo.add %379, %381 : tensor<32x8x3072xf16>
    %383 = stablehlo.multiply %375, %382 : tensor<32x8x3072xf16>
    %384 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %385 = stablehlo.broadcast_in_dim %384, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %386 = stablehlo.divide %383, %385 : tensor<32x8x3072xf16>
    %387 = stablehlo.convert %17 : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
    %388 = stablehlo.convert %18 : (tensor<768xf32>) -> tensor<768xf16>
    %389 = stablehlo.convert %386 : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf32>
    %390 = stablehlo.convert %387 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %391 = stablehlo.dot_general %389, %390, contracting_dims = [2] x [0] : (tensor<32x8x3072xf32>, tensor<3072x768xf32>) -> tensor<32x8x768xf16>
    %392 = stablehlo.reshape %388 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %393 = stablehlo.broadcast_in_dim %392, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %394 = stablehlo.add %391, %393 : tensor<32x8x768xf16>
    %395 = stablehlo.add %394, %367 : tensor<32x8x768xf16>
    %396 = stablehlo.convert %395 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %397 = stablehlo.multiply %396, %396 : tensor<32x8x768xf32>
    %398 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %399 = stablehlo.reduce(%396 init: %398) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %400 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %401 = stablehlo.broadcast_in_dim %400, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %402 = stablehlo.divide %399, %401 : tensor<32x8xf32>
    %403 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %404 = stablehlo.reduce(%397 init: %403) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %405 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %406 = stablehlo.broadcast_in_dim %405, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %407 = stablehlo.divide %404, %406 : tensor<32x8xf32>
    %408 = stablehlo.multiply %402, %402 : tensor<32x8xf32>
    %409 = stablehlo.subtract %407, %408 : tensor<32x8xf32>
    %410 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %411 = stablehlo.broadcast_in_dim %410, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %412 = stablehlo.maximum %411, %409 : tensor<32x8xf32>
    %413 = stablehlo.broadcast_in_dim %402, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %414 = stablehlo.broadcast_in_dim %412, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %415 = stablehlo.convert %395 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %416 = stablehlo.broadcast_in_dim %413, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %417 = stablehlo.subtract %415, %416 : tensor<32x8x768xf32>
    %418 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %419 = stablehlo.broadcast_in_dim %418, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %420 = stablehlo.add %414, %419 : tensor<32x8x1xf32>
    %421 = stablehlo.rsqrt %420 : tensor<32x8x1xf32>
    %422 = stablehlo.reshape %19 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %423 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %424 = stablehlo.broadcast_in_dim %422, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %425 = stablehlo.multiply %423, %424 : tensor<32x8x768xf32>
    %426 = stablehlo.multiply %417, %425 : tensor<32x8x768xf32>
    %427 = stablehlo.reshape %20 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %428 = stablehlo.broadcast_in_dim %427, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %429 = stablehlo.add %426, %428 : tensor<32x8x768xf32>
    %430 = stablehlo.convert %429 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %431 = stablehlo.slice %207 [1:2, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
    %432 = stablehlo.reshape %431 : (tensor<1x12xi32>) -> tensor<12xi32>
    %433 = stablehlo.convert %21 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %434 = stablehlo.convert %22 : (tensor<768xf32>) -> tensor<768xf16>
    %435 = stablehlo.convert %430 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %436 = stablehlo.convert %433 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %437 = stablehlo.dot_general %435, %436, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %438 = stablehlo.reshape %434 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %439 = stablehlo.broadcast_in_dim %438, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %440 = stablehlo.add %437, %439 : tensor<32x8x768xf16>
    %441 = stablehlo.convert %23 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %442 = stablehlo.convert %24 : (tensor<768xf32>) -> tensor<768xf16>
    %443 = stablehlo.convert %430 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %444 = stablehlo.convert %441 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %445 = stablehlo.dot_general %443, %444, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %446 = stablehlo.reshape %442 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %447 = stablehlo.broadcast_in_dim %446, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %448 = stablehlo.add %445, %447 : tensor<32x8x768xf16>
    %449 = stablehlo.convert %25 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %450 = stablehlo.convert %26 : (tensor<768xf32>) -> tensor<768xf16>
    %451 = stablehlo.convert %430 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %452 = stablehlo.convert %449 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %453 = stablehlo.dot_general %451, %452, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %454 = stablehlo.reshape %450 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %455 = stablehlo.broadcast_in_dim %454, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %456 = stablehlo.add %453, %455 : tensor<32x8x768xf16>
    %457 = stablehlo.reshape %440 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %458 = stablehlo.reshape %448 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %459 = stablehlo.reshape %456 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %460 = stablehlo.broadcast_in_dim %204, dims = [0, 3] : (tensor<32x8xi32>) -> tensor<32x1x1x8xi32>
    %461 = stablehlo.constant dense<0> : tensor<i32>
    %462 = stablehlo.broadcast_in_dim %461, dims = [] : (tensor<i32>) -> tensor<32x1x1x8xi32>
    %463 = stablehlo.compare  GT, %460, %462,  SIGNED : (tensor<32x1x1x8xi32>, tensor<32x1x1x8xi32>) -> tensor<32x1x1x8xi1>
    %464 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %465 = stablehlo.broadcast_in_dim %464, dims = [] : (tensor<f32>) -> tensor<32x1x1x8xf32>
    %466 = stablehlo.convert %465 : (tensor<32x1x1x8xf32>) -> tensor<32x1x1x8xf16>
    %467 = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
    %468 = stablehlo.broadcast_in_dim %467, dims = [] : (tensor<f16>) -> tensor<32x1x1x8xf16>
    %469 = stablehlo.select %463, %466, %468 : tensor<32x1x1x8xi1>, tensor<32x1x1x8xf16>
    %470 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %471 = stablehlo.sqrt %470 : tensor<f32>
    %472 = stablehlo.convert %471 : (tensor<f32>) -> tensor<f16>
    %473 = stablehlo.broadcast_in_dim %472, dims = [] : (tensor<f16>) -> tensor<32x8x12x64xf16>
    %474 = stablehlo.divide %457, %473 : tensor<32x8x12x64xf16>
    %475 = stablehlo.convert %474 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %476 = stablehlo.convert %458 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %477 = stablehlo.dot_general %475, %476, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<32x8x12x64xf32>, tensor<32x8x12x64xf32>) -> tensor<32x12x8x8xf16>
    %478 = stablehlo.broadcast_in_dim %469, dims = [0, 1, 2, 3] : (tensor<32x1x1x8xf16>) -> tensor<32x12x8x8xf16>
    %479 = stablehlo.add %477, %478 : tensor<32x12x8x8xf16>
    %480 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %481 = stablehlo.reduce(%479 init: %480) applies stablehlo.maximum across dimensions = [3] : (tensor<32x12x8x8xf16>, tensor<f16>) -> tensor<32x12x8xf16>
    %482 = stablehlo.broadcast_in_dim %481, dims = [0, 1, 2] : (tensor<32x12x8xf16>) -> tensor<32x12x8x1xf16>
    %483 = stablehlo.broadcast_in_dim %482, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %484 = stablehlo.subtract %479, %483 : tensor<32x12x8x8xf16>
    %485 = stablehlo.exponential %484 : tensor<32x12x8x8xf16>
    %486 = stablehlo.convert %485 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %487 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %488 = stablehlo.reduce(%486 init: %487) applies stablehlo.add across dimensions = [3] : (tensor<32x12x8x8xf32>, tensor<f32>) -> tensor<32x12x8xf32>
    %489 = stablehlo.broadcast_in_dim %488, dims = [0, 1, 2] : (tensor<32x12x8xf32>) -> tensor<32x12x8x1xf32>
    %490 = stablehlo.convert %489 : (tensor<32x12x8x1xf32>) -> tensor<32x12x8x1xf16>
    %491 = stablehlo.broadcast_in_dim %490, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %492 = stablehlo.divide %485, %491 : tensor<32x12x8x8xf16>
    %493 = stablehlo.convert %432 : (tensor<12xi32>) -> tensor<12xf16>
    %494 = stablehlo.convert %492 : tensor<32x12x8x8xf16>
    %495 = stablehlo.convert %493 : (tensor<12xf16>) -> tensor<12xf32>
    %496 = stablehlo.convert %494 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %497 = stablehlo.dot_general %495, %496, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<12xf32>, tensor<32x12x8x8xf32>) -> tensor<12x32x8x8xf16>
    %498 = stablehlo.transpose %497, dims = [1, 0, 2, 3] : (tensor<12x32x8x8xf16>) -> tensor<32x12x8x8xf16>
    %499 = stablehlo.convert %459 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %500 = stablehlo.convert %498 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %501 = stablehlo.dot_general %499, %500, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<32x8x12x64xf32>, tensor<32x12x8x8xf32>) -> tensor<32x12x64x8xf16>
    %502 = stablehlo.transpose %501, dims = [0, 3, 1, 2] : (tensor<32x12x64x8xf16>) -> tensor<32x8x12x64xf16>
    %503 = stablehlo.reshape %502 : (tensor<32x8x12x64xf16>) -> tensor<32x8x768xf16>
    %504 = stablehlo.convert %27 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %505 = stablehlo.convert %28 : (tensor<768xf32>) -> tensor<768xf16>
    %506 = stablehlo.convert %503 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %507 = stablehlo.convert %504 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %508 = stablehlo.dot_general %506, %507, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %509 = stablehlo.reshape %505 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %510 = stablehlo.broadcast_in_dim %509, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %511 = stablehlo.add %508, %510 : tensor<32x8x768xf16>
    %512 = stablehlo.add %511, %430 : tensor<32x8x768xf16>
    %513 = stablehlo.convert %512 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %514 = stablehlo.multiply %513, %513 : tensor<32x8x768xf32>
    %515 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %516 = stablehlo.reduce(%513 init: %515) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %517 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %518 = stablehlo.broadcast_in_dim %517, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %519 = stablehlo.divide %516, %518 : tensor<32x8xf32>
    %520 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %521 = stablehlo.reduce(%514 init: %520) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %522 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %523 = stablehlo.broadcast_in_dim %522, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %524 = stablehlo.divide %521, %523 : tensor<32x8xf32>
    %525 = stablehlo.multiply %519, %519 : tensor<32x8xf32>
    %526 = stablehlo.subtract %524, %525 : tensor<32x8xf32>
    %527 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %528 = stablehlo.broadcast_in_dim %527, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %529 = stablehlo.maximum %528, %526 : tensor<32x8xf32>
    %530 = stablehlo.broadcast_in_dim %519, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %531 = stablehlo.broadcast_in_dim %529, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %532 = stablehlo.convert %512 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %533 = stablehlo.broadcast_in_dim %530, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %534 = stablehlo.subtract %532, %533 : tensor<32x8x768xf32>
    %535 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %536 = stablehlo.broadcast_in_dim %535, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %537 = stablehlo.add %531, %536 : tensor<32x8x1xf32>
    %538 = stablehlo.rsqrt %537 : tensor<32x8x1xf32>
    %539 = stablehlo.reshape %29 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %540 = stablehlo.broadcast_in_dim %538, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %541 = stablehlo.broadcast_in_dim %539, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %542 = stablehlo.multiply %540, %541 : tensor<32x8x768xf32>
    %543 = stablehlo.multiply %534, %542 : tensor<32x8x768xf32>
    %544 = stablehlo.reshape %30 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %545 = stablehlo.broadcast_in_dim %544, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %546 = stablehlo.add %543, %545 : tensor<32x8x768xf32>
    %547 = stablehlo.convert %546 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %548 = stablehlo.convert %31 : (tensor<768x3072xf32>) -> tensor<768x3072xf16>
    %549 = stablehlo.convert %32 : (tensor<3072xf32>) -> tensor<3072xf16>
    %550 = stablehlo.convert %547 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %551 = stablehlo.convert %548 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %552 = stablehlo.dot_general %550, %551, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x3072xf32>) -> tensor<32x8x3072xf16>
    %553 = stablehlo.reshape %549 : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %554 = stablehlo.broadcast_in_dim %553, dims = [0, 1, 2] : (tensor<1x1x3072xf16>) -> tensor<32x8x3072xf16>
    %555 = stablehlo.add %552, %554 : tensor<32x8x3072xf16>
    %556 = stablehlo.constant dense<1.414060e+00> : tensor<f16>
    %557 = stablehlo.broadcast_in_dim %556, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %558 = stablehlo.divide %555, %557 : tensor<32x8x3072xf16>
    %559 = stablehlo.custom_call @mhlo.erf(%558) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf16>
    %560 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %561 = stablehlo.broadcast_in_dim %560, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %562 = stablehlo.add %559, %561 : tensor<32x8x3072xf16>
    %563 = stablehlo.multiply %555, %562 : tensor<32x8x3072xf16>
    %564 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %565 = stablehlo.broadcast_in_dim %564, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %566 = stablehlo.divide %563, %565 : tensor<32x8x3072xf16>
    %567 = stablehlo.convert %33 : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
    %568 = stablehlo.convert %34 : (tensor<768xf32>) -> tensor<768xf16>
    %569 = stablehlo.convert %566 : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf32>
    %570 = stablehlo.convert %567 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %571 = stablehlo.dot_general %569, %570, contracting_dims = [2] x [0] : (tensor<32x8x3072xf32>, tensor<3072x768xf32>) -> tensor<32x8x768xf16>
    %572 = stablehlo.reshape %568 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %573 = stablehlo.broadcast_in_dim %572, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %574 = stablehlo.add %571, %573 : tensor<32x8x768xf16>
    %575 = stablehlo.add %574, %547 : tensor<32x8x768xf16>
    %576 = stablehlo.convert %575 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %577 = stablehlo.multiply %576, %576 : tensor<32x8x768xf32>
    %578 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %579 = stablehlo.reduce(%576 init: %578) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %580 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %581 = stablehlo.broadcast_in_dim %580, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %582 = stablehlo.divide %579, %581 : tensor<32x8xf32>
    %583 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %584 = stablehlo.reduce(%577 init: %583) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %585 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %586 = stablehlo.broadcast_in_dim %585, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %587 = stablehlo.divide %584, %586 : tensor<32x8xf32>
    %588 = stablehlo.multiply %582, %582 : tensor<32x8xf32>
    %589 = stablehlo.subtract %587, %588 : tensor<32x8xf32>
    %590 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %591 = stablehlo.broadcast_in_dim %590, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %592 = stablehlo.maximum %591, %589 : tensor<32x8xf32>
    %593 = stablehlo.broadcast_in_dim %582, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %594 = stablehlo.broadcast_in_dim %592, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %595 = stablehlo.convert %575 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %596 = stablehlo.broadcast_in_dim %593, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %597 = stablehlo.subtract %595, %596 : tensor<32x8x768xf32>
    %598 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %599 = stablehlo.broadcast_in_dim %598, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %600 = stablehlo.add %594, %599 : tensor<32x8x1xf32>
    %601 = stablehlo.rsqrt %600 : tensor<32x8x1xf32>
    %602 = stablehlo.reshape %35 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %603 = stablehlo.broadcast_in_dim %601, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %604 = stablehlo.broadcast_in_dim %602, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %605 = stablehlo.multiply %603, %604 : tensor<32x8x768xf32>
    %606 = stablehlo.multiply %597, %605 : tensor<32x8x768xf32>
    %607 = stablehlo.reshape %36 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %608 = stablehlo.broadcast_in_dim %607, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %609 = stablehlo.add %606, %608 : tensor<32x8x768xf32>
    %610 = stablehlo.convert %609 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %611 = stablehlo.slice %207 [2:3, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
    %612 = stablehlo.reshape %611 : (tensor<1x12xi32>) -> tensor<12xi32>
    %613 = stablehlo.convert %37 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %614 = stablehlo.convert %38 : (tensor<768xf32>) -> tensor<768xf16>
    %615 = stablehlo.convert %610 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %616 = stablehlo.convert %613 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %617 = stablehlo.dot_general %615, %616, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %618 = stablehlo.reshape %614 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %619 = stablehlo.broadcast_in_dim %618, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %620 = stablehlo.add %617, %619 : tensor<32x8x768xf16>
    %621 = stablehlo.convert %39 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %622 = stablehlo.convert %40 : (tensor<768xf32>) -> tensor<768xf16>
    %623 = stablehlo.convert %610 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %624 = stablehlo.convert %621 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %625 = stablehlo.dot_general %623, %624, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %626 = stablehlo.reshape %622 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %627 = stablehlo.broadcast_in_dim %626, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %628 = stablehlo.add %625, %627 : tensor<32x8x768xf16>
    %629 = stablehlo.convert %41 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %630 = stablehlo.convert %42 : (tensor<768xf32>) -> tensor<768xf16>
    %631 = stablehlo.convert %610 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %632 = stablehlo.convert %629 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %633 = stablehlo.dot_general %631, %632, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %634 = stablehlo.reshape %630 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %635 = stablehlo.broadcast_in_dim %634, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %636 = stablehlo.add %633, %635 : tensor<32x8x768xf16>
    %637 = stablehlo.reshape %620 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %638 = stablehlo.reshape %628 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %639 = stablehlo.reshape %636 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %640 = stablehlo.broadcast_in_dim %204, dims = [0, 3] : (tensor<32x8xi32>) -> tensor<32x1x1x8xi32>
    %641 = stablehlo.constant dense<0> : tensor<i32>
    %642 = stablehlo.broadcast_in_dim %641, dims = [] : (tensor<i32>) -> tensor<32x1x1x8xi32>
    %643 = stablehlo.compare  GT, %640, %642,  SIGNED : (tensor<32x1x1x8xi32>, tensor<32x1x1x8xi32>) -> tensor<32x1x1x8xi1>
    %644 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %645 = stablehlo.broadcast_in_dim %644, dims = [] : (tensor<f32>) -> tensor<32x1x1x8xf32>
    %646 = stablehlo.convert %645 : (tensor<32x1x1x8xf32>) -> tensor<32x1x1x8xf16>
    %647 = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
    %648 = stablehlo.broadcast_in_dim %647, dims = [] : (tensor<f16>) -> tensor<32x1x1x8xf16>
    %649 = stablehlo.select %643, %646, %648 : tensor<32x1x1x8xi1>, tensor<32x1x1x8xf16>
    %650 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %651 = stablehlo.sqrt %650 : tensor<f32>
    %652 = stablehlo.convert %651 : (tensor<f32>) -> tensor<f16>
    %653 = stablehlo.broadcast_in_dim %652, dims = [] : (tensor<f16>) -> tensor<32x8x12x64xf16>
    %654 = stablehlo.divide %637, %653 : tensor<32x8x12x64xf16>
    %655 = stablehlo.convert %654 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %656 = stablehlo.convert %638 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %657 = stablehlo.dot_general %655, %656, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<32x8x12x64xf32>, tensor<32x8x12x64xf32>) -> tensor<32x12x8x8xf16>
    %658 = stablehlo.broadcast_in_dim %649, dims = [0, 1, 2, 3] : (tensor<32x1x1x8xf16>) -> tensor<32x12x8x8xf16>
    %659 = stablehlo.add %657, %658 : tensor<32x12x8x8xf16>
    %660 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %661 = stablehlo.reduce(%659 init: %660) applies stablehlo.maximum across dimensions = [3] : (tensor<32x12x8x8xf16>, tensor<f16>) -> tensor<32x12x8xf16>
    %662 = stablehlo.broadcast_in_dim %661, dims = [0, 1, 2] : (tensor<32x12x8xf16>) -> tensor<32x12x8x1xf16>
    %663 = stablehlo.broadcast_in_dim %662, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %664 = stablehlo.subtract %659, %663 : tensor<32x12x8x8xf16>
    %665 = stablehlo.exponential %664 : tensor<32x12x8x8xf16>
    %666 = stablehlo.convert %665 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %667 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %668 = stablehlo.reduce(%666 init: %667) applies stablehlo.add across dimensions = [3] : (tensor<32x12x8x8xf32>, tensor<f32>) -> tensor<32x12x8xf32>
    %669 = stablehlo.broadcast_in_dim %668, dims = [0, 1, 2] : (tensor<32x12x8xf32>) -> tensor<32x12x8x1xf32>
    %670 = stablehlo.convert %669 : (tensor<32x12x8x1xf32>) -> tensor<32x12x8x1xf16>
    %671 = stablehlo.broadcast_in_dim %670, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %672 = stablehlo.divide %665, %671 : tensor<32x12x8x8xf16>
    %673 = stablehlo.convert %612 : (tensor<12xi32>) -> tensor<12xf16>
    %674 = stablehlo.convert %672 : tensor<32x12x8x8xf16>
    %675 = stablehlo.convert %673 : (tensor<12xf16>) -> tensor<12xf32>
    %676 = stablehlo.convert %674 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %677 = stablehlo.dot_general %675, %676, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<12xf32>, tensor<32x12x8x8xf32>) -> tensor<12x32x8x8xf16>
    %678 = stablehlo.transpose %677, dims = [1, 0, 2, 3] : (tensor<12x32x8x8xf16>) -> tensor<32x12x8x8xf16>
    %679 = stablehlo.convert %639 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %680 = stablehlo.convert %678 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %681 = stablehlo.dot_general %679, %680, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<32x8x12x64xf32>, tensor<32x12x8x8xf32>) -> tensor<32x12x64x8xf16>
    %682 = stablehlo.transpose %681, dims = [0, 3, 1, 2] : (tensor<32x12x64x8xf16>) -> tensor<32x8x12x64xf16>
    %683 = stablehlo.reshape %682 : (tensor<32x8x12x64xf16>) -> tensor<32x8x768xf16>
    %684 = stablehlo.convert %43 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %685 = stablehlo.convert %44 : (tensor<768xf32>) -> tensor<768xf16>
    %686 = stablehlo.convert %683 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %687 = stablehlo.convert %684 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %688 = stablehlo.dot_general %686, %687, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %689 = stablehlo.reshape %685 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %690 = stablehlo.broadcast_in_dim %689, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %691 = stablehlo.add %688, %690 : tensor<32x8x768xf16>
    %692 = stablehlo.add %691, %610 : tensor<32x8x768xf16>
    %693 = stablehlo.convert %692 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %694 = stablehlo.multiply %693, %693 : tensor<32x8x768xf32>
    %695 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %696 = stablehlo.reduce(%693 init: %695) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %697 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %698 = stablehlo.broadcast_in_dim %697, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %699 = stablehlo.divide %696, %698 : tensor<32x8xf32>
    %700 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %701 = stablehlo.reduce(%694 init: %700) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %702 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %703 = stablehlo.broadcast_in_dim %702, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %704 = stablehlo.divide %701, %703 : tensor<32x8xf32>
    %705 = stablehlo.multiply %699, %699 : tensor<32x8xf32>
    %706 = stablehlo.subtract %704, %705 : tensor<32x8xf32>
    %707 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %708 = stablehlo.broadcast_in_dim %707, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %709 = stablehlo.maximum %708, %706 : tensor<32x8xf32>
    %710 = stablehlo.broadcast_in_dim %699, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %711 = stablehlo.broadcast_in_dim %709, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %712 = stablehlo.convert %692 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %713 = stablehlo.broadcast_in_dim %710, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %714 = stablehlo.subtract %712, %713 : tensor<32x8x768xf32>
    %715 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %716 = stablehlo.broadcast_in_dim %715, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %717 = stablehlo.add %711, %716 : tensor<32x8x1xf32>
    %718 = stablehlo.rsqrt %717 : tensor<32x8x1xf32>
    %719 = stablehlo.reshape %45 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %720 = stablehlo.broadcast_in_dim %718, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %721 = stablehlo.broadcast_in_dim %719, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %722 = stablehlo.multiply %720, %721 : tensor<32x8x768xf32>
    %723 = stablehlo.multiply %714, %722 : tensor<32x8x768xf32>
    %724 = stablehlo.reshape %46 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %725 = stablehlo.broadcast_in_dim %724, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %726 = stablehlo.add %723, %725 : tensor<32x8x768xf32>
    %727 = stablehlo.convert %726 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %728 = stablehlo.convert %47 : (tensor<768x3072xf32>) -> tensor<768x3072xf16>
    %729 = stablehlo.convert %48 : (tensor<3072xf32>) -> tensor<3072xf16>
    %730 = stablehlo.convert %727 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %731 = stablehlo.convert %728 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %732 = stablehlo.dot_general %730, %731, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x3072xf32>) -> tensor<32x8x3072xf16>
    %733 = stablehlo.reshape %729 : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %734 = stablehlo.broadcast_in_dim %733, dims = [0, 1, 2] : (tensor<1x1x3072xf16>) -> tensor<32x8x3072xf16>
    %735 = stablehlo.add %732, %734 : tensor<32x8x3072xf16>
    %736 = stablehlo.constant dense<1.414060e+00> : tensor<f16>
    %737 = stablehlo.broadcast_in_dim %736, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %738 = stablehlo.divide %735, %737 : tensor<32x8x3072xf16>
    %739 = stablehlo.custom_call @mhlo.erf(%738) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf16>
    %740 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %741 = stablehlo.broadcast_in_dim %740, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %742 = stablehlo.add %739, %741 : tensor<32x8x3072xf16>
    %743 = stablehlo.multiply %735, %742 : tensor<32x8x3072xf16>
    %744 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %745 = stablehlo.broadcast_in_dim %744, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %746 = stablehlo.divide %743, %745 : tensor<32x8x3072xf16>
    %747 = stablehlo.convert %49 : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
    %748 = stablehlo.convert %50 : (tensor<768xf32>) -> tensor<768xf16>
    %749 = stablehlo.convert %746 : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf32>
    %750 = stablehlo.convert %747 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %751 = stablehlo.dot_general %749, %750, contracting_dims = [2] x [0] : (tensor<32x8x3072xf32>, tensor<3072x768xf32>) -> tensor<32x8x768xf16>
    %752 = stablehlo.reshape %748 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %753 = stablehlo.broadcast_in_dim %752, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %754 = stablehlo.add %751, %753 : tensor<32x8x768xf16>
    %755 = stablehlo.add %754, %727 : tensor<32x8x768xf16>
    %756 = stablehlo.convert %755 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %757 = stablehlo.multiply %756, %756 : tensor<32x8x768xf32>
    %758 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %759 = stablehlo.reduce(%756 init: %758) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %760 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %761 = stablehlo.broadcast_in_dim %760, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %762 = stablehlo.divide %759, %761 : tensor<32x8xf32>
    %763 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %764 = stablehlo.reduce(%757 init: %763) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %765 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %766 = stablehlo.broadcast_in_dim %765, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %767 = stablehlo.divide %764, %766 : tensor<32x8xf32>
    %768 = stablehlo.multiply %762, %762 : tensor<32x8xf32>
    %769 = stablehlo.subtract %767, %768 : tensor<32x8xf32>
    %770 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %771 = stablehlo.broadcast_in_dim %770, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %772 = stablehlo.maximum %771, %769 : tensor<32x8xf32>
    %773 = stablehlo.broadcast_in_dim %762, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %774 = stablehlo.broadcast_in_dim %772, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %775 = stablehlo.convert %755 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %776 = stablehlo.broadcast_in_dim %773, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %777 = stablehlo.subtract %775, %776 : tensor<32x8x768xf32>
    %778 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %779 = stablehlo.broadcast_in_dim %778, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %780 = stablehlo.add %774, %779 : tensor<32x8x1xf32>
    %781 = stablehlo.rsqrt %780 : tensor<32x8x1xf32>
    %782 = stablehlo.reshape %51 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %783 = stablehlo.broadcast_in_dim %781, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %784 = stablehlo.broadcast_in_dim %782, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %785 = stablehlo.multiply %783, %784 : tensor<32x8x768xf32>
    %786 = stablehlo.multiply %777, %785 : tensor<32x8x768xf32>
    %787 = stablehlo.reshape %52 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %788 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %789 = stablehlo.add %786, %788 : tensor<32x8x768xf32>
    %790 = stablehlo.convert %789 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %791 = stablehlo.slice %207 [3:4, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
    %792 = stablehlo.reshape %791 : (tensor<1x12xi32>) -> tensor<12xi32>
    %793 = stablehlo.convert %53 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %794 = stablehlo.convert %54 : (tensor<768xf32>) -> tensor<768xf16>
    %795 = stablehlo.convert %790 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %796 = stablehlo.convert %793 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %797 = stablehlo.dot_general %795, %796, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %798 = stablehlo.reshape %794 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %799 = stablehlo.broadcast_in_dim %798, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %800 = stablehlo.add %797, %799 : tensor<32x8x768xf16>
    %801 = stablehlo.convert %55 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %802 = stablehlo.convert %56 : (tensor<768xf32>) -> tensor<768xf16>
    %803 = stablehlo.convert %790 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %804 = stablehlo.convert %801 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %805 = stablehlo.dot_general %803, %804, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %806 = stablehlo.reshape %802 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %807 = stablehlo.broadcast_in_dim %806, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %808 = stablehlo.add %805, %807 : tensor<32x8x768xf16>
    %809 = stablehlo.convert %57 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %810 = stablehlo.convert %58 : (tensor<768xf32>) -> tensor<768xf16>
    %811 = stablehlo.convert %790 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %812 = stablehlo.convert %809 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %813 = stablehlo.dot_general %811, %812, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %814 = stablehlo.reshape %810 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %815 = stablehlo.broadcast_in_dim %814, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %816 = stablehlo.add %813, %815 : tensor<32x8x768xf16>
    %817 = stablehlo.reshape %800 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %818 = stablehlo.reshape %808 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %819 = stablehlo.reshape %816 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %820 = stablehlo.broadcast_in_dim %204, dims = [0, 3] : (tensor<32x8xi32>) -> tensor<32x1x1x8xi32>
    %821 = stablehlo.constant dense<0> : tensor<i32>
    %822 = stablehlo.broadcast_in_dim %821, dims = [] : (tensor<i32>) -> tensor<32x1x1x8xi32>
    %823 = stablehlo.compare  GT, %820, %822,  SIGNED : (tensor<32x1x1x8xi32>, tensor<32x1x1x8xi32>) -> tensor<32x1x1x8xi1>
    %824 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %825 = stablehlo.broadcast_in_dim %824, dims = [] : (tensor<f32>) -> tensor<32x1x1x8xf32>
    %826 = stablehlo.convert %825 : (tensor<32x1x1x8xf32>) -> tensor<32x1x1x8xf16>
    %827 = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
    %828 = stablehlo.broadcast_in_dim %827, dims = [] : (tensor<f16>) -> tensor<32x1x1x8xf16>
    %829 = stablehlo.select %823, %826, %828 : tensor<32x1x1x8xi1>, tensor<32x1x1x8xf16>
    %830 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %831 = stablehlo.sqrt %830 : tensor<f32>
    %832 = stablehlo.convert %831 : (tensor<f32>) -> tensor<f16>
    %833 = stablehlo.broadcast_in_dim %832, dims = [] : (tensor<f16>) -> tensor<32x8x12x64xf16>
    %834 = stablehlo.divide %817, %833 : tensor<32x8x12x64xf16>
    %835 = stablehlo.convert %834 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %836 = stablehlo.convert %818 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %837 = stablehlo.dot_general %835, %836, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<32x8x12x64xf32>, tensor<32x8x12x64xf32>) -> tensor<32x12x8x8xf16>
    %838 = stablehlo.broadcast_in_dim %829, dims = [0, 1, 2, 3] : (tensor<32x1x1x8xf16>) -> tensor<32x12x8x8xf16>
    %839 = stablehlo.add %837, %838 : tensor<32x12x8x8xf16>
    %840 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %841 = stablehlo.reduce(%839 init: %840) applies stablehlo.maximum across dimensions = [3] : (tensor<32x12x8x8xf16>, tensor<f16>) -> tensor<32x12x8xf16>
    %842 = stablehlo.broadcast_in_dim %841, dims = [0, 1, 2] : (tensor<32x12x8xf16>) -> tensor<32x12x8x1xf16>
    %843 = stablehlo.broadcast_in_dim %842, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %844 = stablehlo.subtract %839, %843 : tensor<32x12x8x8xf16>
    %845 = stablehlo.exponential %844 : tensor<32x12x8x8xf16>
    %846 = stablehlo.convert %845 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %847 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %848 = stablehlo.reduce(%846 init: %847) applies stablehlo.add across dimensions = [3] : (tensor<32x12x8x8xf32>, tensor<f32>) -> tensor<32x12x8xf32>
    %849 = stablehlo.broadcast_in_dim %848, dims = [0, 1, 2] : (tensor<32x12x8xf32>) -> tensor<32x12x8x1xf32>
    %850 = stablehlo.convert %849 : (tensor<32x12x8x1xf32>) -> tensor<32x12x8x1xf16>
    %851 = stablehlo.broadcast_in_dim %850, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %852 = stablehlo.divide %845, %851 : tensor<32x12x8x8xf16>
    %853 = stablehlo.convert %792 : (tensor<12xi32>) -> tensor<12xf16>
    %854 = stablehlo.convert %852 : tensor<32x12x8x8xf16>
    %855 = stablehlo.convert %853 : (tensor<12xf16>) -> tensor<12xf32>
    %856 = stablehlo.convert %854 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %857 = stablehlo.dot_general %855, %856, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<12xf32>, tensor<32x12x8x8xf32>) -> tensor<12x32x8x8xf16>
    %858 = stablehlo.transpose %857, dims = [1, 0, 2, 3] : (tensor<12x32x8x8xf16>) -> tensor<32x12x8x8xf16>
    %859 = stablehlo.convert %819 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %860 = stablehlo.convert %858 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %861 = stablehlo.dot_general %859, %860, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<32x8x12x64xf32>, tensor<32x12x8x8xf32>) -> tensor<32x12x64x8xf16>
    %862 = stablehlo.transpose %861, dims = [0, 3, 1, 2] : (tensor<32x12x64x8xf16>) -> tensor<32x8x12x64xf16>
    %863 = stablehlo.reshape %862 : (tensor<32x8x12x64xf16>) -> tensor<32x8x768xf16>
    %864 = stablehlo.convert %59 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %865 = stablehlo.convert %60 : (tensor<768xf32>) -> tensor<768xf16>
    %866 = stablehlo.convert %863 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %867 = stablehlo.convert %864 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %868 = stablehlo.dot_general %866, %867, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %869 = stablehlo.reshape %865 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %870 = stablehlo.broadcast_in_dim %869, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %871 = stablehlo.add %868, %870 : tensor<32x8x768xf16>
    %872 = stablehlo.add %871, %790 : tensor<32x8x768xf16>
    %873 = stablehlo.convert %872 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %874 = stablehlo.multiply %873, %873 : tensor<32x8x768xf32>
    %875 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %876 = stablehlo.reduce(%873 init: %875) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %877 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %878 = stablehlo.broadcast_in_dim %877, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %879 = stablehlo.divide %876, %878 : tensor<32x8xf32>
    %880 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %881 = stablehlo.reduce(%874 init: %880) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %882 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %883 = stablehlo.broadcast_in_dim %882, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %884 = stablehlo.divide %881, %883 : tensor<32x8xf32>
    %885 = stablehlo.multiply %879, %879 : tensor<32x8xf32>
    %886 = stablehlo.subtract %884, %885 : tensor<32x8xf32>
    %887 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %888 = stablehlo.broadcast_in_dim %887, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %889 = stablehlo.maximum %888, %886 : tensor<32x8xf32>
    %890 = stablehlo.broadcast_in_dim %879, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %891 = stablehlo.broadcast_in_dim %889, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %892 = stablehlo.convert %872 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %893 = stablehlo.broadcast_in_dim %890, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %894 = stablehlo.subtract %892, %893 : tensor<32x8x768xf32>
    %895 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %896 = stablehlo.broadcast_in_dim %895, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %897 = stablehlo.add %891, %896 : tensor<32x8x1xf32>
    %898 = stablehlo.rsqrt %897 : tensor<32x8x1xf32>
    %899 = stablehlo.reshape %61 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %900 = stablehlo.broadcast_in_dim %898, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %901 = stablehlo.broadcast_in_dim %899, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %902 = stablehlo.multiply %900, %901 : tensor<32x8x768xf32>
    %903 = stablehlo.multiply %894, %902 : tensor<32x8x768xf32>
    %904 = stablehlo.reshape %62 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %905 = stablehlo.broadcast_in_dim %904, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %906 = stablehlo.add %903, %905 : tensor<32x8x768xf32>
    %907 = stablehlo.convert %906 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %908 = stablehlo.convert %63 : (tensor<768x3072xf32>) -> tensor<768x3072xf16>
    %909 = stablehlo.convert %64 : (tensor<3072xf32>) -> tensor<3072xf16>
    %910 = stablehlo.convert %907 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %911 = stablehlo.convert %908 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %912 = stablehlo.dot_general %910, %911, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x3072xf32>) -> tensor<32x8x3072xf16>
    %913 = stablehlo.reshape %909 : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %914 = stablehlo.broadcast_in_dim %913, dims = [0, 1, 2] : (tensor<1x1x3072xf16>) -> tensor<32x8x3072xf16>
    %915 = stablehlo.add %912, %914 : tensor<32x8x3072xf16>
    %916 = stablehlo.constant dense<1.414060e+00> : tensor<f16>
    %917 = stablehlo.broadcast_in_dim %916, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %918 = stablehlo.divide %915, %917 : tensor<32x8x3072xf16>
    %919 = stablehlo.custom_call @mhlo.erf(%918) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf16>
    %920 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %921 = stablehlo.broadcast_in_dim %920, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %922 = stablehlo.add %919, %921 : tensor<32x8x3072xf16>
    %923 = stablehlo.multiply %915, %922 : tensor<32x8x3072xf16>
    %924 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %925 = stablehlo.broadcast_in_dim %924, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %926 = stablehlo.divide %923, %925 : tensor<32x8x3072xf16>
    %927 = stablehlo.convert %65 : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
    %928 = stablehlo.convert %66 : (tensor<768xf32>) -> tensor<768xf16>
    %929 = stablehlo.convert %926 : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf32>
    %930 = stablehlo.convert %927 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %931 = stablehlo.dot_general %929, %930, contracting_dims = [2] x [0] : (tensor<32x8x3072xf32>, tensor<3072x768xf32>) -> tensor<32x8x768xf16>
    %932 = stablehlo.reshape %928 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %933 = stablehlo.broadcast_in_dim %932, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %934 = stablehlo.add %931, %933 : tensor<32x8x768xf16>
    %935 = stablehlo.add %934, %907 : tensor<32x8x768xf16>
    %936 = stablehlo.convert %935 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %937 = stablehlo.multiply %936, %936 : tensor<32x8x768xf32>
    %938 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %939 = stablehlo.reduce(%936 init: %938) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %940 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %941 = stablehlo.broadcast_in_dim %940, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %942 = stablehlo.divide %939, %941 : tensor<32x8xf32>
    %943 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %944 = stablehlo.reduce(%937 init: %943) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %945 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %946 = stablehlo.broadcast_in_dim %945, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %947 = stablehlo.divide %944, %946 : tensor<32x8xf32>
    %948 = stablehlo.multiply %942, %942 : tensor<32x8xf32>
    %949 = stablehlo.subtract %947, %948 : tensor<32x8xf32>
    %950 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %951 = stablehlo.broadcast_in_dim %950, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %952 = stablehlo.maximum %951, %949 : tensor<32x8xf32>
    %953 = stablehlo.broadcast_in_dim %942, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %954 = stablehlo.broadcast_in_dim %952, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %955 = stablehlo.convert %935 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %956 = stablehlo.broadcast_in_dim %953, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %957 = stablehlo.subtract %955, %956 : tensor<32x8x768xf32>
    %958 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %959 = stablehlo.broadcast_in_dim %958, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %960 = stablehlo.add %954, %959 : tensor<32x8x1xf32>
    %961 = stablehlo.rsqrt %960 : tensor<32x8x1xf32>
    %962 = stablehlo.reshape %67 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %963 = stablehlo.broadcast_in_dim %961, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %964 = stablehlo.broadcast_in_dim %962, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %965 = stablehlo.multiply %963, %964 : tensor<32x8x768xf32>
    %966 = stablehlo.multiply %957, %965 : tensor<32x8x768xf32>
    %967 = stablehlo.reshape %68 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %968 = stablehlo.broadcast_in_dim %967, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %969 = stablehlo.add %966, %968 : tensor<32x8x768xf32>
    %970 = stablehlo.convert %969 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %971 = stablehlo.slice %207 [4:5, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
    %972 = stablehlo.reshape %971 : (tensor<1x12xi32>) -> tensor<12xi32>
    %973 = stablehlo.convert %69 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %974 = stablehlo.convert %70 : (tensor<768xf32>) -> tensor<768xf16>
    %975 = stablehlo.convert %970 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %976 = stablehlo.convert %973 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %977 = stablehlo.dot_general %975, %976, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %978 = stablehlo.reshape %974 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %979 = stablehlo.broadcast_in_dim %978, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %980 = stablehlo.add %977, %979 : tensor<32x8x768xf16>
    %981 = stablehlo.convert %71 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %982 = stablehlo.convert %72 : (tensor<768xf32>) -> tensor<768xf16>
    %983 = stablehlo.convert %970 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %984 = stablehlo.convert %981 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %985 = stablehlo.dot_general %983, %984, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %986 = stablehlo.reshape %982 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %987 = stablehlo.broadcast_in_dim %986, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %988 = stablehlo.add %985, %987 : tensor<32x8x768xf16>
    %989 = stablehlo.convert %73 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %990 = stablehlo.convert %74 : (tensor<768xf32>) -> tensor<768xf16>
    %991 = stablehlo.convert %970 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %992 = stablehlo.convert %989 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %993 = stablehlo.dot_general %991, %992, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %994 = stablehlo.reshape %990 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %995 = stablehlo.broadcast_in_dim %994, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %996 = stablehlo.add %993, %995 : tensor<32x8x768xf16>
    %997 = stablehlo.reshape %980 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %998 = stablehlo.reshape %988 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %999 = stablehlo.reshape %996 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1000 = stablehlo.broadcast_in_dim %204, dims = [0, 3] : (tensor<32x8xi32>) -> tensor<32x1x1x8xi32>
    %1001 = stablehlo.constant dense<0> : tensor<i32>
    %1002 = stablehlo.broadcast_in_dim %1001, dims = [] : (tensor<i32>) -> tensor<32x1x1x8xi32>
    %1003 = stablehlo.compare  GT, %1000, %1002,  SIGNED : (tensor<32x1x1x8xi32>, tensor<32x1x1x8xi32>) -> tensor<32x1x1x8xi1>
    %1004 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1005 = stablehlo.broadcast_in_dim %1004, dims = [] : (tensor<f32>) -> tensor<32x1x1x8xf32>
    %1006 = stablehlo.convert %1005 : (tensor<32x1x1x8xf32>) -> tensor<32x1x1x8xf16>
    %1007 = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
    %1008 = stablehlo.broadcast_in_dim %1007, dims = [] : (tensor<f16>) -> tensor<32x1x1x8xf16>
    %1009 = stablehlo.select %1003, %1006, %1008 : tensor<32x1x1x8xi1>, tensor<32x1x1x8xf16>
    %1010 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %1011 = stablehlo.sqrt %1010 : tensor<f32>
    %1012 = stablehlo.convert %1011 : (tensor<f32>) -> tensor<f16>
    %1013 = stablehlo.broadcast_in_dim %1012, dims = [] : (tensor<f16>) -> tensor<32x8x12x64xf16>
    %1014 = stablehlo.divide %997, %1013 : tensor<32x8x12x64xf16>
    %1015 = stablehlo.convert %1014 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1016 = stablehlo.convert %998 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1017 = stablehlo.dot_general %1015, %1016, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<32x8x12x64xf32>, tensor<32x8x12x64xf32>) -> tensor<32x12x8x8xf16>
    %1018 = stablehlo.broadcast_in_dim %1009, dims = [0, 1, 2, 3] : (tensor<32x1x1x8xf16>) -> tensor<32x12x8x8xf16>
    %1019 = stablehlo.add %1017, %1018 : tensor<32x12x8x8xf16>
    %1020 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1021 = stablehlo.reduce(%1019 init: %1020) applies stablehlo.maximum across dimensions = [3] : (tensor<32x12x8x8xf16>, tensor<f16>) -> tensor<32x12x8xf16>
    %1022 = stablehlo.broadcast_in_dim %1021, dims = [0, 1, 2] : (tensor<32x12x8xf16>) -> tensor<32x12x8x1xf16>
    %1023 = stablehlo.broadcast_in_dim %1022, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %1024 = stablehlo.subtract %1019, %1023 : tensor<32x12x8x8xf16>
    %1025 = stablehlo.exponential %1024 : tensor<32x12x8x8xf16>
    %1026 = stablehlo.convert %1025 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1027 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1028 = stablehlo.reduce(%1026 init: %1027) applies stablehlo.add across dimensions = [3] : (tensor<32x12x8x8xf32>, tensor<f32>) -> tensor<32x12x8xf32>
    %1029 = stablehlo.broadcast_in_dim %1028, dims = [0, 1, 2] : (tensor<32x12x8xf32>) -> tensor<32x12x8x1xf32>
    %1030 = stablehlo.convert %1029 : (tensor<32x12x8x1xf32>) -> tensor<32x12x8x1xf16>
    %1031 = stablehlo.broadcast_in_dim %1030, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %1032 = stablehlo.divide %1025, %1031 : tensor<32x12x8x8xf16>
    %1033 = stablehlo.convert %972 : (tensor<12xi32>) -> tensor<12xf16>
    %1034 = stablehlo.convert %1032 : tensor<32x12x8x8xf16>
    %1035 = stablehlo.convert %1033 : (tensor<12xf16>) -> tensor<12xf32>
    %1036 = stablehlo.convert %1034 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1037 = stablehlo.dot_general %1035, %1036, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<12xf32>, tensor<32x12x8x8xf32>) -> tensor<12x32x8x8xf16>
    %1038 = stablehlo.transpose %1037, dims = [1, 0, 2, 3] : (tensor<12x32x8x8xf16>) -> tensor<32x12x8x8xf16>
    %1039 = stablehlo.convert %999 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1040 = stablehlo.convert %1038 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1041 = stablehlo.dot_general %1039, %1040, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<32x8x12x64xf32>, tensor<32x12x8x8xf32>) -> tensor<32x12x64x8xf16>
    %1042 = stablehlo.transpose %1041, dims = [0, 3, 1, 2] : (tensor<32x12x64x8xf16>) -> tensor<32x8x12x64xf16>
    %1043 = stablehlo.reshape %1042 : (tensor<32x8x12x64xf16>) -> tensor<32x8x768xf16>
    %1044 = stablehlo.convert %75 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1045 = stablehlo.convert %76 : (tensor<768xf32>) -> tensor<768xf16>
    %1046 = stablehlo.convert %1043 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1047 = stablehlo.convert %1044 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1048 = stablehlo.dot_general %1046, %1047, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1049 = stablehlo.reshape %1045 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1050 = stablehlo.broadcast_in_dim %1049, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1051 = stablehlo.add %1048, %1050 : tensor<32x8x768xf16>
    %1052 = stablehlo.add %1051, %970 : tensor<32x8x768xf16>
    %1053 = stablehlo.convert %1052 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1054 = stablehlo.multiply %1053, %1053 : tensor<32x8x768xf32>
    %1055 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1056 = stablehlo.reduce(%1053 init: %1055) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1057 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1058 = stablehlo.broadcast_in_dim %1057, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1059 = stablehlo.divide %1056, %1058 : tensor<32x8xf32>
    %1060 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1061 = stablehlo.reduce(%1054 init: %1060) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1062 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1063 = stablehlo.broadcast_in_dim %1062, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1064 = stablehlo.divide %1061, %1063 : tensor<32x8xf32>
    %1065 = stablehlo.multiply %1059, %1059 : tensor<32x8xf32>
    %1066 = stablehlo.subtract %1064, %1065 : tensor<32x8xf32>
    %1067 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1068 = stablehlo.broadcast_in_dim %1067, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1069 = stablehlo.maximum %1068, %1066 : tensor<32x8xf32>
    %1070 = stablehlo.broadcast_in_dim %1059, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1071 = stablehlo.broadcast_in_dim %1069, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1072 = stablehlo.convert %1052 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1073 = stablehlo.broadcast_in_dim %1070, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1074 = stablehlo.subtract %1072, %1073 : tensor<32x8x768xf32>
    %1075 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %1076 = stablehlo.broadcast_in_dim %1075, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %1077 = stablehlo.add %1071, %1076 : tensor<32x8x1xf32>
    %1078 = stablehlo.rsqrt %1077 : tensor<32x8x1xf32>
    %1079 = stablehlo.reshape %77 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1080 = stablehlo.broadcast_in_dim %1078, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1081 = stablehlo.broadcast_in_dim %1079, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1082 = stablehlo.multiply %1080, %1081 : tensor<32x8x768xf32>
    %1083 = stablehlo.multiply %1074, %1082 : tensor<32x8x768xf32>
    %1084 = stablehlo.reshape %78 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1085 = stablehlo.broadcast_in_dim %1084, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1086 = stablehlo.add %1083, %1085 : tensor<32x8x768xf32>
    %1087 = stablehlo.convert %1086 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %1088 = stablehlo.convert %79 : (tensor<768x3072xf32>) -> tensor<768x3072xf16>
    %1089 = stablehlo.convert %80 : (tensor<3072xf32>) -> tensor<3072xf16>
    %1090 = stablehlo.convert %1087 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1091 = stablehlo.convert %1088 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1092 = stablehlo.dot_general %1090, %1091, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x3072xf32>) -> tensor<32x8x3072xf16>
    %1093 = stablehlo.reshape %1089 : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %1094 = stablehlo.broadcast_in_dim %1093, dims = [0, 1, 2] : (tensor<1x1x3072xf16>) -> tensor<32x8x3072xf16>
    %1095 = stablehlo.add %1092, %1094 : tensor<32x8x3072xf16>
    %1096 = stablehlo.constant dense<1.414060e+00> : tensor<f16>
    %1097 = stablehlo.broadcast_in_dim %1096, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1098 = stablehlo.divide %1095, %1097 : tensor<32x8x3072xf16>
    %1099 = stablehlo.custom_call @mhlo.erf(%1098) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf16>
    %1100 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1101 = stablehlo.broadcast_in_dim %1100, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1102 = stablehlo.add %1099, %1101 : tensor<32x8x3072xf16>
    %1103 = stablehlo.multiply %1095, %1102 : tensor<32x8x3072xf16>
    %1104 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %1105 = stablehlo.broadcast_in_dim %1104, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1106 = stablehlo.divide %1103, %1105 : tensor<32x8x3072xf16>
    %1107 = stablehlo.convert %81 : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
    %1108 = stablehlo.convert %82 : (tensor<768xf32>) -> tensor<768xf16>
    %1109 = stablehlo.convert %1106 : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf32>
    %1110 = stablehlo.convert %1107 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1111 = stablehlo.dot_general %1109, %1110, contracting_dims = [2] x [0] : (tensor<32x8x3072xf32>, tensor<3072x768xf32>) -> tensor<32x8x768xf16>
    %1112 = stablehlo.reshape %1108 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1113 = stablehlo.broadcast_in_dim %1112, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1114 = stablehlo.add %1111, %1113 : tensor<32x8x768xf16>
    %1115 = stablehlo.add %1114, %1087 : tensor<32x8x768xf16>
    %1116 = stablehlo.convert %1115 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1117 = stablehlo.multiply %1116, %1116 : tensor<32x8x768xf32>
    %1118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1119 = stablehlo.reduce(%1116 init: %1118) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1120 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1121 = stablehlo.broadcast_in_dim %1120, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1122 = stablehlo.divide %1119, %1121 : tensor<32x8xf32>
    %1123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1124 = stablehlo.reduce(%1117 init: %1123) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1125 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1126 = stablehlo.broadcast_in_dim %1125, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1127 = stablehlo.divide %1124, %1126 : tensor<32x8xf32>
    %1128 = stablehlo.multiply %1122, %1122 : tensor<32x8xf32>
    %1129 = stablehlo.subtract %1127, %1128 : tensor<32x8xf32>
    %1130 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1131 = stablehlo.broadcast_in_dim %1130, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1132 = stablehlo.maximum %1131, %1129 : tensor<32x8xf32>
    %1133 = stablehlo.broadcast_in_dim %1122, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1134 = stablehlo.broadcast_in_dim %1132, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1135 = stablehlo.convert %1115 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1136 = stablehlo.broadcast_in_dim %1133, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1137 = stablehlo.subtract %1135, %1136 : tensor<32x8x768xf32>
    %1138 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %1139 = stablehlo.broadcast_in_dim %1138, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %1140 = stablehlo.add %1134, %1139 : tensor<32x8x1xf32>
    %1141 = stablehlo.rsqrt %1140 : tensor<32x8x1xf32>
    %1142 = stablehlo.reshape %83 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1143 = stablehlo.broadcast_in_dim %1141, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1144 = stablehlo.broadcast_in_dim %1142, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1145 = stablehlo.multiply %1143, %1144 : tensor<32x8x768xf32>
    %1146 = stablehlo.multiply %1137, %1145 : tensor<32x8x768xf32>
    %1147 = stablehlo.reshape %84 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1148 = stablehlo.broadcast_in_dim %1147, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1149 = stablehlo.add %1146, %1148 : tensor<32x8x768xf32>
    %1150 = stablehlo.convert %1149 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %1151 = stablehlo.slice %207 [5:6, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
    %1152 = stablehlo.reshape %1151 : (tensor<1x12xi32>) -> tensor<12xi32>
    %1153 = stablehlo.convert %85 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1154 = stablehlo.convert %86 : (tensor<768xf32>) -> tensor<768xf16>
    %1155 = stablehlo.convert %1150 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1156 = stablehlo.convert %1153 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1157 = stablehlo.dot_general %1155, %1156, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1158 = stablehlo.reshape %1154 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1159 = stablehlo.broadcast_in_dim %1158, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1160 = stablehlo.add %1157, %1159 : tensor<32x8x768xf16>
    %1161 = stablehlo.convert %87 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1162 = stablehlo.convert %88 : (tensor<768xf32>) -> tensor<768xf16>
    %1163 = stablehlo.convert %1150 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1164 = stablehlo.convert %1161 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1165 = stablehlo.dot_general %1163, %1164, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1166 = stablehlo.reshape %1162 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1167 = stablehlo.broadcast_in_dim %1166, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1168 = stablehlo.add %1165, %1167 : tensor<32x8x768xf16>
    %1169 = stablehlo.convert %89 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1170 = stablehlo.convert %90 : (tensor<768xf32>) -> tensor<768xf16>
    %1171 = stablehlo.convert %1150 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1172 = stablehlo.convert %1169 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1173 = stablehlo.dot_general %1171, %1172, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1174 = stablehlo.reshape %1170 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1175 = stablehlo.broadcast_in_dim %1174, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1176 = stablehlo.add %1173, %1175 : tensor<32x8x768xf16>
    %1177 = stablehlo.reshape %1160 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1178 = stablehlo.reshape %1168 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1179 = stablehlo.reshape %1176 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1180 = stablehlo.broadcast_in_dim %204, dims = [0, 3] : (tensor<32x8xi32>) -> tensor<32x1x1x8xi32>
    %1181 = stablehlo.constant dense<0> : tensor<i32>
    %1182 = stablehlo.broadcast_in_dim %1181, dims = [] : (tensor<i32>) -> tensor<32x1x1x8xi32>
    %1183 = stablehlo.compare  GT, %1180, %1182,  SIGNED : (tensor<32x1x1x8xi32>, tensor<32x1x1x8xi32>) -> tensor<32x1x1x8xi1>
    %1184 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1185 = stablehlo.broadcast_in_dim %1184, dims = [] : (tensor<f32>) -> tensor<32x1x1x8xf32>
    %1186 = stablehlo.convert %1185 : (tensor<32x1x1x8xf32>) -> tensor<32x1x1x8xf16>
    %1187 = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
    %1188 = stablehlo.broadcast_in_dim %1187, dims = [] : (tensor<f16>) -> tensor<32x1x1x8xf16>
    %1189 = stablehlo.select %1183, %1186, %1188 : tensor<32x1x1x8xi1>, tensor<32x1x1x8xf16>
    %1190 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %1191 = stablehlo.sqrt %1190 : tensor<f32>
    %1192 = stablehlo.convert %1191 : (tensor<f32>) -> tensor<f16>
    %1193 = stablehlo.broadcast_in_dim %1192, dims = [] : (tensor<f16>) -> tensor<32x8x12x64xf16>
    %1194 = stablehlo.divide %1177, %1193 : tensor<32x8x12x64xf16>
    %1195 = stablehlo.convert %1194 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1196 = stablehlo.convert %1178 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1197 = stablehlo.dot_general %1195, %1196, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<32x8x12x64xf32>, tensor<32x8x12x64xf32>) -> tensor<32x12x8x8xf16>
    %1198 = stablehlo.broadcast_in_dim %1189, dims = [0, 1, 2, 3] : (tensor<32x1x1x8xf16>) -> tensor<32x12x8x8xf16>
    %1199 = stablehlo.add %1197, %1198 : tensor<32x12x8x8xf16>
    %1200 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1201 = stablehlo.reduce(%1199 init: %1200) applies stablehlo.maximum across dimensions = [3] : (tensor<32x12x8x8xf16>, tensor<f16>) -> tensor<32x12x8xf16>
    %1202 = stablehlo.broadcast_in_dim %1201, dims = [0, 1, 2] : (tensor<32x12x8xf16>) -> tensor<32x12x8x1xf16>
    %1203 = stablehlo.broadcast_in_dim %1202, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %1204 = stablehlo.subtract %1199, %1203 : tensor<32x12x8x8xf16>
    %1205 = stablehlo.exponential %1204 : tensor<32x12x8x8xf16>
    %1206 = stablehlo.convert %1205 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1207 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1208 = stablehlo.reduce(%1206 init: %1207) applies stablehlo.add across dimensions = [3] : (tensor<32x12x8x8xf32>, tensor<f32>) -> tensor<32x12x8xf32>
    %1209 = stablehlo.broadcast_in_dim %1208, dims = [0, 1, 2] : (tensor<32x12x8xf32>) -> tensor<32x12x8x1xf32>
    %1210 = stablehlo.convert %1209 : (tensor<32x12x8x1xf32>) -> tensor<32x12x8x1xf16>
    %1211 = stablehlo.broadcast_in_dim %1210, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %1212 = stablehlo.divide %1205, %1211 : tensor<32x12x8x8xf16>
    %1213 = stablehlo.convert %1152 : (tensor<12xi32>) -> tensor<12xf16>
    %1214 = stablehlo.convert %1212 : tensor<32x12x8x8xf16>
    %1215 = stablehlo.convert %1213 : (tensor<12xf16>) -> tensor<12xf32>
    %1216 = stablehlo.convert %1214 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1217 = stablehlo.dot_general %1215, %1216, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<12xf32>, tensor<32x12x8x8xf32>) -> tensor<12x32x8x8xf16>
    %1218 = stablehlo.transpose %1217, dims = [1, 0, 2, 3] : (tensor<12x32x8x8xf16>) -> tensor<32x12x8x8xf16>
    %1219 = stablehlo.convert %1179 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1220 = stablehlo.convert %1218 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1221 = stablehlo.dot_general %1219, %1220, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<32x8x12x64xf32>, tensor<32x12x8x8xf32>) -> tensor<32x12x64x8xf16>
    %1222 = stablehlo.transpose %1221, dims = [0, 3, 1, 2] : (tensor<32x12x64x8xf16>) -> tensor<32x8x12x64xf16>
    %1223 = stablehlo.reshape %1222 : (tensor<32x8x12x64xf16>) -> tensor<32x8x768xf16>
    %1224 = stablehlo.convert %91 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1225 = stablehlo.convert %92 : (tensor<768xf32>) -> tensor<768xf16>
    %1226 = stablehlo.convert %1223 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1227 = stablehlo.convert %1224 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1228 = stablehlo.dot_general %1226, %1227, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1229 = stablehlo.reshape %1225 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1230 = stablehlo.broadcast_in_dim %1229, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1231 = stablehlo.add %1228, %1230 : tensor<32x8x768xf16>
    %1232 = stablehlo.add %1231, %1150 : tensor<32x8x768xf16>
    %1233 = stablehlo.convert %1232 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1234 = stablehlo.multiply %1233, %1233 : tensor<32x8x768xf32>
    %1235 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1236 = stablehlo.reduce(%1233 init: %1235) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1237 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1238 = stablehlo.broadcast_in_dim %1237, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1239 = stablehlo.divide %1236, %1238 : tensor<32x8xf32>
    %1240 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1241 = stablehlo.reduce(%1234 init: %1240) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1242 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1243 = stablehlo.broadcast_in_dim %1242, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1244 = stablehlo.divide %1241, %1243 : tensor<32x8xf32>
    %1245 = stablehlo.multiply %1239, %1239 : tensor<32x8xf32>
    %1246 = stablehlo.subtract %1244, %1245 : tensor<32x8xf32>
    %1247 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1248 = stablehlo.broadcast_in_dim %1247, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1249 = stablehlo.maximum %1248, %1246 : tensor<32x8xf32>
    %1250 = stablehlo.broadcast_in_dim %1239, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1251 = stablehlo.broadcast_in_dim %1249, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1252 = stablehlo.convert %1232 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1253 = stablehlo.broadcast_in_dim %1250, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1254 = stablehlo.subtract %1252, %1253 : tensor<32x8x768xf32>
    %1255 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %1256 = stablehlo.broadcast_in_dim %1255, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %1257 = stablehlo.add %1251, %1256 : tensor<32x8x1xf32>
    %1258 = stablehlo.rsqrt %1257 : tensor<32x8x1xf32>
    %1259 = stablehlo.reshape %93 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1260 = stablehlo.broadcast_in_dim %1258, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1261 = stablehlo.broadcast_in_dim %1259, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1262 = stablehlo.multiply %1260, %1261 : tensor<32x8x768xf32>
    %1263 = stablehlo.multiply %1254, %1262 : tensor<32x8x768xf32>
    %1264 = stablehlo.reshape %94 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1265 = stablehlo.broadcast_in_dim %1264, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1266 = stablehlo.add %1263, %1265 : tensor<32x8x768xf32>
    %1267 = stablehlo.convert %1266 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %1268 = stablehlo.convert %95 : (tensor<768x3072xf32>) -> tensor<768x3072xf16>
    %1269 = stablehlo.convert %96 : (tensor<3072xf32>) -> tensor<3072xf16>
    %1270 = stablehlo.convert %1267 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1271 = stablehlo.convert %1268 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1272 = stablehlo.dot_general %1270, %1271, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x3072xf32>) -> tensor<32x8x3072xf16>
    %1273 = stablehlo.reshape %1269 : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %1274 = stablehlo.broadcast_in_dim %1273, dims = [0, 1, 2] : (tensor<1x1x3072xf16>) -> tensor<32x8x3072xf16>
    %1275 = stablehlo.add %1272, %1274 : tensor<32x8x3072xf16>
    %1276 = stablehlo.constant dense<1.414060e+00> : tensor<f16>
    %1277 = stablehlo.broadcast_in_dim %1276, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1278 = stablehlo.divide %1275, %1277 : tensor<32x8x3072xf16>
    %1279 = stablehlo.custom_call @mhlo.erf(%1278) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf16>
    %1280 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1281 = stablehlo.broadcast_in_dim %1280, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1282 = stablehlo.add %1279, %1281 : tensor<32x8x3072xf16>
    %1283 = stablehlo.multiply %1275, %1282 : tensor<32x8x3072xf16>
    %1284 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %1285 = stablehlo.broadcast_in_dim %1284, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1286 = stablehlo.divide %1283, %1285 : tensor<32x8x3072xf16>
    %1287 = stablehlo.convert %97 : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
    %1288 = stablehlo.convert %98 : (tensor<768xf32>) -> tensor<768xf16>
    %1289 = stablehlo.convert %1286 : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf32>
    %1290 = stablehlo.convert %1287 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1291 = stablehlo.dot_general %1289, %1290, contracting_dims = [2] x [0] : (tensor<32x8x3072xf32>, tensor<3072x768xf32>) -> tensor<32x8x768xf16>
    %1292 = stablehlo.reshape %1288 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1293 = stablehlo.broadcast_in_dim %1292, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1294 = stablehlo.add %1291, %1293 : tensor<32x8x768xf16>
    %1295 = stablehlo.add %1294, %1267 : tensor<32x8x768xf16>
    %1296 = stablehlo.convert %1295 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1297 = stablehlo.multiply %1296, %1296 : tensor<32x8x768xf32>
    %1298 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1299 = stablehlo.reduce(%1296 init: %1298) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1300 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1301 = stablehlo.broadcast_in_dim %1300, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1302 = stablehlo.divide %1299, %1301 : tensor<32x8xf32>
    %1303 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1304 = stablehlo.reduce(%1297 init: %1303) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1305 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1306 = stablehlo.broadcast_in_dim %1305, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1307 = stablehlo.divide %1304, %1306 : tensor<32x8xf32>
    %1308 = stablehlo.multiply %1302, %1302 : tensor<32x8xf32>
    %1309 = stablehlo.subtract %1307, %1308 : tensor<32x8xf32>
    %1310 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1311 = stablehlo.broadcast_in_dim %1310, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1312 = stablehlo.maximum %1311, %1309 : tensor<32x8xf32>
    %1313 = stablehlo.broadcast_in_dim %1302, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1314 = stablehlo.broadcast_in_dim %1312, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1315 = stablehlo.convert %1295 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1316 = stablehlo.broadcast_in_dim %1313, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1317 = stablehlo.subtract %1315, %1316 : tensor<32x8x768xf32>
    %1318 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %1319 = stablehlo.broadcast_in_dim %1318, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %1320 = stablehlo.add %1314, %1319 : tensor<32x8x1xf32>
    %1321 = stablehlo.rsqrt %1320 : tensor<32x8x1xf32>
    %1322 = stablehlo.reshape %99 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1323 = stablehlo.broadcast_in_dim %1321, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1324 = stablehlo.broadcast_in_dim %1322, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1325 = stablehlo.multiply %1323, %1324 : tensor<32x8x768xf32>
    %1326 = stablehlo.multiply %1317, %1325 : tensor<32x8x768xf32>
    %1327 = stablehlo.reshape %100 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1328 = stablehlo.broadcast_in_dim %1327, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1329 = stablehlo.add %1326, %1328 : tensor<32x8x768xf32>
    %1330 = stablehlo.convert %1329 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %1331 = stablehlo.slice %207 [6:7, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
    %1332 = stablehlo.reshape %1331 : (tensor<1x12xi32>) -> tensor<12xi32>
    %1333 = stablehlo.convert %101 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1334 = stablehlo.convert %102 : (tensor<768xf32>) -> tensor<768xf16>
    %1335 = stablehlo.convert %1330 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1336 = stablehlo.convert %1333 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1337 = stablehlo.dot_general %1335, %1336, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1338 = stablehlo.reshape %1334 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1339 = stablehlo.broadcast_in_dim %1338, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1340 = stablehlo.add %1337, %1339 : tensor<32x8x768xf16>
    %1341 = stablehlo.convert %103 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1342 = stablehlo.convert %104 : (tensor<768xf32>) -> tensor<768xf16>
    %1343 = stablehlo.convert %1330 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1344 = stablehlo.convert %1341 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1345 = stablehlo.dot_general %1343, %1344, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1346 = stablehlo.reshape %1342 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1347 = stablehlo.broadcast_in_dim %1346, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1348 = stablehlo.add %1345, %1347 : tensor<32x8x768xf16>
    %1349 = stablehlo.convert %105 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1350 = stablehlo.convert %106 : (tensor<768xf32>) -> tensor<768xf16>
    %1351 = stablehlo.convert %1330 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1352 = stablehlo.convert %1349 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1353 = stablehlo.dot_general %1351, %1352, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1354 = stablehlo.reshape %1350 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1355 = stablehlo.broadcast_in_dim %1354, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1356 = stablehlo.add %1353, %1355 : tensor<32x8x768xf16>
    %1357 = stablehlo.reshape %1340 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1358 = stablehlo.reshape %1348 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1359 = stablehlo.reshape %1356 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1360 = stablehlo.broadcast_in_dim %204, dims = [0, 3] : (tensor<32x8xi32>) -> tensor<32x1x1x8xi32>
    %1361 = stablehlo.constant dense<0> : tensor<i32>
    %1362 = stablehlo.broadcast_in_dim %1361, dims = [] : (tensor<i32>) -> tensor<32x1x1x8xi32>
    %1363 = stablehlo.compare  GT, %1360, %1362,  SIGNED : (tensor<32x1x1x8xi32>, tensor<32x1x1x8xi32>) -> tensor<32x1x1x8xi1>
    %1364 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1365 = stablehlo.broadcast_in_dim %1364, dims = [] : (tensor<f32>) -> tensor<32x1x1x8xf32>
    %1366 = stablehlo.convert %1365 : (tensor<32x1x1x8xf32>) -> tensor<32x1x1x8xf16>
    %1367 = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
    %1368 = stablehlo.broadcast_in_dim %1367, dims = [] : (tensor<f16>) -> tensor<32x1x1x8xf16>
    %1369 = stablehlo.select %1363, %1366, %1368 : tensor<32x1x1x8xi1>, tensor<32x1x1x8xf16>
    %1370 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %1371 = stablehlo.sqrt %1370 : tensor<f32>
    %1372 = stablehlo.convert %1371 : (tensor<f32>) -> tensor<f16>
    %1373 = stablehlo.broadcast_in_dim %1372, dims = [] : (tensor<f16>) -> tensor<32x8x12x64xf16>
    %1374 = stablehlo.divide %1357, %1373 : tensor<32x8x12x64xf16>
    %1375 = stablehlo.convert %1374 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1376 = stablehlo.convert %1358 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1377 = stablehlo.dot_general %1375, %1376, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<32x8x12x64xf32>, tensor<32x8x12x64xf32>) -> tensor<32x12x8x8xf16>
    %1378 = stablehlo.broadcast_in_dim %1369, dims = [0, 1, 2, 3] : (tensor<32x1x1x8xf16>) -> tensor<32x12x8x8xf16>
    %1379 = stablehlo.add %1377, %1378 : tensor<32x12x8x8xf16>
    %1380 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1381 = stablehlo.reduce(%1379 init: %1380) applies stablehlo.maximum across dimensions = [3] : (tensor<32x12x8x8xf16>, tensor<f16>) -> tensor<32x12x8xf16>
    %1382 = stablehlo.broadcast_in_dim %1381, dims = [0, 1, 2] : (tensor<32x12x8xf16>) -> tensor<32x12x8x1xf16>
    %1383 = stablehlo.broadcast_in_dim %1382, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %1384 = stablehlo.subtract %1379, %1383 : tensor<32x12x8x8xf16>
    %1385 = stablehlo.exponential %1384 : tensor<32x12x8x8xf16>
    %1386 = stablehlo.convert %1385 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1387 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1388 = stablehlo.reduce(%1386 init: %1387) applies stablehlo.add across dimensions = [3] : (tensor<32x12x8x8xf32>, tensor<f32>) -> tensor<32x12x8xf32>
    %1389 = stablehlo.broadcast_in_dim %1388, dims = [0, 1, 2] : (tensor<32x12x8xf32>) -> tensor<32x12x8x1xf32>
    %1390 = stablehlo.convert %1389 : (tensor<32x12x8x1xf32>) -> tensor<32x12x8x1xf16>
    %1391 = stablehlo.broadcast_in_dim %1390, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %1392 = stablehlo.divide %1385, %1391 : tensor<32x12x8x8xf16>
    %1393 = stablehlo.convert %1332 : (tensor<12xi32>) -> tensor<12xf16>
    %1394 = stablehlo.convert %1392 : tensor<32x12x8x8xf16>
    %1395 = stablehlo.convert %1393 : (tensor<12xf16>) -> tensor<12xf32>
    %1396 = stablehlo.convert %1394 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1397 = stablehlo.dot_general %1395, %1396, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<12xf32>, tensor<32x12x8x8xf32>) -> tensor<12x32x8x8xf16>
    %1398 = stablehlo.transpose %1397, dims = [1, 0, 2, 3] : (tensor<12x32x8x8xf16>) -> tensor<32x12x8x8xf16>
    %1399 = stablehlo.convert %1359 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1400 = stablehlo.convert %1398 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1401 = stablehlo.dot_general %1399, %1400, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<32x8x12x64xf32>, tensor<32x12x8x8xf32>) -> tensor<32x12x64x8xf16>
    %1402 = stablehlo.transpose %1401, dims = [0, 3, 1, 2] : (tensor<32x12x64x8xf16>) -> tensor<32x8x12x64xf16>
    %1403 = stablehlo.reshape %1402 : (tensor<32x8x12x64xf16>) -> tensor<32x8x768xf16>
    %1404 = stablehlo.convert %107 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1405 = stablehlo.convert %108 : (tensor<768xf32>) -> tensor<768xf16>
    %1406 = stablehlo.convert %1403 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1407 = stablehlo.convert %1404 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1408 = stablehlo.dot_general %1406, %1407, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1409 = stablehlo.reshape %1405 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1410 = stablehlo.broadcast_in_dim %1409, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1411 = stablehlo.add %1408, %1410 : tensor<32x8x768xf16>
    %1412 = stablehlo.add %1411, %1330 : tensor<32x8x768xf16>
    %1413 = stablehlo.convert %1412 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1414 = stablehlo.multiply %1413, %1413 : tensor<32x8x768xf32>
    %1415 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1416 = stablehlo.reduce(%1413 init: %1415) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1417 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1418 = stablehlo.broadcast_in_dim %1417, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1419 = stablehlo.divide %1416, %1418 : tensor<32x8xf32>
    %1420 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1421 = stablehlo.reduce(%1414 init: %1420) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1422 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1423 = stablehlo.broadcast_in_dim %1422, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1424 = stablehlo.divide %1421, %1423 : tensor<32x8xf32>
    %1425 = stablehlo.multiply %1419, %1419 : tensor<32x8xf32>
    %1426 = stablehlo.subtract %1424, %1425 : tensor<32x8xf32>
    %1427 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1428 = stablehlo.broadcast_in_dim %1427, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1429 = stablehlo.maximum %1428, %1426 : tensor<32x8xf32>
    %1430 = stablehlo.broadcast_in_dim %1419, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1431 = stablehlo.broadcast_in_dim %1429, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1432 = stablehlo.convert %1412 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1433 = stablehlo.broadcast_in_dim %1430, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1434 = stablehlo.subtract %1432, %1433 : tensor<32x8x768xf32>
    %1435 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %1436 = stablehlo.broadcast_in_dim %1435, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %1437 = stablehlo.add %1431, %1436 : tensor<32x8x1xf32>
    %1438 = stablehlo.rsqrt %1437 : tensor<32x8x1xf32>
    %1439 = stablehlo.reshape %109 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1440 = stablehlo.broadcast_in_dim %1438, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1441 = stablehlo.broadcast_in_dim %1439, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1442 = stablehlo.multiply %1440, %1441 : tensor<32x8x768xf32>
    %1443 = stablehlo.multiply %1434, %1442 : tensor<32x8x768xf32>
    %1444 = stablehlo.reshape %110 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1445 = stablehlo.broadcast_in_dim %1444, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1446 = stablehlo.add %1443, %1445 : tensor<32x8x768xf32>
    %1447 = stablehlo.convert %1446 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %1448 = stablehlo.convert %111 : (tensor<768x3072xf32>) -> tensor<768x3072xf16>
    %1449 = stablehlo.convert %112 : (tensor<3072xf32>) -> tensor<3072xf16>
    %1450 = stablehlo.convert %1447 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1451 = stablehlo.convert %1448 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1452 = stablehlo.dot_general %1450, %1451, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x3072xf32>) -> tensor<32x8x3072xf16>
    %1453 = stablehlo.reshape %1449 : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %1454 = stablehlo.broadcast_in_dim %1453, dims = [0, 1, 2] : (tensor<1x1x3072xf16>) -> tensor<32x8x3072xf16>
    %1455 = stablehlo.add %1452, %1454 : tensor<32x8x3072xf16>
    %1456 = stablehlo.constant dense<1.414060e+00> : tensor<f16>
    %1457 = stablehlo.broadcast_in_dim %1456, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1458 = stablehlo.divide %1455, %1457 : tensor<32x8x3072xf16>
    %1459 = stablehlo.custom_call @mhlo.erf(%1458) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf16>
    %1460 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1461 = stablehlo.broadcast_in_dim %1460, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1462 = stablehlo.add %1459, %1461 : tensor<32x8x3072xf16>
    %1463 = stablehlo.multiply %1455, %1462 : tensor<32x8x3072xf16>
    %1464 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %1465 = stablehlo.broadcast_in_dim %1464, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1466 = stablehlo.divide %1463, %1465 : tensor<32x8x3072xf16>
    %1467 = stablehlo.convert %113 : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
    %1468 = stablehlo.convert %114 : (tensor<768xf32>) -> tensor<768xf16>
    %1469 = stablehlo.convert %1466 : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf32>
    %1470 = stablehlo.convert %1467 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1471 = stablehlo.dot_general %1469, %1470, contracting_dims = [2] x [0] : (tensor<32x8x3072xf32>, tensor<3072x768xf32>) -> tensor<32x8x768xf16>
    %1472 = stablehlo.reshape %1468 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1473 = stablehlo.broadcast_in_dim %1472, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1474 = stablehlo.add %1471, %1473 : tensor<32x8x768xf16>
    %1475 = stablehlo.add %1474, %1447 : tensor<32x8x768xf16>
    %1476 = stablehlo.convert %1475 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1477 = stablehlo.multiply %1476, %1476 : tensor<32x8x768xf32>
    %1478 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1479 = stablehlo.reduce(%1476 init: %1478) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1480 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1481 = stablehlo.broadcast_in_dim %1480, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1482 = stablehlo.divide %1479, %1481 : tensor<32x8xf32>
    %1483 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1484 = stablehlo.reduce(%1477 init: %1483) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1485 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1486 = stablehlo.broadcast_in_dim %1485, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1487 = stablehlo.divide %1484, %1486 : tensor<32x8xf32>
    %1488 = stablehlo.multiply %1482, %1482 : tensor<32x8xf32>
    %1489 = stablehlo.subtract %1487, %1488 : tensor<32x8xf32>
    %1490 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1491 = stablehlo.broadcast_in_dim %1490, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1492 = stablehlo.maximum %1491, %1489 : tensor<32x8xf32>
    %1493 = stablehlo.broadcast_in_dim %1482, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1494 = stablehlo.broadcast_in_dim %1492, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1495 = stablehlo.convert %1475 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1496 = stablehlo.broadcast_in_dim %1493, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1497 = stablehlo.subtract %1495, %1496 : tensor<32x8x768xf32>
    %1498 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %1499 = stablehlo.broadcast_in_dim %1498, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %1500 = stablehlo.add %1494, %1499 : tensor<32x8x1xf32>
    %1501 = stablehlo.rsqrt %1500 : tensor<32x8x1xf32>
    %1502 = stablehlo.reshape %115 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1503 = stablehlo.broadcast_in_dim %1501, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1504 = stablehlo.broadcast_in_dim %1502, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1505 = stablehlo.multiply %1503, %1504 : tensor<32x8x768xf32>
    %1506 = stablehlo.multiply %1497, %1505 : tensor<32x8x768xf32>
    %1507 = stablehlo.reshape %116 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1508 = stablehlo.broadcast_in_dim %1507, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1509 = stablehlo.add %1506, %1508 : tensor<32x8x768xf32>
    %1510 = stablehlo.convert %1509 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %1511 = stablehlo.slice %207 [7:8, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
    %1512 = stablehlo.reshape %1511 : (tensor<1x12xi32>) -> tensor<12xi32>
    %1513 = stablehlo.convert %117 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1514 = stablehlo.convert %118 : (tensor<768xf32>) -> tensor<768xf16>
    %1515 = stablehlo.convert %1510 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1516 = stablehlo.convert %1513 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1517 = stablehlo.dot_general %1515, %1516, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1518 = stablehlo.reshape %1514 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1519 = stablehlo.broadcast_in_dim %1518, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1520 = stablehlo.add %1517, %1519 : tensor<32x8x768xf16>
    %1521 = stablehlo.convert %119 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1522 = stablehlo.convert %120 : (tensor<768xf32>) -> tensor<768xf16>
    %1523 = stablehlo.convert %1510 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1524 = stablehlo.convert %1521 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1525 = stablehlo.dot_general %1523, %1524, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1526 = stablehlo.reshape %1522 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1527 = stablehlo.broadcast_in_dim %1526, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1528 = stablehlo.add %1525, %1527 : tensor<32x8x768xf16>
    %1529 = stablehlo.convert %121 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1530 = stablehlo.convert %122 : (tensor<768xf32>) -> tensor<768xf16>
    %1531 = stablehlo.convert %1510 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1532 = stablehlo.convert %1529 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1533 = stablehlo.dot_general %1531, %1532, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1534 = stablehlo.reshape %1530 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1535 = stablehlo.broadcast_in_dim %1534, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1536 = stablehlo.add %1533, %1535 : tensor<32x8x768xf16>
    %1537 = stablehlo.reshape %1520 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1538 = stablehlo.reshape %1528 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1539 = stablehlo.reshape %1536 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1540 = stablehlo.broadcast_in_dim %204, dims = [0, 3] : (tensor<32x8xi32>) -> tensor<32x1x1x8xi32>
    %1541 = stablehlo.constant dense<0> : tensor<i32>
    %1542 = stablehlo.broadcast_in_dim %1541, dims = [] : (tensor<i32>) -> tensor<32x1x1x8xi32>
    %1543 = stablehlo.compare  GT, %1540, %1542,  SIGNED : (tensor<32x1x1x8xi32>, tensor<32x1x1x8xi32>) -> tensor<32x1x1x8xi1>
    %1544 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1545 = stablehlo.broadcast_in_dim %1544, dims = [] : (tensor<f32>) -> tensor<32x1x1x8xf32>
    %1546 = stablehlo.convert %1545 : (tensor<32x1x1x8xf32>) -> tensor<32x1x1x8xf16>
    %1547 = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
    %1548 = stablehlo.broadcast_in_dim %1547, dims = [] : (tensor<f16>) -> tensor<32x1x1x8xf16>
    %1549 = stablehlo.select %1543, %1546, %1548 : tensor<32x1x1x8xi1>, tensor<32x1x1x8xf16>
    %1550 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %1551 = stablehlo.sqrt %1550 : tensor<f32>
    %1552 = stablehlo.convert %1551 : (tensor<f32>) -> tensor<f16>
    %1553 = stablehlo.broadcast_in_dim %1552, dims = [] : (tensor<f16>) -> tensor<32x8x12x64xf16>
    %1554 = stablehlo.divide %1537, %1553 : tensor<32x8x12x64xf16>
    %1555 = stablehlo.convert %1554 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1556 = stablehlo.convert %1538 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1557 = stablehlo.dot_general %1555, %1556, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<32x8x12x64xf32>, tensor<32x8x12x64xf32>) -> tensor<32x12x8x8xf16>
    %1558 = stablehlo.broadcast_in_dim %1549, dims = [0, 1, 2, 3] : (tensor<32x1x1x8xf16>) -> tensor<32x12x8x8xf16>
    %1559 = stablehlo.add %1557, %1558 : tensor<32x12x8x8xf16>
    %1560 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1561 = stablehlo.reduce(%1559 init: %1560) applies stablehlo.maximum across dimensions = [3] : (tensor<32x12x8x8xf16>, tensor<f16>) -> tensor<32x12x8xf16>
    %1562 = stablehlo.broadcast_in_dim %1561, dims = [0, 1, 2] : (tensor<32x12x8xf16>) -> tensor<32x12x8x1xf16>
    %1563 = stablehlo.broadcast_in_dim %1562, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %1564 = stablehlo.subtract %1559, %1563 : tensor<32x12x8x8xf16>
    %1565 = stablehlo.exponential %1564 : tensor<32x12x8x8xf16>
    %1566 = stablehlo.convert %1565 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1567 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1568 = stablehlo.reduce(%1566 init: %1567) applies stablehlo.add across dimensions = [3] : (tensor<32x12x8x8xf32>, tensor<f32>) -> tensor<32x12x8xf32>
    %1569 = stablehlo.broadcast_in_dim %1568, dims = [0, 1, 2] : (tensor<32x12x8xf32>) -> tensor<32x12x8x1xf32>
    %1570 = stablehlo.convert %1569 : (tensor<32x12x8x1xf32>) -> tensor<32x12x8x1xf16>
    %1571 = stablehlo.broadcast_in_dim %1570, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %1572 = stablehlo.divide %1565, %1571 : tensor<32x12x8x8xf16>
    %1573 = stablehlo.convert %1512 : (tensor<12xi32>) -> tensor<12xf16>
    %1574 = stablehlo.convert %1572 : tensor<32x12x8x8xf16>
    %1575 = stablehlo.convert %1573 : (tensor<12xf16>) -> tensor<12xf32>
    %1576 = stablehlo.convert %1574 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1577 = stablehlo.dot_general %1575, %1576, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<12xf32>, tensor<32x12x8x8xf32>) -> tensor<12x32x8x8xf16>
    %1578 = stablehlo.transpose %1577, dims = [1, 0, 2, 3] : (tensor<12x32x8x8xf16>) -> tensor<32x12x8x8xf16>
    %1579 = stablehlo.convert %1539 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1580 = stablehlo.convert %1578 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1581 = stablehlo.dot_general %1579, %1580, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<32x8x12x64xf32>, tensor<32x12x8x8xf32>) -> tensor<32x12x64x8xf16>
    %1582 = stablehlo.transpose %1581, dims = [0, 3, 1, 2] : (tensor<32x12x64x8xf16>) -> tensor<32x8x12x64xf16>
    %1583 = stablehlo.reshape %1582 : (tensor<32x8x12x64xf16>) -> tensor<32x8x768xf16>
    %1584 = stablehlo.convert %123 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1585 = stablehlo.convert %124 : (tensor<768xf32>) -> tensor<768xf16>
    %1586 = stablehlo.convert %1583 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1587 = stablehlo.convert %1584 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1588 = stablehlo.dot_general %1586, %1587, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1589 = stablehlo.reshape %1585 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1590 = stablehlo.broadcast_in_dim %1589, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1591 = stablehlo.add %1588, %1590 : tensor<32x8x768xf16>
    %1592 = stablehlo.add %1591, %1510 : tensor<32x8x768xf16>
    %1593 = stablehlo.convert %1592 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1594 = stablehlo.multiply %1593, %1593 : tensor<32x8x768xf32>
    %1595 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1596 = stablehlo.reduce(%1593 init: %1595) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1597 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1598 = stablehlo.broadcast_in_dim %1597, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1599 = stablehlo.divide %1596, %1598 : tensor<32x8xf32>
    %1600 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1601 = stablehlo.reduce(%1594 init: %1600) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1602 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1603 = stablehlo.broadcast_in_dim %1602, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1604 = stablehlo.divide %1601, %1603 : tensor<32x8xf32>
    %1605 = stablehlo.multiply %1599, %1599 : tensor<32x8xf32>
    %1606 = stablehlo.subtract %1604, %1605 : tensor<32x8xf32>
    %1607 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1608 = stablehlo.broadcast_in_dim %1607, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1609 = stablehlo.maximum %1608, %1606 : tensor<32x8xf32>
    %1610 = stablehlo.broadcast_in_dim %1599, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1611 = stablehlo.broadcast_in_dim %1609, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1612 = stablehlo.convert %1592 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1613 = stablehlo.broadcast_in_dim %1610, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1614 = stablehlo.subtract %1612, %1613 : tensor<32x8x768xf32>
    %1615 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %1616 = stablehlo.broadcast_in_dim %1615, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %1617 = stablehlo.add %1611, %1616 : tensor<32x8x1xf32>
    %1618 = stablehlo.rsqrt %1617 : tensor<32x8x1xf32>
    %1619 = stablehlo.reshape %125 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1620 = stablehlo.broadcast_in_dim %1618, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1621 = stablehlo.broadcast_in_dim %1619, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1622 = stablehlo.multiply %1620, %1621 : tensor<32x8x768xf32>
    %1623 = stablehlo.multiply %1614, %1622 : tensor<32x8x768xf32>
    %1624 = stablehlo.reshape %126 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1625 = stablehlo.broadcast_in_dim %1624, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1626 = stablehlo.add %1623, %1625 : tensor<32x8x768xf32>
    %1627 = stablehlo.convert %1626 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %1628 = stablehlo.convert %127 : (tensor<768x3072xf32>) -> tensor<768x3072xf16>
    %1629 = stablehlo.convert %128 : (tensor<3072xf32>) -> tensor<3072xf16>
    %1630 = stablehlo.convert %1627 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1631 = stablehlo.convert %1628 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1632 = stablehlo.dot_general %1630, %1631, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x3072xf32>) -> tensor<32x8x3072xf16>
    %1633 = stablehlo.reshape %1629 : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %1634 = stablehlo.broadcast_in_dim %1633, dims = [0, 1, 2] : (tensor<1x1x3072xf16>) -> tensor<32x8x3072xf16>
    %1635 = stablehlo.add %1632, %1634 : tensor<32x8x3072xf16>
    %1636 = stablehlo.constant dense<1.414060e+00> : tensor<f16>
    %1637 = stablehlo.broadcast_in_dim %1636, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1638 = stablehlo.divide %1635, %1637 : tensor<32x8x3072xf16>
    %1639 = stablehlo.custom_call @mhlo.erf(%1638) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf16>
    %1640 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1641 = stablehlo.broadcast_in_dim %1640, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1642 = stablehlo.add %1639, %1641 : tensor<32x8x3072xf16>
    %1643 = stablehlo.multiply %1635, %1642 : tensor<32x8x3072xf16>
    %1644 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %1645 = stablehlo.broadcast_in_dim %1644, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1646 = stablehlo.divide %1643, %1645 : tensor<32x8x3072xf16>
    %1647 = stablehlo.convert %129 : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
    %1648 = stablehlo.convert %130 : (tensor<768xf32>) -> tensor<768xf16>
    %1649 = stablehlo.convert %1646 : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf32>
    %1650 = stablehlo.convert %1647 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1651 = stablehlo.dot_general %1649, %1650, contracting_dims = [2] x [0] : (tensor<32x8x3072xf32>, tensor<3072x768xf32>) -> tensor<32x8x768xf16>
    %1652 = stablehlo.reshape %1648 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1653 = stablehlo.broadcast_in_dim %1652, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1654 = stablehlo.add %1651, %1653 : tensor<32x8x768xf16>
    %1655 = stablehlo.add %1654, %1627 : tensor<32x8x768xf16>
    %1656 = stablehlo.convert %1655 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1657 = stablehlo.multiply %1656, %1656 : tensor<32x8x768xf32>
    %1658 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1659 = stablehlo.reduce(%1656 init: %1658) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1660 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1661 = stablehlo.broadcast_in_dim %1660, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1662 = stablehlo.divide %1659, %1661 : tensor<32x8xf32>
    %1663 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1664 = stablehlo.reduce(%1657 init: %1663) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1665 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1666 = stablehlo.broadcast_in_dim %1665, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1667 = stablehlo.divide %1664, %1666 : tensor<32x8xf32>
    %1668 = stablehlo.multiply %1662, %1662 : tensor<32x8xf32>
    %1669 = stablehlo.subtract %1667, %1668 : tensor<32x8xf32>
    %1670 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1671 = stablehlo.broadcast_in_dim %1670, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1672 = stablehlo.maximum %1671, %1669 : tensor<32x8xf32>
    %1673 = stablehlo.broadcast_in_dim %1662, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1674 = stablehlo.broadcast_in_dim %1672, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1675 = stablehlo.convert %1655 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1676 = stablehlo.broadcast_in_dim %1673, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1677 = stablehlo.subtract %1675, %1676 : tensor<32x8x768xf32>
    %1678 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %1679 = stablehlo.broadcast_in_dim %1678, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %1680 = stablehlo.add %1674, %1679 : tensor<32x8x1xf32>
    %1681 = stablehlo.rsqrt %1680 : tensor<32x8x1xf32>
    %1682 = stablehlo.reshape %131 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1683 = stablehlo.broadcast_in_dim %1681, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1684 = stablehlo.broadcast_in_dim %1682, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1685 = stablehlo.multiply %1683, %1684 : tensor<32x8x768xf32>
    %1686 = stablehlo.multiply %1677, %1685 : tensor<32x8x768xf32>
    %1687 = stablehlo.reshape %132 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1688 = stablehlo.broadcast_in_dim %1687, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1689 = stablehlo.add %1686, %1688 : tensor<32x8x768xf32>
    %1690 = stablehlo.convert %1689 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %1691 = stablehlo.slice %207 [8:9, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
    %1692 = stablehlo.reshape %1691 : (tensor<1x12xi32>) -> tensor<12xi32>
    %1693 = stablehlo.convert %133 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1694 = stablehlo.convert %134 : (tensor<768xf32>) -> tensor<768xf16>
    %1695 = stablehlo.convert %1690 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1696 = stablehlo.convert %1693 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1697 = stablehlo.dot_general %1695, %1696, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1698 = stablehlo.reshape %1694 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1699 = stablehlo.broadcast_in_dim %1698, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1700 = stablehlo.add %1697, %1699 : tensor<32x8x768xf16>
    %1701 = stablehlo.convert %135 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1702 = stablehlo.convert %136 : (tensor<768xf32>) -> tensor<768xf16>
    %1703 = stablehlo.convert %1690 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1704 = stablehlo.convert %1701 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1705 = stablehlo.dot_general %1703, %1704, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1706 = stablehlo.reshape %1702 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1707 = stablehlo.broadcast_in_dim %1706, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1708 = stablehlo.add %1705, %1707 : tensor<32x8x768xf16>
    %1709 = stablehlo.convert %137 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1710 = stablehlo.convert %138 : (tensor<768xf32>) -> tensor<768xf16>
    %1711 = stablehlo.convert %1690 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1712 = stablehlo.convert %1709 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1713 = stablehlo.dot_general %1711, %1712, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1714 = stablehlo.reshape %1710 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1715 = stablehlo.broadcast_in_dim %1714, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1716 = stablehlo.add %1713, %1715 : tensor<32x8x768xf16>
    %1717 = stablehlo.reshape %1700 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1718 = stablehlo.reshape %1708 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1719 = stablehlo.reshape %1716 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1720 = stablehlo.broadcast_in_dim %204, dims = [0, 3] : (tensor<32x8xi32>) -> tensor<32x1x1x8xi32>
    %1721 = stablehlo.constant dense<0> : tensor<i32>
    %1722 = stablehlo.broadcast_in_dim %1721, dims = [] : (tensor<i32>) -> tensor<32x1x1x8xi32>
    %1723 = stablehlo.compare  GT, %1720, %1722,  SIGNED : (tensor<32x1x1x8xi32>, tensor<32x1x1x8xi32>) -> tensor<32x1x1x8xi1>
    %1724 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1725 = stablehlo.broadcast_in_dim %1724, dims = [] : (tensor<f32>) -> tensor<32x1x1x8xf32>
    %1726 = stablehlo.convert %1725 : (tensor<32x1x1x8xf32>) -> tensor<32x1x1x8xf16>
    %1727 = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
    %1728 = stablehlo.broadcast_in_dim %1727, dims = [] : (tensor<f16>) -> tensor<32x1x1x8xf16>
    %1729 = stablehlo.select %1723, %1726, %1728 : tensor<32x1x1x8xi1>, tensor<32x1x1x8xf16>
    %1730 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %1731 = stablehlo.sqrt %1730 : tensor<f32>
    %1732 = stablehlo.convert %1731 : (tensor<f32>) -> tensor<f16>
    %1733 = stablehlo.broadcast_in_dim %1732, dims = [] : (tensor<f16>) -> tensor<32x8x12x64xf16>
    %1734 = stablehlo.divide %1717, %1733 : tensor<32x8x12x64xf16>
    %1735 = stablehlo.convert %1734 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1736 = stablehlo.convert %1718 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1737 = stablehlo.dot_general %1735, %1736, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<32x8x12x64xf32>, tensor<32x8x12x64xf32>) -> tensor<32x12x8x8xf16>
    %1738 = stablehlo.broadcast_in_dim %1729, dims = [0, 1, 2, 3] : (tensor<32x1x1x8xf16>) -> tensor<32x12x8x8xf16>
    %1739 = stablehlo.add %1737, %1738 : tensor<32x12x8x8xf16>
    %1740 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1741 = stablehlo.reduce(%1739 init: %1740) applies stablehlo.maximum across dimensions = [3] : (tensor<32x12x8x8xf16>, tensor<f16>) -> tensor<32x12x8xf16>
    %1742 = stablehlo.broadcast_in_dim %1741, dims = [0, 1, 2] : (tensor<32x12x8xf16>) -> tensor<32x12x8x1xf16>
    %1743 = stablehlo.broadcast_in_dim %1742, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %1744 = stablehlo.subtract %1739, %1743 : tensor<32x12x8x8xf16>
    %1745 = stablehlo.exponential %1744 : tensor<32x12x8x8xf16>
    %1746 = stablehlo.convert %1745 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1747 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1748 = stablehlo.reduce(%1746 init: %1747) applies stablehlo.add across dimensions = [3] : (tensor<32x12x8x8xf32>, tensor<f32>) -> tensor<32x12x8xf32>
    %1749 = stablehlo.broadcast_in_dim %1748, dims = [0, 1, 2] : (tensor<32x12x8xf32>) -> tensor<32x12x8x1xf32>
    %1750 = stablehlo.convert %1749 : (tensor<32x12x8x1xf32>) -> tensor<32x12x8x1xf16>
    %1751 = stablehlo.broadcast_in_dim %1750, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %1752 = stablehlo.divide %1745, %1751 : tensor<32x12x8x8xf16>
    %1753 = stablehlo.convert %1692 : (tensor<12xi32>) -> tensor<12xf16>
    %1754 = stablehlo.convert %1752 : tensor<32x12x8x8xf16>
    %1755 = stablehlo.convert %1753 : (tensor<12xf16>) -> tensor<12xf32>
    %1756 = stablehlo.convert %1754 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1757 = stablehlo.dot_general %1755, %1756, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<12xf32>, tensor<32x12x8x8xf32>) -> tensor<12x32x8x8xf16>
    %1758 = stablehlo.transpose %1757, dims = [1, 0, 2, 3] : (tensor<12x32x8x8xf16>) -> tensor<32x12x8x8xf16>
    %1759 = stablehlo.convert %1719 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1760 = stablehlo.convert %1758 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1761 = stablehlo.dot_general %1759, %1760, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<32x8x12x64xf32>, tensor<32x12x8x8xf32>) -> tensor<32x12x64x8xf16>
    %1762 = stablehlo.transpose %1761, dims = [0, 3, 1, 2] : (tensor<32x12x64x8xf16>) -> tensor<32x8x12x64xf16>
    %1763 = stablehlo.reshape %1762 : (tensor<32x8x12x64xf16>) -> tensor<32x8x768xf16>
    %1764 = stablehlo.convert %139 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1765 = stablehlo.convert %140 : (tensor<768xf32>) -> tensor<768xf16>
    %1766 = stablehlo.convert %1763 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1767 = stablehlo.convert %1764 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1768 = stablehlo.dot_general %1766, %1767, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1769 = stablehlo.reshape %1765 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1770 = stablehlo.broadcast_in_dim %1769, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1771 = stablehlo.add %1768, %1770 : tensor<32x8x768xf16>
    %1772 = stablehlo.add %1771, %1690 : tensor<32x8x768xf16>
    %1773 = stablehlo.convert %1772 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1774 = stablehlo.multiply %1773, %1773 : tensor<32x8x768xf32>
    %1775 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1776 = stablehlo.reduce(%1773 init: %1775) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1777 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1778 = stablehlo.broadcast_in_dim %1777, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1779 = stablehlo.divide %1776, %1778 : tensor<32x8xf32>
    %1780 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1781 = stablehlo.reduce(%1774 init: %1780) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1782 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1783 = stablehlo.broadcast_in_dim %1782, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1784 = stablehlo.divide %1781, %1783 : tensor<32x8xf32>
    %1785 = stablehlo.multiply %1779, %1779 : tensor<32x8xf32>
    %1786 = stablehlo.subtract %1784, %1785 : tensor<32x8xf32>
    %1787 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1788 = stablehlo.broadcast_in_dim %1787, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1789 = stablehlo.maximum %1788, %1786 : tensor<32x8xf32>
    %1790 = stablehlo.broadcast_in_dim %1779, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1791 = stablehlo.broadcast_in_dim %1789, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1792 = stablehlo.convert %1772 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1793 = stablehlo.broadcast_in_dim %1790, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1794 = stablehlo.subtract %1792, %1793 : tensor<32x8x768xf32>
    %1795 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %1796 = stablehlo.broadcast_in_dim %1795, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %1797 = stablehlo.add %1791, %1796 : tensor<32x8x1xf32>
    %1798 = stablehlo.rsqrt %1797 : tensor<32x8x1xf32>
    %1799 = stablehlo.reshape %141 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1800 = stablehlo.broadcast_in_dim %1798, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1801 = stablehlo.broadcast_in_dim %1799, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1802 = stablehlo.multiply %1800, %1801 : tensor<32x8x768xf32>
    %1803 = stablehlo.multiply %1794, %1802 : tensor<32x8x768xf32>
    %1804 = stablehlo.reshape %142 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1805 = stablehlo.broadcast_in_dim %1804, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1806 = stablehlo.add %1803, %1805 : tensor<32x8x768xf32>
    %1807 = stablehlo.convert %1806 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %1808 = stablehlo.convert %143 : (tensor<768x3072xf32>) -> tensor<768x3072xf16>
    %1809 = stablehlo.convert %144 : (tensor<3072xf32>) -> tensor<3072xf16>
    %1810 = stablehlo.convert %1807 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1811 = stablehlo.convert %1808 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1812 = stablehlo.dot_general %1810, %1811, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x3072xf32>) -> tensor<32x8x3072xf16>
    %1813 = stablehlo.reshape %1809 : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %1814 = stablehlo.broadcast_in_dim %1813, dims = [0, 1, 2] : (tensor<1x1x3072xf16>) -> tensor<32x8x3072xf16>
    %1815 = stablehlo.add %1812, %1814 : tensor<32x8x3072xf16>
    %1816 = stablehlo.constant dense<1.414060e+00> : tensor<f16>
    %1817 = stablehlo.broadcast_in_dim %1816, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1818 = stablehlo.divide %1815, %1817 : tensor<32x8x3072xf16>
    %1819 = stablehlo.custom_call @mhlo.erf(%1818) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf16>
    %1820 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %1821 = stablehlo.broadcast_in_dim %1820, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1822 = stablehlo.add %1819, %1821 : tensor<32x8x3072xf16>
    %1823 = stablehlo.multiply %1815, %1822 : tensor<32x8x3072xf16>
    %1824 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %1825 = stablehlo.broadcast_in_dim %1824, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1826 = stablehlo.divide %1823, %1825 : tensor<32x8x3072xf16>
    %1827 = stablehlo.convert %145 : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
    %1828 = stablehlo.convert %146 : (tensor<768xf32>) -> tensor<768xf16>
    %1829 = stablehlo.convert %1826 : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf32>
    %1830 = stablehlo.convert %1827 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1831 = stablehlo.dot_general %1829, %1830, contracting_dims = [2] x [0] : (tensor<32x8x3072xf32>, tensor<3072x768xf32>) -> tensor<32x8x768xf16>
    %1832 = stablehlo.reshape %1828 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1833 = stablehlo.broadcast_in_dim %1832, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1834 = stablehlo.add %1831, %1833 : tensor<32x8x768xf16>
    %1835 = stablehlo.add %1834, %1807 : tensor<32x8x768xf16>
    %1836 = stablehlo.convert %1835 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1837 = stablehlo.multiply %1836, %1836 : tensor<32x8x768xf32>
    %1838 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1839 = stablehlo.reduce(%1836 init: %1838) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1840 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1841 = stablehlo.broadcast_in_dim %1840, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1842 = stablehlo.divide %1839, %1841 : tensor<32x8xf32>
    %1843 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1844 = stablehlo.reduce(%1837 init: %1843) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1845 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1846 = stablehlo.broadcast_in_dim %1845, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1847 = stablehlo.divide %1844, %1846 : tensor<32x8xf32>
    %1848 = stablehlo.multiply %1842, %1842 : tensor<32x8xf32>
    %1849 = stablehlo.subtract %1847, %1848 : tensor<32x8xf32>
    %1850 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1851 = stablehlo.broadcast_in_dim %1850, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1852 = stablehlo.maximum %1851, %1849 : tensor<32x8xf32>
    %1853 = stablehlo.broadcast_in_dim %1842, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1854 = stablehlo.broadcast_in_dim %1852, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1855 = stablehlo.convert %1835 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1856 = stablehlo.broadcast_in_dim %1853, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1857 = stablehlo.subtract %1855, %1856 : tensor<32x8x768xf32>
    %1858 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %1859 = stablehlo.broadcast_in_dim %1858, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %1860 = stablehlo.add %1854, %1859 : tensor<32x8x1xf32>
    %1861 = stablehlo.rsqrt %1860 : tensor<32x8x1xf32>
    %1862 = stablehlo.reshape %147 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1863 = stablehlo.broadcast_in_dim %1861, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1864 = stablehlo.broadcast_in_dim %1862, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1865 = stablehlo.multiply %1863, %1864 : tensor<32x8x768xf32>
    %1866 = stablehlo.multiply %1857, %1865 : tensor<32x8x768xf32>
    %1867 = stablehlo.reshape %148 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1868 = stablehlo.broadcast_in_dim %1867, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1869 = stablehlo.add %1866, %1868 : tensor<32x8x768xf32>
    %1870 = stablehlo.convert %1869 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %1871 = stablehlo.slice %207 [9:10, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
    %1872 = stablehlo.reshape %1871 : (tensor<1x12xi32>) -> tensor<12xi32>
    %1873 = stablehlo.convert %149 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1874 = stablehlo.convert %150 : (tensor<768xf32>) -> tensor<768xf16>
    %1875 = stablehlo.convert %1870 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1876 = stablehlo.convert %1873 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1877 = stablehlo.dot_general %1875, %1876, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1878 = stablehlo.reshape %1874 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1879 = stablehlo.broadcast_in_dim %1878, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1880 = stablehlo.add %1877, %1879 : tensor<32x8x768xf16>
    %1881 = stablehlo.convert %151 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1882 = stablehlo.convert %152 : (tensor<768xf32>) -> tensor<768xf16>
    %1883 = stablehlo.convert %1870 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1884 = stablehlo.convert %1881 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1885 = stablehlo.dot_general %1883, %1884, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1886 = stablehlo.reshape %1882 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1887 = stablehlo.broadcast_in_dim %1886, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1888 = stablehlo.add %1885, %1887 : tensor<32x8x768xf16>
    %1889 = stablehlo.convert %153 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1890 = stablehlo.convert %154 : (tensor<768xf32>) -> tensor<768xf16>
    %1891 = stablehlo.convert %1870 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1892 = stablehlo.convert %1889 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1893 = stablehlo.dot_general %1891, %1892, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1894 = stablehlo.reshape %1890 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1895 = stablehlo.broadcast_in_dim %1894, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1896 = stablehlo.add %1893, %1895 : tensor<32x8x768xf16>
    %1897 = stablehlo.reshape %1880 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1898 = stablehlo.reshape %1888 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1899 = stablehlo.reshape %1896 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %1900 = stablehlo.broadcast_in_dim %204, dims = [0, 3] : (tensor<32x8xi32>) -> tensor<32x1x1x8xi32>
    %1901 = stablehlo.constant dense<0> : tensor<i32>
    %1902 = stablehlo.broadcast_in_dim %1901, dims = [] : (tensor<i32>) -> tensor<32x1x1x8xi32>
    %1903 = stablehlo.compare  GT, %1900, %1902,  SIGNED : (tensor<32x1x1x8xi32>, tensor<32x1x1x8xi32>) -> tensor<32x1x1x8xi1>
    %1904 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1905 = stablehlo.broadcast_in_dim %1904, dims = [] : (tensor<f32>) -> tensor<32x1x1x8xf32>
    %1906 = stablehlo.convert %1905 : (tensor<32x1x1x8xf32>) -> tensor<32x1x1x8xf16>
    %1907 = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
    %1908 = stablehlo.broadcast_in_dim %1907, dims = [] : (tensor<f16>) -> tensor<32x1x1x8xf16>
    %1909 = stablehlo.select %1903, %1906, %1908 : tensor<32x1x1x8xi1>, tensor<32x1x1x8xf16>
    %1910 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %1911 = stablehlo.sqrt %1910 : tensor<f32>
    %1912 = stablehlo.convert %1911 : (tensor<f32>) -> tensor<f16>
    %1913 = stablehlo.broadcast_in_dim %1912, dims = [] : (tensor<f16>) -> tensor<32x8x12x64xf16>
    %1914 = stablehlo.divide %1897, %1913 : tensor<32x8x12x64xf16>
    %1915 = stablehlo.convert %1914 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1916 = stablehlo.convert %1898 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1917 = stablehlo.dot_general %1915, %1916, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<32x8x12x64xf32>, tensor<32x8x12x64xf32>) -> tensor<32x12x8x8xf16>
    %1918 = stablehlo.broadcast_in_dim %1909, dims = [0, 1, 2, 3] : (tensor<32x1x1x8xf16>) -> tensor<32x12x8x8xf16>
    %1919 = stablehlo.add %1917, %1918 : tensor<32x12x8x8xf16>
    %1920 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1921 = stablehlo.reduce(%1919 init: %1920) applies stablehlo.maximum across dimensions = [3] : (tensor<32x12x8x8xf16>, tensor<f16>) -> tensor<32x12x8xf16>
    %1922 = stablehlo.broadcast_in_dim %1921, dims = [0, 1, 2] : (tensor<32x12x8xf16>) -> tensor<32x12x8x1xf16>
    %1923 = stablehlo.broadcast_in_dim %1922, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %1924 = stablehlo.subtract %1919, %1923 : tensor<32x12x8x8xf16>
    %1925 = stablehlo.exponential %1924 : tensor<32x12x8x8xf16>
    %1926 = stablehlo.convert %1925 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1927 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1928 = stablehlo.reduce(%1926 init: %1927) applies stablehlo.add across dimensions = [3] : (tensor<32x12x8x8xf32>, tensor<f32>) -> tensor<32x12x8xf32>
    %1929 = stablehlo.broadcast_in_dim %1928, dims = [0, 1, 2] : (tensor<32x12x8xf32>) -> tensor<32x12x8x1xf32>
    %1930 = stablehlo.convert %1929 : (tensor<32x12x8x1xf32>) -> tensor<32x12x8x1xf16>
    %1931 = stablehlo.broadcast_in_dim %1930, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %1932 = stablehlo.divide %1925, %1931 : tensor<32x12x8x8xf16>
    %1933 = stablehlo.convert %1872 : (tensor<12xi32>) -> tensor<12xf16>
    %1934 = stablehlo.convert %1932 : tensor<32x12x8x8xf16>
    %1935 = stablehlo.convert %1933 : (tensor<12xf16>) -> tensor<12xf32>
    %1936 = stablehlo.convert %1934 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1937 = stablehlo.dot_general %1935, %1936, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<12xf32>, tensor<32x12x8x8xf32>) -> tensor<12x32x8x8xf16>
    %1938 = stablehlo.transpose %1937, dims = [1, 0, 2, 3] : (tensor<12x32x8x8xf16>) -> tensor<32x12x8x8xf16>
    %1939 = stablehlo.convert %1899 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %1940 = stablehlo.convert %1938 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %1941 = stablehlo.dot_general %1939, %1940, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<32x8x12x64xf32>, tensor<32x12x8x8xf32>) -> tensor<32x12x64x8xf16>
    %1942 = stablehlo.transpose %1941, dims = [0, 3, 1, 2] : (tensor<32x12x64x8xf16>) -> tensor<32x8x12x64xf16>
    %1943 = stablehlo.reshape %1942 : (tensor<32x8x12x64xf16>) -> tensor<32x8x768xf16>
    %1944 = stablehlo.convert %155 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %1945 = stablehlo.convert %156 : (tensor<768xf32>) -> tensor<768xf16>
    %1946 = stablehlo.convert %1943 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1947 = stablehlo.convert %1944 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1948 = stablehlo.dot_general %1946, %1947, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %1949 = stablehlo.reshape %1945 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1950 = stablehlo.broadcast_in_dim %1949, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %1951 = stablehlo.add %1948, %1950 : tensor<32x8x768xf16>
    %1952 = stablehlo.add %1951, %1870 : tensor<32x8x768xf16>
    %1953 = stablehlo.convert %1952 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1954 = stablehlo.multiply %1953, %1953 : tensor<32x8x768xf32>
    %1955 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1956 = stablehlo.reduce(%1953 init: %1955) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1957 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1958 = stablehlo.broadcast_in_dim %1957, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1959 = stablehlo.divide %1956, %1958 : tensor<32x8xf32>
    %1960 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1961 = stablehlo.reduce(%1954 init: %1960) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %1962 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %1963 = stablehlo.broadcast_in_dim %1962, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1964 = stablehlo.divide %1961, %1963 : tensor<32x8xf32>
    %1965 = stablehlo.multiply %1959, %1959 : tensor<32x8xf32>
    %1966 = stablehlo.subtract %1964, %1965 : tensor<32x8xf32>
    %1967 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1968 = stablehlo.broadcast_in_dim %1967, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %1969 = stablehlo.maximum %1968, %1966 : tensor<32x8xf32>
    %1970 = stablehlo.broadcast_in_dim %1959, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1971 = stablehlo.broadcast_in_dim %1969, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %1972 = stablehlo.convert %1952 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1973 = stablehlo.broadcast_in_dim %1970, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1974 = stablehlo.subtract %1972, %1973 : tensor<32x8x768xf32>
    %1975 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %1976 = stablehlo.broadcast_in_dim %1975, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %1977 = stablehlo.add %1971, %1976 : tensor<32x8x1xf32>
    %1978 = stablehlo.rsqrt %1977 : tensor<32x8x1xf32>
    %1979 = stablehlo.reshape %157 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1980 = stablehlo.broadcast_in_dim %1978, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %1981 = stablehlo.broadcast_in_dim %1979, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1982 = stablehlo.multiply %1980, %1981 : tensor<32x8x768xf32>
    %1983 = stablehlo.multiply %1974, %1982 : tensor<32x8x768xf32>
    %1984 = stablehlo.reshape %158 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1985 = stablehlo.broadcast_in_dim %1984, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %1986 = stablehlo.add %1983, %1985 : tensor<32x8x768xf32>
    %1987 = stablehlo.convert %1986 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %1988 = stablehlo.convert %159 : (tensor<768x3072xf32>) -> tensor<768x3072xf16>
    %1989 = stablehlo.convert %160 : (tensor<3072xf32>) -> tensor<3072xf16>
    %1990 = stablehlo.convert %1987 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %1991 = stablehlo.convert %1988 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1992 = stablehlo.dot_general %1990, %1991, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x3072xf32>) -> tensor<32x8x3072xf16>
    %1993 = stablehlo.reshape %1989 : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %1994 = stablehlo.broadcast_in_dim %1993, dims = [0, 1, 2] : (tensor<1x1x3072xf16>) -> tensor<32x8x3072xf16>
    %1995 = stablehlo.add %1992, %1994 : tensor<32x8x3072xf16>
    %1996 = stablehlo.constant dense<1.414060e+00> : tensor<f16>
    %1997 = stablehlo.broadcast_in_dim %1996, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %1998 = stablehlo.divide %1995, %1997 : tensor<32x8x3072xf16>
    %1999 = stablehlo.custom_call @mhlo.erf(%1998) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf16>
    %2000 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %2001 = stablehlo.broadcast_in_dim %2000, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %2002 = stablehlo.add %1999, %2001 : tensor<32x8x3072xf16>
    %2003 = stablehlo.multiply %1995, %2002 : tensor<32x8x3072xf16>
    %2004 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %2005 = stablehlo.broadcast_in_dim %2004, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %2006 = stablehlo.divide %2003, %2005 : tensor<32x8x3072xf16>
    %2007 = stablehlo.convert %161 : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
    %2008 = stablehlo.convert %162 : (tensor<768xf32>) -> tensor<768xf16>
    %2009 = stablehlo.convert %2006 : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf32>
    %2010 = stablehlo.convert %2007 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %2011 = stablehlo.dot_general %2009, %2010, contracting_dims = [2] x [0] : (tensor<32x8x3072xf32>, tensor<3072x768xf32>) -> tensor<32x8x768xf16>
    %2012 = stablehlo.reshape %2008 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2013 = stablehlo.broadcast_in_dim %2012, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %2014 = stablehlo.add %2011, %2013 : tensor<32x8x768xf16>
    %2015 = stablehlo.add %2014, %1987 : tensor<32x8x768xf16>
    %2016 = stablehlo.convert %2015 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2017 = stablehlo.multiply %2016, %2016 : tensor<32x8x768xf32>
    %2018 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2019 = stablehlo.reduce(%2016 init: %2018) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %2020 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2021 = stablehlo.broadcast_in_dim %2020, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2022 = stablehlo.divide %2019, %2021 : tensor<32x8xf32>
    %2023 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2024 = stablehlo.reduce(%2017 init: %2023) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %2025 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2026 = stablehlo.broadcast_in_dim %2025, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2027 = stablehlo.divide %2024, %2026 : tensor<32x8xf32>
    %2028 = stablehlo.multiply %2022, %2022 : tensor<32x8xf32>
    %2029 = stablehlo.subtract %2027, %2028 : tensor<32x8xf32>
    %2030 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2031 = stablehlo.broadcast_in_dim %2030, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2032 = stablehlo.maximum %2031, %2029 : tensor<32x8xf32>
    %2033 = stablehlo.broadcast_in_dim %2022, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %2034 = stablehlo.broadcast_in_dim %2032, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %2035 = stablehlo.convert %2015 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2036 = stablehlo.broadcast_in_dim %2033, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %2037 = stablehlo.subtract %2035, %2036 : tensor<32x8x768xf32>
    %2038 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %2039 = stablehlo.broadcast_in_dim %2038, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %2040 = stablehlo.add %2034, %2039 : tensor<32x8x1xf32>
    %2041 = stablehlo.rsqrt %2040 : tensor<32x8x1xf32>
    %2042 = stablehlo.reshape %163 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2043 = stablehlo.broadcast_in_dim %2041, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %2044 = stablehlo.broadcast_in_dim %2042, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %2045 = stablehlo.multiply %2043, %2044 : tensor<32x8x768xf32>
    %2046 = stablehlo.multiply %2037, %2045 : tensor<32x8x768xf32>
    %2047 = stablehlo.reshape %164 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2048 = stablehlo.broadcast_in_dim %2047, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %2049 = stablehlo.add %2046, %2048 : tensor<32x8x768xf32>
    %2050 = stablehlo.convert %2049 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %2051 = stablehlo.slice %207 [10:11, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
    %2052 = stablehlo.reshape %2051 : (tensor<1x12xi32>) -> tensor<12xi32>
    %2053 = stablehlo.convert %165 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %2054 = stablehlo.convert %166 : (tensor<768xf32>) -> tensor<768xf16>
    %2055 = stablehlo.convert %2050 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2056 = stablehlo.convert %2053 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2057 = stablehlo.dot_general %2055, %2056, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %2058 = stablehlo.reshape %2054 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2059 = stablehlo.broadcast_in_dim %2058, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %2060 = stablehlo.add %2057, %2059 : tensor<32x8x768xf16>
    %2061 = stablehlo.convert %167 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %2062 = stablehlo.convert %168 : (tensor<768xf32>) -> tensor<768xf16>
    %2063 = stablehlo.convert %2050 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2064 = stablehlo.convert %2061 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2065 = stablehlo.dot_general %2063, %2064, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %2066 = stablehlo.reshape %2062 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2067 = stablehlo.broadcast_in_dim %2066, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %2068 = stablehlo.add %2065, %2067 : tensor<32x8x768xf16>
    %2069 = stablehlo.convert %169 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %2070 = stablehlo.convert %170 : (tensor<768xf32>) -> tensor<768xf16>
    %2071 = stablehlo.convert %2050 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2072 = stablehlo.convert %2069 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2073 = stablehlo.dot_general %2071, %2072, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %2074 = stablehlo.reshape %2070 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2075 = stablehlo.broadcast_in_dim %2074, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %2076 = stablehlo.add %2073, %2075 : tensor<32x8x768xf16>
    %2077 = stablehlo.reshape %2060 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %2078 = stablehlo.reshape %2068 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %2079 = stablehlo.reshape %2076 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %2080 = stablehlo.broadcast_in_dim %204, dims = [0, 3] : (tensor<32x8xi32>) -> tensor<32x1x1x8xi32>
    %2081 = stablehlo.constant dense<0> : tensor<i32>
    %2082 = stablehlo.broadcast_in_dim %2081, dims = [] : (tensor<i32>) -> tensor<32x1x1x8xi32>
    %2083 = stablehlo.compare  GT, %2080, %2082,  SIGNED : (tensor<32x1x1x8xi32>, tensor<32x1x1x8xi32>) -> tensor<32x1x1x8xi1>
    %2084 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2085 = stablehlo.broadcast_in_dim %2084, dims = [] : (tensor<f32>) -> tensor<32x1x1x8xf32>
    %2086 = stablehlo.convert %2085 : (tensor<32x1x1x8xf32>) -> tensor<32x1x1x8xf16>
    %2087 = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
    %2088 = stablehlo.broadcast_in_dim %2087, dims = [] : (tensor<f16>) -> tensor<32x1x1x8xf16>
    %2089 = stablehlo.select %2083, %2086, %2088 : tensor<32x1x1x8xi1>, tensor<32x1x1x8xf16>
    %2090 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %2091 = stablehlo.sqrt %2090 : tensor<f32>
    %2092 = stablehlo.convert %2091 : (tensor<f32>) -> tensor<f16>
    %2093 = stablehlo.broadcast_in_dim %2092, dims = [] : (tensor<f16>) -> tensor<32x8x12x64xf16>
    %2094 = stablehlo.divide %2077, %2093 : tensor<32x8x12x64xf16>
    %2095 = stablehlo.convert %2094 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %2096 = stablehlo.convert %2078 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %2097 = stablehlo.dot_general %2095, %2096, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<32x8x12x64xf32>, tensor<32x8x12x64xf32>) -> tensor<32x12x8x8xf16>
    %2098 = stablehlo.broadcast_in_dim %2089, dims = [0, 1, 2, 3] : (tensor<32x1x1x8xf16>) -> tensor<32x12x8x8xf16>
    %2099 = stablehlo.add %2097, %2098 : tensor<32x12x8x8xf16>
    %2100 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %2101 = stablehlo.reduce(%2099 init: %2100) applies stablehlo.maximum across dimensions = [3] : (tensor<32x12x8x8xf16>, tensor<f16>) -> tensor<32x12x8xf16>
    %2102 = stablehlo.broadcast_in_dim %2101, dims = [0, 1, 2] : (tensor<32x12x8xf16>) -> tensor<32x12x8x1xf16>
    %2103 = stablehlo.broadcast_in_dim %2102, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %2104 = stablehlo.subtract %2099, %2103 : tensor<32x12x8x8xf16>
    %2105 = stablehlo.exponential %2104 : tensor<32x12x8x8xf16>
    %2106 = stablehlo.convert %2105 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %2107 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2108 = stablehlo.reduce(%2106 init: %2107) applies stablehlo.add across dimensions = [3] : (tensor<32x12x8x8xf32>, tensor<f32>) -> tensor<32x12x8xf32>
    %2109 = stablehlo.broadcast_in_dim %2108, dims = [0, 1, 2] : (tensor<32x12x8xf32>) -> tensor<32x12x8x1xf32>
    %2110 = stablehlo.convert %2109 : (tensor<32x12x8x1xf32>) -> tensor<32x12x8x1xf16>
    %2111 = stablehlo.broadcast_in_dim %2110, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %2112 = stablehlo.divide %2105, %2111 : tensor<32x12x8x8xf16>
    %2113 = stablehlo.convert %2052 : (tensor<12xi32>) -> tensor<12xf16>
    %2114 = stablehlo.convert %2112 : tensor<32x12x8x8xf16>
    %2115 = stablehlo.convert %2113 : (tensor<12xf16>) -> tensor<12xf32>
    %2116 = stablehlo.convert %2114 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %2117 = stablehlo.dot_general %2115, %2116, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<12xf32>, tensor<32x12x8x8xf32>) -> tensor<12x32x8x8xf16>
    %2118 = stablehlo.transpose %2117, dims = [1, 0, 2, 3] : (tensor<12x32x8x8xf16>) -> tensor<32x12x8x8xf16>
    %2119 = stablehlo.convert %2079 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %2120 = stablehlo.convert %2118 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %2121 = stablehlo.dot_general %2119, %2120, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<32x8x12x64xf32>, tensor<32x12x8x8xf32>) -> tensor<32x12x64x8xf16>
    %2122 = stablehlo.transpose %2121, dims = [0, 3, 1, 2] : (tensor<32x12x64x8xf16>) -> tensor<32x8x12x64xf16>
    %2123 = stablehlo.reshape %2122 : (tensor<32x8x12x64xf16>) -> tensor<32x8x768xf16>
    %2124 = stablehlo.convert %171 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %2125 = stablehlo.convert %172 : (tensor<768xf32>) -> tensor<768xf16>
    %2126 = stablehlo.convert %2123 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2127 = stablehlo.convert %2124 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2128 = stablehlo.dot_general %2126, %2127, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %2129 = stablehlo.reshape %2125 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2130 = stablehlo.broadcast_in_dim %2129, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %2131 = stablehlo.add %2128, %2130 : tensor<32x8x768xf16>
    %2132 = stablehlo.add %2131, %2050 : tensor<32x8x768xf16>
    %2133 = stablehlo.convert %2132 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2134 = stablehlo.multiply %2133, %2133 : tensor<32x8x768xf32>
    %2135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2136 = stablehlo.reduce(%2133 init: %2135) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %2137 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2138 = stablehlo.broadcast_in_dim %2137, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2139 = stablehlo.divide %2136, %2138 : tensor<32x8xf32>
    %2140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2141 = stablehlo.reduce(%2134 init: %2140) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %2142 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2143 = stablehlo.broadcast_in_dim %2142, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2144 = stablehlo.divide %2141, %2143 : tensor<32x8xf32>
    %2145 = stablehlo.multiply %2139, %2139 : tensor<32x8xf32>
    %2146 = stablehlo.subtract %2144, %2145 : tensor<32x8xf32>
    %2147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2148 = stablehlo.broadcast_in_dim %2147, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2149 = stablehlo.maximum %2148, %2146 : tensor<32x8xf32>
    %2150 = stablehlo.broadcast_in_dim %2139, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %2151 = stablehlo.broadcast_in_dim %2149, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %2152 = stablehlo.convert %2132 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2153 = stablehlo.broadcast_in_dim %2150, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %2154 = stablehlo.subtract %2152, %2153 : tensor<32x8x768xf32>
    %2155 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %2156 = stablehlo.broadcast_in_dim %2155, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %2157 = stablehlo.add %2151, %2156 : tensor<32x8x1xf32>
    %2158 = stablehlo.rsqrt %2157 : tensor<32x8x1xf32>
    %2159 = stablehlo.reshape %173 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2160 = stablehlo.broadcast_in_dim %2158, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %2161 = stablehlo.broadcast_in_dim %2159, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %2162 = stablehlo.multiply %2160, %2161 : tensor<32x8x768xf32>
    %2163 = stablehlo.multiply %2154, %2162 : tensor<32x8x768xf32>
    %2164 = stablehlo.reshape %174 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2165 = stablehlo.broadcast_in_dim %2164, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %2166 = stablehlo.add %2163, %2165 : tensor<32x8x768xf32>
    %2167 = stablehlo.convert %2166 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %2168 = stablehlo.convert %175 : (tensor<768x3072xf32>) -> tensor<768x3072xf16>
    %2169 = stablehlo.convert %176 : (tensor<3072xf32>) -> tensor<3072xf16>
    %2170 = stablehlo.convert %2167 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2171 = stablehlo.convert %2168 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %2172 = stablehlo.dot_general %2170, %2171, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x3072xf32>) -> tensor<32x8x3072xf16>
    %2173 = stablehlo.reshape %2169 : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %2174 = stablehlo.broadcast_in_dim %2173, dims = [0, 1, 2] : (tensor<1x1x3072xf16>) -> tensor<32x8x3072xf16>
    %2175 = stablehlo.add %2172, %2174 : tensor<32x8x3072xf16>
    %2176 = stablehlo.constant dense<1.414060e+00> : tensor<f16>
    %2177 = stablehlo.broadcast_in_dim %2176, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %2178 = stablehlo.divide %2175, %2177 : tensor<32x8x3072xf16>
    %2179 = stablehlo.custom_call @mhlo.erf(%2178) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf16>
    %2180 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %2181 = stablehlo.broadcast_in_dim %2180, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %2182 = stablehlo.add %2179, %2181 : tensor<32x8x3072xf16>
    %2183 = stablehlo.multiply %2175, %2182 : tensor<32x8x3072xf16>
    %2184 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %2185 = stablehlo.broadcast_in_dim %2184, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %2186 = stablehlo.divide %2183, %2185 : tensor<32x8x3072xf16>
    %2187 = stablehlo.convert %177 : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
    %2188 = stablehlo.convert %178 : (tensor<768xf32>) -> tensor<768xf16>
    %2189 = stablehlo.convert %2186 : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf32>
    %2190 = stablehlo.convert %2187 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %2191 = stablehlo.dot_general %2189, %2190, contracting_dims = [2] x [0] : (tensor<32x8x3072xf32>, tensor<3072x768xf32>) -> tensor<32x8x768xf16>
    %2192 = stablehlo.reshape %2188 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2193 = stablehlo.broadcast_in_dim %2192, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %2194 = stablehlo.add %2191, %2193 : tensor<32x8x768xf16>
    %2195 = stablehlo.add %2194, %2167 : tensor<32x8x768xf16>
    %2196 = stablehlo.convert %2195 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2197 = stablehlo.multiply %2196, %2196 : tensor<32x8x768xf32>
    %2198 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2199 = stablehlo.reduce(%2196 init: %2198) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %2200 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2201 = stablehlo.broadcast_in_dim %2200, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2202 = stablehlo.divide %2199, %2201 : tensor<32x8xf32>
    %2203 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2204 = stablehlo.reduce(%2197 init: %2203) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %2205 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2206 = stablehlo.broadcast_in_dim %2205, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2207 = stablehlo.divide %2204, %2206 : tensor<32x8xf32>
    %2208 = stablehlo.multiply %2202, %2202 : tensor<32x8xf32>
    %2209 = stablehlo.subtract %2207, %2208 : tensor<32x8xf32>
    %2210 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2211 = stablehlo.broadcast_in_dim %2210, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2212 = stablehlo.maximum %2211, %2209 : tensor<32x8xf32>
    %2213 = stablehlo.broadcast_in_dim %2202, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %2214 = stablehlo.broadcast_in_dim %2212, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %2215 = stablehlo.convert %2195 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2216 = stablehlo.broadcast_in_dim %2213, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %2217 = stablehlo.subtract %2215, %2216 : tensor<32x8x768xf32>
    %2218 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %2219 = stablehlo.broadcast_in_dim %2218, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %2220 = stablehlo.add %2214, %2219 : tensor<32x8x1xf32>
    %2221 = stablehlo.rsqrt %2220 : tensor<32x8x1xf32>
    %2222 = stablehlo.reshape %179 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2223 = stablehlo.broadcast_in_dim %2221, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %2224 = stablehlo.broadcast_in_dim %2222, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %2225 = stablehlo.multiply %2223, %2224 : tensor<32x8x768xf32>
    %2226 = stablehlo.multiply %2217, %2225 : tensor<32x8x768xf32>
    %2227 = stablehlo.reshape %180 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2228 = stablehlo.broadcast_in_dim %2227, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %2229 = stablehlo.add %2226, %2228 : tensor<32x8x768xf32>
    %2230 = stablehlo.convert %2229 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %2231 = stablehlo.slice %207 [11:12, 0:12] : (tensor<12x12xi32>) -> tensor<1x12xi32>
    %2232 = stablehlo.reshape %2231 : (tensor<1x12xi32>) -> tensor<12xi32>
    %2233 = stablehlo.convert %181 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %2234 = stablehlo.convert %182 : (tensor<768xf32>) -> tensor<768xf16>
    %2235 = stablehlo.convert %2230 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2236 = stablehlo.convert %2233 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2237 = stablehlo.dot_general %2235, %2236, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %2238 = stablehlo.reshape %2234 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2239 = stablehlo.broadcast_in_dim %2238, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %2240 = stablehlo.add %2237, %2239 : tensor<32x8x768xf16>
    %2241 = stablehlo.convert %183 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %2242 = stablehlo.convert %184 : (tensor<768xf32>) -> tensor<768xf16>
    %2243 = stablehlo.convert %2230 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2244 = stablehlo.convert %2241 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2245 = stablehlo.dot_general %2243, %2244, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %2246 = stablehlo.reshape %2242 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2247 = stablehlo.broadcast_in_dim %2246, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %2248 = stablehlo.add %2245, %2247 : tensor<32x8x768xf16>
    %2249 = stablehlo.convert %185 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %2250 = stablehlo.convert %186 : (tensor<768xf32>) -> tensor<768xf16>
    %2251 = stablehlo.convert %2230 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2252 = stablehlo.convert %2249 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2253 = stablehlo.dot_general %2251, %2252, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %2254 = stablehlo.reshape %2250 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2255 = stablehlo.broadcast_in_dim %2254, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %2256 = stablehlo.add %2253, %2255 : tensor<32x8x768xf16>
    %2257 = stablehlo.reshape %2240 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %2258 = stablehlo.reshape %2248 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %2259 = stablehlo.reshape %2256 : (tensor<32x8x768xf16>) -> tensor<32x8x12x64xf16>
    %2260 = stablehlo.broadcast_in_dim %204, dims = [0, 3] : (tensor<32x8xi32>) -> tensor<32x1x1x8xi32>
    %2261 = stablehlo.constant dense<0> : tensor<i32>
    %2262 = stablehlo.broadcast_in_dim %2261, dims = [] : (tensor<i32>) -> tensor<32x1x1x8xi32>
    %2263 = stablehlo.compare  GT, %2260, %2262,  SIGNED : (tensor<32x1x1x8xi32>, tensor<32x1x1x8xi32>) -> tensor<32x1x1x8xi1>
    %2264 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2265 = stablehlo.broadcast_in_dim %2264, dims = [] : (tensor<f32>) -> tensor<32x1x1x8xf32>
    %2266 = stablehlo.convert %2265 : (tensor<32x1x1x8xf32>) -> tensor<32x1x1x8xf16>
    %2267 = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
    %2268 = stablehlo.broadcast_in_dim %2267, dims = [] : (tensor<f16>) -> tensor<32x1x1x8xf16>
    %2269 = stablehlo.select %2263, %2266, %2268 : tensor<32x1x1x8xi1>, tensor<32x1x1x8xf16>
    %2270 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %2271 = stablehlo.sqrt %2270 : tensor<f32>
    %2272 = stablehlo.convert %2271 : (tensor<f32>) -> tensor<f16>
    %2273 = stablehlo.broadcast_in_dim %2272, dims = [] : (tensor<f16>) -> tensor<32x8x12x64xf16>
    %2274 = stablehlo.divide %2257, %2273 : tensor<32x8x12x64xf16>
    %2275 = stablehlo.convert %2274 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %2276 = stablehlo.convert %2258 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %2277 = stablehlo.dot_general %2275, %2276, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<32x8x12x64xf32>, tensor<32x8x12x64xf32>) -> tensor<32x12x8x8xf16>
    %2278 = stablehlo.broadcast_in_dim %2269, dims = [0, 1, 2, 3] : (tensor<32x1x1x8xf16>) -> tensor<32x12x8x8xf16>
    %2279 = stablehlo.add %2277, %2278 : tensor<32x12x8x8xf16>
    %2280 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %2281 = stablehlo.reduce(%2279 init: %2280) applies stablehlo.maximum across dimensions = [3] : (tensor<32x12x8x8xf16>, tensor<f16>) -> tensor<32x12x8xf16>
    %2282 = stablehlo.broadcast_in_dim %2281, dims = [0, 1, 2] : (tensor<32x12x8xf16>) -> tensor<32x12x8x1xf16>
    %2283 = stablehlo.broadcast_in_dim %2282, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %2284 = stablehlo.subtract %2279, %2283 : tensor<32x12x8x8xf16>
    %2285 = stablehlo.exponential %2284 : tensor<32x12x8x8xf16>
    %2286 = stablehlo.convert %2285 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %2287 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2288 = stablehlo.reduce(%2286 init: %2287) applies stablehlo.add across dimensions = [3] : (tensor<32x12x8x8xf32>, tensor<f32>) -> tensor<32x12x8xf32>
    %2289 = stablehlo.broadcast_in_dim %2288, dims = [0, 1, 2] : (tensor<32x12x8xf32>) -> tensor<32x12x8x1xf32>
    %2290 = stablehlo.convert %2289 : (tensor<32x12x8x1xf32>) -> tensor<32x12x8x1xf16>
    %2291 = stablehlo.broadcast_in_dim %2290, dims = [0, 1, 2, 3] : (tensor<32x12x8x1xf16>) -> tensor<32x12x8x8xf16>
    %2292 = stablehlo.divide %2285, %2291 : tensor<32x12x8x8xf16>
    %2293 = stablehlo.convert %2232 : (tensor<12xi32>) -> tensor<12xf16>
    %2294 = stablehlo.convert %2292 : tensor<32x12x8x8xf16>
    %2295 = stablehlo.convert %2293 : (tensor<12xf16>) -> tensor<12xf32>
    %2296 = stablehlo.convert %2294 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %2297 = stablehlo.dot_general %2295, %2296, batching_dims = [0] x [1], contracting_dims = [] x [] : (tensor<12xf32>, tensor<32x12x8x8xf32>) -> tensor<12x32x8x8xf16>
    %2298 = stablehlo.transpose %2297, dims = [1, 0, 2, 3] : (tensor<12x32x8x8xf16>) -> tensor<32x12x8x8xf16>
    %2299 = stablehlo.convert %2259 : (tensor<32x8x12x64xf16>) -> tensor<32x8x12x64xf32>
    %2300 = stablehlo.convert %2298 : (tensor<32x12x8x8xf16>) -> tensor<32x12x8x8xf32>
    %2301 = stablehlo.dot_general %2299, %2300, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<32x8x12x64xf32>, tensor<32x12x8x8xf32>) -> tensor<32x12x64x8xf16>
    %2302 = stablehlo.transpose %2301, dims = [0, 3, 1, 2] : (tensor<32x12x64x8xf16>) -> tensor<32x8x12x64xf16>
    %2303 = stablehlo.reshape %2302 : (tensor<32x8x12x64xf16>) -> tensor<32x8x768xf16>
    %2304 = stablehlo.convert %187 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %2305 = stablehlo.convert %188 : (tensor<768xf32>) -> tensor<768xf16>
    %2306 = stablehlo.convert %2303 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2307 = stablehlo.convert %2304 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2308 = stablehlo.dot_general %2306, %2307, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x768xf32>) -> tensor<32x8x768xf16>
    %2309 = stablehlo.reshape %2305 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2310 = stablehlo.broadcast_in_dim %2309, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %2311 = stablehlo.add %2308, %2310 : tensor<32x8x768xf16>
    %2312 = stablehlo.add %2311, %2230 : tensor<32x8x768xf16>
    %2313 = stablehlo.convert %2312 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2314 = stablehlo.multiply %2313, %2313 : tensor<32x8x768xf32>
    %2315 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2316 = stablehlo.reduce(%2313 init: %2315) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %2317 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2318 = stablehlo.broadcast_in_dim %2317, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2319 = stablehlo.divide %2316, %2318 : tensor<32x8xf32>
    %2320 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2321 = stablehlo.reduce(%2314 init: %2320) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %2322 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2323 = stablehlo.broadcast_in_dim %2322, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2324 = stablehlo.divide %2321, %2323 : tensor<32x8xf32>
    %2325 = stablehlo.multiply %2319, %2319 : tensor<32x8xf32>
    %2326 = stablehlo.subtract %2324, %2325 : tensor<32x8xf32>
    %2327 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2328 = stablehlo.broadcast_in_dim %2327, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2329 = stablehlo.maximum %2328, %2326 : tensor<32x8xf32>
    %2330 = stablehlo.broadcast_in_dim %2319, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %2331 = stablehlo.broadcast_in_dim %2329, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %2332 = stablehlo.convert %2312 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2333 = stablehlo.broadcast_in_dim %2330, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %2334 = stablehlo.subtract %2332, %2333 : tensor<32x8x768xf32>
    %2335 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %2336 = stablehlo.broadcast_in_dim %2335, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %2337 = stablehlo.add %2331, %2336 : tensor<32x8x1xf32>
    %2338 = stablehlo.rsqrt %2337 : tensor<32x8x1xf32>
    %2339 = stablehlo.reshape %189 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2340 = stablehlo.broadcast_in_dim %2338, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %2341 = stablehlo.broadcast_in_dim %2339, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %2342 = stablehlo.multiply %2340, %2341 : tensor<32x8x768xf32>
    %2343 = stablehlo.multiply %2334, %2342 : tensor<32x8x768xf32>
    %2344 = stablehlo.reshape %190 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2345 = stablehlo.broadcast_in_dim %2344, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %2346 = stablehlo.add %2343, %2345 : tensor<32x8x768xf32>
    %2347 = stablehlo.convert %2346 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %2348 = stablehlo.convert %191 : (tensor<768x3072xf32>) -> tensor<768x3072xf16>
    %2349 = stablehlo.convert %192 : (tensor<3072xf32>) -> tensor<3072xf16>
    %2350 = stablehlo.convert %2347 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2351 = stablehlo.convert %2348 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %2352 = stablehlo.dot_general %2350, %2351, contracting_dims = [2] x [0] : (tensor<32x8x768xf32>, tensor<768x3072xf32>) -> tensor<32x8x3072xf16>
    %2353 = stablehlo.reshape %2349 : (tensor<3072xf16>) -> tensor<1x1x3072xf16>
    %2354 = stablehlo.broadcast_in_dim %2353, dims = [0, 1, 2] : (tensor<1x1x3072xf16>) -> tensor<32x8x3072xf16>
    %2355 = stablehlo.add %2352, %2354 : tensor<32x8x3072xf16>
    %2356 = stablehlo.constant dense<1.414060e+00> : tensor<f16>
    %2357 = stablehlo.broadcast_in_dim %2356, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %2358 = stablehlo.divide %2355, %2357 : tensor<32x8x3072xf16>
    %2359 = stablehlo.custom_call @mhlo.erf(%2358) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf16>
    %2360 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %2361 = stablehlo.broadcast_in_dim %2360, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %2362 = stablehlo.add %2359, %2361 : tensor<32x8x3072xf16>
    %2363 = stablehlo.multiply %2355, %2362 : tensor<32x8x3072xf16>
    %2364 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %2365 = stablehlo.broadcast_in_dim %2364, dims = [] : (tensor<f16>) -> tensor<32x8x3072xf16>
    %2366 = stablehlo.divide %2363, %2365 : tensor<32x8x3072xf16>
    %2367 = stablehlo.convert %193 : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
    %2368 = stablehlo.convert %194 : (tensor<768xf32>) -> tensor<768xf16>
    %2369 = stablehlo.convert %2366 : (tensor<32x8x3072xf16>) -> tensor<32x8x3072xf32>
    %2370 = stablehlo.convert %2367 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %2371 = stablehlo.dot_general %2369, %2370, contracting_dims = [2] x [0] : (tensor<32x8x3072xf32>, tensor<3072x768xf32>) -> tensor<32x8x768xf16>
    %2372 = stablehlo.reshape %2368 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2373 = stablehlo.broadcast_in_dim %2372, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<32x8x768xf16>
    %2374 = stablehlo.add %2371, %2373 : tensor<32x8x768xf16>
    %2375 = stablehlo.add %2374, %2347 : tensor<32x8x768xf16>
    %2376 = stablehlo.convert %2375 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2377 = stablehlo.multiply %2376, %2376 : tensor<32x8x768xf32>
    %2378 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2379 = stablehlo.reduce(%2376 init: %2378) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %2380 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2381 = stablehlo.broadcast_in_dim %2380, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2382 = stablehlo.divide %2379, %2381 : tensor<32x8xf32>
    %2383 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2384 = stablehlo.reduce(%2377 init: %2383) applies stablehlo.add across dimensions = [2] : (tensor<32x8x768xf32>, tensor<f32>) -> tensor<32x8xf32>
    %2385 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %2386 = stablehlo.broadcast_in_dim %2385, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2387 = stablehlo.divide %2384, %2386 : tensor<32x8xf32>
    %2388 = stablehlo.multiply %2382, %2382 : tensor<32x8xf32>
    %2389 = stablehlo.subtract %2387, %2388 : tensor<32x8xf32>
    %2390 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2391 = stablehlo.broadcast_in_dim %2390, dims = [] : (tensor<f32>) -> tensor<32x8xf32>
    %2392 = stablehlo.maximum %2391, %2389 : tensor<32x8xf32>
    %2393 = stablehlo.broadcast_in_dim %2382, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %2394 = stablehlo.broadcast_in_dim %2392, dims = [0, 1] : (tensor<32x8xf32>) -> tensor<32x8x1xf32>
    %2395 = stablehlo.convert %2375 : (tensor<32x8x768xf16>) -> tensor<32x8x768xf32>
    %2396 = stablehlo.broadcast_in_dim %2393, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %2397 = stablehlo.subtract %2395, %2396 : tensor<32x8x768xf32>
    %2398 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %2399 = stablehlo.broadcast_in_dim %2398, dims = [] : (tensor<f32>) -> tensor<32x8x1xf32>
    %2400 = stablehlo.add %2394, %2399 : tensor<32x8x1xf32>
    %2401 = stablehlo.rsqrt %2400 : tensor<32x8x1xf32>
    %2402 = stablehlo.reshape %195 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2403 = stablehlo.broadcast_in_dim %2401, dims = [0, 1, 2] : (tensor<32x8x1xf32>) -> tensor<32x8x768xf32>
    %2404 = stablehlo.broadcast_in_dim %2402, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %2405 = stablehlo.multiply %2403, %2404 : tensor<32x8x768xf32>
    %2406 = stablehlo.multiply %2397, %2405 : tensor<32x8x768xf32>
    %2407 = stablehlo.reshape %196 : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2408 = stablehlo.broadcast_in_dim %2407, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<32x8x768xf32>
    %2409 = stablehlo.add %2406, %2408 : tensor<32x8x768xf32>
    %2410 = stablehlo.convert %2409 : (tensor<32x8x768xf32>) -> tensor<32x8x768xf16>
    %2411 = stablehlo.slice %2410 [0:32, 0:1, 0:768] : (tensor<32x8x768xf16>) -> tensor<32x1x768xf16>
    %2412 = stablehlo.reshape %2411 : (tensor<32x1x768xf16>) -> tensor<32x768xf16>
    %2413 = stablehlo.convert %197 : (tensor<768x768xf32>) -> tensor<768x768xf16>
    %2414 = stablehlo.convert %198 : (tensor<768xf32>) -> tensor<768xf16>
    %2415 = stablehlo.convert %2412 : (tensor<32x768xf16>) -> tensor<32x768xf32>
    %2416 = stablehlo.convert %2413 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2417 = stablehlo.dot_general %2415, %2416, contracting_dims = [1] x [0] : (tensor<32x768xf32>, tensor<768x768xf32>) -> tensor<32x768xf16>
    %2418 = stablehlo.reshape %2414 : (tensor<768xf16>) -> tensor<1x768xf16>
    %2419 = stablehlo.broadcast_in_dim %2418, dims = [0, 1] : (tensor<1x768xf16>) -> tensor<32x768xf16>
    %2420 = stablehlo.add %2417, %2419 : tensor<32x768xf16>
    %2421 = stablehlo.tanh %2420 : tensor<32x768xf16>
    return %2410, %2421 : tensor<32x8x768xf16>, tensor<32x768xf16>
  }
  func.func private @_take(%arg0: tensor<30522x768xf16>, %arg1: tensor<32x8xi32>) -> tensor<32x8x768xf16> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<32x8xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<32x8xi32>, tensor<32x8xi32>) -> tensor<32x8xi1>
    %3 = stablehlo.constant dense<30522> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<32x8xi32>
    %5 = stablehlo.add %arg1, %4 : tensor<32x8xi32>
    %6 = call @_where(%2, %5, %arg1) : (tensor<32x8xi1>, tensor<32x8xi32>, tensor<32x8xi32>) -> tensor<32x8xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<32x8xi32>) -> tensor<32x8x1xi32>
    %8 = stablehlo.constant dense<0> : tensor<1xi32>
    %9 = stablehlo.constant dense<0> : tensor<1xi32>
    %10 = stablehlo.constant dense<30522> : tensor<i32>
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
    %23 = "stablehlo.gather"(%14, %22) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
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
    %37 = "stablehlo.gather"(%28, %36) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %38 = stablehlo.subtract %23, %37 : tensor<1xi32>
    %39 = stablehlo.constant dense<0> : tensor<i32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<32x8x1xi32>
    %41 = stablehlo.compare  GE, %7, %40,  SIGNED : (tensor<32x8x1xi32>, tensor<32x8x1xi32>) -> tensor<32x8x1xi1>
    %42 = stablehlo.broadcast_in_dim %38, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<32x8x1xi32>
    %44 = stablehlo.compare  LE, %7, %43,  SIGNED : (tensor<32x8x1xi32>, tensor<32x8x1xi32>) -> tensor<32x8x1xi1>
    %45 = stablehlo.and %41, %44 : tensor<32x8x1xi1>
    %46 = stablehlo.constant dense<true> : tensor<i1>
    %47 = stablehlo.reduce(%45 init: %46) applies stablehlo.and across dimensions = [2] : (tensor<32x8x1xi1>, tensor<i1>) -> tensor<32x8xi1>
    %48 = "stablehlo.gather"(%arg0, %7) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 768>} : (tensor<30522x768xf16>, tensor<32x8x1xi32>) -> tensor<32x8x768xf16>
    %49 = stablehlo.broadcast_in_dim %47, dims = [0, 1] : (tensor<32x8xi1>) -> tensor<32x8x768xi1>
    %50 = stablehlo.constant dense<0x7E00> : tensor<f16>
    %51 = stablehlo.broadcast_in_dim %50, dims = [] : (tensor<f16>) -> tensor<32x8x768xf16>
    %52 = stablehlo.select %49, %48, %51 : tensor<32x8x768xi1>, tensor<32x8x768xf16>
    return %52 : tensor<32x8x768xf16>
  }
  func.func private @_where(%arg0: tensor<32x8xi1>, %arg1: tensor<32x8xi32>, %arg2: tensor<32x8xi32>) -> tensor<32x8xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<32x8xi1>, tensor<32x8xi32>
    return %0 : tensor<32x8xi32>
  }
  func.func private @_take_0(%arg0: tensor<512x768xf16>, %arg1: tensor<32x8xi32>) -> tensor<32x8x768xf16> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<32x8xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<32x8xi32>, tensor<32x8xi32>) -> tensor<32x8xi1>
    %3 = stablehlo.constant dense<512> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<32x8xi32>
    %5 = stablehlo.add %arg1, %4 : tensor<32x8xi32>
    %6 = call @_where(%2, %5, %arg1) : (tensor<32x8xi1>, tensor<32x8xi32>, tensor<32x8xi32>) -> tensor<32x8xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<32x8xi32>) -> tensor<32x8x1xi32>
    %8 = stablehlo.constant dense<0> : tensor<1xi32>
    %9 = stablehlo.constant dense<0> : tensor<1xi32>
    %10 = stablehlo.constant dense<512> : tensor<i32>
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
    %23 = "stablehlo.gather"(%14, %22) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
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
    %37 = "stablehlo.gather"(%28, %36) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %38 = stablehlo.subtract %23, %37 : tensor<1xi32>
    %39 = stablehlo.constant dense<0> : tensor<i32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<32x8x1xi32>
    %41 = stablehlo.compare  GE, %7, %40,  SIGNED : (tensor<32x8x1xi32>, tensor<32x8x1xi32>) -> tensor<32x8x1xi1>
    %42 = stablehlo.broadcast_in_dim %38, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<32x8x1xi32>
    %44 = stablehlo.compare  LE, %7, %43,  SIGNED : (tensor<32x8x1xi32>, tensor<32x8x1xi32>) -> tensor<32x8x1xi1>
    %45 = stablehlo.and %41, %44 : tensor<32x8x1xi1>
    %46 = stablehlo.constant dense<true> : tensor<i1>
    %47 = stablehlo.reduce(%45 init: %46) applies stablehlo.and across dimensions = [2] : (tensor<32x8x1xi1>, tensor<i1>) -> tensor<32x8xi1>
    %48 = "stablehlo.gather"(%arg0, %7) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 768>} : (tensor<512x768xf16>, tensor<32x8x1xi32>) -> tensor<32x8x768xf16>
    %49 = stablehlo.broadcast_in_dim %47, dims = [0, 1] : (tensor<32x8xi1>) -> tensor<32x8x768xi1>
    %50 = stablehlo.constant dense<0x7E00> : tensor<f16>
    %51 = stablehlo.broadcast_in_dim %50, dims = [] : (tensor<f16>) -> tensor<32x8x768xf16>
    %52 = stablehlo.select %49, %48, %51 : tensor<32x8x768xi1>, tensor<32x8x768xf16>
    return %52 : tensor<32x8x768xf16>
  }
  func.func private @_take_1(%arg0: tensor<2x768xf16>, %arg1: tensor<32x8xi32>) -> tensor<32x8x768xf16> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<32x8xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<32x8xi32>, tensor<32x8xi32>) -> tensor<32x8xi1>
    %3 = stablehlo.constant dense<2> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<32x8xi32>
    %5 = stablehlo.add %arg1, %4 : tensor<32x8xi32>
    %6 = call @_where(%2, %5, %arg1) : (tensor<32x8xi1>, tensor<32x8xi32>, tensor<32x8xi32>) -> tensor<32x8xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<32x8xi32>) -> tensor<32x8x1xi32>
    %8 = stablehlo.constant dense<0> : tensor<1xi32>
    %9 = stablehlo.constant dense<0> : tensor<1xi32>
    %10 = stablehlo.constant dense<2> : tensor<i32>
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
    %23 = "stablehlo.gather"(%14, %22) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
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
    %37 = "stablehlo.gather"(%28, %36) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %38 = stablehlo.subtract %23, %37 : tensor<1xi32>
    %39 = stablehlo.constant dense<0> : tensor<i32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<32x8x1xi32>
    %41 = stablehlo.compare  GE, %7, %40,  SIGNED : (tensor<32x8x1xi32>, tensor<32x8x1xi32>) -> tensor<32x8x1xi1>
    %42 = stablehlo.broadcast_in_dim %38, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<32x8x1xi32>
    %44 = stablehlo.compare  LE, %7, %43,  SIGNED : (tensor<32x8x1xi32>, tensor<32x8x1xi32>) -> tensor<32x8x1xi1>
    %45 = stablehlo.and %41, %44 : tensor<32x8x1xi1>
    %46 = stablehlo.constant dense<true> : tensor<i1>
    %47 = stablehlo.reduce(%45 init: %46) applies stablehlo.and across dimensions = [2] : (tensor<32x8x1xi1>, tensor<i1>) -> tensor<32x8xi1>
    %48 = "stablehlo.gather"(%arg0, %7) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 768>} : (tensor<2x768xf16>, tensor<32x8x1xi32>) -> tensor<32x8x768xf16>
    %49 = stablehlo.broadcast_in_dim %47, dims = [0, 1] : (tensor<32x8xi1>) -> tensor<32x8x768xi1>
    %50 = stablehlo.constant dense<0x7E00> : tensor<f16>
    %51 = stablehlo.broadcast_in_dim %50, dims = [] : (tensor<f16>) -> tensor<32x8x768xf16>
    %52 = stablehlo.select %49, %48, %51 : tensor<32x8x768xi1>, tensor<32x8x768xf16>
    return %52 : tensor<32x8x768xf16>
  }
}

