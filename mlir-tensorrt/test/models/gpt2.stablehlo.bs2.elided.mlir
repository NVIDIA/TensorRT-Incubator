module @gpt2_bs2 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x6xi32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<2x6xi32> {mhlo.sharding = "{replicated}"}) -> (tensor<2x20xi32> {jax.result_info = ""}) {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1 = stablehlo.constant dense<768> : tensor<i32>
    %2 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %3 = stablehlo.constant dense<true> : tensor<i1>
    %4 = stablehlo.constant dense<50257> : tensor<i32>
    %5 = stablehlo.constant dense<-1> : tensor<i32>
    %6 = stablehlo.constant dense<2> : tensor<i32>
    %7 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %10 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %11 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %12 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %13 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %14 = stablehlo.constant dense<6> : tensor<i32>
    %15 = stablehlo.constant dense<20> : tensor<i32>
    %16 = stablehlo.constant dense<1024> : tensor<i32>
    %17 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %19 = stablehlo.constant dense<1> : tensor<i32>
    %20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.constant dense<false> : tensor<i1>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.constant dense_resource<__elided__> : tensor<50257x768xf16>
    %24 = stablehlo.constant dense_resource<__elided__> : tensor<1024x768xf16>
    %25 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %26 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %27 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %28 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %29 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %30 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %31 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %32 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %33 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %34 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %35 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %36 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %37 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %38 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %39 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %40 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %41 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %42 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %43 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %44 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %45 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %46 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %47 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %48 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %49 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %50 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %51 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %52 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %53 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %54 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %55 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %56 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %57 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %58 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %59 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %60 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %61 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %62 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %63 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %64 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %65 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %66 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %67 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %68 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %69 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %70 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %71 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %72 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %73 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %74 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %75 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %76 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %77 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %78 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %79 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %80 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %81 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %82 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %83 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %84 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %85 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %86 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %87 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %88 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %89 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %90 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %91 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %92 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %93 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %94 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %95 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %96 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %97 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %98 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %99 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %100 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %101 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %102 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %103 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %104 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %105 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %106 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %107 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %108 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %109 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %110 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %111 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %112 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %113 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %114 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %115 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %116 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %117 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %118 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %119 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %120 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %121 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %122 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %123 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %124 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %125 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %126 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %127 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %128 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %129 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %130 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %131 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %132 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %133 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %134 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %135 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %136 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %137 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %138 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %139 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %140 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %141 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %142 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %143 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %144 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %145 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %146 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %147 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %148 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %149 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %150 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %151 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %152 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %153 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %154 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %155 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %156 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %157 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %158 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %159 = stablehlo.constant dense_resource<__elided__> : tensor<2304x768xf16>
    %160 = stablehlo.constant dense_resource<__elided__> : tensor<2304xf16>
    %161 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf16>
    %162 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %163 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %164 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %165 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf16>
    %166 = stablehlo.constant dense_resource<__elided__> : tensor<3072xf16>
    %167 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf16>
    %168 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %169 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %170 = stablehlo.constant dense_resource<__elided__> : tensor<768xf16>
    %171 = stablehlo.constant dense<50256> : tensor<i32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [] : (tensor<i32>) -> tensor<2x20xi32>
    %173 = stablehlo.dynamic_update_slice %172, %arg1, %22, %22 : (tensor<2x20xi32>, tensor<2x6xi32>, tensor<i32>, tensor<i32>) -> tensor<2x20xi32>
    %174 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i1>) -> tensor<2xi1>
    %175 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %176 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %177 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %178 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %179 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %180 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %181 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %182 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %183 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %184 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %185 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %186 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %187 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %188 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %189 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %190 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %191 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %192 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %193 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %194 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %195 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %196 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %197 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %198 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x20x12x64xf32>
    %199 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i32>) -> tensor<2x20xi32>
    %200 = stablehlo.slice %arg0 [0:2, 0:5:2] : (tensor<2x6xi32>) -> tensor<2x3xi32>
    %201 = stablehlo.slice %arg0 [0:2, 1:6:2] : (tensor<2x6xi32>) -> tensor<2x3xi32>
    %202 = stablehlo.add %200, %201 : tensor<2x3xi32>
    %203 = stablehlo.slice %202 [0:2, 0:2:2] : (tensor<2x3xi32>) -> tensor<2x1xi32>
    %204 = stablehlo.slice %202 [0:2, 1:3:2] : (tensor<2x3xi32>) -> tensor<2x1xi32>
    %205 = stablehlo.add %203, %204 : tensor<2x1xi32>
    %206 = stablehlo.slice %202 [0:2, 2:3:2] : (tensor<2x3xi32>) -> tensor<2x1xi32>
    %207 = stablehlo.add %205, %206 : tensor<2x1xi32>
    %208 = stablehlo.slice %202 [0:2, 0:1] : (tensor<2x3xi32>) -> tensor<2x1xi32>
    %209 = stablehlo.concatenate %208, %207, dim = 1 : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x2xi32>
    %210 = stablehlo.pad %209, %22, low = [0, 0], high = [0, 0], interior = [0, 1] : (tensor<2x2xi32>, tensor<i32>) -> tensor<2x3xi32>
    %211 = stablehlo.pad %205, %22, low = [0, 1], high = [0, 1], interior = [0, 1] : (tensor<2x1xi32>, tensor<i32>) -> tensor<2x3xi32>
    %212 = stablehlo.add %210, %211 : tensor<2x3xi32>
    %213 = stablehlo.slice %212 [0:2, 0:2] : (tensor<2x3xi32>) -> tensor<2x2xi32>
    %214 = stablehlo.slice %arg0 [0:2, 2:6:2] : (tensor<2x6xi32>) -> tensor<2x2xi32>
    %215 = stablehlo.add %213, %214 : tensor<2x2xi32>
    %216 = stablehlo.slice %arg0 [0:2, 0:1] : (tensor<2x6xi32>) -> tensor<2x1xi32>
    %217 = stablehlo.concatenate %216, %215, dim = 1 : (tensor<2x1xi32>, tensor<2x2xi32>) -> tensor<2x3xi32>
    %218 = stablehlo.pad %217, %22, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2x6xi32>
    %219 = stablehlo.pad %212, %22, low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2x6xi32>
    %220 = stablehlo.add %218, %219 : tensor<2x6xi32>
    %221 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i32>) -> tensor<2x6xi32>
    %222 = stablehlo.subtract %220, %221 : tensor<2x6xi32>
    %223 = stablehlo.dynamic_update_slice %199, %arg0, %22, %22 : (tensor<2x20xi32>, tensor<2x6xi32>, tensor<i32>, tensor<i32>) -> tensor<2x20xi32>
    %224 = stablehlo.convert %23 : (tensor<50257x768xf16>) -> tensor<50257x768xf32>
    %225 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x6xi32>
    %226 = stablehlo.compare  LT, %arg1, %225,  SIGNED : (tensor<2x6xi32>, tensor<2x6xi32>) -> tensor<2x6xi1>
    %227 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<2x6xi32>
    %228 = stablehlo.add %arg1, %227 : tensor<2x6xi32>
    %229 = stablehlo.select %226, %228, %arg1 : tensor<2x6xi1>, tensor<2x6xi32>
    %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1] : (tensor<2x6xi32>) -> tensor<2x6x1xi32>
    %231 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %232 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %233 = stablehlo.concatenate %231, %232, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %234 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %235 = stablehlo.compare  LT, %0, %234,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %236 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %237 = stablehlo.add %0, %236 : tensor<1xi32>
    %238 = stablehlo.select %235, %237, %0 : tensor<1xi1>, tensor<1xi32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %240 = "stablehlo.gather"(%233, %239) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %241 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %242 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %243 = stablehlo.concatenate %241, %242, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %244 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %245 = stablehlo.compare  LT, %0, %244,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %246 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %247 = stablehlo.add %0, %246 : tensor<1xi32>
    %248 = stablehlo.select %245, %247, %0 : tensor<1xi1>, tensor<1xi32>
    %249 = stablehlo.broadcast_in_dim %248, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %250 = "stablehlo.gather"(%243, %249) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %251 = stablehlo.subtract %240, %250 : tensor<1xi32>
    %252 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x6x1xi32>
    %253 = stablehlo.compare  GE, %230, %252,  SIGNED : (tensor<2x6x1xi32>, tensor<2x6x1xi32>) -> tensor<2x6x1xi1>
    %254 = stablehlo.broadcast_in_dim %251, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<2x6x1xi32>
    %256 = stablehlo.compare  LE, %230, %255,  SIGNED : (tensor<2x6x1xi32>, tensor<2x6x1xi32>) -> tensor<2x6x1xi1>
    %257 = stablehlo.and %253, %256 : tensor<2x6x1xi1>
    %258 = stablehlo.reduce(%257 init: %3) across dimensions = [2] : (tensor<2x6x1xi1>, tensor<i1>) -> tensor<2x6xi1>
     reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
      %2405 = stablehlo.and %arg2, %arg3 : tensor<i1>
      stablehlo.return %2405 : tensor<i1>
    }
    %259 = "stablehlo.gather"(%224, %230) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>} : (tensor<50257x768xf32>, tensor<2x6x1xi32>) -> tensor<2x6x768xf32>
    %260 = stablehlo.broadcast_in_dim %258, dims = [0, 1] : (tensor<2x6xi1>) -> tensor<2x6x768xi1>
    %261 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x6x768xf32>
    %262 = stablehlo.select %260, %259, %261 : tensor<2x6x768xi1>, tensor<2x6x768xf32>
    %263 = stablehlo.convert %24 : (tensor<1024x768xf16>) -> tensor<1024x768xf32>
    %264 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x6xi32>
    %265 = stablehlo.compare  LT, %222, %264,  SIGNED : (tensor<2x6xi32>, tensor<2x6xi32>) -> tensor<2x6xi1>
    %266 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i32>) -> tensor<2x6xi32>
    %267 = stablehlo.add %222, %266 : tensor<2x6xi32>
    %268 = stablehlo.select %265, %267, %222 : tensor<2x6xi1>, tensor<2x6xi32>
    %269 = stablehlo.broadcast_in_dim %268, dims = [0, 1] : (tensor<2x6xi32>) -> tensor<2x6x1xi32>
    %270 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %271 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %272 = stablehlo.concatenate %270, %271, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %273 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %274 = stablehlo.compare  LT, %0, %273,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %275 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %276 = stablehlo.add %0, %275 : tensor<1xi32>
    %277 = stablehlo.select %274, %276, %0 : tensor<1xi1>, tensor<1xi32>
    %278 = stablehlo.broadcast_in_dim %277, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %279 = "stablehlo.gather"(%272, %278) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %280 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %281 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %282 = stablehlo.concatenate %280, %281, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %283 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %284 = stablehlo.compare  LT, %0, %283,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %285 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %286 = stablehlo.add %0, %285 : tensor<1xi32>
    %287 = stablehlo.select %284, %286, %0 : tensor<1xi1>, tensor<1xi32>
    %288 = stablehlo.broadcast_in_dim %287, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %289 = "stablehlo.gather"(%282, %288) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %290 = stablehlo.subtract %279, %289 : tensor<1xi32>
    %291 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x6x1xi32>
    %292 = stablehlo.compare  GE, %269, %291,  SIGNED : (tensor<2x6x1xi32>, tensor<2x6x1xi32>) -> tensor<2x6x1xi1>
    %293 = stablehlo.broadcast_in_dim %290, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<2x6x1xi32>
    %295 = stablehlo.compare  LE, %269, %294,  SIGNED : (tensor<2x6x1xi32>, tensor<2x6x1xi32>) -> tensor<2x6x1xi1>
    %296 = stablehlo.and %292, %295 : tensor<2x6x1xi1>
    %297 = stablehlo.reduce(%296 init: %3) across dimensions = [2] : (tensor<2x6x1xi1>, tensor<i1>) -> tensor<2x6xi1>
     reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
      %2405 = stablehlo.and %arg2, %arg3 : tensor<i1>
      stablehlo.return %2405 : tensor<i1>
    }
    %298 = "stablehlo.gather"(%263, %269) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>} : (tensor<1024x768xf32>, tensor<2x6x1xi32>) -> tensor<2x6x768xf32>
    %299 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<2x6xi1>) -> tensor<2x6x768xi1>
    %300 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x6x768xf32>
    %301 = stablehlo.select %299, %298, %300 : tensor<2x6x768xi1>, tensor<2x6x768xf32>
    %302 = stablehlo.add %262, %301 : tensor<2x6x768xf32>
    %303 = stablehlo.multiply %302, %302 : tensor<2x6x768xf32>
    %304 = stablehlo.reduce(%302 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %305 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %306 = stablehlo.divide %304, %305 : tensor<2x6xf32>
    %307 = stablehlo.reduce(%303 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %308 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %309 = stablehlo.divide %307, %308 : tensor<2x6xf32>
    %310 = stablehlo.multiply %306, %306 : tensor<2x6xf32>
    %311 = stablehlo.subtract %309, %310 : tensor<2x6xf32>
    %312 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %313 = stablehlo.maximum %312, %311 : tensor<2x6xf32>
    %314 = stablehlo.broadcast_in_dim %306, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %315 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %316 = stablehlo.broadcast_in_dim %314, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %317 = stablehlo.subtract %302, %316 : tensor<2x6x768xf32>
    %318 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %319 = stablehlo.add %315, %318 : tensor<2x6x1xf32>
    %320 = stablehlo.rsqrt %319 : tensor<2x6x1xf32>
    %321 = stablehlo.reshape %25 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %322 = stablehlo.convert %321 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %323 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %324 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %325 = stablehlo.multiply %323, %324 : tensor<2x6x768xf32>
    %326 = stablehlo.multiply %317, %325 : tensor<2x6x768xf32>
    %327 = stablehlo.reshape %26 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %328 = stablehlo.convert %327 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %329 = stablehlo.broadcast_in_dim %328, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %330 = stablehlo.add %326, %329 : tensor<2x6x768xf32>
    %331 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %332 = stablehlo.broadcast_in_dim %331, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %333 = stablehlo.broadcast_in_dim %332, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %334 = stablehlo.broadcast_in_dim %332, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %335 = stablehlo.broadcast_in_dim %333, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %336 = stablehlo.broadcast_in_dim %334, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %337 = stablehlo.compare  GE, %335, %336,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %338 = stablehlo.broadcast_in_dim %337, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %339 = stablehlo.transpose %27, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %340 = stablehlo.convert %339 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %341 = stablehlo.dot_general %330, %340, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x2304xf32>) -> tensor<2x6x2304xf32>
    %342 = stablehlo.convert %28 : (tensor<2304xf16>) -> tensor<2304xf32>
    %343 = stablehlo.broadcast_in_dim %342, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %344 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x6x2304xf32>
    %345 = stablehlo.add %341, %344 : tensor<2x6x2304xf32>
    %346 = stablehlo.slice %345 [0:2, 0:6, 0:768] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %347 = stablehlo.slice %345 [0:2, 0:6, 768:1536] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %348 = stablehlo.slice %345 [0:2, 0:6, 1536:2304] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %349 = stablehlo.reshape %346 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %350 = stablehlo.reshape %347 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %351 = stablehlo.reshape %348 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %352 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %353 = stablehlo.add %22, %16 : tensor<i32>
    %354 = stablehlo.select %352, %353, %22 : tensor<i1>, tensor<i32>
    %355 = stablehlo.dynamic_slice %338, %22, %22, %354, %22, sizes = [1, 1, 6, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x20xi1>
    %356 = stablehlo.reshape %355 : (tensor<1x1x6x20xi1>) -> tensor<1x6x20xi1>
    %357 = stablehlo.broadcast_in_dim %356, dims = [1, 2, 3] : (tensor<1x6x20xi1>) -> tensor<2x1x6x20xi1>
    %358 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
    %359 = stablehlo.reshape %358 : (tensor<2x1x1x20xi32>) -> tensor<2x1x20xi32>
    %360 = stablehlo.broadcast_in_dim %359, dims = [0, 1, 3] : (tensor<2x1x20xi32>) -> tensor<2x1x6x20xi32>
    %361 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x6x20xi32>
    %362 = stablehlo.compare  NE, %360, %361,  SIGNED : (tensor<2x1x6x20xi32>, tensor<2x1x6x20xi32>) -> tensor<2x1x6x20xi1>
    %363 = stablehlo.and %362, %357 : tensor<2x1x6x20xi1>
    %364 = stablehlo.convert %363 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %365 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %366 = stablehlo.add %22, %15 : tensor<i32>
    %367 = stablehlo.select %365, %366, %22 : tensor<i1>, tensor<i32>
    %368 = stablehlo.dynamic_update_slice %175, %350, %22, %367, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %369 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %370 = stablehlo.add %22, %15 : tensor<i32>
    %371 = stablehlo.select %369, %370, %22 : tensor<i1>, tensor<i32>
    %372 = stablehlo.dynamic_update_slice %176, %351, %22, %371, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %373 = stablehlo.add %22, %14 : tensor<i32>
    %374 = stablehlo.iota dim = 0 : tensor<20xi32>
    %375 = stablehlo.add %22, %14 : tensor<i32>
    %376 = stablehlo.broadcast_in_dim %375, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %377 = stablehlo.compare  LT, %374, %376,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %378 = stablehlo.broadcast_in_dim %377, dims = [3] : (tensor<20xi1>) -> tensor<2x1x6x20xi1>
    %379 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %380 = stablehlo.compare  NE, %364, %379,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %381 = stablehlo.and %378, %380 : tensor<2x1x6x20xi1>
    %382 = stablehlo.convert %381 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %383 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %384 = stablehlo.compare  GT, %382, %383,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %385 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %386 = stablehlo.convert %385 : tensor<2x1x6x20xf32>
    %387 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %388 = stablehlo.select %384, %386, %387 : tensor<2x1x6x20xi1>, tensor<2x1x6x20xf32>
    %389 = stablehlo.sqrt %12 : tensor<f32>
    %390 = stablehlo.convert %389 : tensor<f32>
    %391 = stablehlo.broadcast_in_dim %390, dims = [] : (tensor<f32>) -> tensor<2x6x12x64xf32>
    %392 = stablehlo.divide %349, %391 : tensor<2x6x12x64xf32>
    %393 = stablehlo.dot_general %392, %368, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x6x20xf32>
    %394 = stablehlo.broadcast_in_dim %388, dims = [0, 1, 2, 3] : (tensor<2x1x6x20xf32>) -> tensor<2x12x6x20xf32>
    %395 = stablehlo.add %393, %394 : tensor<2x12x6x20xf32>
    %396 = stablehlo.reduce(%395 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %397 = stablehlo.broadcast_in_dim %396, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %399 = stablehlo.subtract %395, %398 : tensor<2x12x6x20xf32>
    %400 = stablehlo.exponential %399 : tensor<2x12x6x20xf32>
    %401 = stablehlo.reduce(%400 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %403 = stablehlo.broadcast_in_dim %402, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %404 = stablehlo.divide %400, %403 : tensor<2x12x6x20xf32>
    %405 = stablehlo.dot_general %372, %404, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x6x20xf32>) -> tensor<2x12x64x6xf32>
    %406 = stablehlo.transpose %405, dims = [0, 3, 1, 2] : (tensor<2x12x64x6xf32>) -> tensor<2x6x12x64xf32>
    %407 = stablehlo.reshape %406 : (tensor<2x6x12x64xf32>) -> tensor<2x6x768xf32>
    %408 = stablehlo.transpose %29, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %409 = stablehlo.convert %408 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %410 = stablehlo.dot_general %407, %409, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x768xf32>) -> tensor<2x6x768xf32>
    %411 = stablehlo.convert %30 : (tensor<768xf16>) -> tensor<768xf32>
    %412 = stablehlo.broadcast_in_dim %411, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %413 = stablehlo.broadcast_in_dim %412, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %414 = stablehlo.add %410, %413 : tensor<2x6x768xf32>
    %415 = stablehlo.add %414, %302 : tensor<2x6x768xf32>
    %416 = stablehlo.multiply %415, %415 : tensor<2x6x768xf32>
    %417 = stablehlo.reduce(%415 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %418 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %419 = stablehlo.divide %417, %418 : tensor<2x6xf32>
    %420 = stablehlo.reduce(%416 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %421 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %422 = stablehlo.divide %420, %421 : tensor<2x6xf32>
    %423 = stablehlo.multiply %419, %419 : tensor<2x6xf32>
    %424 = stablehlo.subtract %422, %423 : tensor<2x6xf32>
    %425 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %426 = stablehlo.maximum %425, %424 : tensor<2x6xf32>
    %427 = stablehlo.broadcast_in_dim %419, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %428 = stablehlo.broadcast_in_dim %426, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %429 = stablehlo.broadcast_in_dim %427, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %430 = stablehlo.subtract %415, %429 : tensor<2x6x768xf32>
    %431 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %432 = stablehlo.add %428, %431 : tensor<2x6x1xf32>
    %433 = stablehlo.rsqrt %432 : tensor<2x6x1xf32>
    %434 = stablehlo.reshape %31 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %435 = stablehlo.convert %434 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %436 = stablehlo.broadcast_in_dim %433, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %437 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %438 = stablehlo.multiply %436, %437 : tensor<2x6x768xf32>
    %439 = stablehlo.multiply %430, %438 : tensor<2x6x768xf32>
    %440 = stablehlo.reshape %32 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %441 = stablehlo.convert %440 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %442 = stablehlo.broadcast_in_dim %441, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %443 = stablehlo.add %439, %442 : tensor<2x6x768xf32>
    %444 = stablehlo.transpose %33, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %445 = stablehlo.convert %444 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %446 = stablehlo.dot_general %443, %445, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x3072xf32>) -> tensor<2x6x3072xf32>
    %447 = stablehlo.convert %34 : (tensor<3072xf16>) -> tensor<3072xf32>
    %448 = stablehlo.broadcast_in_dim %447, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %449 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x6x3072xf32>
    %450 = stablehlo.add %446, %449 : tensor<2x6x3072xf32>
    %451 = stablehlo.multiply %450, %450 : tensor<2x6x3072xf32>
    %452 = stablehlo.multiply %450, %451 : tensor<2x6x3072xf32>
    %453 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %454 = stablehlo.multiply %453, %452 : tensor<2x6x3072xf32>
    %455 = stablehlo.add %450, %454 : tensor<2x6x3072xf32>
    %456 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %457 = stablehlo.multiply %456, %455 : tensor<2x6x3072xf32>
    %458 = stablehlo.tanh %457 : tensor<2x6x3072xf32>
    %459 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %460 = stablehlo.add %459, %458 : tensor<2x6x3072xf32>
    %461 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %462 = stablehlo.multiply %461, %460 : tensor<2x6x3072xf32>
    %463 = stablehlo.multiply %450, %462 : tensor<2x6x3072xf32>
    %464 = stablehlo.transpose %35, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %465 = stablehlo.convert %464 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %466 = stablehlo.dot_general %463, %465, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x3072xf32>, tensor<3072x768xf32>) -> tensor<2x6x768xf32>
    %467 = stablehlo.convert %36 : (tensor<768xf16>) -> tensor<768xf32>
    %468 = stablehlo.broadcast_in_dim %467, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %469 = stablehlo.broadcast_in_dim %468, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %470 = stablehlo.add %466, %469 : tensor<2x6x768xf32>
    %471 = stablehlo.add %415, %470 : tensor<2x6x768xf32>
    %472 = stablehlo.multiply %471, %471 : tensor<2x6x768xf32>
    %473 = stablehlo.reduce(%471 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %474 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %475 = stablehlo.divide %473, %474 : tensor<2x6xf32>
    %476 = stablehlo.reduce(%472 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %477 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %478 = stablehlo.divide %476, %477 : tensor<2x6xf32>
    %479 = stablehlo.multiply %475, %475 : tensor<2x6xf32>
    %480 = stablehlo.subtract %478, %479 : tensor<2x6xf32>
    %481 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %482 = stablehlo.maximum %481, %480 : tensor<2x6xf32>
    %483 = stablehlo.broadcast_in_dim %475, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %484 = stablehlo.broadcast_in_dim %482, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %485 = stablehlo.broadcast_in_dim %483, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %486 = stablehlo.subtract %471, %485 : tensor<2x6x768xf32>
    %487 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %488 = stablehlo.add %484, %487 : tensor<2x6x1xf32>
    %489 = stablehlo.rsqrt %488 : tensor<2x6x1xf32>
    %490 = stablehlo.reshape %37 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %491 = stablehlo.convert %490 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %492 = stablehlo.broadcast_in_dim %489, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %493 = stablehlo.broadcast_in_dim %491, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %494 = stablehlo.multiply %492, %493 : tensor<2x6x768xf32>
    %495 = stablehlo.multiply %486, %494 : tensor<2x6x768xf32>
    %496 = stablehlo.reshape %38 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %497 = stablehlo.convert %496 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %498 = stablehlo.broadcast_in_dim %497, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %499 = stablehlo.add %495, %498 : tensor<2x6x768xf32>
    %500 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %501 = stablehlo.broadcast_in_dim %500, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %502 = stablehlo.broadcast_in_dim %501, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %503 = stablehlo.broadcast_in_dim %501, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %504 = stablehlo.broadcast_in_dim %502, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %505 = stablehlo.broadcast_in_dim %503, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %506 = stablehlo.compare  GE, %504, %505,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %507 = stablehlo.broadcast_in_dim %506, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %508 = stablehlo.transpose %39, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %509 = stablehlo.convert %508 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %510 = stablehlo.dot_general %499, %509, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x2304xf32>) -> tensor<2x6x2304xf32>
    %511 = stablehlo.convert %40 : (tensor<2304xf16>) -> tensor<2304xf32>
    %512 = stablehlo.broadcast_in_dim %511, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %513 = stablehlo.broadcast_in_dim %512, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x6x2304xf32>
    %514 = stablehlo.add %510, %513 : tensor<2x6x2304xf32>
    %515 = stablehlo.slice %514 [0:2, 0:6, 0:768] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %516 = stablehlo.slice %514 [0:2, 0:6, 768:1536] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %517 = stablehlo.slice %514 [0:2, 0:6, 1536:2304] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %518 = stablehlo.reshape %515 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %519 = stablehlo.reshape %516 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %520 = stablehlo.reshape %517 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %521 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %522 = stablehlo.add %22, %16 : tensor<i32>
    %523 = stablehlo.select %521, %522, %22 : tensor<i1>, tensor<i32>
    %524 = stablehlo.dynamic_slice %507, %22, %22, %523, %22, sizes = [1, 1, 6, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x20xi1>
    %525 = stablehlo.reshape %524 : (tensor<1x1x6x20xi1>) -> tensor<1x6x20xi1>
    %526 = stablehlo.broadcast_in_dim %525, dims = [1, 2, 3] : (tensor<1x6x20xi1>) -> tensor<2x1x6x20xi1>
    %527 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
    %528 = stablehlo.reshape %527 : (tensor<2x1x1x20xi32>) -> tensor<2x1x20xi32>
    %529 = stablehlo.broadcast_in_dim %528, dims = [0, 1, 3] : (tensor<2x1x20xi32>) -> tensor<2x1x6x20xi32>
    %530 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x6x20xi32>
    %531 = stablehlo.compare  NE, %529, %530,  SIGNED : (tensor<2x1x6x20xi32>, tensor<2x1x6x20xi32>) -> tensor<2x1x6x20xi1>
    %532 = stablehlo.and %531, %526 : tensor<2x1x6x20xi1>
    %533 = stablehlo.convert %532 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %534 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %535 = stablehlo.add %22, %15 : tensor<i32>
    %536 = stablehlo.select %534, %535, %22 : tensor<i1>, tensor<i32>
    %537 = stablehlo.dynamic_update_slice %177, %519, %22, %536, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %538 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %539 = stablehlo.add %22, %15 : tensor<i32>
    %540 = stablehlo.select %538, %539, %22 : tensor<i1>, tensor<i32>
    %541 = stablehlo.dynamic_update_slice %178, %520, %22, %540, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %542 = stablehlo.add %22, %14 : tensor<i32>
    %543 = stablehlo.iota dim = 0 : tensor<20xi32>
    %544 = stablehlo.add %22, %14 : tensor<i32>
    %545 = stablehlo.broadcast_in_dim %544, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %546 = stablehlo.compare  LT, %543, %545,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %547 = stablehlo.broadcast_in_dim %546, dims = [3] : (tensor<20xi1>) -> tensor<2x1x6x20xi1>
    %548 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %549 = stablehlo.compare  NE, %533, %548,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %550 = stablehlo.and %547, %549 : tensor<2x1x6x20xi1>
    %551 = stablehlo.convert %550 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %552 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %553 = stablehlo.compare  GT, %551, %552,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %554 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %555 = stablehlo.convert %554 : tensor<2x1x6x20xf32>
    %556 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %557 = stablehlo.select %553, %555, %556 : tensor<2x1x6x20xi1>, tensor<2x1x6x20xf32>
    %558 = stablehlo.sqrt %12 : tensor<f32>
    %559 = stablehlo.convert %558 : tensor<f32>
    %560 = stablehlo.broadcast_in_dim %559, dims = [] : (tensor<f32>) -> tensor<2x6x12x64xf32>
    %561 = stablehlo.divide %518, %560 : tensor<2x6x12x64xf32>
    %562 = stablehlo.dot_general %561, %537, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x6x20xf32>
    %563 = stablehlo.broadcast_in_dim %557, dims = [0, 1, 2, 3] : (tensor<2x1x6x20xf32>) -> tensor<2x12x6x20xf32>
    %564 = stablehlo.add %562, %563 : tensor<2x12x6x20xf32>
    %565 = stablehlo.reduce(%564 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %566 = stablehlo.broadcast_in_dim %565, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %567 = stablehlo.broadcast_in_dim %566, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %568 = stablehlo.subtract %564, %567 : tensor<2x12x6x20xf32>
    %569 = stablehlo.exponential %568 : tensor<2x12x6x20xf32>
    %570 = stablehlo.reduce(%569 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %571 = stablehlo.broadcast_in_dim %570, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %572 = stablehlo.broadcast_in_dim %571, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %573 = stablehlo.divide %569, %572 : tensor<2x12x6x20xf32>
    %574 = stablehlo.dot_general %541, %573, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x6x20xf32>) -> tensor<2x12x64x6xf32>
    %575 = stablehlo.transpose %574, dims = [0, 3, 1, 2] : (tensor<2x12x64x6xf32>) -> tensor<2x6x12x64xf32>
    %576 = stablehlo.reshape %575 : (tensor<2x6x12x64xf32>) -> tensor<2x6x768xf32>
    %577 = stablehlo.transpose %41, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %578 = stablehlo.convert %577 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %579 = stablehlo.dot_general %576, %578, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x768xf32>) -> tensor<2x6x768xf32>
    %580 = stablehlo.convert %42 : (tensor<768xf16>) -> tensor<768xf32>
    %581 = stablehlo.broadcast_in_dim %580, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %582 = stablehlo.broadcast_in_dim %581, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %583 = stablehlo.add %579, %582 : tensor<2x6x768xf32>
    %584 = stablehlo.add %583, %471 : tensor<2x6x768xf32>
    %585 = stablehlo.multiply %584, %584 : tensor<2x6x768xf32>
    %586 = stablehlo.reduce(%584 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %587 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %588 = stablehlo.divide %586, %587 : tensor<2x6xf32>
    %589 = stablehlo.reduce(%585 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %590 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %591 = stablehlo.divide %589, %590 : tensor<2x6xf32>
    %592 = stablehlo.multiply %588, %588 : tensor<2x6xf32>
    %593 = stablehlo.subtract %591, %592 : tensor<2x6xf32>
    %594 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %595 = stablehlo.maximum %594, %593 : tensor<2x6xf32>
    %596 = stablehlo.broadcast_in_dim %588, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %597 = stablehlo.broadcast_in_dim %595, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %598 = stablehlo.broadcast_in_dim %596, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %599 = stablehlo.subtract %584, %598 : tensor<2x6x768xf32>
    %600 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %601 = stablehlo.add %597, %600 : tensor<2x6x1xf32>
    %602 = stablehlo.rsqrt %601 : tensor<2x6x1xf32>
    %603 = stablehlo.reshape %43 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %604 = stablehlo.convert %603 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %605 = stablehlo.broadcast_in_dim %602, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %606 = stablehlo.broadcast_in_dim %604, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %607 = stablehlo.multiply %605, %606 : tensor<2x6x768xf32>
    %608 = stablehlo.multiply %599, %607 : tensor<2x6x768xf32>
    %609 = stablehlo.reshape %44 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %610 = stablehlo.convert %609 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %611 = stablehlo.broadcast_in_dim %610, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %612 = stablehlo.add %608, %611 : tensor<2x6x768xf32>
    %613 = stablehlo.transpose %45, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %614 = stablehlo.convert %613 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %615 = stablehlo.dot_general %612, %614, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x3072xf32>) -> tensor<2x6x3072xf32>
    %616 = stablehlo.convert %46 : (tensor<3072xf16>) -> tensor<3072xf32>
    %617 = stablehlo.broadcast_in_dim %616, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %618 = stablehlo.broadcast_in_dim %617, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x6x3072xf32>
    %619 = stablehlo.add %615, %618 : tensor<2x6x3072xf32>
    %620 = stablehlo.multiply %619, %619 : tensor<2x6x3072xf32>
    %621 = stablehlo.multiply %619, %620 : tensor<2x6x3072xf32>
    %622 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %623 = stablehlo.multiply %622, %621 : tensor<2x6x3072xf32>
    %624 = stablehlo.add %619, %623 : tensor<2x6x3072xf32>
    %625 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %626 = stablehlo.multiply %625, %624 : tensor<2x6x3072xf32>
    %627 = stablehlo.tanh %626 : tensor<2x6x3072xf32>
    %628 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %629 = stablehlo.add %628, %627 : tensor<2x6x3072xf32>
    %630 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %631 = stablehlo.multiply %630, %629 : tensor<2x6x3072xf32>
    %632 = stablehlo.multiply %619, %631 : tensor<2x6x3072xf32>
    %633 = stablehlo.transpose %47, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %634 = stablehlo.convert %633 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %635 = stablehlo.dot_general %632, %634, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x3072xf32>, tensor<3072x768xf32>) -> tensor<2x6x768xf32>
    %636 = stablehlo.convert %48 : (tensor<768xf16>) -> tensor<768xf32>
    %637 = stablehlo.broadcast_in_dim %636, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %638 = stablehlo.broadcast_in_dim %637, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %639 = stablehlo.add %635, %638 : tensor<2x6x768xf32>
    %640 = stablehlo.add %584, %639 : tensor<2x6x768xf32>
    %641 = stablehlo.multiply %640, %640 : tensor<2x6x768xf32>
    %642 = stablehlo.reduce(%640 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %643 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %644 = stablehlo.divide %642, %643 : tensor<2x6xf32>
    %645 = stablehlo.reduce(%641 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %646 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %647 = stablehlo.divide %645, %646 : tensor<2x6xf32>
    %648 = stablehlo.multiply %644, %644 : tensor<2x6xf32>
    %649 = stablehlo.subtract %647, %648 : tensor<2x6xf32>
    %650 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %651 = stablehlo.maximum %650, %649 : tensor<2x6xf32>
    %652 = stablehlo.broadcast_in_dim %644, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %653 = stablehlo.broadcast_in_dim %651, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %654 = stablehlo.broadcast_in_dim %652, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %655 = stablehlo.subtract %640, %654 : tensor<2x6x768xf32>
    %656 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %657 = stablehlo.add %653, %656 : tensor<2x6x1xf32>
    %658 = stablehlo.rsqrt %657 : tensor<2x6x1xf32>
    %659 = stablehlo.reshape %49 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %660 = stablehlo.convert %659 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %661 = stablehlo.broadcast_in_dim %658, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %662 = stablehlo.broadcast_in_dim %660, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %663 = stablehlo.multiply %661, %662 : tensor<2x6x768xf32>
    %664 = stablehlo.multiply %655, %663 : tensor<2x6x768xf32>
    %665 = stablehlo.reshape %50 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %666 = stablehlo.convert %665 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %667 = stablehlo.broadcast_in_dim %666, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %668 = stablehlo.add %664, %667 : tensor<2x6x768xf32>
    %669 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %670 = stablehlo.broadcast_in_dim %669, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %671 = stablehlo.broadcast_in_dim %670, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %672 = stablehlo.broadcast_in_dim %670, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %673 = stablehlo.broadcast_in_dim %671, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %674 = stablehlo.broadcast_in_dim %672, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %675 = stablehlo.compare  GE, %673, %674,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %676 = stablehlo.broadcast_in_dim %675, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %677 = stablehlo.transpose %51, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %678 = stablehlo.convert %677 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %679 = stablehlo.dot_general %668, %678, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x2304xf32>) -> tensor<2x6x2304xf32>
    %680 = stablehlo.convert %52 : (tensor<2304xf16>) -> tensor<2304xf32>
    %681 = stablehlo.broadcast_in_dim %680, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %682 = stablehlo.broadcast_in_dim %681, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x6x2304xf32>
    %683 = stablehlo.add %679, %682 : tensor<2x6x2304xf32>
    %684 = stablehlo.slice %683 [0:2, 0:6, 0:768] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %685 = stablehlo.slice %683 [0:2, 0:6, 768:1536] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %686 = stablehlo.slice %683 [0:2, 0:6, 1536:2304] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %687 = stablehlo.reshape %684 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %688 = stablehlo.reshape %685 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %689 = stablehlo.reshape %686 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %690 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %691 = stablehlo.add %22, %16 : tensor<i32>
    %692 = stablehlo.select %690, %691, %22 : tensor<i1>, tensor<i32>
    %693 = stablehlo.dynamic_slice %676, %22, %22, %692, %22, sizes = [1, 1, 6, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x20xi1>
    %694 = stablehlo.reshape %693 : (tensor<1x1x6x20xi1>) -> tensor<1x6x20xi1>
    %695 = stablehlo.broadcast_in_dim %694, dims = [1, 2, 3] : (tensor<1x6x20xi1>) -> tensor<2x1x6x20xi1>
    %696 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
    %697 = stablehlo.reshape %696 : (tensor<2x1x1x20xi32>) -> tensor<2x1x20xi32>
    %698 = stablehlo.broadcast_in_dim %697, dims = [0, 1, 3] : (tensor<2x1x20xi32>) -> tensor<2x1x6x20xi32>
    %699 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x6x20xi32>
    %700 = stablehlo.compare  NE, %698, %699,  SIGNED : (tensor<2x1x6x20xi32>, tensor<2x1x6x20xi32>) -> tensor<2x1x6x20xi1>
    %701 = stablehlo.and %700, %695 : tensor<2x1x6x20xi1>
    %702 = stablehlo.convert %701 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %703 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %704 = stablehlo.add %22, %15 : tensor<i32>
    %705 = stablehlo.select %703, %704, %22 : tensor<i1>, tensor<i32>
    %706 = stablehlo.dynamic_update_slice %179, %688, %22, %705, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %707 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %708 = stablehlo.add %22, %15 : tensor<i32>
    %709 = stablehlo.select %707, %708, %22 : tensor<i1>, tensor<i32>
    %710 = stablehlo.dynamic_update_slice %180, %689, %22, %709, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %711 = stablehlo.add %22, %14 : tensor<i32>
    %712 = stablehlo.iota dim = 0 : tensor<20xi32>
    %713 = stablehlo.add %22, %14 : tensor<i32>
    %714 = stablehlo.broadcast_in_dim %713, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %715 = stablehlo.compare  LT, %712, %714,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %716 = stablehlo.broadcast_in_dim %715, dims = [3] : (tensor<20xi1>) -> tensor<2x1x6x20xi1>
    %717 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %718 = stablehlo.compare  NE, %702, %717,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %719 = stablehlo.and %716, %718 : tensor<2x1x6x20xi1>
    %720 = stablehlo.convert %719 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %721 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %722 = stablehlo.compare  GT, %720, %721,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %723 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %724 = stablehlo.convert %723 : tensor<2x1x6x20xf32>
    %725 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %726 = stablehlo.select %722, %724, %725 : tensor<2x1x6x20xi1>, tensor<2x1x6x20xf32>
    %727 = stablehlo.sqrt %12 : tensor<f32>
    %728 = stablehlo.convert %727 : tensor<f32>
    %729 = stablehlo.broadcast_in_dim %728, dims = [] : (tensor<f32>) -> tensor<2x6x12x64xf32>
    %730 = stablehlo.divide %687, %729 : tensor<2x6x12x64xf32>
    %731 = stablehlo.dot_general %730, %706, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x6x20xf32>
    %732 = stablehlo.broadcast_in_dim %726, dims = [0, 1, 2, 3] : (tensor<2x1x6x20xf32>) -> tensor<2x12x6x20xf32>
    %733 = stablehlo.add %731, %732 : tensor<2x12x6x20xf32>
    %734 = stablehlo.reduce(%733 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %735 = stablehlo.broadcast_in_dim %734, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %736 = stablehlo.broadcast_in_dim %735, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %737 = stablehlo.subtract %733, %736 : tensor<2x12x6x20xf32>
    %738 = stablehlo.exponential %737 : tensor<2x12x6x20xf32>
    %739 = stablehlo.reduce(%738 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %740 = stablehlo.broadcast_in_dim %739, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %741 = stablehlo.broadcast_in_dim %740, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %742 = stablehlo.divide %738, %741 : tensor<2x12x6x20xf32>
    %743 = stablehlo.dot_general %710, %742, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x6x20xf32>) -> tensor<2x12x64x6xf32>
    %744 = stablehlo.transpose %743, dims = [0, 3, 1, 2] : (tensor<2x12x64x6xf32>) -> tensor<2x6x12x64xf32>
    %745 = stablehlo.reshape %744 : (tensor<2x6x12x64xf32>) -> tensor<2x6x768xf32>
    %746 = stablehlo.transpose %53, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %747 = stablehlo.convert %746 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %748 = stablehlo.dot_general %745, %747, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x768xf32>) -> tensor<2x6x768xf32>
    %749 = stablehlo.convert %54 : (tensor<768xf16>) -> tensor<768xf32>
    %750 = stablehlo.broadcast_in_dim %749, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %751 = stablehlo.broadcast_in_dim %750, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %752 = stablehlo.add %748, %751 : tensor<2x6x768xf32>
    %753 = stablehlo.add %752, %640 : tensor<2x6x768xf32>
    %754 = stablehlo.multiply %753, %753 : tensor<2x6x768xf32>
    %755 = stablehlo.reduce(%753 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %756 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %757 = stablehlo.divide %755, %756 : tensor<2x6xf32>
    %758 = stablehlo.reduce(%754 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %759 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %760 = stablehlo.divide %758, %759 : tensor<2x6xf32>
    %761 = stablehlo.multiply %757, %757 : tensor<2x6xf32>
    %762 = stablehlo.subtract %760, %761 : tensor<2x6xf32>
    %763 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %764 = stablehlo.maximum %763, %762 : tensor<2x6xf32>
    %765 = stablehlo.broadcast_in_dim %757, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %766 = stablehlo.broadcast_in_dim %764, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %767 = stablehlo.broadcast_in_dim %765, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %768 = stablehlo.subtract %753, %767 : tensor<2x6x768xf32>
    %769 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %770 = stablehlo.add %766, %769 : tensor<2x6x1xf32>
    %771 = stablehlo.rsqrt %770 : tensor<2x6x1xf32>
    %772 = stablehlo.reshape %55 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %773 = stablehlo.convert %772 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %774 = stablehlo.broadcast_in_dim %771, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %775 = stablehlo.broadcast_in_dim %773, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %776 = stablehlo.multiply %774, %775 : tensor<2x6x768xf32>
    %777 = stablehlo.multiply %768, %776 : tensor<2x6x768xf32>
    %778 = stablehlo.reshape %56 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %779 = stablehlo.convert %778 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %780 = stablehlo.broadcast_in_dim %779, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %781 = stablehlo.add %777, %780 : tensor<2x6x768xf32>
    %782 = stablehlo.transpose %57, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %783 = stablehlo.convert %782 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %784 = stablehlo.dot_general %781, %783, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x3072xf32>) -> tensor<2x6x3072xf32>
    %785 = stablehlo.convert %58 : (tensor<3072xf16>) -> tensor<3072xf32>
    %786 = stablehlo.broadcast_in_dim %785, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %787 = stablehlo.broadcast_in_dim %786, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x6x3072xf32>
    %788 = stablehlo.add %784, %787 : tensor<2x6x3072xf32>
    %789 = stablehlo.multiply %788, %788 : tensor<2x6x3072xf32>
    %790 = stablehlo.multiply %788, %789 : tensor<2x6x3072xf32>
    %791 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %792 = stablehlo.multiply %791, %790 : tensor<2x6x3072xf32>
    %793 = stablehlo.add %788, %792 : tensor<2x6x3072xf32>
    %794 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %795 = stablehlo.multiply %794, %793 : tensor<2x6x3072xf32>
    %796 = stablehlo.tanh %795 : tensor<2x6x3072xf32>
    %797 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %798 = stablehlo.add %797, %796 : tensor<2x6x3072xf32>
    %799 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %800 = stablehlo.multiply %799, %798 : tensor<2x6x3072xf32>
    %801 = stablehlo.multiply %788, %800 : tensor<2x6x3072xf32>
    %802 = stablehlo.transpose %59, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %803 = stablehlo.convert %802 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %804 = stablehlo.dot_general %801, %803, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x3072xf32>, tensor<3072x768xf32>) -> tensor<2x6x768xf32>
    %805 = stablehlo.convert %60 : (tensor<768xf16>) -> tensor<768xf32>
    %806 = stablehlo.broadcast_in_dim %805, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %807 = stablehlo.broadcast_in_dim %806, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %808 = stablehlo.add %804, %807 : tensor<2x6x768xf32>
    %809 = stablehlo.add %753, %808 : tensor<2x6x768xf32>
    %810 = stablehlo.multiply %809, %809 : tensor<2x6x768xf32>
    %811 = stablehlo.reduce(%809 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %812 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %813 = stablehlo.divide %811, %812 : tensor<2x6xf32>
    %814 = stablehlo.reduce(%810 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %815 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %816 = stablehlo.divide %814, %815 : tensor<2x6xf32>
    %817 = stablehlo.multiply %813, %813 : tensor<2x6xf32>
    %818 = stablehlo.subtract %816, %817 : tensor<2x6xf32>
    %819 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %820 = stablehlo.maximum %819, %818 : tensor<2x6xf32>
    %821 = stablehlo.broadcast_in_dim %813, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %822 = stablehlo.broadcast_in_dim %820, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %823 = stablehlo.broadcast_in_dim %821, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %824 = stablehlo.subtract %809, %823 : tensor<2x6x768xf32>
    %825 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %826 = stablehlo.add %822, %825 : tensor<2x6x1xf32>
    %827 = stablehlo.rsqrt %826 : tensor<2x6x1xf32>
    %828 = stablehlo.reshape %61 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %829 = stablehlo.convert %828 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %830 = stablehlo.broadcast_in_dim %827, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %831 = stablehlo.broadcast_in_dim %829, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %832 = stablehlo.multiply %830, %831 : tensor<2x6x768xf32>
    %833 = stablehlo.multiply %824, %832 : tensor<2x6x768xf32>
    %834 = stablehlo.reshape %62 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %835 = stablehlo.convert %834 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %836 = stablehlo.broadcast_in_dim %835, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %837 = stablehlo.add %833, %836 : tensor<2x6x768xf32>
    %838 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %839 = stablehlo.broadcast_in_dim %838, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %840 = stablehlo.broadcast_in_dim %839, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %841 = stablehlo.broadcast_in_dim %839, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %842 = stablehlo.broadcast_in_dim %840, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %843 = stablehlo.broadcast_in_dim %841, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %844 = stablehlo.compare  GE, %842, %843,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %845 = stablehlo.broadcast_in_dim %844, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %846 = stablehlo.transpose %63, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %847 = stablehlo.convert %846 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %848 = stablehlo.dot_general %837, %847, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x2304xf32>) -> tensor<2x6x2304xf32>
    %849 = stablehlo.convert %64 : (tensor<2304xf16>) -> tensor<2304xf32>
    %850 = stablehlo.broadcast_in_dim %849, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %851 = stablehlo.broadcast_in_dim %850, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x6x2304xf32>
    %852 = stablehlo.add %848, %851 : tensor<2x6x2304xf32>
    %853 = stablehlo.slice %852 [0:2, 0:6, 0:768] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %854 = stablehlo.slice %852 [0:2, 0:6, 768:1536] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %855 = stablehlo.slice %852 [0:2, 0:6, 1536:2304] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %856 = stablehlo.reshape %853 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %857 = stablehlo.reshape %854 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %858 = stablehlo.reshape %855 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %859 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %860 = stablehlo.add %22, %16 : tensor<i32>
    %861 = stablehlo.select %859, %860, %22 : tensor<i1>, tensor<i32>
    %862 = stablehlo.dynamic_slice %845, %22, %22, %861, %22, sizes = [1, 1, 6, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x20xi1>
    %863 = stablehlo.reshape %862 : (tensor<1x1x6x20xi1>) -> tensor<1x6x20xi1>
    %864 = stablehlo.broadcast_in_dim %863, dims = [1, 2, 3] : (tensor<1x6x20xi1>) -> tensor<2x1x6x20xi1>
    %865 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
    %866 = stablehlo.reshape %865 : (tensor<2x1x1x20xi32>) -> tensor<2x1x20xi32>
    %867 = stablehlo.broadcast_in_dim %866, dims = [0, 1, 3] : (tensor<2x1x20xi32>) -> tensor<2x1x6x20xi32>
    %868 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x6x20xi32>
    %869 = stablehlo.compare  NE, %867, %868,  SIGNED : (tensor<2x1x6x20xi32>, tensor<2x1x6x20xi32>) -> tensor<2x1x6x20xi1>
    %870 = stablehlo.and %869, %864 : tensor<2x1x6x20xi1>
    %871 = stablehlo.convert %870 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %872 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %873 = stablehlo.add %22, %15 : tensor<i32>
    %874 = stablehlo.select %872, %873, %22 : tensor<i1>, tensor<i32>
    %875 = stablehlo.dynamic_update_slice %181, %857, %22, %874, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %876 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %877 = stablehlo.add %22, %15 : tensor<i32>
    %878 = stablehlo.select %876, %877, %22 : tensor<i1>, tensor<i32>
    %879 = stablehlo.dynamic_update_slice %182, %858, %22, %878, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %880 = stablehlo.add %22, %14 : tensor<i32>
    %881 = stablehlo.iota dim = 0 : tensor<20xi32>
    %882 = stablehlo.add %22, %14 : tensor<i32>
    %883 = stablehlo.broadcast_in_dim %882, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %884 = stablehlo.compare  LT, %881, %883,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %885 = stablehlo.broadcast_in_dim %884, dims = [3] : (tensor<20xi1>) -> tensor<2x1x6x20xi1>
    %886 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %887 = stablehlo.compare  NE, %871, %886,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %888 = stablehlo.and %885, %887 : tensor<2x1x6x20xi1>
    %889 = stablehlo.convert %888 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %890 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %891 = stablehlo.compare  GT, %889, %890,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %892 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %893 = stablehlo.convert %892 : tensor<2x1x6x20xf32>
    %894 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %895 = stablehlo.select %891, %893, %894 : tensor<2x1x6x20xi1>, tensor<2x1x6x20xf32>
    %896 = stablehlo.sqrt %12 : tensor<f32>
    %897 = stablehlo.convert %896 : tensor<f32>
    %898 = stablehlo.broadcast_in_dim %897, dims = [] : (tensor<f32>) -> tensor<2x6x12x64xf32>
    %899 = stablehlo.divide %856, %898 : tensor<2x6x12x64xf32>
    %900 = stablehlo.dot_general %899, %875, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x6x20xf32>
    %901 = stablehlo.broadcast_in_dim %895, dims = [0, 1, 2, 3] : (tensor<2x1x6x20xf32>) -> tensor<2x12x6x20xf32>
    %902 = stablehlo.add %900, %901 : tensor<2x12x6x20xf32>
    %903 = stablehlo.reduce(%902 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %904 = stablehlo.broadcast_in_dim %903, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %905 = stablehlo.broadcast_in_dim %904, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %906 = stablehlo.subtract %902, %905 : tensor<2x12x6x20xf32>
    %907 = stablehlo.exponential %906 : tensor<2x12x6x20xf32>
    %908 = stablehlo.reduce(%907 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %909 = stablehlo.broadcast_in_dim %908, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %910 = stablehlo.broadcast_in_dim %909, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %911 = stablehlo.divide %907, %910 : tensor<2x12x6x20xf32>
    %912 = stablehlo.dot_general %879, %911, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x6x20xf32>) -> tensor<2x12x64x6xf32>
    %913 = stablehlo.transpose %912, dims = [0, 3, 1, 2] : (tensor<2x12x64x6xf32>) -> tensor<2x6x12x64xf32>
    %914 = stablehlo.reshape %913 : (tensor<2x6x12x64xf32>) -> tensor<2x6x768xf32>
    %915 = stablehlo.transpose %65, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %916 = stablehlo.convert %915 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %917 = stablehlo.dot_general %914, %916, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x768xf32>) -> tensor<2x6x768xf32>
    %918 = stablehlo.convert %66 : (tensor<768xf16>) -> tensor<768xf32>
    %919 = stablehlo.broadcast_in_dim %918, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %920 = stablehlo.broadcast_in_dim %919, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %921 = stablehlo.add %917, %920 : tensor<2x6x768xf32>
    %922 = stablehlo.add %921, %809 : tensor<2x6x768xf32>
    %923 = stablehlo.multiply %922, %922 : tensor<2x6x768xf32>
    %924 = stablehlo.reduce(%922 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %925 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %926 = stablehlo.divide %924, %925 : tensor<2x6xf32>
    %927 = stablehlo.reduce(%923 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %928 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %929 = stablehlo.divide %927, %928 : tensor<2x6xf32>
    %930 = stablehlo.multiply %926, %926 : tensor<2x6xf32>
    %931 = stablehlo.subtract %929, %930 : tensor<2x6xf32>
    %932 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %933 = stablehlo.maximum %932, %931 : tensor<2x6xf32>
    %934 = stablehlo.broadcast_in_dim %926, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %935 = stablehlo.broadcast_in_dim %933, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %936 = stablehlo.broadcast_in_dim %934, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %937 = stablehlo.subtract %922, %936 : tensor<2x6x768xf32>
    %938 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %939 = stablehlo.add %935, %938 : tensor<2x6x1xf32>
    %940 = stablehlo.rsqrt %939 : tensor<2x6x1xf32>
    %941 = stablehlo.reshape %67 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %942 = stablehlo.convert %941 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %943 = stablehlo.broadcast_in_dim %940, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %944 = stablehlo.broadcast_in_dim %942, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %945 = stablehlo.multiply %943, %944 : tensor<2x6x768xf32>
    %946 = stablehlo.multiply %937, %945 : tensor<2x6x768xf32>
    %947 = stablehlo.reshape %68 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %948 = stablehlo.convert %947 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %949 = stablehlo.broadcast_in_dim %948, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %950 = stablehlo.add %946, %949 : tensor<2x6x768xf32>
    %951 = stablehlo.transpose %69, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %952 = stablehlo.convert %951 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %953 = stablehlo.dot_general %950, %952, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x3072xf32>) -> tensor<2x6x3072xf32>
    %954 = stablehlo.convert %70 : (tensor<3072xf16>) -> tensor<3072xf32>
    %955 = stablehlo.broadcast_in_dim %954, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %956 = stablehlo.broadcast_in_dim %955, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x6x3072xf32>
    %957 = stablehlo.add %953, %956 : tensor<2x6x3072xf32>
    %958 = stablehlo.multiply %957, %957 : tensor<2x6x3072xf32>
    %959 = stablehlo.multiply %957, %958 : tensor<2x6x3072xf32>
    %960 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %961 = stablehlo.multiply %960, %959 : tensor<2x6x3072xf32>
    %962 = stablehlo.add %957, %961 : tensor<2x6x3072xf32>
    %963 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %964 = stablehlo.multiply %963, %962 : tensor<2x6x3072xf32>
    %965 = stablehlo.tanh %964 : tensor<2x6x3072xf32>
    %966 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %967 = stablehlo.add %966, %965 : tensor<2x6x3072xf32>
    %968 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %969 = stablehlo.multiply %968, %967 : tensor<2x6x3072xf32>
    %970 = stablehlo.multiply %957, %969 : tensor<2x6x3072xf32>
    %971 = stablehlo.transpose %71, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %972 = stablehlo.convert %971 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %973 = stablehlo.dot_general %970, %972, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x3072xf32>, tensor<3072x768xf32>) -> tensor<2x6x768xf32>
    %974 = stablehlo.convert %72 : (tensor<768xf16>) -> tensor<768xf32>
    %975 = stablehlo.broadcast_in_dim %974, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %976 = stablehlo.broadcast_in_dim %975, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %977 = stablehlo.add %973, %976 : tensor<2x6x768xf32>
    %978 = stablehlo.add %922, %977 : tensor<2x6x768xf32>
    %979 = stablehlo.multiply %978, %978 : tensor<2x6x768xf32>
    %980 = stablehlo.reduce(%978 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %981 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %982 = stablehlo.divide %980, %981 : tensor<2x6xf32>
    %983 = stablehlo.reduce(%979 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %984 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %985 = stablehlo.divide %983, %984 : tensor<2x6xf32>
    %986 = stablehlo.multiply %982, %982 : tensor<2x6xf32>
    %987 = stablehlo.subtract %985, %986 : tensor<2x6xf32>
    %988 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %989 = stablehlo.maximum %988, %987 : tensor<2x6xf32>
    %990 = stablehlo.broadcast_in_dim %982, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %991 = stablehlo.broadcast_in_dim %989, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %992 = stablehlo.broadcast_in_dim %990, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %993 = stablehlo.subtract %978, %992 : tensor<2x6x768xf32>
    %994 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %995 = stablehlo.add %991, %994 : tensor<2x6x1xf32>
    %996 = stablehlo.rsqrt %995 : tensor<2x6x1xf32>
    %997 = stablehlo.reshape %73 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %998 = stablehlo.convert %997 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %999 = stablehlo.broadcast_in_dim %996, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1000 = stablehlo.broadcast_in_dim %998, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1001 = stablehlo.multiply %999, %1000 : tensor<2x6x768xf32>
    %1002 = stablehlo.multiply %993, %1001 : tensor<2x6x768xf32>
    %1003 = stablehlo.reshape %74 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1004 = stablehlo.convert %1003 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1005 = stablehlo.broadcast_in_dim %1004, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1006 = stablehlo.add %1002, %1005 : tensor<2x6x768xf32>
    %1007 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %1008 = stablehlo.broadcast_in_dim %1007, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %1009 = stablehlo.broadcast_in_dim %1008, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %1010 = stablehlo.broadcast_in_dim %1008, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %1011 = stablehlo.broadcast_in_dim %1009, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %1012 = stablehlo.broadcast_in_dim %1010, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %1013 = stablehlo.compare  GE, %1011, %1012,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %1014 = stablehlo.broadcast_in_dim %1013, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %1015 = stablehlo.transpose %75, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %1016 = stablehlo.convert %1015 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %1017 = stablehlo.dot_general %1006, %1016, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x2304xf32>) -> tensor<2x6x2304xf32>
    %1018 = stablehlo.convert %76 : (tensor<2304xf16>) -> tensor<2304xf32>
    %1019 = stablehlo.broadcast_in_dim %1018, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %1020 = stablehlo.broadcast_in_dim %1019, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x6x2304xf32>
    %1021 = stablehlo.add %1017, %1020 : tensor<2x6x2304xf32>
    %1022 = stablehlo.slice %1021 [0:2, 0:6, 0:768] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1023 = stablehlo.slice %1021 [0:2, 0:6, 768:1536] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1024 = stablehlo.slice %1021 [0:2, 0:6, 1536:2304] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1025 = stablehlo.reshape %1022 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1026 = stablehlo.reshape %1023 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1027 = stablehlo.reshape %1024 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1028 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1029 = stablehlo.add %22, %16 : tensor<i32>
    %1030 = stablehlo.select %1028, %1029, %22 : tensor<i1>, tensor<i32>
    %1031 = stablehlo.dynamic_slice %1014, %22, %22, %1030, %22, sizes = [1, 1, 6, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x20xi1>
    %1032 = stablehlo.reshape %1031 : (tensor<1x1x6x20xi1>) -> tensor<1x6x20xi1>
    %1033 = stablehlo.broadcast_in_dim %1032, dims = [1, 2, 3] : (tensor<1x6x20xi1>) -> tensor<2x1x6x20xi1>
    %1034 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
    %1035 = stablehlo.reshape %1034 : (tensor<2x1x1x20xi32>) -> tensor<2x1x20xi32>
    %1036 = stablehlo.broadcast_in_dim %1035, dims = [0, 1, 3] : (tensor<2x1x20xi32>) -> tensor<2x1x6x20xi32>
    %1037 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x6x20xi32>
    %1038 = stablehlo.compare  NE, %1036, %1037,  SIGNED : (tensor<2x1x6x20xi32>, tensor<2x1x6x20xi32>) -> tensor<2x1x6x20xi1>
    %1039 = stablehlo.and %1038, %1033 : tensor<2x1x6x20xi1>
    %1040 = stablehlo.convert %1039 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %1041 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1042 = stablehlo.add %22, %15 : tensor<i32>
    %1043 = stablehlo.select %1041, %1042, %22 : tensor<i1>, tensor<i32>
    %1044 = stablehlo.dynamic_update_slice %183, %1026, %22, %1043, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %1045 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1046 = stablehlo.add %22, %15 : tensor<i32>
    %1047 = stablehlo.select %1045, %1046, %22 : tensor<i1>, tensor<i32>
    %1048 = stablehlo.dynamic_update_slice %184, %1027, %22, %1047, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %1049 = stablehlo.add %22, %14 : tensor<i32>
    %1050 = stablehlo.iota dim = 0 : tensor<20xi32>
    %1051 = stablehlo.add %22, %14 : tensor<i32>
    %1052 = stablehlo.broadcast_in_dim %1051, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %1053 = stablehlo.compare  LT, %1050, %1052,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %1054 = stablehlo.broadcast_in_dim %1053, dims = [3] : (tensor<20xi1>) -> tensor<2x1x6x20xi1>
    %1055 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1056 = stablehlo.compare  NE, %1040, %1055,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %1057 = stablehlo.and %1054, %1056 : tensor<2x1x6x20xi1>
    %1058 = stablehlo.convert %1057 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %1059 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1060 = stablehlo.compare  GT, %1058, %1059,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %1061 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1062 = stablehlo.convert %1061 : tensor<2x1x6x20xf32>
    %1063 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1064 = stablehlo.select %1060, %1062, %1063 : tensor<2x1x6x20xi1>, tensor<2x1x6x20xf32>
    %1065 = stablehlo.sqrt %12 : tensor<f32>
    %1066 = stablehlo.convert %1065 : tensor<f32>
    %1067 = stablehlo.broadcast_in_dim %1066, dims = [] : (tensor<f32>) -> tensor<2x6x12x64xf32>
    %1068 = stablehlo.divide %1025, %1067 : tensor<2x6x12x64xf32>
    %1069 = stablehlo.dot_general %1068, %1044, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x6x20xf32>
    %1070 = stablehlo.broadcast_in_dim %1064, dims = [0, 1, 2, 3] : (tensor<2x1x6x20xf32>) -> tensor<2x12x6x20xf32>
    %1071 = stablehlo.add %1069, %1070 : tensor<2x12x6x20xf32>
    %1072 = stablehlo.reduce(%1071 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %1073 = stablehlo.broadcast_in_dim %1072, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %1074 = stablehlo.broadcast_in_dim %1073, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %1075 = stablehlo.subtract %1071, %1074 : tensor<2x12x6x20xf32>
    %1076 = stablehlo.exponential %1075 : tensor<2x12x6x20xf32>
    %1077 = stablehlo.reduce(%1076 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %1078 = stablehlo.broadcast_in_dim %1077, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %1079 = stablehlo.broadcast_in_dim %1078, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %1080 = stablehlo.divide %1076, %1079 : tensor<2x12x6x20xf32>
    %1081 = stablehlo.dot_general %1048, %1080, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x6x20xf32>) -> tensor<2x12x64x6xf32>
    %1082 = stablehlo.transpose %1081, dims = [0, 3, 1, 2] : (tensor<2x12x64x6xf32>) -> tensor<2x6x12x64xf32>
    %1083 = stablehlo.reshape %1082 : (tensor<2x6x12x64xf32>) -> tensor<2x6x768xf32>
    %1084 = stablehlo.transpose %77, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %1085 = stablehlo.convert %1084 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1086 = stablehlo.dot_general %1083, %1085, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x768xf32>) -> tensor<2x6x768xf32>
    %1087 = stablehlo.convert %78 : (tensor<768xf16>) -> tensor<768xf32>
    %1088 = stablehlo.broadcast_in_dim %1087, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1089 = stablehlo.broadcast_in_dim %1088, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1090 = stablehlo.add %1086, %1089 : tensor<2x6x768xf32>
    %1091 = stablehlo.add %1090, %978 : tensor<2x6x768xf32>
    %1092 = stablehlo.multiply %1091, %1091 : tensor<2x6x768xf32>
    %1093 = stablehlo.reduce(%1091 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1094 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1095 = stablehlo.divide %1093, %1094 : tensor<2x6xf32>
    %1096 = stablehlo.reduce(%1092 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1097 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1098 = stablehlo.divide %1096, %1097 : tensor<2x6xf32>
    %1099 = stablehlo.multiply %1095, %1095 : tensor<2x6xf32>
    %1100 = stablehlo.subtract %1098, %1099 : tensor<2x6xf32>
    %1101 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1102 = stablehlo.maximum %1101, %1100 : tensor<2x6xf32>
    %1103 = stablehlo.broadcast_in_dim %1095, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1104 = stablehlo.broadcast_in_dim %1102, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1105 = stablehlo.broadcast_in_dim %1103, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1106 = stablehlo.subtract %1091, %1105 : tensor<2x6x768xf32>
    %1107 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %1108 = stablehlo.add %1104, %1107 : tensor<2x6x1xf32>
    %1109 = stablehlo.rsqrt %1108 : tensor<2x6x1xf32>
    %1110 = stablehlo.reshape %79 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1111 = stablehlo.convert %1110 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1112 = stablehlo.broadcast_in_dim %1109, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1113 = stablehlo.broadcast_in_dim %1111, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1114 = stablehlo.multiply %1112, %1113 : tensor<2x6x768xf32>
    %1115 = stablehlo.multiply %1106, %1114 : tensor<2x6x768xf32>
    %1116 = stablehlo.reshape %80 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1117 = stablehlo.convert %1116 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1118 = stablehlo.broadcast_in_dim %1117, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1119 = stablehlo.add %1115, %1118 : tensor<2x6x768xf32>
    %1120 = stablehlo.transpose %81, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %1121 = stablehlo.convert %1120 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1122 = stablehlo.dot_general %1119, %1121, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x3072xf32>) -> tensor<2x6x3072xf32>
    %1123 = stablehlo.convert %82 : (tensor<3072xf16>) -> tensor<3072xf32>
    %1124 = stablehlo.broadcast_in_dim %1123, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %1125 = stablehlo.broadcast_in_dim %1124, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x6x3072xf32>
    %1126 = stablehlo.add %1122, %1125 : tensor<2x6x3072xf32>
    %1127 = stablehlo.multiply %1126, %1126 : tensor<2x6x3072xf32>
    %1128 = stablehlo.multiply %1126, %1127 : tensor<2x6x3072xf32>
    %1129 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1130 = stablehlo.multiply %1129, %1128 : tensor<2x6x3072xf32>
    %1131 = stablehlo.add %1126, %1130 : tensor<2x6x3072xf32>
    %1132 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1133 = stablehlo.multiply %1132, %1131 : tensor<2x6x3072xf32>
    %1134 = stablehlo.tanh %1133 : tensor<2x6x3072xf32>
    %1135 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1136 = stablehlo.add %1135, %1134 : tensor<2x6x3072xf32>
    %1137 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1138 = stablehlo.multiply %1137, %1136 : tensor<2x6x3072xf32>
    %1139 = stablehlo.multiply %1126, %1138 : tensor<2x6x3072xf32>
    %1140 = stablehlo.transpose %83, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %1141 = stablehlo.convert %1140 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1142 = stablehlo.dot_general %1139, %1141, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x3072xf32>, tensor<3072x768xf32>) -> tensor<2x6x768xf32>
    %1143 = stablehlo.convert %84 : (tensor<768xf16>) -> tensor<768xf32>
    %1144 = stablehlo.broadcast_in_dim %1143, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1145 = stablehlo.broadcast_in_dim %1144, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1146 = stablehlo.add %1142, %1145 : tensor<2x6x768xf32>
    %1147 = stablehlo.add %1091, %1146 : tensor<2x6x768xf32>
    %1148 = stablehlo.multiply %1147, %1147 : tensor<2x6x768xf32>
    %1149 = stablehlo.reduce(%1147 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1150 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1151 = stablehlo.divide %1149, %1150 : tensor<2x6xf32>
    %1152 = stablehlo.reduce(%1148 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1153 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1154 = stablehlo.divide %1152, %1153 : tensor<2x6xf32>
    %1155 = stablehlo.multiply %1151, %1151 : tensor<2x6xf32>
    %1156 = stablehlo.subtract %1154, %1155 : tensor<2x6xf32>
    %1157 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1158 = stablehlo.maximum %1157, %1156 : tensor<2x6xf32>
    %1159 = stablehlo.broadcast_in_dim %1151, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1160 = stablehlo.broadcast_in_dim %1158, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1161 = stablehlo.broadcast_in_dim %1159, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1162 = stablehlo.subtract %1147, %1161 : tensor<2x6x768xf32>
    %1163 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %1164 = stablehlo.add %1160, %1163 : tensor<2x6x1xf32>
    %1165 = stablehlo.rsqrt %1164 : tensor<2x6x1xf32>
    %1166 = stablehlo.reshape %85 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1167 = stablehlo.convert %1166 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1168 = stablehlo.broadcast_in_dim %1165, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1169 = stablehlo.broadcast_in_dim %1167, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1170 = stablehlo.multiply %1168, %1169 : tensor<2x6x768xf32>
    %1171 = stablehlo.multiply %1162, %1170 : tensor<2x6x768xf32>
    %1172 = stablehlo.reshape %86 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1173 = stablehlo.convert %1172 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1174 = stablehlo.broadcast_in_dim %1173, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1175 = stablehlo.add %1171, %1174 : tensor<2x6x768xf32>
    %1176 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %1177 = stablehlo.broadcast_in_dim %1176, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %1178 = stablehlo.broadcast_in_dim %1177, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %1179 = stablehlo.broadcast_in_dim %1177, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %1180 = stablehlo.broadcast_in_dim %1178, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %1181 = stablehlo.broadcast_in_dim %1179, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %1182 = stablehlo.compare  GE, %1180, %1181,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %1183 = stablehlo.broadcast_in_dim %1182, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %1184 = stablehlo.transpose %87, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %1185 = stablehlo.convert %1184 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %1186 = stablehlo.dot_general %1175, %1185, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x2304xf32>) -> tensor<2x6x2304xf32>
    %1187 = stablehlo.convert %88 : (tensor<2304xf16>) -> tensor<2304xf32>
    %1188 = stablehlo.broadcast_in_dim %1187, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %1189 = stablehlo.broadcast_in_dim %1188, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x6x2304xf32>
    %1190 = stablehlo.add %1186, %1189 : tensor<2x6x2304xf32>
    %1191 = stablehlo.slice %1190 [0:2, 0:6, 0:768] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1192 = stablehlo.slice %1190 [0:2, 0:6, 768:1536] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1193 = stablehlo.slice %1190 [0:2, 0:6, 1536:2304] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1194 = stablehlo.reshape %1191 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1195 = stablehlo.reshape %1192 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1196 = stablehlo.reshape %1193 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1197 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1198 = stablehlo.add %22, %16 : tensor<i32>
    %1199 = stablehlo.select %1197, %1198, %22 : tensor<i1>, tensor<i32>
    %1200 = stablehlo.dynamic_slice %1183, %22, %22, %1199, %22, sizes = [1, 1, 6, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x20xi1>
    %1201 = stablehlo.reshape %1200 : (tensor<1x1x6x20xi1>) -> tensor<1x6x20xi1>
    %1202 = stablehlo.broadcast_in_dim %1201, dims = [1, 2, 3] : (tensor<1x6x20xi1>) -> tensor<2x1x6x20xi1>
    %1203 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
    %1204 = stablehlo.reshape %1203 : (tensor<2x1x1x20xi32>) -> tensor<2x1x20xi32>
    %1205 = stablehlo.broadcast_in_dim %1204, dims = [0, 1, 3] : (tensor<2x1x20xi32>) -> tensor<2x1x6x20xi32>
    %1206 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x6x20xi32>
    %1207 = stablehlo.compare  NE, %1205, %1206,  SIGNED : (tensor<2x1x6x20xi32>, tensor<2x1x6x20xi32>) -> tensor<2x1x6x20xi1>
    %1208 = stablehlo.and %1207, %1202 : tensor<2x1x6x20xi1>
    %1209 = stablehlo.convert %1208 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %1210 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1211 = stablehlo.add %22, %15 : tensor<i32>
    %1212 = stablehlo.select %1210, %1211, %22 : tensor<i1>, tensor<i32>
    %1213 = stablehlo.dynamic_update_slice %185, %1195, %22, %1212, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %1214 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1215 = stablehlo.add %22, %15 : tensor<i32>
    %1216 = stablehlo.select %1214, %1215, %22 : tensor<i1>, tensor<i32>
    %1217 = stablehlo.dynamic_update_slice %186, %1196, %22, %1216, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %1218 = stablehlo.add %22, %14 : tensor<i32>
    %1219 = stablehlo.iota dim = 0 : tensor<20xi32>
    %1220 = stablehlo.add %22, %14 : tensor<i32>
    %1221 = stablehlo.broadcast_in_dim %1220, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %1222 = stablehlo.compare  LT, %1219, %1221,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %1223 = stablehlo.broadcast_in_dim %1222, dims = [3] : (tensor<20xi1>) -> tensor<2x1x6x20xi1>
    %1224 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1225 = stablehlo.compare  NE, %1209, %1224,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %1226 = stablehlo.and %1223, %1225 : tensor<2x1x6x20xi1>
    %1227 = stablehlo.convert %1226 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %1228 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1229 = stablehlo.compare  GT, %1227, %1228,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %1230 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1231 = stablehlo.convert %1230 : tensor<2x1x6x20xf32>
    %1232 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1233 = stablehlo.select %1229, %1231, %1232 : tensor<2x1x6x20xi1>, tensor<2x1x6x20xf32>
    %1234 = stablehlo.sqrt %12 : tensor<f32>
    %1235 = stablehlo.convert %1234 : tensor<f32>
    %1236 = stablehlo.broadcast_in_dim %1235, dims = [] : (tensor<f32>) -> tensor<2x6x12x64xf32>
    %1237 = stablehlo.divide %1194, %1236 : tensor<2x6x12x64xf32>
    %1238 = stablehlo.dot_general %1237, %1213, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x6x20xf32>
    %1239 = stablehlo.broadcast_in_dim %1233, dims = [0, 1, 2, 3] : (tensor<2x1x6x20xf32>) -> tensor<2x12x6x20xf32>
    %1240 = stablehlo.add %1238, %1239 : tensor<2x12x6x20xf32>
    %1241 = stablehlo.reduce(%1240 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %1242 = stablehlo.broadcast_in_dim %1241, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %1243 = stablehlo.broadcast_in_dim %1242, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %1244 = stablehlo.subtract %1240, %1243 : tensor<2x12x6x20xf32>
    %1245 = stablehlo.exponential %1244 : tensor<2x12x6x20xf32>
    %1246 = stablehlo.reduce(%1245 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %1247 = stablehlo.broadcast_in_dim %1246, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %1248 = stablehlo.broadcast_in_dim %1247, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %1249 = stablehlo.divide %1245, %1248 : tensor<2x12x6x20xf32>
    %1250 = stablehlo.dot_general %1217, %1249, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x6x20xf32>) -> tensor<2x12x64x6xf32>
    %1251 = stablehlo.transpose %1250, dims = [0, 3, 1, 2] : (tensor<2x12x64x6xf32>) -> tensor<2x6x12x64xf32>
    %1252 = stablehlo.reshape %1251 : (tensor<2x6x12x64xf32>) -> tensor<2x6x768xf32>
    %1253 = stablehlo.transpose %89, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %1254 = stablehlo.convert %1253 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1255 = stablehlo.dot_general %1252, %1254, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x768xf32>) -> tensor<2x6x768xf32>
    %1256 = stablehlo.convert %90 : (tensor<768xf16>) -> tensor<768xf32>
    %1257 = stablehlo.broadcast_in_dim %1256, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1258 = stablehlo.broadcast_in_dim %1257, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1259 = stablehlo.add %1255, %1258 : tensor<2x6x768xf32>
    %1260 = stablehlo.add %1259, %1147 : tensor<2x6x768xf32>
    %1261 = stablehlo.multiply %1260, %1260 : tensor<2x6x768xf32>
    %1262 = stablehlo.reduce(%1260 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1263 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1264 = stablehlo.divide %1262, %1263 : tensor<2x6xf32>
    %1265 = stablehlo.reduce(%1261 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1266 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1267 = stablehlo.divide %1265, %1266 : tensor<2x6xf32>
    %1268 = stablehlo.multiply %1264, %1264 : tensor<2x6xf32>
    %1269 = stablehlo.subtract %1267, %1268 : tensor<2x6xf32>
    %1270 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1271 = stablehlo.maximum %1270, %1269 : tensor<2x6xf32>
    %1272 = stablehlo.broadcast_in_dim %1264, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1273 = stablehlo.broadcast_in_dim %1271, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1274 = stablehlo.broadcast_in_dim %1272, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1275 = stablehlo.subtract %1260, %1274 : tensor<2x6x768xf32>
    %1276 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %1277 = stablehlo.add %1273, %1276 : tensor<2x6x1xf32>
    %1278 = stablehlo.rsqrt %1277 : tensor<2x6x1xf32>
    %1279 = stablehlo.reshape %91 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1280 = stablehlo.convert %1279 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1281 = stablehlo.broadcast_in_dim %1278, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1282 = stablehlo.broadcast_in_dim %1280, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1283 = stablehlo.multiply %1281, %1282 : tensor<2x6x768xf32>
    %1284 = stablehlo.multiply %1275, %1283 : tensor<2x6x768xf32>
    %1285 = stablehlo.reshape %92 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1286 = stablehlo.convert %1285 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1287 = stablehlo.broadcast_in_dim %1286, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1288 = stablehlo.add %1284, %1287 : tensor<2x6x768xf32>
    %1289 = stablehlo.transpose %93, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %1290 = stablehlo.convert %1289 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1291 = stablehlo.dot_general %1288, %1290, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x3072xf32>) -> tensor<2x6x3072xf32>
    %1292 = stablehlo.convert %94 : (tensor<3072xf16>) -> tensor<3072xf32>
    %1293 = stablehlo.broadcast_in_dim %1292, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %1294 = stablehlo.broadcast_in_dim %1293, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x6x3072xf32>
    %1295 = stablehlo.add %1291, %1294 : tensor<2x6x3072xf32>
    %1296 = stablehlo.multiply %1295, %1295 : tensor<2x6x3072xf32>
    %1297 = stablehlo.multiply %1295, %1296 : tensor<2x6x3072xf32>
    %1298 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1299 = stablehlo.multiply %1298, %1297 : tensor<2x6x3072xf32>
    %1300 = stablehlo.add %1295, %1299 : tensor<2x6x3072xf32>
    %1301 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1302 = stablehlo.multiply %1301, %1300 : tensor<2x6x3072xf32>
    %1303 = stablehlo.tanh %1302 : tensor<2x6x3072xf32>
    %1304 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1305 = stablehlo.add %1304, %1303 : tensor<2x6x3072xf32>
    %1306 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1307 = stablehlo.multiply %1306, %1305 : tensor<2x6x3072xf32>
    %1308 = stablehlo.multiply %1295, %1307 : tensor<2x6x3072xf32>
    %1309 = stablehlo.transpose %95, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %1310 = stablehlo.convert %1309 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1311 = stablehlo.dot_general %1308, %1310, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x3072xf32>, tensor<3072x768xf32>) -> tensor<2x6x768xf32>
    %1312 = stablehlo.convert %96 : (tensor<768xf16>) -> tensor<768xf32>
    %1313 = stablehlo.broadcast_in_dim %1312, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1314 = stablehlo.broadcast_in_dim %1313, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1315 = stablehlo.add %1311, %1314 : tensor<2x6x768xf32>
    %1316 = stablehlo.add %1260, %1315 : tensor<2x6x768xf32>
    %1317 = stablehlo.multiply %1316, %1316 : tensor<2x6x768xf32>
    %1318 = stablehlo.reduce(%1316 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1319 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1320 = stablehlo.divide %1318, %1319 : tensor<2x6xf32>
    %1321 = stablehlo.reduce(%1317 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1322 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1323 = stablehlo.divide %1321, %1322 : tensor<2x6xf32>
    %1324 = stablehlo.multiply %1320, %1320 : tensor<2x6xf32>
    %1325 = stablehlo.subtract %1323, %1324 : tensor<2x6xf32>
    %1326 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1327 = stablehlo.maximum %1326, %1325 : tensor<2x6xf32>
    %1328 = stablehlo.broadcast_in_dim %1320, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1329 = stablehlo.broadcast_in_dim %1327, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1330 = stablehlo.broadcast_in_dim %1328, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1331 = stablehlo.subtract %1316, %1330 : tensor<2x6x768xf32>
    %1332 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %1333 = stablehlo.add %1329, %1332 : tensor<2x6x1xf32>
    %1334 = stablehlo.rsqrt %1333 : tensor<2x6x1xf32>
    %1335 = stablehlo.reshape %97 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1336 = stablehlo.convert %1335 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1337 = stablehlo.broadcast_in_dim %1334, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1338 = stablehlo.broadcast_in_dim %1336, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1339 = stablehlo.multiply %1337, %1338 : tensor<2x6x768xf32>
    %1340 = stablehlo.multiply %1331, %1339 : tensor<2x6x768xf32>
    %1341 = stablehlo.reshape %98 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1342 = stablehlo.convert %1341 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1343 = stablehlo.broadcast_in_dim %1342, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1344 = stablehlo.add %1340, %1343 : tensor<2x6x768xf32>
    %1345 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %1346 = stablehlo.broadcast_in_dim %1345, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %1347 = stablehlo.broadcast_in_dim %1346, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %1348 = stablehlo.broadcast_in_dim %1346, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %1349 = stablehlo.broadcast_in_dim %1347, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %1350 = stablehlo.broadcast_in_dim %1348, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %1351 = stablehlo.compare  GE, %1349, %1350,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %1352 = stablehlo.broadcast_in_dim %1351, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %1353 = stablehlo.transpose %99, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %1354 = stablehlo.convert %1353 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %1355 = stablehlo.dot_general %1344, %1354, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x2304xf32>) -> tensor<2x6x2304xf32>
    %1356 = stablehlo.convert %100 : (tensor<2304xf16>) -> tensor<2304xf32>
    %1357 = stablehlo.broadcast_in_dim %1356, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %1358 = stablehlo.broadcast_in_dim %1357, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x6x2304xf32>
    %1359 = stablehlo.add %1355, %1358 : tensor<2x6x2304xf32>
    %1360 = stablehlo.slice %1359 [0:2, 0:6, 0:768] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1361 = stablehlo.slice %1359 [0:2, 0:6, 768:1536] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1362 = stablehlo.slice %1359 [0:2, 0:6, 1536:2304] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1363 = stablehlo.reshape %1360 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1364 = stablehlo.reshape %1361 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1365 = stablehlo.reshape %1362 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1366 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1367 = stablehlo.add %22, %16 : tensor<i32>
    %1368 = stablehlo.select %1366, %1367, %22 : tensor<i1>, tensor<i32>
    %1369 = stablehlo.dynamic_slice %1352, %22, %22, %1368, %22, sizes = [1, 1, 6, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x20xi1>
    %1370 = stablehlo.reshape %1369 : (tensor<1x1x6x20xi1>) -> tensor<1x6x20xi1>
    %1371 = stablehlo.broadcast_in_dim %1370, dims = [1, 2, 3] : (tensor<1x6x20xi1>) -> tensor<2x1x6x20xi1>
    %1372 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
    %1373 = stablehlo.reshape %1372 : (tensor<2x1x1x20xi32>) -> tensor<2x1x20xi32>
    %1374 = stablehlo.broadcast_in_dim %1373, dims = [0, 1, 3] : (tensor<2x1x20xi32>) -> tensor<2x1x6x20xi32>
    %1375 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x6x20xi32>
    %1376 = stablehlo.compare  NE, %1374, %1375,  SIGNED : (tensor<2x1x6x20xi32>, tensor<2x1x6x20xi32>) -> tensor<2x1x6x20xi1>
    %1377 = stablehlo.and %1376, %1371 : tensor<2x1x6x20xi1>
    %1378 = stablehlo.convert %1377 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %1379 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1380 = stablehlo.add %22, %15 : tensor<i32>
    %1381 = stablehlo.select %1379, %1380, %22 : tensor<i1>, tensor<i32>
    %1382 = stablehlo.dynamic_update_slice %187, %1364, %22, %1381, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %1383 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1384 = stablehlo.add %22, %15 : tensor<i32>
    %1385 = stablehlo.select %1383, %1384, %22 : tensor<i1>, tensor<i32>
    %1386 = stablehlo.dynamic_update_slice %188, %1365, %22, %1385, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %1387 = stablehlo.add %22, %14 : tensor<i32>
    %1388 = stablehlo.iota dim = 0 : tensor<20xi32>
    %1389 = stablehlo.add %22, %14 : tensor<i32>
    %1390 = stablehlo.broadcast_in_dim %1389, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %1391 = stablehlo.compare  LT, %1388, %1390,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %1392 = stablehlo.broadcast_in_dim %1391, dims = [3] : (tensor<20xi1>) -> tensor<2x1x6x20xi1>
    %1393 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1394 = stablehlo.compare  NE, %1378, %1393,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %1395 = stablehlo.and %1392, %1394 : tensor<2x1x6x20xi1>
    %1396 = stablehlo.convert %1395 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %1397 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1398 = stablehlo.compare  GT, %1396, %1397,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %1399 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1400 = stablehlo.convert %1399 : tensor<2x1x6x20xf32>
    %1401 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1402 = stablehlo.select %1398, %1400, %1401 : tensor<2x1x6x20xi1>, tensor<2x1x6x20xf32>
    %1403 = stablehlo.sqrt %12 : tensor<f32>
    %1404 = stablehlo.convert %1403 : tensor<f32>
    %1405 = stablehlo.broadcast_in_dim %1404, dims = [] : (tensor<f32>) -> tensor<2x6x12x64xf32>
    %1406 = stablehlo.divide %1363, %1405 : tensor<2x6x12x64xf32>
    %1407 = stablehlo.dot_general %1406, %1382, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x6x20xf32>
    %1408 = stablehlo.broadcast_in_dim %1402, dims = [0, 1, 2, 3] : (tensor<2x1x6x20xf32>) -> tensor<2x12x6x20xf32>
    %1409 = stablehlo.add %1407, %1408 : tensor<2x12x6x20xf32>
    %1410 = stablehlo.reduce(%1409 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %1411 = stablehlo.broadcast_in_dim %1410, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %1412 = stablehlo.broadcast_in_dim %1411, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %1413 = stablehlo.subtract %1409, %1412 : tensor<2x12x6x20xf32>
    %1414 = stablehlo.exponential %1413 : tensor<2x12x6x20xf32>
    %1415 = stablehlo.reduce(%1414 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %1416 = stablehlo.broadcast_in_dim %1415, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %1417 = stablehlo.broadcast_in_dim %1416, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %1418 = stablehlo.divide %1414, %1417 : tensor<2x12x6x20xf32>
    %1419 = stablehlo.dot_general %1386, %1418, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x6x20xf32>) -> tensor<2x12x64x6xf32>
    %1420 = stablehlo.transpose %1419, dims = [0, 3, 1, 2] : (tensor<2x12x64x6xf32>) -> tensor<2x6x12x64xf32>
    %1421 = stablehlo.reshape %1420 : (tensor<2x6x12x64xf32>) -> tensor<2x6x768xf32>
    %1422 = stablehlo.transpose %101, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %1423 = stablehlo.convert %1422 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1424 = stablehlo.dot_general %1421, %1423, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x768xf32>) -> tensor<2x6x768xf32>
    %1425 = stablehlo.convert %102 : (tensor<768xf16>) -> tensor<768xf32>
    %1426 = stablehlo.broadcast_in_dim %1425, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1427 = stablehlo.broadcast_in_dim %1426, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1428 = stablehlo.add %1424, %1427 : tensor<2x6x768xf32>
    %1429 = stablehlo.add %1428, %1316 : tensor<2x6x768xf32>
    %1430 = stablehlo.multiply %1429, %1429 : tensor<2x6x768xf32>
    %1431 = stablehlo.reduce(%1429 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1432 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1433 = stablehlo.divide %1431, %1432 : tensor<2x6xf32>
    %1434 = stablehlo.reduce(%1430 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1435 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1436 = stablehlo.divide %1434, %1435 : tensor<2x6xf32>
    %1437 = stablehlo.multiply %1433, %1433 : tensor<2x6xf32>
    %1438 = stablehlo.subtract %1436, %1437 : tensor<2x6xf32>
    %1439 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1440 = stablehlo.maximum %1439, %1438 : tensor<2x6xf32>
    %1441 = stablehlo.broadcast_in_dim %1433, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1442 = stablehlo.broadcast_in_dim %1440, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1443 = stablehlo.broadcast_in_dim %1441, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1444 = stablehlo.subtract %1429, %1443 : tensor<2x6x768xf32>
    %1445 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %1446 = stablehlo.add %1442, %1445 : tensor<2x6x1xf32>
    %1447 = stablehlo.rsqrt %1446 : tensor<2x6x1xf32>
    %1448 = stablehlo.reshape %103 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1449 = stablehlo.convert %1448 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1450 = stablehlo.broadcast_in_dim %1447, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1451 = stablehlo.broadcast_in_dim %1449, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1452 = stablehlo.multiply %1450, %1451 : tensor<2x6x768xf32>
    %1453 = stablehlo.multiply %1444, %1452 : tensor<2x6x768xf32>
    %1454 = stablehlo.reshape %104 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1455 = stablehlo.convert %1454 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1456 = stablehlo.broadcast_in_dim %1455, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1457 = stablehlo.add %1453, %1456 : tensor<2x6x768xf32>
    %1458 = stablehlo.transpose %105, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %1459 = stablehlo.convert %1458 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1460 = stablehlo.dot_general %1457, %1459, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x3072xf32>) -> tensor<2x6x3072xf32>
    %1461 = stablehlo.convert %106 : (tensor<3072xf16>) -> tensor<3072xf32>
    %1462 = stablehlo.broadcast_in_dim %1461, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %1463 = stablehlo.broadcast_in_dim %1462, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x6x3072xf32>
    %1464 = stablehlo.add %1460, %1463 : tensor<2x6x3072xf32>
    %1465 = stablehlo.multiply %1464, %1464 : tensor<2x6x3072xf32>
    %1466 = stablehlo.multiply %1464, %1465 : tensor<2x6x3072xf32>
    %1467 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1468 = stablehlo.multiply %1467, %1466 : tensor<2x6x3072xf32>
    %1469 = stablehlo.add %1464, %1468 : tensor<2x6x3072xf32>
    %1470 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1471 = stablehlo.multiply %1470, %1469 : tensor<2x6x3072xf32>
    %1472 = stablehlo.tanh %1471 : tensor<2x6x3072xf32>
    %1473 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1474 = stablehlo.add %1473, %1472 : tensor<2x6x3072xf32>
    %1475 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1476 = stablehlo.multiply %1475, %1474 : tensor<2x6x3072xf32>
    %1477 = stablehlo.multiply %1464, %1476 : tensor<2x6x3072xf32>
    %1478 = stablehlo.transpose %107, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %1479 = stablehlo.convert %1478 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1480 = stablehlo.dot_general %1477, %1479, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x3072xf32>, tensor<3072x768xf32>) -> tensor<2x6x768xf32>
    %1481 = stablehlo.convert %108 : (tensor<768xf16>) -> tensor<768xf32>
    %1482 = stablehlo.broadcast_in_dim %1481, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1483 = stablehlo.broadcast_in_dim %1482, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1484 = stablehlo.add %1480, %1483 : tensor<2x6x768xf32>
    %1485 = stablehlo.add %1429, %1484 : tensor<2x6x768xf32>
    %1486 = stablehlo.multiply %1485, %1485 : tensor<2x6x768xf32>
    %1487 = stablehlo.reduce(%1485 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1488 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1489 = stablehlo.divide %1487, %1488 : tensor<2x6xf32>
    %1490 = stablehlo.reduce(%1486 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1491 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1492 = stablehlo.divide %1490, %1491 : tensor<2x6xf32>
    %1493 = stablehlo.multiply %1489, %1489 : tensor<2x6xf32>
    %1494 = stablehlo.subtract %1492, %1493 : tensor<2x6xf32>
    %1495 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1496 = stablehlo.maximum %1495, %1494 : tensor<2x6xf32>
    %1497 = stablehlo.broadcast_in_dim %1489, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1498 = stablehlo.broadcast_in_dim %1496, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1499 = stablehlo.broadcast_in_dim %1497, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1500 = stablehlo.subtract %1485, %1499 : tensor<2x6x768xf32>
    %1501 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %1502 = stablehlo.add %1498, %1501 : tensor<2x6x1xf32>
    %1503 = stablehlo.rsqrt %1502 : tensor<2x6x1xf32>
    %1504 = stablehlo.reshape %109 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1505 = stablehlo.convert %1504 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1506 = stablehlo.broadcast_in_dim %1503, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1507 = stablehlo.broadcast_in_dim %1505, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1508 = stablehlo.multiply %1506, %1507 : tensor<2x6x768xf32>
    %1509 = stablehlo.multiply %1500, %1508 : tensor<2x6x768xf32>
    %1510 = stablehlo.reshape %110 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1511 = stablehlo.convert %1510 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1512 = stablehlo.broadcast_in_dim %1511, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1513 = stablehlo.add %1509, %1512 : tensor<2x6x768xf32>
    %1514 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %1515 = stablehlo.broadcast_in_dim %1514, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %1516 = stablehlo.broadcast_in_dim %1515, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %1517 = stablehlo.broadcast_in_dim %1515, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %1518 = stablehlo.broadcast_in_dim %1516, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %1519 = stablehlo.broadcast_in_dim %1517, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %1520 = stablehlo.compare  GE, %1518, %1519,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %1521 = stablehlo.broadcast_in_dim %1520, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %1522 = stablehlo.transpose %111, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %1523 = stablehlo.convert %1522 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %1524 = stablehlo.dot_general %1513, %1523, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x2304xf32>) -> tensor<2x6x2304xf32>
    %1525 = stablehlo.convert %112 : (tensor<2304xf16>) -> tensor<2304xf32>
    %1526 = stablehlo.broadcast_in_dim %1525, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %1527 = stablehlo.broadcast_in_dim %1526, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x6x2304xf32>
    %1528 = stablehlo.add %1524, %1527 : tensor<2x6x2304xf32>
    %1529 = stablehlo.slice %1528 [0:2, 0:6, 0:768] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1530 = stablehlo.slice %1528 [0:2, 0:6, 768:1536] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1531 = stablehlo.slice %1528 [0:2, 0:6, 1536:2304] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1532 = stablehlo.reshape %1529 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1533 = stablehlo.reshape %1530 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1534 = stablehlo.reshape %1531 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1535 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1536 = stablehlo.add %22, %16 : tensor<i32>
    %1537 = stablehlo.select %1535, %1536, %22 : tensor<i1>, tensor<i32>
    %1538 = stablehlo.dynamic_slice %1521, %22, %22, %1537, %22, sizes = [1, 1, 6, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x20xi1>
    %1539 = stablehlo.reshape %1538 : (tensor<1x1x6x20xi1>) -> tensor<1x6x20xi1>
    %1540 = stablehlo.broadcast_in_dim %1539, dims = [1, 2, 3] : (tensor<1x6x20xi1>) -> tensor<2x1x6x20xi1>
    %1541 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
    %1542 = stablehlo.reshape %1541 : (tensor<2x1x1x20xi32>) -> tensor<2x1x20xi32>
    %1543 = stablehlo.broadcast_in_dim %1542, dims = [0, 1, 3] : (tensor<2x1x20xi32>) -> tensor<2x1x6x20xi32>
    %1544 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x6x20xi32>
    %1545 = stablehlo.compare  NE, %1543, %1544,  SIGNED : (tensor<2x1x6x20xi32>, tensor<2x1x6x20xi32>) -> tensor<2x1x6x20xi1>
    %1546 = stablehlo.and %1545, %1540 : tensor<2x1x6x20xi1>
    %1547 = stablehlo.convert %1546 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %1548 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1549 = stablehlo.add %22, %15 : tensor<i32>
    %1550 = stablehlo.select %1548, %1549, %22 : tensor<i1>, tensor<i32>
    %1551 = stablehlo.dynamic_update_slice %189, %1533, %22, %1550, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %1552 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1553 = stablehlo.add %22, %15 : tensor<i32>
    %1554 = stablehlo.select %1552, %1553, %22 : tensor<i1>, tensor<i32>
    %1555 = stablehlo.dynamic_update_slice %190, %1534, %22, %1554, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %1556 = stablehlo.add %22, %14 : tensor<i32>
    %1557 = stablehlo.iota dim = 0 : tensor<20xi32>
    %1558 = stablehlo.add %22, %14 : tensor<i32>
    %1559 = stablehlo.broadcast_in_dim %1558, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %1560 = stablehlo.compare  LT, %1557, %1559,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %1561 = stablehlo.broadcast_in_dim %1560, dims = [3] : (tensor<20xi1>) -> tensor<2x1x6x20xi1>
    %1562 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1563 = stablehlo.compare  NE, %1547, %1562,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %1564 = stablehlo.and %1561, %1563 : tensor<2x1x6x20xi1>
    %1565 = stablehlo.convert %1564 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %1566 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1567 = stablehlo.compare  GT, %1565, %1566,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %1568 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1569 = stablehlo.convert %1568 : tensor<2x1x6x20xf32>
    %1570 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1571 = stablehlo.select %1567, %1569, %1570 : tensor<2x1x6x20xi1>, tensor<2x1x6x20xf32>
    %1572 = stablehlo.sqrt %12 : tensor<f32>
    %1573 = stablehlo.convert %1572 : tensor<f32>
    %1574 = stablehlo.broadcast_in_dim %1573, dims = [] : (tensor<f32>) -> tensor<2x6x12x64xf32>
    %1575 = stablehlo.divide %1532, %1574 : tensor<2x6x12x64xf32>
    %1576 = stablehlo.dot_general %1575, %1551, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x6x20xf32>
    %1577 = stablehlo.broadcast_in_dim %1571, dims = [0, 1, 2, 3] : (tensor<2x1x6x20xf32>) -> tensor<2x12x6x20xf32>
    %1578 = stablehlo.add %1576, %1577 : tensor<2x12x6x20xf32>
    %1579 = stablehlo.reduce(%1578 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %1580 = stablehlo.broadcast_in_dim %1579, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %1581 = stablehlo.broadcast_in_dim %1580, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %1582 = stablehlo.subtract %1578, %1581 : tensor<2x12x6x20xf32>
    %1583 = stablehlo.exponential %1582 : tensor<2x12x6x20xf32>
    %1584 = stablehlo.reduce(%1583 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %1585 = stablehlo.broadcast_in_dim %1584, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %1586 = stablehlo.broadcast_in_dim %1585, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %1587 = stablehlo.divide %1583, %1586 : tensor<2x12x6x20xf32>
    %1588 = stablehlo.dot_general %1555, %1587, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x6x20xf32>) -> tensor<2x12x64x6xf32>
    %1589 = stablehlo.transpose %1588, dims = [0, 3, 1, 2] : (tensor<2x12x64x6xf32>) -> tensor<2x6x12x64xf32>
    %1590 = stablehlo.reshape %1589 : (tensor<2x6x12x64xf32>) -> tensor<2x6x768xf32>
    %1591 = stablehlo.transpose %113, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %1592 = stablehlo.convert %1591 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1593 = stablehlo.dot_general %1590, %1592, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x768xf32>) -> tensor<2x6x768xf32>
    %1594 = stablehlo.convert %114 : (tensor<768xf16>) -> tensor<768xf32>
    %1595 = stablehlo.broadcast_in_dim %1594, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1596 = stablehlo.broadcast_in_dim %1595, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1597 = stablehlo.add %1593, %1596 : tensor<2x6x768xf32>
    %1598 = stablehlo.add %1597, %1485 : tensor<2x6x768xf32>
    %1599 = stablehlo.multiply %1598, %1598 : tensor<2x6x768xf32>
    %1600 = stablehlo.reduce(%1598 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1601 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1602 = stablehlo.divide %1600, %1601 : tensor<2x6xf32>
    %1603 = stablehlo.reduce(%1599 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1604 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1605 = stablehlo.divide %1603, %1604 : tensor<2x6xf32>
    %1606 = stablehlo.multiply %1602, %1602 : tensor<2x6xf32>
    %1607 = stablehlo.subtract %1605, %1606 : tensor<2x6xf32>
    %1608 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1609 = stablehlo.maximum %1608, %1607 : tensor<2x6xf32>
    %1610 = stablehlo.broadcast_in_dim %1602, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1611 = stablehlo.broadcast_in_dim %1609, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1612 = stablehlo.broadcast_in_dim %1610, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1613 = stablehlo.subtract %1598, %1612 : tensor<2x6x768xf32>
    %1614 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %1615 = stablehlo.add %1611, %1614 : tensor<2x6x1xf32>
    %1616 = stablehlo.rsqrt %1615 : tensor<2x6x1xf32>
    %1617 = stablehlo.reshape %115 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1618 = stablehlo.convert %1617 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1619 = stablehlo.broadcast_in_dim %1616, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1620 = stablehlo.broadcast_in_dim %1618, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1621 = stablehlo.multiply %1619, %1620 : tensor<2x6x768xf32>
    %1622 = stablehlo.multiply %1613, %1621 : tensor<2x6x768xf32>
    %1623 = stablehlo.reshape %116 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1624 = stablehlo.convert %1623 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1625 = stablehlo.broadcast_in_dim %1624, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1626 = stablehlo.add %1622, %1625 : tensor<2x6x768xf32>
    %1627 = stablehlo.transpose %117, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %1628 = stablehlo.convert %1627 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1629 = stablehlo.dot_general %1626, %1628, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x3072xf32>) -> tensor<2x6x3072xf32>
    %1630 = stablehlo.convert %118 : (tensor<3072xf16>) -> tensor<3072xf32>
    %1631 = stablehlo.broadcast_in_dim %1630, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %1632 = stablehlo.broadcast_in_dim %1631, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x6x3072xf32>
    %1633 = stablehlo.add %1629, %1632 : tensor<2x6x3072xf32>
    %1634 = stablehlo.multiply %1633, %1633 : tensor<2x6x3072xf32>
    %1635 = stablehlo.multiply %1633, %1634 : tensor<2x6x3072xf32>
    %1636 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1637 = stablehlo.multiply %1636, %1635 : tensor<2x6x3072xf32>
    %1638 = stablehlo.add %1633, %1637 : tensor<2x6x3072xf32>
    %1639 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1640 = stablehlo.multiply %1639, %1638 : tensor<2x6x3072xf32>
    %1641 = stablehlo.tanh %1640 : tensor<2x6x3072xf32>
    %1642 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1643 = stablehlo.add %1642, %1641 : tensor<2x6x3072xf32>
    %1644 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1645 = stablehlo.multiply %1644, %1643 : tensor<2x6x3072xf32>
    %1646 = stablehlo.multiply %1633, %1645 : tensor<2x6x3072xf32>
    %1647 = stablehlo.transpose %119, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %1648 = stablehlo.convert %1647 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1649 = stablehlo.dot_general %1646, %1648, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x3072xf32>, tensor<3072x768xf32>) -> tensor<2x6x768xf32>
    %1650 = stablehlo.convert %120 : (tensor<768xf16>) -> tensor<768xf32>
    %1651 = stablehlo.broadcast_in_dim %1650, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1652 = stablehlo.broadcast_in_dim %1651, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1653 = stablehlo.add %1649, %1652 : tensor<2x6x768xf32>
    %1654 = stablehlo.add %1598, %1653 : tensor<2x6x768xf32>
    %1655 = stablehlo.multiply %1654, %1654 : tensor<2x6x768xf32>
    %1656 = stablehlo.reduce(%1654 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1657 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1658 = stablehlo.divide %1656, %1657 : tensor<2x6xf32>
    %1659 = stablehlo.reduce(%1655 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1660 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1661 = stablehlo.divide %1659, %1660 : tensor<2x6xf32>
    %1662 = stablehlo.multiply %1658, %1658 : tensor<2x6xf32>
    %1663 = stablehlo.subtract %1661, %1662 : tensor<2x6xf32>
    %1664 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1665 = stablehlo.maximum %1664, %1663 : tensor<2x6xf32>
    %1666 = stablehlo.broadcast_in_dim %1658, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1667 = stablehlo.broadcast_in_dim %1665, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1668 = stablehlo.broadcast_in_dim %1666, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1669 = stablehlo.subtract %1654, %1668 : tensor<2x6x768xf32>
    %1670 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %1671 = stablehlo.add %1667, %1670 : tensor<2x6x1xf32>
    %1672 = stablehlo.rsqrt %1671 : tensor<2x6x1xf32>
    %1673 = stablehlo.reshape %121 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1674 = stablehlo.convert %1673 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1675 = stablehlo.broadcast_in_dim %1672, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1676 = stablehlo.broadcast_in_dim %1674, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1677 = stablehlo.multiply %1675, %1676 : tensor<2x6x768xf32>
    %1678 = stablehlo.multiply %1669, %1677 : tensor<2x6x768xf32>
    %1679 = stablehlo.reshape %122 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1680 = stablehlo.convert %1679 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1681 = stablehlo.broadcast_in_dim %1680, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1682 = stablehlo.add %1678, %1681 : tensor<2x6x768xf32>
    %1683 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %1684 = stablehlo.broadcast_in_dim %1683, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %1685 = stablehlo.broadcast_in_dim %1684, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %1686 = stablehlo.broadcast_in_dim %1684, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %1687 = stablehlo.broadcast_in_dim %1685, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %1688 = stablehlo.broadcast_in_dim %1686, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %1689 = stablehlo.compare  GE, %1687, %1688,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %1690 = stablehlo.broadcast_in_dim %1689, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %1691 = stablehlo.transpose %123, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %1692 = stablehlo.convert %1691 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %1693 = stablehlo.dot_general %1682, %1692, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x2304xf32>) -> tensor<2x6x2304xf32>
    %1694 = stablehlo.convert %124 : (tensor<2304xf16>) -> tensor<2304xf32>
    %1695 = stablehlo.broadcast_in_dim %1694, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %1696 = stablehlo.broadcast_in_dim %1695, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x6x2304xf32>
    %1697 = stablehlo.add %1693, %1696 : tensor<2x6x2304xf32>
    %1698 = stablehlo.slice %1697 [0:2, 0:6, 0:768] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1699 = stablehlo.slice %1697 [0:2, 0:6, 768:1536] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1700 = stablehlo.slice %1697 [0:2, 0:6, 1536:2304] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1701 = stablehlo.reshape %1698 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1702 = stablehlo.reshape %1699 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1703 = stablehlo.reshape %1700 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1704 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1705 = stablehlo.add %22, %16 : tensor<i32>
    %1706 = stablehlo.select %1704, %1705, %22 : tensor<i1>, tensor<i32>
    %1707 = stablehlo.dynamic_slice %1690, %22, %22, %1706, %22, sizes = [1, 1, 6, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x20xi1>
    %1708 = stablehlo.reshape %1707 : (tensor<1x1x6x20xi1>) -> tensor<1x6x20xi1>
    %1709 = stablehlo.broadcast_in_dim %1708, dims = [1, 2, 3] : (tensor<1x6x20xi1>) -> tensor<2x1x6x20xi1>
    %1710 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
    %1711 = stablehlo.reshape %1710 : (tensor<2x1x1x20xi32>) -> tensor<2x1x20xi32>
    %1712 = stablehlo.broadcast_in_dim %1711, dims = [0, 1, 3] : (tensor<2x1x20xi32>) -> tensor<2x1x6x20xi32>
    %1713 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x6x20xi32>
    %1714 = stablehlo.compare  NE, %1712, %1713,  SIGNED : (tensor<2x1x6x20xi32>, tensor<2x1x6x20xi32>) -> tensor<2x1x6x20xi1>
    %1715 = stablehlo.and %1714, %1709 : tensor<2x1x6x20xi1>
    %1716 = stablehlo.convert %1715 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %1717 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1718 = stablehlo.add %22, %15 : tensor<i32>
    %1719 = stablehlo.select %1717, %1718, %22 : tensor<i1>, tensor<i32>
    %1720 = stablehlo.dynamic_update_slice %191, %1702, %22, %1719, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %1721 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1722 = stablehlo.add %22, %15 : tensor<i32>
    %1723 = stablehlo.select %1721, %1722, %22 : tensor<i1>, tensor<i32>
    %1724 = stablehlo.dynamic_update_slice %192, %1703, %22, %1723, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %1725 = stablehlo.add %22, %14 : tensor<i32>
    %1726 = stablehlo.iota dim = 0 : tensor<20xi32>
    %1727 = stablehlo.add %22, %14 : tensor<i32>
    %1728 = stablehlo.broadcast_in_dim %1727, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %1729 = stablehlo.compare  LT, %1726, %1728,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %1730 = stablehlo.broadcast_in_dim %1729, dims = [3] : (tensor<20xi1>) -> tensor<2x1x6x20xi1>
    %1731 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1732 = stablehlo.compare  NE, %1716, %1731,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %1733 = stablehlo.and %1730, %1732 : tensor<2x1x6x20xi1>
    %1734 = stablehlo.convert %1733 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %1735 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1736 = stablehlo.compare  GT, %1734, %1735,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %1737 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1738 = stablehlo.convert %1737 : tensor<2x1x6x20xf32>
    %1739 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1740 = stablehlo.select %1736, %1738, %1739 : tensor<2x1x6x20xi1>, tensor<2x1x6x20xf32>
    %1741 = stablehlo.sqrt %12 : tensor<f32>
    %1742 = stablehlo.convert %1741 : tensor<f32>
    %1743 = stablehlo.broadcast_in_dim %1742, dims = [] : (tensor<f32>) -> tensor<2x6x12x64xf32>
    %1744 = stablehlo.divide %1701, %1743 : tensor<2x6x12x64xf32>
    %1745 = stablehlo.dot_general %1744, %1720, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x6x20xf32>
    %1746 = stablehlo.broadcast_in_dim %1740, dims = [0, 1, 2, 3] : (tensor<2x1x6x20xf32>) -> tensor<2x12x6x20xf32>
    %1747 = stablehlo.add %1745, %1746 : tensor<2x12x6x20xf32>
    %1748 = stablehlo.reduce(%1747 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %1749 = stablehlo.broadcast_in_dim %1748, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %1750 = stablehlo.broadcast_in_dim %1749, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %1751 = stablehlo.subtract %1747, %1750 : tensor<2x12x6x20xf32>
    %1752 = stablehlo.exponential %1751 : tensor<2x12x6x20xf32>
    %1753 = stablehlo.reduce(%1752 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %1754 = stablehlo.broadcast_in_dim %1753, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %1755 = stablehlo.broadcast_in_dim %1754, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %1756 = stablehlo.divide %1752, %1755 : tensor<2x12x6x20xf32>
    %1757 = stablehlo.dot_general %1724, %1756, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x6x20xf32>) -> tensor<2x12x64x6xf32>
    %1758 = stablehlo.transpose %1757, dims = [0, 3, 1, 2] : (tensor<2x12x64x6xf32>) -> tensor<2x6x12x64xf32>
    %1759 = stablehlo.reshape %1758 : (tensor<2x6x12x64xf32>) -> tensor<2x6x768xf32>
    %1760 = stablehlo.transpose %125, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %1761 = stablehlo.convert %1760 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1762 = stablehlo.dot_general %1759, %1761, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x768xf32>) -> tensor<2x6x768xf32>
    %1763 = stablehlo.convert %126 : (tensor<768xf16>) -> tensor<768xf32>
    %1764 = stablehlo.broadcast_in_dim %1763, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1765 = stablehlo.broadcast_in_dim %1764, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1766 = stablehlo.add %1762, %1765 : tensor<2x6x768xf32>
    %1767 = stablehlo.add %1766, %1654 : tensor<2x6x768xf32>
    %1768 = stablehlo.multiply %1767, %1767 : tensor<2x6x768xf32>
    %1769 = stablehlo.reduce(%1767 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1770 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1771 = stablehlo.divide %1769, %1770 : tensor<2x6xf32>
    %1772 = stablehlo.reduce(%1768 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1773 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1774 = stablehlo.divide %1772, %1773 : tensor<2x6xf32>
    %1775 = stablehlo.multiply %1771, %1771 : tensor<2x6xf32>
    %1776 = stablehlo.subtract %1774, %1775 : tensor<2x6xf32>
    %1777 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1778 = stablehlo.maximum %1777, %1776 : tensor<2x6xf32>
    %1779 = stablehlo.broadcast_in_dim %1771, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1780 = stablehlo.broadcast_in_dim %1778, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1781 = stablehlo.broadcast_in_dim %1779, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1782 = stablehlo.subtract %1767, %1781 : tensor<2x6x768xf32>
    %1783 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %1784 = stablehlo.add %1780, %1783 : tensor<2x6x1xf32>
    %1785 = stablehlo.rsqrt %1784 : tensor<2x6x1xf32>
    %1786 = stablehlo.reshape %127 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1787 = stablehlo.convert %1786 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1788 = stablehlo.broadcast_in_dim %1785, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1789 = stablehlo.broadcast_in_dim %1787, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1790 = stablehlo.multiply %1788, %1789 : tensor<2x6x768xf32>
    %1791 = stablehlo.multiply %1782, %1790 : tensor<2x6x768xf32>
    %1792 = stablehlo.reshape %128 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1793 = stablehlo.convert %1792 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1794 = stablehlo.broadcast_in_dim %1793, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1795 = stablehlo.add %1791, %1794 : tensor<2x6x768xf32>
    %1796 = stablehlo.transpose %129, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %1797 = stablehlo.convert %1796 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1798 = stablehlo.dot_general %1795, %1797, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x3072xf32>) -> tensor<2x6x3072xf32>
    %1799 = stablehlo.convert %130 : (tensor<3072xf16>) -> tensor<3072xf32>
    %1800 = stablehlo.broadcast_in_dim %1799, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %1801 = stablehlo.broadcast_in_dim %1800, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x6x3072xf32>
    %1802 = stablehlo.add %1798, %1801 : tensor<2x6x3072xf32>
    %1803 = stablehlo.multiply %1802, %1802 : tensor<2x6x3072xf32>
    %1804 = stablehlo.multiply %1802, %1803 : tensor<2x6x3072xf32>
    %1805 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1806 = stablehlo.multiply %1805, %1804 : tensor<2x6x3072xf32>
    %1807 = stablehlo.add %1802, %1806 : tensor<2x6x3072xf32>
    %1808 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1809 = stablehlo.multiply %1808, %1807 : tensor<2x6x3072xf32>
    %1810 = stablehlo.tanh %1809 : tensor<2x6x3072xf32>
    %1811 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1812 = stablehlo.add %1811, %1810 : tensor<2x6x3072xf32>
    %1813 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1814 = stablehlo.multiply %1813, %1812 : tensor<2x6x3072xf32>
    %1815 = stablehlo.multiply %1802, %1814 : tensor<2x6x3072xf32>
    %1816 = stablehlo.transpose %131, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %1817 = stablehlo.convert %1816 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1818 = stablehlo.dot_general %1815, %1817, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x3072xf32>, tensor<3072x768xf32>) -> tensor<2x6x768xf32>
    %1819 = stablehlo.convert %132 : (tensor<768xf16>) -> tensor<768xf32>
    %1820 = stablehlo.broadcast_in_dim %1819, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1821 = stablehlo.broadcast_in_dim %1820, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1822 = stablehlo.add %1818, %1821 : tensor<2x6x768xf32>
    %1823 = stablehlo.add %1767, %1822 : tensor<2x6x768xf32>
    %1824 = stablehlo.multiply %1823, %1823 : tensor<2x6x768xf32>
    %1825 = stablehlo.reduce(%1823 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1826 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1827 = stablehlo.divide %1825, %1826 : tensor<2x6xf32>
    %1828 = stablehlo.reduce(%1824 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1829 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1830 = stablehlo.divide %1828, %1829 : tensor<2x6xf32>
    %1831 = stablehlo.multiply %1827, %1827 : tensor<2x6xf32>
    %1832 = stablehlo.subtract %1830, %1831 : tensor<2x6xf32>
    %1833 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1834 = stablehlo.maximum %1833, %1832 : tensor<2x6xf32>
    %1835 = stablehlo.broadcast_in_dim %1827, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1836 = stablehlo.broadcast_in_dim %1834, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1837 = stablehlo.broadcast_in_dim %1835, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1838 = stablehlo.subtract %1823, %1837 : tensor<2x6x768xf32>
    %1839 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %1840 = stablehlo.add %1836, %1839 : tensor<2x6x1xf32>
    %1841 = stablehlo.rsqrt %1840 : tensor<2x6x1xf32>
    %1842 = stablehlo.reshape %133 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1843 = stablehlo.convert %1842 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1844 = stablehlo.broadcast_in_dim %1841, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1845 = stablehlo.broadcast_in_dim %1843, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1846 = stablehlo.multiply %1844, %1845 : tensor<2x6x768xf32>
    %1847 = stablehlo.multiply %1838, %1846 : tensor<2x6x768xf32>
    %1848 = stablehlo.reshape %134 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1849 = stablehlo.convert %1848 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1850 = stablehlo.broadcast_in_dim %1849, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1851 = stablehlo.add %1847, %1850 : tensor<2x6x768xf32>
    %1852 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %1853 = stablehlo.broadcast_in_dim %1852, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %1854 = stablehlo.broadcast_in_dim %1853, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %1855 = stablehlo.broadcast_in_dim %1853, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %1856 = stablehlo.broadcast_in_dim %1854, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %1857 = stablehlo.broadcast_in_dim %1855, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %1858 = stablehlo.compare  GE, %1856, %1857,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %1859 = stablehlo.broadcast_in_dim %1858, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %1860 = stablehlo.transpose %135, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %1861 = stablehlo.convert %1860 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %1862 = stablehlo.dot_general %1851, %1861, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x2304xf32>) -> tensor<2x6x2304xf32>
    %1863 = stablehlo.convert %136 : (tensor<2304xf16>) -> tensor<2304xf32>
    %1864 = stablehlo.broadcast_in_dim %1863, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %1865 = stablehlo.broadcast_in_dim %1864, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x6x2304xf32>
    %1866 = stablehlo.add %1862, %1865 : tensor<2x6x2304xf32>
    %1867 = stablehlo.slice %1866 [0:2, 0:6, 0:768] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1868 = stablehlo.slice %1866 [0:2, 0:6, 768:1536] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1869 = stablehlo.slice %1866 [0:2, 0:6, 1536:2304] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %1870 = stablehlo.reshape %1867 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1871 = stablehlo.reshape %1868 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1872 = stablehlo.reshape %1869 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %1873 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1874 = stablehlo.add %22, %16 : tensor<i32>
    %1875 = stablehlo.select %1873, %1874, %22 : tensor<i1>, tensor<i32>
    %1876 = stablehlo.dynamic_slice %1859, %22, %22, %1875, %22, sizes = [1, 1, 6, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x20xi1>
    %1877 = stablehlo.reshape %1876 : (tensor<1x1x6x20xi1>) -> tensor<1x6x20xi1>
    %1878 = stablehlo.broadcast_in_dim %1877, dims = [1, 2, 3] : (tensor<1x6x20xi1>) -> tensor<2x1x6x20xi1>
    %1879 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
    %1880 = stablehlo.reshape %1879 : (tensor<2x1x1x20xi32>) -> tensor<2x1x20xi32>
    %1881 = stablehlo.broadcast_in_dim %1880, dims = [0, 1, 3] : (tensor<2x1x20xi32>) -> tensor<2x1x6x20xi32>
    %1882 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x6x20xi32>
    %1883 = stablehlo.compare  NE, %1881, %1882,  SIGNED : (tensor<2x1x6x20xi32>, tensor<2x1x6x20xi32>) -> tensor<2x1x6x20xi1>
    %1884 = stablehlo.and %1883, %1878 : tensor<2x1x6x20xi1>
    %1885 = stablehlo.convert %1884 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %1886 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1887 = stablehlo.add %22, %15 : tensor<i32>
    %1888 = stablehlo.select %1886, %1887, %22 : tensor<i1>, tensor<i32>
    %1889 = stablehlo.dynamic_update_slice %193, %1871, %22, %1888, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %1890 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1891 = stablehlo.add %22, %15 : tensor<i32>
    %1892 = stablehlo.select %1890, %1891, %22 : tensor<i1>, tensor<i32>
    %1893 = stablehlo.dynamic_update_slice %194, %1872, %22, %1892, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %1894 = stablehlo.add %22, %14 : tensor<i32>
    %1895 = stablehlo.iota dim = 0 : tensor<20xi32>
    %1896 = stablehlo.add %22, %14 : tensor<i32>
    %1897 = stablehlo.broadcast_in_dim %1896, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %1898 = stablehlo.compare  LT, %1895, %1897,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %1899 = stablehlo.broadcast_in_dim %1898, dims = [3] : (tensor<20xi1>) -> tensor<2x1x6x20xi1>
    %1900 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1901 = stablehlo.compare  NE, %1885, %1900,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %1902 = stablehlo.and %1899, %1901 : tensor<2x1x6x20xi1>
    %1903 = stablehlo.convert %1902 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %1904 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1905 = stablehlo.compare  GT, %1903, %1904,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %1906 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1907 = stablehlo.convert %1906 : tensor<2x1x6x20xf32>
    %1908 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %1909 = stablehlo.select %1905, %1907, %1908 : tensor<2x1x6x20xi1>, tensor<2x1x6x20xf32>
    %1910 = stablehlo.sqrt %12 : tensor<f32>
    %1911 = stablehlo.convert %1910 : tensor<f32>
    %1912 = stablehlo.broadcast_in_dim %1911, dims = [] : (tensor<f32>) -> tensor<2x6x12x64xf32>
    %1913 = stablehlo.divide %1870, %1912 : tensor<2x6x12x64xf32>
    %1914 = stablehlo.dot_general %1913, %1889, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x6x20xf32>
    %1915 = stablehlo.broadcast_in_dim %1909, dims = [0, 1, 2, 3] : (tensor<2x1x6x20xf32>) -> tensor<2x12x6x20xf32>
    %1916 = stablehlo.add %1914, %1915 : tensor<2x12x6x20xf32>
    %1917 = stablehlo.reduce(%1916 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %1918 = stablehlo.broadcast_in_dim %1917, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %1919 = stablehlo.broadcast_in_dim %1918, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %1920 = stablehlo.subtract %1916, %1919 : tensor<2x12x6x20xf32>
    %1921 = stablehlo.exponential %1920 : tensor<2x12x6x20xf32>
    %1922 = stablehlo.reduce(%1921 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %1923 = stablehlo.broadcast_in_dim %1922, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %1924 = stablehlo.broadcast_in_dim %1923, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %1925 = stablehlo.divide %1921, %1924 : tensor<2x12x6x20xf32>
    %1926 = stablehlo.dot_general %1893, %1925, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x6x20xf32>) -> tensor<2x12x64x6xf32>
    %1927 = stablehlo.transpose %1926, dims = [0, 3, 1, 2] : (tensor<2x12x64x6xf32>) -> tensor<2x6x12x64xf32>
    %1928 = stablehlo.reshape %1927 : (tensor<2x6x12x64xf32>) -> tensor<2x6x768xf32>
    %1929 = stablehlo.transpose %137, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %1930 = stablehlo.convert %1929 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %1931 = stablehlo.dot_general %1928, %1930, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x768xf32>) -> tensor<2x6x768xf32>
    %1932 = stablehlo.convert %138 : (tensor<768xf16>) -> tensor<768xf32>
    %1933 = stablehlo.broadcast_in_dim %1932, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1934 = stablehlo.broadcast_in_dim %1933, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1935 = stablehlo.add %1931, %1934 : tensor<2x6x768xf32>
    %1936 = stablehlo.add %1935, %1823 : tensor<2x6x768xf32>
    %1937 = stablehlo.multiply %1936, %1936 : tensor<2x6x768xf32>
    %1938 = stablehlo.reduce(%1936 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1939 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1940 = stablehlo.divide %1938, %1939 : tensor<2x6xf32>
    %1941 = stablehlo.reduce(%1937 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1942 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1943 = stablehlo.divide %1941, %1942 : tensor<2x6xf32>
    %1944 = stablehlo.multiply %1940, %1940 : tensor<2x6xf32>
    %1945 = stablehlo.subtract %1943, %1944 : tensor<2x6xf32>
    %1946 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1947 = stablehlo.maximum %1946, %1945 : tensor<2x6xf32>
    %1948 = stablehlo.broadcast_in_dim %1940, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1949 = stablehlo.broadcast_in_dim %1947, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %1950 = stablehlo.broadcast_in_dim %1948, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1951 = stablehlo.subtract %1936, %1950 : tensor<2x6x768xf32>
    %1952 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %1953 = stablehlo.add %1949, %1952 : tensor<2x6x1xf32>
    %1954 = stablehlo.rsqrt %1953 : tensor<2x6x1xf32>
    %1955 = stablehlo.reshape %139 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1956 = stablehlo.convert %1955 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1957 = stablehlo.broadcast_in_dim %1954, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %1958 = stablehlo.broadcast_in_dim %1956, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1959 = stablehlo.multiply %1957, %1958 : tensor<2x6x768xf32>
    %1960 = stablehlo.multiply %1951, %1959 : tensor<2x6x768xf32>
    %1961 = stablehlo.reshape %140 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %1962 = stablehlo.convert %1961 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %1963 = stablehlo.broadcast_in_dim %1962, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1964 = stablehlo.add %1960, %1963 : tensor<2x6x768xf32>
    %1965 = stablehlo.transpose %141, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %1966 = stablehlo.convert %1965 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %1967 = stablehlo.dot_general %1964, %1966, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x3072xf32>) -> tensor<2x6x3072xf32>
    %1968 = stablehlo.convert %142 : (tensor<3072xf16>) -> tensor<3072xf32>
    %1969 = stablehlo.broadcast_in_dim %1968, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %1970 = stablehlo.broadcast_in_dim %1969, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x6x3072xf32>
    %1971 = stablehlo.add %1967, %1970 : tensor<2x6x3072xf32>
    %1972 = stablehlo.multiply %1971, %1971 : tensor<2x6x3072xf32>
    %1973 = stablehlo.multiply %1971, %1972 : tensor<2x6x3072xf32>
    %1974 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1975 = stablehlo.multiply %1974, %1973 : tensor<2x6x3072xf32>
    %1976 = stablehlo.add %1971, %1975 : tensor<2x6x3072xf32>
    %1977 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1978 = stablehlo.multiply %1977, %1976 : tensor<2x6x3072xf32>
    %1979 = stablehlo.tanh %1978 : tensor<2x6x3072xf32>
    %1980 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1981 = stablehlo.add %1980, %1979 : tensor<2x6x3072xf32>
    %1982 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %1983 = stablehlo.multiply %1982, %1981 : tensor<2x6x3072xf32>
    %1984 = stablehlo.multiply %1971, %1983 : tensor<2x6x3072xf32>
    %1985 = stablehlo.transpose %143, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %1986 = stablehlo.convert %1985 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %1987 = stablehlo.dot_general %1984, %1986, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x3072xf32>, tensor<3072x768xf32>) -> tensor<2x6x768xf32>
    %1988 = stablehlo.convert %144 : (tensor<768xf16>) -> tensor<768xf32>
    %1989 = stablehlo.broadcast_in_dim %1988, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1990 = stablehlo.broadcast_in_dim %1989, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %1991 = stablehlo.add %1987, %1990 : tensor<2x6x768xf32>
    %1992 = stablehlo.add %1936, %1991 : tensor<2x6x768xf32>
    %1993 = stablehlo.multiply %1992, %1992 : tensor<2x6x768xf32>
    %1994 = stablehlo.reduce(%1992 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1995 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1996 = stablehlo.divide %1994, %1995 : tensor<2x6xf32>
    %1997 = stablehlo.reduce(%1993 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %1998 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %1999 = stablehlo.divide %1997, %1998 : tensor<2x6xf32>
    %2000 = stablehlo.multiply %1996, %1996 : tensor<2x6xf32>
    %2001 = stablehlo.subtract %1999, %2000 : tensor<2x6xf32>
    %2002 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2003 = stablehlo.maximum %2002, %2001 : tensor<2x6xf32>
    %2004 = stablehlo.broadcast_in_dim %1996, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %2005 = stablehlo.broadcast_in_dim %2003, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %2006 = stablehlo.broadcast_in_dim %2004, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %2007 = stablehlo.subtract %1992, %2006 : tensor<2x6x768xf32>
    %2008 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %2009 = stablehlo.add %2005, %2008 : tensor<2x6x1xf32>
    %2010 = stablehlo.rsqrt %2009 : tensor<2x6x1xf32>
    %2011 = stablehlo.reshape %145 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2012 = stablehlo.convert %2011 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2013 = stablehlo.broadcast_in_dim %2010, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %2014 = stablehlo.broadcast_in_dim %2012, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2015 = stablehlo.multiply %2013, %2014 : tensor<2x6x768xf32>
    %2016 = stablehlo.multiply %2007, %2015 : tensor<2x6x768xf32>
    %2017 = stablehlo.reshape %146 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2018 = stablehlo.convert %2017 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2019 = stablehlo.broadcast_in_dim %2018, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2020 = stablehlo.add %2016, %2019 : tensor<2x6x768xf32>
    %2021 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %2022 = stablehlo.broadcast_in_dim %2021, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %2023 = stablehlo.broadcast_in_dim %2022, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %2024 = stablehlo.broadcast_in_dim %2022, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %2025 = stablehlo.broadcast_in_dim %2023, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %2026 = stablehlo.broadcast_in_dim %2024, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %2027 = stablehlo.compare  GE, %2025, %2026,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %2028 = stablehlo.broadcast_in_dim %2027, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %2029 = stablehlo.transpose %147, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %2030 = stablehlo.convert %2029 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %2031 = stablehlo.dot_general %2020, %2030, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x2304xf32>) -> tensor<2x6x2304xf32>
    %2032 = stablehlo.convert %148 : (tensor<2304xf16>) -> tensor<2304xf32>
    %2033 = stablehlo.broadcast_in_dim %2032, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %2034 = stablehlo.broadcast_in_dim %2033, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x6x2304xf32>
    %2035 = stablehlo.add %2031, %2034 : tensor<2x6x2304xf32>
    %2036 = stablehlo.slice %2035 [0:2, 0:6, 0:768] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %2037 = stablehlo.slice %2035 [0:2, 0:6, 768:1536] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %2038 = stablehlo.slice %2035 [0:2, 0:6, 1536:2304] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %2039 = stablehlo.reshape %2036 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %2040 = stablehlo.reshape %2037 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %2041 = stablehlo.reshape %2038 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %2042 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2043 = stablehlo.add %22, %16 : tensor<i32>
    %2044 = stablehlo.select %2042, %2043, %22 : tensor<i1>, tensor<i32>
    %2045 = stablehlo.dynamic_slice %2028, %22, %22, %2044, %22, sizes = [1, 1, 6, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x20xi1>
    %2046 = stablehlo.reshape %2045 : (tensor<1x1x6x20xi1>) -> tensor<1x6x20xi1>
    %2047 = stablehlo.broadcast_in_dim %2046, dims = [1, 2, 3] : (tensor<1x6x20xi1>) -> tensor<2x1x6x20xi1>
    %2048 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
    %2049 = stablehlo.reshape %2048 : (tensor<2x1x1x20xi32>) -> tensor<2x1x20xi32>
    %2050 = stablehlo.broadcast_in_dim %2049, dims = [0, 1, 3] : (tensor<2x1x20xi32>) -> tensor<2x1x6x20xi32>
    %2051 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x6x20xi32>
    %2052 = stablehlo.compare  NE, %2050, %2051,  SIGNED : (tensor<2x1x6x20xi32>, tensor<2x1x6x20xi32>) -> tensor<2x1x6x20xi1>
    %2053 = stablehlo.and %2052, %2047 : tensor<2x1x6x20xi1>
    %2054 = stablehlo.convert %2053 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %2055 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2056 = stablehlo.add %22, %15 : tensor<i32>
    %2057 = stablehlo.select %2055, %2056, %22 : tensor<i1>, tensor<i32>
    %2058 = stablehlo.dynamic_update_slice %195, %2040, %22, %2057, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %2059 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2060 = stablehlo.add %22, %15 : tensor<i32>
    %2061 = stablehlo.select %2059, %2060, %22 : tensor<i1>, tensor<i32>
    %2062 = stablehlo.dynamic_update_slice %196, %2041, %22, %2061, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %2063 = stablehlo.add %22, %14 : tensor<i32>
    %2064 = stablehlo.iota dim = 0 : tensor<20xi32>
    %2065 = stablehlo.add %22, %14 : tensor<i32>
    %2066 = stablehlo.broadcast_in_dim %2065, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %2067 = stablehlo.compare  LT, %2064, %2066,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %2068 = stablehlo.broadcast_in_dim %2067, dims = [3] : (tensor<20xi1>) -> tensor<2x1x6x20xi1>
    %2069 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %2070 = stablehlo.compare  NE, %2054, %2069,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %2071 = stablehlo.and %2068, %2070 : tensor<2x1x6x20xi1>
    %2072 = stablehlo.convert %2071 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %2073 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %2074 = stablehlo.compare  GT, %2072, %2073,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %2075 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %2076 = stablehlo.convert %2075 : tensor<2x1x6x20xf32>
    %2077 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %2078 = stablehlo.select %2074, %2076, %2077 : tensor<2x1x6x20xi1>, tensor<2x1x6x20xf32>
    %2079 = stablehlo.sqrt %12 : tensor<f32>
    %2080 = stablehlo.convert %2079 : tensor<f32>
    %2081 = stablehlo.broadcast_in_dim %2080, dims = [] : (tensor<f32>) -> tensor<2x6x12x64xf32>
    %2082 = stablehlo.divide %2039, %2081 : tensor<2x6x12x64xf32>
    %2083 = stablehlo.dot_general %2082, %2058, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x6x20xf32>
    %2084 = stablehlo.broadcast_in_dim %2078, dims = [0, 1, 2, 3] : (tensor<2x1x6x20xf32>) -> tensor<2x12x6x20xf32>
    %2085 = stablehlo.add %2083, %2084 : tensor<2x12x6x20xf32>
    %2086 = stablehlo.reduce(%2085 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %2087 = stablehlo.broadcast_in_dim %2086, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %2088 = stablehlo.broadcast_in_dim %2087, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %2089 = stablehlo.subtract %2085, %2088 : tensor<2x12x6x20xf32>
    %2090 = stablehlo.exponential %2089 : tensor<2x12x6x20xf32>
    %2091 = stablehlo.reduce(%2090 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %2092 = stablehlo.broadcast_in_dim %2091, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %2093 = stablehlo.broadcast_in_dim %2092, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %2094 = stablehlo.divide %2090, %2093 : tensor<2x12x6x20xf32>
    %2095 = stablehlo.dot_general %2062, %2094, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x6x20xf32>) -> tensor<2x12x64x6xf32>
    %2096 = stablehlo.transpose %2095, dims = [0, 3, 1, 2] : (tensor<2x12x64x6xf32>) -> tensor<2x6x12x64xf32>
    %2097 = stablehlo.reshape %2096 : (tensor<2x6x12x64xf32>) -> tensor<2x6x768xf32>
    %2098 = stablehlo.transpose %149, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %2099 = stablehlo.convert %2098 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2100 = stablehlo.dot_general %2097, %2099, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x768xf32>) -> tensor<2x6x768xf32>
    %2101 = stablehlo.convert %150 : (tensor<768xf16>) -> tensor<768xf32>
    %2102 = stablehlo.broadcast_in_dim %2101, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2103 = stablehlo.broadcast_in_dim %2102, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2104 = stablehlo.add %2100, %2103 : tensor<2x6x768xf32>
    %2105 = stablehlo.add %2104, %1992 : tensor<2x6x768xf32>
    %2106 = stablehlo.multiply %2105, %2105 : tensor<2x6x768xf32>
    %2107 = stablehlo.reduce(%2105 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %2108 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2109 = stablehlo.divide %2107, %2108 : tensor<2x6xf32>
    %2110 = stablehlo.reduce(%2106 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %2111 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2112 = stablehlo.divide %2110, %2111 : tensor<2x6xf32>
    %2113 = stablehlo.multiply %2109, %2109 : tensor<2x6xf32>
    %2114 = stablehlo.subtract %2112, %2113 : tensor<2x6xf32>
    %2115 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2116 = stablehlo.maximum %2115, %2114 : tensor<2x6xf32>
    %2117 = stablehlo.broadcast_in_dim %2109, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %2118 = stablehlo.broadcast_in_dim %2116, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %2119 = stablehlo.broadcast_in_dim %2117, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %2120 = stablehlo.subtract %2105, %2119 : tensor<2x6x768xf32>
    %2121 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %2122 = stablehlo.add %2118, %2121 : tensor<2x6x1xf32>
    %2123 = stablehlo.rsqrt %2122 : tensor<2x6x1xf32>
    %2124 = stablehlo.reshape %151 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2125 = stablehlo.convert %2124 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2126 = stablehlo.broadcast_in_dim %2123, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %2127 = stablehlo.broadcast_in_dim %2125, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2128 = stablehlo.multiply %2126, %2127 : tensor<2x6x768xf32>
    %2129 = stablehlo.multiply %2120, %2128 : tensor<2x6x768xf32>
    %2130 = stablehlo.reshape %152 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2131 = stablehlo.convert %2130 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2132 = stablehlo.broadcast_in_dim %2131, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2133 = stablehlo.add %2129, %2132 : tensor<2x6x768xf32>
    %2134 = stablehlo.transpose %153, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %2135 = stablehlo.convert %2134 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %2136 = stablehlo.dot_general %2133, %2135, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x3072xf32>) -> tensor<2x6x3072xf32>
    %2137 = stablehlo.convert %154 : (tensor<3072xf16>) -> tensor<3072xf32>
    %2138 = stablehlo.broadcast_in_dim %2137, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %2139 = stablehlo.broadcast_in_dim %2138, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x6x3072xf32>
    %2140 = stablehlo.add %2136, %2139 : tensor<2x6x3072xf32>
    %2141 = stablehlo.multiply %2140, %2140 : tensor<2x6x3072xf32>
    %2142 = stablehlo.multiply %2140, %2141 : tensor<2x6x3072xf32>
    %2143 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %2144 = stablehlo.multiply %2143, %2142 : tensor<2x6x3072xf32>
    %2145 = stablehlo.add %2140, %2144 : tensor<2x6x3072xf32>
    %2146 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %2147 = stablehlo.multiply %2146, %2145 : tensor<2x6x3072xf32>
    %2148 = stablehlo.tanh %2147 : tensor<2x6x3072xf32>
    %2149 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %2150 = stablehlo.add %2149, %2148 : tensor<2x6x3072xf32>
    %2151 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %2152 = stablehlo.multiply %2151, %2150 : tensor<2x6x3072xf32>
    %2153 = stablehlo.multiply %2140, %2152 : tensor<2x6x3072xf32>
    %2154 = stablehlo.transpose %155, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %2155 = stablehlo.convert %2154 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %2156 = stablehlo.dot_general %2153, %2155, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x3072xf32>, tensor<3072x768xf32>) -> tensor<2x6x768xf32>
    %2157 = stablehlo.convert %156 : (tensor<768xf16>) -> tensor<768xf32>
    %2158 = stablehlo.broadcast_in_dim %2157, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2159 = stablehlo.broadcast_in_dim %2158, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2160 = stablehlo.add %2156, %2159 : tensor<2x6x768xf32>
    %2161 = stablehlo.add %2105, %2160 : tensor<2x6x768xf32>
    %2162 = stablehlo.multiply %2161, %2161 : tensor<2x6x768xf32>
    %2163 = stablehlo.reduce(%2161 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %2164 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2165 = stablehlo.divide %2163, %2164 : tensor<2x6xf32>
    %2166 = stablehlo.reduce(%2162 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %2167 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2168 = stablehlo.divide %2166, %2167 : tensor<2x6xf32>
    %2169 = stablehlo.multiply %2165, %2165 : tensor<2x6xf32>
    %2170 = stablehlo.subtract %2168, %2169 : tensor<2x6xf32>
    %2171 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2172 = stablehlo.maximum %2171, %2170 : tensor<2x6xf32>
    %2173 = stablehlo.broadcast_in_dim %2165, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %2174 = stablehlo.broadcast_in_dim %2172, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %2175 = stablehlo.broadcast_in_dim %2173, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %2176 = stablehlo.subtract %2161, %2175 : tensor<2x6x768xf32>
    %2177 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %2178 = stablehlo.add %2174, %2177 : tensor<2x6x1xf32>
    %2179 = stablehlo.rsqrt %2178 : tensor<2x6x1xf32>
    %2180 = stablehlo.reshape %157 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2181 = stablehlo.convert %2180 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2182 = stablehlo.broadcast_in_dim %2179, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %2183 = stablehlo.broadcast_in_dim %2181, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2184 = stablehlo.multiply %2182, %2183 : tensor<2x6x768xf32>
    %2185 = stablehlo.multiply %2176, %2184 : tensor<2x6x768xf32>
    %2186 = stablehlo.reshape %158 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2187 = stablehlo.convert %2186 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2188 = stablehlo.broadcast_in_dim %2187, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2189 = stablehlo.add %2185, %2188 : tensor<2x6x768xf32>
    %2190 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %2191 = stablehlo.broadcast_in_dim %2190, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %2192 = stablehlo.broadcast_in_dim %2191, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
    %2193 = stablehlo.broadcast_in_dim %2191, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
    %2194 = stablehlo.broadcast_in_dim %2192, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
    %2195 = stablehlo.broadcast_in_dim %2193, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
    %2196 = stablehlo.compare  GE, %2194, %2195,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
    %2197 = stablehlo.broadcast_in_dim %2196, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
    %2198 = stablehlo.transpose %159, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
    %2199 = stablehlo.convert %2198 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
    %2200 = stablehlo.dot_general %2189, %2199, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x2304xf32>) -> tensor<2x6x2304xf32>
    %2201 = stablehlo.convert %160 : (tensor<2304xf16>) -> tensor<2304xf32>
    %2202 = stablehlo.broadcast_in_dim %2201, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
    %2203 = stablehlo.broadcast_in_dim %2202, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x6x2304xf32>
    %2204 = stablehlo.add %2200, %2203 : tensor<2x6x2304xf32>
    %2205 = stablehlo.slice %2204 [0:2, 0:6, 0:768] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %2206 = stablehlo.slice %2204 [0:2, 0:6, 768:1536] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %2207 = stablehlo.slice %2204 [0:2, 0:6, 1536:2304] : (tensor<2x6x2304xf32>) -> tensor<2x6x768xf32>
    %2208 = stablehlo.reshape %2205 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %2209 = stablehlo.reshape %2206 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %2210 = stablehlo.reshape %2207 : (tensor<2x6x768xf32>) -> tensor<2x6x12x64xf32>
    %2211 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2212 = stablehlo.add %22, %16 : tensor<i32>
    %2213 = stablehlo.select %2211, %2212, %22 : tensor<i1>, tensor<i32>
    %2214 = stablehlo.dynamic_slice %2197, %22, %22, %2213, %22, sizes = [1, 1, 6, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x6x20xi1>
    %2215 = stablehlo.reshape %2214 : (tensor<1x1x6x20xi1>) -> tensor<1x6x20xi1>
    %2216 = stablehlo.broadcast_in_dim %2215, dims = [1, 2, 3] : (tensor<1x6x20xi1>) -> tensor<2x1x6x20xi1>
    %2217 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
    %2218 = stablehlo.reshape %2217 : (tensor<2x1x1x20xi32>) -> tensor<2x1x20xi32>
    %2219 = stablehlo.broadcast_in_dim %2218, dims = [0, 1, 3] : (tensor<2x1x20xi32>) -> tensor<2x1x6x20xi32>
    %2220 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x6x20xi32>
    %2221 = stablehlo.compare  NE, %2219, %2220,  SIGNED : (tensor<2x1x6x20xi32>, tensor<2x1x6x20xi32>) -> tensor<2x1x6x20xi1>
    %2222 = stablehlo.and %2221, %2216 : tensor<2x1x6x20xi1>
    %2223 = stablehlo.convert %2222 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %2224 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2225 = stablehlo.add %22, %15 : tensor<i32>
    %2226 = stablehlo.select %2224, %2225, %22 : tensor<i1>, tensor<i32>
    %2227 = stablehlo.dynamic_update_slice %197, %2209, %22, %2226, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %2228 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2229 = stablehlo.add %22, %15 : tensor<i32>
    %2230 = stablehlo.select %2228, %2229, %22 : tensor<i1>, tensor<i32>
    %2231 = stablehlo.dynamic_update_slice %198, %2210, %22, %2230, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x6x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
    %2232 = stablehlo.add %22, %14 : tensor<i32>
    %2233 = stablehlo.iota dim = 0 : tensor<20xi32>
    %2234 = stablehlo.add %22, %14 : tensor<i32>
    %2235 = stablehlo.broadcast_in_dim %2234, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %2236 = stablehlo.compare  LT, %2233, %2235,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %2237 = stablehlo.broadcast_in_dim %2236, dims = [3] : (tensor<20xi1>) -> tensor<2x1x6x20xi1>
    %2238 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %2239 = stablehlo.compare  NE, %2223, %2238,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %2240 = stablehlo.and %2237, %2239 : tensor<2x1x6x20xi1>
    %2241 = stablehlo.convert %2240 : (tensor<2x1x6x20xi1>) -> tensor<2x1x6x20xf32>
    %2242 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %2243 = stablehlo.compare  GT, %2241, %2242,  FLOAT : (tensor<2x1x6x20xf32>, tensor<2x1x6x20xf32>) -> tensor<2x1x6x20xi1>
    %2244 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %2245 = stablehlo.convert %2244 : tensor<2x1x6x20xf32>
    %2246 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x6x20xf32>
    %2247 = stablehlo.select %2243, %2245, %2246 : tensor<2x1x6x20xi1>, tensor<2x1x6x20xf32>
    %2248 = stablehlo.sqrt %12 : tensor<f32>
    %2249 = stablehlo.convert %2248 : tensor<f32>
    %2250 = stablehlo.broadcast_in_dim %2249, dims = [] : (tensor<f32>) -> tensor<2x6x12x64xf32>
    %2251 = stablehlo.divide %2208, %2250 : tensor<2x6x12x64xf32>
    %2252 = stablehlo.dot_general %2251, %2227, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x6x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x6x20xf32>
    %2253 = stablehlo.broadcast_in_dim %2247, dims = [0, 1, 2, 3] : (tensor<2x1x6x20xf32>) -> tensor<2x12x6x20xf32>
    %2254 = stablehlo.add %2252, %2253 : tensor<2x12x6x20xf32>
    %2255 = stablehlo.reduce(%2254 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %2256 = stablehlo.broadcast_in_dim %2255, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %2257 = stablehlo.broadcast_in_dim %2256, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %2258 = stablehlo.subtract %2254, %2257 : tensor<2x12x6x20xf32>
    %2259 = stablehlo.exponential %2258 : tensor<2x12x6x20xf32>
    %2260 = stablehlo.reduce(%2259 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x6x20xf32>, tensor<f32>) -> tensor<2x12x6xf32>
    %2261 = stablehlo.broadcast_in_dim %2260, dims = [0, 1, 2] : (tensor<2x12x6xf32>) -> tensor<2x12x6x1xf32>
    %2262 = stablehlo.broadcast_in_dim %2261, dims = [0, 1, 2, 3] : (tensor<2x12x6x1xf32>) -> tensor<2x12x6x20xf32>
    %2263 = stablehlo.divide %2259, %2262 : tensor<2x12x6x20xf32>
    %2264 = stablehlo.dot_general %2231, %2263, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x6x20xf32>) -> tensor<2x12x64x6xf32>
    %2265 = stablehlo.transpose %2264, dims = [0, 3, 1, 2] : (tensor<2x12x64x6xf32>) -> tensor<2x6x12x64xf32>
    %2266 = stablehlo.reshape %2265 : (tensor<2x6x12x64xf32>) -> tensor<2x6x768xf32>
    %2267 = stablehlo.transpose %161, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
    %2268 = stablehlo.convert %2267 : (tensor<768x768xf16>) -> tensor<768x768xf32>
    %2269 = stablehlo.dot_general %2266, %2268, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x768xf32>) -> tensor<2x6x768xf32>
    %2270 = stablehlo.convert %162 : (tensor<768xf16>) -> tensor<768xf32>
    %2271 = stablehlo.broadcast_in_dim %2270, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2272 = stablehlo.broadcast_in_dim %2271, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2273 = stablehlo.add %2269, %2272 : tensor<2x6x768xf32>
    %2274 = stablehlo.add %2273, %2161 : tensor<2x6x768xf32>
    %2275 = stablehlo.multiply %2274, %2274 : tensor<2x6x768xf32>
    %2276 = stablehlo.reduce(%2274 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %2277 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2278 = stablehlo.divide %2276, %2277 : tensor<2x6xf32>
    %2279 = stablehlo.reduce(%2275 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %2280 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2281 = stablehlo.divide %2279, %2280 : tensor<2x6xf32>
    %2282 = stablehlo.multiply %2278, %2278 : tensor<2x6xf32>
    %2283 = stablehlo.subtract %2281, %2282 : tensor<2x6xf32>
    %2284 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2285 = stablehlo.maximum %2284, %2283 : tensor<2x6xf32>
    %2286 = stablehlo.broadcast_in_dim %2278, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %2287 = stablehlo.broadcast_in_dim %2285, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %2288 = stablehlo.broadcast_in_dim %2286, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %2289 = stablehlo.subtract %2274, %2288 : tensor<2x6x768xf32>
    %2290 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %2291 = stablehlo.add %2287, %2290 : tensor<2x6x1xf32>
    %2292 = stablehlo.rsqrt %2291 : tensor<2x6x1xf32>
    %2293 = stablehlo.reshape %163 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2294 = stablehlo.convert %2293 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2295 = stablehlo.broadcast_in_dim %2292, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %2296 = stablehlo.broadcast_in_dim %2294, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2297 = stablehlo.multiply %2295, %2296 : tensor<2x6x768xf32>
    %2298 = stablehlo.multiply %2289, %2297 : tensor<2x6x768xf32>
    %2299 = stablehlo.reshape %164 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2300 = stablehlo.convert %2299 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2301 = stablehlo.broadcast_in_dim %2300, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2302 = stablehlo.add %2298, %2301 : tensor<2x6x768xf32>
    %2303 = stablehlo.transpose %165, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
    %2304 = stablehlo.convert %2303 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
    %2305 = stablehlo.dot_general %2302, %2304, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x3072xf32>) -> tensor<2x6x3072xf32>
    %2306 = stablehlo.convert %166 : (tensor<3072xf16>) -> tensor<3072xf32>
    %2307 = stablehlo.broadcast_in_dim %2306, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
    %2308 = stablehlo.broadcast_in_dim %2307, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x6x3072xf32>
    %2309 = stablehlo.add %2305, %2308 : tensor<2x6x3072xf32>
    %2310 = stablehlo.multiply %2309, %2309 : tensor<2x6x3072xf32>
    %2311 = stablehlo.multiply %2309, %2310 : tensor<2x6x3072xf32>
    %2312 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %2313 = stablehlo.multiply %2312, %2311 : tensor<2x6x3072xf32>
    %2314 = stablehlo.add %2309, %2313 : tensor<2x6x3072xf32>
    %2315 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %2316 = stablehlo.multiply %2315, %2314 : tensor<2x6x3072xf32>
    %2317 = stablehlo.tanh %2316 : tensor<2x6x3072xf32>
    %2318 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %2319 = stablehlo.add %2318, %2317 : tensor<2x6x3072xf32>
    %2320 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x6x3072xf32>
    %2321 = stablehlo.multiply %2320, %2319 : tensor<2x6x3072xf32>
    %2322 = stablehlo.multiply %2309, %2321 : tensor<2x6x3072xf32>
    %2323 = stablehlo.transpose %167, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
    %2324 = stablehlo.convert %2323 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
    %2325 = stablehlo.dot_general %2322, %2324, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x3072xf32>, tensor<3072x768xf32>) -> tensor<2x6x768xf32>
    %2326 = stablehlo.convert %168 : (tensor<768xf16>) -> tensor<768xf32>
    %2327 = stablehlo.broadcast_in_dim %2326, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2328 = stablehlo.broadcast_in_dim %2327, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2329 = stablehlo.add %2325, %2328 : tensor<2x6x768xf32>
    %2330 = stablehlo.add %2274, %2329 : tensor<2x6x768xf32>
    %2331 = stablehlo.multiply %2330, %2330 : tensor<2x6x768xf32>
    %2332 = stablehlo.reduce(%2330 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %2333 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2334 = stablehlo.divide %2332, %2333 : tensor<2x6xf32>
    %2335 = stablehlo.reduce(%2331 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x6x768xf32>, tensor<f32>) -> tensor<2x6xf32>
    %2336 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2337 = stablehlo.divide %2335, %2336 : tensor<2x6xf32>
    %2338 = stablehlo.multiply %2334, %2334 : tensor<2x6xf32>
    %2339 = stablehlo.subtract %2337, %2338 : tensor<2x6xf32>
    %2340 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x6xf32>
    %2341 = stablehlo.maximum %2340, %2339 : tensor<2x6xf32>
    %2342 = stablehlo.broadcast_in_dim %2334, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %2343 = stablehlo.broadcast_in_dim %2341, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %2344 = stablehlo.broadcast_in_dim %2342, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %2345 = stablehlo.subtract %2330, %2344 : tensor<2x6x768xf32>
    %2346 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x6x1xf32>
    %2347 = stablehlo.add %2343, %2346 : tensor<2x6x1xf32>
    %2348 = stablehlo.rsqrt %2347 : tensor<2x6x1xf32>
    %2349 = stablehlo.reshape %169 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2350 = stablehlo.convert %2349 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2351 = stablehlo.broadcast_in_dim %2348, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x768xf32>
    %2352 = stablehlo.broadcast_in_dim %2350, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2353 = stablehlo.multiply %2351, %2352 : tensor<2x6x768xf32>
    %2354 = stablehlo.multiply %2345, %2353 : tensor<2x6x768xf32>
    %2355 = stablehlo.reshape %170 : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %2356 = stablehlo.convert %2355 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
    %2357 = stablehlo.broadcast_in_dim %2356, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x6x768xf32>
    %2358 = stablehlo.add %2354, %2357 : tensor<2x6x768xf32>
    %2359 = stablehlo.transpose %23, dims = [1, 0] : (tensor<50257x768xf16>) -> tensor<768x50257xf16>
    %2360 = stablehlo.convert %2359 : (tensor<768x50257xf16>) -> tensor<768x50257xf32>
    %2361 = stablehlo.dot_general %2358, %2360, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x6x768xf32>, tensor<768x50257xf32>) -> tensor<2x6x50257xf32>
    %2362 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2363 = stablehlo.add %22, %6 : tensor<i32>
    %2364 = stablehlo.select %2362, %2363, %22 : tensor<i1>, tensor<i32>
    %2365 = stablehlo.compare  LT, %5, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2366 = stablehlo.add %5, %14 : tensor<i32>
    %2367 = stablehlo.select %2365, %2366, %5 : tensor<i1>, tensor<i32>
    %2368 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2369 = stablehlo.add %22, %4 : tensor<i32>
    %2370 = stablehlo.select %2368, %2369, %22 : tensor<i1>, tensor<i32>
    %2371 = stablehlo.dynamic_slice %2361, %2364, %2367, %2370, sizes = [2, 1, 50257] : (tensor<2x6x50257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x1x50257xf32>
    %2372 = stablehlo.reshape %2371 : (tensor<2x1x50257xf32>) -> tensor<2x50257xf32>
    %2373 = stablehlo.subtract %14, %22 : tensor<i32>
    %2374 = stablehlo.maximum %22, %2373 : tensor<i32>
    %2375 = stablehlo.minimum %19, %2374 : tensor<i32>
    %2376 = stablehlo.subtract %19, %2375 : tensor<i32>
    %2377 = stablehlo.broadcast_in_dim %171, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2378 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %2379 = "stablehlo.scatter"(%2372, %2377, %2378) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      stablehlo.return %arg3 : tensor<f32>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<2x50257xf32>, tensor<1xi32>, tensor<2xf32>) -> tensor<2x50257xf32>
    %2380 = stablehlo.compare  NE, %2376, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2381 = stablehlo.broadcast_in_dim %2380, dims = [] : (tensor<i1>) -> tensor<2x50257xi1>
    %2382 = stablehlo.select %2381, %2379, %2372 : tensor<2x50257xi1>, tensor<2x50257xf32>
    %2383 = stablehlo.iota dim = 1 : tensor<2x50257xi32>
    %2384:2 = stablehlo.reduce(%2382 init: %11), (%2383 init: %22) across dimensions = [1] : (tensor<2x50257xf32>, tensor<2x50257xi32>, tensor<f32>, tensor<i32>) -> (tensor<2xf32>, tensor<2xi32>)
     reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<i32>, %arg5: tensor<i32>)  {
      %2405 = stablehlo.compare  GT, %arg2, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %2406 = stablehlo.compare  NE, %arg2, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %2407 = stablehlo.or %2405, %2406 : tensor<i1>
      %2408 = stablehlo.compare  EQ, %arg2, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %2409 = stablehlo.compare  LT, %arg3, %arg5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2410 = stablehlo.and %2408, %2409 : tensor<i1>
      %2411 = stablehlo.or %2407, %2410 : tensor<i1>
      %2412 = stablehlo.select %2407, %arg2, %arg4 : tensor<i1>, tensor<f32>
      %2413 = stablehlo.select %2411, %arg3, %arg5 : tensor<i1>, tensor<i32>
      stablehlo.return %2412, %2413 : tensor<f32>, tensor<i32>
    }
    %2385 = stablehlo.not %174 : tensor<2xi1>
    %2386 = stablehlo.convert %2385 : (tensor<2xi1>) -> tensor<2xi32>
    %2387 = stablehlo.multiply %2384#1, %2386 : tensor<2xi32>
    %2388 = stablehlo.convert %174 : (tensor<2xi1>) -> tensor<2xi32>
    %2389 = stablehlo.broadcast_in_dim %171, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %2390 = stablehlo.multiply %2389, %2388 : tensor<2xi32>
    %2391 = stablehlo.add %2387, %2390 : tensor<2xi32>
    %2392 = stablehlo.broadcast_in_dim %171, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %2393 = stablehlo.compare  EQ, %2391, %2392,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    %2394 = stablehlo.or %174, %2393 : tensor<2xi1>
    %2395 = stablehlo.broadcast_in_dim %2391, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %2396 = stablehlo.compare  LT, %14, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2397 = stablehlo.add %14, %15 : tensor<i32>
    %2398 = stablehlo.select %2396, %2397, %14 : tensor<i1>, tensor<i32>
    %2399 = stablehlo.dynamic_update_slice %173, %2395, %22, %2398 : (tensor<2x20xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<2x20xi32>
    %2400 = stablehlo.slice %222 [0:2, 5:6] : (tensor<2x6xi32>) -> tensor<2x1xi32>
    %2401 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
    %2402 = stablehlo.add %2400, %2401 : tensor<2x1xi32>
    %2403 = stablehlo.add %14, %19 : tensor<i32>
    %2404:192 = stablehlo.while(%iterArg = %23, %iterArg_0 = %24, %iterArg_1 = %25, %iterArg_2 = %26, %iterArg_3 = %27, %iterArg_4 = %28, %iterArg_5 = %29, %iterArg_6 = %30, %iterArg_7 = %31, %iterArg_8 = %32, %iterArg_9 = %33, %iterArg_10 = %34, %iterArg_11 = %35, %iterArg_12 = %36, %iterArg_13 = %37, %iterArg_14 = %38, %iterArg_15 = %39, %iterArg_16 = %40, %iterArg_17 = %41, %iterArg_18 = %42, %iterArg_19 = %43, %iterArg_20 = %44, %iterArg_21 = %45, %iterArg_22 = %46, %iterArg_23 = %47, %iterArg_24 = %48, %iterArg_25 = %49, %iterArg_26 = %50, %iterArg_27 = %51, %iterArg_28 = %52, %iterArg_29 = %53, %iterArg_30 = %54, %iterArg_31 = %55, %iterArg_32 = %56, %iterArg_33 = %57, %iterArg_34 = %58, %iterArg_35 = %59, %iterArg_36 = %60, %iterArg_37 = %61, %iterArg_38 = %62, %iterArg_39 = %63, %iterArg_40 = %64, %iterArg_41 = %65, %iterArg_42 = %66, %iterArg_43 = %67, %iterArg_44 = %68, %iterArg_45 = %69, %iterArg_46 = %70, %iterArg_47 = %71, %iterArg_48 = %72, %iterArg_49 = %73, %iterArg_50 = %74, %iterArg_51 = %75, %iterArg_52 = %76, %iterArg_53 = %77, %iterArg_54 = %78, %iterArg_55 = %79, %iterArg_56 = %80, %iterArg_57 = %81, %iterArg_58 = %82, %iterArg_59 = %83, %iterArg_60 = %84, %iterArg_61 = %85, %iterArg_62 = %86, %iterArg_63 = %87, %iterArg_64 = %88, %iterArg_65 = %89, %iterArg_66 = %90, %iterArg_67 = %91, %iterArg_68 = %92, %iterArg_69 = %93, %iterArg_70 = %94, %iterArg_71 = %95, %iterArg_72 = %96, %iterArg_73 = %97, %iterArg_74 = %98, %iterArg_75 = %99, %iterArg_76 = %100, %iterArg_77 = %101, %iterArg_78 = %102, %iterArg_79 = %103, %iterArg_80 = %104, %iterArg_81 = %105, %iterArg_82 = %106, %iterArg_83 = %107, %iterArg_84 = %108, %iterArg_85 = %109, %iterArg_86 = %110, %iterArg_87 = %111, %iterArg_88 = %112, %iterArg_89 = %113, %iterArg_90 = %114, %iterArg_91 = %115, %iterArg_92 = %116, %iterArg_93 = %117, %iterArg_94 = %118, %iterArg_95 = %119, %iterArg_96 = %120, %iterArg_97 = %121, %iterArg_98 = %122, %iterArg_99 = %123, %iterArg_100 = %124, %iterArg_101 = %125, %iterArg_102 = %126, %iterArg_103 = %127, %iterArg_104 = %128, %iterArg_105 = %129, %iterArg_106 = %130, %iterArg_107 = %131, %iterArg_108 = %132, %iterArg_109 = %133, %iterArg_110 = %134, %iterArg_111 = %135, %iterArg_112 = %136, %iterArg_113 = %137, %iterArg_114 = %138, %iterArg_115 = %139, %iterArg_116 = %140, %iterArg_117 = %141, %iterArg_118 = %142, %iterArg_119 = %143, %iterArg_120 = %144, %iterArg_121 = %145, %iterArg_122 = %146, %iterArg_123 = %147, %iterArg_124 = %148, %iterArg_125 = %149, %iterArg_126 = %150, %iterArg_127 = %151, %iterArg_128 = %152, %iterArg_129 = %153, %iterArg_130 = %154, %iterArg_131 = %155, %iterArg_132 = %156, %iterArg_133 = %157, %iterArg_134 = %158, %iterArg_135 = %159, %iterArg_136 = %160, %iterArg_137 = %161, %iterArg_138 = %162, %iterArg_139 = %163, %iterArg_140 = %164, %iterArg_141 = %165, %iterArg_142 = %166, %iterArg_143 = %167, %iterArg_144 = %168, %iterArg_145 = %169, %iterArg_146 = %170, %iterArg_147 = %171, %iterArg_148 = %171, %iterArg_149 = %2403, %iterArg_150 = %2399, %iterArg_151 = %2395, %iterArg_152 = %2394, %iterArg_153 = %223, %iterArg_154 = %373, %iterArg_155 = %368, %iterArg_156 = %372, %iterArg_157 = %542, %iterArg_158 = %537, %iterArg_159 = %541, %iterArg_160 = %2063, %iterArg_161 = %2058, %iterArg_162 = %2062, %iterArg_163 = %2232, %iterArg_164 = %2227, %iterArg_165 = %2231, %iterArg_166 = %711, %iterArg_167 = %706, %iterArg_168 = %710, %iterArg_169 = %880, %iterArg_170 = %875, %iterArg_171 = %879, %iterArg_172 = %1049, %iterArg_173 = %1044, %iterArg_174 = %1048, %iterArg_175 = %1218, %iterArg_176 = %1213, %iterArg_177 = %1217, %iterArg_178 = %1387, %iterArg_179 = %1382, %iterArg_180 = %1386, %iterArg_181 = %1556, %iterArg_182 = %1551, %iterArg_183 = %1555, %iterArg_184 = %1725, %iterArg_185 = %1720, %iterArg_186 = %1724, %iterArg_187 = %1894, %iterArg_188 = %1889, %iterArg_189 = %1893, %iterArg_190 = %2402) : tensor<50257x768xf16>, tensor<1024x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<2x20xi32>, tensor<2x1xi32>, tensor<2xi1>, tensor<2x20xi32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<2x1xi32>
     cond {
      %2405 = stablehlo.compare  EQ, %iterArg_149, %15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2406 = stablehlo.reduce(%iterArg_152 init: %3) applies stablehlo.and across dimensions = [0] : (tensor<2xi1>, tensor<i1>) -> tensor<i1>
      %2407 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i1>) -> tensor<i1>
      %2408 = stablehlo.compare  NE, %2405, %2407,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %2409 = stablehlo.or %2408, %2406 : tensor<i1>
      %2410 = stablehlo.not %2409 : tensor<i1>
      stablehlo.return %2410 : tensor<i1>
    } do {
      %2405 = stablehlo.convert %iterArg : (tensor<50257x768xf16>) -> tensor<50257x768xf32>
      %2406 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
      %2407 = stablehlo.compare  LT, %iterArg_151, %2406,  SIGNED : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x1xi1>
      %2408 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
      %2409 = stablehlo.add %iterArg_151, %2408 : tensor<2x1xi32>
      %2410 = stablehlo.select %2407, %2409, %iterArg_151 : tensor<2x1xi1>, tensor<2x1xi32>
      %2411 = stablehlo.broadcast_in_dim %2410, dims = [0, 1] : (tensor<2x1xi32>) -> tensor<2x1x1xi32>
      %2412 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2413 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2414 = stablehlo.concatenate %2412, %2413, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %2415 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2416 = stablehlo.compare  LT, %0, %2415,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      %2417 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2418 = stablehlo.add %0, %2417 : tensor<1xi32>
      %2419 = stablehlo.select %2416, %2418, %0 : tensor<1xi1>, tensor<1xi32>
      %2420 = stablehlo.broadcast_in_dim %2419, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
      %2421 = "stablehlo.gather"(%2414, %2420) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
      %2422 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2423 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2424 = stablehlo.concatenate %2422, %2423, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %2425 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2426 = stablehlo.compare  LT, %0, %2425,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      %2427 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2428 = stablehlo.add %0, %2427 : tensor<1xi32>
      %2429 = stablehlo.select %2426, %2428, %0 : tensor<1xi1>, tensor<1xi32>
      %2430 = stablehlo.broadcast_in_dim %2429, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
      %2431 = "stablehlo.gather"(%2424, %2430) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
      %2432 = stablehlo.subtract %2421, %2431 : tensor<1xi32>
      %2433 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1xi32>
      %2434 = stablehlo.compare  GE, %2411, %2433,  SIGNED : (tensor<2x1x1xi32>, tensor<2x1x1xi32>) -> tensor<2x1x1xi1>
      %2435 = stablehlo.broadcast_in_dim %2432, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
      %2436 = stablehlo.broadcast_in_dim %2435, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<2x1x1xi32>
      %2437 = stablehlo.compare  LE, %2411, %2436,  SIGNED : (tensor<2x1x1xi32>, tensor<2x1x1xi32>) -> tensor<2x1x1xi1>
      %2438 = stablehlo.and %2434, %2437 : tensor<2x1x1xi1>
      %2439 = stablehlo.reduce(%2438 init: %3) across dimensions = [2] : (tensor<2x1x1xi1>, tensor<i1>) -> tensor<2x1xi1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %4561 = stablehlo.and %arg2, %arg3 : tensor<i1>
        stablehlo.return %4561 : tensor<i1>
      }
      %2440 = "stablehlo.gather"(%2405, %2411) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>} : (tensor<50257x768xf32>, tensor<2x1x1xi32>) -> tensor<2x1x768xf32>
      %2441 = stablehlo.broadcast_in_dim %2439, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x1x768xi1>
      %2442 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x1x768xf32>
      %2443 = stablehlo.select %2441, %2440, %2442 : tensor<2x1x768xi1>, tensor<2x1x768xf32>
      %2444 = stablehlo.convert %iterArg_0 : (tensor<1024x768xf16>) -> tensor<1024x768xf32>
      %2445 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
      %2446 = stablehlo.compare  LT, %iterArg_190, %2445,  SIGNED : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x1xi1>
      %2447 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
      %2448 = stablehlo.add %iterArg_190, %2447 : tensor<2x1xi32>
      %2449 = stablehlo.select %2446, %2448, %iterArg_190 : tensor<2x1xi1>, tensor<2x1xi32>
      %2450 = stablehlo.broadcast_in_dim %2449, dims = [0, 1] : (tensor<2x1xi32>) -> tensor<2x1x1xi32>
      %2451 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2452 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2453 = stablehlo.concatenate %2451, %2452, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %2454 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2455 = stablehlo.compare  LT, %0, %2454,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      %2456 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2457 = stablehlo.add %0, %2456 : tensor<1xi32>
      %2458 = stablehlo.select %2455, %2457, %0 : tensor<1xi1>, tensor<1xi32>
      %2459 = stablehlo.broadcast_in_dim %2458, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
      %2460 = "stablehlo.gather"(%2453, %2459) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
      %2461 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2462 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2463 = stablehlo.concatenate %2461, %2462, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %2464 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2465 = stablehlo.compare  LT, %0, %2464,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      %2466 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %2467 = stablehlo.add %0, %2466 : tensor<1xi32>
      %2468 = stablehlo.select %2465, %2467, %0 : tensor<1xi1>, tensor<1xi32>
      %2469 = stablehlo.broadcast_in_dim %2468, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
      %2470 = "stablehlo.gather"(%2463, %2469) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
      %2471 = stablehlo.subtract %2460, %2470 : tensor<1xi32>
      %2472 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1xi32>
      %2473 = stablehlo.compare  GE, %2450, %2472,  SIGNED : (tensor<2x1x1xi32>, tensor<2x1x1xi32>) -> tensor<2x1x1xi1>
      %2474 = stablehlo.broadcast_in_dim %2471, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
      %2475 = stablehlo.broadcast_in_dim %2474, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<2x1x1xi32>
      %2476 = stablehlo.compare  LE, %2450, %2475,  SIGNED : (tensor<2x1x1xi32>, tensor<2x1x1xi32>) -> tensor<2x1x1xi1>
      %2477 = stablehlo.and %2473, %2476 : tensor<2x1x1xi1>
      %2478 = stablehlo.reduce(%2477 init: %3) across dimensions = [2] : (tensor<2x1x1xi1>, tensor<i1>) -> tensor<2x1xi1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %4561 = stablehlo.and %arg2, %arg3 : tensor<i1>
        stablehlo.return %4561 : tensor<i1>
      }
      %2479 = "stablehlo.gather"(%2444, %2450) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>} : (tensor<1024x768xf32>, tensor<2x1x1xi32>) -> tensor<2x1x768xf32>
      %2480 = stablehlo.broadcast_in_dim %2478, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x1x768xi1>
      %2481 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x1x768xf32>
      %2482 = stablehlo.select %2480, %2479, %2481 : tensor<2x1x768xi1>, tensor<2x1x768xf32>
      %2483 = stablehlo.add %2443, %2482 : tensor<2x1x768xf32>
      %2484 = stablehlo.multiply %2483, %2483 : tensor<2x1x768xf32>
      %2485 = stablehlo.reduce(%2483 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2486 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2487 = stablehlo.divide %2485, %2486 : tensor<2x1xf32>
      %2488 = stablehlo.reduce(%2484 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2489 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2490 = stablehlo.divide %2488, %2489 : tensor<2x1xf32>
      %2491 = stablehlo.multiply %2487, %2487 : tensor<2x1xf32>
      %2492 = stablehlo.subtract %2490, %2491 : tensor<2x1xf32>
      %2493 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2494 = stablehlo.maximum %2493, %2492 : tensor<2x1xf32>
      %2495 = stablehlo.broadcast_in_dim %2487, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2496 = stablehlo.broadcast_in_dim %2494, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2497 = stablehlo.broadcast_in_dim %2495, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2498 = stablehlo.subtract %2483, %2497 : tensor<2x1x768xf32>
      %2499 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %2500 = stablehlo.add %2496, %2499 : tensor<2x1x1xf32>
      %2501 = stablehlo.rsqrt %2500 : tensor<2x1x1xf32>
      %2502 = stablehlo.reshape %iterArg_1 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %2503 = stablehlo.convert %2502 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %2504 = stablehlo.broadcast_in_dim %2501, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2505 = stablehlo.broadcast_in_dim %2503, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2506 = stablehlo.multiply %2504, %2505 : tensor<2x1x768xf32>
      %2507 = stablehlo.multiply %2498, %2506 : tensor<2x1x768xf32>
      %2508 = stablehlo.reshape %iterArg_2 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %2509 = stablehlo.convert %2508 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %2510 = stablehlo.broadcast_in_dim %2509, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2511 = stablehlo.add %2507, %2510 : tensor<2x1x768xf32>
      %2512 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %2513 = stablehlo.broadcast_in_dim %2512, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %2514 = stablehlo.broadcast_in_dim %2513, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %2515 = stablehlo.broadcast_in_dim %2513, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %2516 = stablehlo.broadcast_in_dim %2514, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %2517 = stablehlo.broadcast_in_dim %2515, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %2518 = stablehlo.compare  GE, %2516, %2517,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %2519 = stablehlo.broadcast_in_dim %2518, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %2520 = stablehlo.transpose %iterArg_3, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %2521 = stablehlo.convert %2520 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %2522 = stablehlo.dot_general %2511, %2521, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x2304xf32>) -> tensor<2x1x2304xf32>
      %2523 = stablehlo.convert %iterArg_4 : (tensor<2304xf16>) -> tensor<2304xf32>
      %2524 = stablehlo.broadcast_in_dim %2523, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %2525 = stablehlo.broadcast_in_dim %2524, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x1x2304xf32>
      %2526 = stablehlo.add %2522, %2525 : tensor<2x1x2304xf32>
      %2527 = stablehlo.slice %2526 [0:2, 0:1, 0:768] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %2528 = stablehlo.slice %2526 [0:2, 0:1, 768:1536] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %2529 = stablehlo.slice %2526 [0:2, 0:1, 1536:2304] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %2530 = stablehlo.reshape %2527 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %2531 = stablehlo.reshape %2528 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %2532 = stablehlo.reshape %2529 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %2533 = stablehlo.compare  LT, %iterArg_154, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2534 = stablehlo.add %iterArg_154, %16 : tensor<i32>
      %2535 = stablehlo.select %2533, %2534, %iterArg_154 : tensor<i1>, tensor<i32>
      %2536 = stablehlo.dynamic_slice %2519, %22, %22, %2535, %22, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %2537 = stablehlo.reshape %2536 : (tensor<1x1x1x20xi1>) -> tensor<1x1x20xi1>
      %2538 = stablehlo.broadcast_in_dim %2537, dims = [1, 2, 3] : (tensor<1x1x20xi1>) -> tensor<2x1x1x20xi1>
      %2539 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
      %2540 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1x20xi32>
      %2541 = stablehlo.compare  NE, %2539, %2540,  SIGNED : (tensor<2x1x1x20xi32>, tensor<2x1x1x20xi32>) -> tensor<2x1x1x20xi1>
      %2542 = stablehlo.and %2541, %2538 : tensor<2x1x1x20xi1>
      %2543 = stablehlo.convert %2542 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %2544 = stablehlo.compare  LT, %iterArg_154, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2545 = stablehlo.add %iterArg_154, %15 : tensor<i32>
      %2546 = stablehlo.select %2544, %2545, %iterArg_154 : tensor<i1>, tensor<i32>
      %2547 = stablehlo.dynamic_update_slice %iterArg_155, %2531, %22, %2546, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %2548 = stablehlo.compare  LT, %iterArg_154, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2549 = stablehlo.add %iterArg_154, %15 : tensor<i32>
      %2550 = stablehlo.select %2548, %2549, %iterArg_154 : tensor<i1>, tensor<i32>
      %2551 = stablehlo.dynamic_update_slice %iterArg_156, %2532, %22, %2550, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %2552 = stablehlo.add %iterArg_154, %19 : tensor<i32>
      %2553 = stablehlo.iota dim = 0 : tensor<20xi32>
      %2554 = stablehlo.add %iterArg_154, %19 : tensor<i32>
      %2555 = stablehlo.broadcast_in_dim %2554, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %2556 = stablehlo.compare  LT, %2553, %2555,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %2557 = stablehlo.broadcast_in_dim %2556, dims = [3] : (tensor<20xi1>) -> tensor<2x1x1x20xi1>
      %2558 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %2559 = stablehlo.compare  NE, %2543, %2558,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %2560 = stablehlo.and %2557, %2559 : tensor<2x1x1x20xi1>
      %2561 = stablehlo.convert %2560 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %2562 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %2563 = stablehlo.compare  GT, %2561, %2562,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %2564 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %2565 = stablehlo.convert %2564 : tensor<2x1x1x20xf32>
      %2566 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %2567 = stablehlo.select %2563, %2565, %2566 : tensor<2x1x1x20xi1>, tensor<2x1x1x20xf32>
      %2568 = stablehlo.sqrt %12 : tensor<f32>
      %2569 = stablehlo.convert %2568 : tensor<f32>
      %2570 = stablehlo.broadcast_in_dim %2569, dims = [] : (tensor<f32>) -> tensor<2x1x12x64xf32>
      %2571 = stablehlo.divide %2530, %2570 : tensor<2x1x12x64xf32>
      %2572 = stablehlo.dot_general %2571, %2547, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x1x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x1x20xf32>
      %2573 = stablehlo.broadcast_in_dim %2567, dims = [0, 1, 2, 3] : (tensor<2x1x1x20xf32>) -> tensor<2x12x1x20xf32>
      %2574 = stablehlo.add %2572, %2573 : tensor<2x12x1x20xf32>
      %2575 = stablehlo.reduce(%2574 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %2576 = stablehlo.broadcast_in_dim %2575, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %2577 = stablehlo.broadcast_in_dim %2576, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %2578 = stablehlo.subtract %2574, %2577 : tensor<2x12x1x20xf32>
      %2579 = stablehlo.exponential %2578 : tensor<2x12x1x20xf32>
      %2580 = stablehlo.reduce(%2579 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %2581 = stablehlo.broadcast_in_dim %2580, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %2582 = stablehlo.broadcast_in_dim %2581, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %2583 = stablehlo.divide %2579, %2582 : tensor<2x12x1x20xf32>
      %2584 = stablehlo.dot_general %2551, %2583, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x1x20xf32>) -> tensor<2x12x64x1xf32>
      %2585 = stablehlo.transpose %2584, dims = [0, 3, 1, 2] : (tensor<2x12x64x1xf32>) -> tensor<2x1x12x64xf32>
      %2586 = stablehlo.reshape %2585 : (tensor<2x1x12x64xf32>) -> tensor<2x1x768xf32>
      %2587 = stablehlo.transpose %iterArg_5, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %2588 = stablehlo.convert %2587 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %2589 = stablehlo.dot_general %2586, %2588, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x768xf32>) -> tensor<2x1x768xf32>
      %2590 = stablehlo.convert %iterArg_6 : (tensor<768xf16>) -> tensor<768xf32>
      %2591 = stablehlo.broadcast_in_dim %2590, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %2592 = stablehlo.broadcast_in_dim %2591, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2593 = stablehlo.add %2589, %2592 : tensor<2x1x768xf32>
      %2594 = stablehlo.add %2593, %2483 : tensor<2x1x768xf32>
      %2595 = stablehlo.multiply %2594, %2594 : tensor<2x1x768xf32>
      %2596 = stablehlo.reduce(%2594 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2597 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2598 = stablehlo.divide %2596, %2597 : tensor<2x1xf32>
      %2599 = stablehlo.reduce(%2595 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2600 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2601 = stablehlo.divide %2599, %2600 : tensor<2x1xf32>
      %2602 = stablehlo.multiply %2598, %2598 : tensor<2x1xf32>
      %2603 = stablehlo.subtract %2601, %2602 : tensor<2x1xf32>
      %2604 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2605 = stablehlo.maximum %2604, %2603 : tensor<2x1xf32>
      %2606 = stablehlo.broadcast_in_dim %2598, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2607 = stablehlo.broadcast_in_dim %2605, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2608 = stablehlo.broadcast_in_dim %2606, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2609 = stablehlo.subtract %2594, %2608 : tensor<2x1x768xf32>
      %2610 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %2611 = stablehlo.add %2607, %2610 : tensor<2x1x1xf32>
      %2612 = stablehlo.rsqrt %2611 : tensor<2x1x1xf32>
      %2613 = stablehlo.reshape %iterArg_7 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %2614 = stablehlo.convert %2613 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %2615 = stablehlo.broadcast_in_dim %2612, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2616 = stablehlo.broadcast_in_dim %2614, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2617 = stablehlo.multiply %2615, %2616 : tensor<2x1x768xf32>
      %2618 = stablehlo.multiply %2609, %2617 : tensor<2x1x768xf32>
      %2619 = stablehlo.reshape %iterArg_8 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %2620 = stablehlo.convert %2619 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %2621 = stablehlo.broadcast_in_dim %2620, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2622 = stablehlo.add %2618, %2621 : tensor<2x1x768xf32>
      %2623 = stablehlo.transpose %iterArg_9, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %2624 = stablehlo.convert %2623 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %2625 = stablehlo.dot_general %2622, %2624, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x3072xf32>) -> tensor<2x1x3072xf32>
      %2626 = stablehlo.convert %iterArg_10 : (tensor<3072xf16>) -> tensor<3072xf32>
      %2627 = stablehlo.broadcast_in_dim %2626, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %2628 = stablehlo.broadcast_in_dim %2627, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
      %2629 = stablehlo.add %2625, %2628 : tensor<2x1x3072xf32>
      %2630 = stablehlo.multiply %2629, %2629 : tensor<2x1x3072xf32>
      %2631 = stablehlo.multiply %2629, %2630 : tensor<2x1x3072xf32>
      %2632 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %2633 = stablehlo.multiply %2632, %2631 : tensor<2x1x3072xf32>
      %2634 = stablehlo.add %2629, %2633 : tensor<2x1x3072xf32>
      %2635 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %2636 = stablehlo.multiply %2635, %2634 : tensor<2x1x3072xf32>
      %2637 = stablehlo.tanh %2636 : tensor<2x1x3072xf32>
      %2638 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %2639 = stablehlo.add %2638, %2637 : tensor<2x1x3072xf32>
      %2640 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %2641 = stablehlo.multiply %2640, %2639 : tensor<2x1x3072xf32>
      %2642 = stablehlo.multiply %2629, %2641 : tensor<2x1x3072xf32>
      %2643 = stablehlo.transpose %iterArg_11, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %2644 = stablehlo.convert %2643 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %2645 = stablehlo.dot_general %2642, %2644, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x3072xf32>, tensor<3072x768xf32>) -> tensor<2x1x768xf32>
      %2646 = stablehlo.convert %iterArg_12 : (tensor<768xf16>) -> tensor<768xf32>
      %2647 = stablehlo.broadcast_in_dim %2646, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %2648 = stablehlo.broadcast_in_dim %2647, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2649 = stablehlo.add %2645, %2648 : tensor<2x1x768xf32>
      %2650 = stablehlo.add %2594, %2649 : tensor<2x1x768xf32>
      %2651 = stablehlo.multiply %2650, %2650 : tensor<2x1x768xf32>
      %2652 = stablehlo.reduce(%2650 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2653 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2654 = stablehlo.divide %2652, %2653 : tensor<2x1xf32>
      %2655 = stablehlo.reduce(%2651 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2656 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2657 = stablehlo.divide %2655, %2656 : tensor<2x1xf32>
      %2658 = stablehlo.multiply %2654, %2654 : tensor<2x1xf32>
      %2659 = stablehlo.subtract %2657, %2658 : tensor<2x1xf32>
      %2660 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2661 = stablehlo.maximum %2660, %2659 : tensor<2x1xf32>
      %2662 = stablehlo.broadcast_in_dim %2654, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2663 = stablehlo.broadcast_in_dim %2661, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2664 = stablehlo.broadcast_in_dim %2662, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2665 = stablehlo.subtract %2650, %2664 : tensor<2x1x768xf32>
      %2666 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %2667 = stablehlo.add %2663, %2666 : tensor<2x1x1xf32>
      %2668 = stablehlo.rsqrt %2667 : tensor<2x1x1xf32>
      %2669 = stablehlo.reshape %iterArg_13 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %2670 = stablehlo.convert %2669 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %2671 = stablehlo.broadcast_in_dim %2668, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2672 = stablehlo.broadcast_in_dim %2670, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2673 = stablehlo.multiply %2671, %2672 : tensor<2x1x768xf32>
      %2674 = stablehlo.multiply %2665, %2673 : tensor<2x1x768xf32>
      %2675 = stablehlo.reshape %iterArg_14 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %2676 = stablehlo.convert %2675 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %2677 = stablehlo.broadcast_in_dim %2676, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2678 = stablehlo.add %2674, %2677 : tensor<2x1x768xf32>
      %2679 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %2680 = stablehlo.broadcast_in_dim %2679, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %2681 = stablehlo.broadcast_in_dim %2680, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %2682 = stablehlo.broadcast_in_dim %2680, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %2683 = stablehlo.broadcast_in_dim %2681, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %2684 = stablehlo.broadcast_in_dim %2682, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %2685 = stablehlo.compare  GE, %2683, %2684,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %2686 = stablehlo.broadcast_in_dim %2685, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %2687 = stablehlo.transpose %iterArg_15, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %2688 = stablehlo.convert %2687 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %2689 = stablehlo.dot_general %2678, %2688, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x2304xf32>) -> tensor<2x1x2304xf32>
      %2690 = stablehlo.convert %iterArg_16 : (tensor<2304xf16>) -> tensor<2304xf32>
      %2691 = stablehlo.broadcast_in_dim %2690, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %2692 = stablehlo.broadcast_in_dim %2691, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x1x2304xf32>
      %2693 = stablehlo.add %2689, %2692 : tensor<2x1x2304xf32>
      %2694 = stablehlo.slice %2693 [0:2, 0:1, 0:768] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %2695 = stablehlo.slice %2693 [0:2, 0:1, 768:1536] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %2696 = stablehlo.slice %2693 [0:2, 0:1, 1536:2304] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %2697 = stablehlo.reshape %2694 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %2698 = stablehlo.reshape %2695 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %2699 = stablehlo.reshape %2696 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %2700 = stablehlo.compare  LT, %iterArg_157, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2701 = stablehlo.add %iterArg_157, %16 : tensor<i32>
      %2702 = stablehlo.select %2700, %2701, %iterArg_157 : tensor<i1>, tensor<i32>
      %2703 = stablehlo.dynamic_slice %2686, %22, %22, %2702, %22, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %2704 = stablehlo.reshape %2703 : (tensor<1x1x1x20xi1>) -> tensor<1x1x20xi1>
      %2705 = stablehlo.broadcast_in_dim %2704, dims = [1, 2, 3] : (tensor<1x1x20xi1>) -> tensor<2x1x1x20xi1>
      %2706 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
      %2707 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1x20xi32>
      %2708 = stablehlo.compare  NE, %2706, %2707,  SIGNED : (tensor<2x1x1x20xi32>, tensor<2x1x1x20xi32>) -> tensor<2x1x1x20xi1>
      %2709 = stablehlo.and %2708, %2705 : tensor<2x1x1x20xi1>
      %2710 = stablehlo.convert %2709 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %2711 = stablehlo.compare  LT, %iterArg_157, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2712 = stablehlo.add %iterArg_157, %15 : tensor<i32>
      %2713 = stablehlo.select %2711, %2712, %iterArg_157 : tensor<i1>, tensor<i32>
      %2714 = stablehlo.dynamic_update_slice %iterArg_158, %2698, %22, %2713, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %2715 = stablehlo.compare  LT, %iterArg_157, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2716 = stablehlo.add %iterArg_157, %15 : tensor<i32>
      %2717 = stablehlo.select %2715, %2716, %iterArg_157 : tensor<i1>, tensor<i32>
      %2718 = stablehlo.dynamic_update_slice %iterArg_159, %2699, %22, %2717, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %2719 = stablehlo.add %iterArg_157, %19 : tensor<i32>
      %2720 = stablehlo.iota dim = 0 : tensor<20xi32>
      %2721 = stablehlo.add %iterArg_157, %19 : tensor<i32>
      %2722 = stablehlo.broadcast_in_dim %2721, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %2723 = stablehlo.compare  LT, %2720, %2722,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %2724 = stablehlo.broadcast_in_dim %2723, dims = [3] : (tensor<20xi1>) -> tensor<2x1x1x20xi1>
      %2725 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %2726 = stablehlo.compare  NE, %2710, %2725,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %2727 = stablehlo.and %2724, %2726 : tensor<2x1x1x20xi1>
      %2728 = stablehlo.convert %2727 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %2729 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %2730 = stablehlo.compare  GT, %2728, %2729,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %2731 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %2732 = stablehlo.convert %2731 : tensor<2x1x1x20xf32>
      %2733 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %2734 = stablehlo.select %2730, %2732, %2733 : tensor<2x1x1x20xi1>, tensor<2x1x1x20xf32>
      %2735 = stablehlo.sqrt %12 : tensor<f32>
      %2736 = stablehlo.convert %2735 : tensor<f32>
      %2737 = stablehlo.broadcast_in_dim %2736, dims = [] : (tensor<f32>) -> tensor<2x1x12x64xf32>
      %2738 = stablehlo.divide %2697, %2737 : tensor<2x1x12x64xf32>
      %2739 = stablehlo.dot_general %2738, %2714, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x1x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x1x20xf32>
      %2740 = stablehlo.broadcast_in_dim %2734, dims = [0, 1, 2, 3] : (tensor<2x1x1x20xf32>) -> tensor<2x12x1x20xf32>
      %2741 = stablehlo.add %2739, %2740 : tensor<2x12x1x20xf32>
      %2742 = stablehlo.reduce(%2741 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %2743 = stablehlo.broadcast_in_dim %2742, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %2744 = stablehlo.broadcast_in_dim %2743, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %2745 = stablehlo.subtract %2741, %2744 : tensor<2x12x1x20xf32>
      %2746 = stablehlo.exponential %2745 : tensor<2x12x1x20xf32>
      %2747 = stablehlo.reduce(%2746 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %2748 = stablehlo.broadcast_in_dim %2747, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %2749 = stablehlo.broadcast_in_dim %2748, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %2750 = stablehlo.divide %2746, %2749 : tensor<2x12x1x20xf32>
      %2751 = stablehlo.dot_general %2718, %2750, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x1x20xf32>) -> tensor<2x12x64x1xf32>
      %2752 = stablehlo.transpose %2751, dims = [0, 3, 1, 2] : (tensor<2x12x64x1xf32>) -> tensor<2x1x12x64xf32>
      %2753 = stablehlo.reshape %2752 : (tensor<2x1x12x64xf32>) -> tensor<2x1x768xf32>
      %2754 = stablehlo.transpose %iterArg_17, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %2755 = stablehlo.convert %2754 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %2756 = stablehlo.dot_general %2753, %2755, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x768xf32>) -> tensor<2x1x768xf32>
      %2757 = stablehlo.convert %iterArg_18 : (tensor<768xf16>) -> tensor<768xf32>
      %2758 = stablehlo.broadcast_in_dim %2757, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %2759 = stablehlo.broadcast_in_dim %2758, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2760 = stablehlo.add %2756, %2759 : tensor<2x1x768xf32>
      %2761 = stablehlo.add %2760, %2650 : tensor<2x1x768xf32>
      %2762 = stablehlo.multiply %2761, %2761 : tensor<2x1x768xf32>
      %2763 = stablehlo.reduce(%2761 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2764 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2765 = stablehlo.divide %2763, %2764 : tensor<2x1xf32>
      %2766 = stablehlo.reduce(%2762 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2767 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2768 = stablehlo.divide %2766, %2767 : tensor<2x1xf32>
      %2769 = stablehlo.multiply %2765, %2765 : tensor<2x1xf32>
      %2770 = stablehlo.subtract %2768, %2769 : tensor<2x1xf32>
      %2771 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2772 = stablehlo.maximum %2771, %2770 : tensor<2x1xf32>
      %2773 = stablehlo.broadcast_in_dim %2765, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2774 = stablehlo.broadcast_in_dim %2772, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2775 = stablehlo.broadcast_in_dim %2773, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2776 = stablehlo.subtract %2761, %2775 : tensor<2x1x768xf32>
      %2777 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %2778 = stablehlo.add %2774, %2777 : tensor<2x1x1xf32>
      %2779 = stablehlo.rsqrt %2778 : tensor<2x1x1xf32>
      %2780 = stablehlo.reshape %iterArg_19 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %2781 = stablehlo.convert %2780 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %2782 = stablehlo.broadcast_in_dim %2779, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2783 = stablehlo.broadcast_in_dim %2781, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2784 = stablehlo.multiply %2782, %2783 : tensor<2x1x768xf32>
      %2785 = stablehlo.multiply %2776, %2784 : tensor<2x1x768xf32>
      %2786 = stablehlo.reshape %iterArg_20 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %2787 = stablehlo.convert %2786 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %2788 = stablehlo.broadcast_in_dim %2787, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2789 = stablehlo.add %2785, %2788 : tensor<2x1x768xf32>
      %2790 = stablehlo.transpose %iterArg_21, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %2791 = stablehlo.convert %2790 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %2792 = stablehlo.dot_general %2789, %2791, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x3072xf32>) -> tensor<2x1x3072xf32>
      %2793 = stablehlo.convert %iterArg_22 : (tensor<3072xf16>) -> tensor<3072xf32>
      %2794 = stablehlo.broadcast_in_dim %2793, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %2795 = stablehlo.broadcast_in_dim %2794, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
      %2796 = stablehlo.add %2792, %2795 : tensor<2x1x3072xf32>
      %2797 = stablehlo.multiply %2796, %2796 : tensor<2x1x3072xf32>
      %2798 = stablehlo.multiply %2796, %2797 : tensor<2x1x3072xf32>
      %2799 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %2800 = stablehlo.multiply %2799, %2798 : tensor<2x1x3072xf32>
      %2801 = stablehlo.add %2796, %2800 : tensor<2x1x3072xf32>
      %2802 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %2803 = stablehlo.multiply %2802, %2801 : tensor<2x1x3072xf32>
      %2804 = stablehlo.tanh %2803 : tensor<2x1x3072xf32>
      %2805 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %2806 = stablehlo.add %2805, %2804 : tensor<2x1x3072xf32>
      %2807 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %2808 = stablehlo.multiply %2807, %2806 : tensor<2x1x3072xf32>
      %2809 = stablehlo.multiply %2796, %2808 : tensor<2x1x3072xf32>
      %2810 = stablehlo.transpose %iterArg_23, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %2811 = stablehlo.convert %2810 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %2812 = stablehlo.dot_general %2809, %2811, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x3072xf32>, tensor<3072x768xf32>) -> tensor<2x1x768xf32>
      %2813 = stablehlo.convert %iterArg_24 : (tensor<768xf16>) -> tensor<768xf32>
      %2814 = stablehlo.broadcast_in_dim %2813, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %2815 = stablehlo.broadcast_in_dim %2814, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2816 = stablehlo.add %2812, %2815 : tensor<2x1x768xf32>
      %2817 = stablehlo.add %2761, %2816 : tensor<2x1x768xf32>
      %2818 = stablehlo.multiply %2817, %2817 : tensor<2x1x768xf32>
      %2819 = stablehlo.reduce(%2817 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2820 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2821 = stablehlo.divide %2819, %2820 : tensor<2x1xf32>
      %2822 = stablehlo.reduce(%2818 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2823 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2824 = stablehlo.divide %2822, %2823 : tensor<2x1xf32>
      %2825 = stablehlo.multiply %2821, %2821 : tensor<2x1xf32>
      %2826 = stablehlo.subtract %2824, %2825 : tensor<2x1xf32>
      %2827 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2828 = stablehlo.maximum %2827, %2826 : tensor<2x1xf32>
      %2829 = stablehlo.broadcast_in_dim %2821, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2830 = stablehlo.broadcast_in_dim %2828, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2831 = stablehlo.broadcast_in_dim %2829, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2832 = stablehlo.subtract %2817, %2831 : tensor<2x1x768xf32>
      %2833 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %2834 = stablehlo.add %2830, %2833 : tensor<2x1x1xf32>
      %2835 = stablehlo.rsqrt %2834 : tensor<2x1x1xf32>
      %2836 = stablehlo.reshape %iterArg_25 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %2837 = stablehlo.convert %2836 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %2838 = stablehlo.broadcast_in_dim %2835, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2839 = stablehlo.broadcast_in_dim %2837, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2840 = stablehlo.multiply %2838, %2839 : tensor<2x1x768xf32>
      %2841 = stablehlo.multiply %2832, %2840 : tensor<2x1x768xf32>
      %2842 = stablehlo.reshape %iterArg_26 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %2843 = stablehlo.convert %2842 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %2844 = stablehlo.broadcast_in_dim %2843, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2845 = stablehlo.add %2841, %2844 : tensor<2x1x768xf32>
      %2846 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %2847 = stablehlo.broadcast_in_dim %2846, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %2848 = stablehlo.broadcast_in_dim %2847, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %2849 = stablehlo.broadcast_in_dim %2847, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %2850 = stablehlo.broadcast_in_dim %2848, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %2851 = stablehlo.broadcast_in_dim %2849, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %2852 = stablehlo.compare  GE, %2850, %2851,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %2853 = stablehlo.broadcast_in_dim %2852, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %2854 = stablehlo.transpose %iterArg_27, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %2855 = stablehlo.convert %2854 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %2856 = stablehlo.dot_general %2845, %2855, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x2304xf32>) -> tensor<2x1x2304xf32>
      %2857 = stablehlo.convert %iterArg_28 : (tensor<2304xf16>) -> tensor<2304xf32>
      %2858 = stablehlo.broadcast_in_dim %2857, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %2859 = stablehlo.broadcast_in_dim %2858, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x1x2304xf32>
      %2860 = stablehlo.add %2856, %2859 : tensor<2x1x2304xf32>
      %2861 = stablehlo.slice %2860 [0:2, 0:1, 0:768] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %2862 = stablehlo.slice %2860 [0:2, 0:1, 768:1536] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %2863 = stablehlo.slice %2860 [0:2, 0:1, 1536:2304] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %2864 = stablehlo.reshape %2861 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %2865 = stablehlo.reshape %2862 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %2866 = stablehlo.reshape %2863 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %2867 = stablehlo.compare  LT, %iterArg_166, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2868 = stablehlo.add %iterArg_166, %16 : tensor<i32>
      %2869 = stablehlo.select %2867, %2868, %iterArg_166 : tensor<i1>, tensor<i32>
      %2870 = stablehlo.dynamic_slice %2853, %22, %22, %2869, %22, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %2871 = stablehlo.reshape %2870 : (tensor<1x1x1x20xi1>) -> tensor<1x1x20xi1>
      %2872 = stablehlo.broadcast_in_dim %2871, dims = [1, 2, 3] : (tensor<1x1x20xi1>) -> tensor<2x1x1x20xi1>
      %2873 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
      %2874 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1x20xi32>
      %2875 = stablehlo.compare  NE, %2873, %2874,  SIGNED : (tensor<2x1x1x20xi32>, tensor<2x1x1x20xi32>) -> tensor<2x1x1x20xi1>
      %2876 = stablehlo.and %2875, %2872 : tensor<2x1x1x20xi1>
      %2877 = stablehlo.convert %2876 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %2878 = stablehlo.compare  LT, %iterArg_166, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2879 = stablehlo.add %iterArg_166, %15 : tensor<i32>
      %2880 = stablehlo.select %2878, %2879, %iterArg_166 : tensor<i1>, tensor<i32>
      %2881 = stablehlo.dynamic_update_slice %iterArg_167, %2865, %22, %2880, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %2882 = stablehlo.compare  LT, %iterArg_166, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %2883 = stablehlo.add %iterArg_166, %15 : tensor<i32>
      %2884 = stablehlo.select %2882, %2883, %iterArg_166 : tensor<i1>, tensor<i32>
      %2885 = stablehlo.dynamic_update_slice %iterArg_168, %2866, %22, %2884, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %2886 = stablehlo.add %iterArg_166, %19 : tensor<i32>
      %2887 = stablehlo.iota dim = 0 : tensor<20xi32>
      %2888 = stablehlo.add %iterArg_166, %19 : tensor<i32>
      %2889 = stablehlo.broadcast_in_dim %2888, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %2890 = stablehlo.compare  LT, %2887, %2889,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %2891 = stablehlo.broadcast_in_dim %2890, dims = [3] : (tensor<20xi1>) -> tensor<2x1x1x20xi1>
      %2892 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %2893 = stablehlo.compare  NE, %2877, %2892,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %2894 = stablehlo.and %2891, %2893 : tensor<2x1x1x20xi1>
      %2895 = stablehlo.convert %2894 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %2896 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %2897 = stablehlo.compare  GT, %2895, %2896,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %2898 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %2899 = stablehlo.convert %2898 : tensor<2x1x1x20xf32>
      %2900 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %2901 = stablehlo.select %2897, %2899, %2900 : tensor<2x1x1x20xi1>, tensor<2x1x1x20xf32>
      %2902 = stablehlo.sqrt %12 : tensor<f32>
      %2903 = stablehlo.convert %2902 : tensor<f32>
      %2904 = stablehlo.broadcast_in_dim %2903, dims = [] : (tensor<f32>) -> tensor<2x1x12x64xf32>
      %2905 = stablehlo.divide %2864, %2904 : tensor<2x1x12x64xf32>
      %2906 = stablehlo.dot_general %2905, %2881, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x1x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x1x20xf32>
      %2907 = stablehlo.broadcast_in_dim %2901, dims = [0, 1, 2, 3] : (tensor<2x1x1x20xf32>) -> tensor<2x12x1x20xf32>
      %2908 = stablehlo.add %2906, %2907 : tensor<2x12x1x20xf32>
      %2909 = stablehlo.reduce(%2908 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %2910 = stablehlo.broadcast_in_dim %2909, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %2911 = stablehlo.broadcast_in_dim %2910, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %2912 = stablehlo.subtract %2908, %2911 : tensor<2x12x1x20xf32>
      %2913 = stablehlo.exponential %2912 : tensor<2x12x1x20xf32>
      %2914 = stablehlo.reduce(%2913 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %2915 = stablehlo.broadcast_in_dim %2914, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %2916 = stablehlo.broadcast_in_dim %2915, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %2917 = stablehlo.divide %2913, %2916 : tensor<2x12x1x20xf32>
      %2918 = stablehlo.dot_general %2885, %2917, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x1x20xf32>) -> tensor<2x12x64x1xf32>
      %2919 = stablehlo.transpose %2918, dims = [0, 3, 1, 2] : (tensor<2x12x64x1xf32>) -> tensor<2x1x12x64xf32>
      %2920 = stablehlo.reshape %2919 : (tensor<2x1x12x64xf32>) -> tensor<2x1x768xf32>
      %2921 = stablehlo.transpose %iterArg_29, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %2922 = stablehlo.convert %2921 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %2923 = stablehlo.dot_general %2920, %2922, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x768xf32>) -> tensor<2x1x768xf32>
      %2924 = stablehlo.convert %iterArg_30 : (tensor<768xf16>) -> tensor<768xf32>
      %2925 = stablehlo.broadcast_in_dim %2924, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %2926 = stablehlo.broadcast_in_dim %2925, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2927 = stablehlo.add %2923, %2926 : tensor<2x1x768xf32>
      %2928 = stablehlo.add %2927, %2817 : tensor<2x1x768xf32>
      %2929 = stablehlo.multiply %2928, %2928 : tensor<2x1x768xf32>
      %2930 = stablehlo.reduce(%2928 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2931 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2932 = stablehlo.divide %2930, %2931 : tensor<2x1xf32>
      %2933 = stablehlo.reduce(%2929 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2934 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2935 = stablehlo.divide %2933, %2934 : tensor<2x1xf32>
      %2936 = stablehlo.multiply %2932, %2932 : tensor<2x1xf32>
      %2937 = stablehlo.subtract %2935, %2936 : tensor<2x1xf32>
      %2938 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2939 = stablehlo.maximum %2938, %2937 : tensor<2x1xf32>
      %2940 = stablehlo.broadcast_in_dim %2932, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2941 = stablehlo.broadcast_in_dim %2939, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2942 = stablehlo.broadcast_in_dim %2940, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2943 = stablehlo.subtract %2928, %2942 : tensor<2x1x768xf32>
      %2944 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %2945 = stablehlo.add %2941, %2944 : tensor<2x1x1xf32>
      %2946 = stablehlo.rsqrt %2945 : tensor<2x1x1xf32>
      %2947 = stablehlo.reshape %iterArg_31 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %2948 = stablehlo.convert %2947 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %2949 = stablehlo.broadcast_in_dim %2946, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2950 = stablehlo.broadcast_in_dim %2948, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2951 = stablehlo.multiply %2949, %2950 : tensor<2x1x768xf32>
      %2952 = stablehlo.multiply %2943, %2951 : tensor<2x1x768xf32>
      %2953 = stablehlo.reshape %iterArg_32 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %2954 = stablehlo.convert %2953 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %2955 = stablehlo.broadcast_in_dim %2954, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2956 = stablehlo.add %2952, %2955 : tensor<2x1x768xf32>
      %2957 = stablehlo.transpose %iterArg_33, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %2958 = stablehlo.convert %2957 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %2959 = stablehlo.dot_general %2956, %2958, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x3072xf32>) -> tensor<2x1x3072xf32>
      %2960 = stablehlo.convert %iterArg_34 : (tensor<3072xf16>) -> tensor<3072xf32>
      %2961 = stablehlo.broadcast_in_dim %2960, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %2962 = stablehlo.broadcast_in_dim %2961, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
      %2963 = stablehlo.add %2959, %2962 : tensor<2x1x3072xf32>
      %2964 = stablehlo.multiply %2963, %2963 : tensor<2x1x3072xf32>
      %2965 = stablehlo.multiply %2963, %2964 : tensor<2x1x3072xf32>
      %2966 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %2967 = stablehlo.multiply %2966, %2965 : tensor<2x1x3072xf32>
      %2968 = stablehlo.add %2963, %2967 : tensor<2x1x3072xf32>
      %2969 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %2970 = stablehlo.multiply %2969, %2968 : tensor<2x1x3072xf32>
      %2971 = stablehlo.tanh %2970 : tensor<2x1x3072xf32>
      %2972 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %2973 = stablehlo.add %2972, %2971 : tensor<2x1x3072xf32>
      %2974 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %2975 = stablehlo.multiply %2974, %2973 : tensor<2x1x3072xf32>
      %2976 = stablehlo.multiply %2963, %2975 : tensor<2x1x3072xf32>
      %2977 = stablehlo.transpose %iterArg_35, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %2978 = stablehlo.convert %2977 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %2979 = stablehlo.dot_general %2976, %2978, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x3072xf32>, tensor<3072x768xf32>) -> tensor<2x1x768xf32>
      %2980 = stablehlo.convert %iterArg_36 : (tensor<768xf16>) -> tensor<768xf32>
      %2981 = stablehlo.broadcast_in_dim %2980, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %2982 = stablehlo.broadcast_in_dim %2981, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %2983 = stablehlo.add %2979, %2982 : tensor<2x1x768xf32>
      %2984 = stablehlo.add %2928, %2983 : tensor<2x1x768xf32>
      %2985 = stablehlo.multiply %2984, %2984 : tensor<2x1x768xf32>
      %2986 = stablehlo.reduce(%2984 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2987 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2988 = stablehlo.divide %2986, %2987 : tensor<2x1xf32>
      %2989 = stablehlo.reduce(%2985 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %2990 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2991 = stablehlo.divide %2989, %2990 : tensor<2x1xf32>
      %2992 = stablehlo.multiply %2988, %2988 : tensor<2x1xf32>
      %2993 = stablehlo.subtract %2991, %2992 : tensor<2x1xf32>
      %2994 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %2995 = stablehlo.maximum %2994, %2993 : tensor<2x1xf32>
      %2996 = stablehlo.broadcast_in_dim %2988, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2997 = stablehlo.broadcast_in_dim %2995, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %2998 = stablehlo.broadcast_in_dim %2996, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %2999 = stablehlo.subtract %2984, %2998 : tensor<2x1x768xf32>
      %3000 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %3001 = stablehlo.add %2997, %3000 : tensor<2x1x1xf32>
      %3002 = stablehlo.rsqrt %3001 : tensor<2x1x1xf32>
      %3003 = stablehlo.reshape %iterArg_37 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3004 = stablehlo.convert %3003 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3005 = stablehlo.broadcast_in_dim %3002, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3006 = stablehlo.broadcast_in_dim %3004, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3007 = stablehlo.multiply %3005, %3006 : tensor<2x1x768xf32>
      %3008 = stablehlo.multiply %2999, %3007 : tensor<2x1x768xf32>
      %3009 = stablehlo.reshape %iterArg_38 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3010 = stablehlo.convert %3009 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3011 = stablehlo.broadcast_in_dim %3010, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3012 = stablehlo.add %3008, %3011 : tensor<2x1x768xf32>
      %3013 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %3014 = stablehlo.broadcast_in_dim %3013, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %3015 = stablehlo.broadcast_in_dim %3014, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %3016 = stablehlo.broadcast_in_dim %3014, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %3017 = stablehlo.broadcast_in_dim %3015, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %3018 = stablehlo.broadcast_in_dim %3016, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %3019 = stablehlo.compare  GE, %3017, %3018,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %3020 = stablehlo.broadcast_in_dim %3019, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %3021 = stablehlo.transpose %iterArg_39, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %3022 = stablehlo.convert %3021 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %3023 = stablehlo.dot_general %3012, %3022, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x2304xf32>) -> tensor<2x1x2304xf32>
      %3024 = stablehlo.convert %iterArg_40 : (tensor<2304xf16>) -> tensor<2304xf32>
      %3025 = stablehlo.broadcast_in_dim %3024, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %3026 = stablehlo.broadcast_in_dim %3025, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x1x2304xf32>
      %3027 = stablehlo.add %3023, %3026 : tensor<2x1x2304xf32>
      %3028 = stablehlo.slice %3027 [0:2, 0:1, 0:768] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3029 = stablehlo.slice %3027 [0:2, 0:1, 768:1536] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3030 = stablehlo.slice %3027 [0:2, 0:1, 1536:2304] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3031 = stablehlo.reshape %3028 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3032 = stablehlo.reshape %3029 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3033 = stablehlo.reshape %3030 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3034 = stablehlo.compare  LT, %iterArg_169, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3035 = stablehlo.add %iterArg_169, %16 : tensor<i32>
      %3036 = stablehlo.select %3034, %3035, %iterArg_169 : tensor<i1>, tensor<i32>
      %3037 = stablehlo.dynamic_slice %3020, %22, %22, %3036, %22, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %3038 = stablehlo.reshape %3037 : (tensor<1x1x1x20xi1>) -> tensor<1x1x20xi1>
      %3039 = stablehlo.broadcast_in_dim %3038, dims = [1, 2, 3] : (tensor<1x1x20xi1>) -> tensor<2x1x1x20xi1>
      %3040 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
      %3041 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1x20xi32>
      %3042 = stablehlo.compare  NE, %3040, %3041,  SIGNED : (tensor<2x1x1x20xi32>, tensor<2x1x1x20xi32>) -> tensor<2x1x1x20xi1>
      %3043 = stablehlo.and %3042, %3039 : tensor<2x1x1x20xi1>
      %3044 = stablehlo.convert %3043 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %3045 = stablehlo.compare  LT, %iterArg_169, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3046 = stablehlo.add %iterArg_169, %15 : tensor<i32>
      %3047 = stablehlo.select %3045, %3046, %iterArg_169 : tensor<i1>, tensor<i32>
      %3048 = stablehlo.dynamic_update_slice %iterArg_170, %3032, %22, %3047, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %3049 = stablehlo.compare  LT, %iterArg_169, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3050 = stablehlo.add %iterArg_169, %15 : tensor<i32>
      %3051 = stablehlo.select %3049, %3050, %iterArg_169 : tensor<i1>, tensor<i32>
      %3052 = stablehlo.dynamic_update_slice %iterArg_171, %3033, %22, %3051, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %3053 = stablehlo.add %iterArg_169, %19 : tensor<i32>
      %3054 = stablehlo.iota dim = 0 : tensor<20xi32>
      %3055 = stablehlo.add %iterArg_169, %19 : tensor<i32>
      %3056 = stablehlo.broadcast_in_dim %3055, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %3057 = stablehlo.compare  LT, %3054, %3056,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %3058 = stablehlo.broadcast_in_dim %3057, dims = [3] : (tensor<20xi1>) -> tensor<2x1x1x20xi1>
      %3059 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3060 = stablehlo.compare  NE, %3044, %3059,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %3061 = stablehlo.and %3058, %3060 : tensor<2x1x1x20xi1>
      %3062 = stablehlo.convert %3061 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %3063 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3064 = stablehlo.compare  GT, %3062, %3063,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %3065 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3066 = stablehlo.convert %3065 : tensor<2x1x1x20xf32>
      %3067 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3068 = stablehlo.select %3064, %3066, %3067 : tensor<2x1x1x20xi1>, tensor<2x1x1x20xf32>
      %3069 = stablehlo.sqrt %12 : tensor<f32>
      %3070 = stablehlo.convert %3069 : tensor<f32>
      %3071 = stablehlo.broadcast_in_dim %3070, dims = [] : (tensor<f32>) -> tensor<2x1x12x64xf32>
      %3072 = stablehlo.divide %3031, %3071 : tensor<2x1x12x64xf32>
      %3073 = stablehlo.dot_general %3072, %3048, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x1x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x1x20xf32>
      %3074 = stablehlo.broadcast_in_dim %3068, dims = [0, 1, 2, 3] : (tensor<2x1x1x20xf32>) -> tensor<2x12x1x20xf32>
      %3075 = stablehlo.add %3073, %3074 : tensor<2x12x1x20xf32>
      %3076 = stablehlo.reduce(%3075 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %3077 = stablehlo.broadcast_in_dim %3076, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %3078 = stablehlo.broadcast_in_dim %3077, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %3079 = stablehlo.subtract %3075, %3078 : tensor<2x12x1x20xf32>
      %3080 = stablehlo.exponential %3079 : tensor<2x12x1x20xf32>
      %3081 = stablehlo.reduce(%3080 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %3082 = stablehlo.broadcast_in_dim %3081, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %3083 = stablehlo.broadcast_in_dim %3082, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %3084 = stablehlo.divide %3080, %3083 : tensor<2x12x1x20xf32>
      %3085 = stablehlo.dot_general %3052, %3084, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x1x20xf32>) -> tensor<2x12x64x1xf32>
      %3086 = stablehlo.transpose %3085, dims = [0, 3, 1, 2] : (tensor<2x12x64x1xf32>) -> tensor<2x1x12x64xf32>
      %3087 = stablehlo.reshape %3086 : (tensor<2x1x12x64xf32>) -> tensor<2x1x768xf32>
      %3088 = stablehlo.transpose %iterArg_41, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %3089 = stablehlo.convert %3088 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %3090 = stablehlo.dot_general %3087, %3089, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x768xf32>) -> tensor<2x1x768xf32>
      %3091 = stablehlo.convert %iterArg_42 : (tensor<768xf16>) -> tensor<768xf32>
      %3092 = stablehlo.broadcast_in_dim %3091, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3093 = stablehlo.broadcast_in_dim %3092, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3094 = stablehlo.add %3090, %3093 : tensor<2x1x768xf32>
      %3095 = stablehlo.add %3094, %2984 : tensor<2x1x768xf32>
      %3096 = stablehlo.multiply %3095, %3095 : tensor<2x1x768xf32>
      %3097 = stablehlo.reduce(%3095 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3098 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3099 = stablehlo.divide %3097, %3098 : tensor<2x1xf32>
      %3100 = stablehlo.reduce(%3096 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3101 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3102 = stablehlo.divide %3100, %3101 : tensor<2x1xf32>
      %3103 = stablehlo.multiply %3099, %3099 : tensor<2x1xf32>
      %3104 = stablehlo.subtract %3102, %3103 : tensor<2x1xf32>
      %3105 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3106 = stablehlo.maximum %3105, %3104 : tensor<2x1xf32>
      %3107 = stablehlo.broadcast_in_dim %3099, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3108 = stablehlo.broadcast_in_dim %3106, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3109 = stablehlo.broadcast_in_dim %3107, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3110 = stablehlo.subtract %3095, %3109 : tensor<2x1x768xf32>
      %3111 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %3112 = stablehlo.add %3108, %3111 : tensor<2x1x1xf32>
      %3113 = stablehlo.rsqrt %3112 : tensor<2x1x1xf32>
      %3114 = stablehlo.reshape %iterArg_43 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3115 = stablehlo.convert %3114 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3116 = stablehlo.broadcast_in_dim %3113, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3117 = stablehlo.broadcast_in_dim %3115, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3118 = stablehlo.multiply %3116, %3117 : tensor<2x1x768xf32>
      %3119 = stablehlo.multiply %3110, %3118 : tensor<2x1x768xf32>
      %3120 = stablehlo.reshape %iterArg_44 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3121 = stablehlo.convert %3120 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3122 = stablehlo.broadcast_in_dim %3121, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3123 = stablehlo.add %3119, %3122 : tensor<2x1x768xf32>
      %3124 = stablehlo.transpose %iterArg_45, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %3125 = stablehlo.convert %3124 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %3126 = stablehlo.dot_general %3123, %3125, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x3072xf32>) -> tensor<2x1x3072xf32>
      %3127 = stablehlo.convert %iterArg_46 : (tensor<3072xf16>) -> tensor<3072xf32>
      %3128 = stablehlo.broadcast_in_dim %3127, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %3129 = stablehlo.broadcast_in_dim %3128, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
      %3130 = stablehlo.add %3126, %3129 : tensor<2x1x3072xf32>
      %3131 = stablehlo.multiply %3130, %3130 : tensor<2x1x3072xf32>
      %3132 = stablehlo.multiply %3130, %3131 : tensor<2x1x3072xf32>
      %3133 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3134 = stablehlo.multiply %3133, %3132 : tensor<2x1x3072xf32>
      %3135 = stablehlo.add %3130, %3134 : tensor<2x1x3072xf32>
      %3136 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3137 = stablehlo.multiply %3136, %3135 : tensor<2x1x3072xf32>
      %3138 = stablehlo.tanh %3137 : tensor<2x1x3072xf32>
      %3139 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3140 = stablehlo.add %3139, %3138 : tensor<2x1x3072xf32>
      %3141 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3142 = stablehlo.multiply %3141, %3140 : tensor<2x1x3072xf32>
      %3143 = stablehlo.multiply %3130, %3142 : tensor<2x1x3072xf32>
      %3144 = stablehlo.transpose %iterArg_47, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %3145 = stablehlo.convert %3144 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %3146 = stablehlo.dot_general %3143, %3145, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x3072xf32>, tensor<3072x768xf32>) -> tensor<2x1x768xf32>
      %3147 = stablehlo.convert %iterArg_48 : (tensor<768xf16>) -> tensor<768xf32>
      %3148 = stablehlo.broadcast_in_dim %3147, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3149 = stablehlo.broadcast_in_dim %3148, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3150 = stablehlo.add %3146, %3149 : tensor<2x1x768xf32>
      %3151 = stablehlo.add %3095, %3150 : tensor<2x1x768xf32>
      %3152 = stablehlo.multiply %3151, %3151 : tensor<2x1x768xf32>
      %3153 = stablehlo.reduce(%3151 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3154 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3155 = stablehlo.divide %3153, %3154 : tensor<2x1xf32>
      %3156 = stablehlo.reduce(%3152 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3157 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3158 = stablehlo.divide %3156, %3157 : tensor<2x1xf32>
      %3159 = stablehlo.multiply %3155, %3155 : tensor<2x1xf32>
      %3160 = stablehlo.subtract %3158, %3159 : tensor<2x1xf32>
      %3161 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3162 = stablehlo.maximum %3161, %3160 : tensor<2x1xf32>
      %3163 = stablehlo.broadcast_in_dim %3155, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3164 = stablehlo.broadcast_in_dim %3162, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3165 = stablehlo.broadcast_in_dim %3163, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3166 = stablehlo.subtract %3151, %3165 : tensor<2x1x768xf32>
      %3167 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %3168 = stablehlo.add %3164, %3167 : tensor<2x1x1xf32>
      %3169 = stablehlo.rsqrt %3168 : tensor<2x1x1xf32>
      %3170 = stablehlo.reshape %iterArg_49 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3171 = stablehlo.convert %3170 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3172 = stablehlo.broadcast_in_dim %3169, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3173 = stablehlo.broadcast_in_dim %3171, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3174 = stablehlo.multiply %3172, %3173 : tensor<2x1x768xf32>
      %3175 = stablehlo.multiply %3166, %3174 : tensor<2x1x768xf32>
      %3176 = stablehlo.reshape %iterArg_50 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3177 = stablehlo.convert %3176 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3178 = stablehlo.broadcast_in_dim %3177, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3179 = stablehlo.add %3175, %3178 : tensor<2x1x768xf32>
      %3180 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %3181 = stablehlo.broadcast_in_dim %3180, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %3182 = stablehlo.broadcast_in_dim %3181, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %3183 = stablehlo.broadcast_in_dim %3181, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %3184 = stablehlo.broadcast_in_dim %3182, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %3185 = stablehlo.broadcast_in_dim %3183, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %3186 = stablehlo.compare  GE, %3184, %3185,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %3187 = stablehlo.broadcast_in_dim %3186, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %3188 = stablehlo.transpose %iterArg_51, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %3189 = stablehlo.convert %3188 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %3190 = stablehlo.dot_general %3179, %3189, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x2304xf32>) -> tensor<2x1x2304xf32>
      %3191 = stablehlo.convert %iterArg_52 : (tensor<2304xf16>) -> tensor<2304xf32>
      %3192 = stablehlo.broadcast_in_dim %3191, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %3193 = stablehlo.broadcast_in_dim %3192, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x1x2304xf32>
      %3194 = stablehlo.add %3190, %3193 : tensor<2x1x2304xf32>
      %3195 = stablehlo.slice %3194 [0:2, 0:1, 0:768] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3196 = stablehlo.slice %3194 [0:2, 0:1, 768:1536] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3197 = stablehlo.slice %3194 [0:2, 0:1, 1536:2304] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3198 = stablehlo.reshape %3195 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3199 = stablehlo.reshape %3196 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3200 = stablehlo.reshape %3197 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3201 = stablehlo.compare  LT, %iterArg_172, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3202 = stablehlo.add %iterArg_172, %16 : tensor<i32>
      %3203 = stablehlo.select %3201, %3202, %iterArg_172 : tensor<i1>, tensor<i32>
      %3204 = stablehlo.dynamic_slice %3187, %22, %22, %3203, %22, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %3205 = stablehlo.reshape %3204 : (tensor<1x1x1x20xi1>) -> tensor<1x1x20xi1>
      %3206 = stablehlo.broadcast_in_dim %3205, dims = [1, 2, 3] : (tensor<1x1x20xi1>) -> tensor<2x1x1x20xi1>
      %3207 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
      %3208 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1x20xi32>
      %3209 = stablehlo.compare  NE, %3207, %3208,  SIGNED : (tensor<2x1x1x20xi32>, tensor<2x1x1x20xi32>) -> tensor<2x1x1x20xi1>
      %3210 = stablehlo.and %3209, %3206 : tensor<2x1x1x20xi1>
      %3211 = stablehlo.convert %3210 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %3212 = stablehlo.compare  LT, %iterArg_172, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3213 = stablehlo.add %iterArg_172, %15 : tensor<i32>
      %3214 = stablehlo.select %3212, %3213, %iterArg_172 : tensor<i1>, tensor<i32>
      %3215 = stablehlo.dynamic_update_slice %iterArg_173, %3199, %22, %3214, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %3216 = stablehlo.compare  LT, %iterArg_172, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3217 = stablehlo.add %iterArg_172, %15 : tensor<i32>
      %3218 = stablehlo.select %3216, %3217, %iterArg_172 : tensor<i1>, tensor<i32>
      %3219 = stablehlo.dynamic_update_slice %iterArg_174, %3200, %22, %3218, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %3220 = stablehlo.add %iterArg_172, %19 : tensor<i32>
      %3221 = stablehlo.iota dim = 0 : tensor<20xi32>
      %3222 = stablehlo.add %iterArg_172, %19 : tensor<i32>
      %3223 = stablehlo.broadcast_in_dim %3222, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %3224 = stablehlo.compare  LT, %3221, %3223,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %3225 = stablehlo.broadcast_in_dim %3224, dims = [3] : (tensor<20xi1>) -> tensor<2x1x1x20xi1>
      %3226 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3227 = stablehlo.compare  NE, %3211, %3226,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %3228 = stablehlo.and %3225, %3227 : tensor<2x1x1x20xi1>
      %3229 = stablehlo.convert %3228 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %3230 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3231 = stablehlo.compare  GT, %3229, %3230,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %3232 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3233 = stablehlo.convert %3232 : tensor<2x1x1x20xf32>
      %3234 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3235 = stablehlo.select %3231, %3233, %3234 : tensor<2x1x1x20xi1>, tensor<2x1x1x20xf32>
      %3236 = stablehlo.sqrt %12 : tensor<f32>
      %3237 = stablehlo.convert %3236 : tensor<f32>
      %3238 = stablehlo.broadcast_in_dim %3237, dims = [] : (tensor<f32>) -> tensor<2x1x12x64xf32>
      %3239 = stablehlo.divide %3198, %3238 : tensor<2x1x12x64xf32>
      %3240 = stablehlo.dot_general %3239, %3215, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x1x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x1x20xf32>
      %3241 = stablehlo.broadcast_in_dim %3235, dims = [0, 1, 2, 3] : (tensor<2x1x1x20xf32>) -> tensor<2x12x1x20xf32>
      %3242 = stablehlo.add %3240, %3241 : tensor<2x12x1x20xf32>
      %3243 = stablehlo.reduce(%3242 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %3244 = stablehlo.broadcast_in_dim %3243, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %3245 = stablehlo.broadcast_in_dim %3244, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %3246 = stablehlo.subtract %3242, %3245 : tensor<2x12x1x20xf32>
      %3247 = stablehlo.exponential %3246 : tensor<2x12x1x20xf32>
      %3248 = stablehlo.reduce(%3247 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %3249 = stablehlo.broadcast_in_dim %3248, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %3250 = stablehlo.broadcast_in_dim %3249, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %3251 = stablehlo.divide %3247, %3250 : tensor<2x12x1x20xf32>
      %3252 = stablehlo.dot_general %3219, %3251, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x1x20xf32>) -> tensor<2x12x64x1xf32>
      %3253 = stablehlo.transpose %3252, dims = [0, 3, 1, 2] : (tensor<2x12x64x1xf32>) -> tensor<2x1x12x64xf32>
      %3254 = stablehlo.reshape %3253 : (tensor<2x1x12x64xf32>) -> tensor<2x1x768xf32>
      %3255 = stablehlo.transpose %iterArg_53, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %3256 = stablehlo.convert %3255 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %3257 = stablehlo.dot_general %3254, %3256, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x768xf32>) -> tensor<2x1x768xf32>
      %3258 = stablehlo.convert %iterArg_54 : (tensor<768xf16>) -> tensor<768xf32>
      %3259 = stablehlo.broadcast_in_dim %3258, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3260 = stablehlo.broadcast_in_dim %3259, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3261 = stablehlo.add %3257, %3260 : tensor<2x1x768xf32>
      %3262 = stablehlo.add %3261, %3151 : tensor<2x1x768xf32>
      %3263 = stablehlo.multiply %3262, %3262 : tensor<2x1x768xf32>
      %3264 = stablehlo.reduce(%3262 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3265 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3266 = stablehlo.divide %3264, %3265 : tensor<2x1xf32>
      %3267 = stablehlo.reduce(%3263 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3268 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3269 = stablehlo.divide %3267, %3268 : tensor<2x1xf32>
      %3270 = stablehlo.multiply %3266, %3266 : tensor<2x1xf32>
      %3271 = stablehlo.subtract %3269, %3270 : tensor<2x1xf32>
      %3272 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3273 = stablehlo.maximum %3272, %3271 : tensor<2x1xf32>
      %3274 = stablehlo.broadcast_in_dim %3266, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3275 = stablehlo.broadcast_in_dim %3273, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3276 = stablehlo.broadcast_in_dim %3274, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3277 = stablehlo.subtract %3262, %3276 : tensor<2x1x768xf32>
      %3278 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %3279 = stablehlo.add %3275, %3278 : tensor<2x1x1xf32>
      %3280 = stablehlo.rsqrt %3279 : tensor<2x1x1xf32>
      %3281 = stablehlo.reshape %iterArg_55 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3282 = stablehlo.convert %3281 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3283 = stablehlo.broadcast_in_dim %3280, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3284 = stablehlo.broadcast_in_dim %3282, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3285 = stablehlo.multiply %3283, %3284 : tensor<2x1x768xf32>
      %3286 = stablehlo.multiply %3277, %3285 : tensor<2x1x768xf32>
      %3287 = stablehlo.reshape %iterArg_56 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3288 = stablehlo.convert %3287 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3289 = stablehlo.broadcast_in_dim %3288, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3290 = stablehlo.add %3286, %3289 : tensor<2x1x768xf32>
      %3291 = stablehlo.transpose %iterArg_57, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %3292 = stablehlo.convert %3291 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %3293 = stablehlo.dot_general %3290, %3292, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x3072xf32>) -> tensor<2x1x3072xf32>
      %3294 = stablehlo.convert %iterArg_58 : (tensor<3072xf16>) -> tensor<3072xf32>
      %3295 = stablehlo.broadcast_in_dim %3294, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %3296 = stablehlo.broadcast_in_dim %3295, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
      %3297 = stablehlo.add %3293, %3296 : tensor<2x1x3072xf32>
      %3298 = stablehlo.multiply %3297, %3297 : tensor<2x1x3072xf32>
      %3299 = stablehlo.multiply %3297, %3298 : tensor<2x1x3072xf32>
      %3300 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3301 = stablehlo.multiply %3300, %3299 : tensor<2x1x3072xf32>
      %3302 = stablehlo.add %3297, %3301 : tensor<2x1x3072xf32>
      %3303 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3304 = stablehlo.multiply %3303, %3302 : tensor<2x1x3072xf32>
      %3305 = stablehlo.tanh %3304 : tensor<2x1x3072xf32>
      %3306 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3307 = stablehlo.add %3306, %3305 : tensor<2x1x3072xf32>
      %3308 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3309 = stablehlo.multiply %3308, %3307 : tensor<2x1x3072xf32>
      %3310 = stablehlo.multiply %3297, %3309 : tensor<2x1x3072xf32>
      %3311 = stablehlo.transpose %iterArg_59, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %3312 = stablehlo.convert %3311 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %3313 = stablehlo.dot_general %3310, %3312, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x3072xf32>, tensor<3072x768xf32>) -> tensor<2x1x768xf32>
      %3314 = stablehlo.convert %iterArg_60 : (tensor<768xf16>) -> tensor<768xf32>
      %3315 = stablehlo.broadcast_in_dim %3314, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3316 = stablehlo.broadcast_in_dim %3315, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3317 = stablehlo.add %3313, %3316 : tensor<2x1x768xf32>
      %3318 = stablehlo.add %3262, %3317 : tensor<2x1x768xf32>
      %3319 = stablehlo.multiply %3318, %3318 : tensor<2x1x768xf32>
      %3320 = stablehlo.reduce(%3318 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3321 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3322 = stablehlo.divide %3320, %3321 : tensor<2x1xf32>
      %3323 = stablehlo.reduce(%3319 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3324 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3325 = stablehlo.divide %3323, %3324 : tensor<2x1xf32>
      %3326 = stablehlo.multiply %3322, %3322 : tensor<2x1xf32>
      %3327 = stablehlo.subtract %3325, %3326 : tensor<2x1xf32>
      %3328 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3329 = stablehlo.maximum %3328, %3327 : tensor<2x1xf32>
      %3330 = stablehlo.broadcast_in_dim %3322, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3331 = stablehlo.broadcast_in_dim %3329, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3332 = stablehlo.broadcast_in_dim %3330, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3333 = stablehlo.subtract %3318, %3332 : tensor<2x1x768xf32>
      %3334 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %3335 = stablehlo.add %3331, %3334 : tensor<2x1x1xf32>
      %3336 = stablehlo.rsqrt %3335 : tensor<2x1x1xf32>
      %3337 = stablehlo.reshape %iterArg_61 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3338 = stablehlo.convert %3337 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3339 = stablehlo.broadcast_in_dim %3336, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3340 = stablehlo.broadcast_in_dim %3338, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3341 = stablehlo.multiply %3339, %3340 : tensor<2x1x768xf32>
      %3342 = stablehlo.multiply %3333, %3341 : tensor<2x1x768xf32>
      %3343 = stablehlo.reshape %iterArg_62 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3344 = stablehlo.convert %3343 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3345 = stablehlo.broadcast_in_dim %3344, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3346 = stablehlo.add %3342, %3345 : tensor<2x1x768xf32>
      %3347 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %3348 = stablehlo.broadcast_in_dim %3347, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %3349 = stablehlo.broadcast_in_dim %3348, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %3350 = stablehlo.broadcast_in_dim %3348, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %3351 = stablehlo.broadcast_in_dim %3349, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %3352 = stablehlo.broadcast_in_dim %3350, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %3353 = stablehlo.compare  GE, %3351, %3352,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %3354 = stablehlo.broadcast_in_dim %3353, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %3355 = stablehlo.transpose %iterArg_63, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %3356 = stablehlo.convert %3355 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %3357 = stablehlo.dot_general %3346, %3356, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x2304xf32>) -> tensor<2x1x2304xf32>
      %3358 = stablehlo.convert %iterArg_64 : (tensor<2304xf16>) -> tensor<2304xf32>
      %3359 = stablehlo.broadcast_in_dim %3358, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %3360 = stablehlo.broadcast_in_dim %3359, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x1x2304xf32>
      %3361 = stablehlo.add %3357, %3360 : tensor<2x1x2304xf32>
      %3362 = stablehlo.slice %3361 [0:2, 0:1, 0:768] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3363 = stablehlo.slice %3361 [0:2, 0:1, 768:1536] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3364 = stablehlo.slice %3361 [0:2, 0:1, 1536:2304] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3365 = stablehlo.reshape %3362 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3366 = stablehlo.reshape %3363 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3367 = stablehlo.reshape %3364 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3368 = stablehlo.compare  LT, %iterArg_175, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3369 = stablehlo.add %iterArg_175, %16 : tensor<i32>
      %3370 = stablehlo.select %3368, %3369, %iterArg_175 : tensor<i1>, tensor<i32>
      %3371 = stablehlo.dynamic_slice %3354, %22, %22, %3370, %22, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %3372 = stablehlo.reshape %3371 : (tensor<1x1x1x20xi1>) -> tensor<1x1x20xi1>
      %3373 = stablehlo.broadcast_in_dim %3372, dims = [1, 2, 3] : (tensor<1x1x20xi1>) -> tensor<2x1x1x20xi1>
      %3374 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
      %3375 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1x20xi32>
      %3376 = stablehlo.compare  NE, %3374, %3375,  SIGNED : (tensor<2x1x1x20xi32>, tensor<2x1x1x20xi32>) -> tensor<2x1x1x20xi1>
      %3377 = stablehlo.and %3376, %3373 : tensor<2x1x1x20xi1>
      %3378 = stablehlo.convert %3377 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %3379 = stablehlo.compare  LT, %iterArg_175, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3380 = stablehlo.add %iterArg_175, %15 : tensor<i32>
      %3381 = stablehlo.select %3379, %3380, %iterArg_175 : tensor<i1>, tensor<i32>
      %3382 = stablehlo.dynamic_update_slice %iterArg_176, %3366, %22, %3381, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %3383 = stablehlo.compare  LT, %iterArg_175, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3384 = stablehlo.add %iterArg_175, %15 : tensor<i32>
      %3385 = stablehlo.select %3383, %3384, %iterArg_175 : tensor<i1>, tensor<i32>
      %3386 = stablehlo.dynamic_update_slice %iterArg_177, %3367, %22, %3385, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %3387 = stablehlo.add %iterArg_175, %19 : tensor<i32>
      %3388 = stablehlo.iota dim = 0 : tensor<20xi32>
      %3389 = stablehlo.add %iterArg_175, %19 : tensor<i32>
      %3390 = stablehlo.broadcast_in_dim %3389, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %3391 = stablehlo.compare  LT, %3388, %3390,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %3392 = stablehlo.broadcast_in_dim %3391, dims = [3] : (tensor<20xi1>) -> tensor<2x1x1x20xi1>
      %3393 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3394 = stablehlo.compare  NE, %3378, %3393,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %3395 = stablehlo.and %3392, %3394 : tensor<2x1x1x20xi1>
      %3396 = stablehlo.convert %3395 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %3397 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3398 = stablehlo.compare  GT, %3396, %3397,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %3399 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3400 = stablehlo.convert %3399 : tensor<2x1x1x20xf32>
      %3401 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3402 = stablehlo.select %3398, %3400, %3401 : tensor<2x1x1x20xi1>, tensor<2x1x1x20xf32>
      %3403 = stablehlo.sqrt %12 : tensor<f32>
      %3404 = stablehlo.convert %3403 : tensor<f32>
      %3405 = stablehlo.broadcast_in_dim %3404, dims = [] : (tensor<f32>) -> tensor<2x1x12x64xf32>
      %3406 = stablehlo.divide %3365, %3405 : tensor<2x1x12x64xf32>
      %3407 = stablehlo.dot_general %3406, %3382, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x1x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x1x20xf32>
      %3408 = stablehlo.broadcast_in_dim %3402, dims = [0, 1, 2, 3] : (tensor<2x1x1x20xf32>) -> tensor<2x12x1x20xf32>
      %3409 = stablehlo.add %3407, %3408 : tensor<2x12x1x20xf32>
      %3410 = stablehlo.reduce(%3409 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %3411 = stablehlo.broadcast_in_dim %3410, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %3412 = stablehlo.broadcast_in_dim %3411, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %3413 = stablehlo.subtract %3409, %3412 : tensor<2x12x1x20xf32>
      %3414 = stablehlo.exponential %3413 : tensor<2x12x1x20xf32>
      %3415 = stablehlo.reduce(%3414 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %3416 = stablehlo.broadcast_in_dim %3415, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %3417 = stablehlo.broadcast_in_dim %3416, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %3418 = stablehlo.divide %3414, %3417 : tensor<2x12x1x20xf32>
      %3419 = stablehlo.dot_general %3386, %3418, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x1x20xf32>) -> tensor<2x12x64x1xf32>
      %3420 = stablehlo.transpose %3419, dims = [0, 3, 1, 2] : (tensor<2x12x64x1xf32>) -> tensor<2x1x12x64xf32>
      %3421 = stablehlo.reshape %3420 : (tensor<2x1x12x64xf32>) -> tensor<2x1x768xf32>
      %3422 = stablehlo.transpose %iterArg_65, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %3423 = stablehlo.convert %3422 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %3424 = stablehlo.dot_general %3421, %3423, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x768xf32>) -> tensor<2x1x768xf32>
      %3425 = stablehlo.convert %iterArg_66 : (tensor<768xf16>) -> tensor<768xf32>
      %3426 = stablehlo.broadcast_in_dim %3425, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3427 = stablehlo.broadcast_in_dim %3426, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3428 = stablehlo.add %3424, %3427 : tensor<2x1x768xf32>
      %3429 = stablehlo.add %3428, %3318 : tensor<2x1x768xf32>
      %3430 = stablehlo.multiply %3429, %3429 : tensor<2x1x768xf32>
      %3431 = stablehlo.reduce(%3429 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3432 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3433 = stablehlo.divide %3431, %3432 : tensor<2x1xf32>
      %3434 = stablehlo.reduce(%3430 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3435 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3436 = stablehlo.divide %3434, %3435 : tensor<2x1xf32>
      %3437 = stablehlo.multiply %3433, %3433 : tensor<2x1xf32>
      %3438 = stablehlo.subtract %3436, %3437 : tensor<2x1xf32>
      %3439 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3440 = stablehlo.maximum %3439, %3438 : tensor<2x1xf32>
      %3441 = stablehlo.broadcast_in_dim %3433, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3442 = stablehlo.broadcast_in_dim %3440, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3443 = stablehlo.broadcast_in_dim %3441, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3444 = stablehlo.subtract %3429, %3443 : tensor<2x1x768xf32>
      %3445 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %3446 = stablehlo.add %3442, %3445 : tensor<2x1x1xf32>
      %3447 = stablehlo.rsqrt %3446 : tensor<2x1x1xf32>
      %3448 = stablehlo.reshape %iterArg_67 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3449 = stablehlo.convert %3448 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3450 = stablehlo.broadcast_in_dim %3447, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3451 = stablehlo.broadcast_in_dim %3449, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3452 = stablehlo.multiply %3450, %3451 : tensor<2x1x768xf32>
      %3453 = stablehlo.multiply %3444, %3452 : tensor<2x1x768xf32>
      %3454 = stablehlo.reshape %iterArg_68 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3455 = stablehlo.convert %3454 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3456 = stablehlo.broadcast_in_dim %3455, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3457 = stablehlo.add %3453, %3456 : tensor<2x1x768xf32>
      %3458 = stablehlo.transpose %iterArg_69, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %3459 = stablehlo.convert %3458 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %3460 = stablehlo.dot_general %3457, %3459, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x3072xf32>) -> tensor<2x1x3072xf32>
      %3461 = stablehlo.convert %iterArg_70 : (tensor<3072xf16>) -> tensor<3072xf32>
      %3462 = stablehlo.broadcast_in_dim %3461, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %3463 = stablehlo.broadcast_in_dim %3462, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
      %3464 = stablehlo.add %3460, %3463 : tensor<2x1x3072xf32>
      %3465 = stablehlo.multiply %3464, %3464 : tensor<2x1x3072xf32>
      %3466 = stablehlo.multiply %3464, %3465 : tensor<2x1x3072xf32>
      %3467 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3468 = stablehlo.multiply %3467, %3466 : tensor<2x1x3072xf32>
      %3469 = stablehlo.add %3464, %3468 : tensor<2x1x3072xf32>
      %3470 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3471 = stablehlo.multiply %3470, %3469 : tensor<2x1x3072xf32>
      %3472 = stablehlo.tanh %3471 : tensor<2x1x3072xf32>
      %3473 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3474 = stablehlo.add %3473, %3472 : tensor<2x1x3072xf32>
      %3475 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3476 = stablehlo.multiply %3475, %3474 : tensor<2x1x3072xf32>
      %3477 = stablehlo.multiply %3464, %3476 : tensor<2x1x3072xf32>
      %3478 = stablehlo.transpose %iterArg_71, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %3479 = stablehlo.convert %3478 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %3480 = stablehlo.dot_general %3477, %3479, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x3072xf32>, tensor<3072x768xf32>) -> tensor<2x1x768xf32>
      %3481 = stablehlo.convert %iterArg_72 : (tensor<768xf16>) -> tensor<768xf32>
      %3482 = stablehlo.broadcast_in_dim %3481, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3483 = stablehlo.broadcast_in_dim %3482, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3484 = stablehlo.add %3480, %3483 : tensor<2x1x768xf32>
      %3485 = stablehlo.add %3429, %3484 : tensor<2x1x768xf32>
      %3486 = stablehlo.multiply %3485, %3485 : tensor<2x1x768xf32>
      %3487 = stablehlo.reduce(%3485 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3488 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3489 = stablehlo.divide %3487, %3488 : tensor<2x1xf32>
      %3490 = stablehlo.reduce(%3486 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3491 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3492 = stablehlo.divide %3490, %3491 : tensor<2x1xf32>
      %3493 = stablehlo.multiply %3489, %3489 : tensor<2x1xf32>
      %3494 = stablehlo.subtract %3492, %3493 : tensor<2x1xf32>
      %3495 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3496 = stablehlo.maximum %3495, %3494 : tensor<2x1xf32>
      %3497 = stablehlo.broadcast_in_dim %3489, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3498 = stablehlo.broadcast_in_dim %3496, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3499 = stablehlo.broadcast_in_dim %3497, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3500 = stablehlo.subtract %3485, %3499 : tensor<2x1x768xf32>
      %3501 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %3502 = stablehlo.add %3498, %3501 : tensor<2x1x1xf32>
      %3503 = stablehlo.rsqrt %3502 : tensor<2x1x1xf32>
      %3504 = stablehlo.reshape %iterArg_73 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3505 = stablehlo.convert %3504 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3506 = stablehlo.broadcast_in_dim %3503, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3507 = stablehlo.broadcast_in_dim %3505, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3508 = stablehlo.multiply %3506, %3507 : tensor<2x1x768xf32>
      %3509 = stablehlo.multiply %3500, %3508 : tensor<2x1x768xf32>
      %3510 = stablehlo.reshape %iterArg_74 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3511 = stablehlo.convert %3510 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3512 = stablehlo.broadcast_in_dim %3511, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3513 = stablehlo.add %3509, %3512 : tensor<2x1x768xf32>
      %3514 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %3515 = stablehlo.broadcast_in_dim %3514, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %3516 = stablehlo.broadcast_in_dim %3515, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %3517 = stablehlo.broadcast_in_dim %3515, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %3518 = stablehlo.broadcast_in_dim %3516, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %3519 = stablehlo.broadcast_in_dim %3517, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %3520 = stablehlo.compare  GE, %3518, %3519,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %3521 = stablehlo.broadcast_in_dim %3520, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %3522 = stablehlo.transpose %iterArg_75, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %3523 = stablehlo.convert %3522 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %3524 = stablehlo.dot_general %3513, %3523, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x2304xf32>) -> tensor<2x1x2304xf32>
      %3525 = stablehlo.convert %iterArg_76 : (tensor<2304xf16>) -> tensor<2304xf32>
      %3526 = stablehlo.broadcast_in_dim %3525, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %3527 = stablehlo.broadcast_in_dim %3526, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x1x2304xf32>
      %3528 = stablehlo.add %3524, %3527 : tensor<2x1x2304xf32>
      %3529 = stablehlo.slice %3528 [0:2, 0:1, 0:768] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3530 = stablehlo.slice %3528 [0:2, 0:1, 768:1536] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3531 = stablehlo.slice %3528 [0:2, 0:1, 1536:2304] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3532 = stablehlo.reshape %3529 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3533 = stablehlo.reshape %3530 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3534 = stablehlo.reshape %3531 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3535 = stablehlo.compare  LT, %iterArg_178, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3536 = stablehlo.add %iterArg_178, %16 : tensor<i32>
      %3537 = stablehlo.select %3535, %3536, %iterArg_178 : tensor<i1>, tensor<i32>
      %3538 = stablehlo.dynamic_slice %3521, %22, %22, %3537, %22, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %3539 = stablehlo.reshape %3538 : (tensor<1x1x1x20xi1>) -> tensor<1x1x20xi1>
      %3540 = stablehlo.broadcast_in_dim %3539, dims = [1, 2, 3] : (tensor<1x1x20xi1>) -> tensor<2x1x1x20xi1>
      %3541 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
      %3542 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1x20xi32>
      %3543 = stablehlo.compare  NE, %3541, %3542,  SIGNED : (tensor<2x1x1x20xi32>, tensor<2x1x1x20xi32>) -> tensor<2x1x1x20xi1>
      %3544 = stablehlo.and %3543, %3540 : tensor<2x1x1x20xi1>
      %3545 = stablehlo.convert %3544 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %3546 = stablehlo.compare  LT, %iterArg_178, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3547 = stablehlo.add %iterArg_178, %15 : tensor<i32>
      %3548 = stablehlo.select %3546, %3547, %iterArg_178 : tensor<i1>, tensor<i32>
      %3549 = stablehlo.dynamic_update_slice %iterArg_179, %3533, %22, %3548, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %3550 = stablehlo.compare  LT, %iterArg_178, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3551 = stablehlo.add %iterArg_178, %15 : tensor<i32>
      %3552 = stablehlo.select %3550, %3551, %iterArg_178 : tensor<i1>, tensor<i32>
      %3553 = stablehlo.dynamic_update_slice %iterArg_180, %3534, %22, %3552, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %3554 = stablehlo.add %iterArg_178, %19 : tensor<i32>
      %3555 = stablehlo.iota dim = 0 : tensor<20xi32>
      %3556 = stablehlo.add %iterArg_178, %19 : tensor<i32>
      %3557 = stablehlo.broadcast_in_dim %3556, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %3558 = stablehlo.compare  LT, %3555, %3557,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %3559 = stablehlo.broadcast_in_dim %3558, dims = [3] : (tensor<20xi1>) -> tensor<2x1x1x20xi1>
      %3560 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3561 = stablehlo.compare  NE, %3545, %3560,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %3562 = stablehlo.and %3559, %3561 : tensor<2x1x1x20xi1>
      %3563 = stablehlo.convert %3562 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %3564 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3565 = stablehlo.compare  GT, %3563, %3564,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %3566 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3567 = stablehlo.convert %3566 : tensor<2x1x1x20xf32>
      %3568 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3569 = stablehlo.select %3565, %3567, %3568 : tensor<2x1x1x20xi1>, tensor<2x1x1x20xf32>
      %3570 = stablehlo.sqrt %12 : tensor<f32>
      %3571 = stablehlo.convert %3570 : tensor<f32>
      %3572 = stablehlo.broadcast_in_dim %3571, dims = [] : (tensor<f32>) -> tensor<2x1x12x64xf32>
      %3573 = stablehlo.divide %3532, %3572 : tensor<2x1x12x64xf32>
      %3574 = stablehlo.dot_general %3573, %3549, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x1x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x1x20xf32>
      %3575 = stablehlo.broadcast_in_dim %3569, dims = [0, 1, 2, 3] : (tensor<2x1x1x20xf32>) -> tensor<2x12x1x20xf32>
      %3576 = stablehlo.add %3574, %3575 : tensor<2x12x1x20xf32>
      %3577 = stablehlo.reduce(%3576 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %3578 = stablehlo.broadcast_in_dim %3577, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %3579 = stablehlo.broadcast_in_dim %3578, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %3580 = stablehlo.subtract %3576, %3579 : tensor<2x12x1x20xf32>
      %3581 = stablehlo.exponential %3580 : tensor<2x12x1x20xf32>
      %3582 = stablehlo.reduce(%3581 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %3583 = stablehlo.broadcast_in_dim %3582, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %3584 = stablehlo.broadcast_in_dim %3583, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %3585 = stablehlo.divide %3581, %3584 : tensor<2x12x1x20xf32>
      %3586 = stablehlo.dot_general %3553, %3585, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x1x20xf32>) -> tensor<2x12x64x1xf32>
      %3587 = stablehlo.transpose %3586, dims = [0, 3, 1, 2] : (tensor<2x12x64x1xf32>) -> tensor<2x1x12x64xf32>
      %3588 = stablehlo.reshape %3587 : (tensor<2x1x12x64xf32>) -> tensor<2x1x768xf32>
      %3589 = stablehlo.transpose %iterArg_77, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %3590 = stablehlo.convert %3589 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %3591 = stablehlo.dot_general %3588, %3590, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x768xf32>) -> tensor<2x1x768xf32>
      %3592 = stablehlo.convert %iterArg_78 : (tensor<768xf16>) -> tensor<768xf32>
      %3593 = stablehlo.broadcast_in_dim %3592, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3594 = stablehlo.broadcast_in_dim %3593, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3595 = stablehlo.add %3591, %3594 : tensor<2x1x768xf32>
      %3596 = stablehlo.add %3595, %3485 : tensor<2x1x768xf32>
      %3597 = stablehlo.multiply %3596, %3596 : tensor<2x1x768xf32>
      %3598 = stablehlo.reduce(%3596 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3599 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3600 = stablehlo.divide %3598, %3599 : tensor<2x1xf32>
      %3601 = stablehlo.reduce(%3597 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3602 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3603 = stablehlo.divide %3601, %3602 : tensor<2x1xf32>
      %3604 = stablehlo.multiply %3600, %3600 : tensor<2x1xf32>
      %3605 = stablehlo.subtract %3603, %3604 : tensor<2x1xf32>
      %3606 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3607 = stablehlo.maximum %3606, %3605 : tensor<2x1xf32>
      %3608 = stablehlo.broadcast_in_dim %3600, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3609 = stablehlo.broadcast_in_dim %3607, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3610 = stablehlo.broadcast_in_dim %3608, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3611 = stablehlo.subtract %3596, %3610 : tensor<2x1x768xf32>
      %3612 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %3613 = stablehlo.add %3609, %3612 : tensor<2x1x1xf32>
      %3614 = stablehlo.rsqrt %3613 : tensor<2x1x1xf32>
      %3615 = stablehlo.reshape %iterArg_79 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3616 = stablehlo.convert %3615 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3617 = stablehlo.broadcast_in_dim %3614, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3618 = stablehlo.broadcast_in_dim %3616, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3619 = stablehlo.multiply %3617, %3618 : tensor<2x1x768xf32>
      %3620 = stablehlo.multiply %3611, %3619 : tensor<2x1x768xf32>
      %3621 = stablehlo.reshape %iterArg_80 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3622 = stablehlo.convert %3621 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3623 = stablehlo.broadcast_in_dim %3622, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3624 = stablehlo.add %3620, %3623 : tensor<2x1x768xf32>
      %3625 = stablehlo.transpose %iterArg_81, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %3626 = stablehlo.convert %3625 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %3627 = stablehlo.dot_general %3624, %3626, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x3072xf32>) -> tensor<2x1x3072xf32>
      %3628 = stablehlo.convert %iterArg_82 : (tensor<3072xf16>) -> tensor<3072xf32>
      %3629 = stablehlo.broadcast_in_dim %3628, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %3630 = stablehlo.broadcast_in_dim %3629, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
      %3631 = stablehlo.add %3627, %3630 : tensor<2x1x3072xf32>
      %3632 = stablehlo.multiply %3631, %3631 : tensor<2x1x3072xf32>
      %3633 = stablehlo.multiply %3631, %3632 : tensor<2x1x3072xf32>
      %3634 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3635 = stablehlo.multiply %3634, %3633 : tensor<2x1x3072xf32>
      %3636 = stablehlo.add %3631, %3635 : tensor<2x1x3072xf32>
      %3637 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3638 = stablehlo.multiply %3637, %3636 : tensor<2x1x3072xf32>
      %3639 = stablehlo.tanh %3638 : tensor<2x1x3072xf32>
      %3640 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3641 = stablehlo.add %3640, %3639 : tensor<2x1x3072xf32>
      %3642 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3643 = stablehlo.multiply %3642, %3641 : tensor<2x1x3072xf32>
      %3644 = stablehlo.multiply %3631, %3643 : tensor<2x1x3072xf32>
      %3645 = stablehlo.transpose %iterArg_83, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %3646 = stablehlo.convert %3645 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %3647 = stablehlo.dot_general %3644, %3646, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x3072xf32>, tensor<3072x768xf32>) -> tensor<2x1x768xf32>
      %3648 = stablehlo.convert %iterArg_84 : (tensor<768xf16>) -> tensor<768xf32>
      %3649 = stablehlo.broadcast_in_dim %3648, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3650 = stablehlo.broadcast_in_dim %3649, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3651 = stablehlo.add %3647, %3650 : tensor<2x1x768xf32>
      %3652 = stablehlo.add %3596, %3651 : tensor<2x1x768xf32>
      %3653 = stablehlo.multiply %3652, %3652 : tensor<2x1x768xf32>
      %3654 = stablehlo.reduce(%3652 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3655 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3656 = stablehlo.divide %3654, %3655 : tensor<2x1xf32>
      %3657 = stablehlo.reduce(%3653 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3658 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3659 = stablehlo.divide %3657, %3658 : tensor<2x1xf32>
      %3660 = stablehlo.multiply %3656, %3656 : tensor<2x1xf32>
      %3661 = stablehlo.subtract %3659, %3660 : tensor<2x1xf32>
      %3662 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3663 = stablehlo.maximum %3662, %3661 : tensor<2x1xf32>
      %3664 = stablehlo.broadcast_in_dim %3656, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3665 = stablehlo.broadcast_in_dim %3663, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3666 = stablehlo.broadcast_in_dim %3664, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3667 = stablehlo.subtract %3652, %3666 : tensor<2x1x768xf32>
      %3668 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %3669 = stablehlo.add %3665, %3668 : tensor<2x1x1xf32>
      %3670 = stablehlo.rsqrt %3669 : tensor<2x1x1xf32>
      %3671 = stablehlo.reshape %iterArg_85 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3672 = stablehlo.convert %3671 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3673 = stablehlo.broadcast_in_dim %3670, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3674 = stablehlo.broadcast_in_dim %3672, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3675 = stablehlo.multiply %3673, %3674 : tensor<2x1x768xf32>
      %3676 = stablehlo.multiply %3667, %3675 : tensor<2x1x768xf32>
      %3677 = stablehlo.reshape %iterArg_86 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3678 = stablehlo.convert %3677 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3679 = stablehlo.broadcast_in_dim %3678, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3680 = stablehlo.add %3676, %3679 : tensor<2x1x768xf32>
      %3681 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %3682 = stablehlo.broadcast_in_dim %3681, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %3683 = stablehlo.broadcast_in_dim %3682, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %3684 = stablehlo.broadcast_in_dim %3682, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %3685 = stablehlo.broadcast_in_dim %3683, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %3686 = stablehlo.broadcast_in_dim %3684, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %3687 = stablehlo.compare  GE, %3685, %3686,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %3688 = stablehlo.broadcast_in_dim %3687, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %3689 = stablehlo.transpose %iterArg_87, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %3690 = stablehlo.convert %3689 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %3691 = stablehlo.dot_general %3680, %3690, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x2304xf32>) -> tensor<2x1x2304xf32>
      %3692 = stablehlo.convert %iterArg_88 : (tensor<2304xf16>) -> tensor<2304xf32>
      %3693 = stablehlo.broadcast_in_dim %3692, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %3694 = stablehlo.broadcast_in_dim %3693, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x1x2304xf32>
      %3695 = stablehlo.add %3691, %3694 : tensor<2x1x2304xf32>
      %3696 = stablehlo.slice %3695 [0:2, 0:1, 0:768] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3697 = stablehlo.slice %3695 [0:2, 0:1, 768:1536] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3698 = stablehlo.slice %3695 [0:2, 0:1, 1536:2304] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3699 = stablehlo.reshape %3696 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3700 = stablehlo.reshape %3697 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3701 = stablehlo.reshape %3698 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3702 = stablehlo.compare  LT, %iterArg_181, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3703 = stablehlo.add %iterArg_181, %16 : tensor<i32>
      %3704 = stablehlo.select %3702, %3703, %iterArg_181 : tensor<i1>, tensor<i32>
      %3705 = stablehlo.dynamic_slice %3688, %22, %22, %3704, %22, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %3706 = stablehlo.reshape %3705 : (tensor<1x1x1x20xi1>) -> tensor<1x1x20xi1>
      %3707 = stablehlo.broadcast_in_dim %3706, dims = [1, 2, 3] : (tensor<1x1x20xi1>) -> tensor<2x1x1x20xi1>
      %3708 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
      %3709 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1x20xi32>
      %3710 = stablehlo.compare  NE, %3708, %3709,  SIGNED : (tensor<2x1x1x20xi32>, tensor<2x1x1x20xi32>) -> tensor<2x1x1x20xi1>
      %3711 = stablehlo.and %3710, %3707 : tensor<2x1x1x20xi1>
      %3712 = stablehlo.convert %3711 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %3713 = stablehlo.compare  LT, %iterArg_181, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3714 = stablehlo.add %iterArg_181, %15 : tensor<i32>
      %3715 = stablehlo.select %3713, %3714, %iterArg_181 : tensor<i1>, tensor<i32>
      %3716 = stablehlo.dynamic_update_slice %iterArg_182, %3700, %22, %3715, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %3717 = stablehlo.compare  LT, %iterArg_181, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3718 = stablehlo.add %iterArg_181, %15 : tensor<i32>
      %3719 = stablehlo.select %3717, %3718, %iterArg_181 : tensor<i1>, tensor<i32>
      %3720 = stablehlo.dynamic_update_slice %iterArg_183, %3701, %22, %3719, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %3721 = stablehlo.add %iterArg_181, %19 : tensor<i32>
      %3722 = stablehlo.iota dim = 0 : tensor<20xi32>
      %3723 = stablehlo.add %iterArg_181, %19 : tensor<i32>
      %3724 = stablehlo.broadcast_in_dim %3723, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %3725 = stablehlo.compare  LT, %3722, %3724,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %3726 = stablehlo.broadcast_in_dim %3725, dims = [3] : (tensor<20xi1>) -> tensor<2x1x1x20xi1>
      %3727 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3728 = stablehlo.compare  NE, %3712, %3727,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %3729 = stablehlo.and %3726, %3728 : tensor<2x1x1x20xi1>
      %3730 = stablehlo.convert %3729 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %3731 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3732 = stablehlo.compare  GT, %3730, %3731,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %3733 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3734 = stablehlo.convert %3733 : tensor<2x1x1x20xf32>
      %3735 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3736 = stablehlo.select %3732, %3734, %3735 : tensor<2x1x1x20xi1>, tensor<2x1x1x20xf32>
      %3737 = stablehlo.sqrt %12 : tensor<f32>
      %3738 = stablehlo.convert %3737 : tensor<f32>
      %3739 = stablehlo.broadcast_in_dim %3738, dims = [] : (tensor<f32>) -> tensor<2x1x12x64xf32>
      %3740 = stablehlo.divide %3699, %3739 : tensor<2x1x12x64xf32>
      %3741 = stablehlo.dot_general %3740, %3716, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x1x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x1x20xf32>
      %3742 = stablehlo.broadcast_in_dim %3736, dims = [0, 1, 2, 3] : (tensor<2x1x1x20xf32>) -> tensor<2x12x1x20xf32>
      %3743 = stablehlo.add %3741, %3742 : tensor<2x12x1x20xf32>
      %3744 = stablehlo.reduce(%3743 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %3745 = stablehlo.broadcast_in_dim %3744, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %3746 = stablehlo.broadcast_in_dim %3745, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %3747 = stablehlo.subtract %3743, %3746 : tensor<2x12x1x20xf32>
      %3748 = stablehlo.exponential %3747 : tensor<2x12x1x20xf32>
      %3749 = stablehlo.reduce(%3748 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %3750 = stablehlo.broadcast_in_dim %3749, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %3751 = stablehlo.broadcast_in_dim %3750, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %3752 = stablehlo.divide %3748, %3751 : tensor<2x12x1x20xf32>
      %3753 = stablehlo.dot_general %3720, %3752, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x1x20xf32>) -> tensor<2x12x64x1xf32>
      %3754 = stablehlo.transpose %3753, dims = [0, 3, 1, 2] : (tensor<2x12x64x1xf32>) -> tensor<2x1x12x64xf32>
      %3755 = stablehlo.reshape %3754 : (tensor<2x1x12x64xf32>) -> tensor<2x1x768xf32>
      %3756 = stablehlo.transpose %iterArg_89, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %3757 = stablehlo.convert %3756 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %3758 = stablehlo.dot_general %3755, %3757, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x768xf32>) -> tensor<2x1x768xf32>
      %3759 = stablehlo.convert %iterArg_90 : (tensor<768xf16>) -> tensor<768xf32>
      %3760 = stablehlo.broadcast_in_dim %3759, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3761 = stablehlo.broadcast_in_dim %3760, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3762 = stablehlo.add %3758, %3761 : tensor<2x1x768xf32>
      %3763 = stablehlo.add %3762, %3652 : tensor<2x1x768xf32>
      %3764 = stablehlo.multiply %3763, %3763 : tensor<2x1x768xf32>
      %3765 = stablehlo.reduce(%3763 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3766 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3767 = stablehlo.divide %3765, %3766 : tensor<2x1xf32>
      %3768 = stablehlo.reduce(%3764 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3769 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3770 = stablehlo.divide %3768, %3769 : tensor<2x1xf32>
      %3771 = stablehlo.multiply %3767, %3767 : tensor<2x1xf32>
      %3772 = stablehlo.subtract %3770, %3771 : tensor<2x1xf32>
      %3773 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3774 = stablehlo.maximum %3773, %3772 : tensor<2x1xf32>
      %3775 = stablehlo.broadcast_in_dim %3767, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3776 = stablehlo.broadcast_in_dim %3774, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3777 = stablehlo.broadcast_in_dim %3775, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3778 = stablehlo.subtract %3763, %3777 : tensor<2x1x768xf32>
      %3779 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %3780 = stablehlo.add %3776, %3779 : tensor<2x1x1xf32>
      %3781 = stablehlo.rsqrt %3780 : tensor<2x1x1xf32>
      %3782 = stablehlo.reshape %iterArg_91 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3783 = stablehlo.convert %3782 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3784 = stablehlo.broadcast_in_dim %3781, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3785 = stablehlo.broadcast_in_dim %3783, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3786 = stablehlo.multiply %3784, %3785 : tensor<2x1x768xf32>
      %3787 = stablehlo.multiply %3778, %3786 : tensor<2x1x768xf32>
      %3788 = stablehlo.reshape %iterArg_92 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3789 = stablehlo.convert %3788 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3790 = stablehlo.broadcast_in_dim %3789, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3791 = stablehlo.add %3787, %3790 : tensor<2x1x768xf32>
      %3792 = stablehlo.transpose %iterArg_93, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %3793 = stablehlo.convert %3792 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %3794 = stablehlo.dot_general %3791, %3793, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x3072xf32>) -> tensor<2x1x3072xf32>
      %3795 = stablehlo.convert %iterArg_94 : (tensor<3072xf16>) -> tensor<3072xf32>
      %3796 = stablehlo.broadcast_in_dim %3795, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %3797 = stablehlo.broadcast_in_dim %3796, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
      %3798 = stablehlo.add %3794, %3797 : tensor<2x1x3072xf32>
      %3799 = stablehlo.multiply %3798, %3798 : tensor<2x1x3072xf32>
      %3800 = stablehlo.multiply %3798, %3799 : tensor<2x1x3072xf32>
      %3801 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3802 = stablehlo.multiply %3801, %3800 : tensor<2x1x3072xf32>
      %3803 = stablehlo.add %3798, %3802 : tensor<2x1x3072xf32>
      %3804 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3805 = stablehlo.multiply %3804, %3803 : tensor<2x1x3072xf32>
      %3806 = stablehlo.tanh %3805 : tensor<2x1x3072xf32>
      %3807 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3808 = stablehlo.add %3807, %3806 : tensor<2x1x3072xf32>
      %3809 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3810 = stablehlo.multiply %3809, %3808 : tensor<2x1x3072xf32>
      %3811 = stablehlo.multiply %3798, %3810 : tensor<2x1x3072xf32>
      %3812 = stablehlo.transpose %iterArg_95, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %3813 = stablehlo.convert %3812 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %3814 = stablehlo.dot_general %3811, %3813, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x3072xf32>, tensor<3072x768xf32>) -> tensor<2x1x768xf32>
      %3815 = stablehlo.convert %iterArg_96 : (tensor<768xf16>) -> tensor<768xf32>
      %3816 = stablehlo.broadcast_in_dim %3815, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3817 = stablehlo.broadcast_in_dim %3816, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3818 = stablehlo.add %3814, %3817 : tensor<2x1x768xf32>
      %3819 = stablehlo.add %3763, %3818 : tensor<2x1x768xf32>
      %3820 = stablehlo.multiply %3819, %3819 : tensor<2x1x768xf32>
      %3821 = stablehlo.reduce(%3819 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3822 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3823 = stablehlo.divide %3821, %3822 : tensor<2x1xf32>
      %3824 = stablehlo.reduce(%3820 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3825 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3826 = stablehlo.divide %3824, %3825 : tensor<2x1xf32>
      %3827 = stablehlo.multiply %3823, %3823 : tensor<2x1xf32>
      %3828 = stablehlo.subtract %3826, %3827 : tensor<2x1xf32>
      %3829 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3830 = stablehlo.maximum %3829, %3828 : tensor<2x1xf32>
      %3831 = stablehlo.broadcast_in_dim %3823, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3832 = stablehlo.broadcast_in_dim %3830, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3833 = stablehlo.broadcast_in_dim %3831, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3834 = stablehlo.subtract %3819, %3833 : tensor<2x1x768xf32>
      %3835 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %3836 = stablehlo.add %3832, %3835 : tensor<2x1x1xf32>
      %3837 = stablehlo.rsqrt %3836 : tensor<2x1x1xf32>
      %3838 = stablehlo.reshape %iterArg_97 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3839 = stablehlo.convert %3838 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3840 = stablehlo.broadcast_in_dim %3837, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3841 = stablehlo.broadcast_in_dim %3839, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3842 = stablehlo.multiply %3840, %3841 : tensor<2x1x768xf32>
      %3843 = stablehlo.multiply %3834, %3842 : tensor<2x1x768xf32>
      %3844 = stablehlo.reshape %iterArg_98 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3845 = stablehlo.convert %3844 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3846 = stablehlo.broadcast_in_dim %3845, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3847 = stablehlo.add %3843, %3846 : tensor<2x1x768xf32>
      %3848 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %3849 = stablehlo.broadcast_in_dim %3848, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %3850 = stablehlo.broadcast_in_dim %3849, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %3851 = stablehlo.broadcast_in_dim %3849, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %3852 = stablehlo.broadcast_in_dim %3850, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %3853 = stablehlo.broadcast_in_dim %3851, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %3854 = stablehlo.compare  GE, %3852, %3853,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %3855 = stablehlo.broadcast_in_dim %3854, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %3856 = stablehlo.transpose %iterArg_99, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %3857 = stablehlo.convert %3856 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %3858 = stablehlo.dot_general %3847, %3857, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x2304xf32>) -> tensor<2x1x2304xf32>
      %3859 = stablehlo.convert %iterArg_100 : (tensor<2304xf16>) -> tensor<2304xf32>
      %3860 = stablehlo.broadcast_in_dim %3859, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %3861 = stablehlo.broadcast_in_dim %3860, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x1x2304xf32>
      %3862 = stablehlo.add %3858, %3861 : tensor<2x1x2304xf32>
      %3863 = stablehlo.slice %3862 [0:2, 0:1, 0:768] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3864 = stablehlo.slice %3862 [0:2, 0:1, 768:1536] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3865 = stablehlo.slice %3862 [0:2, 0:1, 1536:2304] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %3866 = stablehlo.reshape %3863 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3867 = stablehlo.reshape %3864 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3868 = stablehlo.reshape %3865 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %3869 = stablehlo.compare  LT, %iterArg_184, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3870 = stablehlo.add %iterArg_184, %16 : tensor<i32>
      %3871 = stablehlo.select %3869, %3870, %iterArg_184 : tensor<i1>, tensor<i32>
      %3872 = stablehlo.dynamic_slice %3855, %22, %22, %3871, %22, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %3873 = stablehlo.reshape %3872 : (tensor<1x1x1x20xi1>) -> tensor<1x1x20xi1>
      %3874 = stablehlo.broadcast_in_dim %3873, dims = [1, 2, 3] : (tensor<1x1x20xi1>) -> tensor<2x1x1x20xi1>
      %3875 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
      %3876 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1x20xi32>
      %3877 = stablehlo.compare  NE, %3875, %3876,  SIGNED : (tensor<2x1x1x20xi32>, tensor<2x1x1x20xi32>) -> tensor<2x1x1x20xi1>
      %3878 = stablehlo.and %3877, %3874 : tensor<2x1x1x20xi1>
      %3879 = stablehlo.convert %3878 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %3880 = stablehlo.compare  LT, %iterArg_184, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3881 = stablehlo.add %iterArg_184, %15 : tensor<i32>
      %3882 = stablehlo.select %3880, %3881, %iterArg_184 : tensor<i1>, tensor<i32>
      %3883 = stablehlo.dynamic_update_slice %iterArg_185, %3867, %22, %3882, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %3884 = stablehlo.compare  LT, %iterArg_184, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %3885 = stablehlo.add %iterArg_184, %15 : tensor<i32>
      %3886 = stablehlo.select %3884, %3885, %iterArg_184 : tensor<i1>, tensor<i32>
      %3887 = stablehlo.dynamic_update_slice %iterArg_186, %3868, %22, %3886, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %3888 = stablehlo.add %iterArg_184, %19 : tensor<i32>
      %3889 = stablehlo.iota dim = 0 : tensor<20xi32>
      %3890 = stablehlo.add %iterArg_184, %19 : tensor<i32>
      %3891 = stablehlo.broadcast_in_dim %3890, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %3892 = stablehlo.compare  LT, %3889, %3891,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %3893 = stablehlo.broadcast_in_dim %3892, dims = [3] : (tensor<20xi1>) -> tensor<2x1x1x20xi1>
      %3894 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3895 = stablehlo.compare  NE, %3879, %3894,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %3896 = stablehlo.and %3893, %3895 : tensor<2x1x1x20xi1>
      %3897 = stablehlo.convert %3896 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %3898 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3899 = stablehlo.compare  GT, %3897, %3898,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %3900 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3901 = stablehlo.convert %3900 : tensor<2x1x1x20xf32>
      %3902 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %3903 = stablehlo.select %3899, %3901, %3902 : tensor<2x1x1x20xi1>, tensor<2x1x1x20xf32>
      %3904 = stablehlo.sqrt %12 : tensor<f32>
      %3905 = stablehlo.convert %3904 : tensor<f32>
      %3906 = stablehlo.broadcast_in_dim %3905, dims = [] : (tensor<f32>) -> tensor<2x1x12x64xf32>
      %3907 = stablehlo.divide %3866, %3906 : tensor<2x1x12x64xf32>
      %3908 = stablehlo.dot_general %3907, %3883, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x1x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x1x20xf32>
      %3909 = stablehlo.broadcast_in_dim %3903, dims = [0, 1, 2, 3] : (tensor<2x1x1x20xf32>) -> tensor<2x12x1x20xf32>
      %3910 = stablehlo.add %3908, %3909 : tensor<2x12x1x20xf32>
      %3911 = stablehlo.reduce(%3910 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %3912 = stablehlo.broadcast_in_dim %3911, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %3913 = stablehlo.broadcast_in_dim %3912, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %3914 = stablehlo.subtract %3910, %3913 : tensor<2x12x1x20xf32>
      %3915 = stablehlo.exponential %3914 : tensor<2x12x1x20xf32>
      %3916 = stablehlo.reduce(%3915 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %3917 = stablehlo.broadcast_in_dim %3916, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %3918 = stablehlo.broadcast_in_dim %3917, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %3919 = stablehlo.divide %3915, %3918 : tensor<2x12x1x20xf32>
      %3920 = stablehlo.dot_general %3887, %3919, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x1x20xf32>) -> tensor<2x12x64x1xf32>
      %3921 = stablehlo.transpose %3920, dims = [0, 3, 1, 2] : (tensor<2x12x64x1xf32>) -> tensor<2x1x12x64xf32>
      %3922 = stablehlo.reshape %3921 : (tensor<2x1x12x64xf32>) -> tensor<2x1x768xf32>
      %3923 = stablehlo.transpose %iterArg_101, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %3924 = stablehlo.convert %3923 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %3925 = stablehlo.dot_general %3922, %3924, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x768xf32>) -> tensor<2x1x768xf32>
      %3926 = stablehlo.convert %iterArg_102 : (tensor<768xf16>) -> tensor<768xf32>
      %3927 = stablehlo.broadcast_in_dim %3926, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3928 = stablehlo.broadcast_in_dim %3927, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3929 = stablehlo.add %3925, %3928 : tensor<2x1x768xf32>
      %3930 = stablehlo.add %3929, %3819 : tensor<2x1x768xf32>
      %3931 = stablehlo.multiply %3930, %3930 : tensor<2x1x768xf32>
      %3932 = stablehlo.reduce(%3930 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3933 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3934 = stablehlo.divide %3932, %3933 : tensor<2x1xf32>
      %3935 = stablehlo.reduce(%3931 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3936 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3937 = stablehlo.divide %3935, %3936 : tensor<2x1xf32>
      %3938 = stablehlo.multiply %3934, %3934 : tensor<2x1xf32>
      %3939 = stablehlo.subtract %3937, %3938 : tensor<2x1xf32>
      %3940 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3941 = stablehlo.maximum %3940, %3939 : tensor<2x1xf32>
      %3942 = stablehlo.broadcast_in_dim %3934, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3943 = stablehlo.broadcast_in_dim %3941, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3944 = stablehlo.broadcast_in_dim %3942, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3945 = stablehlo.subtract %3930, %3944 : tensor<2x1x768xf32>
      %3946 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %3947 = stablehlo.add %3943, %3946 : tensor<2x1x1xf32>
      %3948 = stablehlo.rsqrt %3947 : tensor<2x1x1xf32>
      %3949 = stablehlo.reshape %iterArg_103 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3950 = stablehlo.convert %3949 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3951 = stablehlo.broadcast_in_dim %3948, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %3952 = stablehlo.broadcast_in_dim %3950, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3953 = stablehlo.multiply %3951, %3952 : tensor<2x1x768xf32>
      %3954 = stablehlo.multiply %3945, %3953 : tensor<2x1x768xf32>
      %3955 = stablehlo.reshape %iterArg_104 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %3956 = stablehlo.convert %3955 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %3957 = stablehlo.broadcast_in_dim %3956, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3958 = stablehlo.add %3954, %3957 : tensor<2x1x768xf32>
      %3959 = stablehlo.transpose %iterArg_105, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %3960 = stablehlo.convert %3959 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %3961 = stablehlo.dot_general %3958, %3960, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x3072xf32>) -> tensor<2x1x3072xf32>
      %3962 = stablehlo.convert %iterArg_106 : (tensor<3072xf16>) -> tensor<3072xf32>
      %3963 = stablehlo.broadcast_in_dim %3962, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %3964 = stablehlo.broadcast_in_dim %3963, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
      %3965 = stablehlo.add %3961, %3964 : tensor<2x1x3072xf32>
      %3966 = stablehlo.multiply %3965, %3965 : tensor<2x1x3072xf32>
      %3967 = stablehlo.multiply %3965, %3966 : tensor<2x1x3072xf32>
      %3968 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3969 = stablehlo.multiply %3968, %3967 : tensor<2x1x3072xf32>
      %3970 = stablehlo.add %3965, %3969 : tensor<2x1x3072xf32>
      %3971 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3972 = stablehlo.multiply %3971, %3970 : tensor<2x1x3072xf32>
      %3973 = stablehlo.tanh %3972 : tensor<2x1x3072xf32>
      %3974 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3975 = stablehlo.add %3974, %3973 : tensor<2x1x3072xf32>
      %3976 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3977 = stablehlo.multiply %3976, %3975 : tensor<2x1x3072xf32>
      %3978 = stablehlo.multiply %3965, %3977 : tensor<2x1x3072xf32>
      %3979 = stablehlo.transpose %iterArg_107, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %3980 = stablehlo.convert %3979 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %3981 = stablehlo.dot_general %3978, %3980, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x3072xf32>, tensor<3072x768xf32>) -> tensor<2x1x768xf32>
      %3982 = stablehlo.convert %iterArg_108 : (tensor<768xf16>) -> tensor<768xf32>
      %3983 = stablehlo.broadcast_in_dim %3982, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %3984 = stablehlo.broadcast_in_dim %3983, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %3985 = stablehlo.add %3981, %3984 : tensor<2x1x768xf32>
      %3986 = stablehlo.add %3930, %3985 : tensor<2x1x768xf32>
      %3987 = stablehlo.multiply %3986, %3986 : tensor<2x1x768xf32>
      %3988 = stablehlo.reduce(%3986 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3989 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3990 = stablehlo.divide %3988, %3989 : tensor<2x1xf32>
      %3991 = stablehlo.reduce(%3987 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %3992 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3993 = stablehlo.divide %3991, %3992 : tensor<2x1xf32>
      %3994 = stablehlo.multiply %3990, %3990 : tensor<2x1xf32>
      %3995 = stablehlo.subtract %3993, %3994 : tensor<2x1xf32>
      %3996 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %3997 = stablehlo.maximum %3996, %3995 : tensor<2x1xf32>
      %3998 = stablehlo.broadcast_in_dim %3990, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %3999 = stablehlo.broadcast_in_dim %3997, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4000 = stablehlo.broadcast_in_dim %3998, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4001 = stablehlo.subtract %3986, %4000 : tensor<2x1x768xf32>
      %4002 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %4003 = stablehlo.add %3999, %4002 : tensor<2x1x1xf32>
      %4004 = stablehlo.rsqrt %4003 : tensor<2x1x1xf32>
      %4005 = stablehlo.reshape %iterArg_109 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4006 = stablehlo.convert %4005 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4007 = stablehlo.broadcast_in_dim %4004, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4008 = stablehlo.broadcast_in_dim %4006, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4009 = stablehlo.multiply %4007, %4008 : tensor<2x1x768xf32>
      %4010 = stablehlo.multiply %4001, %4009 : tensor<2x1x768xf32>
      %4011 = stablehlo.reshape %iterArg_110 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4012 = stablehlo.convert %4011 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4013 = stablehlo.broadcast_in_dim %4012, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4014 = stablehlo.add %4010, %4013 : tensor<2x1x768xf32>
      %4015 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %4016 = stablehlo.broadcast_in_dim %4015, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %4017 = stablehlo.broadcast_in_dim %4016, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %4018 = stablehlo.broadcast_in_dim %4016, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %4019 = stablehlo.broadcast_in_dim %4017, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %4020 = stablehlo.broadcast_in_dim %4018, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %4021 = stablehlo.compare  GE, %4019, %4020,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %4022 = stablehlo.broadcast_in_dim %4021, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %4023 = stablehlo.transpose %iterArg_111, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %4024 = stablehlo.convert %4023 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %4025 = stablehlo.dot_general %4014, %4024, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x2304xf32>) -> tensor<2x1x2304xf32>
      %4026 = stablehlo.convert %iterArg_112 : (tensor<2304xf16>) -> tensor<2304xf32>
      %4027 = stablehlo.broadcast_in_dim %4026, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %4028 = stablehlo.broadcast_in_dim %4027, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x1x2304xf32>
      %4029 = stablehlo.add %4025, %4028 : tensor<2x1x2304xf32>
      %4030 = stablehlo.slice %4029 [0:2, 0:1, 0:768] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %4031 = stablehlo.slice %4029 [0:2, 0:1, 768:1536] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %4032 = stablehlo.slice %4029 [0:2, 0:1, 1536:2304] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %4033 = stablehlo.reshape %4030 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %4034 = stablehlo.reshape %4031 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %4035 = stablehlo.reshape %4032 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %4036 = stablehlo.compare  LT, %iterArg_187, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4037 = stablehlo.add %iterArg_187, %16 : tensor<i32>
      %4038 = stablehlo.select %4036, %4037, %iterArg_187 : tensor<i1>, tensor<i32>
      %4039 = stablehlo.dynamic_slice %4022, %22, %22, %4038, %22, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %4040 = stablehlo.reshape %4039 : (tensor<1x1x1x20xi1>) -> tensor<1x1x20xi1>
      %4041 = stablehlo.broadcast_in_dim %4040, dims = [1, 2, 3] : (tensor<1x1x20xi1>) -> tensor<2x1x1x20xi1>
      %4042 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
      %4043 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1x20xi32>
      %4044 = stablehlo.compare  NE, %4042, %4043,  SIGNED : (tensor<2x1x1x20xi32>, tensor<2x1x1x20xi32>) -> tensor<2x1x1x20xi1>
      %4045 = stablehlo.and %4044, %4041 : tensor<2x1x1x20xi1>
      %4046 = stablehlo.convert %4045 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %4047 = stablehlo.compare  LT, %iterArg_187, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4048 = stablehlo.add %iterArg_187, %15 : tensor<i32>
      %4049 = stablehlo.select %4047, %4048, %iterArg_187 : tensor<i1>, tensor<i32>
      %4050 = stablehlo.dynamic_update_slice %iterArg_188, %4034, %22, %4049, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %4051 = stablehlo.compare  LT, %iterArg_187, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4052 = stablehlo.add %iterArg_187, %15 : tensor<i32>
      %4053 = stablehlo.select %4051, %4052, %iterArg_187 : tensor<i1>, tensor<i32>
      %4054 = stablehlo.dynamic_update_slice %iterArg_189, %4035, %22, %4053, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %4055 = stablehlo.add %iterArg_187, %19 : tensor<i32>
      %4056 = stablehlo.iota dim = 0 : tensor<20xi32>
      %4057 = stablehlo.add %iterArg_187, %19 : tensor<i32>
      %4058 = stablehlo.broadcast_in_dim %4057, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %4059 = stablehlo.compare  LT, %4056, %4058,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %4060 = stablehlo.broadcast_in_dim %4059, dims = [3] : (tensor<20xi1>) -> tensor<2x1x1x20xi1>
      %4061 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %4062 = stablehlo.compare  NE, %4046, %4061,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %4063 = stablehlo.and %4060, %4062 : tensor<2x1x1x20xi1>
      %4064 = stablehlo.convert %4063 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %4065 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %4066 = stablehlo.compare  GT, %4064, %4065,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %4067 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %4068 = stablehlo.convert %4067 : tensor<2x1x1x20xf32>
      %4069 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %4070 = stablehlo.select %4066, %4068, %4069 : tensor<2x1x1x20xi1>, tensor<2x1x1x20xf32>
      %4071 = stablehlo.sqrt %12 : tensor<f32>
      %4072 = stablehlo.convert %4071 : tensor<f32>
      %4073 = stablehlo.broadcast_in_dim %4072, dims = [] : (tensor<f32>) -> tensor<2x1x12x64xf32>
      %4074 = stablehlo.divide %4033, %4073 : tensor<2x1x12x64xf32>
      %4075 = stablehlo.dot_general %4074, %4050, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x1x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x1x20xf32>
      %4076 = stablehlo.broadcast_in_dim %4070, dims = [0, 1, 2, 3] : (tensor<2x1x1x20xf32>) -> tensor<2x12x1x20xf32>
      %4077 = stablehlo.add %4075, %4076 : tensor<2x12x1x20xf32>
      %4078 = stablehlo.reduce(%4077 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %4079 = stablehlo.broadcast_in_dim %4078, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %4080 = stablehlo.broadcast_in_dim %4079, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %4081 = stablehlo.subtract %4077, %4080 : tensor<2x12x1x20xf32>
      %4082 = stablehlo.exponential %4081 : tensor<2x12x1x20xf32>
      %4083 = stablehlo.reduce(%4082 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %4084 = stablehlo.broadcast_in_dim %4083, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %4085 = stablehlo.broadcast_in_dim %4084, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %4086 = stablehlo.divide %4082, %4085 : tensor<2x12x1x20xf32>
      %4087 = stablehlo.dot_general %4054, %4086, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x1x20xf32>) -> tensor<2x12x64x1xf32>
      %4088 = stablehlo.transpose %4087, dims = [0, 3, 1, 2] : (tensor<2x12x64x1xf32>) -> tensor<2x1x12x64xf32>
      %4089 = stablehlo.reshape %4088 : (tensor<2x1x12x64xf32>) -> tensor<2x1x768xf32>
      %4090 = stablehlo.transpose %iterArg_113, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %4091 = stablehlo.convert %4090 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %4092 = stablehlo.dot_general %4089, %4091, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x768xf32>) -> tensor<2x1x768xf32>
      %4093 = stablehlo.convert %iterArg_114 : (tensor<768xf16>) -> tensor<768xf32>
      %4094 = stablehlo.broadcast_in_dim %4093, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4095 = stablehlo.broadcast_in_dim %4094, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4096 = stablehlo.add %4092, %4095 : tensor<2x1x768xf32>
      %4097 = stablehlo.add %4096, %3986 : tensor<2x1x768xf32>
      %4098 = stablehlo.multiply %4097, %4097 : tensor<2x1x768xf32>
      %4099 = stablehlo.reduce(%4097 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %4100 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4101 = stablehlo.divide %4099, %4100 : tensor<2x1xf32>
      %4102 = stablehlo.reduce(%4098 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %4103 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4104 = stablehlo.divide %4102, %4103 : tensor<2x1xf32>
      %4105 = stablehlo.multiply %4101, %4101 : tensor<2x1xf32>
      %4106 = stablehlo.subtract %4104, %4105 : tensor<2x1xf32>
      %4107 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4108 = stablehlo.maximum %4107, %4106 : tensor<2x1xf32>
      %4109 = stablehlo.broadcast_in_dim %4101, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4110 = stablehlo.broadcast_in_dim %4108, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4111 = stablehlo.broadcast_in_dim %4109, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4112 = stablehlo.subtract %4097, %4111 : tensor<2x1x768xf32>
      %4113 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %4114 = stablehlo.add %4110, %4113 : tensor<2x1x1xf32>
      %4115 = stablehlo.rsqrt %4114 : tensor<2x1x1xf32>
      %4116 = stablehlo.reshape %iterArg_115 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4117 = stablehlo.convert %4116 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4118 = stablehlo.broadcast_in_dim %4115, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4119 = stablehlo.broadcast_in_dim %4117, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4120 = stablehlo.multiply %4118, %4119 : tensor<2x1x768xf32>
      %4121 = stablehlo.multiply %4112, %4120 : tensor<2x1x768xf32>
      %4122 = stablehlo.reshape %iterArg_116 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4123 = stablehlo.convert %4122 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4124 = stablehlo.broadcast_in_dim %4123, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4125 = stablehlo.add %4121, %4124 : tensor<2x1x768xf32>
      %4126 = stablehlo.transpose %iterArg_117, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %4127 = stablehlo.convert %4126 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %4128 = stablehlo.dot_general %4125, %4127, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x3072xf32>) -> tensor<2x1x3072xf32>
      %4129 = stablehlo.convert %iterArg_118 : (tensor<3072xf16>) -> tensor<3072xf32>
      %4130 = stablehlo.broadcast_in_dim %4129, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %4131 = stablehlo.broadcast_in_dim %4130, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
      %4132 = stablehlo.add %4128, %4131 : tensor<2x1x3072xf32>
      %4133 = stablehlo.multiply %4132, %4132 : tensor<2x1x3072xf32>
      %4134 = stablehlo.multiply %4132, %4133 : tensor<2x1x3072xf32>
      %4135 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %4136 = stablehlo.multiply %4135, %4134 : tensor<2x1x3072xf32>
      %4137 = stablehlo.add %4132, %4136 : tensor<2x1x3072xf32>
      %4138 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %4139 = stablehlo.multiply %4138, %4137 : tensor<2x1x3072xf32>
      %4140 = stablehlo.tanh %4139 : tensor<2x1x3072xf32>
      %4141 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %4142 = stablehlo.add %4141, %4140 : tensor<2x1x3072xf32>
      %4143 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %4144 = stablehlo.multiply %4143, %4142 : tensor<2x1x3072xf32>
      %4145 = stablehlo.multiply %4132, %4144 : tensor<2x1x3072xf32>
      %4146 = stablehlo.transpose %iterArg_119, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %4147 = stablehlo.convert %4146 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %4148 = stablehlo.dot_general %4145, %4147, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x3072xf32>, tensor<3072x768xf32>) -> tensor<2x1x768xf32>
      %4149 = stablehlo.convert %iterArg_120 : (tensor<768xf16>) -> tensor<768xf32>
      %4150 = stablehlo.broadcast_in_dim %4149, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4151 = stablehlo.broadcast_in_dim %4150, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4152 = stablehlo.add %4148, %4151 : tensor<2x1x768xf32>
      %4153 = stablehlo.add %4097, %4152 : tensor<2x1x768xf32>
      %4154 = stablehlo.multiply %4153, %4153 : tensor<2x1x768xf32>
      %4155 = stablehlo.reduce(%4153 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %4156 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4157 = stablehlo.divide %4155, %4156 : tensor<2x1xf32>
      %4158 = stablehlo.reduce(%4154 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %4159 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4160 = stablehlo.divide %4158, %4159 : tensor<2x1xf32>
      %4161 = stablehlo.multiply %4157, %4157 : tensor<2x1xf32>
      %4162 = stablehlo.subtract %4160, %4161 : tensor<2x1xf32>
      %4163 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4164 = stablehlo.maximum %4163, %4162 : tensor<2x1xf32>
      %4165 = stablehlo.broadcast_in_dim %4157, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4166 = stablehlo.broadcast_in_dim %4164, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4167 = stablehlo.broadcast_in_dim %4165, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4168 = stablehlo.subtract %4153, %4167 : tensor<2x1x768xf32>
      %4169 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %4170 = stablehlo.add %4166, %4169 : tensor<2x1x1xf32>
      %4171 = stablehlo.rsqrt %4170 : tensor<2x1x1xf32>
      %4172 = stablehlo.reshape %iterArg_121 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4173 = stablehlo.convert %4172 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4174 = stablehlo.broadcast_in_dim %4171, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4175 = stablehlo.broadcast_in_dim %4173, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4176 = stablehlo.multiply %4174, %4175 : tensor<2x1x768xf32>
      %4177 = stablehlo.multiply %4168, %4176 : tensor<2x1x768xf32>
      %4178 = stablehlo.reshape %iterArg_122 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4179 = stablehlo.convert %4178 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4180 = stablehlo.broadcast_in_dim %4179, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4181 = stablehlo.add %4177, %4180 : tensor<2x1x768xf32>
      %4182 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %4183 = stablehlo.broadcast_in_dim %4182, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %4184 = stablehlo.broadcast_in_dim %4183, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %4185 = stablehlo.broadcast_in_dim %4183, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %4186 = stablehlo.broadcast_in_dim %4184, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %4187 = stablehlo.broadcast_in_dim %4185, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %4188 = stablehlo.compare  GE, %4186, %4187,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %4189 = stablehlo.broadcast_in_dim %4188, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %4190 = stablehlo.transpose %iterArg_123, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %4191 = stablehlo.convert %4190 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %4192 = stablehlo.dot_general %4181, %4191, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x2304xf32>) -> tensor<2x1x2304xf32>
      %4193 = stablehlo.convert %iterArg_124 : (tensor<2304xf16>) -> tensor<2304xf32>
      %4194 = stablehlo.broadcast_in_dim %4193, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %4195 = stablehlo.broadcast_in_dim %4194, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x1x2304xf32>
      %4196 = stablehlo.add %4192, %4195 : tensor<2x1x2304xf32>
      %4197 = stablehlo.slice %4196 [0:2, 0:1, 0:768] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %4198 = stablehlo.slice %4196 [0:2, 0:1, 768:1536] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %4199 = stablehlo.slice %4196 [0:2, 0:1, 1536:2304] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %4200 = stablehlo.reshape %4197 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %4201 = stablehlo.reshape %4198 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %4202 = stablehlo.reshape %4199 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %4203 = stablehlo.compare  LT, %iterArg_160, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4204 = stablehlo.add %iterArg_160, %16 : tensor<i32>
      %4205 = stablehlo.select %4203, %4204, %iterArg_160 : tensor<i1>, tensor<i32>
      %4206 = stablehlo.dynamic_slice %4189, %22, %22, %4205, %22, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %4207 = stablehlo.reshape %4206 : (tensor<1x1x1x20xi1>) -> tensor<1x1x20xi1>
      %4208 = stablehlo.broadcast_in_dim %4207, dims = [1, 2, 3] : (tensor<1x1x20xi1>) -> tensor<2x1x1x20xi1>
      %4209 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
      %4210 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1x20xi32>
      %4211 = stablehlo.compare  NE, %4209, %4210,  SIGNED : (tensor<2x1x1x20xi32>, tensor<2x1x1x20xi32>) -> tensor<2x1x1x20xi1>
      %4212 = stablehlo.and %4211, %4208 : tensor<2x1x1x20xi1>
      %4213 = stablehlo.convert %4212 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %4214 = stablehlo.compare  LT, %iterArg_160, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4215 = stablehlo.add %iterArg_160, %15 : tensor<i32>
      %4216 = stablehlo.select %4214, %4215, %iterArg_160 : tensor<i1>, tensor<i32>
      %4217 = stablehlo.dynamic_update_slice %iterArg_161, %4201, %22, %4216, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %4218 = stablehlo.compare  LT, %iterArg_160, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4219 = stablehlo.add %iterArg_160, %15 : tensor<i32>
      %4220 = stablehlo.select %4218, %4219, %iterArg_160 : tensor<i1>, tensor<i32>
      %4221 = stablehlo.dynamic_update_slice %iterArg_162, %4202, %22, %4220, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %4222 = stablehlo.add %iterArg_160, %19 : tensor<i32>
      %4223 = stablehlo.iota dim = 0 : tensor<20xi32>
      %4224 = stablehlo.add %iterArg_160, %19 : tensor<i32>
      %4225 = stablehlo.broadcast_in_dim %4224, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %4226 = stablehlo.compare  LT, %4223, %4225,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %4227 = stablehlo.broadcast_in_dim %4226, dims = [3] : (tensor<20xi1>) -> tensor<2x1x1x20xi1>
      %4228 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %4229 = stablehlo.compare  NE, %4213, %4228,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %4230 = stablehlo.and %4227, %4229 : tensor<2x1x1x20xi1>
      %4231 = stablehlo.convert %4230 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %4232 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %4233 = stablehlo.compare  GT, %4231, %4232,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %4234 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %4235 = stablehlo.convert %4234 : tensor<2x1x1x20xf32>
      %4236 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %4237 = stablehlo.select %4233, %4235, %4236 : tensor<2x1x1x20xi1>, tensor<2x1x1x20xf32>
      %4238 = stablehlo.sqrt %12 : tensor<f32>
      %4239 = stablehlo.convert %4238 : tensor<f32>
      %4240 = stablehlo.broadcast_in_dim %4239, dims = [] : (tensor<f32>) -> tensor<2x1x12x64xf32>
      %4241 = stablehlo.divide %4200, %4240 : tensor<2x1x12x64xf32>
      %4242 = stablehlo.dot_general %4241, %4217, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x1x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x1x20xf32>
      %4243 = stablehlo.broadcast_in_dim %4237, dims = [0, 1, 2, 3] : (tensor<2x1x1x20xf32>) -> tensor<2x12x1x20xf32>
      %4244 = stablehlo.add %4242, %4243 : tensor<2x12x1x20xf32>
      %4245 = stablehlo.reduce(%4244 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %4246 = stablehlo.broadcast_in_dim %4245, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %4247 = stablehlo.broadcast_in_dim %4246, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %4248 = stablehlo.subtract %4244, %4247 : tensor<2x12x1x20xf32>
      %4249 = stablehlo.exponential %4248 : tensor<2x12x1x20xf32>
      %4250 = stablehlo.reduce(%4249 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %4251 = stablehlo.broadcast_in_dim %4250, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %4252 = stablehlo.broadcast_in_dim %4251, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %4253 = stablehlo.divide %4249, %4252 : tensor<2x12x1x20xf32>
      %4254 = stablehlo.dot_general %4221, %4253, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x1x20xf32>) -> tensor<2x12x64x1xf32>
      %4255 = stablehlo.transpose %4254, dims = [0, 3, 1, 2] : (tensor<2x12x64x1xf32>) -> tensor<2x1x12x64xf32>
      %4256 = stablehlo.reshape %4255 : (tensor<2x1x12x64xf32>) -> tensor<2x1x768xf32>
      %4257 = stablehlo.transpose %iterArg_125, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %4258 = stablehlo.convert %4257 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %4259 = stablehlo.dot_general %4256, %4258, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x768xf32>) -> tensor<2x1x768xf32>
      %4260 = stablehlo.convert %iterArg_126 : (tensor<768xf16>) -> tensor<768xf32>
      %4261 = stablehlo.broadcast_in_dim %4260, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4262 = stablehlo.broadcast_in_dim %4261, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4263 = stablehlo.add %4259, %4262 : tensor<2x1x768xf32>
      %4264 = stablehlo.add %4263, %4153 : tensor<2x1x768xf32>
      %4265 = stablehlo.multiply %4264, %4264 : tensor<2x1x768xf32>
      %4266 = stablehlo.reduce(%4264 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %4267 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4268 = stablehlo.divide %4266, %4267 : tensor<2x1xf32>
      %4269 = stablehlo.reduce(%4265 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %4270 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4271 = stablehlo.divide %4269, %4270 : tensor<2x1xf32>
      %4272 = stablehlo.multiply %4268, %4268 : tensor<2x1xf32>
      %4273 = stablehlo.subtract %4271, %4272 : tensor<2x1xf32>
      %4274 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4275 = stablehlo.maximum %4274, %4273 : tensor<2x1xf32>
      %4276 = stablehlo.broadcast_in_dim %4268, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4277 = stablehlo.broadcast_in_dim %4275, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4278 = stablehlo.broadcast_in_dim %4276, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4279 = stablehlo.subtract %4264, %4278 : tensor<2x1x768xf32>
      %4280 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %4281 = stablehlo.add %4277, %4280 : tensor<2x1x1xf32>
      %4282 = stablehlo.rsqrt %4281 : tensor<2x1x1xf32>
      %4283 = stablehlo.reshape %iterArg_127 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4284 = stablehlo.convert %4283 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4285 = stablehlo.broadcast_in_dim %4282, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4286 = stablehlo.broadcast_in_dim %4284, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4287 = stablehlo.multiply %4285, %4286 : tensor<2x1x768xf32>
      %4288 = stablehlo.multiply %4279, %4287 : tensor<2x1x768xf32>
      %4289 = stablehlo.reshape %iterArg_128 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4290 = stablehlo.convert %4289 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4291 = stablehlo.broadcast_in_dim %4290, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4292 = stablehlo.add %4288, %4291 : tensor<2x1x768xf32>
      %4293 = stablehlo.transpose %iterArg_129, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %4294 = stablehlo.convert %4293 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %4295 = stablehlo.dot_general %4292, %4294, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x3072xf32>) -> tensor<2x1x3072xf32>
      %4296 = stablehlo.convert %iterArg_130 : (tensor<3072xf16>) -> tensor<3072xf32>
      %4297 = stablehlo.broadcast_in_dim %4296, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %4298 = stablehlo.broadcast_in_dim %4297, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
      %4299 = stablehlo.add %4295, %4298 : tensor<2x1x3072xf32>
      %4300 = stablehlo.multiply %4299, %4299 : tensor<2x1x3072xf32>
      %4301 = stablehlo.multiply %4299, %4300 : tensor<2x1x3072xf32>
      %4302 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %4303 = stablehlo.multiply %4302, %4301 : tensor<2x1x3072xf32>
      %4304 = stablehlo.add %4299, %4303 : tensor<2x1x3072xf32>
      %4305 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %4306 = stablehlo.multiply %4305, %4304 : tensor<2x1x3072xf32>
      %4307 = stablehlo.tanh %4306 : tensor<2x1x3072xf32>
      %4308 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %4309 = stablehlo.add %4308, %4307 : tensor<2x1x3072xf32>
      %4310 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %4311 = stablehlo.multiply %4310, %4309 : tensor<2x1x3072xf32>
      %4312 = stablehlo.multiply %4299, %4311 : tensor<2x1x3072xf32>
      %4313 = stablehlo.transpose %iterArg_131, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %4314 = stablehlo.convert %4313 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %4315 = stablehlo.dot_general %4312, %4314, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x3072xf32>, tensor<3072x768xf32>) -> tensor<2x1x768xf32>
      %4316 = stablehlo.convert %iterArg_132 : (tensor<768xf16>) -> tensor<768xf32>
      %4317 = stablehlo.broadcast_in_dim %4316, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4318 = stablehlo.broadcast_in_dim %4317, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4319 = stablehlo.add %4315, %4318 : tensor<2x1x768xf32>
      %4320 = stablehlo.add %4264, %4319 : tensor<2x1x768xf32>
      %4321 = stablehlo.multiply %4320, %4320 : tensor<2x1x768xf32>
      %4322 = stablehlo.reduce(%4320 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %4323 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4324 = stablehlo.divide %4322, %4323 : tensor<2x1xf32>
      %4325 = stablehlo.reduce(%4321 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %4326 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4327 = stablehlo.divide %4325, %4326 : tensor<2x1xf32>
      %4328 = stablehlo.multiply %4324, %4324 : tensor<2x1xf32>
      %4329 = stablehlo.subtract %4327, %4328 : tensor<2x1xf32>
      %4330 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4331 = stablehlo.maximum %4330, %4329 : tensor<2x1xf32>
      %4332 = stablehlo.broadcast_in_dim %4324, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4333 = stablehlo.broadcast_in_dim %4331, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4334 = stablehlo.broadcast_in_dim %4332, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4335 = stablehlo.subtract %4320, %4334 : tensor<2x1x768xf32>
      %4336 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %4337 = stablehlo.add %4333, %4336 : tensor<2x1x1xf32>
      %4338 = stablehlo.rsqrt %4337 : tensor<2x1x1xf32>
      %4339 = stablehlo.reshape %iterArg_133 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4340 = stablehlo.convert %4339 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4341 = stablehlo.broadcast_in_dim %4338, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4342 = stablehlo.broadcast_in_dim %4340, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4343 = stablehlo.multiply %4341, %4342 : tensor<2x1x768xf32>
      %4344 = stablehlo.multiply %4335, %4343 : tensor<2x1x768xf32>
      %4345 = stablehlo.reshape %iterArg_134 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4346 = stablehlo.convert %4345 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4347 = stablehlo.broadcast_in_dim %4346, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4348 = stablehlo.add %4344, %4347 : tensor<2x1x768xf32>
      %4349 = stablehlo.iota dim = 0 : tensor<1024xi32>
      %4350 = stablehlo.broadcast_in_dim %4349, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
      %4351 = stablehlo.broadcast_in_dim %4350, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<1x1024x1xi32>
      %4352 = stablehlo.broadcast_in_dim %4350, dims = [0, 2] : (tensor<1x1024xi32>) -> tensor<1x1x1024xi32>
      %4353 = stablehlo.broadcast_in_dim %4351, dims = [0, 1, 2] : (tensor<1x1024x1xi32>) -> tensor<1x1024x1024xi32>
      %4354 = stablehlo.broadcast_in_dim %4352, dims = [0, 1, 2] : (tensor<1x1x1024xi32>) -> tensor<1x1024x1024xi32>
      %4355 = stablehlo.compare  GE, %4353, %4354,  SIGNED : (tensor<1x1024x1024xi32>, tensor<1x1024x1024xi32>) -> tensor<1x1024x1024xi1>
      %4356 = stablehlo.broadcast_in_dim %4355, dims = [0, 2, 3] : (tensor<1x1024x1024xi1>) -> tensor<1x1x1024x1024xi1>
      %4357 = stablehlo.transpose %iterArg_135, dims = [1, 0] : (tensor<2304x768xf16>) -> tensor<768x2304xf16>
      %4358 = stablehlo.convert %4357 : (tensor<768x2304xf16>) -> tensor<768x2304xf32>
      %4359 = stablehlo.dot_general %4348, %4358, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x2304xf32>) -> tensor<2x1x2304xf32>
      %4360 = stablehlo.convert %iterArg_136 : (tensor<2304xf16>) -> tensor<2304xf32>
      %4361 = stablehlo.broadcast_in_dim %4360, dims = [2] : (tensor<2304xf32>) -> tensor<1x1x2304xf32>
      %4362 = stablehlo.broadcast_in_dim %4361, dims = [0, 1, 2] : (tensor<1x1x2304xf32>) -> tensor<2x1x2304xf32>
      %4363 = stablehlo.add %4359, %4362 : tensor<2x1x2304xf32>
      %4364 = stablehlo.slice %4363 [0:2, 0:1, 0:768] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %4365 = stablehlo.slice %4363 [0:2, 0:1, 768:1536] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %4366 = stablehlo.slice %4363 [0:2, 0:1, 1536:2304] : (tensor<2x1x2304xf32>) -> tensor<2x1x768xf32>
      %4367 = stablehlo.reshape %4364 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %4368 = stablehlo.reshape %4365 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %4369 = stablehlo.reshape %4366 : (tensor<2x1x768xf32>) -> tensor<2x1x12x64xf32>
      %4370 = stablehlo.compare  LT, %iterArg_163, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4371 = stablehlo.add %iterArg_163, %16 : tensor<i32>
      %4372 = stablehlo.select %4370, %4371, %iterArg_163 : tensor<i1>, tensor<i32>
      %4373 = stablehlo.dynamic_slice %4356, %22, %22, %4372, %22, sizes = [1, 1, 1, 20] : (tensor<1x1x1024x1024xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %4374 = stablehlo.reshape %4373 : (tensor<1x1x1x20xi1>) -> tensor<1x1x20xi1>
      %4375 = stablehlo.broadcast_in_dim %4374, dims = [1, 2, 3] : (tensor<1x1x20xi1>) -> tensor<2x1x1x20xi1>
      %4376 = stablehlo.broadcast_in_dim %iterArg_153, dims = [0, 3] : (tensor<2x20xi32>) -> tensor<2x1x1x20xi32>
      %4377 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<2x1x1x20xi32>
      %4378 = stablehlo.compare  NE, %4376, %4377,  SIGNED : (tensor<2x1x1x20xi32>, tensor<2x1x1x20xi32>) -> tensor<2x1x1x20xi1>
      %4379 = stablehlo.and %4378, %4375 : tensor<2x1x1x20xi1>
      %4380 = stablehlo.convert %4379 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %4381 = stablehlo.compare  LT, %iterArg_163, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4382 = stablehlo.add %iterArg_163, %15 : tensor<i32>
      %4383 = stablehlo.select %4381, %4382, %iterArg_163 : tensor<i1>, tensor<i32>
      %4384 = stablehlo.dynamic_update_slice %iterArg_164, %4368, %22, %4383, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %4385 = stablehlo.compare  LT, %iterArg_163, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4386 = stablehlo.add %iterArg_163, %15 : tensor<i32>
      %4387 = stablehlo.select %4385, %4386, %iterArg_163 : tensor<i1>, tensor<i32>
      %4388 = stablehlo.dynamic_update_slice %iterArg_165, %4369, %22, %4387, %22, %22 : (tensor<2x20x12x64xf32>, tensor<2x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x20x12x64xf32>
      %4389 = stablehlo.add %iterArg_163, %19 : tensor<i32>
      %4390 = stablehlo.iota dim = 0 : tensor<20xi32>
      %4391 = stablehlo.add %iterArg_163, %19 : tensor<i32>
      %4392 = stablehlo.broadcast_in_dim %4391, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %4393 = stablehlo.compare  LT, %4390, %4392,  SIGNED : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %4394 = stablehlo.broadcast_in_dim %4393, dims = [3] : (tensor<20xi1>) -> tensor<2x1x1x20xi1>
      %4395 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %4396 = stablehlo.compare  NE, %4380, %4395,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %4397 = stablehlo.and %4394, %4396 : tensor<2x1x1x20xi1>
      %4398 = stablehlo.convert %4397 : (tensor<2x1x1x20xi1>) -> tensor<2x1x1x20xf32>
      %4399 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %4400 = stablehlo.compare  GT, %4398, %4399,  FLOAT : (tensor<2x1x1x20xf32>, tensor<2x1x1x20xf32>) -> tensor<2x1x1x20xi1>
      %4401 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %4402 = stablehlo.convert %4401 : tensor<2x1x1x20xf32>
      %4403 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1x1x20xf32>
      %4404 = stablehlo.select %4400, %4402, %4403 : tensor<2x1x1x20xi1>, tensor<2x1x1x20xf32>
      %4405 = stablehlo.sqrt %12 : tensor<f32>
      %4406 = stablehlo.convert %4405 : tensor<f32>
      %4407 = stablehlo.broadcast_in_dim %4406, dims = [] : (tensor<f32>) -> tensor<2x1x12x64xf32>
      %4408 = stablehlo.divide %4367, %4407 : tensor<2x1x12x64xf32>
      %4409 = stablehlo.dot_general %4408, %4384, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x1x12x64xf32>, tensor<2x20x12x64xf32>) -> tensor<2x12x1x20xf32>
      %4410 = stablehlo.broadcast_in_dim %4404, dims = [0, 1, 2, 3] : (tensor<2x1x1x20xf32>) -> tensor<2x12x1x20xf32>
      %4411 = stablehlo.add %4409, %4410 : tensor<2x12x1x20xf32>
      %4412 = stablehlo.reduce(%4411 init: %11) applies stablehlo.maximum across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %4413 = stablehlo.broadcast_in_dim %4412, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %4414 = stablehlo.broadcast_in_dim %4413, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %4415 = stablehlo.subtract %4411, %4414 : tensor<2x12x1x20xf32>
      %4416 = stablehlo.exponential %4415 : tensor<2x12x1x20xf32>
      %4417 = stablehlo.reduce(%4416 init: %20) applies stablehlo.add across dimensions = [3] : (tensor<2x12x1x20xf32>, tensor<f32>) -> tensor<2x12x1xf32>
      %4418 = stablehlo.broadcast_in_dim %4417, dims = [0, 1, 2] : (tensor<2x12x1xf32>) -> tensor<2x12x1x1xf32>
      %4419 = stablehlo.broadcast_in_dim %4418, dims = [0, 1, 2, 3] : (tensor<2x12x1x1xf32>) -> tensor<2x12x1x20xf32>
      %4420 = stablehlo.divide %4416, %4419 : tensor<2x12x1x20xf32>
      %4421 = stablehlo.dot_general %4388, %4420, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x20x12x64xf32>, tensor<2x12x1x20xf32>) -> tensor<2x12x64x1xf32>
      %4422 = stablehlo.transpose %4421, dims = [0, 3, 1, 2] : (tensor<2x12x64x1xf32>) -> tensor<2x1x12x64xf32>
      %4423 = stablehlo.reshape %4422 : (tensor<2x1x12x64xf32>) -> tensor<2x1x768xf32>
      %4424 = stablehlo.transpose %iterArg_137, dims = [1, 0] : (tensor<768x768xf16>) -> tensor<768x768xf16>
      %4425 = stablehlo.convert %4424 : (tensor<768x768xf16>) -> tensor<768x768xf32>
      %4426 = stablehlo.dot_general %4423, %4425, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x768xf32>) -> tensor<2x1x768xf32>
      %4427 = stablehlo.convert %iterArg_138 : (tensor<768xf16>) -> tensor<768xf32>
      %4428 = stablehlo.broadcast_in_dim %4427, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4429 = stablehlo.broadcast_in_dim %4428, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4430 = stablehlo.add %4426, %4429 : tensor<2x1x768xf32>
      %4431 = stablehlo.add %4430, %4320 : tensor<2x1x768xf32>
      %4432 = stablehlo.multiply %4431, %4431 : tensor<2x1x768xf32>
      %4433 = stablehlo.reduce(%4431 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %4434 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4435 = stablehlo.divide %4433, %4434 : tensor<2x1xf32>
      %4436 = stablehlo.reduce(%4432 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %4437 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4438 = stablehlo.divide %4436, %4437 : tensor<2x1xf32>
      %4439 = stablehlo.multiply %4435, %4435 : tensor<2x1xf32>
      %4440 = stablehlo.subtract %4438, %4439 : tensor<2x1xf32>
      %4441 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4442 = stablehlo.maximum %4441, %4440 : tensor<2x1xf32>
      %4443 = stablehlo.broadcast_in_dim %4435, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4444 = stablehlo.broadcast_in_dim %4442, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4445 = stablehlo.broadcast_in_dim %4443, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4446 = stablehlo.subtract %4431, %4445 : tensor<2x1x768xf32>
      %4447 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %4448 = stablehlo.add %4444, %4447 : tensor<2x1x1xf32>
      %4449 = stablehlo.rsqrt %4448 : tensor<2x1x1xf32>
      %4450 = stablehlo.reshape %iterArg_139 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4451 = stablehlo.convert %4450 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4452 = stablehlo.broadcast_in_dim %4449, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4453 = stablehlo.broadcast_in_dim %4451, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4454 = stablehlo.multiply %4452, %4453 : tensor<2x1x768xf32>
      %4455 = stablehlo.multiply %4446, %4454 : tensor<2x1x768xf32>
      %4456 = stablehlo.reshape %iterArg_140 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4457 = stablehlo.convert %4456 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4458 = stablehlo.broadcast_in_dim %4457, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4459 = stablehlo.add %4455, %4458 : tensor<2x1x768xf32>
      %4460 = stablehlo.transpose %iterArg_141, dims = [1, 0] : (tensor<3072x768xf16>) -> tensor<768x3072xf16>
      %4461 = stablehlo.convert %4460 : (tensor<768x3072xf16>) -> tensor<768x3072xf32>
      %4462 = stablehlo.dot_general %4459, %4461, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x3072xf32>) -> tensor<2x1x3072xf32>
      %4463 = stablehlo.convert %iterArg_142 : (tensor<3072xf16>) -> tensor<3072xf32>
      %4464 = stablehlo.broadcast_in_dim %4463, dims = [2] : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %4465 = stablehlo.broadcast_in_dim %4464, dims = [0, 1, 2] : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
      %4466 = stablehlo.add %4462, %4465 : tensor<2x1x3072xf32>
      %4467 = stablehlo.multiply %4466, %4466 : tensor<2x1x3072xf32>
      %4468 = stablehlo.multiply %4466, %4467 : tensor<2x1x3072xf32>
      %4469 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %4470 = stablehlo.multiply %4469, %4468 : tensor<2x1x3072xf32>
      %4471 = stablehlo.add %4466, %4470 : tensor<2x1x3072xf32>
      %4472 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %4473 = stablehlo.multiply %4472, %4471 : tensor<2x1x3072xf32>
      %4474 = stablehlo.tanh %4473 : tensor<2x1x3072xf32>
      %4475 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %4476 = stablehlo.add %4475, %4474 : tensor<2x1x3072xf32>
      %4477 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %4478 = stablehlo.multiply %4477, %4476 : tensor<2x1x3072xf32>
      %4479 = stablehlo.multiply %4466, %4478 : tensor<2x1x3072xf32>
      %4480 = stablehlo.transpose %iterArg_143, dims = [1, 0] : (tensor<768x3072xf16>) -> tensor<3072x768xf16>
      %4481 = stablehlo.convert %4480 : (tensor<3072x768xf16>) -> tensor<3072x768xf32>
      %4482 = stablehlo.dot_general %4479, %4481, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x3072xf32>, tensor<3072x768xf32>) -> tensor<2x1x768xf32>
      %4483 = stablehlo.convert %iterArg_144 : (tensor<768xf16>) -> tensor<768xf32>
      %4484 = stablehlo.broadcast_in_dim %4483, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %4485 = stablehlo.broadcast_in_dim %4484, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4486 = stablehlo.add %4482, %4485 : tensor<2x1x768xf32>
      %4487 = stablehlo.add %4431, %4486 : tensor<2x1x768xf32>
      %4488 = stablehlo.multiply %4487, %4487 : tensor<2x1x768xf32>
      %4489 = stablehlo.reduce(%4487 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %4490 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4491 = stablehlo.divide %4489, %4490 : tensor<2x1xf32>
      %4492 = stablehlo.reduce(%4488 init: %20) applies stablehlo.add across dimensions = [2] : (tensor<2x1x768xf32>, tensor<f32>) -> tensor<2x1xf32>
      %4493 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4494 = stablehlo.divide %4492, %4493 : tensor<2x1xf32>
      %4495 = stablehlo.multiply %4491, %4491 : tensor<2x1xf32>
      %4496 = stablehlo.subtract %4494, %4495 : tensor<2x1xf32>
      %4497 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
      %4498 = stablehlo.maximum %4497, %4496 : tensor<2x1xf32>
      %4499 = stablehlo.broadcast_in_dim %4491, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4500 = stablehlo.broadcast_in_dim %4498, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %4501 = stablehlo.broadcast_in_dim %4499, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4502 = stablehlo.subtract %4487, %4501 : tensor<2x1x768xf32>
      %4503 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
      %4504 = stablehlo.add %4500, %4503 : tensor<2x1x1xf32>
      %4505 = stablehlo.rsqrt %4504 : tensor<2x1x1xf32>
      %4506 = stablehlo.reshape %iterArg_145 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4507 = stablehlo.convert %4506 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4508 = stablehlo.broadcast_in_dim %4505, dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x1x768xf32>
      %4509 = stablehlo.broadcast_in_dim %4507, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4510 = stablehlo.multiply %4508, %4509 : tensor<2x1x768xf32>
      %4511 = stablehlo.multiply %4502, %4510 : tensor<2x1x768xf32>
      %4512 = stablehlo.reshape %iterArg_146 : (tensor<768xf16>) -> tensor<1x1x768xf16>
      %4513 = stablehlo.convert %4512 : (tensor<1x1x768xf16>) -> tensor<1x1x768xf32>
      %4514 = stablehlo.broadcast_in_dim %4513, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<2x1x768xf32>
      %4515 = stablehlo.add %4511, %4514 : tensor<2x1x768xf32>
      %4516 = stablehlo.transpose %iterArg, dims = [1, 0] : (tensor<50257x768xf16>) -> tensor<768x50257xf16>
      %4517 = stablehlo.convert %4516 : (tensor<768x50257xf16>) -> tensor<768x50257xf32>
      %4518 = stablehlo.dot_general %4515, %4517, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x1x768xf32>, tensor<768x50257xf32>) -> tensor<2x1x50257xf32>
      %4519 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4520 = stablehlo.add %22, %6 : tensor<i32>
      %4521 = stablehlo.select %4519, %4520, %22 : tensor<i1>, tensor<i32>
      %4522 = stablehlo.compare  LT, %5, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4523 = stablehlo.add %5, %19 : tensor<i32>
      %4524 = stablehlo.select %4522, %4523, %5 : tensor<i1>, tensor<i32>
      %4525 = stablehlo.compare  LT, %22, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4526 = stablehlo.add %22, %4 : tensor<i32>
      %4527 = stablehlo.select %4525, %4526, %22 : tensor<i1>, tensor<i32>
      %4528 = stablehlo.dynamic_slice %4518, %4521, %4524, %4527, sizes = [2, 1, 50257] : (tensor<2x1x50257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x1x50257xf32>
      %4529 = stablehlo.reshape %4528 : (tensor<2x1x50257xf32>) -> tensor<2x50257xf32>
      %4530 = stablehlo.subtract %iterArg_149, %22 : tensor<i32>
      %4531 = stablehlo.maximum %22, %4530 : tensor<i32>
      %4532 = stablehlo.minimum %19, %4531 : tensor<i32>
      %4533 = stablehlo.subtract %19, %4532 : tensor<i32>
      %4534 = stablehlo.broadcast_in_dim %171, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %4535 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<2xf32>
      %4536 = "stablehlo.scatter"(%4529, %4534, %4535) ({
      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        stablehlo.return %arg3 : tensor<f32>
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<2x50257xf32>, tensor<1xi32>, tensor<2xf32>) -> tensor<2x50257xf32>
      %4537 = stablehlo.compare  NE, %4533, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4538 = stablehlo.broadcast_in_dim %4537, dims = [] : (tensor<i1>) -> tensor<2x50257xi1>
      %4539 = stablehlo.select %4538, %4536, %4529 : tensor<2x50257xi1>, tensor<2x50257xf32>
      %4540 = stablehlo.iota dim = 1 : tensor<2x50257xi32>
      %4541:2 = stablehlo.reduce(%4539 init: %11), (%4540 init: %22) across dimensions = [1] : (tensor<2x50257xf32>, tensor<2x50257xi32>, tensor<f32>, tensor<i32>) -> (tensor<2xf32>, tensor<2xi32>)
       reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<i32>, %arg5: tensor<i32>)  {
        %4561 = stablehlo.compare  GT, %arg2, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
        %4562 = stablehlo.compare  NE, %arg2, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
        %4563 = stablehlo.or %4561, %4562 : tensor<i1>
        %4564 = stablehlo.compare  EQ, %arg2, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
        %4565 = stablehlo.compare  LT, %arg3, %arg5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %4566 = stablehlo.and %4564, %4565 : tensor<i1>
        %4567 = stablehlo.or %4563, %4566 : tensor<i1>
        %4568 = stablehlo.select %4563, %arg2, %arg4 : tensor<i1>, tensor<f32>
        %4569 = stablehlo.select %4567, %arg3, %arg5 : tensor<i1>, tensor<i32>
        stablehlo.return %4568, %4569 : tensor<f32>, tensor<i32>
      }
      %4542 = stablehlo.not %iterArg_152 : tensor<2xi1>
      %4543 = stablehlo.convert %4542 : (tensor<2xi1>) -> tensor<2xi32>
      %4544 = stablehlo.multiply %4541#1, %4543 : tensor<2xi32>
      %4545 = stablehlo.convert %iterArg_152 : (tensor<2xi1>) -> tensor<2xi32>
      %4546 = stablehlo.broadcast_in_dim %iterArg_147, dims = [] : (tensor<i32>) -> tensor<2xi32>
      %4547 = stablehlo.multiply %4546, %4545 : tensor<2xi32>
      %4548 = stablehlo.add %4544, %4547 : tensor<2xi32>
      %4549 = stablehlo.broadcast_in_dim %iterArg_148, dims = [] : (tensor<i32>) -> tensor<2xi32>
      %4550 = stablehlo.compare  EQ, %4548, %4549,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
      %4551 = stablehlo.or %iterArg_152, %4550 : tensor<2xi1>
      %4552 = stablehlo.broadcast_in_dim %4548, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
      %4553 = stablehlo.compare  LT, %iterArg_149, %22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4554 = stablehlo.convert %iterArg_149 : tensor<i32>
      %4555 = stablehlo.add %4554, %15 : tensor<i32>
      %4556 = stablehlo.select %4553, %4555, %iterArg_149 : tensor<i1>, tensor<i32>
      %4557 = stablehlo.dynamic_update_slice %iterArg_150, %4552, %22, %4556 : (tensor<2x20xi32>, tensor<2x1xi32>, tensor<i32>, tensor<i32>) -> tensor<2x20xi32>
      %4558 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i32>) -> tensor<2x1xi32>
      %4559 = stablehlo.add %iterArg_190, %4558 : tensor<2x1xi32>
      %4560 = stablehlo.add %iterArg_149, %19 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_0, %iterArg_1, %iterArg_2, %iterArg_3, %iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %iterArg_13, %iterArg_14, %iterArg_15, %iterArg_16, %iterArg_17, %iterArg_18, %iterArg_19, %iterArg_20, %iterArg_21, %iterArg_22, %iterArg_23, %iterArg_24, %iterArg_25, %iterArg_26, %iterArg_27, %iterArg_28, %iterArg_29, %iterArg_30, %iterArg_31, %iterArg_32, %iterArg_33, %iterArg_34, %iterArg_35, %iterArg_36, %iterArg_37, %iterArg_38, %iterArg_39, %iterArg_40, %iterArg_41, %iterArg_42, %iterArg_43, %iterArg_44, %iterArg_45, %iterArg_46, %iterArg_47, %iterArg_48, %iterArg_49, %iterArg_50, %iterArg_51, %iterArg_52, %iterArg_53, %iterArg_54, %iterArg_55, %iterArg_56, %iterArg_57, %iterArg_58, %iterArg_59, %iterArg_60, %iterArg_61, %iterArg_62, %iterArg_63, %iterArg_64, %iterArg_65, %iterArg_66, %iterArg_67, %iterArg_68, %iterArg_69, %iterArg_70, %iterArg_71, %iterArg_72, %iterArg_73, %iterArg_74, %iterArg_75, %iterArg_76, %iterArg_77, %iterArg_78, %iterArg_79, %iterArg_80, %iterArg_81, %iterArg_82, %iterArg_83, %iterArg_84, %iterArg_85, %iterArg_86, %iterArg_87, %iterArg_88, %iterArg_89, %iterArg_90, %iterArg_91, %iterArg_92, %iterArg_93, %iterArg_94, %iterArg_95, %iterArg_96, %iterArg_97, %iterArg_98, %iterArg_99, %iterArg_100, %iterArg_101, %iterArg_102, %iterArg_103, %iterArg_104, %iterArg_105, %iterArg_106, %iterArg_107, %iterArg_108, %iterArg_109, %iterArg_110, %iterArg_111, %iterArg_112, %iterArg_113, %iterArg_114, %iterArg_115, %iterArg_116, %iterArg_117, %iterArg_118, %iterArg_119, %iterArg_120, %iterArg_121, %iterArg_122, %iterArg_123, %iterArg_124, %iterArg_125, %iterArg_126, %iterArg_127, %iterArg_128, %iterArg_129, %iterArg_130, %iterArg_131, %iterArg_132, %iterArg_133, %iterArg_134, %iterArg_135, %iterArg_136, %iterArg_137, %iterArg_138, %iterArg_139, %iterArg_140, %iterArg_141, %iterArg_142, %iterArg_143, %iterArg_144, %iterArg_145, %iterArg_146, %iterArg_147, %iterArg_148, %4560, %4557, %4552, %4551, %iterArg_153, %2552, %2547, %2551, %2719, %2714, %2718, %4222, %4217, %4221, %4389, %4384, %4388, %2886, %2881, %2885, %3053, %3048, %3052, %3220, %3215, %3219, %3387, %3382, %3386, %3554, %3549, %3553, %3721, %3716, %3720, %3888, %3883, %3887, %4055, %4050, %4054, %4559 : tensor<50257x768xf16>, tensor<1024x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<2304x768xf16>, tensor<2304xf16>, tensor<768x768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<3072x768xf16>, tensor<3072xf16>, tensor<768x3072xf16>, tensor<768xf16>, tensor<768xf16>, tensor<768xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<2x20xi32>, tensor<2x1xi32>, tensor<2xi1>, tensor<2x20xi32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<i32>, tensor<2x20x12x64xf32>, tensor<2x20x12x64xf32>, tensor<2x1xi32>
    }
    return %2404#151 : tensor<2x20xi32>
  }
}


