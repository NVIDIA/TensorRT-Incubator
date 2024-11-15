module @whisper_jax attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x80x3000xf32> {jax.arg_info = "input_features", mhlo.sharding = "{replicated}"}) -> (tensor<1x448xi32> {jax.result_info = ""}) {
    %0 = stablehlo.constant dense_resource<__elided__> : tensor<3x80x384xf32>
    %1 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %2 = stablehlo.constant dense_resource<__elided__> : tensor<3x384x384xf32>
    %3 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %4 = stablehlo.constant dense_resource<__elided__> : tensor<1500x384xf32>
    %5 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %6 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %7 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %8 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %9 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %10 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %11 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %12 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %13 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %14 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %15 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %16 = stablehlo.constant dense_resource<__elided__> : tensor<384x1536xf32>
    %17 = stablehlo.constant dense_resource<__elided__> : tensor<1536xf32>
    %18 = stablehlo.constant dense_resource<__elided__> : tensor<1536x384xf32>
    %19 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %20 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %21 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %22 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %23 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %24 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %25 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %26 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %27 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %28 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %29 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %30 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %31 = stablehlo.constant dense_resource<__elided__> : tensor<384x1536xf32>
    %32 = stablehlo.constant dense_resource<__elided__> : tensor<1536xf32>
    %33 = stablehlo.constant dense_resource<__elided__> : tensor<1536x384xf32>
    %34 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %35 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %36 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %37 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %38 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %39 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %40 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %41 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %42 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %43 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %44 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %45 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %46 = stablehlo.constant dense_resource<__elided__> : tensor<384x1536xf32>
    %47 = stablehlo.constant dense_resource<__elided__> : tensor<1536xf32>
    %48 = stablehlo.constant dense_resource<__elided__> : tensor<1536x384xf32>
    %49 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %50 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %51 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %52 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %53 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %54 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %55 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %56 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %57 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %58 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %59 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %60 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %61 = stablehlo.constant dense_resource<__elided__> : tensor<384x1536xf32>
    %62 = stablehlo.constant dense_resource<__elided__> : tensor<1536xf32>
    %63 = stablehlo.constant dense_resource<__elided__> : tensor<1536x384xf32>
    %64 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %65 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %66 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %67 = stablehlo.constant dense_resource<__elided__> : tensor<51865x384xf32>
    %68 = stablehlo.constant dense_resource<__elided__> : tensor<448x384xf32>
    %69 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %70 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %71 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %72 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %73 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %74 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %75 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %76 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %77 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %78 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %79 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %80 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %81 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %82 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %83 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %84 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %85 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %86 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %87 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %88 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %89 = stablehlo.constant dense_resource<__elided__> : tensor<384x1536xf32>
    %90 = stablehlo.constant dense_resource<__elided__> : tensor<1536xf32>
    %91 = stablehlo.constant dense_resource<__elided__> : tensor<1536x384xf32>
    %92 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %93 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %94 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %95 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %96 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %97 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %98 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %99 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %100 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %101 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %102 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %103 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %104 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %105 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %106 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %107 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %108 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %109 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %110 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %111 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %112 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %113 = stablehlo.constant dense_resource<__elided__> : tensor<384x1536xf32>
    %114 = stablehlo.constant dense_resource<__elided__> : tensor<1536xf32>
    %115 = stablehlo.constant dense_resource<__elided__> : tensor<1536x384xf32>
    %116 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %117 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %118 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %119 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %120 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %121 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %122 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %123 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %124 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %125 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %126 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %127 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %128 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %129 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %130 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %131 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %132 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %133 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %134 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %135 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %136 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %137 = stablehlo.constant dense_resource<__elided__> : tensor<384x1536xf32>
    %138 = stablehlo.constant dense_resource<__elided__> : tensor<1536xf32>
    %139 = stablehlo.constant dense_resource<__elided__> : tensor<1536x384xf32>
    %140 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %141 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %142 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %143 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %144 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %145 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %146 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %147 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %148 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %149 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %150 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %151 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %152 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %153 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %154 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %155 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %156 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %157 = stablehlo.constant dense_resource<__elided__> : tensor<384x384xf32>
    %158 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %159 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %160 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %161 = stablehlo.constant dense_resource<__elided__> : tensor<384x1536xf32>
    %162 = stablehlo.constant dense_resource<__elided__> : tensor<1536xf32>
    %163 = stablehlo.constant dense_resource<__elided__> : tensor<1536x384xf32>
    %164 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %165 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %166 = stablehlo.constant dense_resource<__elided__> : tensor<384xf32>
    %167 = stablehlo.constant dense_resource<__elided__> : tensor<88xi32>
    %168 = stablehlo.constant dense<[220, 50257]> : tensor<2xi32>
    %169 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<1x80x3000xf32>) -> tensor<1x3000x80xf32>
    %170 = stablehlo.convolution(%169, %0) dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f], window = {stride = [1], pad = [[1, 1]], lhs_dilate = [1], rhs_dilate = [1], reverse = [0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x3000x80xf32>, tensor<3x80x384xf32>) -> tensor<1x3000x384xf32>
    %171 = stablehlo.reshape %1 : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x3000x384xf32>
    %173 = stablehlo.add %170, %172 : tensor<1x3000x384xf32>
    %174 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %175 = stablehlo.broadcast_in_dim %174, dims = [] : (tensor<f32>) -> tensor<1x3000x384xf32>
    %176 = stablehlo.divide %173, %175 : tensor<1x3000x384xf32>
    %177 = chlo.erf %176 : tensor<1x3000x384xf32> -> tensor<1x3000x384xf32>
    %178 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %179 = stablehlo.broadcast_in_dim %178, dims = [] : (tensor<f32>) -> tensor<1x3000x384xf32>
    %180 = stablehlo.add %177, %179 : tensor<1x3000x384xf32>
    %181 = stablehlo.multiply %173, %180 : tensor<1x3000x384xf32>
    %182 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %183 = stablehlo.broadcast_in_dim %182, dims = [] : (tensor<f32>) -> tensor<1x3000x384xf32>
    %184 = stablehlo.divide %181, %183 : tensor<1x3000x384xf32>
    %185 = stablehlo.convolution(%184, %2) dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f], window = {stride = [2], pad = [[1, 1]], lhs_dilate = [1], rhs_dilate = [1], reverse = [0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x3000x384xf32>, tensor<3x384x384xf32>) -> tensor<1x1500x384xf32>
    %186 = stablehlo.reshape %3 : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %187 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %188 = stablehlo.add %185, %187 : tensor<1x1500x384xf32>
    %189 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %190 = stablehlo.broadcast_in_dim %189, dims = [] : (tensor<f32>) -> tensor<1x1500x384xf32>
    %191 = stablehlo.divide %188, %190 : tensor<1x1500x384xf32>
    %192 = chlo.erf %191 : tensor<1x1500x384xf32> -> tensor<1x1500x384xf32>
    %193 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [] : (tensor<f32>) -> tensor<1x1500x384xf32>
    %195 = stablehlo.add %192, %194 : tensor<1x1500x384xf32>
    %196 = stablehlo.multiply %188, %195 : tensor<1x1500x384xf32>
    %197 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [] : (tensor<f32>) -> tensor<1x1500x384xf32>
    %199 = stablehlo.divide %196, %198 : tensor<1x1500x384xf32>
    %200 = stablehlo.iota dim = 0 : tensor<1500xi32>
    %201 = stablehlo.iota dim = 0 : tensor<1500xi32>
    %202 = stablehlo.broadcast_in_dim %200, dims = [0] : (tensor<1500xi32>) -> tensor<1500x1xi32>
    %203 = stablehlo.broadcast_in_dim %201, dims = [1] : (tensor<1500xi32>) -> tensor<1x1500xi32>
    %204 = stablehlo.broadcast_in_dim %202, dims = [0, 1] : (tensor<1500x1xi32>) -> tensor<1500x1500xi32>
    %205 = stablehlo.broadcast_in_dim %203, dims = [0, 1] : (tensor<1x1500xi32>) -> tensor<1500x1500xi32>
    %206 = stablehlo.compare  EQ, %204, %205,  SIGNED : (tensor<1500x1500xi32>, tensor<1500x1500xi32>) -> tensor<1500x1500xi1>
    %207 = stablehlo.convert %206 : (tensor<1500x1500xi1>) -> tensor<1500x1500xf32>
    %208 = stablehlo.dot_general %207, %4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1500x1500xf32>, tensor<1500x384xf32>) -> tensor<1500x384xf32>
    %209 = stablehlo.broadcast_in_dim %208, dims = [1, 2] : (tensor<1500x384xf32>) -> tensor<1x1500x384xf32>
    %210 = stablehlo.add %199, %209 : tensor<1x1500x384xf32>
    %211 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %212 = stablehlo.reduce(%210 init: %211) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %214 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %215 = stablehlo.broadcast_in_dim %214, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %216 = stablehlo.divide %213, %215 : tensor<1x1500x1xf32>
    %217 = stablehlo.multiply %210, %210 : tensor<1x1500x384xf32>
    %218 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %219 = stablehlo.reduce(%217 init: %218) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %220 = stablehlo.broadcast_in_dim %219, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %221 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %222 = stablehlo.broadcast_in_dim %221, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %223 = stablehlo.divide %220, %222 : tensor<1x1500x1xf32>
    %224 = stablehlo.multiply %216, %216 : tensor<1x1500x1xf32>
    %225 = stablehlo.subtract %223, %224 : tensor<1x1500x1xf32>
    %226 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %228 = stablehlo.add %225, %227 : tensor<1x1500x1xf32>
    %229 = stablehlo.rsqrt %228 : tensor<1x1500x1xf32>
    %230 = stablehlo.broadcast_in_dim %5, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %231 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %232 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %233 = stablehlo.multiply %231, %232 : tensor<1x1500x384xf32>
    %234 = stablehlo.broadcast_in_dim %216, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %235 = stablehlo.subtract %210, %234 : tensor<1x1500x384xf32>
    %236 = stablehlo.multiply %235, %233 : tensor<1x1500x384xf32>
    %237 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %238 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %239 = stablehlo.add %236, %238 : tensor<1x1500x384xf32>
    %240 = stablehlo.dot_general %239, %7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %241 = stablehlo.broadcast_in_dim %8, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %242 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %243 = stablehlo.add %240, %242 : tensor<1x1500x384xf32>
    %244 = stablehlo.dot_general %239, %9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %245 = stablehlo.dot_general %239, %10, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %246 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %247 = stablehlo.broadcast_in_dim %246, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %248 = stablehlo.add %245, %247 : tensor<1x1500x384xf32>
    %249 = stablehlo.reshape %243 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
    %250 = stablehlo.reshape %244 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
    %251 = stablehlo.reshape %248 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
    %252 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %253 = stablehlo.sqrt %252 : tensor<f32>
    %254 = stablehlo.convert %253 : tensor<f32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [] : (tensor<f32>) -> tensor<1x1500x6x64xf32>
    %256 = stablehlo.divide %249, %255 : tensor<1x1500x6x64xf32>
    %257 = stablehlo.dot_general %256, %250, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x6x64xf32>, tensor<1x1500x6x64xf32>) -> tensor<1x6x1500x1500xf32>
    %258 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %259 = stablehlo.reduce(%257 init: %258) across dimensions = [3] : (tensor<1x6x1500x1500xf32>, tensor<f32>) -> tensor<1x6x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %260 = stablehlo.broadcast_in_dim %259, dims = [0, 1, 2] : (tensor<1x6x1500xf32>) -> tensor<1x6x1500x1xf32>
    %261 = stablehlo.broadcast_in_dim %260, dims = [0, 1, 2, 3] : (tensor<1x6x1500x1xf32>) -> tensor<1x6x1500x1500xf32>
    %262 = stablehlo.subtract %257, %261 : tensor<1x6x1500x1500xf32>
    %263 = stablehlo.exponential %262 : tensor<1x6x1500x1500xf32>
    %264 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %265 = stablehlo.reduce(%263 init: %264) across dimensions = [3] : (tensor<1x6x1500x1500xf32>, tensor<f32>) -> tensor<1x6x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2] : (tensor<1x6x1500xf32>) -> tensor<1x6x1500x1xf32>
    %267 = stablehlo.broadcast_in_dim %266, dims = [0, 1, 2, 3] : (tensor<1x6x1500x1xf32>) -> tensor<1x6x1500x1500xf32>
    %268 = stablehlo.divide %263, %267 : tensor<1x6x1500x1500xf32>
    %269 = stablehlo.dot_general %251, %268, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x6x64xf32>, tensor<1x6x1500x1500xf32>) -> tensor<1x6x64x1500xf32>
    %270 = stablehlo.transpose %269, dims = [0, 3, 1, 2] : (tensor<1x6x64x1500xf32>) -> tensor<1x1500x6x64xf32>
    %271 = stablehlo.reshape %270 : (tensor<1x1500x6x64xf32>) -> tensor<1x1500x384xf32>
    %272 = stablehlo.dot_general %271, %12, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %273 = stablehlo.broadcast_in_dim %13, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %275 = stablehlo.add %272, %274 : tensor<1x1500x384xf32>
    %276 = stablehlo.add %210, %275 : tensor<1x1500x384xf32>
    %277 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %278 = stablehlo.reduce(%276 init: %277) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %279 = stablehlo.broadcast_in_dim %278, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %280 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %281 = stablehlo.broadcast_in_dim %280, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %282 = stablehlo.divide %279, %281 : tensor<1x1500x1xf32>
    %283 = stablehlo.multiply %276, %276 : tensor<1x1500x384xf32>
    %284 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %285 = stablehlo.reduce(%283 init: %284) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %287 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %288 = stablehlo.broadcast_in_dim %287, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %289 = stablehlo.divide %286, %288 : tensor<1x1500x1xf32>
    %290 = stablehlo.multiply %282, %282 : tensor<1x1500x1xf32>
    %291 = stablehlo.subtract %289, %290 : tensor<1x1500x1xf32>
    %292 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %293 = stablehlo.broadcast_in_dim %292, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %294 = stablehlo.add %291, %293 : tensor<1x1500x1xf32>
    %295 = stablehlo.rsqrt %294 : tensor<1x1500x1xf32>
    %296 = stablehlo.broadcast_in_dim %14, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %297 = stablehlo.broadcast_in_dim %295, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %298 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %299 = stablehlo.multiply %297, %298 : tensor<1x1500x384xf32>
    %300 = stablehlo.broadcast_in_dim %282, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %301 = stablehlo.subtract %276, %300 : tensor<1x1500x384xf32>
    %302 = stablehlo.multiply %301, %299 : tensor<1x1500x384xf32>
    %303 = stablehlo.broadcast_in_dim %15, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %305 = stablehlo.add %302, %304 : tensor<1x1500x384xf32>
    %306 = stablehlo.dot_general %305, %16, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x1536xf32>) -> tensor<1x1500x1536xf32>
    %307 = stablehlo.broadcast_in_dim %17, dims = [2] : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2] : (tensor<1x1x1536xf32>) -> tensor<1x1500x1536xf32>
    %309 = stablehlo.add %306, %308 : tensor<1x1500x1536xf32>
    %310 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %311 = stablehlo.broadcast_in_dim %310, dims = [] : (tensor<f32>) -> tensor<1x1500x1536xf32>
    %312 = stablehlo.divide %309, %311 : tensor<1x1500x1536xf32>
    %313 = chlo.erf %312 : tensor<1x1500x1536xf32> -> tensor<1x1500x1536xf32>
    %314 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %315 = stablehlo.broadcast_in_dim %314, dims = [] : (tensor<f32>) -> tensor<1x1500x1536xf32>
    %316 = stablehlo.add %313, %315 : tensor<1x1500x1536xf32>
    %317 = stablehlo.multiply %309, %316 : tensor<1x1500x1536xf32>
    %318 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %319 = stablehlo.broadcast_in_dim %318, dims = [] : (tensor<f32>) -> tensor<1x1500x1536xf32>
    %320 = stablehlo.divide %317, %319 : tensor<1x1500x1536xf32>
    %321 = stablehlo.dot_general %320, %18, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x1536xf32>, tensor<1536x384xf32>) -> tensor<1x1500x384xf32>
    %322 = stablehlo.broadcast_in_dim %19, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %324 = stablehlo.add %321, %323 : tensor<1x1500x384xf32>
    %325 = stablehlo.add %276, %324 : tensor<1x1500x384xf32>
    %326 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %327 = stablehlo.reduce(%325 init: %326) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %328 = stablehlo.broadcast_in_dim %327, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %329 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %330 = stablehlo.broadcast_in_dim %329, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %331 = stablehlo.divide %328, %330 : tensor<1x1500x1xf32>
    %332 = stablehlo.multiply %325, %325 : tensor<1x1500x384xf32>
    %333 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %334 = stablehlo.reduce(%332 init: %333) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %335 = stablehlo.broadcast_in_dim %334, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %336 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %337 = stablehlo.broadcast_in_dim %336, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %338 = stablehlo.divide %335, %337 : tensor<1x1500x1xf32>
    %339 = stablehlo.multiply %331, %331 : tensor<1x1500x1xf32>
    %340 = stablehlo.subtract %338, %339 : tensor<1x1500x1xf32>
    %341 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %342 = stablehlo.broadcast_in_dim %341, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %343 = stablehlo.add %340, %342 : tensor<1x1500x1xf32>
    %344 = stablehlo.rsqrt %343 : tensor<1x1500x1xf32>
    %345 = stablehlo.broadcast_in_dim %20, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %346 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %347 = stablehlo.broadcast_in_dim %345, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %348 = stablehlo.multiply %346, %347 : tensor<1x1500x384xf32>
    %349 = stablehlo.broadcast_in_dim %331, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %350 = stablehlo.subtract %325, %349 : tensor<1x1500x384xf32>
    %351 = stablehlo.multiply %350, %348 : tensor<1x1500x384xf32>
    %352 = stablehlo.broadcast_in_dim %21, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %353 = stablehlo.broadcast_in_dim %352, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %354 = stablehlo.add %351, %353 : tensor<1x1500x384xf32>
    %355 = stablehlo.dot_general %354, %22, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %356 = stablehlo.broadcast_in_dim %23, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %357 = stablehlo.broadcast_in_dim %356, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %358 = stablehlo.add %355, %357 : tensor<1x1500x384xf32>
    %359 = stablehlo.dot_general %354, %24, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %360 = stablehlo.dot_general %354, %25, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %361 = stablehlo.broadcast_in_dim %26, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %362 = stablehlo.broadcast_in_dim %361, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %363 = stablehlo.add %360, %362 : tensor<1x1500x384xf32>
    %364 = stablehlo.reshape %358 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
    %365 = stablehlo.reshape %359 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
    %366 = stablehlo.reshape %363 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
    %367 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %368 = stablehlo.sqrt %367 : tensor<f32>
    %369 = stablehlo.convert %368 : tensor<f32>
    %370 = stablehlo.broadcast_in_dim %369, dims = [] : (tensor<f32>) -> tensor<1x1500x6x64xf32>
    %371 = stablehlo.divide %364, %370 : tensor<1x1500x6x64xf32>
    %372 = stablehlo.dot_general %371, %365, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x6x64xf32>, tensor<1x1500x6x64xf32>) -> tensor<1x6x1500x1500xf32>
    %373 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %374 = stablehlo.reduce(%372 init: %373) across dimensions = [3] : (tensor<1x6x1500x1500xf32>, tensor<f32>) -> tensor<1x6x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %375 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2] : (tensor<1x6x1500xf32>) -> tensor<1x6x1500x1xf32>
    %376 = stablehlo.broadcast_in_dim %375, dims = [0, 1, 2, 3] : (tensor<1x6x1500x1xf32>) -> tensor<1x6x1500x1500xf32>
    %377 = stablehlo.subtract %372, %376 : tensor<1x6x1500x1500xf32>
    %378 = stablehlo.exponential %377 : tensor<1x6x1500x1500xf32>
    %379 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %380 = stablehlo.reduce(%378 init: %379) across dimensions = [3] : (tensor<1x6x1500x1500xf32>, tensor<f32>) -> tensor<1x6x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %381 = stablehlo.broadcast_in_dim %380, dims = [0, 1, 2] : (tensor<1x6x1500xf32>) -> tensor<1x6x1500x1xf32>
    %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1, 2, 3] : (tensor<1x6x1500x1xf32>) -> tensor<1x6x1500x1500xf32>
    %383 = stablehlo.divide %378, %382 : tensor<1x6x1500x1500xf32>
    %384 = stablehlo.dot_general %366, %383, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x6x64xf32>, tensor<1x6x1500x1500xf32>) -> tensor<1x6x64x1500xf32>
    %385 = stablehlo.transpose %384, dims = [0, 3, 1, 2] : (tensor<1x6x64x1500xf32>) -> tensor<1x1500x6x64xf32>
    %386 = stablehlo.reshape %385 : (tensor<1x1500x6x64xf32>) -> tensor<1x1500x384xf32>
    %387 = stablehlo.dot_general %386, %27, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %388 = stablehlo.broadcast_in_dim %28, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %389 = stablehlo.broadcast_in_dim %388, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %390 = stablehlo.add %387, %389 : tensor<1x1500x384xf32>
    %391 = stablehlo.add %325, %390 : tensor<1x1500x384xf32>
    %392 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %393 = stablehlo.reduce(%391 init: %392) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %395 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %396 = stablehlo.broadcast_in_dim %395, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %397 = stablehlo.divide %394, %396 : tensor<1x1500x1xf32>
    %398 = stablehlo.multiply %391, %391 : tensor<1x1500x384xf32>
    %399 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %400 = stablehlo.reduce(%398 init: %399) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %401 = stablehlo.broadcast_in_dim %400, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %402 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %403 = stablehlo.broadcast_in_dim %402, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %404 = stablehlo.divide %401, %403 : tensor<1x1500x1xf32>
    %405 = stablehlo.multiply %397, %397 : tensor<1x1500x1xf32>
    %406 = stablehlo.subtract %404, %405 : tensor<1x1500x1xf32>
    %407 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %408 = stablehlo.broadcast_in_dim %407, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %409 = stablehlo.add %406, %408 : tensor<1x1500x1xf32>
    %410 = stablehlo.rsqrt %409 : tensor<1x1500x1xf32>
    %411 = stablehlo.broadcast_in_dim %29, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %412 = stablehlo.broadcast_in_dim %410, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %413 = stablehlo.broadcast_in_dim %411, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %414 = stablehlo.multiply %412, %413 : tensor<1x1500x384xf32>
    %415 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %416 = stablehlo.subtract %391, %415 : tensor<1x1500x384xf32>
    %417 = stablehlo.multiply %416, %414 : tensor<1x1500x384xf32>
    %418 = stablehlo.broadcast_in_dim %30, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %420 = stablehlo.add %417, %419 : tensor<1x1500x384xf32>
    %421 = stablehlo.dot_general %420, %31, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x1536xf32>) -> tensor<1x1500x1536xf32>
    %422 = stablehlo.broadcast_in_dim %32, dims = [2] : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %423 = stablehlo.broadcast_in_dim %422, dims = [0, 1, 2] : (tensor<1x1x1536xf32>) -> tensor<1x1500x1536xf32>
    %424 = stablehlo.add %421, %423 : tensor<1x1500x1536xf32>
    %425 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %426 = stablehlo.broadcast_in_dim %425, dims = [] : (tensor<f32>) -> tensor<1x1500x1536xf32>
    %427 = stablehlo.divide %424, %426 : tensor<1x1500x1536xf32>
    %428 = chlo.erf %427 : tensor<1x1500x1536xf32> -> tensor<1x1500x1536xf32>
    %429 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %430 = stablehlo.broadcast_in_dim %429, dims = [] : (tensor<f32>) -> tensor<1x1500x1536xf32>
    %431 = stablehlo.add %428, %430 : tensor<1x1500x1536xf32>
    %432 = stablehlo.multiply %424, %431 : tensor<1x1500x1536xf32>
    %433 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %434 = stablehlo.broadcast_in_dim %433, dims = [] : (tensor<f32>) -> tensor<1x1500x1536xf32>
    %435 = stablehlo.divide %432, %434 : tensor<1x1500x1536xf32>
    %436 = stablehlo.dot_general %435, %33, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x1536xf32>, tensor<1536x384xf32>) -> tensor<1x1500x384xf32>
    %437 = stablehlo.broadcast_in_dim %34, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %439 = stablehlo.add %436, %438 : tensor<1x1500x384xf32>
    %440 = stablehlo.add %391, %439 : tensor<1x1500x384xf32>
    %441 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %442 = stablehlo.reduce(%440 init: %441) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %443 = stablehlo.broadcast_in_dim %442, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %444 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %445 = stablehlo.broadcast_in_dim %444, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %446 = stablehlo.divide %443, %445 : tensor<1x1500x1xf32>
    %447 = stablehlo.multiply %440, %440 : tensor<1x1500x384xf32>
    %448 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %449 = stablehlo.reduce(%447 init: %448) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %450 = stablehlo.broadcast_in_dim %449, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %451 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %452 = stablehlo.broadcast_in_dim %451, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %453 = stablehlo.divide %450, %452 : tensor<1x1500x1xf32>
    %454 = stablehlo.multiply %446, %446 : tensor<1x1500x1xf32>
    %455 = stablehlo.subtract %453, %454 : tensor<1x1500x1xf32>
    %456 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %457 = stablehlo.broadcast_in_dim %456, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %458 = stablehlo.add %455, %457 : tensor<1x1500x1xf32>
    %459 = stablehlo.rsqrt %458 : tensor<1x1500x1xf32>
    %460 = stablehlo.broadcast_in_dim %35, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %461 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %462 = stablehlo.broadcast_in_dim %460, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %463 = stablehlo.multiply %461, %462 : tensor<1x1500x384xf32>
    %464 = stablehlo.broadcast_in_dim %446, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %465 = stablehlo.subtract %440, %464 : tensor<1x1500x384xf32>
    %466 = stablehlo.multiply %465, %463 : tensor<1x1500x384xf32>
    %467 = stablehlo.broadcast_in_dim %36, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %468 = stablehlo.broadcast_in_dim %467, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %469 = stablehlo.add %466, %468 : tensor<1x1500x384xf32>
    %470 = stablehlo.dot_general %469, %37, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %471 = stablehlo.broadcast_in_dim %38, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %472 = stablehlo.broadcast_in_dim %471, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %473 = stablehlo.add %470, %472 : tensor<1x1500x384xf32>
    %474 = stablehlo.dot_general %469, %39, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %475 = stablehlo.dot_general %469, %40, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %476 = stablehlo.broadcast_in_dim %41, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %477 = stablehlo.broadcast_in_dim %476, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %478 = stablehlo.add %475, %477 : tensor<1x1500x384xf32>
    %479 = stablehlo.reshape %473 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
    %480 = stablehlo.reshape %474 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
    %481 = stablehlo.reshape %478 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
    %482 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %483 = stablehlo.sqrt %482 : tensor<f32>
    %484 = stablehlo.convert %483 : tensor<f32>
    %485 = stablehlo.broadcast_in_dim %484, dims = [] : (tensor<f32>) -> tensor<1x1500x6x64xf32>
    %486 = stablehlo.divide %479, %485 : tensor<1x1500x6x64xf32>
    %487 = stablehlo.dot_general %486, %480, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x6x64xf32>, tensor<1x1500x6x64xf32>) -> tensor<1x6x1500x1500xf32>
    %488 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %489 = stablehlo.reduce(%487 init: %488) across dimensions = [3] : (tensor<1x6x1500x1500xf32>, tensor<f32>) -> tensor<1x6x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %490 = stablehlo.broadcast_in_dim %489, dims = [0, 1, 2] : (tensor<1x6x1500xf32>) -> tensor<1x6x1500x1xf32>
    %491 = stablehlo.broadcast_in_dim %490, dims = [0, 1, 2, 3] : (tensor<1x6x1500x1xf32>) -> tensor<1x6x1500x1500xf32>
    %492 = stablehlo.subtract %487, %491 : tensor<1x6x1500x1500xf32>
    %493 = stablehlo.exponential %492 : tensor<1x6x1500x1500xf32>
    %494 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %495 = stablehlo.reduce(%493 init: %494) across dimensions = [3] : (tensor<1x6x1500x1500xf32>, tensor<f32>) -> tensor<1x6x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %496 = stablehlo.broadcast_in_dim %495, dims = [0, 1, 2] : (tensor<1x6x1500xf32>) -> tensor<1x6x1500x1xf32>
    %497 = stablehlo.broadcast_in_dim %496, dims = [0, 1, 2, 3] : (tensor<1x6x1500x1xf32>) -> tensor<1x6x1500x1500xf32>
    %498 = stablehlo.divide %493, %497 : tensor<1x6x1500x1500xf32>
    %499 = stablehlo.dot_general %481, %498, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x6x64xf32>, tensor<1x6x1500x1500xf32>) -> tensor<1x6x64x1500xf32>
    %500 = stablehlo.transpose %499, dims = [0, 3, 1, 2] : (tensor<1x6x64x1500xf32>) -> tensor<1x1500x6x64xf32>
    %501 = stablehlo.reshape %500 : (tensor<1x1500x6x64xf32>) -> tensor<1x1500x384xf32>
    %502 = stablehlo.dot_general %501, %42, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %503 = stablehlo.broadcast_in_dim %43, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %504 = stablehlo.broadcast_in_dim %503, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %505 = stablehlo.add %502, %504 : tensor<1x1500x384xf32>
    %506 = stablehlo.add %440, %505 : tensor<1x1500x384xf32>
    %507 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %508 = stablehlo.reduce(%506 init: %507) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %509 = stablehlo.broadcast_in_dim %508, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %510 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %511 = stablehlo.broadcast_in_dim %510, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %512 = stablehlo.divide %509, %511 : tensor<1x1500x1xf32>
    %513 = stablehlo.multiply %506, %506 : tensor<1x1500x384xf32>
    %514 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %515 = stablehlo.reduce(%513 init: %514) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %516 = stablehlo.broadcast_in_dim %515, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %517 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %518 = stablehlo.broadcast_in_dim %517, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %519 = stablehlo.divide %516, %518 : tensor<1x1500x1xf32>
    %520 = stablehlo.multiply %512, %512 : tensor<1x1500x1xf32>
    %521 = stablehlo.subtract %519, %520 : tensor<1x1500x1xf32>
    %522 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %523 = stablehlo.broadcast_in_dim %522, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %524 = stablehlo.add %521, %523 : tensor<1x1500x1xf32>
    %525 = stablehlo.rsqrt %524 : tensor<1x1500x1xf32>
    %526 = stablehlo.broadcast_in_dim %44, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %527 = stablehlo.broadcast_in_dim %525, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %528 = stablehlo.broadcast_in_dim %526, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %529 = stablehlo.multiply %527, %528 : tensor<1x1500x384xf32>
    %530 = stablehlo.broadcast_in_dim %512, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %531 = stablehlo.subtract %506, %530 : tensor<1x1500x384xf32>
    %532 = stablehlo.multiply %531, %529 : tensor<1x1500x384xf32>
    %533 = stablehlo.broadcast_in_dim %45, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %534 = stablehlo.broadcast_in_dim %533, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %535 = stablehlo.add %532, %534 : tensor<1x1500x384xf32>
    %536 = stablehlo.dot_general %535, %46, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x1536xf32>) -> tensor<1x1500x1536xf32>
    %537 = stablehlo.broadcast_in_dim %47, dims = [2] : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %538 = stablehlo.broadcast_in_dim %537, dims = [0, 1, 2] : (tensor<1x1x1536xf32>) -> tensor<1x1500x1536xf32>
    %539 = stablehlo.add %536, %538 : tensor<1x1500x1536xf32>
    %540 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %541 = stablehlo.broadcast_in_dim %540, dims = [] : (tensor<f32>) -> tensor<1x1500x1536xf32>
    %542 = stablehlo.divide %539, %541 : tensor<1x1500x1536xf32>
    %543 = chlo.erf %542 : tensor<1x1500x1536xf32> -> tensor<1x1500x1536xf32>
    %544 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %545 = stablehlo.broadcast_in_dim %544, dims = [] : (tensor<f32>) -> tensor<1x1500x1536xf32>
    %546 = stablehlo.add %543, %545 : tensor<1x1500x1536xf32>
    %547 = stablehlo.multiply %539, %546 : tensor<1x1500x1536xf32>
    %548 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %549 = stablehlo.broadcast_in_dim %548, dims = [] : (tensor<f32>) -> tensor<1x1500x1536xf32>
    %550 = stablehlo.divide %547, %549 : tensor<1x1500x1536xf32>
    %551 = stablehlo.dot_general %550, %48, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x1536xf32>, tensor<1536x384xf32>) -> tensor<1x1500x384xf32>
    %552 = stablehlo.broadcast_in_dim %49, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %553 = stablehlo.broadcast_in_dim %552, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %554 = stablehlo.add %551, %553 : tensor<1x1500x384xf32>
    %555 = stablehlo.add %506, %554 : tensor<1x1500x384xf32>
    %556 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %557 = stablehlo.reduce(%555 init: %556) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %558 = stablehlo.broadcast_in_dim %557, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %559 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %560 = stablehlo.broadcast_in_dim %559, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %561 = stablehlo.divide %558, %560 : tensor<1x1500x1xf32>
    %562 = stablehlo.multiply %555, %555 : tensor<1x1500x384xf32>
    %563 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %564 = stablehlo.reduce(%562 init: %563) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %565 = stablehlo.broadcast_in_dim %564, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %566 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %567 = stablehlo.broadcast_in_dim %566, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %568 = stablehlo.divide %565, %567 : tensor<1x1500x1xf32>
    %569 = stablehlo.multiply %561, %561 : tensor<1x1500x1xf32>
    %570 = stablehlo.subtract %568, %569 : tensor<1x1500x1xf32>
    %571 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %572 = stablehlo.broadcast_in_dim %571, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %573 = stablehlo.add %570, %572 : tensor<1x1500x1xf32>
    %574 = stablehlo.rsqrt %573 : tensor<1x1500x1xf32>
    %575 = stablehlo.broadcast_in_dim %50, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %576 = stablehlo.broadcast_in_dim %574, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %577 = stablehlo.broadcast_in_dim %575, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %578 = stablehlo.multiply %576, %577 : tensor<1x1500x384xf32>
    %579 = stablehlo.broadcast_in_dim %561, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %580 = stablehlo.subtract %555, %579 : tensor<1x1500x384xf32>
    %581 = stablehlo.multiply %580, %578 : tensor<1x1500x384xf32>
    %582 = stablehlo.broadcast_in_dim %51, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %583 = stablehlo.broadcast_in_dim %582, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %584 = stablehlo.add %581, %583 : tensor<1x1500x384xf32>
    %585 = stablehlo.dot_general %584, %52, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %586 = stablehlo.broadcast_in_dim %53, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %587 = stablehlo.broadcast_in_dim %586, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %588 = stablehlo.add %585, %587 : tensor<1x1500x384xf32>
    %589 = stablehlo.dot_general %584, %54, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %590 = stablehlo.dot_general %584, %55, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %591 = stablehlo.broadcast_in_dim %56, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %592 = stablehlo.broadcast_in_dim %591, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %593 = stablehlo.add %590, %592 : tensor<1x1500x384xf32>
    %594 = stablehlo.reshape %588 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
    %595 = stablehlo.reshape %589 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
    %596 = stablehlo.reshape %593 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
    %597 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %598 = stablehlo.sqrt %597 : tensor<f32>
    %599 = stablehlo.convert %598 : tensor<f32>
    %600 = stablehlo.broadcast_in_dim %599, dims = [] : (tensor<f32>) -> tensor<1x1500x6x64xf32>
    %601 = stablehlo.divide %594, %600 : tensor<1x1500x6x64xf32>
    %602 = stablehlo.dot_general %601, %595, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x6x64xf32>, tensor<1x1500x6x64xf32>) -> tensor<1x6x1500x1500xf32>
    %603 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %604 = stablehlo.reduce(%602 init: %603) across dimensions = [3] : (tensor<1x6x1500x1500xf32>, tensor<f32>) -> tensor<1x6x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %605 = stablehlo.broadcast_in_dim %604, dims = [0, 1, 2] : (tensor<1x6x1500xf32>) -> tensor<1x6x1500x1xf32>
    %606 = stablehlo.broadcast_in_dim %605, dims = [0, 1, 2, 3] : (tensor<1x6x1500x1xf32>) -> tensor<1x6x1500x1500xf32>
    %607 = stablehlo.subtract %602, %606 : tensor<1x6x1500x1500xf32>
    %608 = stablehlo.exponential %607 : tensor<1x6x1500x1500xf32>
    %609 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %610 = stablehlo.reduce(%608 init: %609) across dimensions = [3] : (tensor<1x6x1500x1500xf32>, tensor<f32>) -> tensor<1x6x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %611 = stablehlo.broadcast_in_dim %610, dims = [0, 1, 2] : (tensor<1x6x1500xf32>) -> tensor<1x6x1500x1xf32>
    %612 = stablehlo.broadcast_in_dim %611, dims = [0, 1, 2, 3] : (tensor<1x6x1500x1xf32>) -> tensor<1x6x1500x1500xf32>
    %613 = stablehlo.divide %608, %612 : tensor<1x6x1500x1500xf32>
    %614 = stablehlo.dot_general %596, %613, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x6x64xf32>, tensor<1x6x1500x1500xf32>) -> tensor<1x6x64x1500xf32>
    %615 = stablehlo.transpose %614, dims = [0, 3, 1, 2] : (tensor<1x6x64x1500xf32>) -> tensor<1x1500x6x64xf32>
    %616 = stablehlo.reshape %615 : (tensor<1x1500x6x64xf32>) -> tensor<1x1500x384xf32>
    %617 = stablehlo.dot_general %616, %57, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
    %618 = stablehlo.broadcast_in_dim %58, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %619 = stablehlo.broadcast_in_dim %618, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %620 = stablehlo.add %617, %619 : tensor<1x1500x384xf32>
    %621 = stablehlo.add %555, %620 : tensor<1x1500x384xf32>
    %622 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %623 = stablehlo.reduce(%621 init: %622) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %624 = stablehlo.broadcast_in_dim %623, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %625 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %626 = stablehlo.broadcast_in_dim %625, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %627 = stablehlo.divide %624, %626 : tensor<1x1500x1xf32>
    %628 = stablehlo.multiply %621, %621 : tensor<1x1500x384xf32>
    %629 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %630 = stablehlo.reduce(%628 init: %629) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %632 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %633 = stablehlo.broadcast_in_dim %632, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %634 = stablehlo.divide %631, %633 : tensor<1x1500x1xf32>
    %635 = stablehlo.multiply %627, %627 : tensor<1x1500x1xf32>
    %636 = stablehlo.subtract %634, %635 : tensor<1x1500x1xf32>
    %637 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %638 = stablehlo.broadcast_in_dim %637, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %639 = stablehlo.add %636, %638 : tensor<1x1500x1xf32>
    %640 = stablehlo.rsqrt %639 : tensor<1x1500x1xf32>
    %641 = stablehlo.broadcast_in_dim %59, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %642 = stablehlo.broadcast_in_dim %640, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %643 = stablehlo.broadcast_in_dim %641, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %644 = stablehlo.multiply %642, %643 : tensor<1x1500x384xf32>
    %645 = stablehlo.broadcast_in_dim %627, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %646 = stablehlo.subtract %621, %645 : tensor<1x1500x384xf32>
    %647 = stablehlo.multiply %646, %644 : tensor<1x1500x384xf32>
    %648 = stablehlo.broadcast_in_dim %60, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %649 = stablehlo.broadcast_in_dim %648, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %650 = stablehlo.add %647, %649 : tensor<1x1500x384xf32>
    %651 = stablehlo.dot_general %650, %61, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x1536xf32>) -> tensor<1x1500x1536xf32>
    %652 = stablehlo.broadcast_in_dim %62, dims = [2] : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %653 = stablehlo.broadcast_in_dim %652, dims = [0, 1, 2] : (tensor<1x1x1536xf32>) -> tensor<1x1500x1536xf32>
    %654 = stablehlo.add %651, %653 : tensor<1x1500x1536xf32>
    %655 = stablehlo.constant dense<1.41421354> : tensor<f32>
    %656 = stablehlo.broadcast_in_dim %655, dims = [] : (tensor<f32>) -> tensor<1x1500x1536xf32>
    %657 = stablehlo.divide %654, %656 : tensor<1x1500x1536xf32>
    %658 = chlo.erf %657 : tensor<1x1500x1536xf32> -> tensor<1x1500x1536xf32>
    %659 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %660 = stablehlo.broadcast_in_dim %659, dims = [] : (tensor<f32>) -> tensor<1x1500x1536xf32>
    %661 = stablehlo.add %658, %660 : tensor<1x1500x1536xf32>
    %662 = stablehlo.multiply %654, %661 : tensor<1x1500x1536xf32>
    %663 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %664 = stablehlo.broadcast_in_dim %663, dims = [] : (tensor<f32>) -> tensor<1x1500x1536xf32>
    %665 = stablehlo.divide %662, %664 : tensor<1x1500x1536xf32>
    %666 = stablehlo.dot_general %665, %63, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x1536xf32>, tensor<1536x384xf32>) -> tensor<1x1500x384xf32>
    %667 = stablehlo.broadcast_in_dim %64, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %668 = stablehlo.broadcast_in_dim %667, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %669 = stablehlo.add %666, %668 : tensor<1x1500x384xf32>
    %670 = stablehlo.add %621, %669 : tensor<1x1500x384xf32>
    %671 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %672 = stablehlo.reduce(%670 init: %671) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %673 = stablehlo.broadcast_in_dim %672, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %674 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %675 = stablehlo.broadcast_in_dim %674, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %676 = stablehlo.divide %673, %675 : tensor<1x1500x1xf32>
    %677 = stablehlo.multiply %670, %670 : tensor<1x1500x384xf32>
    %678 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %679 = stablehlo.reduce(%677 init: %678) across dimensions = [2] : (tensor<1x1500x384xf32>, tensor<f32>) -> tensor<1x1500xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %754 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %754 : tensor<f32>
    }
    %680 = stablehlo.broadcast_in_dim %679, dims = [0, 1] : (tensor<1x1500xf32>) -> tensor<1x1500x1xf32>
    %681 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
    %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %683 = stablehlo.divide %680, %682 : tensor<1x1500x1xf32>
    %684 = stablehlo.multiply %676, %676 : tensor<1x1500x1xf32>
    %685 = stablehlo.subtract %683, %684 : tensor<1x1500x1xf32>
    %686 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %687 = stablehlo.broadcast_in_dim %686, dims = [] : (tensor<f32>) -> tensor<1x1500x1xf32>
    %688 = stablehlo.add %685, %687 : tensor<1x1500x1xf32>
    %689 = stablehlo.rsqrt %688 : tensor<1x1500x1xf32>
    %690 = stablehlo.broadcast_in_dim %65, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %691 = stablehlo.broadcast_in_dim %689, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %692 = stablehlo.broadcast_in_dim %690, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %693 = stablehlo.multiply %691, %692 : tensor<1x1500x384xf32>
    %694 = stablehlo.broadcast_in_dim %676, dims = [0, 1, 2] : (tensor<1x1500x1xf32>) -> tensor<1x1500x384xf32>
    %695 = stablehlo.subtract %670, %694 : tensor<1x1500x384xf32>
    %696 = stablehlo.multiply %695, %693 : tensor<1x1500x384xf32>
    %697 = stablehlo.broadcast_in_dim %66, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
    %698 = stablehlo.broadcast_in_dim %697, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
    %699 = stablehlo.add %696, %698 : tensor<1x1500x384xf32>
    %700 = stablehlo.constant dense<50258> : tensor<i32>
    %701 = stablehlo.reshape %700 : (tensor<i32>) -> tensor<1x1xi32>
    %702 = stablehlo.broadcast_in_dim %701, dims = [0, 2] : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
    %703 = stablehlo.reshape %702 : (tensor<1x1x1xi32>) -> tensor<1x1x1x1x1x1xi32>
    %704 = stablehlo.reshape %703 : (tensor<1x1x1x1x1x1xi32>) -> tensor<1x1x1xi32>
    %705 = stablehlo.reshape %704 : (tensor<1x1x1xi32>) -> tensor<1x1xi32>
    %706 = stablehlo.constant dense<1> : tensor<i32>
    %707 = stablehlo.broadcast_in_dim %706, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %708 = stablehlo.constant dense<-1> : tensor<i32>
    %709 = stablehlo.broadcast_in_dim %708, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %710 = stablehlo.multiply %707, %709 : tensor<4xi32>
    %711 = stablehlo.constant dense<2> : tensor<i32>
    %712 = stablehlo.broadcast_in_dim %711, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %713 = stablehlo.constant dense<50359> : tensor<i32>
    %714 = "stablehlo.scatter"(%710, %712, %713) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      stablehlo.return %arg2 : tensor<i32>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>
    %715 = stablehlo.constant dense<3> : tensor<i32>
    %716 = stablehlo.broadcast_in_dim %715, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %717 = stablehlo.constant dense<50363> : tensor<i32>
    %718 = "stablehlo.scatter"(%714, %716, %717) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      stablehlo.return %arg2 : tensor<i32>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>
    %719 = stablehlo.constant dense<50257> : tensor<i32>
    %720 = stablehlo.broadcast_in_dim %719, dims = [] : (tensor<i32>) -> tensor<1x448xi32>
    %721 = stablehlo.constant dense<0> : tensor<i32>
    %722 = stablehlo.constant dense<0> : tensor<i32>
    %723 = stablehlo.dynamic_update_slice %720, %705, %721, %722 : (tensor<1x448xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x448xi32>
    %724 = stablehlo.constant dense<false> : tensor<i1>
    %725 = stablehlo.broadcast_in_dim %724, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %726 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %727 = stablehlo.broadcast_in_dim %726, dims = [] : (tensor<f32>) -> tensor<1x6x64x448xf32>
    %728 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %729 = stablehlo.broadcast_in_dim %728, dims = [] : (tensor<f32>) -> tensor<1x6x64x448xf32>
    %730 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %731 = stablehlo.broadcast_in_dim %730, dims = [] : (tensor<f32>) -> tensor<1x6x64x448xf32>
    %732 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %733 = stablehlo.broadcast_in_dim %732, dims = [] : (tensor<f32>) -> tensor<1x6x64x448xf32>
    %734 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %735 = stablehlo.broadcast_in_dim %734, dims = [] : (tensor<f32>) -> tensor<1x6x64x448xf32>
    %736 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %737 = stablehlo.broadcast_in_dim %736, dims = [] : (tensor<f32>) -> tensor<1x6x64x448xf32>
    %738 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %739 = stablehlo.broadcast_in_dim %738, dims = [] : (tensor<f32>) -> tensor<1x6x64x448xf32>
    %740 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %741 = stablehlo.broadcast_in_dim %740, dims = [] : (tensor<f32>) -> tensor<1x6x64x448xf32>
    %742 = stablehlo.constant dense<1> : tensor<i32>
    %743 = stablehlo.broadcast_in_dim %742, dims = [] : (tensor<i32>) -> tensor<1x448xi32>
    %744 = stablehlo.iota dim = 0 : tensor<1xi32>
    %745 = stablehlo.broadcast_in_dim %744, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %746 = stablehlo.constant dense<50257> : tensor<i32>
    %747 = stablehlo.constant dense<50257> : tensor<i32>
    %748 = stablehlo.constant dense<1> : tensor<i32>
    %749 = stablehlo.constant dense<0> : tensor<i32>
    %750 = stablehlo.constant dense<0> : tensor<i32>
    %751 = stablehlo.constant dense<0> : tensor<i32>
    %752 = stablehlo.constant dense<0> : tensor<i32>
    %753:124 = stablehlo.while(%iterArg = %67, %iterArg_0 = %68, %iterArg_1 = %69, %iterArg_2 = %70, %iterArg_3 = %71, %iterArg_4 = %72, %iterArg_5 = %73, %iterArg_6 = %74, %iterArg_7 = %75, %iterArg_8 = %76, %iterArg_9 = %77, %iterArg_10 = %78, %iterArg_11 = %79, %iterArg_12 = %80, %iterArg_13 = %81, %iterArg_14 = %82, %iterArg_15 = %83, %iterArg_16 = %84, %iterArg_17 = %85, %iterArg_18 = %86, %iterArg_19 = %87, %iterArg_20 = %88, %iterArg_21 = %89, %iterArg_22 = %90, %iterArg_23 = %91, %iterArg_24 = %92, %iterArg_25 = %93, %iterArg_26 = %94, %iterArg_27 = %95, %iterArg_28 = %96, %iterArg_29 = %97, %iterArg_30 = %98, %iterArg_31 = %99, %iterArg_32 = %100, %iterArg_33 = %101, %iterArg_34 = %102, %iterArg_35 = %103, %iterArg_36 = %104, %iterArg_37 = %105, %iterArg_38 = %106, %iterArg_39 = %107, %iterArg_40 = %108, %iterArg_41 = %109, %iterArg_42 = %110, %iterArg_43 = %111, %iterArg_44 = %112, %iterArg_45 = %113, %iterArg_46 = %114, %iterArg_47 = %115, %iterArg_48 = %116, %iterArg_49 = %117, %iterArg_50 = %118, %iterArg_51 = %119, %iterArg_52 = %120, %iterArg_53 = %121, %iterArg_54 = %122, %iterArg_55 = %123, %iterArg_56 = %124, %iterArg_57 = %125, %iterArg_58 = %126, %iterArg_59 = %127, %iterArg_60 = %128, %iterArg_61 = %129, %iterArg_62 = %130, %iterArg_63 = %131, %iterArg_64 = %132, %iterArg_65 = %133, %iterArg_66 = %134, %iterArg_67 = %135, %iterArg_68 = %136, %iterArg_69 = %137, %iterArg_70 = %138, %iterArg_71 = %139, %iterArg_72 = %140, %iterArg_73 = %141, %iterArg_74 = %142, %iterArg_75 = %143, %iterArg_76 = %144, %iterArg_77 = %145, %iterArg_78 = %146, %iterArg_79 = %147, %iterArg_80 = %148, %iterArg_81 = %149, %iterArg_82 = %150, %iterArg_83 = %151, %iterArg_84 = %152, %iterArg_85 = %153, %iterArg_86 = %154, %iterArg_87 = %155, %iterArg_88 = %156, %iterArg_89 = %157, %iterArg_90 = %158, %iterArg_91 = %159, %iterArg_92 = %160, %iterArg_93 = %161, %iterArg_94 = %162, %iterArg_95 = %163, %iterArg_96 = %164, %iterArg_97 = %165, %iterArg_98 = %166, %iterArg_99 = %167, %iterArg_100 = %168, %iterArg_101 = %718, %iterArg_102 = %746, %iterArg_103 = %747, %iterArg_104 = %748, %iterArg_105 = %723, %iterArg_106 = %705, %iterArg_107 = %725, %iterArg_108 = %743, %iterArg_109 = %745, %iterArg_110 = %699, %iterArg_111 = %749, %iterArg_112 = %727, %iterArg_113 = %729, %iterArg_114 = %750, %iterArg_115 = %731, %iterArg_116 = %733, %iterArg_117 = %751, %iterArg_118 = %735, %iterArg_119 = %737, %iterArg_120 = %752, %iterArg_121 = %739, %iterArg_122 = %741) : tensor<51865x384xf32>, tensor<448x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1536xf32>, tensor<1536xf32>, tensor<1536x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1536xf32>, tensor<1536xf32>, tensor<1536x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1536xf32>, tensor<1536xf32>, tensor<1536x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1536xf32>, tensor<1536xf32>, tensor<1536x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<88xi32>, tensor<2xi32>, tensor<4xi32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x448xi32>, tensor<1x1xi32>, tensor<1xi1>, tensor<1x448xi32>, tensor<1x1xi32>, tensor<1x1500x384xf32>, tensor<i32>, tensor<1x6x64x448xf32>, tensor<1x6x64x448xf32>, tensor<i32>, tensor<1x6x64x448xf32>, tensor<1x6x64x448xf32>, tensor<i32>, tensor<1x6x64x448xf32>, tensor<1x6x64x448xf32>, tensor<i32>, tensor<1x6x64x448xf32>, tensor<1x6x64x448xf32>
     cond {
      %754 = stablehlo.constant dense<448> : tensor<i32>
      %755 = stablehlo.compare  EQ, %iterArg_104, %754,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %756 = stablehlo.constant dense<true> : tensor<i1>
      %757 = stablehlo.reduce(%iterArg_107 init: %756) across dimensions = [0] : (tensor<1xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg1: tensor<i1>, %arg2: tensor<i1>)  {
        %763 = stablehlo.and %arg1, %arg2 : tensor<i1>
        stablehlo.return %763 : tensor<i1>
      }
      %758 = stablehlo.constant dense<false> : tensor<i1>
      %759 = stablehlo.broadcast_in_dim %758, dims = [] : (tensor<i1>) -> tensor<i1>
      %760 = stablehlo.compare  NE, %755, %759,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %761 = stablehlo.or %760, %757 : tensor<i1>
      %762 = stablehlo.not %761 : tensor<i1>
      stablehlo.return %762 : tensor<i1>
    } do {
      %754 = stablehlo.iota dim = 0 : tensor<51865xi32>
      %755 = stablehlo.broadcast_in_dim %iterArg_106, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
      %756 = stablehlo.broadcast_in_dim %754, dims = [2] : (tensor<51865xi32>) -> tensor<1x1x51865xi32>
      %757 = stablehlo.broadcast_in_dim %755, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x1x51865xi32>
      %758 = stablehlo.compare  EQ, %757, %756,  SIGNED : (tensor<1x1x51865xi32>, tensor<1x1x51865xi32>) -> tensor<1x1x51865xi1>
      %759 = stablehlo.convert %758 : (tensor<1x1x51865xi1>) -> tensor<1x1x51865xf32>
      %760 = stablehlo.dot_general %759, %iterArg, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x51865xf32>, tensor<51865x384xf32>) -> tensor<1x1x384xf32>
      %761 = stablehlo.iota dim = 0 : tensor<448xi32>
      %762 = stablehlo.broadcast_in_dim %iterArg_109, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
      %763 = stablehlo.broadcast_in_dim %761, dims = [2] : (tensor<448xi32>) -> tensor<1x1x448xi32>
      %764 = stablehlo.broadcast_in_dim %762, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x1x448xi32>
      %765 = stablehlo.compare  EQ, %764, %763,  SIGNED : (tensor<1x1x448xi32>, tensor<1x1x448xi32>) -> tensor<1x1x448xi1>
      %766 = stablehlo.convert %765 : (tensor<1x1x448xi1>) -> tensor<1x1x448xf32>
      %767 = stablehlo.dot_general %766, %iterArg_0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x448xf32>, tensor<448x384xf32>) -> tensor<1x1x384xf32>
      %768 = stablehlo.add %760, %767 : tensor<1x1x384xf32>
      %769 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %770 = stablehlo.reduce(%768 init: %769) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %771 = stablehlo.broadcast_in_dim %770, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %772 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %773 = stablehlo.broadcast_in_dim %772, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %774 = stablehlo.divide %771, %773 : tensor<1x1x1xf32>
      %775 = stablehlo.multiply %768, %768 : tensor<1x1x384xf32>
      %776 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %777 = stablehlo.reduce(%775 init: %776) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %778 = stablehlo.broadcast_in_dim %777, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %779 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %780 = stablehlo.broadcast_in_dim %779, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %781 = stablehlo.divide %778, %780 : tensor<1x1x1xf32>
      %782 = stablehlo.multiply %774, %774 : tensor<1x1x1xf32>
      %783 = stablehlo.subtract %781, %782 : tensor<1x1x1xf32>
      %784 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %785 = stablehlo.broadcast_in_dim %784, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %786 = stablehlo.add %783, %785 : tensor<1x1x1xf32>
      %787 = stablehlo.rsqrt %786 : tensor<1x1x1xf32>
      %788 = stablehlo.broadcast_in_dim %iterArg_1, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %789 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %790 = stablehlo.multiply %789, %788 : tensor<1x1x384xf32>
      %791 = stablehlo.broadcast_in_dim %774, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %792 = stablehlo.subtract %768, %791 : tensor<1x1x384xf32>
      %793 = stablehlo.multiply %792, %790 : tensor<1x1x384xf32>
      %794 = stablehlo.broadcast_in_dim %iterArg_2, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %795 = stablehlo.add %793, %794 : tensor<1x1x384xf32>
      %796 = stablehlo.constant dense<true> : tensor<i1>
      %797 = stablehlo.broadcast_in_dim %796, dims = [] : (tensor<i1>) -> tensor<1x448xi1>
      %798 = stablehlo.iota dim = 0 : tensor<448xi32>
      %799 = stablehlo.broadcast_in_dim %798, dims = [1] : (tensor<448xi32>) -> tensor<1x448xi32>
      %800 = stablehlo.broadcast_in_dim %799, dims = [0, 1] : (tensor<1x448xi32>) -> tensor<1x448x1xi32>
      %801 = stablehlo.broadcast_in_dim %799, dims = [0, 2] : (tensor<1x448xi32>) -> tensor<1x1x448xi32>
      %802 = stablehlo.broadcast_in_dim %800, dims = [0, 1, 2] : (tensor<1x448x1xi32>) -> tensor<1x448x448xi32>
      %803 = stablehlo.broadcast_in_dim %801, dims = [0, 1, 2] : (tensor<1x1x448xi32>) -> tensor<1x448x448xi32>
      %804 = stablehlo.compare  GE, %802, %803,  SIGNED : (tensor<1x448x448xi32>, tensor<1x448x448xi32>) -> tensor<1x448x448xi1>
      %805 = stablehlo.broadcast_in_dim %804, dims = [0, 2, 3] : (tensor<1x448x448xi1>) -> tensor<1x1x448x448xi1>
      %806 = stablehlo.dot_general %795, %iterArg_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %807 = stablehlo.broadcast_in_dim %iterArg_4, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %808 = stablehlo.add %806, %807 : tensor<1x1x384xf32>
      %809 = stablehlo.dot_general %795, %iterArg_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %810 = stablehlo.dot_general %795, %iterArg_6, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %811 = stablehlo.broadcast_in_dim %iterArg_7, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %812 = stablehlo.add %810, %811 : tensor<1x1x384xf32>
      %813 = stablehlo.reshape %808 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %814 = stablehlo.reshape %809 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %815 = stablehlo.reshape %812 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %816 = stablehlo.constant dense<0> : tensor<i32>
      %817 = stablehlo.compare  LT, %iterArg_111, %816,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %818 = stablehlo.constant dense<448> : tensor<i32>
      %819 = stablehlo.add %iterArg_111, %818 : tensor<i32>
      %820 = stablehlo.select %817, %819, %iterArg_111 : tensor<i1>, tensor<i32>
      %821 = stablehlo.constant dense<0> : tensor<i32>
      %822 = stablehlo.constant dense<0> : tensor<i32>
      %823 = stablehlo.constant dense<0> : tensor<i32>
      %824 = stablehlo.dynamic_slice %805, %821, %822, %820, %823, sizes = [1, 1, 1, 448] : (tensor<1x1x448x448xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x448xi1>
      %825 = stablehlo.broadcast_in_dim %iterArg_108, dims = [0, 3] : (tensor<1x448xi32>) -> tensor<1x1x1x448xi32>
      %826 = stablehlo.constant dense<0> : tensor<i32>
      %827 = stablehlo.broadcast_in_dim %826, dims = [] : (tensor<i32>) -> tensor<1x1x1x448xi32>
      %828 = stablehlo.compare  NE, %825, %827,  SIGNED : (tensor<1x1x1x448xi32>, tensor<1x1x1x448xi32>) -> tensor<1x1x1x448xi1>
      %829 = stablehlo.and %828, %824 : tensor<1x1x1x448xi1>
      %830 = stablehlo.convert %829 : (tensor<1x1x1x448xi1>) -> tensor<1x1x1x448xf32>
      %831 = stablehlo.transpose %814, dims = [0, 2, 3, 1] : (tensor<1x1x6x64xf32>) -> tensor<1x6x64x1xf32>
      %832 = stablehlo.transpose %815, dims = [0, 2, 3, 1] : (tensor<1x1x6x64xf32>) -> tensor<1x6x64x1xf32>
      %833 = func.call @_one_hot(%iterArg_111) : (tensor<i32>) -> tensor<448xf32>
      %834 = stablehlo.broadcast_in_dim %833, dims = [3] : (tensor<448xf32>) -> tensor<1x1x1x448xf32>
      %835 = stablehlo.broadcast_in_dim %831, dims = [0, 1, 2, 3] : (tensor<1x6x64x1xf32>) -> tensor<1x6x64x448xf32>
      %836 = stablehlo.broadcast_in_dim %834, dims = [0, 1, 2, 3] : (tensor<1x1x1x448xf32>) -> tensor<1x6x64x448xf32>
      %837 = stablehlo.multiply %835, %836 : tensor<1x6x64x448xf32>
      %838 = stablehlo.add %iterArg_112, %837 : tensor<1x6x64x448xf32>
      %839 = stablehlo.broadcast_in_dim %833, dims = [3] : (tensor<448xf32>) -> tensor<1x1x1x448xf32>
      %840 = stablehlo.broadcast_in_dim %832, dims = [0, 1, 2, 3] : (tensor<1x6x64x1xf32>) -> tensor<1x6x64x448xf32>
      %841 = stablehlo.broadcast_in_dim %839, dims = [0, 1, 2, 3] : (tensor<1x1x1x448xf32>) -> tensor<1x6x64x448xf32>
      %842 = stablehlo.multiply %840, %841 : tensor<1x6x64x448xf32>
      %843 = stablehlo.add %iterArg_113, %842 : tensor<1x6x64x448xf32>
      %844 = stablehlo.constant dense<1> : tensor<i32>
      %845 = stablehlo.add %iterArg_111, %844 : tensor<i32>
      %846 = stablehlo.transpose %838, dims = [0, 3, 1, 2] : (tensor<1x6x64x448xf32>) -> tensor<1x448x6x64xf32>
      %847 = stablehlo.transpose %843, dims = [0, 3, 1, 2] : (tensor<1x6x64x448xf32>) -> tensor<1x448x6x64xf32>
      %848 = stablehlo.iota dim = 0 : tensor<448xi32>
      %849 = stablehlo.constant dense<1> : tensor<i32>
      %850 = stablehlo.add %iterArg_111, %849 : tensor<i32>
      %851 = stablehlo.broadcast_in_dim %850, dims = [] : (tensor<i32>) -> tensor<448xi32>
      %852 = stablehlo.compare  LT, %848, %851,  SIGNED : (tensor<448xi32>, tensor<448xi32>) -> tensor<448xi1>
      %853 = stablehlo.broadcast_in_dim %852, dims = [3] : (tensor<448xi1>) -> tensor<1x1x1x448xi1>
      %854 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %855 = stablehlo.broadcast_in_dim %854, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %856 = stablehlo.compare  NE, %830, %855,  FLOAT : (tensor<1x1x1x448xf32>, tensor<1x1x1x448xf32>) -> tensor<1x1x1x448xi1>
      %857 = stablehlo.and %853, %856 : tensor<1x1x1x448xi1>
      %858 = stablehlo.convert %857 : (tensor<1x1x1x448xi1>) -> tensor<1x1x1x448xf32>
      %859 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %860 = stablehlo.broadcast_in_dim %859, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %861 = stablehlo.compare  GT, %858, %860,  FLOAT : (tensor<1x1x1x448xf32>, tensor<1x1x1x448xf32>) -> tensor<1x1x1x448xi1>
      %862 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %863 = stablehlo.broadcast_in_dim %862, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %864 = stablehlo.convert %863 : tensor<1x1x1x448xf32>
      %865 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %866 = stablehlo.broadcast_in_dim %865, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %867 = stablehlo.select %861, %864, %866 : tensor<1x1x1x448xi1>, tensor<1x1x1x448xf32>
      %868 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %869 = stablehlo.sqrt %868 : tensor<f32>
      %870 = stablehlo.convert %869 : tensor<f32>
      %871 = stablehlo.broadcast_in_dim %870, dims = [] : (tensor<f32>) -> tensor<1x1x6x64xf32>
      %872 = stablehlo.divide %813, %871 : tensor<1x1x6x64xf32>
      %873 = stablehlo.dot_general %872, %846, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x6x64xf32>, tensor<1x448x6x64xf32>) -> tensor<1x6x1x448xf32>
      %874 = stablehlo.broadcast_in_dim %867, dims = [0, 1, 2, 3] : (tensor<1x1x1x448xf32>) -> tensor<1x6x1x448xf32>
      %875 = stablehlo.add %873, %874 : tensor<1x6x1x448xf32>
      %876 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %877 = stablehlo.reduce(%875 init: %876) across dimensions = [3] : (tensor<1x6x1x448xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %878 = stablehlo.broadcast_in_dim %877, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %879 = stablehlo.broadcast_in_dim %878, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x448xf32>
      %880 = stablehlo.subtract %875, %879 : tensor<1x6x1x448xf32>
      %881 = stablehlo.exponential %880 : tensor<1x6x1x448xf32>
      %882 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %883 = stablehlo.reduce(%881 init: %882) across dimensions = [3] : (tensor<1x6x1x448xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %884 = stablehlo.broadcast_in_dim %883, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %885 = stablehlo.broadcast_in_dim %884, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x448xf32>
      %886 = stablehlo.divide %881, %885 : tensor<1x6x1x448xf32>
      %887 = stablehlo.dot_general %847, %886, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x448x6x64xf32>, tensor<1x6x1x448xf32>) -> tensor<1x6x64x1xf32>
      %888 = stablehlo.transpose %887, dims = [0, 3, 1, 2] : (tensor<1x6x64x1xf32>) -> tensor<1x1x6x64xf32>
      %889 = stablehlo.reshape %888 : (tensor<1x1x6x64xf32>) -> tensor<1x1x384xf32>
      %890 = stablehlo.dot_general %889, %iterArg_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %891 = stablehlo.broadcast_in_dim %iterArg_9, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %892 = stablehlo.add %890, %891 : tensor<1x1x384xf32>
      %893 = stablehlo.add %768, %892 : tensor<1x1x384xf32>
      %894 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %895 = stablehlo.reduce(%893 init: %894) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %896 = stablehlo.broadcast_in_dim %895, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %897 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %898 = stablehlo.broadcast_in_dim %897, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %899 = stablehlo.divide %896, %898 : tensor<1x1x1xf32>
      %900 = stablehlo.multiply %893, %893 : tensor<1x1x384xf32>
      %901 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %902 = stablehlo.reduce(%900 init: %901) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %903 = stablehlo.broadcast_in_dim %902, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %904 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %905 = stablehlo.broadcast_in_dim %904, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %906 = stablehlo.divide %903, %905 : tensor<1x1x1xf32>
      %907 = stablehlo.multiply %899, %899 : tensor<1x1x1xf32>
      %908 = stablehlo.subtract %906, %907 : tensor<1x1x1xf32>
      %909 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %910 = stablehlo.broadcast_in_dim %909, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %911 = stablehlo.add %908, %910 : tensor<1x1x1xf32>
      %912 = stablehlo.rsqrt %911 : tensor<1x1x1xf32>
      %913 = stablehlo.broadcast_in_dim %iterArg_10, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %914 = stablehlo.broadcast_in_dim %912, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %915 = stablehlo.multiply %914, %913 : tensor<1x1x384xf32>
      %916 = stablehlo.broadcast_in_dim %899, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %917 = stablehlo.subtract %893, %916 : tensor<1x1x384xf32>
      %918 = stablehlo.multiply %917, %915 : tensor<1x1x384xf32>
      %919 = stablehlo.broadcast_in_dim %iterArg_11, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %920 = stablehlo.add %918, %919 : tensor<1x1x384xf32>
      %921 = stablehlo.dot_general %920, %iterArg_12, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %922 = stablehlo.broadcast_in_dim %iterArg_13, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %923 = stablehlo.add %921, %922 : tensor<1x1x384xf32>
      %924 = stablehlo.dot_general %iterArg_110, %iterArg_14, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
      %925 = stablehlo.dot_general %iterArg_110, %iterArg_15, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
      %926 = stablehlo.broadcast_in_dim %iterArg_16, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %927 = stablehlo.broadcast_in_dim %926, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
      %928 = stablehlo.add %925, %927 : tensor<1x1500x384xf32>
      %929 = stablehlo.reshape %923 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %930 = stablehlo.reshape %924 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
      %931 = stablehlo.reshape %928 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
      %932 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %933 = stablehlo.sqrt %932 : tensor<f32>
      %934 = stablehlo.convert %933 : tensor<f32>
      %935 = stablehlo.broadcast_in_dim %934, dims = [] : (tensor<f32>) -> tensor<1x1x6x64xf32>
      %936 = stablehlo.divide %929, %935 : tensor<1x1x6x64xf32>
      %937 = stablehlo.dot_general %936, %930, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x6x64xf32>, tensor<1x1500x6x64xf32>) -> tensor<1x6x1x1500xf32>
      %938 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %939 = stablehlo.reduce(%937 init: %938) across dimensions = [3] : (tensor<1x6x1x1500xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %940 = stablehlo.broadcast_in_dim %939, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %941 = stablehlo.broadcast_in_dim %940, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x1500xf32>
      %942 = stablehlo.subtract %937, %941 : tensor<1x6x1x1500xf32>
      %943 = stablehlo.exponential %942 : tensor<1x6x1x1500xf32>
      %944 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %945 = stablehlo.reduce(%943 init: %944) across dimensions = [3] : (tensor<1x6x1x1500xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %946 = stablehlo.broadcast_in_dim %945, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %947 = stablehlo.broadcast_in_dim %946, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x1500xf32>
      %948 = stablehlo.divide %943, %947 : tensor<1x6x1x1500xf32>
      %949 = stablehlo.dot_general %931, %948, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x6x64xf32>, tensor<1x6x1x1500xf32>) -> tensor<1x6x64x1xf32>
      %950 = stablehlo.transpose %949, dims = [0, 3, 1, 2] : (tensor<1x6x64x1xf32>) -> tensor<1x1x6x64xf32>
      %951 = stablehlo.reshape %950 : (tensor<1x1x6x64xf32>) -> tensor<1x1x384xf32>
      %952 = stablehlo.dot_general %951, %iterArg_17, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %953 = stablehlo.broadcast_in_dim %iterArg_18, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %954 = stablehlo.add %952, %953 : tensor<1x1x384xf32>
      %955 = stablehlo.add %893, %954 : tensor<1x1x384xf32>
      %956 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %957 = stablehlo.reduce(%955 init: %956) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %958 = stablehlo.broadcast_in_dim %957, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %959 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %960 = stablehlo.broadcast_in_dim %959, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %961 = stablehlo.divide %958, %960 : tensor<1x1x1xf32>
      %962 = stablehlo.multiply %955, %955 : tensor<1x1x384xf32>
      %963 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %964 = stablehlo.reduce(%962 init: %963) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %965 = stablehlo.broadcast_in_dim %964, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %966 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %967 = stablehlo.broadcast_in_dim %966, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %968 = stablehlo.divide %965, %967 : tensor<1x1x1xf32>
      %969 = stablehlo.multiply %961, %961 : tensor<1x1x1xf32>
      %970 = stablehlo.subtract %968, %969 : tensor<1x1x1xf32>
      %971 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %972 = stablehlo.broadcast_in_dim %971, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %973 = stablehlo.add %970, %972 : tensor<1x1x1xf32>
      %974 = stablehlo.rsqrt %973 : tensor<1x1x1xf32>
      %975 = stablehlo.broadcast_in_dim %iterArg_19, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %976 = stablehlo.broadcast_in_dim %974, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %977 = stablehlo.multiply %976, %975 : tensor<1x1x384xf32>
      %978 = stablehlo.broadcast_in_dim %961, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %979 = stablehlo.subtract %955, %978 : tensor<1x1x384xf32>
      %980 = stablehlo.multiply %979, %977 : tensor<1x1x384xf32>
      %981 = stablehlo.broadcast_in_dim %iterArg_20, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %982 = stablehlo.add %980, %981 : tensor<1x1x384xf32>
      %983 = stablehlo.dot_general %982, %iterArg_21, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x1536xf32>) -> tensor<1x1x1536xf32>
      %984 = stablehlo.broadcast_in_dim %iterArg_22, dims = [2] : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
      %985 = stablehlo.add %983, %984 : tensor<1x1x1536xf32>
      %986 = stablehlo.constant dense<1.41421354> : tensor<f32>
      %987 = stablehlo.broadcast_in_dim %986, dims = [] : (tensor<f32>) -> tensor<1x1x1536xf32>
      %988 = stablehlo.divide %985, %987 : tensor<1x1x1536xf32>
      %989 = chlo.erf %988 : tensor<1x1x1536xf32> -> tensor<1x1x1536xf32>
      %990 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %991 = stablehlo.broadcast_in_dim %990, dims = [] : (tensor<f32>) -> tensor<1x1x1536xf32>
      %992 = stablehlo.add %989, %991 : tensor<1x1x1536xf32>
      %993 = stablehlo.multiply %985, %992 : tensor<1x1x1536xf32>
      %994 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %995 = stablehlo.broadcast_in_dim %994, dims = [] : (tensor<f32>) -> tensor<1x1x1536xf32>
      %996 = stablehlo.divide %993, %995 : tensor<1x1x1536xf32>
      %997 = stablehlo.dot_general %996, %iterArg_23, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x1536xf32>, tensor<1536x384xf32>) -> tensor<1x1x384xf32>
      %998 = stablehlo.broadcast_in_dim %iterArg_24, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %999 = stablehlo.add %997, %998 : tensor<1x1x384xf32>
      %1000 = stablehlo.add %955, %999 : tensor<1x1x384xf32>
      %1001 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1002 = stablehlo.reduce(%1000 init: %1001) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1003 = stablehlo.broadcast_in_dim %1002, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1004 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1005 = stablehlo.broadcast_in_dim %1004, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1006 = stablehlo.divide %1003, %1005 : tensor<1x1x1xf32>
      %1007 = stablehlo.multiply %1000, %1000 : tensor<1x1x384xf32>
      %1008 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1009 = stablehlo.reduce(%1007 init: %1008) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1010 = stablehlo.broadcast_in_dim %1009, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1011 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1012 = stablehlo.broadcast_in_dim %1011, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1013 = stablehlo.divide %1010, %1012 : tensor<1x1x1xf32>
      %1014 = stablehlo.multiply %1006, %1006 : tensor<1x1x1xf32>
      %1015 = stablehlo.subtract %1013, %1014 : tensor<1x1x1xf32>
      %1016 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1017 = stablehlo.broadcast_in_dim %1016, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1018 = stablehlo.add %1015, %1017 : tensor<1x1x1xf32>
      %1019 = stablehlo.rsqrt %1018 : tensor<1x1x1xf32>
      %1020 = stablehlo.broadcast_in_dim %iterArg_25, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1021 = stablehlo.broadcast_in_dim %1019, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1022 = stablehlo.multiply %1021, %1020 : tensor<1x1x384xf32>
      %1023 = stablehlo.broadcast_in_dim %1006, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1024 = stablehlo.subtract %1000, %1023 : tensor<1x1x384xf32>
      %1025 = stablehlo.multiply %1024, %1022 : tensor<1x1x384xf32>
      %1026 = stablehlo.broadcast_in_dim %iterArg_26, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1027 = stablehlo.add %1025, %1026 : tensor<1x1x384xf32>
      %1028 = stablehlo.constant dense<true> : tensor<i1>
      %1029 = stablehlo.broadcast_in_dim %1028, dims = [] : (tensor<i1>) -> tensor<1x448xi1>
      %1030 = stablehlo.iota dim = 0 : tensor<448xi32>
      %1031 = stablehlo.broadcast_in_dim %1030, dims = [1] : (tensor<448xi32>) -> tensor<1x448xi32>
      %1032 = stablehlo.broadcast_in_dim %1031, dims = [0, 1] : (tensor<1x448xi32>) -> tensor<1x448x1xi32>
      %1033 = stablehlo.broadcast_in_dim %1031, dims = [0, 2] : (tensor<1x448xi32>) -> tensor<1x1x448xi32>
      %1034 = stablehlo.broadcast_in_dim %1032, dims = [0, 1, 2] : (tensor<1x448x1xi32>) -> tensor<1x448x448xi32>
      %1035 = stablehlo.broadcast_in_dim %1033, dims = [0, 1, 2] : (tensor<1x1x448xi32>) -> tensor<1x448x448xi32>
      %1036 = stablehlo.compare  GE, %1034, %1035,  SIGNED : (tensor<1x448x448xi32>, tensor<1x448x448xi32>) -> tensor<1x448x448xi1>
      %1037 = stablehlo.broadcast_in_dim %1036, dims = [0, 2, 3] : (tensor<1x448x448xi1>) -> tensor<1x1x448x448xi1>
      %1038 = stablehlo.dot_general %1027, %iterArg_27, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1039 = stablehlo.broadcast_in_dim %iterArg_28, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1040 = stablehlo.add %1038, %1039 : tensor<1x1x384xf32>
      %1041 = stablehlo.dot_general %1027, %iterArg_29, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1042 = stablehlo.dot_general %1027, %iterArg_30, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1043 = stablehlo.broadcast_in_dim %iterArg_31, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1044 = stablehlo.add %1042, %1043 : tensor<1x1x384xf32>
      %1045 = stablehlo.reshape %1040 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %1046 = stablehlo.reshape %1041 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %1047 = stablehlo.reshape %1044 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %1048 = stablehlo.constant dense<0> : tensor<i32>
      %1049 = stablehlo.compare  LT, %iterArg_114, %1048,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1050 = stablehlo.constant dense<448> : tensor<i32>
      %1051 = stablehlo.add %iterArg_114, %1050 : tensor<i32>
      %1052 = stablehlo.select %1049, %1051, %iterArg_114 : tensor<i1>, tensor<i32>
      %1053 = stablehlo.constant dense<0> : tensor<i32>
      %1054 = stablehlo.constant dense<0> : tensor<i32>
      %1055 = stablehlo.constant dense<0> : tensor<i32>
      %1056 = stablehlo.dynamic_slice %1037, %1053, %1054, %1052, %1055, sizes = [1, 1, 1, 448] : (tensor<1x1x448x448xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x448xi1>
      %1057 = stablehlo.broadcast_in_dim %iterArg_108, dims = [0, 3] : (tensor<1x448xi32>) -> tensor<1x1x1x448xi32>
      %1058 = stablehlo.constant dense<0> : tensor<i32>
      %1059 = stablehlo.broadcast_in_dim %1058, dims = [] : (tensor<i32>) -> tensor<1x1x1x448xi32>
      %1060 = stablehlo.compare  NE, %1057, %1059,  SIGNED : (tensor<1x1x1x448xi32>, tensor<1x1x1x448xi32>) -> tensor<1x1x1x448xi1>
      %1061 = stablehlo.and %1060, %1056 : tensor<1x1x1x448xi1>
      %1062 = stablehlo.convert %1061 : (tensor<1x1x1x448xi1>) -> tensor<1x1x1x448xf32>
      %1063 = stablehlo.transpose %1046, dims = [0, 2, 3, 1] : (tensor<1x1x6x64xf32>) -> tensor<1x6x64x1xf32>
      %1064 = stablehlo.transpose %1047, dims = [0, 2, 3, 1] : (tensor<1x1x6x64xf32>) -> tensor<1x6x64x1xf32>
      %1065 = func.call @_one_hot_0(%iterArg_114) : (tensor<i32>) -> tensor<448xf32>
      %1066 = stablehlo.broadcast_in_dim %1065, dims = [3] : (tensor<448xf32>) -> tensor<1x1x1x448xf32>
      %1067 = stablehlo.broadcast_in_dim %1063, dims = [0, 1, 2, 3] : (tensor<1x6x64x1xf32>) -> tensor<1x6x64x448xf32>
      %1068 = stablehlo.broadcast_in_dim %1066, dims = [0, 1, 2, 3] : (tensor<1x1x1x448xf32>) -> tensor<1x6x64x448xf32>
      %1069 = stablehlo.multiply %1067, %1068 : tensor<1x6x64x448xf32>
      %1070 = stablehlo.add %iterArg_115, %1069 : tensor<1x6x64x448xf32>
      %1071 = stablehlo.broadcast_in_dim %1065, dims = [3] : (tensor<448xf32>) -> tensor<1x1x1x448xf32>
      %1072 = stablehlo.broadcast_in_dim %1064, dims = [0, 1, 2, 3] : (tensor<1x6x64x1xf32>) -> tensor<1x6x64x448xf32>
      %1073 = stablehlo.broadcast_in_dim %1071, dims = [0, 1, 2, 3] : (tensor<1x1x1x448xf32>) -> tensor<1x6x64x448xf32>
      %1074 = stablehlo.multiply %1072, %1073 : tensor<1x6x64x448xf32>
      %1075 = stablehlo.add %iterArg_116, %1074 : tensor<1x6x64x448xf32>
      %1076 = stablehlo.constant dense<1> : tensor<i32>
      %1077 = stablehlo.add %iterArg_114, %1076 : tensor<i32>
      %1078 = stablehlo.transpose %1070, dims = [0, 3, 1, 2] : (tensor<1x6x64x448xf32>) -> tensor<1x448x6x64xf32>
      %1079 = stablehlo.transpose %1075, dims = [0, 3, 1, 2] : (tensor<1x6x64x448xf32>) -> tensor<1x448x6x64xf32>
      %1080 = stablehlo.iota dim = 0 : tensor<448xi32>
      %1081 = stablehlo.constant dense<1> : tensor<i32>
      %1082 = stablehlo.add %iterArg_114, %1081 : tensor<i32>
      %1083 = stablehlo.broadcast_in_dim %1082, dims = [] : (tensor<i32>) -> tensor<448xi32>
      %1084 = stablehlo.compare  LT, %1080, %1083,  SIGNED : (tensor<448xi32>, tensor<448xi32>) -> tensor<448xi1>
      %1085 = stablehlo.broadcast_in_dim %1084, dims = [3] : (tensor<448xi1>) -> tensor<1x1x1x448xi1>
      %1086 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1087 = stablehlo.broadcast_in_dim %1086, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %1088 = stablehlo.compare  NE, %1062, %1087,  FLOAT : (tensor<1x1x1x448xf32>, tensor<1x1x1x448xf32>) -> tensor<1x1x1x448xi1>
      %1089 = stablehlo.and %1085, %1088 : tensor<1x1x1x448xi1>
      %1090 = stablehlo.convert %1089 : (tensor<1x1x1x448xi1>) -> tensor<1x1x1x448xf32>
      %1091 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1092 = stablehlo.broadcast_in_dim %1091, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %1093 = stablehlo.compare  GT, %1090, %1092,  FLOAT : (tensor<1x1x1x448xf32>, tensor<1x1x1x448xf32>) -> tensor<1x1x1x448xi1>
      %1094 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1095 = stablehlo.broadcast_in_dim %1094, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %1096 = stablehlo.convert %1095 : tensor<1x1x1x448xf32>
      %1097 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %1098 = stablehlo.broadcast_in_dim %1097, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %1099 = stablehlo.select %1093, %1096, %1098 : tensor<1x1x1x448xi1>, tensor<1x1x1x448xf32>
      %1100 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %1101 = stablehlo.sqrt %1100 : tensor<f32>
      %1102 = stablehlo.convert %1101 : tensor<f32>
      %1103 = stablehlo.broadcast_in_dim %1102, dims = [] : (tensor<f32>) -> tensor<1x1x6x64xf32>
      %1104 = stablehlo.divide %1045, %1103 : tensor<1x1x6x64xf32>
      %1105 = stablehlo.dot_general %1104, %1078, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x6x64xf32>, tensor<1x448x6x64xf32>) -> tensor<1x6x1x448xf32>
      %1106 = stablehlo.broadcast_in_dim %1099, dims = [0, 1, 2, 3] : (tensor<1x1x1x448xf32>) -> tensor<1x6x1x448xf32>
      %1107 = stablehlo.add %1105, %1106 : tensor<1x6x1x448xf32>
      %1108 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %1109 = stablehlo.reduce(%1107 init: %1108) across dimensions = [3] : (tensor<1x6x1x448xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1110 = stablehlo.broadcast_in_dim %1109, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %1111 = stablehlo.broadcast_in_dim %1110, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x448xf32>
      %1112 = stablehlo.subtract %1107, %1111 : tensor<1x6x1x448xf32>
      %1113 = stablehlo.exponential %1112 : tensor<1x6x1x448xf32>
      %1114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1115 = stablehlo.reduce(%1113 init: %1114) across dimensions = [3] : (tensor<1x6x1x448xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1116 = stablehlo.broadcast_in_dim %1115, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %1117 = stablehlo.broadcast_in_dim %1116, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x448xf32>
      %1118 = stablehlo.divide %1113, %1117 : tensor<1x6x1x448xf32>
      %1119 = stablehlo.dot_general %1079, %1118, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x448x6x64xf32>, tensor<1x6x1x448xf32>) -> tensor<1x6x64x1xf32>
      %1120 = stablehlo.transpose %1119, dims = [0, 3, 1, 2] : (tensor<1x6x64x1xf32>) -> tensor<1x1x6x64xf32>
      %1121 = stablehlo.reshape %1120 : (tensor<1x1x6x64xf32>) -> tensor<1x1x384xf32>
      %1122 = stablehlo.dot_general %1121, %iterArg_32, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1123 = stablehlo.broadcast_in_dim %iterArg_33, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1124 = stablehlo.add %1122, %1123 : tensor<1x1x384xf32>
      %1125 = stablehlo.add %1000, %1124 : tensor<1x1x384xf32>
      %1126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1127 = stablehlo.reduce(%1125 init: %1126) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1128 = stablehlo.broadcast_in_dim %1127, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1129 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1130 = stablehlo.broadcast_in_dim %1129, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1131 = stablehlo.divide %1128, %1130 : tensor<1x1x1xf32>
      %1132 = stablehlo.multiply %1125, %1125 : tensor<1x1x384xf32>
      %1133 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1134 = stablehlo.reduce(%1132 init: %1133) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1135 = stablehlo.broadcast_in_dim %1134, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1136 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1137 = stablehlo.broadcast_in_dim %1136, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1138 = stablehlo.divide %1135, %1137 : tensor<1x1x1xf32>
      %1139 = stablehlo.multiply %1131, %1131 : tensor<1x1x1xf32>
      %1140 = stablehlo.subtract %1138, %1139 : tensor<1x1x1xf32>
      %1141 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1142 = stablehlo.broadcast_in_dim %1141, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1143 = stablehlo.add %1140, %1142 : tensor<1x1x1xf32>
      %1144 = stablehlo.rsqrt %1143 : tensor<1x1x1xf32>
      %1145 = stablehlo.broadcast_in_dim %iterArg_34, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1146 = stablehlo.broadcast_in_dim %1144, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1147 = stablehlo.multiply %1146, %1145 : tensor<1x1x384xf32>
      %1148 = stablehlo.broadcast_in_dim %1131, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1149 = stablehlo.subtract %1125, %1148 : tensor<1x1x384xf32>
      %1150 = stablehlo.multiply %1149, %1147 : tensor<1x1x384xf32>
      %1151 = stablehlo.broadcast_in_dim %iterArg_35, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1152 = stablehlo.add %1150, %1151 : tensor<1x1x384xf32>
      %1153 = stablehlo.dot_general %1152, %iterArg_36, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1154 = stablehlo.broadcast_in_dim %iterArg_37, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1155 = stablehlo.add %1153, %1154 : tensor<1x1x384xf32>
      %1156 = stablehlo.dot_general %iterArg_110, %iterArg_38, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
      %1157 = stablehlo.dot_general %iterArg_110, %iterArg_39, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
      %1158 = stablehlo.broadcast_in_dim %iterArg_40, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1159 = stablehlo.broadcast_in_dim %1158, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
      %1160 = stablehlo.add %1157, %1159 : tensor<1x1500x384xf32>
      %1161 = stablehlo.reshape %1155 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %1162 = stablehlo.reshape %1156 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
      %1163 = stablehlo.reshape %1160 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
      %1164 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %1165 = stablehlo.sqrt %1164 : tensor<f32>
      %1166 = stablehlo.convert %1165 : tensor<f32>
      %1167 = stablehlo.broadcast_in_dim %1166, dims = [] : (tensor<f32>) -> tensor<1x1x6x64xf32>
      %1168 = stablehlo.divide %1161, %1167 : tensor<1x1x6x64xf32>
      %1169 = stablehlo.dot_general %1168, %1162, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x6x64xf32>, tensor<1x1500x6x64xf32>) -> tensor<1x6x1x1500xf32>
      %1170 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %1171 = stablehlo.reduce(%1169 init: %1170) across dimensions = [3] : (tensor<1x6x1x1500xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1172 = stablehlo.broadcast_in_dim %1171, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %1173 = stablehlo.broadcast_in_dim %1172, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x1500xf32>
      %1174 = stablehlo.subtract %1169, %1173 : tensor<1x6x1x1500xf32>
      %1175 = stablehlo.exponential %1174 : tensor<1x6x1x1500xf32>
      %1176 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1177 = stablehlo.reduce(%1175 init: %1176) across dimensions = [3] : (tensor<1x6x1x1500xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1178 = stablehlo.broadcast_in_dim %1177, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %1179 = stablehlo.broadcast_in_dim %1178, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x1500xf32>
      %1180 = stablehlo.divide %1175, %1179 : tensor<1x6x1x1500xf32>
      %1181 = stablehlo.dot_general %1163, %1180, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x6x64xf32>, tensor<1x6x1x1500xf32>) -> tensor<1x6x64x1xf32>
      %1182 = stablehlo.transpose %1181, dims = [0, 3, 1, 2] : (tensor<1x6x64x1xf32>) -> tensor<1x1x6x64xf32>
      %1183 = stablehlo.reshape %1182 : (tensor<1x1x6x64xf32>) -> tensor<1x1x384xf32>
      %1184 = stablehlo.dot_general %1183, %iterArg_41, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1185 = stablehlo.broadcast_in_dim %iterArg_42, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1186 = stablehlo.add %1184, %1185 : tensor<1x1x384xf32>
      %1187 = stablehlo.add %1125, %1186 : tensor<1x1x384xf32>
      %1188 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1189 = stablehlo.reduce(%1187 init: %1188) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1190 = stablehlo.broadcast_in_dim %1189, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1191 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1192 = stablehlo.broadcast_in_dim %1191, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1193 = stablehlo.divide %1190, %1192 : tensor<1x1x1xf32>
      %1194 = stablehlo.multiply %1187, %1187 : tensor<1x1x384xf32>
      %1195 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1196 = stablehlo.reduce(%1194 init: %1195) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1197 = stablehlo.broadcast_in_dim %1196, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1198 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1199 = stablehlo.broadcast_in_dim %1198, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1200 = stablehlo.divide %1197, %1199 : tensor<1x1x1xf32>
      %1201 = stablehlo.multiply %1193, %1193 : tensor<1x1x1xf32>
      %1202 = stablehlo.subtract %1200, %1201 : tensor<1x1x1xf32>
      %1203 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1204 = stablehlo.broadcast_in_dim %1203, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1205 = stablehlo.add %1202, %1204 : tensor<1x1x1xf32>
      %1206 = stablehlo.rsqrt %1205 : tensor<1x1x1xf32>
      %1207 = stablehlo.broadcast_in_dim %iterArg_43, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1208 = stablehlo.broadcast_in_dim %1206, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1209 = stablehlo.multiply %1208, %1207 : tensor<1x1x384xf32>
      %1210 = stablehlo.broadcast_in_dim %1193, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1211 = stablehlo.subtract %1187, %1210 : tensor<1x1x384xf32>
      %1212 = stablehlo.multiply %1211, %1209 : tensor<1x1x384xf32>
      %1213 = stablehlo.broadcast_in_dim %iterArg_44, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1214 = stablehlo.add %1212, %1213 : tensor<1x1x384xf32>
      %1215 = stablehlo.dot_general %1214, %iterArg_45, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x1536xf32>) -> tensor<1x1x1536xf32>
      %1216 = stablehlo.broadcast_in_dim %iterArg_46, dims = [2] : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
      %1217 = stablehlo.add %1215, %1216 : tensor<1x1x1536xf32>
      %1218 = stablehlo.constant dense<1.41421354> : tensor<f32>
      %1219 = stablehlo.broadcast_in_dim %1218, dims = [] : (tensor<f32>) -> tensor<1x1x1536xf32>
      %1220 = stablehlo.divide %1217, %1219 : tensor<1x1x1536xf32>
      %1221 = chlo.erf %1220 : tensor<1x1x1536xf32> -> tensor<1x1x1536xf32>
      %1222 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1223 = stablehlo.broadcast_in_dim %1222, dims = [] : (tensor<f32>) -> tensor<1x1x1536xf32>
      %1224 = stablehlo.add %1221, %1223 : tensor<1x1x1536xf32>
      %1225 = stablehlo.multiply %1217, %1224 : tensor<1x1x1536xf32>
      %1226 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1227 = stablehlo.broadcast_in_dim %1226, dims = [] : (tensor<f32>) -> tensor<1x1x1536xf32>
      %1228 = stablehlo.divide %1225, %1227 : tensor<1x1x1536xf32>
      %1229 = stablehlo.dot_general %1228, %iterArg_47, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x1536xf32>, tensor<1536x384xf32>) -> tensor<1x1x384xf32>
      %1230 = stablehlo.broadcast_in_dim %iterArg_48, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1231 = stablehlo.add %1229, %1230 : tensor<1x1x384xf32>
      %1232 = stablehlo.add %1187, %1231 : tensor<1x1x384xf32>
      %1233 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1234 = stablehlo.reduce(%1232 init: %1233) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1235 = stablehlo.broadcast_in_dim %1234, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1236 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1237 = stablehlo.broadcast_in_dim %1236, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1238 = stablehlo.divide %1235, %1237 : tensor<1x1x1xf32>
      %1239 = stablehlo.multiply %1232, %1232 : tensor<1x1x384xf32>
      %1240 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1241 = stablehlo.reduce(%1239 init: %1240) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1242 = stablehlo.broadcast_in_dim %1241, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1243 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1244 = stablehlo.broadcast_in_dim %1243, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1245 = stablehlo.divide %1242, %1244 : tensor<1x1x1xf32>
      %1246 = stablehlo.multiply %1238, %1238 : tensor<1x1x1xf32>
      %1247 = stablehlo.subtract %1245, %1246 : tensor<1x1x1xf32>
      %1248 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1249 = stablehlo.broadcast_in_dim %1248, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1250 = stablehlo.add %1247, %1249 : tensor<1x1x1xf32>
      %1251 = stablehlo.rsqrt %1250 : tensor<1x1x1xf32>
      %1252 = stablehlo.broadcast_in_dim %iterArg_49, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1253 = stablehlo.broadcast_in_dim %1251, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1254 = stablehlo.multiply %1253, %1252 : tensor<1x1x384xf32>
      %1255 = stablehlo.broadcast_in_dim %1238, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1256 = stablehlo.subtract %1232, %1255 : tensor<1x1x384xf32>
      %1257 = stablehlo.multiply %1256, %1254 : tensor<1x1x384xf32>
      %1258 = stablehlo.broadcast_in_dim %iterArg_50, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1259 = stablehlo.add %1257, %1258 : tensor<1x1x384xf32>
      %1260 = stablehlo.constant dense<true> : tensor<i1>
      %1261 = stablehlo.broadcast_in_dim %1260, dims = [] : (tensor<i1>) -> tensor<1x448xi1>
      %1262 = stablehlo.iota dim = 0 : tensor<448xi32>
      %1263 = stablehlo.broadcast_in_dim %1262, dims = [1] : (tensor<448xi32>) -> tensor<1x448xi32>
      %1264 = stablehlo.broadcast_in_dim %1263, dims = [0, 1] : (tensor<1x448xi32>) -> tensor<1x448x1xi32>
      %1265 = stablehlo.broadcast_in_dim %1263, dims = [0, 2] : (tensor<1x448xi32>) -> tensor<1x1x448xi32>
      %1266 = stablehlo.broadcast_in_dim %1264, dims = [0, 1, 2] : (tensor<1x448x1xi32>) -> tensor<1x448x448xi32>
      %1267 = stablehlo.broadcast_in_dim %1265, dims = [0, 1, 2] : (tensor<1x1x448xi32>) -> tensor<1x448x448xi32>
      %1268 = stablehlo.compare  GE, %1266, %1267,  SIGNED : (tensor<1x448x448xi32>, tensor<1x448x448xi32>) -> tensor<1x448x448xi1>
      %1269 = stablehlo.broadcast_in_dim %1268, dims = [0, 2, 3] : (tensor<1x448x448xi1>) -> tensor<1x1x448x448xi1>
      %1270 = stablehlo.dot_general %1259, %iterArg_51, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1271 = stablehlo.broadcast_in_dim %iterArg_52, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1272 = stablehlo.add %1270, %1271 : tensor<1x1x384xf32>
      %1273 = stablehlo.dot_general %1259, %iterArg_53, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1274 = stablehlo.dot_general %1259, %iterArg_54, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1275 = stablehlo.broadcast_in_dim %iterArg_55, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1276 = stablehlo.add %1274, %1275 : tensor<1x1x384xf32>
      %1277 = stablehlo.reshape %1272 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %1278 = stablehlo.reshape %1273 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %1279 = stablehlo.reshape %1276 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %1280 = stablehlo.constant dense<0> : tensor<i32>
      %1281 = stablehlo.compare  LT, %iterArg_117, %1280,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1282 = stablehlo.constant dense<448> : tensor<i32>
      %1283 = stablehlo.add %iterArg_117, %1282 : tensor<i32>
      %1284 = stablehlo.select %1281, %1283, %iterArg_117 : tensor<i1>, tensor<i32>
      %1285 = stablehlo.constant dense<0> : tensor<i32>
      %1286 = stablehlo.constant dense<0> : tensor<i32>
      %1287 = stablehlo.constant dense<0> : tensor<i32>
      %1288 = stablehlo.dynamic_slice %1269, %1285, %1286, %1284, %1287, sizes = [1, 1, 1, 448] : (tensor<1x1x448x448xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x448xi1>
      %1289 = stablehlo.broadcast_in_dim %iterArg_108, dims = [0, 3] : (tensor<1x448xi32>) -> tensor<1x1x1x448xi32>
      %1290 = stablehlo.constant dense<0> : tensor<i32>
      %1291 = stablehlo.broadcast_in_dim %1290, dims = [] : (tensor<i32>) -> tensor<1x1x1x448xi32>
      %1292 = stablehlo.compare  NE, %1289, %1291,  SIGNED : (tensor<1x1x1x448xi32>, tensor<1x1x1x448xi32>) -> tensor<1x1x1x448xi1>
      %1293 = stablehlo.and %1292, %1288 : tensor<1x1x1x448xi1>
      %1294 = stablehlo.convert %1293 : (tensor<1x1x1x448xi1>) -> tensor<1x1x1x448xf32>
      %1295 = stablehlo.transpose %1278, dims = [0, 2, 3, 1] : (tensor<1x1x6x64xf32>) -> tensor<1x6x64x1xf32>
      %1296 = stablehlo.transpose %1279, dims = [0, 2, 3, 1] : (tensor<1x1x6x64xf32>) -> tensor<1x6x64x1xf32>
      %1297 = func.call @_one_hot_1(%iterArg_117) : (tensor<i32>) -> tensor<448xf32>
      %1298 = stablehlo.broadcast_in_dim %1297, dims = [3] : (tensor<448xf32>) -> tensor<1x1x1x448xf32>
      %1299 = stablehlo.broadcast_in_dim %1295, dims = [0, 1, 2, 3] : (tensor<1x6x64x1xf32>) -> tensor<1x6x64x448xf32>
      %1300 = stablehlo.broadcast_in_dim %1298, dims = [0, 1, 2, 3] : (tensor<1x1x1x448xf32>) -> tensor<1x6x64x448xf32>
      %1301 = stablehlo.multiply %1299, %1300 : tensor<1x6x64x448xf32>
      %1302 = stablehlo.add %iterArg_118, %1301 : tensor<1x6x64x448xf32>
      %1303 = stablehlo.broadcast_in_dim %1297, dims = [3] : (tensor<448xf32>) -> tensor<1x1x1x448xf32>
      %1304 = stablehlo.broadcast_in_dim %1296, dims = [0, 1, 2, 3] : (tensor<1x6x64x1xf32>) -> tensor<1x6x64x448xf32>
      %1305 = stablehlo.broadcast_in_dim %1303, dims = [0, 1, 2, 3] : (tensor<1x1x1x448xf32>) -> tensor<1x6x64x448xf32>
      %1306 = stablehlo.multiply %1304, %1305 : tensor<1x6x64x448xf32>
      %1307 = stablehlo.add %iterArg_119, %1306 : tensor<1x6x64x448xf32>
      %1308 = stablehlo.constant dense<1> : tensor<i32>
      %1309 = stablehlo.add %iterArg_117, %1308 : tensor<i32>
      %1310 = stablehlo.transpose %1302, dims = [0, 3, 1, 2] : (tensor<1x6x64x448xf32>) -> tensor<1x448x6x64xf32>
      %1311 = stablehlo.transpose %1307, dims = [0, 3, 1, 2] : (tensor<1x6x64x448xf32>) -> tensor<1x448x6x64xf32>
      %1312 = stablehlo.iota dim = 0 : tensor<448xi32>
      %1313 = stablehlo.constant dense<1> : tensor<i32>
      %1314 = stablehlo.add %iterArg_117, %1313 : tensor<i32>
      %1315 = stablehlo.broadcast_in_dim %1314, dims = [] : (tensor<i32>) -> tensor<448xi32>
      %1316 = stablehlo.compare  LT, %1312, %1315,  SIGNED : (tensor<448xi32>, tensor<448xi32>) -> tensor<448xi1>
      %1317 = stablehlo.broadcast_in_dim %1316, dims = [3] : (tensor<448xi1>) -> tensor<1x1x1x448xi1>
      %1318 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1319 = stablehlo.broadcast_in_dim %1318, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %1320 = stablehlo.compare  NE, %1294, %1319,  FLOAT : (tensor<1x1x1x448xf32>, tensor<1x1x1x448xf32>) -> tensor<1x1x1x448xi1>
      %1321 = stablehlo.and %1317, %1320 : tensor<1x1x1x448xi1>
      %1322 = stablehlo.convert %1321 : (tensor<1x1x1x448xi1>) -> tensor<1x1x1x448xf32>
      %1323 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1324 = stablehlo.broadcast_in_dim %1323, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %1325 = stablehlo.compare  GT, %1322, %1324,  FLOAT : (tensor<1x1x1x448xf32>, tensor<1x1x1x448xf32>) -> tensor<1x1x1x448xi1>
      %1326 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1327 = stablehlo.broadcast_in_dim %1326, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %1328 = stablehlo.convert %1327 : tensor<1x1x1x448xf32>
      %1329 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %1330 = stablehlo.broadcast_in_dim %1329, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %1331 = stablehlo.select %1325, %1328, %1330 : tensor<1x1x1x448xi1>, tensor<1x1x1x448xf32>
      %1332 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %1333 = stablehlo.sqrt %1332 : tensor<f32>
      %1334 = stablehlo.convert %1333 : tensor<f32>
      %1335 = stablehlo.broadcast_in_dim %1334, dims = [] : (tensor<f32>) -> tensor<1x1x6x64xf32>
      %1336 = stablehlo.divide %1277, %1335 : tensor<1x1x6x64xf32>
      %1337 = stablehlo.dot_general %1336, %1310, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x6x64xf32>, tensor<1x448x6x64xf32>) -> tensor<1x6x1x448xf32>
      %1338 = stablehlo.broadcast_in_dim %1331, dims = [0, 1, 2, 3] : (tensor<1x1x1x448xf32>) -> tensor<1x6x1x448xf32>
      %1339 = stablehlo.add %1337, %1338 : tensor<1x6x1x448xf32>
      %1340 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %1341 = stablehlo.reduce(%1339 init: %1340) across dimensions = [3] : (tensor<1x6x1x448xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1342 = stablehlo.broadcast_in_dim %1341, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %1343 = stablehlo.broadcast_in_dim %1342, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x448xf32>
      %1344 = stablehlo.subtract %1339, %1343 : tensor<1x6x1x448xf32>
      %1345 = stablehlo.exponential %1344 : tensor<1x6x1x448xf32>
      %1346 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1347 = stablehlo.reduce(%1345 init: %1346) across dimensions = [3] : (tensor<1x6x1x448xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1348 = stablehlo.broadcast_in_dim %1347, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %1349 = stablehlo.broadcast_in_dim %1348, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x448xf32>
      %1350 = stablehlo.divide %1345, %1349 : tensor<1x6x1x448xf32>
      %1351 = stablehlo.dot_general %1311, %1350, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x448x6x64xf32>, tensor<1x6x1x448xf32>) -> tensor<1x6x64x1xf32>
      %1352 = stablehlo.transpose %1351, dims = [0, 3, 1, 2] : (tensor<1x6x64x1xf32>) -> tensor<1x1x6x64xf32>
      %1353 = stablehlo.reshape %1352 : (tensor<1x1x6x64xf32>) -> tensor<1x1x384xf32>
      %1354 = stablehlo.dot_general %1353, %iterArg_56, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1355 = stablehlo.broadcast_in_dim %iterArg_57, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1356 = stablehlo.add %1354, %1355 : tensor<1x1x384xf32>
      %1357 = stablehlo.add %1232, %1356 : tensor<1x1x384xf32>
      %1358 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1359 = stablehlo.reduce(%1357 init: %1358) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1360 = stablehlo.broadcast_in_dim %1359, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1361 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1362 = stablehlo.broadcast_in_dim %1361, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1363 = stablehlo.divide %1360, %1362 : tensor<1x1x1xf32>
      %1364 = stablehlo.multiply %1357, %1357 : tensor<1x1x384xf32>
      %1365 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1366 = stablehlo.reduce(%1364 init: %1365) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1367 = stablehlo.broadcast_in_dim %1366, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1368 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1369 = stablehlo.broadcast_in_dim %1368, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1370 = stablehlo.divide %1367, %1369 : tensor<1x1x1xf32>
      %1371 = stablehlo.multiply %1363, %1363 : tensor<1x1x1xf32>
      %1372 = stablehlo.subtract %1370, %1371 : tensor<1x1x1xf32>
      %1373 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1374 = stablehlo.broadcast_in_dim %1373, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1375 = stablehlo.add %1372, %1374 : tensor<1x1x1xf32>
      %1376 = stablehlo.rsqrt %1375 : tensor<1x1x1xf32>
      %1377 = stablehlo.broadcast_in_dim %iterArg_58, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1378 = stablehlo.broadcast_in_dim %1376, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1379 = stablehlo.multiply %1378, %1377 : tensor<1x1x384xf32>
      %1380 = stablehlo.broadcast_in_dim %1363, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1381 = stablehlo.subtract %1357, %1380 : tensor<1x1x384xf32>
      %1382 = stablehlo.multiply %1381, %1379 : tensor<1x1x384xf32>
      %1383 = stablehlo.broadcast_in_dim %iterArg_59, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1384 = stablehlo.add %1382, %1383 : tensor<1x1x384xf32>
      %1385 = stablehlo.dot_general %1384, %iterArg_60, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1386 = stablehlo.broadcast_in_dim %iterArg_61, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1387 = stablehlo.add %1385, %1386 : tensor<1x1x384xf32>
      %1388 = stablehlo.dot_general %iterArg_110, %iterArg_62, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
      %1389 = stablehlo.dot_general %iterArg_110, %iterArg_63, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
      %1390 = stablehlo.broadcast_in_dim %iterArg_64, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1391 = stablehlo.broadcast_in_dim %1390, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
      %1392 = stablehlo.add %1389, %1391 : tensor<1x1500x384xf32>
      %1393 = stablehlo.reshape %1387 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %1394 = stablehlo.reshape %1388 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
      %1395 = stablehlo.reshape %1392 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
      %1396 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %1397 = stablehlo.sqrt %1396 : tensor<f32>
      %1398 = stablehlo.convert %1397 : tensor<f32>
      %1399 = stablehlo.broadcast_in_dim %1398, dims = [] : (tensor<f32>) -> tensor<1x1x6x64xf32>
      %1400 = stablehlo.divide %1393, %1399 : tensor<1x1x6x64xf32>
      %1401 = stablehlo.dot_general %1400, %1394, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x6x64xf32>, tensor<1x1500x6x64xf32>) -> tensor<1x6x1x1500xf32>
      %1402 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %1403 = stablehlo.reduce(%1401 init: %1402) across dimensions = [3] : (tensor<1x6x1x1500xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1404 = stablehlo.broadcast_in_dim %1403, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %1405 = stablehlo.broadcast_in_dim %1404, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x1500xf32>
      %1406 = stablehlo.subtract %1401, %1405 : tensor<1x6x1x1500xf32>
      %1407 = stablehlo.exponential %1406 : tensor<1x6x1x1500xf32>
      %1408 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1409 = stablehlo.reduce(%1407 init: %1408) across dimensions = [3] : (tensor<1x6x1x1500xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1410 = stablehlo.broadcast_in_dim %1409, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %1411 = stablehlo.broadcast_in_dim %1410, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x1500xf32>
      %1412 = stablehlo.divide %1407, %1411 : tensor<1x6x1x1500xf32>
      %1413 = stablehlo.dot_general %1395, %1412, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x6x64xf32>, tensor<1x6x1x1500xf32>) -> tensor<1x6x64x1xf32>
      %1414 = stablehlo.transpose %1413, dims = [0, 3, 1, 2] : (tensor<1x6x64x1xf32>) -> tensor<1x1x6x64xf32>
      %1415 = stablehlo.reshape %1414 : (tensor<1x1x6x64xf32>) -> tensor<1x1x384xf32>
      %1416 = stablehlo.dot_general %1415, %iterArg_65, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1417 = stablehlo.broadcast_in_dim %iterArg_66, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1418 = stablehlo.add %1416, %1417 : tensor<1x1x384xf32>
      %1419 = stablehlo.add %1357, %1418 : tensor<1x1x384xf32>
      %1420 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1421 = stablehlo.reduce(%1419 init: %1420) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1422 = stablehlo.broadcast_in_dim %1421, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1423 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1424 = stablehlo.broadcast_in_dim %1423, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1425 = stablehlo.divide %1422, %1424 : tensor<1x1x1xf32>
      %1426 = stablehlo.multiply %1419, %1419 : tensor<1x1x384xf32>
      %1427 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1428 = stablehlo.reduce(%1426 init: %1427) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1429 = stablehlo.broadcast_in_dim %1428, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1430 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1431 = stablehlo.broadcast_in_dim %1430, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1432 = stablehlo.divide %1429, %1431 : tensor<1x1x1xf32>
      %1433 = stablehlo.multiply %1425, %1425 : tensor<1x1x1xf32>
      %1434 = stablehlo.subtract %1432, %1433 : tensor<1x1x1xf32>
      %1435 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1436 = stablehlo.broadcast_in_dim %1435, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1437 = stablehlo.add %1434, %1436 : tensor<1x1x1xf32>
      %1438 = stablehlo.rsqrt %1437 : tensor<1x1x1xf32>
      %1439 = stablehlo.broadcast_in_dim %iterArg_67, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1440 = stablehlo.broadcast_in_dim %1438, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1441 = stablehlo.multiply %1440, %1439 : tensor<1x1x384xf32>
      %1442 = stablehlo.broadcast_in_dim %1425, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1443 = stablehlo.subtract %1419, %1442 : tensor<1x1x384xf32>
      %1444 = stablehlo.multiply %1443, %1441 : tensor<1x1x384xf32>
      %1445 = stablehlo.broadcast_in_dim %iterArg_68, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1446 = stablehlo.add %1444, %1445 : tensor<1x1x384xf32>
      %1447 = stablehlo.dot_general %1446, %iterArg_69, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x1536xf32>) -> tensor<1x1x1536xf32>
      %1448 = stablehlo.broadcast_in_dim %iterArg_70, dims = [2] : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
      %1449 = stablehlo.add %1447, %1448 : tensor<1x1x1536xf32>
      %1450 = stablehlo.constant dense<1.41421354> : tensor<f32>
      %1451 = stablehlo.broadcast_in_dim %1450, dims = [] : (tensor<f32>) -> tensor<1x1x1536xf32>
      %1452 = stablehlo.divide %1449, %1451 : tensor<1x1x1536xf32>
      %1453 = chlo.erf %1452 : tensor<1x1x1536xf32> -> tensor<1x1x1536xf32>
      %1454 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1455 = stablehlo.broadcast_in_dim %1454, dims = [] : (tensor<f32>) -> tensor<1x1x1536xf32>
      %1456 = stablehlo.add %1453, %1455 : tensor<1x1x1536xf32>
      %1457 = stablehlo.multiply %1449, %1456 : tensor<1x1x1536xf32>
      %1458 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1459 = stablehlo.broadcast_in_dim %1458, dims = [] : (tensor<f32>) -> tensor<1x1x1536xf32>
      %1460 = stablehlo.divide %1457, %1459 : tensor<1x1x1536xf32>
      %1461 = stablehlo.dot_general %1460, %iterArg_71, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x1536xf32>, tensor<1536x384xf32>) -> tensor<1x1x384xf32>
      %1462 = stablehlo.broadcast_in_dim %iterArg_72, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1463 = stablehlo.add %1461, %1462 : tensor<1x1x384xf32>
      %1464 = stablehlo.add %1419, %1463 : tensor<1x1x384xf32>
      %1465 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1466 = stablehlo.reduce(%1464 init: %1465) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1467 = stablehlo.broadcast_in_dim %1466, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1468 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1469 = stablehlo.broadcast_in_dim %1468, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1470 = stablehlo.divide %1467, %1469 : tensor<1x1x1xf32>
      %1471 = stablehlo.multiply %1464, %1464 : tensor<1x1x384xf32>
      %1472 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1473 = stablehlo.reduce(%1471 init: %1472) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1474 = stablehlo.broadcast_in_dim %1473, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1475 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1476 = stablehlo.broadcast_in_dim %1475, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1477 = stablehlo.divide %1474, %1476 : tensor<1x1x1xf32>
      %1478 = stablehlo.multiply %1470, %1470 : tensor<1x1x1xf32>
      %1479 = stablehlo.subtract %1477, %1478 : tensor<1x1x1xf32>
      %1480 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1481 = stablehlo.broadcast_in_dim %1480, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1482 = stablehlo.add %1479, %1481 : tensor<1x1x1xf32>
      %1483 = stablehlo.rsqrt %1482 : tensor<1x1x1xf32>
      %1484 = stablehlo.broadcast_in_dim %iterArg_73, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1485 = stablehlo.broadcast_in_dim %1483, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1486 = stablehlo.multiply %1485, %1484 : tensor<1x1x384xf32>
      %1487 = stablehlo.broadcast_in_dim %1470, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1488 = stablehlo.subtract %1464, %1487 : tensor<1x1x384xf32>
      %1489 = stablehlo.multiply %1488, %1486 : tensor<1x1x384xf32>
      %1490 = stablehlo.broadcast_in_dim %iterArg_74, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1491 = stablehlo.add %1489, %1490 : tensor<1x1x384xf32>
      %1492 = stablehlo.constant dense<true> : tensor<i1>
      %1493 = stablehlo.broadcast_in_dim %1492, dims = [] : (tensor<i1>) -> tensor<1x448xi1>
      %1494 = stablehlo.iota dim = 0 : tensor<448xi32>
      %1495 = stablehlo.broadcast_in_dim %1494, dims = [1] : (tensor<448xi32>) -> tensor<1x448xi32>
      %1496 = stablehlo.broadcast_in_dim %1495, dims = [0, 1] : (tensor<1x448xi32>) -> tensor<1x448x1xi32>
      %1497 = stablehlo.broadcast_in_dim %1495, dims = [0, 2] : (tensor<1x448xi32>) -> tensor<1x1x448xi32>
      %1498 = stablehlo.broadcast_in_dim %1496, dims = [0, 1, 2] : (tensor<1x448x1xi32>) -> tensor<1x448x448xi32>
      %1499 = stablehlo.broadcast_in_dim %1497, dims = [0, 1, 2] : (tensor<1x1x448xi32>) -> tensor<1x448x448xi32>
      %1500 = stablehlo.compare  GE, %1498, %1499,  SIGNED : (tensor<1x448x448xi32>, tensor<1x448x448xi32>) -> tensor<1x448x448xi1>
      %1501 = stablehlo.broadcast_in_dim %1500, dims = [0, 2, 3] : (tensor<1x448x448xi1>) -> tensor<1x1x448x448xi1>
      %1502 = stablehlo.dot_general %1491, %iterArg_75, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1503 = stablehlo.broadcast_in_dim %iterArg_76, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1504 = stablehlo.add %1502, %1503 : tensor<1x1x384xf32>
      %1505 = stablehlo.dot_general %1491, %iterArg_77, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1506 = stablehlo.dot_general %1491, %iterArg_78, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1507 = stablehlo.broadcast_in_dim %iterArg_79, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1508 = stablehlo.add %1506, %1507 : tensor<1x1x384xf32>
      %1509 = stablehlo.reshape %1504 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %1510 = stablehlo.reshape %1505 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %1511 = stablehlo.reshape %1508 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %1512 = stablehlo.constant dense<0> : tensor<i32>
      %1513 = stablehlo.compare  LT, %iterArg_120, %1512,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1514 = stablehlo.constant dense<448> : tensor<i32>
      %1515 = stablehlo.add %iterArg_120, %1514 : tensor<i32>
      %1516 = stablehlo.select %1513, %1515, %iterArg_120 : tensor<i1>, tensor<i32>
      %1517 = stablehlo.constant dense<0> : tensor<i32>
      %1518 = stablehlo.constant dense<0> : tensor<i32>
      %1519 = stablehlo.constant dense<0> : tensor<i32>
      %1520 = stablehlo.dynamic_slice %1501, %1517, %1518, %1516, %1519, sizes = [1, 1, 1, 448] : (tensor<1x1x448x448xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x448xi1>
      %1521 = stablehlo.broadcast_in_dim %iterArg_108, dims = [0, 3] : (tensor<1x448xi32>) -> tensor<1x1x1x448xi32>
      %1522 = stablehlo.constant dense<0> : tensor<i32>
      %1523 = stablehlo.broadcast_in_dim %1522, dims = [] : (tensor<i32>) -> tensor<1x1x1x448xi32>
      %1524 = stablehlo.compare  NE, %1521, %1523,  SIGNED : (tensor<1x1x1x448xi32>, tensor<1x1x1x448xi32>) -> tensor<1x1x1x448xi1>
      %1525 = stablehlo.and %1524, %1520 : tensor<1x1x1x448xi1>
      %1526 = stablehlo.convert %1525 : (tensor<1x1x1x448xi1>) -> tensor<1x1x1x448xf32>
      %1527 = stablehlo.transpose %1510, dims = [0, 2, 3, 1] : (tensor<1x1x6x64xf32>) -> tensor<1x6x64x1xf32>
      %1528 = stablehlo.transpose %1511, dims = [0, 2, 3, 1] : (tensor<1x1x6x64xf32>) -> tensor<1x6x64x1xf32>
      %1529 = func.call @_one_hot_2(%iterArg_120) : (tensor<i32>) -> tensor<448xf32>
      %1530 = stablehlo.broadcast_in_dim %1529, dims = [3] : (tensor<448xf32>) -> tensor<1x1x1x448xf32>
      %1531 = stablehlo.broadcast_in_dim %1527, dims = [0, 1, 2, 3] : (tensor<1x6x64x1xf32>) -> tensor<1x6x64x448xf32>
      %1532 = stablehlo.broadcast_in_dim %1530, dims = [0, 1, 2, 3] : (tensor<1x1x1x448xf32>) -> tensor<1x6x64x448xf32>
      %1533 = stablehlo.multiply %1531, %1532 : tensor<1x6x64x448xf32>
      %1534 = stablehlo.add %iterArg_121, %1533 : tensor<1x6x64x448xf32>
      %1535 = stablehlo.broadcast_in_dim %1529, dims = [3] : (tensor<448xf32>) -> tensor<1x1x1x448xf32>
      %1536 = stablehlo.broadcast_in_dim %1528, dims = [0, 1, 2, 3] : (tensor<1x6x64x1xf32>) -> tensor<1x6x64x448xf32>
      %1537 = stablehlo.broadcast_in_dim %1535, dims = [0, 1, 2, 3] : (tensor<1x1x1x448xf32>) -> tensor<1x6x64x448xf32>
      %1538 = stablehlo.multiply %1536, %1537 : tensor<1x6x64x448xf32>
      %1539 = stablehlo.add %iterArg_122, %1538 : tensor<1x6x64x448xf32>
      %1540 = stablehlo.constant dense<1> : tensor<i32>
      %1541 = stablehlo.add %iterArg_120, %1540 : tensor<i32>
      %1542 = stablehlo.transpose %1534, dims = [0, 3, 1, 2] : (tensor<1x6x64x448xf32>) -> tensor<1x448x6x64xf32>
      %1543 = stablehlo.transpose %1539, dims = [0, 3, 1, 2] : (tensor<1x6x64x448xf32>) -> tensor<1x448x6x64xf32>
      %1544 = stablehlo.iota dim = 0 : tensor<448xi32>
      %1545 = stablehlo.constant dense<1> : tensor<i32>
      %1546 = stablehlo.add %iterArg_120, %1545 : tensor<i32>
      %1547 = stablehlo.broadcast_in_dim %1546, dims = [] : (tensor<i32>) -> tensor<448xi32>
      %1548 = stablehlo.compare  LT, %1544, %1547,  SIGNED : (tensor<448xi32>, tensor<448xi32>) -> tensor<448xi1>
      %1549 = stablehlo.broadcast_in_dim %1548, dims = [3] : (tensor<448xi1>) -> tensor<1x1x1x448xi1>
      %1550 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1551 = stablehlo.broadcast_in_dim %1550, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %1552 = stablehlo.compare  NE, %1526, %1551,  FLOAT : (tensor<1x1x1x448xf32>, tensor<1x1x1x448xf32>) -> tensor<1x1x1x448xi1>
      %1553 = stablehlo.and %1549, %1552 : tensor<1x1x1x448xi1>
      %1554 = stablehlo.convert %1553 : (tensor<1x1x1x448xi1>) -> tensor<1x1x1x448xf32>
      %1555 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1556 = stablehlo.broadcast_in_dim %1555, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %1557 = stablehlo.compare  GT, %1554, %1556,  FLOAT : (tensor<1x1x1x448xf32>, tensor<1x1x1x448xf32>) -> tensor<1x1x1x448xi1>
      %1558 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1559 = stablehlo.broadcast_in_dim %1558, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %1560 = stablehlo.convert %1559 : tensor<1x1x1x448xf32>
      %1561 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
      %1562 = stablehlo.broadcast_in_dim %1561, dims = [] : (tensor<f32>) -> tensor<1x1x1x448xf32>
      %1563 = stablehlo.select %1557, %1560, %1562 : tensor<1x1x1x448xi1>, tensor<1x1x1x448xf32>
      %1564 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %1565 = stablehlo.sqrt %1564 : tensor<f32>
      %1566 = stablehlo.convert %1565 : tensor<f32>
      %1567 = stablehlo.broadcast_in_dim %1566, dims = [] : (tensor<f32>) -> tensor<1x1x6x64xf32>
      %1568 = stablehlo.divide %1509, %1567 : tensor<1x1x6x64xf32>
      %1569 = stablehlo.dot_general %1568, %1542, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x6x64xf32>, tensor<1x448x6x64xf32>) -> tensor<1x6x1x448xf32>
      %1570 = stablehlo.broadcast_in_dim %1563, dims = [0, 1, 2, 3] : (tensor<1x1x1x448xf32>) -> tensor<1x6x1x448xf32>
      %1571 = stablehlo.add %1569, %1570 : tensor<1x6x1x448xf32>
      %1572 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %1573 = stablehlo.reduce(%1571 init: %1572) across dimensions = [3] : (tensor<1x6x1x448xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1574 = stablehlo.broadcast_in_dim %1573, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %1575 = stablehlo.broadcast_in_dim %1574, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x448xf32>
      %1576 = stablehlo.subtract %1571, %1575 : tensor<1x6x1x448xf32>
      %1577 = stablehlo.exponential %1576 : tensor<1x6x1x448xf32>
      %1578 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1579 = stablehlo.reduce(%1577 init: %1578) across dimensions = [3] : (tensor<1x6x1x448xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1580 = stablehlo.broadcast_in_dim %1579, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %1581 = stablehlo.broadcast_in_dim %1580, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x448xf32>
      %1582 = stablehlo.divide %1577, %1581 : tensor<1x6x1x448xf32>
      %1583 = stablehlo.dot_general %1543, %1582, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x448x6x64xf32>, tensor<1x6x1x448xf32>) -> tensor<1x6x64x1xf32>
      %1584 = stablehlo.transpose %1583, dims = [0, 3, 1, 2] : (tensor<1x6x64x1xf32>) -> tensor<1x1x6x64xf32>
      %1585 = stablehlo.reshape %1584 : (tensor<1x1x6x64xf32>) -> tensor<1x1x384xf32>
      %1586 = stablehlo.dot_general %1585, %iterArg_80, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1587 = stablehlo.broadcast_in_dim %iterArg_81, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1588 = stablehlo.add %1586, %1587 : tensor<1x1x384xf32>
      %1589 = stablehlo.add %1464, %1588 : tensor<1x1x384xf32>
      %1590 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1591 = stablehlo.reduce(%1589 init: %1590) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1592 = stablehlo.broadcast_in_dim %1591, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1593 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1594 = stablehlo.broadcast_in_dim %1593, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1595 = stablehlo.divide %1592, %1594 : tensor<1x1x1xf32>
      %1596 = stablehlo.multiply %1589, %1589 : tensor<1x1x384xf32>
      %1597 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1598 = stablehlo.reduce(%1596 init: %1597) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1599 = stablehlo.broadcast_in_dim %1598, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1600 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1601 = stablehlo.broadcast_in_dim %1600, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1602 = stablehlo.divide %1599, %1601 : tensor<1x1x1xf32>
      %1603 = stablehlo.multiply %1595, %1595 : tensor<1x1x1xf32>
      %1604 = stablehlo.subtract %1602, %1603 : tensor<1x1x1xf32>
      %1605 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1606 = stablehlo.broadcast_in_dim %1605, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1607 = stablehlo.add %1604, %1606 : tensor<1x1x1xf32>
      %1608 = stablehlo.rsqrt %1607 : tensor<1x1x1xf32>
      %1609 = stablehlo.broadcast_in_dim %iterArg_82, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1610 = stablehlo.broadcast_in_dim %1608, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1611 = stablehlo.multiply %1610, %1609 : tensor<1x1x384xf32>
      %1612 = stablehlo.broadcast_in_dim %1595, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1613 = stablehlo.subtract %1589, %1612 : tensor<1x1x384xf32>
      %1614 = stablehlo.multiply %1613, %1611 : tensor<1x1x384xf32>
      %1615 = stablehlo.broadcast_in_dim %iterArg_83, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1616 = stablehlo.add %1614, %1615 : tensor<1x1x384xf32>
      %1617 = stablehlo.dot_general %1616, %iterArg_84, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1618 = stablehlo.broadcast_in_dim %iterArg_85, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1619 = stablehlo.add %1617, %1618 : tensor<1x1x384xf32>
      %1620 = stablehlo.dot_general %iterArg_110, %iterArg_86, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
      %1621 = stablehlo.dot_general %iterArg_110, %iterArg_87, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x384xf32>, tensor<384x384xf32>) -> tensor<1x1500x384xf32>
      %1622 = stablehlo.broadcast_in_dim %iterArg_88, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1623 = stablehlo.broadcast_in_dim %1622, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x1500x384xf32>
      %1624 = stablehlo.add %1621, %1623 : tensor<1x1500x384xf32>
      %1625 = stablehlo.reshape %1619 : (tensor<1x1x384xf32>) -> tensor<1x1x6x64xf32>
      %1626 = stablehlo.reshape %1620 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
      %1627 = stablehlo.reshape %1624 : (tensor<1x1500x384xf32>) -> tensor<1x1500x6x64xf32>
      %1628 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
      %1629 = stablehlo.sqrt %1628 : tensor<f32>
      %1630 = stablehlo.convert %1629 : tensor<f32>
      %1631 = stablehlo.broadcast_in_dim %1630, dims = [] : (tensor<f32>) -> tensor<1x1x6x64xf32>
      %1632 = stablehlo.divide %1625, %1631 : tensor<1x1x6x64xf32>
      %1633 = stablehlo.dot_general %1632, %1626, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x6x64xf32>, tensor<1x1500x6x64xf32>) -> tensor<1x6x1x1500xf32>
      %1634 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %1635 = stablehlo.reduce(%1633 init: %1634) across dimensions = [3] : (tensor<1x6x1x1500xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1636 = stablehlo.broadcast_in_dim %1635, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %1637 = stablehlo.broadcast_in_dim %1636, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x1500xf32>
      %1638 = stablehlo.subtract %1633, %1637 : tensor<1x6x1x1500xf32>
      %1639 = stablehlo.exponential %1638 : tensor<1x6x1x1500xf32>
      %1640 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1641 = stablehlo.reduce(%1639 init: %1640) across dimensions = [3] : (tensor<1x6x1x1500xf32>, tensor<f32>) -> tensor<1x6x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1642 = stablehlo.broadcast_in_dim %1641, dims = [0, 1, 2] : (tensor<1x6x1xf32>) -> tensor<1x6x1x1xf32>
      %1643 = stablehlo.broadcast_in_dim %1642, dims = [0, 1, 2, 3] : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x1500xf32>
      %1644 = stablehlo.divide %1639, %1643 : tensor<1x6x1x1500xf32>
      %1645 = stablehlo.dot_general %1627, %1644, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1500x6x64xf32>, tensor<1x6x1x1500xf32>) -> tensor<1x6x64x1xf32>
      %1646 = stablehlo.transpose %1645, dims = [0, 3, 1, 2] : (tensor<1x6x64x1xf32>) -> tensor<1x1x6x64xf32>
      %1647 = stablehlo.reshape %1646 : (tensor<1x1x6x64xf32>) -> tensor<1x1x384xf32>
      %1648 = stablehlo.dot_general %1647, %iterArg_89, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x384xf32>) -> tensor<1x1x384xf32>
      %1649 = stablehlo.broadcast_in_dim %iterArg_90, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1650 = stablehlo.add %1648, %1649 : tensor<1x1x384xf32>
      %1651 = stablehlo.add %1589, %1650 : tensor<1x1x384xf32>
      %1652 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1653 = stablehlo.reduce(%1651 init: %1652) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1654 = stablehlo.broadcast_in_dim %1653, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1655 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1656 = stablehlo.broadcast_in_dim %1655, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1657 = stablehlo.divide %1654, %1656 : tensor<1x1x1xf32>
      %1658 = stablehlo.multiply %1651, %1651 : tensor<1x1x384xf32>
      %1659 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1660 = stablehlo.reduce(%1658 init: %1659) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1661 = stablehlo.broadcast_in_dim %1660, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1662 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1663 = stablehlo.broadcast_in_dim %1662, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1664 = stablehlo.divide %1661, %1663 : tensor<1x1x1xf32>
      %1665 = stablehlo.multiply %1657, %1657 : tensor<1x1x1xf32>
      %1666 = stablehlo.subtract %1664, %1665 : tensor<1x1x1xf32>
      %1667 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1668 = stablehlo.broadcast_in_dim %1667, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1669 = stablehlo.add %1666, %1668 : tensor<1x1x1xf32>
      %1670 = stablehlo.rsqrt %1669 : tensor<1x1x1xf32>
      %1671 = stablehlo.broadcast_in_dim %iterArg_91, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1672 = stablehlo.broadcast_in_dim %1670, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1673 = stablehlo.multiply %1672, %1671 : tensor<1x1x384xf32>
      %1674 = stablehlo.broadcast_in_dim %1657, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1675 = stablehlo.subtract %1651, %1674 : tensor<1x1x384xf32>
      %1676 = stablehlo.multiply %1675, %1673 : tensor<1x1x384xf32>
      %1677 = stablehlo.broadcast_in_dim %iterArg_92, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1678 = stablehlo.add %1676, %1677 : tensor<1x1x384xf32>
      %1679 = stablehlo.dot_general %1678, %iterArg_93, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x1536xf32>) -> tensor<1x1x1536xf32>
      %1680 = stablehlo.broadcast_in_dim %iterArg_94, dims = [2] : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
      %1681 = stablehlo.add %1679, %1680 : tensor<1x1x1536xf32>
      %1682 = stablehlo.constant dense<1.41421354> : tensor<f32>
      %1683 = stablehlo.broadcast_in_dim %1682, dims = [] : (tensor<f32>) -> tensor<1x1x1536xf32>
      %1684 = stablehlo.divide %1681, %1683 : tensor<1x1x1536xf32>
      %1685 = chlo.erf %1684 : tensor<1x1x1536xf32> -> tensor<1x1x1536xf32>
      %1686 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1687 = stablehlo.broadcast_in_dim %1686, dims = [] : (tensor<f32>) -> tensor<1x1x1536xf32>
      %1688 = stablehlo.add %1685, %1687 : tensor<1x1x1536xf32>
      %1689 = stablehlo.multiply %1681, %1688 : tensor<1x1x1536xf32>
      %1690 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1691 = stablehlo.broadcast_in_dim %1690, dims = [] : (tensor<f32>) -> tensor<1x1x1536xf32>
      %1692 = stablehlo.divide %1689, %1691 : tensor<1x1x1536xf32>
      %1693 = stablehlo.dot_general %1692, %iterArg_95, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x1536xf32>, tensor<1536x384xf32>) -> tensor<1x1x384xf32>
      %1694 = stablehlo.broadcast_in_dim %iterArg_96, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1695 = stablehlo.add %1693, %1694 : tensor<1x1x384xf32>
      %1696 = stablehlo.add %1651, %1695 : tensor<1x1x384xf32>
      %1697 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1698 = stablehlo.reduce(%1696 init: %1697) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1699 = stablehlo.broadcast_in_dim %1698, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1700 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1701 = stablehlo.broadcast_in_dim %1700, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1702 = stablehlo.divide %1699, %1701 : tensor<1x1x1xf32>
      %1703 = stablehlo.multiply %1696, %1696 : tensor<1x1x384xf32>
      %1704 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1705 = stablehlo.reduce(%1703 init: %1704) across dimensions = [2] : (tensor<1x1x384xf32>, tensor<f32>) -> tensor<1x1xf32>
       reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
        %1825 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1825 : tensor<f32>
      }
      %1706 = stablehlo.broadcast_in_dim %1705, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %1707 = stablehlo.constant dense<3.840000e+02> : tensor<f32>
      %1708 = stablehlo.broadcast_in_dim %1707, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1709 = stablehlo.divide %1706, %1708 : tensor<1x1x1xf32>
      %1710 = stablehlo.multiply %1702, %1702 : tensor<1x1x1xf32>
      %1711 = stablehlo.subtract %1709, %1710 : tensor<1x1x1xf32>
      %1712 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1713 = stablehlo.broadcast_in_dim %1712, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
      %1714 = stablehlo.add %1711, %1713 : tensor<1x1x1xf32>
      %1715 = stablehlo.rsqrt %1714 : tensor<1x1x1xf32>
      %1716 = stablehlo.broadcast_in_dim %iterArg_97, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1717 = stablehlo.broadcast_in_dim %1715, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1718 = stablehlo.multiply %1717, %1716 : tensor<1x1x384xf32>
      %1719 = stablehlo.broadcast_in_dim %1702, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x384xf32>
      %1720 = stablehlo.subtract %1696, %1719 : tensor<1x1x384xf32>
      %1721 = stablehlo.multiply %1720, %1718 : tensor<1x1x384xf32>
      %1722 = stablehlo.broadcast_in_dim %iterArg_98, dims = [2] : (tensor<384xf32>) -> tensor<1x1x384xf32>
      %1723 = stablehlo.add %1721, %1722 : tensor<1x1x384xf32>
      %1724 = stablehlo.transpose %iterArg, dims = [1, 0] : (tensor<51865x384xf32>) -> tensor<384x51865xf32>
      %1725 = stablehlo.dot_general %1723, %1724, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x384xf32>, tensor<384x51865xf32>) -> tensor<1x1x51865xf32>
      %1726 = stablehlo.constant dense<0> : tensor<i32>
      %1727 = stablehlo.constant dense<0> : tensor<i32>
      %1728 = stablehlo.compare  LT, %1726, %1727,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1729 = stablehlo.constant dense<0> : tensor<i32>
      %1730 = stablehlo.constant dense<1> : tensor<i32>
      %1731 = stablehlo.add %1729, %1730 : tensor<i32>
      %1732 = stablehlo.constant dense<0> : tensor<i32>
      %1733 = stablehlo.select %1728, %1731, %1732 : tensor<i1>, tensor<i32>
      %1734 = stablehlo.constant dense<-1> : tensor<i32>
      %1735 = stablehlo.constant dense<0> : tensor<i32>
      %1736 = stablehlo.compare  LT, %1734, %1735,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1737 = stablehlo.constant dense<-1> : tensor<i32>
      %1738 = stablehlo.constant dense<1> : tensor<i32>
      %1739 = stablehlo.add %1737, %1738 : tensor<i32>
      %1740 = stablehlo.constant dense<-1> : tensor<i32>
      %1741 = stablehlo.select %1736, %1739, %1740 : tensor<i1>, tensor<i32>
      %1742 = stablehlo.constant dense<0> : tensor<i32>
      %1743 = stablehlo.constant dense<0> : tensor<i32>
      %1744 = stablehlo.compare  LT, %1742, %1743,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1745 = stablehlo.constant dense<0> : tensor<i32>
      %1746 = stablehlo.constant dense<51865> : tensor<i32>
      %1747 = stablehlo.add %1745, %1746 : tensor<i32>
      %1748 = stablehlo.constant dense<0> : tensor<i32>
      %1749 = stablehlo.select %1744, %1747, %1748 : tensor<i1>, tensor<i32>
      %1750 = stablehlo.dynamic_slice %1725, %1733, %1741, %1749, sizes = [1, 1, 51865] : (tensor<1x1x51865xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x51865xf32>
      %1751 = stablehlo.reshape %1750 : (tensor<1x1x51865xf32>) -> tensor<1x51865xf32>
      %1752 = stablehlo.constant dense<0> : tensor<i32>
      %1753 = stablehlo.subtract %iterArg_104, %1752 : tensor<i32>
      %1754 = stablehlo.constant dense<0> : tensor<i32>
      %1755 = stablehlo.constant dense<1> : tensor<i32>
      %1756 = func.call @clip(%1753, %1754, %1755) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %1757 = stablehlo.constant dense<1> : tensor<i32>
      %1758 = stablehlo.subtract %1757, %1756 : tensor<i32>
      %1759 = stablehlo.constant dense<50257> : tensor<i32>
      %1760 = stablehlo.broadcast_in_dim %1759, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1761 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %1762 = stablehlo.broadcast_in_dim %1761, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %1763 = "stablehlo.scatter"(%1751, %1760, %1762) ({
      ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
        stablehlo.return %arg2 : tensor<f32>
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x51865xf32>, tensor<1xi32>, tensor<1xf32>) -> tensor<1x51865xf32>
      %1764 = func.call @_where(%1758, %1763, %1751) : (tensor<i32>, tensor<1x51865xf32>, tensor<1x51865xf32>) -> tensor<1x51865xf32>
      %1765 = stablehlo.constant dense<0> : tensor<i32>
      %1766 = stablehlo.broadcast_in_dim %1765, dims = [] : (tensor<i32>) -> tensor<88xi32>
      %1767 = stablehlo.compare  LT, %iterArg_99, %1766,  SIGNED : (tensor<88xi32>, tensor<88xi32>) -> tensor<88xi1>
      %1768 = stablehlo.constant dense<51865> : tensor<i32>
      %1769 = stablehlo.broadcast_in_dim %1768, dims = [] : (tensor<i32>) -> tensor<88xi32>
      %1770 = stablehlo.add %iterArg_99, %1769 : tensor<88xi32>
      %1771 = stablehlo.select %1767, %1770, %iterArg_99 : tensor<88xi1>, tensor<88xi32>
      %1772 = stablehlo.broadcast_in_dim %1771, dims = [0] : (tensor<88xi32>) -> tensor<88x1xi32>
      %1773 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %1774 = stablehlo.broadcast_in_dim %1773, dims = [] : (tensor<f32>) -> tensor<1x88xf32>
      %1775 = "stablehlo.scatter"(%1764, %1772, %1774) ({
      ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
        stablehlo.return %arg2 : tensor<f32>
      }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<1x51865xf32>, tensor<88x1xi32>, tensor<1x88xf32>) -> tensor<1x51865xf32>
      %1776 = stablehlo.constant dense<4> : tensor<i32>
      %1777 = stablehlo.subtract %iterArg_104, %1776 : tensor<i32>
      %1778 = stablehlo.constant dense<0> : tensor<i32>
      %1779 = stablehlo.broadcast_in_dim %1778, dims = [] : (tensor<i32>) -> tensor<i32>
      %1780 = stablehlo.compare  NE, %1777, %1779,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1781 = stablehlo.convert %1780 : (tensor<i1>) -> tensor<i32>
      %1782 = stablehlo.constant dense<1> : tensor<i32>
      %1783 = stablehlo.subtract %1782, %1781 : tensor<i32>
      %1784 = stablehlo.constant dense<0> : tensor<i32>
      %1785 = stablehlo.broadcast_in_dim %1784, dims = [] : (tensor<i32>) -> tensor<2xi32>
      %1786 = stablehlo.compare  LT, %iterArg_100, %1785,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
      %1787 = stablehlo.constant dense<51865> : tensor<i32>
      %1788 = stablehlo.broadcast_in_dim %1787, dims = [] : (tensor<i32>) -> tensor<2xi32>
      %1789 = stablehlo.add %iterArg_100, %1788 : tensor<2xi32>
      %1790 = stablehlo.select %1786, %1789, %iterArg_100 : tensor<2xi1>, tensor<2xi32>
      %1791 = stablehlo.broadcast_in_dim %1790, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
      %1792 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %1793 = stablehlo.broadcast_in_dim %1792, dims = [] : (tensor<f32>) -> tensor<1x2xf32>
      %1794 = "stablehlo.scatter"(%1775, %1791, %1793) ({
      ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
        stablehlo.return %arg2 : tensor<f32>
      }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<1x51865xf32>, tensor<2x1xi32>, tensor<1x2xf32>) -> tensor<1x51865xf32>
      %1795 = func.call @_where_3(%1783, %1794, %1775) : (tensor<i32>, tensor<1x51865xf32>, tensor<1x51865xf32>) -> tensor<1x51865xf32>
      %1796 = stablehlo.constant dense<4> : tensor<i32>
      %1797 = stablehlo.compare  GE, %iterArg_104, %1796,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1798 = stablehlo.convert %1797 : (tensor<i1>) -> tensor<i32>
      %1799 = "stablehlo.case"(%1798) ({
        %1825 = stablehlo.constant dense<0> : tensor<i32>
        %1826 = stablehlo.compare  LT, %iterArg_104, %1825,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %1827 = stablehlo.convert %iterArg_104 : tensor<i32>
        %1828 = stablehlo.constant dense<4> : tensor<i32>
        %1829 = stablehlo.add %1827, %1828 : tensor<i32>
        %1830 = stablehlo.select %1826, %1829, %iterArg_104 : tensor<i1>, tensor<i32>
        %1831 = stablehlo.dynamic_slice %iterArg_101, %1830, sizes = [1] : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
        %1832 = stablehlo.reshape %1831 : (tensor<1xi32>) -> tensor<i32>
        %1833 = stablehlo.constant dense<0> : tensor<i32>
        %1834 = stablehlo.compare  GE, %1832, %1833,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %1835 = stablehlo.convert %1834 : (tensor<i1>) -> tensor<i32>
        %1836 = "stablehlo.case"(%1835) ({
          stablehlo.return %1795 : tensor<1x51865xf32>
        }, {
          %1837 = stablehlo.constant dense<0> : tensor<i32>
          %1838 = stablehlo.compare  LT, %iterArg_104, %1837,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
          %1839 = stablehlo.convert %iterArg_104 : tensor<i32>
          %1840 = stablehlo.constant dense<4> : tensor<i32>
          %1841 = stablehlo.add %1839, %1840 : tensor<i32>
          %1842 = stablehlo.select %1838, %1841, %iterArg_104 : tensor<i1>, tensor<i32>
          %1843 = stablehlo.dynamic_slice %iterArg_101, %1842, sizes = [1] : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
          %1844 = stablehlo.reshape %1843 : (tensor<1xi32>) -> tensor<i32>
          %1845 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
          %1846 = stablehlo.broadcast_in_dim %1845, dims = [] : (tensor<f32>) -> tensor<1x51865xf32>
          %1847 = stablehlo.constant dense<0xFF800000> : tensor<f32>
          %1848 = stablehlo.broadcast_in_dim %1847, dims = [] : (tensor<f32>) -> tensor<1x51865xf32>
          %1849 = stablehlo.multiply %1846, %1848 : tensor<1x51865xf32>
          %1850 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %1851 = stablehlo.broadcast_in_dim %1850, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
          %1852 = stablehlo.constant dense<0> : tensor<i32>
          %1853 = stablehlo.compare  LT, %1844, %1852,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
          %1854 = stablehlo.constant dense<51865> : tensor<i32>
          %1855 = stablehlo.add %1844, %1854 : tensor<i32>
          %1856 = stablehlo.select %1853, %1855, %1844 : tensor<i1>, tensor<i32>
          %1857 = stablehlo.constant dense<0> : tensor<i32>
          %1858 = stablehlo.dynamic_update_slice %1849, %1851, %1857, %1856 : (tensor<1x51865xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x51865xf32>
          stablehlo.return %1858 : tensor<1x51865xf32>
        }) : (tensor<i32>) -> tensor<1x51865xf32>
        stablehlo.return %1836 : tensor<1x51865xf32>
      }, {
        stablehlo.return %1795 : tensor<1x51865xf32>
      }) : (tensor<i32>) -> tensor<1x51865xf32>
      %1800 = func.call @argmax(%1799) : (tensor<1x51865xf32>) -> tensor<1xi32>
      %1801 = stablehlo.not %iterArg_107 : tensor<1xi1>
      %1802 = stablehlo.convert %1801 : (tensor<1xi1>) -> tensor<1xi32>
      %1803 = stablehlo.multiply %1800, %1802 : tensor<1xi32>
      %1804 = stablehlo.convert %iterArg_107 : (tensor<1xi1>) -> tensor<1xi32>
      %1805 = stablehlo.broadcast_in_dim %iterArg_102, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1806 = stablehlo.multiply %1805, %1804 : tensor<1xi32>
      %1807 = stablehlo.add %1803, %1806 : tensor<1xi32>
      %1808 = stablehlo.broadcast_in_dim %iterArg_103, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1809 = stablehlo.compare  EQ, %1807, %1808,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      %1810 = stablehlo.or %iterArg_107, %1809 : tensor<1xi1>
      %1811 = stablehlo.broadcast_in_dim %1807, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
      %1812 = stablehlo.constant dense<0> : tensor<i32>
      %1813 = stablehlo.compare  LT, %iterArg_104, %1812,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1814 = stablehlo.convert %iterArg_104 : tensor<i32>
      %1815 = stablehlo.constant dense<448> : tensor<i32>
      %1816 = stablehlo.add %1814, %1815 : tensor<i32>
      %1817 = stablehlo.select %1813, %1816, %iterArg_104 : tensor<i1>, tensor<i32>
      %1818 = stablehlo.constant dense<0> : tensor<i32>
      %1819 = stablehlo.dynamic_update_slice %iterArg_105, %1811, %1818, %1817 : (tensor<1x448xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x448xi32>
      %1820 = stablehlo.constant dense<1> : tensor<i32>
      %1821 = stablehlo.broadcast_in_dim %1820, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
      %1822 = stablehlo.add %iterArg_109, %1821 : tensor<1x1xi32>
      %1823 = stablehlo.constant dense<1> : tensor<i32>
      %1824 = stablehlo.add %iterArg_104, %1823 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_0, %iterArg_1, %iterArg_2, %iterArg_3, %iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %iterArg_13, %iterArg_14, %iterArg_15, %iterArg_16, %iterArg_17, %iterArg_18, %iterArg_19, %iterArg_20, %iterArg_21, %iterArg_22, %iterArg_23, %iterArg_24, %iterArg_25, %iterArg_26, %iterArg_27, %iterArg_28, %iterArg_29, %iterArg_30, %iterArg_31, %iterArg_32, %iterArg_33, %iterArg_34, %iterArg_35, %iterArg_36, %iterArg_37, %iterArg_38, %iterArg_39, %iterArg_40, %iterArg_41, %iterArg_42, %iterArg_43, %iterArg_44, %iterArg_45, %iterArg_46, %iterArg_47, %iterArg_48, %iterArg_49, %iterArg_50, %iterArg_51, %iterArg_52, %iterArg_53, %iterArg_54, %iterArg_55, %iterArg_56, %iterArg_57, %iterArg_58, %iterArg_59, %iterArg_60, %iterArg_61, %iterArg_62, %iterArg_63, %iterArg_64, %iterArg_65, %iterArg_66, %iterArg_67, %iterArg_68, %iterArg_69, %iterArg_70, %iterArg_71, %iterArg_72, %iterArg_73, %iterArg_74, %iterArg_75, %iterArg_76, %iterArg_77, %iterArg_78, %iterArg_79, %iterArg_80, %iterArg_81, %iterArg_82, %iterArg_83, %iterArg_84, %iterArg_85, %iterArg_86, %iterArg_87, %iterArg_88, %iterArg_89, %iterArg_90, %iterArg_91, %iterArg_92, %iterArg_93, %iterArg_94, %iterArg_95, %iterArg_96, %iterArg_97, %iterArg_98, %iterArg_99, %iterArg_100, %iterArg_101, %iterArg_102, %iterArg_103, %1824, %1819, %1811, %1810, %iterArg_108, %1822, %iterArg_110, %845, %838, %843, %1077, %1070, %1075, %1309, %1302, %1307, %1541, %1534, %1539 : tensor<51865x384xf32>, tensor<448x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1536xf32>, tensor<1536xf32>, tensor<1536x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1536xf32>, tensor<1536xf32>, tensor<1536x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1536xf32>, tensor<1536xf32>, tensor<1536x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1536xf32>, tensor<1536xf32>, tensor<1536x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<88xi32>, tensor<2xi32>, tensor<4xi32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x448xi32>, tensor<1x1xi32>, tensor<1xi1>, tensor<1x448xi32>, tensor<1x1xi32>, tensor<1x1500x384xf32>, tensor<i32>, tensor<1x6x64x448xf32>, tensor<1x6x64x448xf32>, tensor<i32>, tensor<1x6x64x448xf32>, tensor<1x6x64x448xf32>, tensor<i32>, tensor<1x6x64x448xf32>, tensor<1x6x64x448xf32>, tensor<i32>, tensor<1x6x64x448xf32>, tensor<1x6x64x448xf32>
    }
    return %753#106 : tensor<1x448xi32>
  }
  func.func private @_one_hot(%arg0: tensor<i32>) -> tensor<448xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.iota dim = 0 : tensor<448xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xi32>) -> tensor<448xi32>
    %3 = stablehlo.compare  EQ, %2, %1,  SIGNED : (tensor<448xi32>, tensor<448xi32>) -> tensor<448xi1>
    %4 = stablehlo.convert %3 : (tensor<448xi1>) -> tensor<448xf32>
    return %4 : tensor<448xf32>
  }
  func.func private @_one_hot_0(%arg0: tensor<i32>) -> tensor<448xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.iota dim = 0 : tensor<448xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xi32>) -> tensor<448xi32>
    %3 = stablehlo.compare  EQ, %2, %1,  SIGNED : (tensor<448xi32>, tensor<448xi32>) -> tensor<448xi1>
    %4 = stablehlo.convert %3 : (tensor<448xi1>) -> tensor<448xf32>
    return %4 : tensor<448xf32>
  }
  func.func private @_one_hot_1(%arg0: tensor<i32>) -> tensor<448xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.iota dim = 0 : tensor<448xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xi32>) -> tensor<448xi32>
    %3 = stablehlo.compare  EQ, %2, %1,  SIGNED : (tensor<448xi32>, tensor<448xi32>) -> tensor<448xi1>
    %4 = stablehlo.convert %3 : (tensor<448xi1>) -> tensor<448xf32>
    return %4 : tensor<448xf32>
  }
  func.func private @_one_hot_2(%arg0: tensor<i32>) -> tensor<448xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.iota dim = 0 : tensor<448xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xi32>) -> tensor<448xi32>
    %3 = stablehlo.compare  EQ, %2, %1,  SIGNED : (tensor<448xi32>, tensor<448xi32>) -> tensor<448xi1>
    %4 = stablehlo.convert %3 : (tensor<448xi1>) -> tensor<448xf32>
    return %4 : tensor<448xf32>
  }
  func.func private @clip(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<i32>
    %1 = stablehlo.minimum %arg2, %0 : tensor<i32>
    return %1 : tensor<i32>
  }
  func.func private @_where(%arg0: tensor<i32>, %arg1: tensor<1x51865xf32>, %arg2: tensor<1x51865xf32>) -> tensor<1x51865xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.compare  NE, %arg0, %0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i1>) -> tensor<1x51865xi1>
    %3 = stablehlo.select %2, %arg1, %arg2 : tensor<1x51865xi1>, tensor<1x51865xf32>
    return %3 : tensor<1x51865xf32>
  }
  func.func private @_where_3(%arg0: tensor<i32>, %arg1: tensor<1x51865xf32>, %arg2: tensor<1x51865xf32>) -> tensor<1x51865xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.compare  NE, %arg0, %0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i1>) -> tensor<1x51865xi1>
    %3 = stablehlo.select %2, %arg1, %arg2 : tensor<1x51865xi1>, tensor<1x51865xf32>
    return %3 : tensor<1x51865xf32>
  }
  func.func private @argmax(%arg0: tensor<1x51865xf32>) -> tensor<1xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<1x51865xi32>
    %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [1] : (tensor<1x51865xf32>, tensor<1x51865xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
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
}