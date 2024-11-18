module @resnet50 attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<16x3x224x224xf16> {mhlo.layout_mode = "default"}) -> (tensor<16x1000xf16> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.constant dense_resource<__elided__> : tensor<7x7x3x64xf32>
    %1 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %2 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %3 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %4 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %5 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x64x256xf32>
    %6 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %7 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %8 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %9 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %10 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x64x64xf32>
    %11 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %12 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %13 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %14 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %15 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x64x64xf32>
    %16 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %17 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %18 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %19 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %20 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x64x256xf32>
    %21 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %22 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %23 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %24 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %25 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x64xf32>
    %26 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %27 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %28 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %29 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %30 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x64x64xf32>
    %31 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %32 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %33 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %34 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %35 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x64x256xf32>
    %36 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %37 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %38 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %39 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %40 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x64xf32>
    %41 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %42 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %43 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %44 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %45 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x64x64xf32>
    %46 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %47 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %48 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %49 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %50 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x64x256xf32>
    %51 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %52 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %53 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %54 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %55 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x512xf32>
    %56 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %57 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %58 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %59 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %60 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x128xf32>
    %61 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %62 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %63 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %64 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %65 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x128x128xf32>
    %66 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %67 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %68 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %69 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %70 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x128x512xf32>
    %71 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %72 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %73 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %74 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %75 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x128xf32>
    %76 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %77 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %78 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %79 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %80 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x128x128xf32>
    %81 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %82 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %83 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %84 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %85 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x128x512xf32>
    %86 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %87 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %88 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %89 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %90 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x128xf32>
    %91 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %92 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %93 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %94 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %95 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x128x128xf32>
    %96 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %97 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %98 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %99 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %100 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x128x512xf32>
    %101 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %102 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %103 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %104 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %105 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x128xf32>
    %106 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %107 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %108 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %109 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %110 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x128x128xf32>
    %111 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %112 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %113 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %114 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %115 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x128x512xf32>
    %116 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %117 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %118 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %119 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %120 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x1024xf32>
    %121 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %122 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %123 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %124 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %125 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x256xf32>
    %126 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %127 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %128 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %129 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %130 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %131 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %132 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %133 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %134 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %135 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x1024xf32>
    %136 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %137 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %138 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %139 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %140 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x256xf32>
    %141 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %142 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %143 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %144 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %145 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %146 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %147 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %148 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %149 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %150 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x1024xf32>
    %151 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %152 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %153 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %154 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %155 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x256xf32>
    %156 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %157 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %158 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %159 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %160 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %161 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %162 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %163 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %164 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %165 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x1024xf32>
    %166 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %167 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %168 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %169 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %170 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x256xf32>
    %171 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %172 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %173 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %174 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %175 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %176 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %177 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %178 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %179 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %180 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x1024xf32>
    %181 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %182 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %183 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %184 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %185 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x256xf32>
    %186 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %187 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %188 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %189 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %190 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %191 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %192 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %193 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %194 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %195 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x1024xf32>
    %196 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %197 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %198 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %199 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %200 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x256xf32>
    %201 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %202 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %203 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %204 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %205 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %206 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %207 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %208 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %209 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %210 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x1024xf32>
    %211 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %212 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %213 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %214 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %215 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x2048xf32>
    %216 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %217 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %218 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %219 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %220 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x512xf32>
    %221 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %222 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %223 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %224 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %225 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x512x512xf32>
    %226 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %227 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %228 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %229 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %230 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x2048xf32>
    %231 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %232 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %233 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %234 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %235 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048x512xf32>
    %236 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %237 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %238 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %239 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %240 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x512x512xf32>
    %241 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %242 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %243 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %244 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %245 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x2048xf32>
    %246 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %247 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %248 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %249 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %250 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048x512xf32>
    %251 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %252 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %253 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %254 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %255 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x512x512xf32>
    %256 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %257 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %258 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %259 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %260 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x2048xf32>
    %261 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %262 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %263 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %264 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %265 = stablehlo.constant dense_resource<__elided__> : tensor<2048x1000xf32>
    %266 = stablehlo.constant dense_resource<__elided__> : tensor<1000xf32>
    %267 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] : (tensor<16x3x224x224xf16>) -> tensor<16x224x224x3xf16>
    %268 = stablehlo.convert %267 : (tensor<16x224x224x3xf16>) -> tensor<16x224x224x3xf32>
    %269 = stablehlo.convert %268 : (tensor<16x224x224x3xf32>) -> tensor<16x224x224x3xf16>
    %270 = stablehlo.convert %0 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf16>
    %271 = stablehlo.convolution(%269, %270) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x224x224x3xf16>, tensor<7x7x3x64xf16>) -> tensor<16x112x112x64xf16>
    %272 = stablehlo.broadcast_in_dim %1, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %273 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %274 = stablehlo.convert %271 : (tensor<16x112x112x64xf16>) -> tensor<16x112x112x64xf32>
    %275 = stablehlo.broadcast_in_dim %272, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x112x112x64xf32>
    %276 = stablehlo.subtract %274, %275 : tensor<16x112x112x64xf32>
    %277 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %278 = stablehlo.broadcast_in_dim %277, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %279 = stablehlo.add %273, %278 : tensor<1x1x1x64xf32>
    %280 = stablehlo.rsqrt %279 : tensor<1x1x1x64xf32>
    %281 = stablehlo.reshape %3 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %282 = stablehlo.multiply %280, %281 : tensor<1x1x1x64xf32>
    %283 = stablehlo.broadcast_in_dim %282, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x112x112x64xf32>
    %284 = stablehlo.multiply %276, %283 : tensor<16x112x112x64xf32>
    %285 = stablehlo.reshape %4 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x112x112x64xf32>
    %287 = stablehlo.add %284, %286 : tensor<16x112x112x64xf32>
    %288 = stablehlo.convert %287 : (tensor<16x112x112x64xf32>) -> tensor<16x112x112x64xf16>
    %289 = call @relu(%288) : (tensor<16x112x112x64xf16>) -> tensor<16x112x112x64xf16>
    %290 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f16>) -> tensor<f16>
    %292 = "stablehlo.reduce_window"(%289, %291) ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
      %1363 = stablehlo.maximum %arg1, %arg2 : tensor<f16>
      stablehlo.return %1363 : tensor<f16>
    }) {padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>} : (tensor<16x112x112x64xf16>, tensor<f16>) -> tensor<16x56x56x64xf16>
    %293 = stablehlo.convert %5 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
    %294 = stablehlo.convolution(%292, %293) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<16x56x56x256xf16>
    %295 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %296 = stablehlo.broadcast_in_dim %7, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %297 = stablehlo.convert %294 : (tensor<16x56x56x256xf16>) -> tensor<16x56x56x256xf32>
    %298 = stablehlo.broadcast_in_dim %295, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x56x56x256xf32>
    %299 = stablehlo.subtract %297, %298 : tensor<16x56x56x256xf32>
    %300 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %301 = stablehlo.broadcast_in_dim %300, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %302 = stablehlo.add %296, %301 : tensor<1x1x1x256xf32>
    %303 = stablehlo.rsqrt %302 : tensor<1x1x1x256xf32>
    %304 = stablehlo.reshape %8 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %305 = stablehlo.multiply %303, %304 : tensor<1x1x1x256xf32>
    %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x56x56x256xf32>
    %307 = stablehlo.multiply %299, %306 : tensor<16x56x56x256xf32>
    %308 = stablehlo.reshape %9 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x56x56x256xf32>
    %310 = stablehlo.add %307, %309 : tensor<16x56x56x256xf32>
    %311 = stablehlo.convert %310 : (tensor<16x56x56x256xf32>) -> tensor<16x56x56x256xf16>
    %312 = stablehlo.convert %10 : (tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xf16>
    %313 = stablehlo.convolution(%292, %312) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x64xf16>, tensor<1x1x64x64xf16>) -> tensor<16x56x56x64xf16>
    %314 = stablehlo.broadcast_in_dim %11, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %315 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %316 = stablehlo.convert %313 : (tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf32>
    %317 = stablehlo.broadcast_in_dim %314, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %318 = stablehlo.subtract %316, %317 : tensor<16x56x56x64xf32>
    %319 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %320 = stablehlo.broadcast_in_dim %319, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %321 = stablehlo.add %315, %320 : tensor<1x1x1x64xf32>
    %322 = stablehlo.rsqrt %321 : tensor<1x1x1x64xf32>
    %323 = stablehlo.reshape %13 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %324 = stablehlo.multiply %322, %323 : tensor<1x1x1x64xf32>
    %325 = stablehlo.broadcast_in_dim %324, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %326 = stablehlo.multiply %318, %325 : tensor<16x56x56x64xf32>
    %327 = stablehlo.reshape %14 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %328 = stablehlo.broadcast_in_dim %327, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %329 = stablehlo.add %326, %328 : tensor<16x56x56x64xf32>
    %330 = stablehlo.convert %329 : (tensor<16x56x56x64xf32>) -> tensor<16x56x56x64xf16>
    %331 = call @relu_0(%330) : (tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf16>
    %332 = stablehlo.convert %15 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
    %333 = stablehlo.convolution(%331, %332) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<16x56x56x64xf16>
    %334 = stablehlo.broadcast_in_dim %16, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %335 = stablehlo.broadcast_in_dim %17, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %336 = stablehlo.convert %333 : (tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf32>
    %337 = stablehlo.broadcast_in_dim %334, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %338 = stablehlo.subtract %336, %337 : tensor<16x56x56x64xf32>
    %339 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %340 = stablehlo.broadcast_in_dim %339, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %341 = stablehlo.add %335, %340 : tensor<1x1x1x64xf32>
    %342 = stablehlo.rsqrt %341 : tensor<1x1x1x64xf32>
    %343 = stablehlo.reshape %18 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %344 = stablehlo.multiply %342, %343 : tensor<1x1x1x64xf32>
    %345 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %346 = stablehlo.multiply %338, %345 : tensor<16x56x56x64xf32>
    %347 = stablehlo.reshape %19 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %348 = stablehlo.broadcast_in_dim %347, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %349 = stablehlo.add %346, %348 : tensor<16x56x56x64xf32>
    %350 = stablehlo.convert %349 : (tensor<16x56x56x64xf32>) -> tensor<16x56x56x64xf16>
    %351 = call @relu_0(%350) : (tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf16>
    %352 = stablehlo.convert %20 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
    %353 = stablehlo.convolution(%351, %352) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<16x56x56x256xf16>
    %354 = stablehlo.broadcast_in_dim %21, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %355 = stablehlo.broadcast_in_dim %22, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %356 = stablehlo.convert %353 : (tensor<16x56x56x256xf16>) -> tensor<16x56x56x256xf32>
    %357 = stablehlo.broadcast_in_dim %354, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x56x56x256xf32>
    %358 = stablehlo.subtract %356, %357 : tensor<16x56x56x256xf32>
    %359 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %360 = stablehlo.broadcast_in_dim %359, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %361 = stablehlo.add %355, %360 : tensor<1x1x1x256xf32>
    %362 = stablehlo.rsqrt %361 : tensor<1x1x1x256xf32>
    %363 = stablehlo.reshape %23 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %364 = stablehlo.multiply %362, %363 : tensor<1x1x1x256xf32>
    %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x56x56x256xf32>
    %366 = stablehlo.multiply %358, %365 : tensor<16x56x56x256xf32>
    %367 = stablehlo.reshape %24 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %368 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x56x56x256xf32>
    %369 = stablehlo.add %366, %368 : tensor<16x56x56x256xf32>
    %370 = stablehlo.convert %369 : (tensor<16x56x56x256xf32>) -> tensor<16x56x56x256xf16>
    %371 = stablehlo.add %370, %311 : tensor<16x56x56x256xf16>
    %372 = call @relu_1(%371) : (tensor<16x56x56x256xf16>) -> tensor<16x56x56x256xf16>
    %373 = stablehlo.convert %25 : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xf16>
    %374 = stablehlo.convolution(%372, %373) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x256xf16>, tensor<1x1x256x64xf16>) -> tensor<16x56x56x64xf16>
    %375 = stablehlo.broadcast_in_dim %26, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %376 = stablehlo.broadcast_in_dim %27, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %377 = stablehlo.convert %374 : (tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf32>
    %378 = stablehlo.broadcast_in_dim %375, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %379 = stablehlo.subtract %377, %378 : tensor<16x56x56x64xf32>
    %380 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %381 = stablehlo.broadcast_in_dim %380, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %382 = stablehlo.add %376, %381 : tensor<1x1x1x64xf32>
    %383 = stablehlo.rsqrt %382 : tensor<1x1x1x64xf32>
    %384 = stablehlo.reshape %28 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %385 = stablehlo.multiply %383, %384 : tensor<1x1x1x64xf32>
    %386 = stablehlo.broadcast_in_dim %385, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %387 = stablehlo.multiply %379, %386 : tensor<16x56x56x64xf32>
    %388 = stablehlo.reshape %29 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %389 = stablehlo.broadcast_in_dim %388, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %390 = stablehlo.add %387, %389 : tensor<16x56x56x64xf32>
    %391 = stablehlo.convert %390 : (tensor<16x56x56x64xf32>) -> tensor<16x56x56x64xf16>
    %392 = call @relu_0(%391) : (tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf16>
    %393 = stablehlo.convert %30 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
    %394 = stablehlo.convolution(%392, %393) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<16x56x56x64xf16>
    %395 = stablehlo.broadcast_in_dim %31, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %396 = stablehlo.broadcast_in_dim %32, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %397 = stablehlo.convert %394 : (tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf32>
    %398 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %399 = stablehlo.subtract %397, %398 : tensor<16x56x56x64xf32>
    %400 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %401 = stablehlo.broadcast_in_dim %400, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %402 = stablehlo.add %396, %401 : tensor<1x1x1x64xf32>
    %403 = stablehlo.rsqrt %402 : tensor<1x1x1x64xf32>
    %404 = stablehlo.reshape %33 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %405 = stablehlo.multiply %403, %404 : tensor<1x1x1x64xf32>
    %406 = stablehlo.broadcast_in_dim %405, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %407 = stablehlo.multiply %399, %406 : tensor<16x56x56x64xf32>
    %408 = stablehlo.reshape %34 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %410 = stablehlo.add %407, %409 : tensor<16x56x56x64xf32>
    %411 = stablehlo.convert %410 : (tensor<16x56x56x64xf32>) -> tensor<16x56x56x64xf16>
    %412 = call @relu_0(%411) : (tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf16>
    %413 = stablehlo.convert %35 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
    %414 = stablehlo.convolution(%412, %413) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<16x56x56x256xf16>
    %415 = stablehlo.broadcast_in_dim %36, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %416 = stablehlo.broadcast_in_dim %37, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %417 = stablehlo.convert %414 : (tensor<16x56x56x256xf16>) -> tensor<16x56x56x256xf32>
    %418 = stablehlo.broadcast_in_dim %415, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x56x56x256xf32>
    %419 = stablehlo.subtract %417, %418 : tensor<16x56x56x256xf32>
    %420 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %421 = stablehlo.broadcast_in_dim %420, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %422 = stablehlo.add %416, %421 : tensor<1x1x1x256xf32>
    %423 = stablehlo.rsqrt %422 : tensor<1x1x1x256xf32>
    %424 = stablehlo.reshape %38 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %425 = stablehlo.multiply %423, %424 : tensor<1x1x1x256xf32>
    %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x56x56x256xf32>
    %427 = stablehlo.multiply %419, %426 : tensor<16x56x56x256xf32>
    %428 = stablehlo.reshape %39 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %429 = stablehlo.broadcast_in_dim %428, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x56x56x256xf32>
    %430 = stablehlo.add %427, %429 : tensor<16x56x56x256xf32>
    %431 = stablehlo.convert %430 : (tensor<16x56x56x256xf32>) -> tensor<16x56x56x256xf16>
    %432 = stablehlo.add %431, %372 : tensor<16x56x56x256xf16>
    %433 = call @relu_1(%432) : (tensor<16x56x56x256xf16>) -> tensor<16x56x56x256xf16>
    %434 = stablehlo.convert %40 : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xf16>
    %435 = stablehlo.convolution(%433, %434) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x256xf16>, tensor<1x1x256x64xf16>) -> tensor<16x56x56x64xf16>
    %436 = stablehlo.broadcast_in_dim %41, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %437 = stablehlo.broadcast_in_dim %42, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %438 = stablehlo.convert %435 : (tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf32>
    %439 = stablehlo.broadcast_in_dim %436, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %440 = stablehlo.subtract %438, %439 : tensor<16x56x56x64xf32>
    %441 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %442 = stablehlo.broadcast_in_dim %441, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %443 = stablehlo.add %437, %442 : tensor<1x1x1x64xf32>
    %444 = stablehlo.rsqrt %443 : tensor<1x1x1x64xf32>
    %445 = stablehlo.reshape %43 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %446 = stablehlo.multiply %444, %445 : tensor<1x1x1x64xf32>
    %447 = stablehlo.broadcast_in_dim %446, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %448 = stablehlo.multiply %440, %447 : tensor<16x56x56x64xf32>
    %449 = stablehlo.reshape %44 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %450 = stablehlo.broadcast_in_dim %449, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %451 = stablehlo.add %448, %450 : tensor<16x56x56x64xf32>
    %452 = stablehlo.convert %451 : (tensor<16x56x56x64xf32>) -> tensor<16x56x56x64xf16>
    %453 = call @relu_0(%452) : (tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf16>
    %454 = stablehlo.convert %45 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
    %455 = stablehlo.convolution(%453, %454) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<16x56x56x64xf16>
    %456 = stablehlo.broadcast_in_dim %46, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %457 = stablehlo.broadcast_in_dim %47, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %458 = stablehlo.convert %455 : (tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf32>
    %459 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %460 = stablehlo.subtract %458, %459 : tensor<16x56x56x64xf32>
    %461 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %462 = stablehlo.broadcast_in_dim %461, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %463 = stablehlo.add %457, %462 : tensor<1x1x1x64xf32>
    %464 = stablehlo.rsqrt %463 : tensor<1x1x1x64xf32>
    %465 = stablehlo.reshape %48 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %466 = stablehlo.multiply %464, %465 : tensor<1x1x1x64xf32>
    %467 = stablehlo.broadcast_in_dim %466, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %468 = stablehlo.multiply %460, %467 : tensor<16x56x56x64xf32>
    %469 = stablehlo.reshape %49 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %470 = stablehlo.broadcast_in_dim %469, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<16x56x56x64xf32>
    %471 = stablehlo.add %468, %470 : tensor<16x56x56x64xf32>
    %472 = stablehlo.convert %471 : (tensor<16x56x56x64xf32>) -> tensor<16x56x56x64xf16>
    %473 = call @relu_0(%472) : (tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf16>
    %474 = stablehlo.convert %50 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
    %475 = stablehlo.convolution(%473, %474) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<16x56x56x256xf16>
    %476 = stablehlo.broadcast_in_dim %51, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %477 = stablehlo.broadcast_in_dim %52, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %478 = stablehlo.convert %475 : (tensor<16x56x56x256xf16>) -> tensor<16x56x56x256xf32>
    %479 = stablehlo.broadcast_in_dim %476, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x56x56x256xf32>
    %480 = stablehlo.subtract %478, %479 : tensor<16x56x56x256xf32>
    %481 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %482 = stablehlo.broadcast_in_dim %481, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %483 = stablehlo.add %477, %482 : tensor<1x1x1x256xf32>
    %484 = stablehlo.rsqrt %483 : tensor<1x1x1x256xf32>
    %485 = stablehlo.reshape %53 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %486 = stablehlo.multiply %484, %485 : tensor<1x1x1x256xf32>
    %487 = stablehlo.broadcast_in_dim %486, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x56x56x256xf32>
    %488 = stablehlo.multiply %480, %487 : tensor<16x56x56x256xf32>
    %489 = stablehlo.reshape %54 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %490 = stablehlo.broadcast_in_dim %489, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x56x56x256xf32>
    %491 = stablehlo.add %488, %490 : tensor<16x56x56x256xf32>
    %492 = stablehlo.convert %491 : (tensor<16x56x56x256xf32>) -> tensor<16x56x56x256xf16>
    %493 = stablehlo.add %492, %433 : tensor<16x56x56x256xf16>
    %494 = call @relu_1(%493) : (tensor<16x56x56x256xf16>) -> tensor<16x56x56x256xf16>
    %495 = stablehlo.convert %55 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf16>
    %496 = stablehlo.convolution(%494, %495) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x256xf16>, tensor<1x1x256x512xf16>) -> tensor<16x28x28x512xf16>
    %497 = stablehlo.broadcast_in_dim %56, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %498 = stablehlo.broadcast_in_dim %57, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %499 = stablehlo.convert %496 : (tensor<16x28x28x512xf16>) -> tensor<16x28x28x512xf32>
    %500 = stablehlo.broadcast_in_dim %497, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %501 = stablehlo.subtract %499, %500 : tensor<16x28x28x512xf32>
    %502 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %503 = stablehlo.broadcast_in_dim %502, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %504 = stablehlo.add %498, %503 : tensor<1x1x1x512xf32>
    %505 = stablehlo.rsqrt %504 : tensor<1x1x1x512xf32>
    %506 = stablehlo.reshape %58 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %507 = stablehlo.multiply %505, %506 : tensor<1x1x1x512xf32>
    %508 = stablehlo.broadcast_in_dim %507, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %509 = stablehlo.multiply %501, %508 : tensor<16x28x28x512xf32>
    %510 = stablehlo.reshape %59 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %511 = stablehlo.broadcast_in_dim %510, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %512 = stablehlo.add %509, %511 : tensor<16x28x28x512xf32>
    %513 = stablehlo.convert %512 : (tensor<16x28x28x512xf32>) -> tensor<16x28x28x512xf16>
    %514 = stablehlo.convert %60 : (tensor<1x1x256x128xf32>) -> tensor<1x1x256x128xf16>
    %515 = stablehlo.convolution(%494, %514) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x256xf16>, tensor<1x1x256x128xf16>) -> tensor<16x56x56x128xf16>
    %516 = stablehlo.broadcast_in_dim %61, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %517 = stablehlo.broadcast_in_dim %62, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %518 = stablehlo.convert %515 : (tensor<16x56x56x128xf16>) -> tensor<16x56x56x128xf32>
    %519 = stablehlo.broadcast_in_dim %516, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x56x56x128xf32>
    %520 = stablehlo.subtract %518, %519 : tensor<16x56x56x128xf32>
    %521 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %522 = stablehlo.broadcast_in_dim %521, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %523 = stablehlo.add %517, %522 : tensor<1x1x1x128xf32>
    %524 = stablehlo.rsqrt %523 : tensor<1x1x1x128xf32>
    %525 = stablehlo.reshape %63 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %526 = stablehlo.multiply %524, %525 : tensor<1x1x1x128xf32>
    %527 = stablehlo.broadcast_in_dim %526, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x56x56x128xf32>
    %528 = stablehlo.multiply %520, %527 : tensor<16x56x56x128xf32>
    %529 = stablehlo.reshape %64 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %530 = stablehlo.broadcast_in_dim %529, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x56x56x128xf32>
    %531 = stablehlo.add %528, %530 : tensor<16x56x56x128xf32>
    %532 = stablehlo.convert %531 : (tensor<16x56x56x128xf32>) -> tensor<16x56x56x128xf16>
    %533 = call @relu_2(%532) : (tensor<16x56x56x128xf16>) -> tensor<16x56x56x128xf16>
    %534 = stablehlo.convert %65 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
    %535 = stablehlo.convolution(%533, %534) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x56x56x128xf16>, tensor<3x3x128x128xf16>) -> tensor<16x28x28x128xf16>
    %536 = stablehlo.broadcast_in_dim %66, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %537 = stablehlo.broadcast_in_dim %67, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %538 = stablehlo.convert %535 : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf32>
    %539 = stablehlo.broadcast_in_dim %536, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %540 = stablehlo.subtract %538, %539 : tensor<16x28x28x128xf32>
    %541 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %542 = stablehlo.broadcast_in_dim %541, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %543 = stablehlo.add %537, %542 : tensor<1x1x1x128xf32>
    %544 = stablehlo.rsqrt %543 : tensor<1x1x1x128xf32>
    %545 = stablehlo.reshape %68 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %546 = stablehlo.multiply %544, %545 : tensor<1x1x1x128xf32>
    %547 = stablehlo.broadcast_in_dim %546, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %548 = stablehlo.multiply %540, %547 : tensor<16x28x28x128xf32>
    %549 = stablehlo.reshape %69 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %550 = stablehlo.broadcast_in_dim %549, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %551 = stablehlo.add %548, %550 : tensor<16x28x28x128xf32>
    %552 = stablehlo.convert %551 : (tensor<16x28x28x128xf32>) -> tensor<16x28x28x128xf16>
    %553 = call @relu_3(%552) : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf16>
    %554 = stablehlo.convert %70 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
    %555 = stablehlo.convolution(%553, %554) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<16x28x28x512xf16>
    %556 = stablehlo.broadcast_in_dim %71, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %557 = stablehlo.broadcast_in_dim %72, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %558 = stablehlo.convert %555 : (tensor<16x28x28x512xf16>) -> tensor<16x28x28x512xf32>
    %559 = stablehlo.broadcast_in_dim %556, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %560 = stablehlo.subtract %558, %559 : tensor<16x28x28x512xf32>
    %561 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %562 = stablehlo.broadcast_in_dim %561, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %563 = stablehlo.add %557, %562 : tensor<1x1x1x512xf32>
    %564 = stablehlo.rsqrt %563 : tensor<1x1x1x512xf32>
    %565 = stablehlo.reshape %73 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %566 = stablehlo.multiply %564, %565 : tensor<1x1x1x512xf32>
    %567 = stablehlo.broadcast_in_dim %566, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %568 = stablehlo.multiply %560, %567 : tensor<16x28x28x512xf32>
    %569 = stablehlo.reshape %74 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %570 = stablehlo.broadcast_in_dim %569, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %571 = stablehlo.add %568, %570 : tensor<16x28x28x512xf32>
    %572 = stablehlo.convert %571 : (tensor<16x28x28x512xf32>) -> tensor<16x28x28x512xf16>
    %573 = stablehlo.add %572, %513 : tensor<16x28x28x512xf16>
    %574 = call @relu_4(%573) : (tensor<16x28x28x512xf16>) -> tensor<16x28x28x512xf16>
    %575 = stablehlo.convert %75 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
    %576 = stablehlo.convolution(%574, %575) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<16x28x28x128xf16>
    %577 = stablehlo.broadcast_in_dim %76, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %578 = stablehlo.broadcast_in_dim %77, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %579 = stablehlo.convert %576 : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf32>
    %580 = stablehlo.broadcast_in_dim %577, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %581 = stablehlo.subtract %579, %580 : tensor<16x28x28x128xf32>
    %582 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %583 = stablehlo.broadcast_in_dim %582, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %584 = stablehlo.add %578, %583 : tensor<1x1x1x128xf32>
    %585 = stablehlo.rsqrt %584 : tensor<1x1x1x128xf32>
    %586 = stablehlo.reshape %78 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %587 = stablehlo.multiply %585, %586 : tensor<1x1x1x128xf32>
    %588 = stablehlo.broadcast_in_dim %587, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %589 = stablehlo.multiply %581, %588 : tensor<16x28x28x128xf32>
    %590 = stablehlo.reshape %79 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %591 = stablehlo.broadcast_in_dim %590, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %592 = stablehlo.add %589, %591 : tensor<16x28x28x128xf32>
    %593 = stablehlo.convert %592 : (tensor<16x28x28x128xf32>) -> tensor<16x28x28x128xf16>
    %594 = call @relu_3(%593) : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf16>
    %595 = stablehlo.convert %80 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
    %596 = stablehlo.convolution(%594, %595) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<16x28x28x128xf16>
    %597 = stablehlo.broadcast_in_dim %81, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %598 = stablehlo.broadcast_in_dim %82, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %599 = stablehlo.convert %596 : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf32>
    %600 = stablehlo.broadcast_in_dim %597, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %601 = stablehlo.subtract %599, %600 : tensor<16x28x28x128xf32>
    %602 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %603 = stablehlo.broadcast_in_dim %602, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %604 = stablehlo.add %598, %603 : tensor<1x1x1x128xf32>
    %605 = stablehlo.rsqrt %604 : tensor<1x1x1x128xf32>
    %606 = stablehlo.reshape %83 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %607 = stablehlo.multiply %605, %606 : tensor<1x1x1x128xf32>
    %608 = stablehlo.broadcast_in_dim %607, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %609 = stablehlo.multiply %601, %608 : tensor<16x28x28x128xf32>
    %610 = stablehlo.reshape %84 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %611 = stablehlo.broadcast_in_dim %610, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %612 = stablehlo.add %609, %611 : tensor<16x28x28x128xf32>
    %613 = stablehlo.convert %612 : (tensor<16x28x28x128xf32>) -> tensor<16x28x28x128xf16>
    %614 = call @relu_3(%613) : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf16>
    %615 = stablehlo.convert %85 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
    %616 = stablehlo.convolution(%614, %615) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<16x28x28x512xf16>
    %617 = stablehlo.broadcast_in_dim %86, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %618 = stablehlo.broadcast_in_dim %87, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %619 = stablehlo.convert %616 : (tensor<16x28x28x512xf16>) -> tensor<16x28x28x512xf32>
    %620 = stablehlo.broadcast_in_dim %617, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %621 = stablehlo.subtract %619, %620 : tensor<16x28x28x512xf32>
    %622 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %623 = stablehlo.broadcast_in_dim %622, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %624 = stablehlo.add %618, %623 : tensor<1x1x1x512xf32>
    %625 = stablehlo.rsqrt %624 : tensor<1x1x1x512xf32>
    %626 = stablehlo.reshape %88 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %627 = stablehlo.multiply %625, %626 : tensor<1x1x1x512xf32>
    %628 = stablehlo.broadcast_in_dim %627, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %629 = stablehlo.multiply %621, %628 : tensor<16x28x28x512xf32>
    %630 = stablehlo.reshape %89 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %632 = stablehlo.add %629, %631 : tensor<16x28x28x512xf32>
    %633 = stablehlo.convert %632 : (tensor<16x28x28x512xf32>) -> tensor<16x28x28x512xf16>
    %634 = stablehlo.add %633, %574 : tensor<16x28x28x512xf16>
    %635 = call @relu_4(%634) : (tensor<16x28x28x512xf16>) -> tensor<16x28x28x512xf16>
    %636 = stablehlo.convert %90 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
    %637 = stablehlo.convolution(%635, %636) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<16x28x28x128xf16>
    %638 = stablehlo.broadcast_in_dim %91, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %639 = stablehlo.broadcast_in_dim %92, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %640 = stablehlo.convert %637 : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf32>
    %641 = stablehlo.broadcast_in_dim %638, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %642 = stablehlo.subtract %640, %641 : tensor<16x28x28x128xf32>
    %643 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %644 = stablehlo.broadcast_in_dim %643, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %645 = stablehlo.add %639, %644 : tensor<1x1x1x128xf32>
    %646 = stablehlo.rsqrt %645 : tensor<1x1x1x128xf32>
    %647 = stablehlo.reshape %93 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %648 = stablehlo.multiply %646, %647 : tensor<1x1x1x128xf32>
    %649 = stablehlo.broadcast_in_dim %648, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %650 = stablehlo.multiply %642, %649 : tensor<16x28x28x128xf32>
    %651 = stablehlo.reshape %94 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %652 = stablehlo.broadcast_in_dim %651, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %653 = stablehlo.add %650, %652 : tensor<16x28x28x128xf32>
    %654 = stablehlo.convert %653 : (tensor<16x28x28x128xf32>) -> tensor<16x28x28x128xf16>
    %655 = call @relu_3(%654) : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf16>
    %656 = stablehlo.convert %95 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
    %657 = stablehlo.convolution(%655, %656) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<16x28x28x128xf16>
    %658 = stablehlo.broadcast_in_dim %96, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %659 = stablehlo.broadcast_in_dim %97, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %660 = stablehlo.convert %657 : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf32>
    %661 = stablehlo.broadcast_in_dim %658, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %662 = stablehlo.subtract %660, %661 : tensor<16x28x28x128xf32>
    %663 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %664 = stablehlo.broadcast_in_dim %663, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %665 = stablehlo.add %659, %664 : tensor<1x1x1x128xf32>
    %666 = stablehlo.rsqrt %665 : tensor<1x1x1x128xf32>
    %667 = stablehlo.reshape %98 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %668 = stablehlo.multiply %666, %667 : tensor<1x1x1x128xf32>
    %669 = stablehlo.broadcast_in_dim %668, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %670 = stablehlo.multiply %662, %669 : tensor<16x28x28x128xf32>
    %671 = stablehlo.reshape %99 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %672 = stablehlo.broadcast_in_dim %671, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %673 = stablehlo.add %670, %672 : tensor<16x28x28x128xf32>
    %674 = stablehlo.convert %673 : (tensor<16x28x28x128xf32>) -> tensor<16x28x28x128xf16>
    %675 = call @relu_3(%674) : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf16>
    %676 = stablehlo.convert %100 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
    %677 = stablehlo.convolution(%675, %676) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<16x28x28x512xf16>
    %678 = stablehlo.broadcast_in_dim %101, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %679 = stablehlo.broadcast_in_dim %102, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %680 = stablehlo.convert %677 : (tensor<16x28x28x512xf16>) -> tensor<16x28x28x512xf32>
    %681 = stablehlo.broadcast_in_dim %678, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %682 = stablehlo.subtract %680, %681 : tensor<16x28x28x512xf32>
    %683 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %684 = stablehlo.broadcast_in_dim %683, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %685 = stablehlo.add %679, %684 : tensor<1x1x1x512xf32>
    %686 = stablehlo.rsqrt %685 : tensor<1x1x1x512xf32>
    %687 = stablehlo.reshape %103 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %688 = stablehlo.multiply %686, %687 : tensor<1x1x1x512xf32>
    %689 = stablehlo.broadcast_in_dim %688, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %690 = stablehlo.multiply %682, %689 : tensor<16x28x28x512xf32>
    %691 = stablehlo.reshape %104 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %692 = stablehlo.broadcast_in_dim %691, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %693 = stablehlo.add %690, %692 : tensor<16x28x28x512xf32>
    %694 = stablehlo.convert %693 : (tensor<16x28x28x512xf32>) -> tensor<16x28x28x512xf16>
    %695 = stablehlo.add %694, %635 : tensor<16x28x28x512xf16>
    %696 = call @relu_4(%695) : (tensor<16x28x28x512xf16>) -> tensor<16x28x28x512xf16>
    %697 = stablehlo.convert %105 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
    %698 = stablehlo.convolution(%696, %697) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<16x28x28x128xf16>
    %699 = stablehlo.broadcast_in_dim %106, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %700 = stablehlo.broadcast_in_dim %107, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %701 = stablehlo.convert %698 : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf32>
    %702 = stablehlo.broadcast_in_dim %699, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %703 = stablehlo.subtract %701, %702 : tensor<16x28x28x128xf32>
    %704 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %705 = stablehlo.broadcast_in_dim %704, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %706 = stablehlo.add %700, %705 : tensor<1x1x1x128xf32>
    %707 = stablehlo.rsqrt %706 : tensor<1x1x1x128xf32>
    %708 = stablehlo.reshape %108 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %709 = stablehlo.multiply %707, %708 : tensor<1x1x1x128xf32>
    %710 = stablehlo.broadcast_in_dim %709, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %711 = stablehlo.multiply %703, %710 : tensor<16x28x28x128xf32>
    %712 = stablehlo.reshape %109 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %713 = stablehlo.broadcast_in_dim %712, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %714 = stablehlo.add %711, %713 : tensor<16x28x28x128xf32>
    %715 = stablehlo.convert %714 : (tensor<16x28x28x128xf32>) -> tensor<16x28x28x128xf16>
    %716 = call @relu_3(%715) : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf16>
    %717 = stablehlo.convert %110 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
    %718 = stablehlo.convolution(%716, %717) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<16x28x28x128xf16>
    %719 = stablehlo.broadcast_in_dim %111, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %720 = stablehlo.broadcast_in_dim %112, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %721 = stablehlo.convert %718 : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf32>
    %722 = stablehlo.broadcast_in_dim %719, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %723 = stablehlo.subtract %721, %722 : tensor<16x28x28x128xf32>
    %724 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %725 = stablehlo.broadcast_in_dim %724, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %726 = stablehlo.add %720, %725 : tensor<1x1x1x128xf32>
    %727 = stablehlo.rsqrt %726 : tensor<1x1x1x128xf32>
    %728 = stablehlo.reshape %113 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %729 = stablehlo.multiply %727, %728 : tensor<1x1x1x128xf32>
    %730 = stablehlo.broadcast_in_dim %729, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %731 = stablehlo.multiply %723, %730 : tensor<16x28x28x128xf32>
    %732 = stablehlo.reshape %114 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %733 = stablehlo.broadcast_in_dim %732, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<16x28x28x128xf32>
    %734 = stablehlo.add %731, %733 : tensor<16x28x28x128xf32>
    %735 = stablehlo.convert %734 : (tensor<16x28x28x128xf32>) -> tensor<16x28x28x128xf16>
    %736 = call @relu_3(%735) : (tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf16>
    %737 = stablehlo.convert %115 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
    %738 = stablehlo.convolution(%736, %737) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<16x28x28x512xf16>
    %739 = stablehlo.broadcast_in_dim %116, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %740 = stablehlo.broadcast_in_dim %117, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %741 = stablehlo.convert %738 : (tensor<16x28x28x512xf16>) -> tensor<16x28x28x512xf32>
    %742 = stablehlo.broadcast_in_dim %739, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %743 = stablehlo.subtract %741, %742 : tensor<16x28x28x512xf32>
    %744 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %745 = stablehlo.broadcast_in_dim %744, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %746 = stablehlo.add %740, %745 : tensor<1x1x1x512xf32>
    %747 = stablehlo.rsqrt %746 : tensor<1x1x1x512xf32>
    %748 = stablehlo.reshape %118 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %749 = stablehlo.multiply %747, %748 : tensor<1x1x1x512xf32>
    %750 = stablehlo.broadcast_in_dim %749, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %751 = stablehlo.multiply %743, %750 : tensor<16x28x28x512xf32>
    %752 = stablehlo.reshape %119 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %753 = stablehlo.broadcast_in_dim %752, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x28x28x512xf32>
    %754 = stablehlo.add %751, %753 : tensor<16x28x28x512xf32>
    %755 = stablehlo.convert %754 : (tensor<16x28x28x512xf32>) -> tensor<16x28x28x512xf16>
    %756 = stablehlo.add %755, %696 : tensor<16x28x28x512xf16>
    %757 = call @relu_4(%756) : (tensor<16x28x28x512xf16>) -> tensor<16x28x28x512xf16>
    %758 = stablehlo.convert %120 : (tensor<1x1x512x1024xf32>) -> tensor<1x1x512x1024xf16>
    %759 = stablehlo.convolution(%757, %758) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x512xf16>, tensor<1x1x512x1024xf16>) -> tensor<16x14x14x1024xf16>
    %760 = stablehlo.broadcast_in_dim %121, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %761 = stablehlo.broadcast_in_dim %122, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %762 = stablehlo.convert %759 : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf32>
    %763 = stablehlo.broadcast_in_dim %760, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %764 = stablehlo.subtract %762, %763 : tensor<16x14x14x1024xf32>
    %765 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %766 = stablehlo.broadcast_in_dim %765, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %767 = stablehlo.add %761, %766 : tensor<1x1x1x1024xf32>
    %768 = stablehlo.rsqrt %767 : tensor<1x1x1x1024xf32>
    %769 = stablehlo.reshape %123 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %770 = stablehlo.multiply %768, %769 : tensor<1x1x1x1024xf32>
    %771 = stablehlo.broadcast_in_dim %770, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %772 = stablehlo.multiply %764, %771 : tensor<16x14x14x1024xf32>
    %773 = stablehlo.reshape %124 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %774 = stablehlo.broadcast_in_dim %773, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %775 = stablehlo.add %772, %774 : tensor<16x14x14x1024xf32>
    %776 = stablehlo.convert %775 : (tensor<16x14x14x1024xf32>) -> tensor<16x14x14x1024xf16>
    %777 = stablehlo.convert %125 : (tensor<1x1x512x256xf32>) -> tensor<1x1x512x256xf16>
    %778 = stablehlo.convolution(%757, %777) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x512xf16>, tensor<1x1x512x256xf16>) -> tensor<16x28x28x256xf16>
    %779 = stablehlo.broadcast_in_dim %126, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %780 = stablehlo.broadcast_in_dim %127, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %781 = stablehlo.convert %778 : (tensor<16x28x28x256xf16>) -> tensor<16x28x28x256xf32>
    %782 = stablehlo.broadcast_in_dim %779, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x28x28x256xf32>
    %783 = stablehlo.subtract %781, %782 : tensor<16x28x28x256xf32>
    %784 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %785 = stablehlo.broadcast_in_dim %784, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %786 = stablehlo.add %780, %785 : tensor<1x1x1x256xf32>
    %787 = stablehlo.rsqrt %786 : tensor<1x1x1x256xf32>
    %788 = stablehlo.reshape %128 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %789 = stablehlo.multiply %787, %788 : tensor<1x1x1x256xf32>
    %790 = stablehlo.broadcast_in_dim %789, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x28x28x256xf32>
    %791 = stablehlo.multiply %783, %790 : tensor<16x28x28x256xf32>
    %792 = stablehlo.reshape %129 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %793 = stablehlo.broadcast_in_dim %792, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x28x28x256xf32>
    %794 = stablehlo.add %791, %793 : tensor<16x28x28x256xf32>
    %795 = stablehlo.convert %794 : (tensor<16x28x28x256xf32>) -> tensor<16x28x28x256xf16>
    %796 = call @relu_5(%795) : (tensor<16x28x28x256xf16>) -> tensor<16x28x28x256xf16>
    %797 = stablehlo.convert %130 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
    %798 = stablehlo.convolution(%796, %797) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x28x28x256xf16>, tensor<3x3x256x256xf16>) -> tensor<16x14x14x256xf16>
    %799 = stablehlo.broadcast_in_dim %131, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %800 = stablehlo.broadcast_in_dim %132, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %801 = stablehlo.convert %798 : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf32>
    %802 = stablehlo.broadcast_in_dim %799, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %803 = stablehlo.subtract %801, %802 : tensor<16x14x14x256xf32>
    %804 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %805 = stablehlo.broadcast_in_dim %804, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %806 = stablehlo.add %800, %805 : tensor<1x1x1x256xf32>
    %807 = stablehlo.rsqrt %806 : tensor<1x1x1x256xf32>
    %808 = stablehlo.reshape %133 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %809 = stablehlo.multiply %807, %808 : tensor<1x1x1x256xf32>
    %810 = stablehlo.broadcast_in_dim %809, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %811 = stablehlo.multiply %803, %810 : tensor<16x14x14x256xf32>
    %812 = stablehlo.reshape %134 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %813 = stablehlo.broadcast_in_dim %812, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %814 = stablehlo.add %811, %813 : tensor<16x14x14x256xf32>
    %815 = stablehlo.convert %814 : (tensor<16x14x14x256xf32>) -> tensor<16x14x14x256xf16>
    %816 = call @relu_6(%815) : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf16>
    %817 = stablehlo.convert %135 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
    %818 = stablehlo.convolution(%816, %817) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<16x14x14x1024xf16>
    %819 = stablehlo.broadcast_in_dim %136, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %820 = stablehlo.broadcast_in_dim %137, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %821 = stablehlo.convert %818 : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf32>
    %822 = stablehlo.broadcast_in_dim %819, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %823 = stablehlo.subtract %821, %822 : tensor<16x14x14x1024xf32>
    %824 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %825 = stablehlo.broadcast_in_dim %824, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %826 = stablehlo.add %820, %825 : tensor<1x1x1x1024xf32>
    %827 = stablehlo.rsqrt %826 : tensor<1x1x1x1024xf32>
    %828 = stablehlo.reshape %138 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %829 = stablehlo.multiply %827, %828 : tensor<1x1x1x1024xf32>
    %830 = stablehlo.broadcast_in_dim %829, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %831 = stablehlo.multiply %823, %830 : tensor<16x14x14x1024xf32>
    %832 = stablehlo.reshape %139 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %833 = stablehlo.broadcast_in_dim %832, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %834 = stablehlo.add %831, %833 : tensor<16x14x14x1024xf32>
    %835 = stablehlo.convert %834 : (tensor<16x14x14x1024xf32>) -> tensor<16x14x14x1024xf16>
    %836 = stablehlo.add %835, %776 : tensor<16x14x14x1024xf16>
    %837 = call @relu_7(%836) : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf16>
    %838 = stablehlo.convert %140 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
    %839 = stablehlo.convolution(%837, %838) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<16x14x14x256xf16>
    %840 = stablehlo.broadcast_in_dim %141, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %841 = stablehlo.broadcast_in_dim %142, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %842 = stablehlo.convert %839 : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf32>
    %843 = stablehlo.broadcast_in_dim %840, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %844 = stablehlo.subtract %842, %843 : tensor<16x14x14x256xf32>
    %845 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %846 = stablehlo.broadcast_in_dim %845, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %847 = stablehlo.add %841, %846 : tensor<1x1x1x256xf32>
    %848 = stablehlo.rsqrt %847 : tensor<1x1x1x256xf32>
    %849 = stablehlo.reshape %143 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %850 = stablehlo.multiply %848, %849 : tensor<1x1x1x256xf32>
    %851 = stablehlo.broadcast_in_dim %850, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %852 = stablehlo.multiply %844, %851 : tensor<16x14x14x256xf32>
    %853 = stablehlo.reshape %144 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %854 = stablehlo.broadcast_in_dim %853, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %855 = stablehlo.add %852, %854 : tensor<16x14x14x256xf32>
    %856 = stablehlo.convert %855 : (tensor<16x14x14x256xf32>) -> tensor<16x14x14x256xf16>
    %857 = call @relu_6(%856) : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf16>
    %858 = stablehlo.convert %145 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
    %859 = stablehlo.convolution(%857, %858) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<16x14x14x256xf16>
    %860 = stablehlo.broadcast_in_dim %146, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %861 = stablehlo.broadcast_in_dim %147, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %862 = stablehlo.convert %859 : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf32>
    %863 = stablehlo.broadcast_in_dim %860, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %864 = stablehlo.subtract %862, %863 : tensor<16x14x14x256xf32>
    %865 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %866 = stablehlo.broadcast_in_dim %865, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %867 = stablehlo.add %861, %866 : tensor<1x1x1x256xf32>
    %868 = stablehlo.rsqrt %867 : tensor<1x1x1x256xf32>
    %869 = stablehlo.reshape %148 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %870 = stablehlo.multiply %868, %869 : tensor<1x1x1x256xf32>
    %871 = stablehlo.broadcast_in_dim %870, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %872 = stablehlo.multiply %864, %871 : tensor<16x14x14x256xf32>
    %873 = stablehlo.reshape %149 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %874 = stablehlo.broadcast_in_dim %873, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %875 = stablehlo.add %872, %874 : tensor<16x14x14x256xf32>
    %876 = stablehlo.convert %875 : (tensor<16x14x14x256xf32>) -> tensor<16x14x14x256xf16>
    %877 = call @relu_6(%876) : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf16>
    %878 = stablehlo.convert %150 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
    %879 = stablehlo.convolution(%877, %878) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<16x14x14x1024xf16>
    %880 = stablehlo.broadcast_in_dim %151, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %881 = stablehlo.broadcast_in_dim %152, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %882 = stablehlo.convert %879 : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf32>
    %883 = stablehlo.broadcast_in_dim %880, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %884 = stablehlo.subtract %882, %883 : tensor<16x14x14x1024xf32>
    %885 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %886 = stablehlo.broadcast_in_dim %885, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %887 = stablehlo.add %881, %886 : tensor<1x1x1x1024xf32>
    %888 = stablehlo.rsqrt %887 : tensor<1x1x1x1024xf32>
    %889 = stablehlo.reshape %153 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %890 = stablehlo.multiply %888, %889 : tensor<1x1x1x1024xf32>
    %891 = stablehlo.broadcast_in_dim %890, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %892 = stablehlo.multiply %884, %891 : tensor<16x14x14x1024xf32>
    %893 = stablehlo.reshape %154 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %894 = stablehlo.broadcast_in_dim %893, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %895 = stablehlo.add %892, %894 : tensor<16x14x14x1024xf32>
    %896 = stablehlo.convert %895 : (tensor<16x14x14x1024xf32>) -> tensor<16x14x14x1024xf16>
    %897 = stablehlo.add %896, %837 : tensor<16x14x14x1024xf16>
    %898 = call @relu_7(%897) : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf16>
    %899 = stablehlo.convert %155 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
    %900 = stablehlo.convolution(%898, %899) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<16x14x14x256xf16>
    %901 = stablehlo.broadcast_in_dim %156, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %902 = stablehlo.broadcast_in_dim %157, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %903 = stablehlo.convert %900 : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf32>
    %904 = stablehlo.broadcast_in_dim %901, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %905 = stablehlo.subtract %903, %904 : tensor<16x14x14x256xf32>
    %906 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %907 = stablehlo.broadcast_in_dim %906, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %908 = stablehlo.add %902, %907 : tensor<1x1x1x256xf32>
    %909 = stablehlo.rsqrt %908 : tensor<1x1x1x256xf32>
    %910 = stablehlo.reshape %158 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %911 = stablehlo.multiply %909, %910 : tensor<1x1x1x256xf32>
    %912 = stablehlo.broadcast_in_dim %911, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %913 = stablehlo.multiply %905, %912 : tensor<16x14x14x256xf32>
    %914 = stablehlo.reshape %159 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %915 = stablehlo.broadcast_in_dim %914, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %916 = stablehlo.add %913, %915 : tensor<16x14x14x256xf32>
    %917 = stablehlo.convert %916 : (tensor<16x14x14x256xf32>) -> tensor<16x14x14x256xf16>
    %918 = call @relu_6(%917) : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf16>
    %919 = stablehlo.convert %160 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
    %920 = stablehlo.convolution(%918, %919) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<16x14x14x256xf16>
    %921 = stablehlo.broadcast_in_dim %161, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %922 = stablehlo.broadcast_in_dim %162, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %923 = stablehlo.convert %920 : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf32>
    %924 = stablehlo.broadcast_in_dim %921, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %925 = stablehlo.subtract %923, %924 : tensor<16x14x14x256xf32>
    %926 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %927 = stablehlo.broadcast_in_dim %926, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %928 = stablehlo.add %922, %927 : tensor<1x1x1x256xf32>
    %929 = stablehlo.rsqrt %928 : tensor<1x1x1x256xf32>
    %930 = stablehlo.reshape %163 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %931 = stablehlo.multiply %929, %930 : tensor<1x1x1x256xf32>
    %932 = stablehlo.broadcast_in_dim %931, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %933 = stablehlo.multiply %925, %932 : tensor<16x14x14x256xf32>
    %934 = stablehlo.reshape %164 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %935 = stablehlo.broadcast_in_dim %934, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %936 = stablehlo.add %933, %935 : tensor<16x14x14x256xf32>
    %937 = stablehlo.convert %936 : (tensor<16x14x14x256xf32>) -> tensor<16x14x14x256xf16>
    %938 = call @relu_6(%937) : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf16>
    %939 = stablehlo.convert %165 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
    %940 = stablehlo.convolution(%938, %939) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<16x14x14x1024xf16>
    %941 = stablehlo.broadcast_in_dim %166, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %942 = stablehlo.broadcast_in_dim %167, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %943 = stablehlo.convert %940 : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf32>
    %944 = stablehlo.broadcast_in_dim %941, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %945 = stablehlo.subtract %943, %944 : tensor<16x14x14x1024xf32>
    %946 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %947 = stablehlo.broadcast_in_dim %946, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %948 = stablehlo.add %942, %947 : tensor<1x1x1x1024xf32>
    %949 = stablehlo.rsqrt %948 : tensor<1x1x1x1024xf32>
    %950 = stablehlo.reshape %168 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %951 = stablehlo.multiply %949, %950 : tensor<1x1x1x1024xf32>
    %952 = stablehlo.broadcast_in_dim %951, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %953 = stablehlo.multiply %945, %952 : tensor<16x14x14x1024xf32>
    %954 = stablehlo.reshape %169 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %955 = stablehlo.broadcast_in_dim %954, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %956 = stablehlo.add %953, %955 : tensor<16x14x14x1024xf32>
    %957 = stablehlo.convert %956 : (tensor<16x14x14x1024xf32>) -> tensor<16x14x14x1024xf16>
    %958 = stablehlo.add %957, %898 : tensor<16x14x14x1024xf16>
    %959 = call @relu_7(%958) : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf16>
    %960 = stablehlo.convert %170 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
    %961 = stablehlo.convolution(%959, %960) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<16x14x14x256xf16>
    %962 = stablehlo.broadcast_in_dim %171, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %963 = stablehlo.broadcast_in_dim %172, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %964 = stablehlo.convert %961 : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf32>
    %965 = stablehlo.broadcast_in_dim %962, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %966 = stablehlo.subtract %964, %965 : tensor<16x14x14x256xf32>
    %967 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %968 = stablehlo.broadcast_in_dim %967, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %969 = stablehlo.add %963, %968 : tensor<1x1x1x256xf32>
    %970 = stablehlo.rsqrt %969 : tensor<1x1x1x256xf32>
    %971 = stablehlo.reshape %173 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %972 = stablehlo.multiply %970, %971 : tensor<1x1x1x256xf32>
    %973 = stablehlo.broadcast_in_dim %972, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %974 = stablehlo.multiply %966, %973 : tensor<16x14x14x256xf32>
    %975 = stablehlo.reshape %174 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %976 = stablehlo.broadcast_in_dim %975, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %977 = stablehlo.add %974, %976 : tensor<16x14x14x256xf32>
    %978 = stablehlo.convert %977 : (tensor<16x14x14x256xf32>) -> tensor<16x14x14x256xf16>
    %979 = call @relu_6(%978) : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf16>
    %980 = stablehlo.convert %175 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
    %981 = stablehlo.convolution(%979, %980) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<16x14x14x256xf16>
    %982 = stablehlo.broadcast_in_dim %176, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %983 = stablehlo.broadcast_in_dim %177, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %984 = stablehlo.convert %981 : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf32>
    %985 = stablehlo.broadcast_in_dim %982, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %986 = stablehlo.subtract %984, %985 : tensor<16x14x14x256xf32>
    %987 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %988 = stablehlo.broadcast_in_dim %987, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %989 = stablehlo.add %983, %988 : tensor<1x1x1x256xf32>
    %990 = stablehlo.rsqrt %989 : tensor<1x1x1x256xf32>
    %991 = stablehlo.reshape %178 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %992 = stablehlo.multiply %990, %991 : tensor<1x1x1x256xf32>
    %993 = stablehlo.broadcast_in_dim %992, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %994 = stablehlo.multiply %986, %993 : tensor<16x14x14x256xf32>
    %995 = stablehlo.reshape %179 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %996 = stablehlo.broadcast_in_dim %995, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %997 = stablehlo.add %994, %996 : tensor<16x14x14x256xf32>
    %998 = stablehlo.convert %997 : (tensor<16x14x14x256xf32>) -> tensor<16x14x14x256xf16>
    %999 = call @relu_6(%998) : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf16>
    %1000 = stablehlo.convert %180 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
    %1001 = stablehlo.convolution(%999, %1000) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<16x14x14x1024xf16>
    %1002 = stablehlo.broadcast_in_dim %181, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1003 = stablehlo.broadcast_in_dim %182, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1004 = stablehlo.convert %1001 : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf32>
    %1005 = stablehlo.broadcast_in_dim %1002, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %1006 = stablehlo.subtract %1004, %1005 : tensor<16x14x14x1024xf32>
    %1007 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1008 = stablehlo.broadcast_in_dim %1007, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1009 = stablehlo.add %1003, %1008 : tensor<1x1x1x1024xf32>
    %1010 = stablehlo.rsqrt %1009 : tensor<1x1x1x1024xf32>
    %1011 = stablehlo.reshape %183 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1012 = stablehlo.multiply %1010, %1011 : tensor<1x1x1x1024xf32>
    %1013 = stablehlo.broadcast_in_dim %1012, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %1014 = stablehlo.multiply %1006, %1013 : tensor<16x14x14x1024xf32>
    %1015 = stablehlo.reshape %184 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1016 = stablehlo.broadcast_in_dim %1015, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %1017 = stablehlo.add %1014, %1016 : tensor<16x14x14x1024xf32>
    %1018 = stablehlo.convert %1017 : (tensor<16x14x14x1024xf32>) -> tensor<16x14x14x1024xf16>
    %1019 = stablehlo.add %1018, %959 : tensor<16x14x14x1024xf16>
    %1020 = call @relu_7(%1019) : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf16>
    %1021 = stablehlo.convert %185 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
    %1022 = stablehlo.convolution(%1020, %1021) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<16x14x14x256xf16>
    %1023 = stablehlo.broadcast_in_dim %186, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1024 = stablehlo.broadcast_in_dim %187, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1025 = stablehlo.convert %1022 : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf32>
    %1026 = stablehlo.broadcast_in_dim %1023, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %1027 = stablehlo.subtract %1025, %1026 : tensor<16x14x14x256xf32>
    %1028 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1029 = stablehlo.broadcast_in_dim %1028, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1030 = stablehlo.add %1024, %1029 : tensor<1x1x1x256xf32>
    %1031 = stablehlo.rsqrt %1030 : tensor<1x1x1x256xf32>
    %1032 = stablehlo.reshape %188 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1033 = stablehlo.multiply %1031, %1032 : tensor<1x1x1x256xf32>
    %1034 = stablehlo.broadcast_in_dim %1033, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %1035 = stablehlo.multiply %1027, %1034 : tensor<16x14x14x256xf32>
    %1036 = stablehlo.reshape %189 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1037 = stablehlo.broadcast_in_dim %1036, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %1038 = stablehlo.add %1035, %1037 : tensor<16x14x14x256xf32>
    %1039 = stablehlo.convert %1038 : (tensor<16x14x14x256xf32>) -> tensor<16x14x14x256xf16>
    %1040 = call @relu_6(%1039) : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf16>
    %1041 = stablehlo.convert %190 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
    %1042 = stablehlo.convolution(%1040, %1041) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<16x14x14x256xf16>
    %1043 = stablehlo.broadcast_in_dim %191, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1044 = stablehlo.broadcast_in_dim %192, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1045 = stablehlo.convert %1042 : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf32>
    %1046 = stablehlo.broadcast_in_dim %1043, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %1047 = stablehlo.subtract %1045, %1046 : tensor<16x14x14x256xf32>
    %1048 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1049 = stablehlo.broadcast_in_dim %1048, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1050 = stablehlo.add %1044, %1049 : tensor<1x1x1x256xf32>
    %1051 = stablehlo.rsqrt %1050 : tensor<1x1x1x256xf32>
    %1052 = stablehlo.reshape %193 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1053 = stablehlo.multiply %1051, %1052 : tensor<1x1x1x256xf32>
    %1054 = stablehlo.broadcast_in_dim %1053, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %1055 = stablehlo.multiply %1047, %1054 : tensor<16x14x14x256xf32>
    %1056 = stablehlo.reshape %194 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1057 = stablehlo.broadcast_in_dim %1056, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %1058 = stablehlo.add %1055, %1057 : tensor<16x14x14x256xf32>
    %1059 = stablehlo.convert %1058 : (tensor<16x14x14x256xf32>) -> tensor<16x14x14x256xf16>
    %1060 = call @relu_6(%1059) : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf16>
    %1061 = stablehlo.convert %195 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
    %1062 = stablehlo.convolution(%1060, %1061) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<16x14x14x1024xf16>
    %1063 = stablehlo.broadcast_in_dim %196, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1064 = stablehlo.broadcast_in_dim %197, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1065 = stablehlo.convert %1062 : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf32>
    %1066 = stablehlo.broadcast_in_dim %1063, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %1067 = stablehlo.subtract %1065, %1066 : tensor<16x14x14x1024xf32>
    %1068 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1069 = stablehlo.broadcast_in_dim %1068, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1070 = stablehlo.add %1064, %1069 : tensor<1x1x1x1024xf32>
    %1071 = stablehlo.rsqrt %1070 : tensor<1x1x1x1024xf32>
    %1072 = stablehlo.reshape %198 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1073 = stablehlo.multiply %1071, %1072 : tensor<1x1x1x1024xf32>
    %1074 = stablehlo.broadcast_in_dim %1073, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %1075 = stablehlo.multiply %1067, %1074 : tensor<16x14x14x1024xf32>
    %1076 = stablehlo.reshape %199 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1077 = stablehlo.broadcast_in_dim %1076, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %1078 = stablehlo.add %1075, %1077 : tensor<16x14x14x1024xf32>
    %1079 = stablehlo.convert %1078 : (tensor<16x14x14x1024xf32>) -> tensor<16x14x14x1024xf16>
    %1080 = stablehlo.add %1079, %1020 : tensor<16x14x14x1024xf16>
    %1081 = call @relu_7(%1080) : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf16>
    %1082 = stablehlo.convert %200 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
    %1083 = stablehlo.convolution(%1081, %1082) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<16x14x14x256xf16>
    %1084 = stablehlo.broadcast_in_dim %201, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1085 = stablehlo.broadcast_in_dim %202, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1086 = stablehlo.convert %1083 : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf32>
    %1087 = stablehlo.broadcast_in_dim %1084, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %1088 = stablehlo.subtract %1086, %1087 : tensor<16x14x14x256xf32>
    %1089 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1090 = stablehlo.broadcast_in_dim %1089, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1091 = stablehlo.add %1085, %1090 : tensor<1x1x1x256xf32>
    %1092 = stablehlo.rsqrt %1091 : tensor<1x1x1x256xf32>
    %1093 = stablehlo.reshape %203 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1094 = stablehlo.multiply %1092, %1093 : tensor<1x1x1x256xf32>
    %1095 = stablehlo.broadcast_in_dim %1094, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %1096 = stablehlo.multiply %1088, %1095 : tensor<16x14x14x256xf32>
    %1097 = stablehlo.reshape %204 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1098 = stablehlo.broadcast_in_dim %1097, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %1099 = stablehlo.add %1096, %1098 : tensor<16x14x14x256xf32>
    %1100 = stablehlo.convert %1099 : (tensor<16x14x14x256xf32>) -> tensor<16x14x14x256xf16>
    %1101 = call @relu_6(%1100) : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf16>
    %1102 = stablehlo.convert %205 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
    %1103 = stablehlo.convolution(%1101, %1102) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<16x14x14x256xf16>
    %1104 = stablehlo.broadcast_in_dim %206, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1105 = stablehlo.broadcast_in_dim %207, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1106 = stablehlo.convert %1103 : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf32>
    %1107 = stablehlo.broadcast_in_dim %1104, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %1108 = stablehlo.subtract %1106, %1107 : tensor<16x14x14x256xf32>
    %1109 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1110 = stablehlo.broadcast_in_dim %1109, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1111 = stablehlo.add %1105, %1110 : tensor<1x1x1x256xf32>
    %1112 = stablehlo.rsqrt %1111 : tensor<1x1x1x256xf32>
    %1113 = stablehlo.reshape %208 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1114 = stablehlo.multiply %1112, %1113 : tensor<1x1x1x256xf32>
    %1115 = stablehlo.broadcast_in_dim %1114, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %1116 = stablehlo.multiply %1108, %1115 : tensor<16x14x14x256xf32>
    %1117 = stablehlo.reshape %209 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1118 = stablehlo.broadcast_in_dim %1117, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<16x14x14x256xf32>
    %1119 = stablehlo.add %1116, %1118 : tensor<16x14x14x256xf32>
    %1120 = stablehlo.convert %1119 : (tensor<16x14x14x256xf32>) -> tensor<16x14x14x256xf16>
    %1121 = call @relu_6(%1120) : (tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf16>
    %1122 = stablehlo.convert %210 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
    %1123 = stablehlo.convolution(%1121, %1122) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<16x14x14x1024xf16>
    %1124 = stablehlo.broadcast_in_dim %211, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1125 = stablehlo.broadcast_in_dim %212, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1126 = stablehlo.convert %1123 : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf32>
    %1127 = stablehlo.broadcast_in_dim %1124, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %1128 = stablehlo.subtract %1126, %1127 : tensor<16x14x14x1024xf32>
    %1129 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1130 = stablehlo.broadcast_in_dim %1129, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1131 = stablehlo.add %1125, %1130 : tensor<1x1x1x1024xf32>
    %1132 = stablehlo.rsqrt %1131 : tensor<1x1x1x1024xf32>
    %1133 = stablehlo.reshape %213 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1134 = stablehlo.multiply %1132, %1133 : tensor<1x1x1x1024xf32>
    %1135 = stablehlo.broadcast_in_dim %1134, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %1136 = stablehlo.multiply %1128, %1135 : tensor<16x14x14x1024xf32>
    %1137 = stablehlo.reshape %214 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1138 = stablehlo.broadcast_in_dim %1137, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<16x14x14x1024xf32>
    %1139 = stablehlo.add %1136, %1138 : tensor<16x14x14x1024xf32>
    %1140 = stablehlo.convert %1139 : (tensor<16x14x14x1024xf32>) -> tensor<16x14x14x1024xf16>
    %1141 = stablehlo.add %1140, %1081 : tensor<16x14x14x1024xf16>
    %1142 = call @relu_7(%1141) : (tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf16>
    %1143 = stablehlo.convert %215 : (tensor<1x1x1024x2048xf32>) -> tensor<1x1x1024x2048xf16>
    %1144 = stablehlo.convolution(%1142, %1143) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x1024xf16>, tensor<1x1x1024x2048xf16>) -> tensor<16x7x7x2048xf16>
    %1145 = stablehlo.broadcast_in_dim %216, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1146 = stablehlo.broadcast_in_dim %217, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1147 = stablehlo.convert %1144 : (tensor<16x7x7x2048xf16>) -> tensor<16x7x7x2048xf32>
    %1148 = stablehlo.broadcast_in_dim %1145, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<16x7x7x2048xf32>
    %1149 = stablehlo.subtract %1147, %1148 : tensor<16x7x7x2048xf32>
    %1150 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1151 = stablehlo.broadcast_in_dim %1150, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %1152 = stablehlo.add %1146, %1151 : tensor<1x1x1x2048xf32>
    %1153 = stablehlo.rsqrt %1152 : tensor<1x1x1x2048xf32>
    %1154 = stablehlo.reshape %218 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1155 = stablehlo.multiply %1153, %1154 : tensor<1x1x1x2048xf32>
    %1156 = stablehlo.broadcast_in_dim %1155, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<16x7x7x2048xf32>
    %1157 = stablehlo.multiply %1149, %1156 : tensor<16x7x7x2048xf32>
    %1158 = stablehlo.reshape %219 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1159 = stablehlo.broadcast_in_dim %1158, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<16x7x7x2048xf32>
    %1160 = stablehlo.add %1157, %1159 : tensor<16x7x7x2048xf32>
    %1161 = stablehlo.convert %1160 : (tensor<16x7x7x2048xf32>) -> tensor<16x7x7x2048xf16>
    %1162 = stablehlo.convert %220 : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf16>
    %1163 = stablehlo.convolution(%1142, %1162) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x1024xf16>, tensor<1x1x1024x512xf16>) -> tensor<16x14x14x512xf16>
    %1164 = stablehlo.broadcast_in_dim %221, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1165 = stablehlo.broadcast_in_dim %222, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1166 = stablehlo.convert %1163 : (tensor<16x14x14x512xf16>) -> tensor<16x14x14x512xf32>
    %1167 = stablehlo.broadcast_in_dim %1164, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x14x14x512xf32>
    %1168 = stablehlo.subtract %1166, %1167 : tensor<16x14x14x512xf32>
    %1169 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1170 = stablehlo.broadcast_in_dim %1169, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1171 = stablehlo.add %1165, %1170 : tensor<1x1x1x512xf32>
    %1172 = stablehlo.rsqrt %1171 : tensor<1x1x1x512xf32>
    %1173 = stablehlo.reshape %223 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1174 = stablehlo.multiply %1172, %1173 : tensor<1x1x1x512xf32>
    %1175 = stablehlo.broadcast_in_dim %1174, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x14x14x512xf32>
    %1176 = stablehlo.multiply %1168, %1175 : tensor<16x14x14x512xf32>
    %1177 = stablehlo.reshape %224 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1178 = stablehlo.broadcast_in_dim %1177, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x14x14x512xf32>
    %1179 = stablehlo.add %1176, %1178 : tensor<16x14x14x512xf32>
    %1180 = stablehlo.convert %1179 : (tensor<16x14x14x512xf32>) -> tensor<16x14x14x512xf16>
    %1181 = call @relu_8(%1180) : (tensor<16x14x14x512xf16>) -> tensor<16x14x14x512xf16>
    %1182 = stablehlo.convert %225 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
    %1183 = stablehlo.convolution(%1181, %1182) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x14x14x512xf16>, tensor<3x3x512x512xf16>) -> tensor<16x7x7x512xf16>
    %1184 = stablehlo.broadcast_in_dim %226, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1185 = stablehlo.broadcast_in_dim %227, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1186 = stablehlo.convert %1183 : (tensor<16x7x7x512xf16>) -> tensor<16x7x7x512xf32>
    %1187 = stablehlo.broadcast_in_dim %1184, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1188 = stablehlo.subtract %1186, %1187 : tensor<16x7x7x512xf32>
    %1189 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1190 = stablehlo.broadcast_in_dim %1189, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1191 = stablehlo.add %1185, %1190 : tensor<1x1x1x512xf32>
    %1192 = stablehlo.rsqrt %1191 : tensor<1x1x1x512xf32>
    %1193 = stablehlo.reshape %228 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1194 = stablehlo.multiply %1192, %1193 : tensor<1x1x1x512xf32>
    %1195 = stablehlo.broadcast_in_dim %1194, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1196 = stablehlo.multiply %1188, %1195 : tensor<16x7x7x512xf32>
    %1197 = stablehlo.reshape %229 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1198 = stablehlo.broadcast_in_dim %1197, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1199 = stablehlo.add %1196, %1198 : tensor<16x7x7x512xf32>
    %1200 = stablehlo.convert %1199 : (tensor<16x7x7x512xf32>) -> tensor<16x7x7x512xf16>
    %1201 = call @relu_9(%1200) : (tensor<16x7x7x512xf16>) -> tensor<16x7x7x512xf16>
    %1202 = stablehlo.convert %230 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
    %1203 = stablehlo.convolution(%1201, %1202) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<16x7x7x2048xf16>
    %1204 = stablehlo.broadcast_in_dim %231, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1205 = stablehlo.broadcast_in_dim %232, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1206 = stablehlo.convert %1203 : (tensor<16x7x7x2048xf16>) -> tensor<16x7x7x2048xf32>
    %1207 = stablehlo.broadcast_in_dim %1204, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<16x7x7x2048xf32>
    %1208 = stablehlo.subtract %1206, %1207 : tensor<16x7x7x2048xf32>
    %1209 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1210 = stablehlo.broadcast_in_dim %1209, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %1211 = stablehlo.add %1205, %1210 : tensor<1x1x1x2048xf32>
    %1212 = stablehlo.rsqrt %1211 : tensor<1x1x1x2048xf32>
    %1213 = stablehlo.reshape %233 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1214 = stablehlo.multiply %1212, %1213 : tensor<1x1x1x2048xf32>
    %1215 = stablehlo.broadcast_in_dim %1214, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<16x7x7x2048xf32>
    %1216 = stablehlo.multiply %1208, %1215 : tensor<16x7x7x2048xf32>
    %1217 = stablehlo.reshape %234 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1218 = stablehlo.broadcast_in_dim %1217, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<16x7x7x2048xf32>
    %1219 = stablehlo.add %1216, %1218 : tensor<16x7x7x2048xf32>
    %1220 = stablehlo.convert %1219 : (tensor<16x7x7x2048xf32>) -> tensor<16x7x7x2048xf16>
    %1221 = stablehlo.add %1220, %1161 : tensor<16x7x7x2048xf16>
    %1222 = call @relu_10(%1221) : (tensor<16x7x7x2048xf16>) -> tensor<16x7x7x2048xf16>
    %1223 = stablehlo.convert %235 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf16>
    %1224 = stablehlo.convolution(%1222, %1223) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x7x7x2048xf16>, tensor<1x1x2048x512xf16>) -> tensor<16x7x7x512xf16>
    %1225 = stablehlo.broadcast_in_dim %236, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1226 = stablehlo.broadcast_in_dim %237, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1227 = stablehlo.convert %1224 : (tensor<16x7x7x512xf16>) -> tensor<16x7x7x512xf32>
    %1228 = stablehlo.broadcast_in_dim %1225, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1229 = stablehlo.subtract %1227, %1228 : tensor<16x7x7x512xf32>
    %1230 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1231 = stablehlo.broadcast_in_dim %1230, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1232 = stablehlo.add %1226, %1231 : tensor<1x1x1x512xf32>
    %1233 = stablehlo.rsqrt %1232 : tensor<1x1x1x512xf32>
    %1234 = stablehlo.reshape %238 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1235 = stablehlo.multiply %1233, %1234 : tensor<1x1x1x512xf32>
    %1236 = stablehlo.broadcast_in_dim %1235, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1237 = stablehlo.multiply %1229, %1236 : tensor<16x7x7x512xf32>
    %1238 = stablehlo.reshape %239 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1239 = stablehlo.broadcast_in_dim %1238, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1240 = stablehlo.add %1237, %1239 : tensor<16x7x7x512xf32>
    %1241 = stablehlo.convert %1240 : (tensor<16x7x7x512xf32>) -> tensor<16x7x7x512xf16>
    %1242 = call @relu_9(%1241) : (tensor<16x7x7x512xf16>) -> tensor<16x7x7x512xf16>
    %1243 = stablehlo.convert %240 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
    %1244 = stablehlo.convolution(%1242, %1243) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<16x7x7x512xf16>
    %1245 = stablehlo.broadcast_in_dim %241, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1246 = stablehlo.broadcast_in_dim %242, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1247 = stablehlo.convert %1244 : (tensor<16x7x7x512xf16>) -> tensor<16x7x7x512xf32>
    %1248 = stablehlo.broadcast_in_dim %1245, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1249 = stablehlo.subtract %1247, %1248 : tensor<16x7x7x512xf32>
    %1250 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1251 = stablehlo.broadcast_in_dim %1250, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1252 = stablehlo.add %1246, %1251 : tensor<1x1x1x512xf32>
    %1253 = stablehlo.rsqrt %1252 : tensor<1x1x1x512xf32>
    %1254 = stablehlo.reshape %243 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1255 = stablehlo.multiply %1253, %1254 : tensor<1x1x1x512xf32>
    %1256 = stablehlo.broadcast_in_dim %1255, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1257 = stablehlo.multiply %1249, %1256 : tensor<16x7x7x512xf32>
    %1258 = stablehlo.reshape %244 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1259 = stablehlo.broadcast_in_dim %1258, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1260 = stablehlo.add %1257, %1259 : tensor<16x7x7x512xf32>
    %1261 = stablehlo.convert %1260 : (tensor<16x7x7x512xf32>) -> tensor<16x7x7x512xf16>
    %1262 = call @relu_9(%1261) : (tensor<16x7x7x512xf16>) -> tensor<16x7x7x512xf16>
    %1263 = stablehlo.convert %245 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
    %1264 = stablehlo.convolution(%1262, %1263) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<16x7x7x2048xf16>
    %1265 = stablehlo.broadcast_in_dim %246, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1266 = stablehlo.broadcast_in_dim %247, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1267 = stablehlo.convert %1264 : (tensor<16x7x7x2048xf16>) -> tensor<16x7x7x2048xf32>
    %1268 = stablehlo.broadcast_in_dim %1265, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<16x7x7x2048xf32>
    %1269 = stablehlo.subtract %1267, %1268 : tensor<16x7x7x2048xf32>
    %1270 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1271 = stablehlo.broadcast_in_dim %1270, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %1272 = stablehlo.add %1266, %1271 : tensor<1x1x1x2048xf32>
    %1273 = stablehlo.rsqrt %1272 : tensor<1x1x1x2048xf32>
    %1274 = stablehlo.reshape %248 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1275 = stablehlo.multiply %1273, %1274 : tensor<1x1x1x2048xf32>
    %1276 = stablehlo.broadcast_in_dim %1275, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<16x7x7x2048xf32>
    %1277 = stablehlo.multiply %1269, %1276 : tensor<16x7x7x2048xf32>
    %1278 = stablehlo.reshape %249 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1279 = stablehlo.broadcast_in_dim %1278, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<16x7x7x2048xf32>
    %1280 = stablehlo.add %1277, %1279 : tensor<16x7x7x2048xf32>
    %1281 = stablehlo.convert %1280 : (tensor<16x7x7x2048xf32>) -> tensor<16x7x7x2048xf16>
    %1282 = stablehlo.add %1281, %1222 : tensor<16x7x7x2048xf16>
    %1283 = call @relu_10(%1282) : (tensor<16x7x7x2048xf16>) -> tensor<16x7x7x2048xf16>
    %1284 = stablehlo.convert %250 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf16>
    %1285 = stablehlo.convolution(%1283, %1284) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x7x7x2048xf16>, tensor<1x1x2048x512xf16>) -> tensor<16x7x7x512xf16>
    %1286 = stablehlo.broadcast_in_dim %251, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1287 = stablehlo.broadcast_in_dim %252, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1288 = stablehlo.convert %1285 : (tensor<16x7x7x512xf16>) -> tensor<16x7x7x512xf32>
    %1289 = stablehlo.broadcast_in_dim %1286, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1290 = stablehlo.subtract %1288, %1289 : tensor<16x7x7x512xf32>
    %1291 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1292 = stablehlo.broadcast_in_dim %1291, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1293 = stablehlo.add %1287, %1292 : tensor<1x1x1x512xf32>
    %1294 = stablehlo.rsqrt %1293 : tensor<1x1x1x512xf32>
    %1295 = stablehlo.reshape %253 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1296 = stablehlo.multiply %1294, %1295 : tensor<1x1x1x512xf32>
    %1297 = stablehlo.broadcast_in_dim %1296, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1298 = stablehlo.multiply %1290, %1297 : tensor<16x7x7x512xf32>
    %1299 = stablehlo.reshape %254 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1300 = stablehlo.broadcast_in_dim %1299, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1301 = stablehlo.add %1298, %1300 : tensor<16x7x7x512xf32>
    %1302 = stablehlo.convert %1301 : (tensor<16x7x7x512xf32>) -> tensor<16x7x7x512xf16>
    %1303 = call @relu_9(%1302) : (tensor<16x7x7x512xf16>) -> tensor<16x7x7x512xf16>
    %1304 = stablehlo.convert %255 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
    %1305 = stablehlo.convolution(%1303, %1304) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<16x7x7x512xf16>
    %1306 = stablehlo.broadcast_in_dim %256, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1307 = stablehlo.broadcast_in_dim %257, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1308 = stablehlo.convert %1305 : (tensor<16x7x7x512xf16>) -> tensor<16x7x7x512xf32>
    %1309 = stablehlo.broadcast_in_dim %1306, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1310 = stablehlo.subtract %1308, %1309 : tensor<16x7x7x512xf32>
    %1311 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1312 = stablehlo.broadcast_in_dim %1311, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1313 = stablehlo.add %1307, %1312 : tensor<1x1x1x512xf32>
    %1314 = stablehlo.rsqrt %1313 : tensor<1x1x1x512xf32>
    %1315 = stablehlo.reshape %258 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1316 = stablehlo.multiply %1314, %1315 : tensor<1x1x1x512xf32>
    %1317 = stablehlo.broadcast_in_dim %1316, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1318 = stablehlo.multiply %1310, %1317 : tensor<16x7x7x512xf32>
    %1319 = stablehlo.reshape %259 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1320 = stablehlo.broadcast_in_dim %1319, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<16x7x7x512xf32>
    %1321 = stablehlo.add %1318, %1320 : tensor<16x7x7x512xf32>
    %1322 = stablehlo.convert %1321 : (tensor<16x7x7x512xf32>) -> tensor<16x7x7x512xf16>
    %1323 = call @relu_9(%1322) : (tensor<16x7x7x512xf16>) -> tensor<16x7x7x512xf16>
    %1324 = stablehlo.convert %260 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
    %1325 = stablehlo.convolution(%1323, %1324) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<16x7x7x2048xf16>
    %1326 = stablehlo.broadcast_in_dim %261, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1327 = stablehlo.broadcast_in_dim %262, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1328 = stablehlo.convert %1325 : (tensor<16x7x7x2048xf16>) -> tensor<16x7x7x2048xf32>
    %1329 = stablehlo.broadcast_in_dim %1326, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<16x7x7x2048xf32>
    %1330 = stablehlo.subtract %1328, %1329 : tensor<16x7x7x2048xf32>
    %1331 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1332 = stablehlo.broadcast_in_dim %1331, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %1333 = stablehlo.add %1327, %1332 : tensor<1x1x1x2048xf32>
    %1334 = stablehlo.rsqrt %1333 : tensor<1x1x1x2048xf32>
    %1335 = stablehlo.reshape %263 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1336 = stablehlo.multiply %1334, %1335 : tensor<1x1x1x2048xf32>
    %1337 = stablehlo.broadcast_in_dim %1336, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<16x7x7x2048xf32>
    %1338 = stablehlo.multiply %1330, %1337 : tensor<16x7x7x2048xf32>
    %1339 = stablehlo.reshape %264 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1340 = stablehlo.broadcast_in_dim %1339, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<16x7x7x2048xf32>
    %1341 = stablehlo.add %1338, %1340 : tensor<16x7x7x2048xf32>
    %1342 = stablehlo.convert %1341 : (tensor<16x7x7x2048xf32>) -> tensor<16x7x7x2048xf16>
    %1343 = stablehlo.add %1342, %1283 : tensor<16x7x7x2048xf16>
    %1344 = call @relu_10(%1343) : (tensor<16x7x7x2048xf16>) -> tensor<16x7x7x2048xf16>
    %1345 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1346 = stablehlo.broadcast_in_dim %1345, dims = [] : (tensor<f16>) -> tensor<f16>
    %1347 = "stablehlo.reduce_window"(%1344, %1346) ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
      %1363 = stablehlo.add %arg1, %arg2 : tensor<f16>
      stablehlo.return %1363 : tensor<f16>
    }) {window_dimensions = array<i64: 1, 7, 7, 1>, window_strides = array<i64: 1, 7, 7, 1>} : (tensor<16x7x7x2048xf16>, tensor<f16>) -> tensor<16x1x1x2048xf16>
    %1348 = stablehlo.constant dense<49> : tensor<i32>
    %1349 = stablehlo.convert %1348 : (tensor<i32>) -> tensor<f16>
    %1350 = stablehlo.broadcast_in_dim %1349, dims = [] : (tensor<f16>) -> tensor<16x1x1x2048xf16>
    %1351 = stablehlo.divide %1347, %1350 : tensor<16x1x1x2048xf16>
    %1352 = stablehlo.transpose %1351, dims = [0, 3, 1, 2] : (tensor<16x1x1x2048xf16>) -> tensor<16x2048x1x1xf16>
    %1353 = stablehlo.slice %1352 [0:16, 0:2048, 0:1, 0:1] : (tensor<16x2048x1x1xf16>) -> tensor<16x2048x1x1xf16>
    %1354 = stablehlo.reshape %1353 : (tensor<16x2048x1x1xf16>) -> tensor<16x2048xf16>
    %1355 = stablehlo.convert %265 : (tensor<2048x1000xf32>) -> tensor<2048x1000xf16>
    %1356 = stablehlo.convert %266 : (tensor<1000xf32>) -> tensor<1000xf16>
    %1357 = stablehlo.convert %1354 : (tensor<16x2048xf16>) -> tensor<16x2048xf32>
    %1358 = stablehlo.convert %1355 : (tensor<2048x1000xf16>) -> tensor<2048x1000xf32>
    %1359 = stablehlo.dot_general %1357, %1358, contracting_dims = [1] x [0] : (tensor<16x2048xf32>, tensor<2048x1000xf32>) -> tensor<16x1000xf16>
    %1360 = stablehlo.reshape %1356 : (tensor<1000xf16>) -> tensor<1x1000xf16>
    %1361 = stablehlo.broadcast_in_dim %1360, dims = [0, 1] : (tensor<1x1000xf16>) -> tensor<16x1000xf16>
    %1362 = stablehlo.add %1359, %1361 : tensor<16x1000xf16>
    return %1362 : tensor<16x1000xf16>
  }
  func.func private @relu(%arg0: tensor<16x112x112x64xf16>) -> tensor<16x112x112x64xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<16x112x112x64xf16>
    %2 = stablehlo.maximum %arg0, %1 : tensor<16x112x112x64xf16>
    return %2 : tensor<16x112x112x64xf16>
  }
  func.func private @relu_0(%arg0: tensor<16x56x56x64xf16>) -> tensor<16x56x56x64xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<16x56x56x64xf16>
    %2 = stablehlo.maximum %arg0, %1 : tensor<16x56x56x64xf16>
    return %2 : tensor<16x56x56x64xf16>
  }
  func.func private @relu_1(%arg0: tensor<16x56x56x256xf16>) -> tensor<16x56x56x256xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<16x56x56x256xf16>
    %2 = stablehlo.maximum %arg0, %1 : tensor<16x56x56x256xf16>
    return %2 : tensor<16x56x56x256xf16>
  }
  func.func private @relu_2(%arg0: tensor<16x56x56x128xf16>) -> tensor<16x56x56x128xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<16x56x56x128xf16>
    %2 = stablehlo.maximum %arg0, %1 : tensor<16x56x56x128xf16>
    return %2 : tensor<16x56x56x128xf16>
  }
  func.func private @relu_3(%arg0: tensor<16x28x28x128xf16>) -> tensor<16x28x28x128xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<16x28x28x128xf16>
    %2 = stablehlo.maximum %arg0, %1 : tensor<16x28x28x128xf16>
    return %2 : tensor<16x28x28x128xf16>
  }
  func.func private @relu_4(%arg0: tensor<16x28x28x512xf16>) -> tensor<16x28x28x512xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<16x28x28x512xf16>
    %2 = stablehlo.maximum %arg0, %1 : tensor<16x28x28x512xf16>
    return %2 : tensor<16x28x28x512xf16>
  }
  func.func private @relu_5(%arg0: tensor<16x28x28x256xf16>) -> tensor<16x28x28x256xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<16x28x28x256xf16>
    %2 = stablehlo.maximum %arg0, %1 : tensor<16x28x28x256xf16>
    return %2 : tensor<16x28x28x256xf16>
  }
  func.func private @relu_6(%arg0: tensor<16x14x14x256xf16>) -> tensor<16x14x14x256xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<16x14x14x256xf16>
    %2 = stablehlo.maximum %arg0, %1 : tensor<16x14x14x256xf16>
    return %2 : tensor<16x14x14x256xf16>
  }
  func.func private @relu_7(%arg0: tensor<16x14x14x1024xf16>) -> tensor<16x14x14x1024xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<16x14x14x1024xf16>
    %2 = stablehlo.maximum %arg0, %1 : tensor<16x14x14x1024xf16>
    return %2 : tensor<16x14x14x1024xf16>
  }
  func.func private @relu_8(%arg0: tensor<16x14x14x512xf16>) -> tensor<16x14x14x512xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<16x14x14x512xf16>
    %2 = stablehlo.maximum %arg0, %1 : tensor<16x14x14x512xf16>
    return %2 : tensor<16x14x14x512xf16>
  }
  func.func private @relu_9(%arg0: tensor<16x7x7x512xf16>) -> tensor<16x7x7x512xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<16x7x7x512xf16>
    %2 = stablehlo.maximum %arg0, %1 : tensor<16x7x7x512xf16>
    return %2 : tensor<16x7x7x512xf16>
  }
  func.func private @relu_10(%arg0: tensor<16x7x7x2048xf16>) -> tensor<16x7x7x2048xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f16>) -> tensor<16x7x7x2048xf16>
    %2 = stablehlo.maximum %arg0, %1 : tensor<16x7x7x2048xf16>
    return %2 : tensor<16x7x7x2048xf16>
  }
}

