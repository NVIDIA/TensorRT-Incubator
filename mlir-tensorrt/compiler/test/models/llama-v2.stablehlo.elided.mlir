module @llama_v2 attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x27xf32> {mhlo.layout_mode = "default"}) -> (tensor<1x27x32000xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.constant dense_resource<__elided__> : tensor<32000x4096xf16>
    %1 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %2 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %3 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %4 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %5 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %6 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %7 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %8 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %9 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %10 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %11 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %12 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %13 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %14 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %15 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %16 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %17 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %18 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %19 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %20 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %21 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %22 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %23 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %24 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %25 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %26 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %27 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %28 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %29 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %30 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %31 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %32 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %33 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %34 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %35 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %36 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %37 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %38 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %39 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %40 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %41 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %42 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %43 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %44 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %45 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %46 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %47 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %48 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %49 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %50 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %51 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %52 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %53 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %54 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %55 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %56 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %57 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %58 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %59 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %60 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %61 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %62 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %63 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %64 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %65 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %66 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %67 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %68 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %69 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %70 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %71 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %72 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %73 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %74 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %75 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %76 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %77 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %78 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %79 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %80 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %81 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %82 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %83 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %84 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %85 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %86 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %87 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %88 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %89 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %90 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %91 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %92 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %93 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %94 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %95 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %96 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %97 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %98 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %99 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %100 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %101 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %102 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %103 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %104 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %105 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %106 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %107 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %108 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %109 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %110 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %111 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %112 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %113 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %114 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %115 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %116 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %117 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %118 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %119 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %120 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %121 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %122 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %123 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %124 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %125 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %126 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %127 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %128 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %129 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %130 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %131 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %132 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %133 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %134 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %135 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %136 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %137 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %138 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %139 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %140 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %141 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %142 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %143 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %144 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %145 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %146 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %147 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %148 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %149 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %150 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %151 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %152 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %153 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %154 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %155 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %156 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %157 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %158 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %159 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %160 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %161 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %162 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %163 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %164 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %165 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %166 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %167 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %168 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %169 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %170 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %171 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %172 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %173 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %174 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %175 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %176 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %177 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %178 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %179 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %180 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %181 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %182 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %183 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %184 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %185 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %186 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %187 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %188 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %189 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %190 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %191 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %192 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %193 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %194 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %195 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %196 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %197 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %198 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %199 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %200 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %201 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %202 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %203 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %204 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %205 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %206 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %207 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %208 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %209 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %210 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %211 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %212 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %213 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %214 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %215 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %216 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %217 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %218 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %219 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %220 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %221 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %222 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %223 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %224 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %225 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %226 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %227 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %228 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %229 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %230 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %231 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %232 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %233 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %234 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %235 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %236 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %237 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %238 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %239 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %240 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %241 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %242 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %243 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %244 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %245 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %246 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %247 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %248 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %249 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %250 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %251 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %252 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %253 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %254 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %255 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %256 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %257 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %258 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %259 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %260 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %261 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %262 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %263 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %264 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %265 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %266 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %267 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %268 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %269 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %270 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %271 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %272 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %273 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %274 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %275 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %276 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %277 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %278 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %279 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %280 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %281 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %282 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %283 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %284 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %285 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %286 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %287 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %288 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %289 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %290 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %291 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %292 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %293 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %294 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %295 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %296 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %297 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %298 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %299 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %300 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %301 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %302 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %303 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %304 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %305 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %306 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %307 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %308 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %309 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %310 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %311 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %312 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %313 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %314 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %315 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1x256xf32>
    %316 = stablehlo.constant dense_resource<__elided__> : tensor<4096x4096xf16>
    %317 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %318 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %319 = stablehlo.constant dense_resource<__elided__> : tensor<4096x11008xf16>
    %320 = stablehlo.constant dense_resource<__elided__> : tensor<11008x4096xf16>
    %321 = stablehlo.constant dense_resource<__elided__> : tensor<4096xf16>
    %322 = stablehlo.constant dense_resource<__elided__> : tensor<4096x32000xf16>
    %323 = stablehlo.iota dim = 0 : tensor<27xi32>
    %324 = stablehlo.broadcast_in_dim %323, dims = [1] : (tensor<27xi32>) -> tensor<1x27xi32>
    %325 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %326 = stablehlo.broadcast_in_dim %325, dims = [] : (tensor<f32>) -> tensor<1x27xf32>
    %327 = stablehlo.convert %arg0 : (tensor<1x27xf32>) -> tensor<1x27xi32>
    %328 = stablehlo.convert %326 : (tensor<1x27xf32>) -> tensor<1x27xi32>
    %329 = stablehlo.convert %0 : (tensor<32000x4096xf16>) -> tensor<32000x4096xf32>
    %330 = call @_take(%329, %327) : (tensor<32000x4096xf32>, tensor<1x27xi32>) -> tensor<1x27x4096xf32>
    %331 = stablehlo.multiply %330, %330 : tensor<1x27x4096xf32>
    %332 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %333 = stablehlo.reduce(%331 init: %332) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %334 = stablehlo.broadcast_in_dim %333, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %335 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %336 = stablehlo.broadcast_in_dim %335, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %337 = stablehlo.divide %334, %336 : tensor<1x27x1xf32>
    %338 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %339 = stablehlo.broadcast_in_dim %338, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %340 = stablehlo.add %337, %339 : tensor<1x27x1xf32>
    %341 = stablehlo.sqrt %340 : tensor<1x27x1xf32>
    %342 = stablehlo.broadcast_in_dim %341, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %343 = stablehlo.divide %330, %342 : tensor<1x27x4096xf32>
    %344 = stablehlo.convert %1 : (tensor<4096xf16>) -> tensor<4096xf32>
    %345 = stablehlo.broadcast_in_dim %344, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %347 = stablehlo.multiply %346, %343 : tensor<1x27x4096xf32>
    %348 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %349 = stablehlo.broadcast_in_dim %348, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %350 = stablehlo.broadcast_in_dim %349, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %351 = stablehlo.broadcast_in_dim %349, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %352 = stablehlo.broadcast_in_dim %350, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %353 = stablehlo.broadcast_in_dim %351, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %354 = stablehlo.compare  GE, %352, %353,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %355 = stablehlo.broadcast_in_dim %354, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %356 = stablehlo.convert %2 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %357 = stablehlo.dot_general %347, %356, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %358 = stablehlo.convert %3 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %359 = stablehlo.dot_general %347, %358, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %360 = stablehlo.convert %4 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %361 = stablehlo.dot_general %347, %360, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %362 = stablehlo.reshape %357 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %363 = stablehlo.reshape %359 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %364 = stablehlo.reshape %361 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %365 = stablehlo.constant dense<0> : tensor<i32>
    %366 = stablehlo.broadcast_in_dim %365, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %367 = stablehlo.compare  LT, %324, %366,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %368 = stablehlo.constant dense<4096> : tensor<i32>
    %369 = stablehlo.broadcast_in_dim %368, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %370 = stablehlo.add %324, %369 : tensor<1x27xi32>
    %371 = stablehlo.select %367, %370, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %372 = stablehlo.broadcast_in_dim %371, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %373 = "stablehlo.gather"(%5, %372) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %374 = stablehlo.slice %373 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %375 = stablehlo.slice %373 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %376 = stablehlo.broadcast_in_dim %375, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %377 = stablehlo.multiply %363, %376 : tensor<1x27x32x128xf32>
    %378 = stablehlo.constant dense<64> : tensor<i32>
    %379 = stablehlo.broadcast_in_dim %378, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %380 = "stablehlo.gather"(%363, %379) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %381 = stablehlo.negate %380 : tensor<1x27x32x64xf32>
    %382 = stablehlo.constant dense<0> : tensor<i32>
    %383 = stablehlo.broadcast_in_dim %382, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %384 = "stablehlo.gather"(%363, %383) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %385 = stablehlo.concatenate %381, %384, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %386 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %387 = stablehlo.multiply %385, %386 : tensor<1x27x32x128xf32>
    %388 = stablehlo.add %377, %387 : tensor<1x27x32x128xf32>
    %389 = stablehlo.broadcast_in_dim %375, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %390 = stablehlo.multiply %362, %389 : tensor<1x27x32x128xf32>
    %391 = stablehlo.constant dense<64> : tensor<i32>
    %392 = stablehlo.broadcast_in_dim %391, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %393 = "stablehlo.gather"(%362, %392) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %394 = stablehlo.negate %393 : tensor<1x27x32x64xf32>
    %395 = stablehlo.constant dense<0> : tensor<i32>
    %396 = stablehlo.broadcast_in_dim %395, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %397 = "stablehlo.gather"(%362, %396) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %398 = stablehlo.concatenate %394, %397, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %399 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %400 = stablehlo.multiply %398, %399 : tensor<1x27x32x128xf32>
    %401 = stablehlo.add %390, %400 : tensor<1x27x32x128xf32>
    %402 = stablehlo.slice %355 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %403 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %404 = stablehlo.reshape %403 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %405 = stablehlo.broadcast_in_dim %404, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %406 = stablehlo.constant dense<0> : tensor<i32>
    %407 = stablehlo.broadcast_in_dim %406, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %408 = stablehlo.compare  NE, %405, %407,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %409 = stablehlo.and %408, %402 : tensor<1x1x27x27xi1>
    %410 = stablehlo.convert %409 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %411 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %412 = stablehlo.broadcast_in_dim %411, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %413 = stablehlo.compare  GT, %410, %412,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %414 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %415 = stablehlo.broadcast_in_dim %414, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %416 = stablehlo.convert %415 : tensor<1x1x27x27xf32>
    %417 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %418 = stablehlo.broadcast_in_dim %417, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %419 = stablehlo.select %413, %416, %418 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %420 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %421 = stablehlo.sqrt %420 : tensor<f32>
    %422 = stablehlo.convert %421 : tensor<f32>
    %423 = stablehlo.broadcast_in_dim %422, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %424 = stablehlo.divide %401, %423 : tensor<1x27x32x128xf32>
    %425 = stablehlo.dot_general %424, %388, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %426 = stablehlo.broadcast_in_dim %419, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %427 = stablehlo.add %425, %426 : tensor<1x32x27x27xf32>
    %428 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %429 = stablehlo.reduce(%427 init: %428) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %430 = stablehlo.broadcast_in_dim %429, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %432 = stablehlo.subtract %427, %431 : tensor<1x32x27x27xf32>
    %433 = stablehlo.exponential %432 : tensor<1x32x27x27xf32>
    %434 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %435 = stablehlo.reduce(%433 init: %434) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %438 = stablehlo.divide %433, %437 : tensor<1x32x27x27xf32>
    %439 = stablehlo.dot_general %364, %438, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %440 = stablehlo.transpose %439, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %441 = stablehlo.reshape %440 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %442 = stablehlo.convert %6 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %443 = stablehlo.dot_general %441, %442, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %444 = stablehlo.add %330, %443 : tensor<1x27x4096xf32>
    %445 = stablehlo.multiply %444, %444 : tensor<1x27x4096xf32>
    %446 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %447 = stablehlo.reduce(%445 init: %446) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %448 = stablehlo.broadcast_in_dim %447, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %449 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %450 = stablehlo.broadcast_in_dim %449, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %451 = stablehlo.divide %448, %450 : tensor<1x27x1xf32>
    %452 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %453 = stablehlo.broadcast_in_dim %452, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %454 = stablehlo.add %451, %453 : tensor<1x27x1xf32>
    %455 = stablehlo.sqrt %454 : tensor<1x27x1xf32>
    %456 = stablehlo.broadcast_in_dim %455, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %457 = stablehlo.divide %444, %456 : tensor<1x27x4096xf32>
    %458 = stablehlo.convert %7 : (tensor<4096xf16>) -> tensor<4096xf32>
    %459 = stablehlo.broadcast_in_dim %458, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %461 = stablehlo.multiply %460, %457 : tensor<1x27x4096xf32>
    %462 = stablehlo.convert %8 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %463 = stablehlo.dot_general %461, %462, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %464 = stablehlo.convert %9 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %465 = stablehlo.dot_general %461, %464, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %466 = call @silu(%465) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %467 = stablehlo.multiply %463, %466 : tensor<1x27x11008xf32>
    %468 = stablehlo.convert %10 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %469 = stablehlo.dot_general %467, %468, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %470 = stablehlo.add %444, %469 : tensor<1x27x4096xf32>
    %471 = stablehlo.multiply %470, %470 : tensor<1x27x4096xf32>
    %472 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %473 = stablehlo.reduce(%471 init: %472) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %474 = stablehlo.broadcast_in_dim %473, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %475 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %476 = stablehlo.broadcast_in_dim %475, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %477 = stablehlo.divide %474, %476 : tensor<1x27x1xf32>
    %478 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %479 = stablehlo.broadcast_in_dim %478, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %480 = stablehlo.add %477, %479 : tensor<1x27x1xf32>
    %481 = stablehlo.sqrt %480 : tensor<1x27x1xf32>
    %482 = stablehlo.broadcast_in_dim %481, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %483 = stablehlo.divide %470, %482 : tensor<1x27x4096xf32>
    %484 = stablehlo.convert %11 : (tensor<4096xf16>) -> tensor<4096xf32>
    %485 = stablehlo.broadcast_in_dim %484, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %486 = stablehlo.broadcast_in_dim %485, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %487 = stablehlo.multiply %486, %483 : tensor<1x27x4096xf32>
    %488 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %489 = stablehlo.broadcast_in_dim %488, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %490 = stablehlo.broadcast_in_dim %489, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %491 = stablehlo.broadcast_in_dim %489, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %492 = stablehlo.broadcast_in_dim %490, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %493 = stablehlo.broadcast_in_dim %491, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %494 = stablehlo.compare  GE, %492, %493,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %495 = stablehlo.broadcast_in_dim %494, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %496 = stablehlo.convert %12 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %497 = stablehlo.dot_general %487, %496, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %498 = stablehlo.convert %13 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %499 = stablehlo.dot_general %487, %498, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %500 = stablehlo.convert %14 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %501 = stablehlo.dot_general %487, %500, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %502 = stablehlo.reshape %497 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %503 = stablehlo.reshape %499 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %504 = stablehlo.reshape %501 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %505 = stablehlo.constant dense<0> : tensor<i32>
    %506 = stablehlo.broadcast_in_dim %505, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %507 = stablehlo.compare  LT, %324, %506,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %508 = stablehlo.constant dense<4096> : tensor<i32>
    %509 = stablehlo.broadcast_in_dim %508, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %510 = stablehlo.add %324, %509 : tensor<1x27xi32>
    %511 = stablehlo.select %507, %510, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %512 = stablehlo.broadcast_in_dim %511, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %513 = "stablehlo.gather"(%15, %512) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %514 = stablehlo.slice %513 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %515 = stablehlo.slice %513 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %516 = stablehlo.broadcast_in_dim %515, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %517 = stablehlo.multiply %503, %516 : tensor<1x27x32x128xf32>
    %518 = stablehlo.constant dense<64> : tensor<i32>
    %519 = stablehlo.broadcast_in_dim %518, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %520 = "stablehlo.gather"(%503, %519) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %521 = stablehlo.negate %520 : tensor<1x27x32x64xf32>
    %522 = stablehlo.constant dense<0> : tensor<i32>
    %523 = stablehlo.broadcast_in_dim %522, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %524 = "stablehlo.gather"(%503, %523) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %525 = stablehlo.concatenate %521, %524, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %526 = stablehlo.broadcast_in_dim %514, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %527 = stablehlo.multiply %525, %526 : tensor<1x27x32x128xf32>
    %528 = stablehlo.add %517, %527 : tensor<1x27x32x128xf32>
    %529 = stablehlo.broadcast_in_dim %515, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %530 = stablehlo.multiply %502, %529 : tensor<1x27x32x128xf32>
    %531 = stablehlo.constant dense<64> : tensor<i32>
    %532 = stablehlo.broadcast_in_dim %531, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %533 = "stablehlo.gather"(%502, %532) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %534 = stablehlo.negate %533 : tensor<1x27x32x64xf32>
    %535 = stablehlo.constant dense<0> : tensor<i32>
    %536 = stablehlo.broadcast_in_dim %535, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %537 = "stablehlo.gather"(%502, %536) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %538 = stablehlo.concatenate %534, %537, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %539 = stablehlo.broadcast_in_dim %514, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %540 = stablehlo.multiply %538, %539 : tensor<1x27x32x128xf32>
    %541 = stablehlo.add %530, %540 : tensor<1x27x32x128xf32>
    %542 = stablehlo.slice %495 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %543 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %544 = stablehlo.reshape %543 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %545 = stablehlo.broadcast_in_dim %544, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %546 = stablehlo.constant dense<0> : tensor<i32>
    %547 = stablehlo.broadcast_in_dim %546, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %548 = stablehlo.compare  NE, %545, %547,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %549 = stablehlo.and %548, %542 : tensor<1x1x27x27xi1>
    %550 = stablehlo.convert %549 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %551 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %552 = stablehlo.broadcast_in_dim %551, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %553 = stablehlo.compare  GT, %550, %552,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %554 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %555 = stablehlo.broadcast_in_dim %554, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %556 = stablehlo.convert %555 : tensor<1x1x27x27xf32>
    %557 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %558 = stablehlo.broadcast_in_dim %557, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %559 = stablehlo.select %553, %556, %558 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %560 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %561 = stablehlo.sqrt %560 : tensor<f32>
    %562 = stablehlo.convert %561 : tensor<f32>
    %563 = stablehlo.broadcast_in_dim %562, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %564 = stablehlo.divide %541, %563 : tensor<1x27x32x128xf32>
    %565 = stablehlo.dot_general %564, %528, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %566 = stablehlo.broadcast_in_dim %559, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %567 = stablehlo.add %565, %566 : tensor<1x32x27x27xf32>
    %568 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %569 = stablehlo.reduce(%567 init: %568) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %570 = stablehlo.broadcast_in_dim %569, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %571 = stablehlo.broadcast_in_dim %570, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %572 = stablehlo.subtract %567, %571 : tensor<1x32x27x27xf32>
    %573 = stablehlo.exponential %572 : tensor<1x32x27x27xf32>
    %574 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %575 = stablehlo.reduce(%573 init: %574) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %576 = stablehlo.broadcast_in_dim %575, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %577 = stablehlo.broadcast_in_dim %576, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %578 = stablehlo.divide %573, %577 : tensor<1x32x27x27xf32>
    %579 = stablehlo.dot_general %504, %578, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %580 = stablehlo.transpose %579, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %581 = stablehlo.reshape %580 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %582 = stablehlo.convert %16 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %583 = stablehlo.dot_general %581, %582, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %584 = stablehlo.add %470, %583 : tensor<1x27x4096xf32>
    %585 = stablehlo.multiply %584, %584 : tensor<1x27x4096xf32>
    %586 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %587 = stablehlo.reduce(%585 init: %586) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %588 = stablehlo.broadcast_in_dim %587, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %589 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %590 = stablehlo.broadcast_in_dim %589, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %591 = stablehlo.divide %588, %590 : tensor<1x27x1xf32>
    %592 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %593 = stablehlo.broadcast_in_dim %592, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %594 = stablehlo.add %591, %593 : tensor<1x27x1xf32>
    %595 = stablehlo.sqrt %594 : tensor<1x27x1xf32>
    %596 = stablehlo.broadcast_in_dim %595, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %597 = stablehlo.divide %584, %596 : tensor<1x27x4096xf32>
    %598 = stablehlo.convert %17 : (tensor<4096xf16>) -> tensor<4096xf32>
    %599 = stablehlo.broadcast_in_dim %598, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %600 = stablehlo.broadcast_in_dim %599, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %601 = stablehlo.multiply %600, %597 : tensor<1x27x4096xf32>
    %602 = stablehlo.convert %18 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %603 = stablehlo.dot_general %601, %602, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %604 = stablehlo.convert %19 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %605 = stablehlo.dot_general %601, %604, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %606 = call @silu(%605) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %607 = stablehlo.multiply %603, %606 : tensor<1x27x11008xf32>
    %608 = stablehlo.convert %20 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %609 = stablehlo.dot_general %607, %608, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %610 = stablehlo.add %584, %609 : tensor<1x27x4096xf32>
    %611 = stablehlo.multiply %610, %610 : tensor<1x27x4096xf32>
    %612 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %613 = stablehlo.reduce(%611 init: %612) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %614 = stablehlo.broadcast_in_dim %613, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %615 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %616 = stablehlo.broadcast_in_dim %615, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %617 = stablehlo.divide %614, %616 : tensor<1x27x1xf32>
    %618 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %619 = stablehlo.broadcast_in_dim %618, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %620 = stablehlo.add %617, %619 : tensor<1x27x1xf32>
    %621 = stablehlo.sqrt %620 : tensor<1x27x1xf32>
    %622 = stablehlo.broadcast_in_dim %621, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %623 = stablehlo.divide %610, %622 : tensor<1x27x4096xf32>
    %624 = stablehlo.convert %21 : (tensor<4096xf16>) -> tensor<4096xf32>
    %625 = stablehlo.broadcast_in_dim %624, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %626 = stablehlo.broadcast_in_dim %625, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %627 = stablehlo.multiply %626, %623 : tensor<1x27x4096xf32>
    %628 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %629 = stablehlo.broadcast_in_dim %628, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %630 = stablehlo.broadcast_in_dim %629, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %631 = stablehlo.broadcast_in_dim %629, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %632 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %633 = stablehlo.broadcast_in_dim %631, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %634 = stablehlo.compare  GE, %632, %633,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %635 = stablehlo.broadcast_in_dim %634, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %636 = stablehlo.convert %22 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %637 = stablehlo.dot_general %627, %636, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %638 = stablehlo.convert %23 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %639 = stablehlo.dot_general %627, %638, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %640 = stablehlo.convert %24 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %641 = stablehlo.dot_general %627, %640, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %642 = stablehlo.reshape %637 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %643 = stablehlo.reshape %639 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %644 = stablehlo.reshape %641 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %645 = stablehlo.constant dense<0> : tensor<i32>
    %646 = stablehlo.broadcast_in_dim %645, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %647 = stablehlo.compare  LT, %324, %646,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %648 = stablehlo.constant dense<4096> : tensor<i32>
    %649 = stablehlo.broadcast_in_dim %648, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %650 = stablehlo.add %324, %649 : tensor<1x27xi32>
    %651 = stablehlo.select %647, %650, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %652 = stablehlo.broadcast_in_dim %651, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %653 = "stablehlo.gather"(%25, %652) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %654 = stablehlo.slice %653 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %655 = stablehlo.slice %653 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %656 = stablehlo.broadcast_in_dim %655, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %657 = stablehlo.multiply %643, %656 : tensor<1x27x32x128xf32>
    %658 = stablehlo.constant dense<64> : tensor<i32>
    %659 = stablehlo.broadcast_in_dim %658, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %660 = "stablehlo.gather"(%643, %659) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %661 = stablehlo.negate %660 : tensor<1x27x32x64xf32>
    %662 = stablehlo.constant dense<0> : tensor<i32>
    %663 = stablehlo.broadcast_in_dim %662, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %664 = "stablehlo.gather"(%643, %663) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %665 = stablehlo.concatenate %661, %664, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %666 = stablehlo.broadcast_in_dim %654, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %667 = stablehlo.multiply %665, %666 : tensor<1x27x32x128xf32>
    %668 = stablehlo.add %657, %667 : tensor<1x27x32x128xf32>
    %669 = stablehlo.broadcast_in_dim %655, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %670 = stablehlo.multiply %642, %669 : tensor<1x27x32x128xf32>
    %671 = stablehlo.constant dense<64> : tensor<i32>
    %672 = stablehlo.broadcast_in_dim %671, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %673 = "stablehlo.gather"(%642, %672) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %674 = stablehlo.negate %673 : tensor<1x27x32x64xf32>
    %675 = stablehlo.constant dense<0> : tensor<i32>
    %676 = stablehlo.broadcast_in_dim %675, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %677 = "stablehlo.gather"(%642, %676) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %678 = stablehlo.concatenate %674, %677, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %679 = stablehlo.broadcast_in_dim %654, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %680 = stablehlo.multiply %678, %679 : tensor<1x27x32x128xf32>
    %681 = stablehlo.add %670, %680 : tensor<1x27x32x128xf32>
    %682 = stablehlo.slice %635 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %683 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %684 = stablehlo.reshape %683 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %685 = stablehlo.broadcast_in_dim %684, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %686 = stablehlo.constant dense<0> : tensor<i32>
    %687 = stablehlo.broadcast_in_dim %686, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %688 = stablehlo.compare  NE, %685, %687,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %689 = stablehlo.and %688, %682 : tensor<1x1x27x27xi1>
    %690 = stablehlo.convert %689 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %691 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %692 = stablehlo.broadcast_in_dim %691, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %693 = stablehlo.compare  GT, %690, %692,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %694 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %695 = stablehlo.broadcast_in_dim %694, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %696 = stablehlo.convert %695 : tensor<1x1x27x27xf32>
    %697 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %698 = stablehlo.broadcast_in_dim %697, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %699 = stablehlo.select %693, %696, %698 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %700 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %701 = stablehlo.sqrt %700 : tensor<f32>
    %702 = stablehlo.convert %701 : tensor<f32>
    %703 = stablehlo.broadcast_in_dim %702, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %704 = stablehlo.divide %681, %703 : tensor<1x27x32x128xf32>
    %705 = stablehlo.dot_general %704, %668, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %706 = stablehlo.broadcast_in_dim %699, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %707 = stablehlo.add %705, %706 : tensor<1x32x27x27xf32>
    %708 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %709 = stablehlo.reduce(%707 init: %708) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %710 = stablehlo.broadcast_in_dim %709, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %711 = stablehlo.broadcast_in_dim %710, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %712 = stablehlo.subtract %707, %711 : tensor<1x32x27x27xf32>
    %713 = stablehlo.exponential %712 : tensor<1x32x27x27xf32>
    %714 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %715 = stablehlo.reduce(%713 init: %714) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %716 = stablehlo.broadcast_in_dim %715, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %717 = stablehlo.broadcast_in_dim %716, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %718 = stablehlo.divide %713, %717 : tensor<1x32x27x27xf32>
    %719 = stablehlo.dot_general %644, %718, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %720 = stablehlo.transpose %719, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %721 = stablehlo.reshape %720 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %722 = stablehlo.convert %26 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %723 = stablehlo.dot_general %721, %722, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %724 = stablehlo.add %610, %723 : tensor<1x27x4096xf32>
    %725 = stablehlo.multiply %724, %724 : tensor<1x27x4096xf32>
    %726 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %727 = stablehlo.reduce(%725 init: %726) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %728 = stablehlo.broadcast_in_dim %727, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %729 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %730 = stablehlo.broadcast_in_dim %729, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %731 = stablehlo.divide %728, %730 : tensor<1x27x1xf32>
    %732 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %733 = stablehlo.broadcast_in_dim %732, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %734 = stablehlo.add %731, %733 : tensor<1x27x1xf32>
    %735 = stablehlo.sqrt %734 : tensor<1x27x1xf32>
    %736 = stablehlo.broadcast_in_dim %735, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %737 = stablehlo.divide %724, %736 : tensor<1x27x4096xf32>
    %738 = stablehlo.convert %27 : (tensor<4096xf16>) -> tensor<4096xf32>
    %739 = stablehlo.broadcast_in_dim %738, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %740 = stablehlo.broadcast_in_dim %739, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %741 = stablehlo.multiply %740, %737 : tensor<1x27x4096xf32>
    %742 = stablehlo.convert %28 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %743 = stablehlo.dot_general %741, %742, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %744 = stablehlo.convert %29 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %745 = stablehlo.dot_general %741, %744, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %746 = call @silu(%745) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %747 = stablehlo.multiply %743, %746 : tensor<1x27x11008xf32>
    %748 = stablehlo.convert %30 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %749 = stablehlo.dot_general %747, %748, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %750 = stablehlo.add %724, %749 : tensor<1x27x4096xf32>
    %751 = stablehlo.multiply %750, %750 : tensor<1x27x4096xf32>
    %752 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %753 = stablehlo.reduce(%751 init: %752) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %754 = stablehlo.broadcast_in_dim %753, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %755 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %756 = stablehlo.broadcast_in_dim %755, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %757 = stablehlo.divide %754, %756 : tensor<1x27x1xf32>
    %758 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %759 = stablehlo.broadcast_in_dim %758, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %760 = stablehlo.add %757, %759 : tensor<1x27x1xf32>
    %761 = stablehlo.sqrt %760 : tensor<1x27x1xf32>
    %762 = stablehlo.broadcast_in_dim %761, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %763 = stablehlo.divide %750, %762 : tensor<1x27x4096xf32>
    %764 = stablehlo.convert %31 : (tensor<4096xf16>) -> tensor<4096xf32>
    %765 = stablehlo.broadcast_in_dim %764, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %766 = stablehlo.broadcast_in_dim %765, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %767 = stablehlo.multiply %766, %763 : tensor<1x27x4096xf32>
    %768 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %769 = stablehlo.broadcast_in_dim %768, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %770 = stablehlo.broadcast_in_dim %769, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %771 = stablehlo.broadcast_in_dim %769, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %772 = stablehlo.broadcast_in_dim %770, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %773 = stablehlo.broadcast_in_dim %771, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %774 = stablehlo.compare  GE, %772, %773,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %775 = stablehlo.broadcast_in_dim %774, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %776 = stablehlo.convert %32 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %777 = stablehlo.dot_general %767, %776, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %778 = stablehlo.convert %33 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %779 = stablehlo.dot_general %767, %778, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %780 = stablehlo.convert %34 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %781 = stablehlo.dot_general %767, %780, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %782 = stablehlo.reshape %777 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %783 = stablehlo.reshape %779 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %784 = stablehlo.reshape %781 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %785 = stablehlo.constant dense<0> : tensor<i32>
    %786 = stablehlo.broadcast_in_dim %785, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %787 = stablehlo.compare  LT, %324, %786,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %788 = stablehlo.constant dense<4096> : tensor<i32>
    %789 = stablehlo.broadcast_in_dim %788, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %790 = stablehlo.add %324, %789 : tensor<1x27xi32>
    %791 = stablehlo.select %787, %790, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %792 = stablehlo.broadcast_in_dim %791, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %793 = "stablehlo.gather"(%35, %792) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %794 = stablehlo.slice %793 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %795 = stablehlo.slice %793 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %796 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %797 = stablehlo.multiply %783, %796 : tensor<1x27x32x128xf32>
    %798 = stablehlo.constant dense<64> : tensor<i32>
    %799 = stablehlo.broadcast_in_dim %798, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %800 = "stablehlo.gather"(%783, %799) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %801 = stablehlo.negate %800 : tensor<1x27x32x64xf32>
    %802 = stablehlo.constant dense<0> : tensor<i32>
    %803 = stablehlo.broadcast_in_dim %802, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %804 = "stablehlo.gather"(%783, %803) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %805 = stablehlo.concatenate %801, %804, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %806 = stablehlo.broadcast_in_dim %794, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %807 = stablehlo.multiply %805, %806 : tensor<1x27x32x128xf32>
    %808 = stablehlo.add %797, %807 : tensor<1x27x32x128xf32>
    %809 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %810 = stablehlo.multiply %782, %809 : tensor<1x27x32x128xf32>
    %811 = stablehlo.constant dense<64> : tensor<i32>
    %812 = stablehlo.broadcast_in_dim %811, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %813 = "stablehlo.gather"(%782, %812) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %814 = stablehlo.negate %813 : tensor<1x27x32x64xf32>
    %815 = stablehlo.constant dense<0> : tensor<i32>
    %816 = stablehlo.broadcast_in_dim %815, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %817 = "stablehlo.gather"(%782, %816) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %818 = stablehlo.concatenate %814, %817, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %819 = stablehlo.broadcast_in_dim %794, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %820 = stablehlo.multiply %818, %819 : tensor<1x27x32x128xf32>
    %821 = stablehlo.add %810, %820 : tensor<1x27x32x128xf32>
    %822 = stablehlo.slice %775 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %823 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %824 = stablehlo.reshape %823 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %825 = stablehlo.broadcast_in_dim %824, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %826 = stablehlo.constant dense<0> : tensor<i32>
    %827 = stablehlo.broadcast_in_dim %826, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %828 = stablehlo.compare  NE, %825, %827,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %829 = stablehlo.and %828, %822 : tensor<1x1x27x27xi1>
    %830 = stablehlo.convert %829 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %831 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %832 = stablehlo.broadcast_in_dim %831, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %833 = stablehlo.compare  GT, %830, %832,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %834 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %835 = stablehlo.broadcast_in_dim %834, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %836 = stablehlo.convert %835 : tensor<1x1x27x27xf32>
    %837 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %838 = stablehlo.broadcast_in_dim %837, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %839 = stablehlo.select %833, %836, %838 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %840 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %841 = stablehlo.sqrt %840 : tensor<f32>
    %842 = stablehlo.convert %841 : tensor<f32>
    %843 = stablehlo.broadcast_in_dim %842, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %844 = stablehlo.divide %821, %843 : tensor<1x27x32x128xf32>
    %845 = stablehlo.dot_general %844, %808, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %846 = stablehlo.broadcast_in_dim %839, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %847 = stablehlo.add %845, %846 : tensor<1x32x27x27xf32>
    %848 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %849 = stablehlo.reduce(%847 init: %848) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %850 = stablehlo.broadcast_in_dim %849, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %851 = stablehlo.broadcast_in_dim %850, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %852 = stablehlo.subtract %847, %851 : tensor<1x32x27x27xf32>
    %853 = stablehlo.exponential %852 : tensor<1x32x27x27xf32>
    %854 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %855 = stablehlo.reduce(%853 init: %854) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %856 = stablehlo.broadcast_in_dim %855, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %857 = stablehlo.broadcast_in_dim %856, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %858 = stablehlo.divide %853, %857 : tensor<1x32x27x27xf32>
    %859 = stablehlo.dot_general %784, %858, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %860 = stablehlo.transpose %859, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %861 = stablehlo.reshape %860 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %862 = stablehlo.convert %36 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %863 = stablehlo.dot_general %861, %862, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %864 = stablehlo.add %750, %863 : tensor<1x27x4096xf32>
    %865 = stablehlo.multiply %864, %864 : tensor<1x27x4096xf32>
    %866 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %867 = stablehlo.reduce(%865 init: %866) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %868 = stablehlo.broadcast_in_dim %867, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %869 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %870 = stablehlo.broadcast_in_dim %869, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %871 = stablehlo.divide %868, %870 : tensor<1x27x1xf32>
    %872 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %873 = stablehlo.broadcast_in_dim %872, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %874 = stablehlo.add %871, %873 : tensor<1x27x1xf32>
    %875 = stablehlo.sqrt %874 : tensor<1x27x1xf32>
    %876 = stablehlo.broadcast_in_dim %875, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %877 = stablehlo.divide %864, %876 : tensor<1x27x4096xf32>
    %878 = stablehlo.convert %37 : (tensor<4096xf16>) -> tensor<4096xf32>
    %879 = stablehlo.broadcast_in_dim %878, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %880 = stablehlo.broadcast_in_dim %879, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %881 = stablehlo.multiply %880, %877 : tensor<1x27x4096xf32>
    %882 = stablehlo.convert %38 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %883 = stablehlo.dot_general %881, %882, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %884 = stablehlo.convert %39 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %885 = stablehlo.dot_general %881, %884, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %886 = call @silu(%885) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %887 = stablehlo.multiply %883, %886 : tensor<1x27x11008xf32>
    %888 = stablehlo.convert %40 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %889 = stablehlo.dot_general %887, %888, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %890 = stablehlo.add %864, %889 : tensor<1x27x4096xf32>
    %891 = stablehlo.multiply %890, %890 : tensor<1x27x4096xf32>
    %892 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %893 = stablehlo.reduce(%891 init: %892) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %894 = stablehlo.broadcast_in_dim %893, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %895 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %896 = stablehlo.broadcast_in_dim %895, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %897 = stablehlo.divide %894, %896 : tensor<1x27x1xf32>
    %898 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %899 = stablehlo.broadcast_in_dim %898, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %900 = stablehlo.add %897, %899 : tensor<1x27x1xf32>
    %901 = stablehlo.sqrt %900 : tensor<1x27x1xf32>
    %902 = stablehlo.broadcast_in_dim %901, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %903 = stablehlo.divide %890, %902 : tensor<1x27x4096xf32>
    %904 = stablehlo.convert %41 : (tensor<4096xf16>) -> tensor<4096xf32>
    %905 = stablehlo.broadcast_in_dim %904, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %906 = stablehlo.broadcast_in_dim %905, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %907 = stablehlo.multiply %906, %903 : tensor<1x27x4096xf32>
    %908 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %909 = stablehlo.broadcast_in_dim %908, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %910 = stablehlo.broadcast_in_dim %909, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %911 = stablehlo.broadcast_in_dim %909, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %912 = stablehlo.broadcast_in_dim %910, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %913 = stablehlo.broadcast_in_dim %911, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %914 = stablehlo.compare  GE, %912, %913,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %915 = stablehlo.broadcast_in_dim %914, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %916 = stablehlo.convert %42 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %917 = stablehlo.dot_general %907, %916, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %918 = stablehlo.convert %43 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %919 = stablehlo.dot_general %907, %918, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %920 = stablehlo.convert %44 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %921 = stablehlo.dot_general %907, %920, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %922 = stablehlo.reshape %917 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %923 = stablehlo.reshape %919 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %924 = stablehlo.reshape %921 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %925 = stablehlo.constant dense<0> : tensor<i32>
    %926 = stablehlo.broadcast_in_dim %925, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %927 = stablehlo.compare  LT, %324, %926,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %928 = stablehlo.constant dense<4096> : tensor<i32>
    %929 = stablehlo.broadcast_in_dim %928, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %930 = stablehlo.add %324, %929 : tensor<1x27xi32>
    %931 = stablehlo.select %927, %930, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %932 = stablehlo.broadcast_in_dim %931, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %933 = "stablehlo.gather"(%45, %932) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %934 = stablehlo.slice %933 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %935 = stablehlo.slice %933 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %936 = stablehlo.broadcast_in_dim %935, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %937 = stablehlo.multiply %923, %936 : tensor<1x27x32x128xf32>
    %938 = stablehlo.constant dense<64> : tensor<i32>
    %939 = stablehlo.broadcast_in_dim %938, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %940 = "stablehlo.gather"(%923, %939) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %941 = stablehlo.negate %940 : tensor<1x27x32x64xf32>
    %942 = stablehlo.constant dense<0> : tensor<i32>
    %943 = stablehlo.broadcast_in_dim %942, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %944 = "stablehlo.gather"(%923, %943) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %945 = stablehlo.concatenate %941, %944, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %946 = stablehlo.broadcast_in_dim %934, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %947 = stablehlo.multiply %945, %946 : tensor<1x27x32x128xf32>
    %948 = stablehlo.add %937, %947 : tensor<1x27x32x128xf32>
    %949 = stablehlo.broadcast_in_dim %935, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %950 = stablehlo.multiply %922, %949 : tensor<1x27x32x128xf32>
    %951 = stablehlo.constant dense<64> : tensor<i32>
    %952 = stablehlo.broadcast_in_dim %951, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %953 = "stablehlo.gather"(%922, %952) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %954 = stablehlo.negate %953 : tensor<1x27x32x64xf32>
    %955 = stablehlo.constant dense<0> : tensor<i32>
    %956 = stablehlo.broadcast_in_dim %955, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %957 = "stablehlo.gather"(%922, %956) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %958 = stablehlo.concatenate %954, %957, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %959 = stablehlo.broadcast_in_dim %934, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %960 = stablehlo.multiply %958, %959 : tensor<1x27x32x128xf32>
    %961 = stablehlo.add %950, %960 : tensor<1x27x32x128xf32>
    %962 = stablehlo.slice %915 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %963 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %964 = stablehlo.reshape %963 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %965 = stablehlo.broadcast_in_dim %964, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %966 = stablehlo.constant dense<0> : tensor<i32>
    %967 = stablehlo.broadcast_in_dim %966, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %968 = stablehlo.compare  NE, %965, %967,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %969 = stablehlo.and %968, %962 : tensor<1x1x27x27xi1>
    %970 = stablehlo.convert %969 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %971 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %972 = stablehlo.broadcast_in_dim %971, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %973 = stablehlo.compare  GT, %970, %972,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %974 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %975 = stablehlo.broadcast_in_dim %974, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %976 = stablehlo.convert %975 : tensor<1x1x27x27xf32>
    %977 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %978 = stablehlo.broadcast_in_dim %977, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %979 = stablehlo.select %973, %976, %978 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %980 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %981 = stablehlo.sqrt %980 : tensor<f32>
    %982 = stablehlo.convert %981 : tensor<f32>
    %983 = stablehlo.broadcast_in_dim %982, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %984 = stablehlo.divide %961, %983 : tensor<1x27x32x128xf32>
    %985 = stablehlo.dot_general %984, %948, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %986 = stablehlo.broadcast_in_dim %979, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %987 = stablehlo.add %985, %986 : tensor<1x32x27x27xf32>
    %988 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %989 = stablehlo.reduce(%987 init: %988) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %990 = stablehlo.broadcast_in_dim %989, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %991 = stablehlo.broadcast_in_dim %990, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %992 = stablehlo.subtract %987, %991 : tensor<1x32x27x27xf32>
    %993 = stablehlo.exponential %992 : tensor<1x32x27x27xf32>
    %994 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %995 = stablehlo.reduce(%993 init: %994) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %996 = stablehlo.broadcast_in_dim %995, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %997 = stablehlo.broadcast_in_dim %996, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %998 = stablehlo.divide %993, %997 : tensor<1x32x27x27xf32>
    %999 = stablehlo.dot_general %924, %998, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %1000 = stablehlo.transpose %999, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %1001 = stablehlo.reshape %1000 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %1002 = stablehlo.convert %46 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1003 = stablehlo.dot_general %1001, %1002, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1004 = stablehlo.add %890, %1003 : tensor<1x27x4096xf32>
    %1005 = stablehlo.multiply %1004, %1004 : tensor<1x27x4096xf32>
    %1006 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1007 = stablehlo.reduce(%1005 init: %1006) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1008 = stablehlo.broadcast_in_dim %1007, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1009 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1010 = stablehlo.broadcast_in_dim %1009, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1011 = stablehlo.divide %1008, %1010 : tensor<1x27x1xf32>
    %1012 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1013 = stablehlo.broadcast_in_dim %1012, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1014 = stablehlo.add %1011, %1013 : tensor<1x27x1xf32>
    %1015 = stablehlo.sqrt %1014 : tensor<1x27x1xf32>
    %1016 = stablehlo.broadcast_in_dim %1015, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1017 = stablehlo.divide %1004, %1016 : tensor<1x27x4096xf32>
    %1018 = stablehlo.convert %47 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1019 = stablehlo.broadcast_in_dim %1018, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1020 = stablehlo.broadcast_in_dim %1019, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1021 = stablehlo.multiply %1020, %1017 : tensor<1x27x4096xf32>
    %1022 = stablehlo.convert %48 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1023 = stablehlo.dot_general %1021, %1022, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1024 = stablehlo.convert %49 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1025 = stablehlo.dot_general %1021, %1024, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1026 = call @silu(%1025) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %1027 = stablehlo.multiply %1023, %1026 : tensor<1x27x11008xf32>
    %1028 = stablehlo.convert %50 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %1029 = stablehlo.dot_general %1027, %1028, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %1030 = stablehlo.add %1004, %1029 : tensor<1x27x4096xf32>
    %1031 = stablehlo.multiply %1030, %1030 : tensor<1x27x4096xf32>
    %1032 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1033 = stablehlo.reduce(%1031 init: %1032) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1034 = stablehlo.broadcast_in_dim %1033, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1035 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1036 = stablehlo.broadcast_in_dim %1035, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1037 = stablehlo.divide %1034, %1036 : tensor<1x27x1xf32>
    %1038 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1039 = stablehlo.broadcast_in_dim %1038, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1040 = stablehlo.add %1037, %1039 : tensor<1x27x1xf32>
    %1041 = stablehlo.sqrt %1040 : tensor<1x27x1xf32>
    %1042 = stablehlo.broadcast_in_dim %1041, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1043 = stablehlo.divide %1030, %1042 : tensor<1x27x4096xf32>
    %1044 = stablehlo.convert %51 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1045 = stablehlo.broadcast_in_dim %1044, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1046 = stablehlo.broadcast_in_dim %1045, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1047 = stablehlo.multiply %1046, %1043 : tensor<1x27x4096xf32>
    %1048 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %1049 = stablehlo.broadcast_in_dim %1048, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %1050 = stablehlo.broadcast_in_dim %1049, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %1051 = stablehlo.broadcast_in_dim %1049, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %1052 = stablehlo.broadcast_in_dim %1050, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %1053 = stablehlo.broadcast_in_dim %1051, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1054 = stablehlo.compare  GE, %1052, %1053,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %1055 = stablehlo.broadcast_in_dim %1054, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %1056 = stablehlo.convert %52 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1057 = stablehlo.dot_general %1047, %1056, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1058 = stablehlo.convert %53 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1059 = stablehlo.dot_general %1047, %1058, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1060 = stablehlo.convert %54 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1061 = stablehlo.dot_general %1047, %1060, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1062 = stablehlo.reshape %1057 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1063 = stablehlo.reshape %1059 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1064 = stablehlo.reshape %1061 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1065 = stablehlo.constant dense<0> : tensor<i32>
    %1066 = stablehlo.broadcast_in_dim %1065, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1067 = stablehlo.compare  LT, %324, %1066,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %1068 = stablehlo.constant dense<4096> : tensor<i32>
    %1069 = stablehlo.broadcast_in_dim %1068, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1070 = stablehlo.add %324, %1069 : tensor<1x27xi32>
    %1071 = stablehlo.select %1067, %1070, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %1072 = stablehlo.broadcast_in_dim %1071, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %1073 = "stablehlo.gather"(%55, %1072) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %1074 = stablehlo.slice %1073 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1075 = stablehlo.slice %1073 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1076 = stablehlo.broadcast_in_dim %1075, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1077 = stablehlo.multiply %1063, %1076 : tensor<1x27x32x128xf32>
    %1078 = stablehlo.constant dense<64> : tensor<i32>
    %1079 = stablehlo.broadcast_in_dim %1078, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1080 = "stablehlo.gather"(%1063, %1079) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1081 = stablehlo.negate %1080 : tensor<1x27x32x64xf32>
    %1082 = stablehlo.constant dense<0> : tensor<i32>
    %1083 = stablehlo.broadcast_in_dim %1082, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1084 = "stablehlo.gather"(%1063, %1083) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1085 = stablehlo.concatenate %1081, %1084, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1086 = stablehlo.broadcast_in_dim %1074, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1087 = stablehlo.multiply %1085, %1086 : tensor<1x27x32x128xf32>
    %1088 = stablehlo.add %1077, %1087 : tensor<1x27x32x128xf32>
    %1089 = stablehlo.broadcast_in_dim %1075, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1090 = stablehlo.multiply %1062, %1089 : tensor<1x27x32x128xf32>
    %1091 = stablehlo.constant dense<64> : tensor<i32>
    %1092 = stablehlo.broadcast_in_dim %1091, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1093 = "stablehlo.gather"(%1062, %1092) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1094 = stablehlo.negate %1093 : tensor<1x27x32x64xf32>
    %1095 = stablehlo.constant dense<0> : tensor<i32>
    %1096 = stablehlo.broadcast_in_dim %1095, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1097 = "stablehlo.gather"(%1062, %1096) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1098 = stablehlo.concatenate %1094, %1097, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1099 = stablehlo.broadcast_in_dim %1074, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1100 = stablehlo.multiply %1098, %1099 : tensor<1x27x32x128xf32>
    %1101 = stablehlo.add %1090, %1100 : tensor<1x27x32x128xf32>
    %1102 = stablehlo.slice %1055 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %1103 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %1104 = stablehlo.reshape %1103 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %1105 = stablehlo.broadcast_in_dim %1104, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %1106 = stablehlo.constant dense<0> : tensor<i32>
    %1107 = stablehlo.broadcast_in_dim %1106, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %1108 = stablehlo.compare  NE, %1105, %1107,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %1109 = stablehlo.and %1108, %1102 : tensor<1x1x27x27xi1>
    %1110 = stablehlo.convert %1109 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %1111 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1112 = stablehlo.broadcast_in_dim %1111, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1113 = stablehlo.compare  GT, %1110, %1112,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %1114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1115 = stablehlo.broadcast_in_dim %1114, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1116 = stablehlo.convert %1115 : tensor<1x1x27x27xf32>
    %1117 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %1118 = stablehlo.broadcast_in_dim %1117, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1119 = stablehlo.select %1113, %1116, %1118 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %1120 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1121 = stablehlo.sqrt %1120 : tensor<f32>
    %1122 = stablehlo.convert %1121 : tensor<f32>
    %1123 = stablehlo.broadcast_in_dim %1122, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %1124 = stablehlo.divide %1101, %1123 : tensor<1x27x32x128xf32>
    %1125 = stablehlo.dot_general %1124, %1088, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %1126 = stablehlo.broadcast_in_dim %1119, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %1127 = stablehlo.add %1125, %1126 : tensor<1x32x27x27xf32>
    %1128 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1129 = stablehlo.reduce(%1127 init: %1128) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1130 = stablehlo.broadcast_in_dim %1129, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1131 = stablehlo.broadcast_in_dim %1130, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1132 = stablehlo.subtract %1127, %1131 : tensor<1x32x27x27xf32>
    %1133 = stablehlo.exponential %1132 : tensor<1x32x27x27xf32>
    %1134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1135 = stablehlo.reduce(%1133 init: %1134) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1136 = stablehlo.broadcast_in_dim %1135, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1137 = stablehlo.broadcast_in_dim %1136, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1138 = stablehlo.divide %1133, %1137 : tensor<1x32x27x27xf32>
    %1139 = stablehlo.dot_general %1064, %1138, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %1140 = stablehlo.transpose %1139, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %1141 = stablehlo.reshape %1140 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %1142 = stablehlo.convert %56 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1143 = stablehlo.dot_general %1141, %1142, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1144 = stablehlo.add %1030, %1143 : tensor<1x27x4096xf32>
    %1145 = stablehlo.multiply %1144, %1144 : tensor<1x27x4096xf32>
    %1146 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1147 = stablehlo.reduce(%1145 init: %1146) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1148 = stablehlo.broadcast_in_dim %1147, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1149 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1150 = stablehlo.broadcast_in_dim %1149, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1151 = stablehlo.divide %1148, %1150 : tensor<1x27x1xf32>
    %1152 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1153 = stablehlo.broadcast_in_dim %1152, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1154 = stablehlo.add %1151, %1153 : tensor<1x27x1xf32>
    %1155 = stablehlo.sqrt %1154 : tensor<1x27x1xf32>
    %1156 = stablehlo.broadcast_in_dim %1155, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1157 = stablehlo.divide %1144, %1156 : tensor<1x27x4096xf32>
    %1158 = stablehlo.convert %57 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1159 = stablehlo.broadcast_in_dim %1158, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1160 = stablehlo.broadcast_in_dim %1159, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1161 = stablehlo.multiply %1160, %1157 : tensor<1x27x4096xf32>
    %1162 = stablehlo.convert %58 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1163 = stablehlo.dot_general %1161, %1162, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1164 = stablehlo.convert %59 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1165 = stablehlo.dot_general %1161, %1164, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1166 = call @silu(%1165) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %1167 = stablehlo.multiply %1163, %1166 : tensor<1x27x11008xf32>
    %1168 = stablehlo.convert %60 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %1169 = stablehlo.dot_general %1167, %1168, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %1170 = stablehlo.add %1144, %1169 : tensor<1x27x4096xf32>
    %1171 = stablehlo.multiply %1170, %1170 : tensor<1x27x4096xf32>
    %1172 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1173 = stablehlo.reduce(%1171 init: %1172) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1174 = stablehlo.broadcast_in_dim %1173, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1175 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1176 = stablehlo.broadcast_in_dim %1175, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1177 = stablehlo.divide %1174, %1176 : tensor<1x27x1xf32>
    %1178 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1179 = stablehlo.broadcast_in_dim %1178, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1180 = stablehlo.add %1177, %1179 : tensor<1x27x1xf32>
    %1181 = stablehlo.sqrt %1180 : tensor<1x27x1xf32>
    %1182 = stablehlo.broadcast_in_dim %1181, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1183 = stablehlo.divide %1170, %1182 : tensor<1x27x4096xf32>
    %1184 = stablehlo.convert %61 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1185 = stablehlo.broadcast_in_dim %1184, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1186 = stablehlo.broadcast_in_dim %1185, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1187 = stablehlo.multiply %1186, %1183 : tensor<1x27x4096xf32>
    %1188 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %1189 = stablehlo.broadcast_in_dim %1188, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %1190 = stablehlo.broadcast_in_dim %1189, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %1191 = stablehlo.broadcast_in_dim %1189, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %1192 = stablehlo.broadcast_in_dim %1190, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %1193 = stablehlo.broadcast_in_dim %1191, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1194 = stablehlo.compare  GE, %1192, %1193,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %1195 = stablehlo.broadcast_in_dim %1194, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %1196 = stablehlo.convert %62 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1197 = stablehlo.dot_general %1187, %1196, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1198 = stablehlo.convert %63 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1199 = stablehlo.dot_general %1187, %1198, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1200 = stablehlo.convert %64 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1201 = stablehlo.dot_general %1187, %1200, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1202 = stablehlo.reshape %1197 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1203 = stablehlo.reshape %1199 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1204 = stablehlo.reshape %1201 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1205 = stablehlo.constant dense<0> : tensor<i32>
    %1206 = stablehlo.broadcast_in_dim %1205, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1207 = stablehlo.compare  LT, %324, %1206,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %1208 = stablehlo.constant dense<4096> : tensor<i32>
    %1209 = stablehlo.broadcast_in_dim %1208, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1210 = stablehlo.add %324, %1209 : tensor<1x27xi32>
    %1211 = stablehlo.select %1207, %1210, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %1212 = stablehlo.broadcast_in_dim %1211, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %1213 = "stablehlo.gather"(%65, %1212) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %1214 = stablehlo.slice %1213 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1215 = stablehlo.slice %1213 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1216 = stablehlo.broadcast_in_dim %1215, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1217 = stablehlo.multiply %1203, %1216 : tensor<1x27x32x128xf32>
    %1218 = stablehlo.constant dense<64> : tensor<i32>
    %1219 = stablehlo.broadcast_in_dim %1218, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1220 = "stablehlo.gather"(%1203, %1219) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1221 = stablehlo.negate %1220 : tensor<1x27x32x64xf32>
    %1222 = stablehlo.constant dense<0> : tensor<i32>
    %1223 = stablehlo.broadcast_in_dim %1222, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1224 = "stablehlo.gather"(%1203, %1223) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1225 = stablehlo.concatenate %1221, %1224, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1226 = stablehlo.broadcast_in_dim %1214, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1227 = stablehlo.multiply %1225, %1226 : tensor<1x27x32x128xf32>
    %1228 = stablehlo.add %1217, %1227 : tensor<1x27x32x128xf32>
    %1229 = stablehlo.broadcast_in_dim %1215, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1230 = stablehlo.multiply %1202, %1229 : tensor<1x27x32x128xf32>
    %1231 = stablehlo.constant dense<64> : tensor<i32>
    %1232 = stablehlo.broadcast_in_dim %1231, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1233 = "stablehlo.gather"(%1202, %1232) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1234 = stablehlo.negate %1233 : tensor<1x27x32x64xf32>
    %1235 = stablehlo.constant dense<0> : tensor<i32>
    %1236 = stablehlo.broadcast_in_dim %1235, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1237 = "stablehlo.gather"(%1202, %1236) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1238 = stablehlo.concatenate %1234, %1237, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1239 = stablehlo.broadcast_in_dim %1214, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1240 = stablehlo.multiply %1238, %1239 : tensor<1x27x32x128xf32>
    %1241 = stablehlo.add %1230, %1240 : tensor<1x27x32x128xf32>
    %1242 = stablehlo.slice %1195 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %1243 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %1244 = stablehlo.reshape %1243 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %1245 = stablehlo.broadcast_in_dim %1244, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %1246 = stablehlo.constant dense<0> : tensor<i32>
    %1247 = stablehlo.broadcast_in_dim %1246, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %1248 = stablehlo.compare  NE, %1245, %1247,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %1249 = stablehlo.and %1248, %1242 : tensor<1x1x27x27xi1>
    %1250 = stablehlo.convert %1249 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %1251 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1252 = stablehlo.broadcast_in_dim %1251, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1253 = stablehlo.compare  GT, %1250, %1252,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %1254 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1255 = stablehlo.broadcast_in_dim %1254, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1256 = stablehlo.convert %1255 : tensor<1x1x27x27xf32>
    %1257 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %1258 = stablehlo.broadcast_in_dim %1257, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1259 = stablehlo.select %1253, %1256, %1258 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %1260 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1261 = stablehlo.sqrt %1260 : tensor<f32>
    %1262 = stablehlo.convert %1261 : tensor<f32>
    %1263 = stablehlo.broadcast_in_dim %1262, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %1264 = stablehlo.divide %1241, %1263 : tensor<1x27x32x128xf32>
    %1265 = stablehlo.dot_general %1264, %1228, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %1266 = stablehlo.broadcast_in_dim %1259, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %1267 = stablehlo.add %1265, %1266 : tensor<1x32x27x27xf32>
    %1268 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1269 = stablehlo.reduce(%1267 init: %1268) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1270 = stablehlo.broadcast_in_dim %1269, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1271 = stablehlo.broadcast_in_dim %1270, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1272 = stablehlo.subtract %1267, %1271 : tensor<1x32x27x27xf32>
    %1273 = stablehlo.exponential %1272 : tensor<1x32x27x27xf32>
    %1274 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1275 = stablehlo.reduce(%1273 init: %1274) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1276 = stablehlo.broadcast_in_dim %1275, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1277 = stablehlo.broadcast_in_dim %1276, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1278 = stablehlo.divide %1273, %1277 : tensor<1x32x27x27xf32>
    %1279 = stablehlo.dot_general %1204, %1278, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %1280 = stablehlo.transpose %1279, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %1281 = stablehlo.reshape %1280 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %1282 = stablehlo.convert %66 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1283 = stablehlo.dot_general %1281, %1282, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1284 = stablehlo.add %1170, %1283 : tensor<1x27x4096xf32>
    %1285 = stablehlo.multiply %1284, %1284 : tensor<1x27x4096xf32>
    %1286 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1287 = stablehlo.reduce(%1285 init: %1286) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1288 = stablehlo.broadcast_in_dim %1287, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1289 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1290 = stablehlo.broadcast_in_dim %1289, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1291 = stablehlo.divide %1288, %1290 : tensor<1x27x1xf32>
    %1292 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1293 = stablehlo.broadcast_in_dim %1292, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1294 = stablehlo.add %1291, %1293 : tensor<1x27x1xf32>
    %1295 = stablehlo.sqrt %1294 : tensor<1x27x1xf32>
    %1296 = stablehlo.broadcast_in_dim %1295, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1297 = stablehlo.divide %1284, %1296 : tensor<1x27x4096xf32>
    %1298 = stablehlo.convert %67 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1299 = stablehlo.broadcast_in_dim %1298, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1300 = stablehlo.broadcast_in_dim %1299, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1301 = stablehlo.multiply %1300, %1297 : tensor<1x27x4096xf32>
    %1302 = stablehlo.convert %68 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1303 = stablehlo.dot_general %1301, %1302, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1304 = stablehlo.convert %69 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1305 = stablehlo.dot_general %1301, %1304, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1306 = call @silu(%1305) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %1307 = stablehlo.multiply %1303, %1306 : tensor<1x27x11008xf32>
    %1308 = stablehlo.convert %70 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %1309 = stablehlo.dot_general %1307, %1308, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %1310 = stablehlo.add %1284, %1309 : tensor<1x27x4096xf32>
    %1311 = stablehlo.multiply %1310, %1310 : tensor<1x27x4096xf32>
    %1312 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1313 = stablehlo.reduce(%1311 init: %1312) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1314 = stablehlo.broadcast_in_dim %1313, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1315 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1316 = stablehlo.broadcast_in_dim %1315, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1317 = stablehlo.divide %1314, %1316 : tensor<1x27x1xf32>
    %1318 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1319 = stablehlo.broadcast_in_dim %1318, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1320 = stablehlo.add %1317, %1319 : tensor<1x27x1xf32>
    %1321 = stablehlo.sqrt %1320 : tensor<1x27x1xf32>
    %1322 = stablehlo.broadcast_in_dim %1321, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1323 = stablehlo.divide %1310, %1322 : tensor<1x27x4096xf32>
    %1324 = stablehlo.convert %71 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1325 = stablehlo.broadcast_in_dim %1324, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1326 = stablehlo.broadcast_in_dim %1325, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1327 = stablehlo.multiply %1326, %1323 : tensor<1x27x4096xf32>
    %1328 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %1329 = stablehlo.broadcast_in_dim %1328, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %1330 = stablehlo.broadcast_in_dim %1329, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %1331 = stablehlo.broadcast_in_dim %1329, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %1332 = stablehlo.broadcast_in_dim %1330, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %1333 = stablehlo.broadcast_in_dim %1331, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1334 = stablehlo.compare  GE, %1332, %1333,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %1335 = stablehlo.broadcast_in_dim %1334, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %1336 = stablehlo.convert %72 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1337 = stablehlo.dot_general %1327, %1336, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1338 = stablehlo.convert %73 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1339 = stablehlo.dot_general %1327, %1338, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1340 = stablehlo.convert %74 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1341 = stablehlo.dot_general %1327, %1340, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1342 = stablehlo.reshape %1337 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1343 = stablehlo.reshape %1339 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1344 = stablehlo.reshape %1341 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1345 = stablehlo.constant dense<0> : tensor<i32>
    %1346 = stablehlo.broadcast_in_dim %1345, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1347 = stablehlo.compare  LT, %324, %1346,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %1348 = stablehlo.constant dense<4096> : tensor<i32>
    %1349 = stablehlo.broadcast_in_dim %1348, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1350 = stablehlo.add %324, %1349 : tensor<1x27xi32>
    %1351 = stablehlo.select %1347, %1350, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %1352 = stablehlo.broadcast_in_dim %1351, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %1353 = "stablehlo.gather"(%75, %1352) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %1354 = stablehlo.slice %1353 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1355 = stablehlo.slice %1353 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1356 = stablehlo.broadcast_in_dim %1355, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1357 = stablehlo.multiply %1343, %1356 : tensor<1x27x32x128xf32>
    %1358 = stablehlo.constant dense<64> : tensor<i32>
    %1359 = stablehlo.broadcast_in_dim %1358, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1360 = "stablehlo.gather"(%1343, %1359) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1361 = stablehlo.negate %1360 : tensor<1x27x32x64xf32>
    %1362 = stablehlo.constant dense<0> : tensor<i32>
    %1363 = stablehlo.broadcast_in_dim %1362, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1364 = "stablehlo.gather"(%1343, %1363) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1365 = stablehlo.concatenate %1361, %1364, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1366 = stablehlo.broadcast_in_dim %1354, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1367 = stablehlo.multiply %1365, %1366 : tensor<1x27x32x128xf32>
    %1368 = stablehlo.add %1357, %1367 : tensor<1x27x32x128xf32>
    %1369 = stablehlo.broadcast_in_dim %1355, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1370 = stablehlo.multiply %1342, %1369 : tensor<1x27x32x128xf32>
    %1371 = stablehlo.constant dense<64> : tensor<i32>
    %1372 = stablehlo.broadcast_in_dim %1371, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1373 = "stablehlo.gather"(%1342, %1372) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1374 = stablehlo.negate %1373 : tensor<1x27x32x64xf32>
    %1375 = stablehlo.constant dense<0> : tensor<i32>
    %1376 = stablehlo.broadcast_in_dim %1375, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1377 = "stablehlo.gather"(%1342, %1376) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1378 = stablehlo.concatenate %1374, %1377, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1379 = stablehlo.broadcast_in_dim %1354, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1380 = stablehlo.multiply %1378, %1379 : tensor<1x27x32x128xf32>
    %1381 = stablehlo.add %1370, %1380 : tensor<1x27x32x128xf32>
    %1382 = stablehlo.slice %1335 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %1383 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %1384 = stablehlo.reshape %1383 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %1385 = stablehlo.broadcast_in_dim %1384, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %1386 = stablehlo.constant dense<0> : tensor<i32>
    %1387 = stablehlo.broadcast_in_dim %1386, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %1388 = stablehlo.compare  NE, %1385, %1387,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %1389 = stablehlo.and %1388, %1382 : tensor<1x1x27x27xi1>
    %1390 = stablehlo.convert %1389 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %1391 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1392 = stablehlo.broadcast_in_dim %1391, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1393 = stablehlo.compare  GT, %1390, %1392,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %1394 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1395 = stablehlo.broadcast_in_dim %1394, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1396 = stablehlo.convert %1395 : tensor<1x1x27x27xf32>
    %1397 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %1398 = stablehlo.broadcast_in_dim %1397, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1399 = stablehlo.select %1393, %1396, %1398 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %1400 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1401 = stablehlo.sqrt %1400 : tensor<f32>
    %1402 = stablehlo.convert %1401 : tensor<f32>
    %1403 = stablehlo.broadcast_in_dim %1402, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %1404 = stablehlo.divide %1381, %1403 : tensor<1x27x32x128xf32>
    %1405 = stablehlo.dot_general %1404, %1368, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %1406 = stablehlo.broadcast_in_dim %1399, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %1407 = stablehlo.add %1405, %1406 : tensor<1x32x27x27xf32>
    %1408 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1409 = stablehlo.reduce(%1407 init: %1408) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1410 = stablehlo.broadcast_in_dim %1409, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1411 = stablehlo.broadcast_in_dim %1410, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1412 = stablehlo.subtract %1407, %1411 : tensor<1x32x27x27xf32>
    %1413 = stablehlo.exponential %1412 : tensor<1x32x27x27xf32>
    %1414 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1415 = stablehlo.reduce(%1413 init: %1414) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1416 = stablehlo.broadcast_in_dim %1415, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1417 = stablehlo.broadcast_in_dim %1416, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1418 = stablehlo.divide %1413, %1417 : tensor<1x32x27x27xf32>
    %1419 = stablehlo.dot_general %1344, %1418, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %1420 = stablehlo.transpose %1419, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %1421 = stablehlo.reshape %1420 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %1422 = stablehlo.convert %76 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1423 = stablehlo.dot_general %1421, %1422, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1424 = stablehlo.add %1310, %1423 : tensor<1x27x4096xf32>
    %1425 = stablehlo.multiply %1424, %1424 : tensor<1x27x4096xf32>
    %1426 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1427 = stablehlo.reduce(%1425 init: %1426) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1428 = stablehlo.broadcast_in_dim %1427, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1429 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1430 = stablehlo.broadcast_in_dim %1429, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1431 = stablehlo.divide %1428, %1430 : tensor<1x27x1xf32>
    %1432 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1433 = stablehlo.broadcast_in_dim %1432, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1434 = stablehlo.add %1431, %1433 : tensor<1x27x1xf32>
    %1435 = stablehlo.sqrt %1434 : tensor<1x27x1xf32>
    %1436 = stablehlo.broadcast_in_dim %1435, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1437 = stablehlo.divide %1424, %1436 : tensor<1x27x4096xf32>
    %1438 = stablehlo.convert %77 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1439 = stablehlo.broadcast_in_dim %1438, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1440 = stablehlo.broadcast_in_dim %1439, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1441 = stablehlo.multiply %1440, %1437 : tensor<1x27x4096xf32>
    %1442 = stablehlo.convert %78 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1443 = stablehlo.dot_general %1441, %1442, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1444 = stablehlo.convert %79 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1445 = stablehlo.dot_general %1441, %1444, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1446 = call @silu(%1445) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %1447 = stablehlo.multiply %1443, %1446 : tensor<1x27x11008xf32>
    %1448 = stablehlo.convert %80 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %1449 = stablehlo.dot_general %1447, %1448, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %1450 = stablehlo.add %1424, %1449 : tensor<1x27x4096xf32>
    %1451 = stablehlo.multiply %1450, %1450 : tensor<1x27x4096xf32>
    %1452 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1453 = stablehlo.reduce(%1451 init: %1452) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1454 = stablehlo.broadcast_in_dim %1453, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1455 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1456 = stablehlo.broadcast_in_dim %1455, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1457 = stablehlo.divide %1454, %1456 : tensor<1x27x1xf32>
    %1458 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1459 = stablehlo.broadcast_in_dim %1458, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1460 = stablehlo.add %1457, %1459 : tensor<1x27x1xf32>
    %1461 = stablehlo.sqrt %1460 : tensor<1x27x1xf32>
    %1462 = stablehlo.broadcast_in_dim %1461, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1463 = stablehlo.divide %1450, %1462 : tensor<1x27x4096xf32>
    %1464 = stablehlo.convert %81 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1465 = stablehlo.broadcast_in_dim %1464, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1466 = stablehlo.broadcast_in_dim %1465, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1467 = stablehlo.multiply %1466, %1463 : tensor<1x27x4096xf32>
    %1468 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %1469 = stablehlo.broadcast_in_dim %1468, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %1470 = stablehlo.broadcast_in_dim %1469, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %1471 = stablehlo.broadcast_in_dim %1469, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %1472 = stablehlo.broadcast_in_dim %1470, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %1473 = stablehlo.broadcast_in_dim %1471, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1474 = stablehlo.compare  GE, %1472, %1473,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %1475 = stablehlo.broadcast_in_dim %1474, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %1476 = stablehlo.convert %82 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1477 = stablehlo.dot_general %1467, %1476, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1478 = stablehlo.convert %83 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1479 = stablehlo.dot_general %1467, %1478, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1480 = stablehlo.convert %84 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1481 = stablehlo.dot_general %1467, %1480, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1482 = stablehlo.reshape %1477 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1483 = stablehlo.reshape %1479 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1484 = stablehlo.reshape %1481 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1485 = stablehlo.constant dense<0> : tensor<i32>
    %1486 = stablehlo.broadcast_in_dim %1485, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1487 = stablehlo.compare  LT, %324, %1486,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %1488 = stablehlo.constant dense<4096> : tensor<i32>
    %1489 = stablehlo.broadcast_in_dim %1488, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1490 = stablehlo.add %324, %1489 : tensor<1x27xi32>
    %1491 = stablehlo.select %1487, %1490, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %1492 = stablehlo.broadcast_in_dim %1491, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %1493 = "stablehlo.gather"(%85, %1492) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %1494 = stablehlo.slice %1493 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1495 = stablehlo.slice %1493 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1496 = stablehlo.broadcast_in_dim %1495, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1497 = stablehlo.multiply %1483, %1496 : tensor<1x27x32x128xf32>
    %1498 = stablehlo.constant dense<64> : tensor<i32>
    %1499 = stablehlo.broadcast_in_dim %1498, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1500 = "stablehlo.gather"(%1483, %1499) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1501 = stablehlo.negate %1500 : tensor<1x27x32x64xf32>
    %1502 = stablehlo.constant dense<0> : tensor<i32>
    %1503 = stablehlo.broadcast_in_dim %1502, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1504 = "stablehlo.gather"(%1483, %1503) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1505 = stablehlo.concatenate %1501, %1504, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1506 = stablehlo.broadcast_in_dim %1494, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1507 = stablehlo.multiply %1505, %1506 : tensor<1x27x32x128xf32>
    %1508 = stablehlo.add %1497, %1507 : tensor<1x27x32x128xf32>
    %1509 = stablehlo.broadcast_in_dim %1495, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1510 = stablehlo.multiply %1482, %1509 : tensor<1x27x32x128xf32>
    %1511 = stablehlo.constant dense<64> : tensor<i32>
    %1512 = stablehlo.broadcast_in_dim %1511, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1513 = "stablehlo.gather"(%1482, %1512) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1514 = stablehlo.negate %1513 : tensor<1x27x32x64xf32>
    %1515 = stablehlo.constant dense<0> : tensor<i32>
    %1516 = stablehlo.broadcast_in_dim %1515, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1517 = "stablehlo.gather"(%1482, %1516) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1518 = stablehlo.concatenate %1514, %1517, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1519 = stablehlo.broadcast_in_dim %1494, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1520 = stablehlo.multiply %1518, %1519 : tensor<1x27x32x128xf32>
    %1521 = stablehlo.add %1510, %1520 : tensor<1x27x32x128xf32>
    %1522 = stablehlo.slice %1475 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %1523 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %1524 = stablehlo.reshape %1523 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %1525 = stablehlo.broadcast_in_dim %1524, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %1526 = stablehlo.constant dense<0> : tensor<i32>
    %1527 = stablehlo.broadcast_in_dim %1526, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %1528 = stablehlo.compare  NE, %1525, %1527,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %1529 = stablehlo.and %1528, %1522 : tensor<1x1x27x27xi1>
    %1530 = stablehlo.convert %1529 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %1531 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1532 = stablehlo.broadcast_in_dim %1531, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1533 = stablehlo.compare  GT, %1530, %1532,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %1534 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1535 = stablehlo.broadcast_in_dim %1534, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1536 = stablehlo.convert %1535 : tensor<1x1x27x27xf32>
    %1537 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %1538 = stablehlo.broadcast_in_dim %1537, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1539 = stablehlo.select %1533, %1536, %1538 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %1540 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1541 = stablehlo.sqrt %1540 : tensor<f32>
    %1542 = stablehlo.convert %1541 : tensor<f32>
    %1543 = stablehlo.broadcast_in_dim %1542, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %1544 = stablehlo.divide %1521, %1543 : tensor<1x27x32x128xf32>
    %1545 = stablehlo.dot_general %1544, %1508, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %1546 = stablehlo.broadcast_in_dim %1539, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %1547 = stablehlo.add %1545, %1546 : tensor<1x32x27x27xf32>
    %1548 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1549 = stablehlo.reduce(%1547 init: %1548) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1550 = stablehlo.broadcast_in_dim %1549, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1551 = stablehlo.broadcast_in_dim %1550, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1552 = stablehlo.subtract %1547, %1551 : tensor<1x32x27x27xf32>
    %1553 = stablehlo.exponential %1552 : tensor<1x32x27x27xf32>
    %1554 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1555 = stablehlo.reduce(%1553 init: %1554) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1556 = stablehlo.broadcast_in_dim %1555, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1557 = stablehlo.broadcast_in_dim %1556, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1558 = stablehlo.divide %1553, %1557 : tensor<1x32x27x27xf32>
    %1559 = stablehlo.dot_general %1484, %1558, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %1560 = stablehlo.transpose %1559, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %1561 = stablehlo.reshape %1560 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %1562 = stablehlo.convert %86 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1563 = stablehlo.dot_general %1561, %1562, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1564 = stablehlo.add %1450, %1563 : tensor<1x27x4096xf32>
    %1565 = stablehlo.multiply %1564, %1564 : tensor<1x27x4096xf32>
    %1566 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1567 = stablehlo.reduce(%1565 init: %1566) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1568 = stablehlo.broadcast_in_dim %1567, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1569 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1570 = stablehlo.broadcast_in_dim %1569, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1571 = stablehlo.divide %1568, %1570 : tensor<1x27x1xf32>
    %1572 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1573 = stablehlo.broadcast_in_dim %1572, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1574 = stablehlo.add %1571, %1573 : tensor<1x27x1xf32>
    %1575 = stablehlo.sqrt %1574 : tensor<1x27x1xf32>
    %1576 = stablehlo.broadcast_in_dim %1575, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1577 = stablehlo.divide %1564, %1576 : tensor<1x27x4096xf32>
    %1578 = stablehlo.convert %87 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1579 = stablehlo.broadcast_in_dim %1578, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1580 = stablehlo.broadcast_in_dim %1579, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1581 = stablehlo.multiply %1580, %1577 : tensor<1x27x4096xf32>
    %1582 = stablehlo.convert %88 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1583 = stablehlo.dot_general %1581, %1582, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1584 = stablehlo.convert %89 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1585 = stablehlo.dot_general %1581, %1584, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1586 = call @silu(%1585) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %1587 = stablehlo.multiply %1583, %1586 : tensor<1x27x11008xf32>
    %1588 = stablehlo.convert %90 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %1589 = stablehlo.dot_general %1587, %1588, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %1590 = stablehlo.add %1564, %1589 : tensor<1x27x4096xf32>
    %1591 = stablehlo.multiply %1590, %1590 : tensor<1x27x4096xf32>
    %1592 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1593 = stablehlo.reduce(%1591 init: %1592) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1594 = stablehlo.broadcast_in_dim %1593, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1595 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1596 = stablehlo.broadcast_in_dim %1595, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1597 = stablehlo.divide %1594, %1596 : tensor<1x27x1xf32>
    %1598 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1599 = stablehlo.broadcast_in_dim %1598, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1600 = stablehlo.add %1597, %1599 : tensor<1x27x1xf32>
    %1601 = stablehlo.sqrt %1600 : tensor<1x27x1xf32>
    %1602 = stablehlo.broadcast_in_dim %1601, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1603 = stablehlo.divide %1590, %1602 : tensor<1x27x4096xf32>
    %1604 = stablehlo.convert %91 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1605 = stablehlo.broadcast_in_dim %1604, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1606 = stablehlo.broadcast_in_dim %1605, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1607 = stablehlo.multiply %1606, %1603 : tensor<1x27x4096xf32>
    %1608 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %1609 = stablehlo.broadcast_in_dim %1608, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %1610 = stablehlo.broadcast_in_dim %1609, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %1611 = stablehlo.broadcast_in_dim %1609, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %1612 = stablehlo.broadcast_in_dim %1610, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %1613 = stablehlo.broadcast_in_dim %1611, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1614 = stablehlo.compare  GE, %1612, %1613,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %1615 = stablehlo.broadcast_in_dim %1614, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %1616 = stablehlo.convert %92 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1617 = stablehlo.dot_general %1607, %1616, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1618 = stablehlo.convert %93 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1619 = stablehlo.dot_general %1607, %1618, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1620 = stablehlo.convert %94 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1621 = stablehlo.dot_general %1607, %1620, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1622 = stablehlo.reshape %1617 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1623 = stablehlo.reshape %1619 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1624 = stablehlo.reshape %1621 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1625 = stablehlo.constant dense<0> : tensor<i32>
    %1626 = stablehlo.broadcast_in_dim %1625, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1627 = stablehlo.compare  LT, %324, %1626,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %1628 = stablehlo.constant dense<4096> : tensor<i32>
    %1629 = stablehlo.broadcast_in_dim %1628, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1630 = stablehlo.add %324, %1629 : tensor<1x27xi32>
    %1631 = stablehlo.select %1627, %1630, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %1632 = stablehlo.broadcast_in_dim %1631, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %1633 = "stablehlo.gather"(%95, %1632) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %1634 = stablehlo.slice %1633 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1635 = stablehlo.slice %1633 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1636 = stablehlo.broadcast_in_dim %1635, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1637 = stablehlo.multiply %1623, %1636 : tensor<1x27x32x128xf32>
    %1638 = stablehlo.constant dense<64> : tensor<i32>
    %1639 = stablehlo.broadcast_in_dim %1638, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1640 = "stablehlo.gather"(%1623, %1639) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1641 = stablehlo.negate %1640 : tensor<1x27x32x64xf32>
    %1642 = stablehlo.constant dense<0> : tensor<i32>
    %1643 = stablehlo.broadcast_in_dim %1642, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1644 = "stablehlo.gather"(%1623, %1643) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1645 = stablehlo.concatenate %1641, %1644, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1646 = stablehlo.broadcast_in_dim %1634, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1647 = stablehlo.multiply %1645, %1646 : tensor<1x27x32x128xf32>
    %1648 = stablehlo.add %1637, %1647 : tensor<1x27x32x128xf32>
    %1649 = stablehlo.broadcast_in_dim %1635, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1650 = stablehlo.multiply %1622, %1649 : tensor<1x27x32x128xf32>
    %1651 = stablehlo.constant dense<64> : tensor<i32>
    %1652 = stablehlo.broadcast_in_dim %1651, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1653 = "stablehlo.gather"(%1622, %1652) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1654 = stablehlo.negate %1653 : tensor<1x27x32x64xf32>
    %1655 = stablehlo.constant dense<0> : tensor<i32>
    %1656 = stablehlo.broadcast_in_dim %1655, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1657 = "stablehlo.gather"(%1622, %1656) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1658 = stablehlo.concatenate %1654, %1657, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1659 = stablehlo.broadcast_in_dim %1634, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1660 = stablehlo.multiply %1658, %1659 : tensor<1x27x32x128xf32>
    %1661 = stablehlo.add %1650, %1660 : tensor<1x27x32x128xf32>
    %1662 = stablehlo.slice %1615 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %1663 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %1664 = stablehlo.reshape %1663 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %1665 = stablehlo.broadcast_in_dim %1664, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %1666 = stablehlo.constant dense<0> : tensor<i32>
    %1667 = stablehlo.broadcast_in_dim %1666, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %1668 = stablehlo.compare  NE, %1665, %1667,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %1669 = stablehlo.and %1668, %1662 : tensor<1x1x27x27xi1>
    %1670 = stablehlo.convert %1669 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %1671 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1672 = stablehlo.broadcast_in_dim %1671, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1673 = stablehlo.compare  GT, %1670, %1672,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %1674 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1675 = stablehlo.broadcast_in_dim %1674, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1676 = stablehlo.convert %1675 : tensor<1x1x27x27xf32>
    %1677 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %1678 = stablehlo.broadcast_in_dim %1677, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1679 = stablehlo.select %1673, %1676, %1678 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %1680 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1681 = stablehlo.sqrt %1680 : tensor<f32>
    %1682 = stablehlo.convert %1681 : tensor<f32>
    %1683 = stablehlo.broadcast_in_dim %1682, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %1684 = stablehlo.divide %1661, %1683 : tensor<1x27x32x128xf32>
    %1685 = stablehlo.dot_general %1684, %1648, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %1686 = stablehlo.broadcast_in_dim %1679, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %1687 = stablehlo.add %1685, %1686 : tensor<1x32x27x27xf32>
    %1688 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1689 = stablehlo.reduce(%1687 init: %1688) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1690 = stablehlo.broadcast_in_dim %1689, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1691 = stablehlo.broadcast_in_dim %1690, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1692 = stablehlo.subtract %1687, %1691 : tensor<1x32x27x27xf32>
    %1693 = stablehlo.exponential %1692 : tensor<1x32x27x27xf32>
    %1694 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1695 = stablehlo.reduce(%1693 init: %1694) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1696 = stablehlo.broadcast_in_dim %1695, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1697 = stablehlo.broadcast_in_dim %1696, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1698 = stablehlo.divide %1693, %1697 : tensor<1x32x27x27xf32>
    %1699 = stablehlo.dot_general %1624, %1698, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %1700 = stablehlo.transpose %1699, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %1701 = stablehlo.reshape %1700 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %1702 = stablehlo.convert %96 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1703 = stablehlo.dot_general %1701, %1702, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1704 = stablehlo.add %1590, %1703 : tensor<1x27x4096xf32>
    %1705 = stablehlo.multiply %1704, %1704 : tensor<1x27x4096xf32>
    %1706 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1707 = stablehlo.reduce(%1705 init: %1706) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1708 = stablehlo.broadcast_in_dim %1707, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1709 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1710 = stablehlo.broadcast_in_dim %1709, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1711 = stablehlo.divide %1708, %1710 : tensor<1x27x1xf32>
    %1712 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1713 = stablehlo.broadcast_in_dim %1712, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1714 = stablehlo.add %1711, %1713 : tensor<1x27x1xf32>
    %1715 = stablehlo.sqrt %1714 : tensor<1x27x1xf32>
    %1716 = stablehlo.broadcast_in_dim %1715, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1717 = stablehlo.divide %1704, %1716 : tensor<1x27x4096xf32>
    %1718 = stablehlo.convert %97 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1719 = stablehlo.broadcast_in_dim %1718, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1720 = stablehlo.broadcast_in_dim %1719, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1721 = stablehlo.multiply %1720, %1717 : tensor<1x27x4096xf32>
    %1722 = stablehlo.convert %98 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1723 = stablehlo.dot_general %1721, %1722, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1724 = stablehlo.convert %99 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1725 = stablehlo.dot_general %1721, %1724, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1726 = call @silu(%1725) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %1727 = stablehlo.multiply %1723, %1726 : tensor<1x27x11008xf32>
    %1728 = stablehlo.convert %100 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %1729 = stablehlo.dot_general %1727, %1728, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %1730 = stablehlo.add %1704, %1729 : tensor<1x27x4096xf32>
    %1731 = stablehlo.multiply %1730, %1730 : tensor<1x27x4096xf32>
    %1732 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1733 = stablehlo.reduce(%1731 init: %1732) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1734 = stablehlo.broadcast_in_dim %1733, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1735 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1736 = stablehlo.broadcast_in_dim %1735, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1737 = stablehlo.divide %1734, %1736 : tensor<1x27x1xf32>
    %1738 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1739 = stablehlo.broadcast_in_dim %1738, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1740 = stablehlo.add %1737, %1739 : tensor<1x27x1xf32>
    %1741 = stablehlo.sqrt %1740 : tensor<1x27x1xf32>
    %1742 = stablehlo.broadcast_in_dim %1741, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1743 = stablehlo.divide %1730, %1742 : tensor<1x27x4096xf32>
    %1744 = stablehlo.convert %101 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1745 = stablehlo.broadcast_in_dim %1744, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1746 = stablehlo.broadcast_in_dim %1745, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1747 = stablehlo.multiply %1746, %1743 : tensor<1x27x4096xf32>
    %1748 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %1749 = stablehlo.broadcast_in_dim %1748, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %1750 = stablehlo.broadcast_in_dim %1749, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %1751 = stablehlo.broadcast_in_dim %1749, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %1752 = stablehlo.broadcast_in_dim %1750, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %1753 = stablehlo.broadcast_in_dim %1751, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1754 = stablehlo.compare  GE, %1752, %1753,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %1755 = stablehlo.broadcast_in_dim %1754, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %1756 = stablehlo.convert %102 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1757 = stablehlo.dot_general %1747, %1756, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1758 = stablehlo.convert %103 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1759 = stablehlo.dot_general %1747, %1758, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1760 = stablehlo.convert %104 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1761 = stablehlo.dot_general %1747, %1760, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1762 = stablehlo.reshape %1757 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1763 = stablehlo.reshape %1759 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1764 = stablehlo.reshape %1761 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1765 = stablehlo.constant dense<0> : tensor<i32>
    %1766 = stablehlo.broadcast_in_dim %1765, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1767 = stablehlo.compare  LT, %324, %1766,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %1768 = stablehlo.constant dense<4096> : tensor<i32>
    %1769 = stablehlo.broadcast_in_dim %1768, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1770 = stablehlo.add %324, %1769 : tensor<1x27xi32>
    %1771 = stablehlo.select %1767, %1770, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %1772 = stablehlo.broadcast_in_dim %1771, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %1773 = "stablehlo.gather"(%105, %1772) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %1774 = stablehlo.slice %1773 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1775 = stablehlo.slice %1773 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1776 = stablehlo.broadcast_in_dim %1775, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1777 = stablehlo.multiply %1763, %1776 : tensor<1x27x32x128xf32>
    %1778 = stablehlo.constant dense<64> : tensor<i32>
    %1779 = stablehlo.broadcast_in_dim %1778, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1780 = "stablehlo.gather"(%1763, %1779) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1781 = stablehlo.negate %1780 : tensor<1x27x32x64xf32>
    %1782 = stablehlo.constant dense<0> : tensor<i32>
    %1783 = stablehlo.broadcast_in_dim %1782, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1784 = "stablehlo.gather"(%1763, %1783) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1785 = stablehlo.concatenate %1781, %1784, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1786 = stablehlo.broadcast_in_dim %1774, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1787 = stablehlo.multiply %1785, %1786 : tensor<1x27x32x128xf32>
    %1788 = stablehlo.add %1777, %1787 : tensor<1x27x32x128xf32>
    %1789 = stablehlo.broadcast_in_dim %1775, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1790 = stablehlo.multiply %1762, %1789 : tensor<1x27x32x128xf32>
    %1791 = stablehlo.constant dense<64> : tensor<i32>
    %1792 = stablehlo.broadcast_in_dim %1791, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1793 = "stablehlo.gather"(%1762, %1792) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1794 = stablehlo.negate %1793 : tensor<1x27x32x64xf32>
    %1795 = stablehlo.constant dense<0> : tensor<i32>
    %1796 = stablehlo.broadcast_in_dim %1795, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1797 = "stablehlo.gather"(%1762, %1796) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1798 = stablehlo.concatenate %1794, %1797, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1799 = stablehlo.broadcast_in_dim %1774, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1800 = stablehlo.multiply %1798, %1799 : tensor<1x27x32x128xf32>
    %1801 = stablehlo.add %1790, %1800 : tensor<1x27x32x128xf32>
    %1802 = stablehlo.slice %1755 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %1803 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %1804 = stablehlo.reshape %1803 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %1805 = stablehlo.broadcast_in_dim %1804, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %1806 = stablehlo.constant dense<0> : tensor<i32>
    %1807 = stablehlo.broadcast_in_dim %1806, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %1808 = stablehlo.compare  NE, %1805, %1807,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %1809 = stablehlo.and %1808, %1802 : tensor<1x1x27x27xi1>
    %1810 = stablehlo.convert %1809 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %1811 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1812 = stablehlo.broadcast_in_dim %1811, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1813 = stablehlo.compare  GT, %1810, %1812,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %1814 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1815 = stablehlo.broadcast_in_dim %1814, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1816 = stablehlo.convert %1815 : tensor<1x1x27x27xf32>
    %1817 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %1818 = stablehlo.broadcast_in_dim %1817, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1819 = stablehlo.select %1813, %1816, %1818 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %1820 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1821 = stablehlo.sqrt %1820 : tensor<f32>
    %1822 = stablehlo.convert %1821 : tensor<f32>
    %1823 = stablehlo.broadcast_in_dim %1822, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %1824 = stablehlo.divide %1801, %1823 : tensor<1x27x32x128xf32>
    %1825 = stablehlo.dot_general %1824, %1788, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %1826 = stablehlo.broadcast_in_dim %1819, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %1827 = stablehlo.add %1825, %1826 : tensor<1x32x27x27xf32>
    %1828 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1829 = stablehlo.reduce(%1827 init: %1828) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1830 = stablehlo.broadcast_in_dim %1829, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1831 = stablehlo.broadcast_in_dim %1830, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1832 = stablehlo.subtract %1827, %1831 : tensor<1x32x27x27xf32>
    %1833 = stablehlo.exponential %1832 : tensor<1x32x27x27xf32>
    %1834 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1835 = stablehlo.reduce(%1833 init: %1834) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1836 = stablehlo.broadcast_in_dim %1835, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1837 = stablehlo.broadcast_in_dim %1836, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1838 = stablehlo.divide %1833, %1837 : tensor<1x32x27x27xf32>
    %1839 = stablehlo.dot_general %1764, %1838, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %1840 = stablehlo.transpose %1839, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %1841 = stablehlo.reshape %1840 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %1842 = stablehlo.convert %106 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1843 = stablehlo.dot_general %1841, %1842, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1844 = stablehlo.add %1730, %1843 : tensor<1x27x4096xf32>
    %1845 = stablehlo.multiply %1844, %1844 : tensor<1x27x4096xf32>
    %1846 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1847 = stablehlo.reduce(%1845 init: %1846) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1848 = stablehlo.broadcast_in_dim %1847, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1849 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1850 = stablehlo.broadcast_in_dim %1849, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1851 = stablehlo.divide %1848, %1850 : tensor<1x27x1xf32>
    %1852 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1853 = stablehlo.broadcast_in_dim %1852, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1854 = stablehlo.add %1851, %1853 : tensor<1x27x1xf32>
    %1855 = stablehlo.sqrt %1854 : tensor<1x27x1xf32>
    %1856 = stablehlo.broadcast_in_dim %1855, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1857 = stablehlo.divide %1844, %1856 : tensor<1x27x4096xf32>
    %1858 = stablehlo.convert %107 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1859 = stablehlo.broadcast_in_dim %1858, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1860 = stablehlo.broadcast_in_dim %1859, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1861 = stablehlo.multiply %1860, %1857 : tensor<1x27x4096xf32>
    %1862 = stablehlo.convert %108 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1863 = stablehlo.dot_general %1861, %1862, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1864 = stablehlo.convert %109 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %1865 = stablehlo.dot_general %1861, %1864, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %1866 = call @silu(%1865) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %1867 = stablehlo.multiply %1863, %1866 : tensor<1x27x11008xf32>
    %1868 = stablehlo.convert %110 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %1869 = stablehlo.dot_general %1867, %1868, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %1870 = stablehlo.add %1844, %1869 : tensor<1x27x4096xf32>
    %1871 = stablehlo.multiply %1870, %1870 : tensor<1x27x4096xf32>
    %1872 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1873 = stablehlo.reduce(%1871 init: %1872) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1874 = stablehlo.broadcast_in_dim %1873, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1875 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1876 = stablehlo.broadcast_in_dim %1875, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1877 = stablehlo.divide %1874, %1876 : tensor<1x27x1xf32>
    %1878 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1879 = stablehlo.broadcast_in_dim %1878, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1880 = stablehlo.add %1877, %1879 : tensor<1x27x1xf32>
    %1881 = stablehlo.sqrt %1880 : tensor<1x27x1xf32>
    %1882 = stablehlo.broadcast_in_dim %1881, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1883 = stablehlo.divide %1870, %1882 : tensor<1x27x4096xf32>
    %1884 = stablehlo.convert %111 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1885 = stablehlo.broadcast_in_dim %1884, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1886 = stablehlo.broadcast_in_dim %1885, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %1887 = stablehlo.multiply %1886, %1883 : tensor<1x27x4096xf32>
    %1888 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %1889 = stablehlo.broadcast_in_dim %1888, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %1890 = stablehlo.broadcast_in_dim %1889, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %1891 = stablehlo.broadcast_in_dim %1889, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %1892 = stablehlo.broadcast_in_dim %1890, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %1893 = stablehlo.broadcast_in_dim %1891, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %1894 = stablehlo.compare  GE, %1892, %1893,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %1895 = stablehlo.broadcast_in_dim %1894, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %1896 = stablehlo.convert %112 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1897 = stablehlo.dot_general %1887, %1896, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1898 = stablehlo.convert %113 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1899 = stablehlo.dot_general %1887, %1898, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1900 = stablehlo.convert %114 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1901 = stablehlo.dot_general %1887, %1900, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1902 = stablehlo.reshape %1897 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1903 = stablehlo.reshape %1899 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1904 = stablehlo.reshape %1901 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %1905 = stablehlo.constant dense<0> : tensor<i32>
    %1906 = stablehlo.broadcast_in_dim %1905, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1907 = stablehlo.compare  LT, %324, %1906,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %1908 = stablehlo.constant dense<4096> : tensor<i32>
    %1909 = stablehlo.broadcast_in_dim %1908, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %1910 = stablehlo.add %324, %1909 : tensor<1x27xi32>
    %1911 = stablehlo.select %1907, %1910, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %1912 = stablehlo.broadcast_in_dim %1911, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %1913 = "stablehlo.gather"(%115, %1912) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %1914 = stablehlo.slice %1913 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1915 = stablehlo.slice %1913 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %1916 = stablehlo.broadcast_in_dim %1915, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1917 = stablehlo.multiply %1903, %1916 : tensor<1x27x32x128xf32>
    %1918 = stablehlo.constant dense<64> : tensor<i32>
    %1919 = stablehlo.broadcast_in_dim %1918, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1920 = "stablehlo.gather"(%1903, %1919) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1921 = stablehlo.negate %1920 : tensor<1x27x32x64xf32>
    %1922 = stablehlo.constant dense<0> : tensor<i32>
    %1923 = stablehlo.broadcast_in_dim %1922, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1924 = "stablehlo.gather"(%1903, %1923) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1925 = stablehlo.concatenate %1921, %1924, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1926 = stablehlo.broadcast_in_dim %1914, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1927 = stablehlo.multiply %1925, %1926 : tensor<1x27x32x128xf32>
    %1928 = stablehlo.add %1917, %1927 : tensor<1x27x32x128xf32>
    %1929 = stablehlo.broadcast_in_dim %1915, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1930 = stablehlo.multiply %1902, %1929 : tensor<1x27x32x128xf32>
    %1931 = stablehlo.constant dense<64> : tensor<i32>
    %1932 = stablehlo.broadcast_in_dim %1931, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1933 = "stablehlo.gather"(%1902, %1932) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1934 = stablehlo.negate %1933 : tensor<1x27x32x64xf32>
    %1935 = stablehlo.constant dense<0> : tensor<i32>
    %1936 = stablehlo.broadcast_in_dim %1935, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1937 = "stablehlo.gather"(%1902, %1936) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %1938 = stablehlo.concatenate %1934, %1937, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %1939 = stablehlo.broadcast_in_dim %1914, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %1940 = stablehlo.multiply %1938, %1939 : tensor<1x27x32x128xf32>
    %1941 = stablehlo.add %1930, %1940 : tensor<1x27x32x128xf32>
    %1942 = stablehlo.slice %1895 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %1943 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %1944 = stablehlo.reshape %1943 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %1945 = stablehlo.broadcast_in_dim %1944, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %1946 = stablehlo.constant dense<0> : tensor<i32>
    %1947 = stablehlo.broadcast_in_dim %1946, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %1948 = stablehlo.compare  NE, %1945, %1947,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %1949 = stablehlo.and %1948, %1942 : tensor<1x1x27x27xi1>
    %1950 = stablehlo.convert %1949 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %1951 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1952 = stablehlo.broadcast_in_dim %1951, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1953 = stablehlo.compare  GT, %1950, %1952,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %1954 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1955 = stablehlo.broadcast_in_dim %1954, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1956 = stablehlo.convert %1955 : tensor<1x1x27x27xf32>
    %1957 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %1958 = stablehlo.broadcast_in_dim %1957, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %1959 = stablehlo.select %1953, %1956, %1958 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %1960 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %1961 = stablehlo.sqrt %1960 : tensor<f32>
    %1962 = stablehlo.convert %1961 : tensor<f32>
    %1963 = stablehlo.broadcast_in_dim %1962, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %1964 = stablehlo.divide %1941, %1963 : tensor<1x27x32x128xf32>
    %1965 = stablehlo.dot_general %1964, %1928, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %1966 = stablehlo.broadcast_in_dim %1959, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %1967 = stablehlo.add %1965, %1966 : tensor<1x32x27x27xf32>
    %1968 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1969 = stablehlo.reduce(%1967 init: %1968) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1970 = stablehlo.broadcast_in_dim %1969, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1971 = stablehlo.broadcast_in_dim %1970, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1972 = stablehlo.subtract %1967, %1971 : tensor<1x32x27x27xf32>
    %1973 = stablehlo.exponential %1972 : tensor<1x32x27x27xf32>
    %1974 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1975 = stablehlo.reduce(%1973 init: %1974) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %1976 = stablehlo.broadcast_in_dim %1975, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %1977 = stablehlo.broadcast_in_dim %1976, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %1978 = stablehlo.divide %1973, %1977 : tensor<1x32x27x27xf32>
    %1979 = stablehlo.dot_general %1904, %1978, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %1980 = stablehlo.transpose %1979, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %1981 = stablehlo.reshape %1980 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %1982 = stablehlo.convert %116 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %1983 = stablehlo.dot_general %1981, %1982, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %1984 = stablehlo.add %1870, %1983 : tensor<1x27x4096xf32>
    %1985 = stablehlo.multiply %1984, %1984 : tensor<1x27x4096xf32>
    %1986 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1987 = stablehlo.reduce(%1985 init: %1986) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %1988 = stablehlo.broadcast_in_dim %1987, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %1989 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %1990 = stablehlo.broadcast_in_dim %1989, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1991 = stablehlo.divide %1988, %1990 : tensor<1x27x1xf32>
    %1992 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %1993 = stablehlo.broadcast_in_dim %1992, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %1994 = stablehlo.add %1991, %1993 : tensor<1x27x1xf32>
    %1995 = stablehlo.sqrt %1994 : tensor<1x27x1xf32>
    %1996 = stablehlo.broadcast_in_dim %1995, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %1997 = stablehlo.divide %1984, %1996 : tensor<1x27x4096xf32>
    %1998 = stablehlo.convert %117 : (tensor<4096xf16>) -> tensor<4096xf32>
    %1999 = stablehlo.broadcast_in_dim %1998, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2000 = stablehlo.broadcast_in_dim %1999, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2001 = stablehlo.multiply %2000, %1997 : tensor<1x27x4096xf32>
    %2002 = stablehlo.convert %118 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2003 = stablehlo.dot_general %2001, %2002, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2004 = stablehlo.convert %119 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2005 = stablehlo.dot_general %2001, %2004, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2006 = call @silu(%2005) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %2007 = stablehlo.multiply %2003, %2006 : tensor<1x27x11008xf32>
    %2008 = stablehlo.convert %120 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %2009 = stablehlo.dot_general %2007, %2008, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %2010 = stablehlo.add %1984, %2009 : tensor<1x27x4096xf32>
    %2011 = stablehlo.multiply %2010, %2010 : tensor<1x27x4096xf32>
    %2012 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2013 = stablehlo.reduce(%2011 init: %2012) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2014 = stablehlo.broadcast_in_dim %2013, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2015 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2016 = stablehlo.broadcast_in_dim %2015, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2017 = stablehlo.divide %2014, %2016 : tensor<1x27x1xf32>
    %2018 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2019 = stablehlo.broadcast_in_dim %2018, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2020 = stablehlo.add %2017, %2019 : tensor<1x27x1xf32>
    %2021 = stablehlo.sqrt %2020 : tensor<1x27x1xf32>
    %2022 = stablehlo.broadcast_in_dim %2021, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2023 = stablehlo.divide %2010, %2022 : tensor<1x27x4096xf32>
    %2024 = stablehlo.convert %121 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2025 = stablehlo.broadcast_in_dim %2024, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2026 = stablehlo.broadcast_in_dim %2025, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2027 = stablehlo.multiply %2026, %2023 : tensor<1x27x4096xf32>
    %2028 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %2029 = stablehlo.broadcast_in_dim %2028, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %2030 = stablehlo.broadcast_in_dim %2029, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %2031 = stablehlo.broadcast_in_dim %2029, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %2032 = stablehlo.broadcast_in_dim %2030, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %2033 = stablehlo.broadcast_in_dim %2031, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %2034 = stablehlo.compare  GE, %2032, %2033,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %2035 = stablehlo.broadcast_in_dim %2034, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %2036 = stablehlo.convert %122 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2037 = stablehlo.dot_general %2027, %2036, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2038 = stablehlo.convert %123 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2039 = stablehlo.dot_general %2027, %2038, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2040 = stablehlo.convert %124 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2041 = stablehlo.dot_general %2027, %2040, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2042 = stablehlo.reshape %2037 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2043 = stablehlo.reshape %2039 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2044 = stablehlo.reshape %2041 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2045 = stablehlo.constant dense<0> : tensor<i32>
    %2046 = stablehlo.broadcast_in_dim %2045, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2047 = stablehlo.compare  LT, %324, %2046,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %2048 = stablehlo.constant dense<4096> : tensor<i32>
    %2049 = stablehlo.broadcast_in_dim %2048, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2050 = stablehlo.add %324, %2049 : tensor<1x27xi32>
    %2051 = stablehlo.select %2047, %2050, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %2052 = stablehlo.broadcast_in_dim %2051, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %2053 = "stablehlo.gather"(%125, %2052) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %2054 = stablehlo.slice %2053 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2055 = stablehlo.slice %2053 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2056 = stablehlo.broadcast_in_dim %2055, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2057 = stablehlo.multiply %2043, %2056 : tensor<1x27x32x128xf32>
    %2058 = stablehlo.constant dense<64> : tensor<i32>
    %2059 = stablehlo.broadcast_in_dim %2058, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2060 = "stablehlo.gather"(%2043, %2059) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2061 = stablehlo.negate %2060 : tensor<1x27x32x64xf32>
    %2062 = stablehlo.constant dense<0> : tensor<i32>
    %2063 = stablehlo.broadcast_in_dim %2062, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2064 = "stablehlo.gather"(%2043, %2063) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2065 = stablehlo.concatenate %2061, %2064, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2066 = stablehlo.broadcast_in_dim %2054, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2067 = stablehlo.multiply %2065, %2066 : tensor<1x27x32x128xf32>
    %2068 = stablehlo.add %2057, %2067 : tensor<1x27x32x128xf32>
    %2069 = stablehlo.broadcast_in_dim %2055, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2070 = stablehlo.multiply %2042, %2069 : tensor<1x27x32x128xf32>
    %2071 = stablehlo.constant dense<64> : tensor<i32>
    %2072 = stablehlo.broadcast_in_dim %2071, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2073 = "stablehlo.gather"(%2042, %2072) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2074 = stablehlo.negate %2073 : tensor<1x27x32x64xf32>
    %2075 = stablehlo.constant dense<0> : tensor<i32>
    %2076 = stablehlo.broadcast_in_dim %2075, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2077 = "stablehlo.gather"(%2042, %2076) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2078 = stablehlo.concatenate %2074, %2077, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2079 = stablehlo.broadcast_in_dim %2054, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2080 = stablehlo.multiply %2078, %2079 : tensor<1x27x32x128xf32>
    %2081 = stablehlo.add %2070, %2080 : tensor<1x27x32x128xf32>
    %2082 = stablehlo.slice %2035 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %2083 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %2084 = stablehlo.reshape %2083 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %2085 = stablehlo.broadcast_in_dim %2084, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %2086 = stablehlo.constant dense<0> : tensor<i32>
    %2087 = stablehlo.broadcast_in_dim %2086, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %2088 = stablehlo.compare  NE, %2085, %2087,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %2089 = stablehlo.and %2088, %2082 : tensor<1x1x27x27xi1>
    %2090 = stablehlo.convert %2089 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %2091 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2092 = stablehlo.broadcast_in_dim %2091, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2093 = stablehlo.compare  GT, %2090, %2092,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %2094 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2095 = stablehlo.broadcast_in_dim %2094, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2096 = stablehlo.convert %2095 : tensor<1x1x27x27xf32>
    %2097 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %2098 = stablehlo.broadcast_in_dim %2097, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2099 = stablehlo.select %2093, %2096, %2098 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %2100 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2101 = stablehlo.sqrt %2100 : tensor<f32>
    %2102 = stablehlo.convert %2101 : tensor<f32>
    %2103 = stablehlo.broadcast_in_dim %2102, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %2104 = stablehlo.divide %2081, %2103 : tensor<1x27x32x128xf32>
    %2105 = stablehlo.dot_general %2104, %2068, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %2106 = stablehlo.broadcast_in_dim %2099, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %2107 = stablehlo.add %2105, %2106 : tensor<1x32x27x27xf32>
    %2108 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2109 = stablehlo.reduce(%2107 init: %2108) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2110 = stablehlo.broadcast_in_dim %2109, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2111 = stablehlo.broadcast_in_dim %2110, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2112 = stablehlo.subtract %2107, %2111 : tensor<1x32x27x27xf32>
    %2113 = stablehlo.exponential %2112 : tensor<1x32x27x27xf32>
    %2114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2115 = stablehlo.reduce(%2113 init: %2114) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2116 = stablehlo.broadcast_in_dim %2115, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2117 = stablehlo.broadcast_in_dim %2116, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2118 = stablehlo.divide %2113, %2117 : tensor<1x32x27x27xf32>
    %2119 = stablehlo.dot_general %2044, %2118, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %2120 = stablehlo.transpose %2119, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %2121 = stablehlo.reshape %2120 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %2122 = stablehlo.convert %126 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2123 = stablehlo.dot_general %2121, %2122, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2124 = stablehlo.add %2010, %2123 : tensor<1x27x4096xf32>
    %2125 = stablehlo.multiply %2124, %2124 : tensor<1x27x4096xf32>
    %2126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2127 = stablehlo.reduce(%2125 init: %2126) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2128 = stablehlo.broadcast_in_dim %2127, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2129 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2130 = stablehlo.broadcast_in_dim %2129, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2131 = stablehlo.divide %2128, %2130 : tensor<1x27x1xf32>
    %2132 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2133 = stablehlo.broadcast_in_dim %2132, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2134 = stablehlo.add %2131, %2133 : tensor<1x27x1xf32>
    %2135 = stablehlo.sqrt %2134 : tensor<1x27x1xf32>
    %2136 = stablehlo.broadcast_in_dim %2135, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2137 = stablehlo.divide %2124, %2136 : tensor<1x27x4096xf32>
    %2138 = stablehlo.convert %127 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2139 = stablehlo.broadcast_in_dim %2138, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2140 = stablehlo.broadcast_in_dim %2139, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2141 = stablehlo.multiply %2140, %2137 : tensor<1x27x4096xf32>
    %2142 = stablehlo.convert %128 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2143 = stablehlo.dot_general %2141, %2142, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2144 = stablehlo.convert %129 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2145 = stablehlo.dot_general %2141, %2144, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2146 = call @silu(%2145) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %2147 = stablehlo.multiply %2143, %2146 : tensor<1x27x11008xf32>
    %2148 = stablehlo.convert %130 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %2149 = stablehlo.dot_general %2147, %2148, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %2150 = stablehlo.add %2124, %2149 : tensor<1x27x4096xf32>
    %2151 = stablehlo.multiply %2150, %2150 : tensor<1x27x4096xf32>
    %2152 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2153 = stablehlo.reduce(%2151 init: %2152) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2154 = stablehlo.broadcast_in_dim %2153, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2155 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2156 = stablehlo.broadcast_in_dim %2155, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2157 = stablehlo.divide %2154, %2156 : tensor<1x27x1xf32>
    %2158 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2159 = stablehlo.broadcast_in_dim %2158, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2160 = stablehlo.add %2157, %2159 : tensor<1x27x1xf32>
    %2161 = stablehlo.sqrt %2160 : tensor<1x27x1xf32>
    %2162 = stablehlo.broadcast_in_dim %2161, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2163 = stablehlo.divide %2150, %2162 : tensor<1x27x4096xf32>
    %2164 = stablehlo.convert %131 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2165 = stablehlo.broadcast_in_dim %2164, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2166 = stablehlo.broadcast_in_dim %2165, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2167 = stablehlo.multiply %2166, %2163 : tensor<1x27x4096xf32>
    %2168 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %2169 = stablehlo.broadcast_in_dim %2168, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %2170 = stablehlo.broadcast_in_dim %2169, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %2171 = stablehlo.broadcast_in_dim %2169, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %2172 = stablehlo.broadcast_in_dim %2170, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %2173 = stablehlo.broadcast_in_dim %2171, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %2174 = stablehlo.compare  GE, %2172, %2173,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %2175 = stablehlo.broadcast_in_dim %2174, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %2176 = stablehlo.convert %132 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2177 = stablehlo.dot_general %2167, %2176, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2178 = stablehlo.convert %133 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2179 = stablehlo.dot_general %2167, %2178, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2180 = stablehlo.convert %134 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2181 = stablehlo.dot_general %2167, %2180, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2182 = stablehlo.reshape %2177 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2183 = stablehlo.reshape %2179 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2184 = stablehlo.reshape %2181 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2185 = stablehlo.constant dense<0> : tensor<i32>
    %2186 = stablehlo.broadcast_in_dim %2185, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2187 = stablehlo.compare  LT, %324, %2186,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %2188 = stablehlo.constant dense<4096> : tensor<i32>
    %2189 = stablehlo.broadcast_in_dim %2188, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2190 = stablehlo.add %324, %2189 : tensor<1x27xi32>
    %2191 = stablehlo.select %2187, %2190, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %2192 = stablehlo.broadcast_in_dim %2191, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %2193 = "stablehlo.gather"(%135, %2192) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %2194 = stablehlo.slice %2193 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2195 = stablehlo.slice %2193 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2196 = stablehlo.broadcast_in_dim %2195, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2197 = stablehlo.multiply %2183, %2196 : tensor<1x27x32x128xf32>
    %2198 = stablehlo.constant dense<64> : tensor<i32>
    %2199 = stablehlo.broadcast_in_dim %2198, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2200 = "stablehlo.gather"(%2183, %2199) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2201 = stablehlo.negate %2200 : tensor<1x27x32x64xf32>
    %2202 = stablehlo.constant dense<0> : tensor<i32>
    %2203 = stablehlo.broadcast_in_dim %2202, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2204 = "stablehlo.gather"(%2183, %2203) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2205 = stablehlo.concatenate %2201, %2204, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2206 = stablehlo.broadcast_in_dim %2194, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2207 = stablehlo.multiply %2205, %2206 : tensor<1x27x32x128xf32>
    %2208 = stablehlo.add %2197, %2207 : tensor<1x27x32x128xf32>
    %2209 = stablehlo.broadcast_in_dim %2195, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2210 = stablehlo.multiply %2182, %2209 : tensor<1x27x32x128xf32>
    %2211 = stablehlo.constant dense<64> : tensor<i32>
    %2212 = stablehlo.broadcast_in_dim %2211, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2213 = "stablehlo.gather"(%2182, %2212) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2214 = stablehlo.negate %2213 : tensor<1x27x32x64xf32>
    %2215 = stablehlo.constant dense<0> : tensor<i32>
    %2216 = stablehlo.broadcast_in_dim %2215, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2217 = "stablehlo.gather"(%2182, %2216) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2218 = stablehlo.concatenate %2214, %2217, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2219 = stablehlo.broadcast_in_dim %2194, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2220 = stablehlo.multiply %2218, %2219 : tensor<1x27x32x128xf32>
    %2221 = stablehlo.add %2210, %2220 : tensor<1x27x32x128xf32>
    %2222 = stablehlo.slice %2175 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %2223 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %2224 = stablehlo.reshape %2223 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %2225 = stablehlo.broadcast_in_dim %2224, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %2226 = stablehlo.constant dense<0> : tensor<i32>
    %2227 = stablehlo.broadcast_in_dim %2226, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %2228 = stablehlo.compare  NE, %2225, %2227,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %2229 = stablehlo.and %2228, %2222 : tensor<1x1x27x27xi1>
    %2230 = stablehlo.convert %2229 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %2231 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2232 = stablehlo.broadcast_in_dim %2231, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2233 = stablehlo.compare  GT, %2230, %2232,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %2234 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2235 = stablehlo.broadcast_in_dim %2234, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2236 = stablehlo.convert %2235 : tensor<1x1x27x27xf32>
    %2237 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %2238 = stablehlo.broadcast_in_dim %2237, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2239 = stablehlo.select %2233, %2236, %2238 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %2240 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2241 = stablehlo.sqrt %2240 : tensor<f32>
    %2242 = stablehlo.convert %2241 : tensor<f32>
    %2243 = stablehlo.broadcast_in_dim %2242, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %2244 = stablehlo.divide %2221, %2243 : tensor<1x27x32x128xf32>
    %2245 = stablehlo.dot_general %2244, %2208, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %2246 = stablehlo.broadcast_in_dim %2239, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %2247 = stablehlo.add %2245, %2246 : tensor<1x32x27x27xf32>
    %2248 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2249 = stablehlo.reduce(%2247 init: %2248) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2250 = stablehlo.broadcast_in_dim %2249, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2251 = stablehlo.broadcast_in_dim %2250, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2252 = stablehlo.subtract %2247, %2251 : tensor<1x32x27x27xf32>
    %2253 = stablehlo.exponential %2252 : tensor<1x32x27x27xf32>
    %2254 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2255 = stablehlo.reduce(%2253 init: %2254) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2256 = stablehlo.broadcast_in_dim %2255, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2257 = stablehlo.broadcast_in_dim %2256, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2258 = stablehlo.divide %2253, %2257 : tensor<1x32x27x27xf32>
    %2259 = stablehlo.dot_general %2184, %2258, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %2260 = stablehlo.transpose %2259, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %2261 = stablehlo.reshape %2260 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %2262 = stablehlo.convert %136 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2263 = stablehlo.dot_general %2261, %2262, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2264 = stablehlo.add %2150, %2263 : tensor<1x27x4096xf32>
    %2265 = stablehlo.multiply %2264, %2264 : tensor<1x27x4096xf32>
    %2266 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2267 = stablehlo.reduce(%2265 init: %2266) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2268 = stablehlo.broadcast_in_dim %2267, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2269 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2270 = stablehlo.broadcast_in_dim %2269, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2271 = stablehlo.divide %2268, %2270 : tensor<1x27x1xf32>
    %2272 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2273 = stablehlo.broadcast_in_dim %2272, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2274 = stablehlo.add %2271, %2273 : tensor<1x27x1xf32>
    %2275 = stablehlo.sqrt %2274 : tensor<1x27x1xf32>
    %2276 = stablehlo.broadcast_in_dim %2275, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2277 = stablehlo.divide %2264, %2276 : tensor<1x27x4096xf32>
    %2278 = stablehlo.convert %137 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2279 = stablehlo.broadcast_in_dim %2278, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2280 = stablehlo.broadcast_in_dim %2279, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2281 = stablehlo.multiply %2280, %2277 : tensor<1x27x4096xf32>
    %2282 = stablehlo.convert %138 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2283 = stablehlo.dot_general %2281, %2282, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2284 = stablehlo.convert %139 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2285 = stablehlo.dot_general %2281, %2284, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2286 = call @silu(%2285) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %2287 = stablehlo.multiply %2283, %2286 : tensor<1x27x11008xf32>
    %2288 = stablehlo.convert %140 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %2289 = stablehlo.dot_general %2287, %2288, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %2290 = stablehlo.add %2264, %2289 : tensor<1x27x4096xf32>
    %2291 = stablehlo.multiply %2290, %2290 : tensor<1x27x4096xf32>
    %2292 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2293 = stablehlo.reduce(%2291 init: %2292) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2294 = stablehlo.broadcast_in_dim %2293, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2295 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2296 = stablehlo.broadcast_in_dim %2295, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2297 = stablehlo.divide %2294, %2296 : tensor<1x27x1xf32>
    %2298 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2299 = stablehlo.broadcast_in_dim %2298, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2300 = stablehlo.add %2297, %2299 : tensor<1x27x1xf32>
    %2301 = stablehlo.sqrt %2300 : tensor<1x27x1xf32>
    %2302 = stablehlo.broadcast_in_dim %2301, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2303 = stablehlo.divide %2290, %2302 : tensor<1x27x4096xf32>
    %2304 = stablehlo.convert %141 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2305 = stablehlo.broadcast_in_dim %2304, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2306 = stablehlo.broadcast_in_dim %2305, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2307 = stablehlo.multiply %2306, %2303 : tensor<1x27x4096xf32>
    %2308 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %2309 = stablehlo.broadcast_in_dim %2308, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %2310 = stablehlo.broadcast_in_dim %2309, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %2311 = stablehlo.broadcast_in_dim %2309, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %2312 = stablehlo.broadcast_in_dim %2310, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %2313 = stablehlo.broadcast_in_dim %2311, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %2314 = stablehlo.compare  GE, %2312, %2313,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %2315 = stablehlo.broadcast_in_dim %2314, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %2316 = stablehlo.convert %142 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2317 = stablehlo.dot_general %2307, %2316, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2318 = stablehlo.convert %143 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2319 = stablehlo.dot_general %2307, %2318, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2320 = stablehlo.convert %144 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2321 = stablehlo.dot_general %2307, %2320, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2322 = stablehlo.reshape %2317 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2323 = stablehlo.reshape %2319 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2324 = stablehlo.reshape %2321 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2325 = stablehlo.constant dense<0> : tensor<i32>
    %2326 = stablehlo.broadcast_in_dim %2325, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2327 = stablehlo.compare  LT, %324, %2326,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %2328 = stablehlo.constant dense<4096> : tensor<i32>
    %2329 = stablehlo.broadcast_in_dim %2328, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2330 = stablehlo.add %324, %2329 : tensor<1x27xi32>
    %2331 = stablehlo.select %2327, %2330, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %2332 = stablehlo.broadcast_in_dim %2331, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %2333 = "stablehlo.gather"(%145, %2332) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %2334 = stablehlo.slice %2333 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2335 = stablehlo.slice %2333 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2336 = stablehlo.broadcast_in_dim %2335, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2337 = stablehlo.multiply %2323, %2336 : tensor<1x27x32x128xf32>
    %2338 = stablehlo.constant dense<64> : tensor<i32>
    %2339 = stablehlo.broadcast_in_dim %2338, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2340 = "stablehlo.gather"(%2323, %2339) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2341 = stablehlo.negate %2340 : tensor<1x27x32x64xf32>
    %2342 = stablehlo.constant dense<0> : tensor<i32>
    %2343 = stablehlo.broadcast_in_dim %2342, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2344 = "stablehlo.gather"(%2323, %2343) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2345 = stablehlo.concatenate %2341, %2344, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2346 = stablehlo.broadcast_in_dim %2334, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2347 = stablehlo.multiply %2345, %2346 : tensor<1x27x32x128xf32>
    %2348 = stablehlo.add %2337, %2347 : tensor<1x27x32x128xf32>
    %2349 = stablehlo.broadcast_in_dim %2335, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2350 = stablehlo.multiply %2322, %2349 : tensor<1x27x32x128xf32>
    %2351 = stablehlo.constant dense<64> : tensor<i32>
    %2352 = stablehlo.broadcast_in_dim %2351, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2353 = "stablehlo.gather"(%2322, %2352) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2354 = stablehlo.negate %2353 : tensor<1x27x32x64xf32>
    %2355 = stablehlo.constant dense<0> : tensor<i32>
    %2356 = stablehlo.broadcast_in_dim %2355, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2357 = "stablehlo.gather"(%2322, %2356) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2358 = stablehlo.concatenate %2354, %2357, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2359 = stablehlo.broadcast_in_dim %2334, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2360 = stablehlo.multiply %2358, %2359 : tensor<1x27x32x128xf32>
    %2361 = stablehlo.add %2350, %2360 : tensor<1x27x32x128xf32>
    %2362 = stablehlo.slice %2315 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %2363 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %2364 = stablehlo.reshape %2363 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %2365 = stablehlo.broadcast_in_dim %2364, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %2366 = stablehlo.constant dense<0> : tensor<i32>
    %2367 = stablehlo.broadcast_in_dim %2366, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %2368 = stablehlo.compare  NE, %2365, %2367,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %2369 = stablehlo.and %2368, %2362 : tensor<1x1x27x27xi1>
    %2370 = stablehlo.convert %2369 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %2371 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2372 = stablehlo.broadcast_in_dim %2371, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2373 = stablehlo.compare  GT, %2370, %2372,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %2374 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2375 = stablehlo.broadcast_in_dim %2374, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2376 = stablehlo.convert %2375 : tensor<1x1x27x27xf32>
    %2377 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %2378 = stablehlo.broadcast_in_dim %2377, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2379 = stablehlo.select %2373, %2376, %2378 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %2380 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2381 = stablehlo.sqrt %2380 : tensor<f32>
    %2382 = stablehlo.convert %2381 : tensor<f32>
    %2383 = stablehlo.broadcast_in_dim %2382, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %2384 = stablehlo.divide %2361, %2383 : tensor<1x27x32x128xf32>
    %2385 = stablehlo.dot_general %2384, %2348, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %2386 = stablehlo.broadcast_in_dim %2379, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %2387 = stablehlo.add %2385, %2386 : tensor<1x32x27x27xf32>
    %2388 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2389 = stablehlo.reduce(%2387 init: %2388) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2390 = stablehlo.broadcast_in_dim %2389, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2391 = stablehlo.broadcast_in_dim %2390, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2392 = stablehlo.subtract %2387, %2391 : tensor<1x32x27x27xf32>
    %2393 = stablehlo.exponential %2392 : tensor<1x32x27x27xf32>
    %2394 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2395 = stablehlo.reduce(%2393 init: %2394) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2396 = stablehlo.broadcast_in_dim %2395, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2397 = stablehlo.broadcast_in_dim %2396, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2398 = stablehlo.divide %2393, %2397 : tensor<1x32x27x27xf32>
    %2399 = stablehlo.dot_general %2324, %2398, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %2400 = stablehlo.transpose %2399, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %2401 = stablehlo.reshape %2400 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %2402 = stablehlo.convert %146 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2403 = stablehlo.dot_general %2401, %2402, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2404 = stablehlo.add %2290, %2403 : tensor<1x27x4096xf32>
    %2405 = stablehlo.multiply %2404, %2404 : tensor<1x27x4096xf32>
    %2406 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2407 = stablehlo.reduce(%2405 init: %2406) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2408 = stablehlo.broadcast_in_dim %2407, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2409 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2410 = stablehlo.broadcast_in_dim %2409, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2411 = stablehlo.divide %2408, %2410 : tensor<1x27x1xf32>
    %2412 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2413 = stablehlo.broadcast_in_dim %2412, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2414 = stablehlo.add %2411, %2413 : tensor<1x27x1xf32>
    %2415 = stablehlo.sqrt %2414 : tensor<1x27x1xf32>
    %2416 = stablehlo.broadcast_in_dim %2415, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2417 = stablehlo.divide %2404, %2416 : tensor<1x27x4096xf32>
    %2418 = stablehlo.convert %147 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2419 = stablehlo.broadcast_in_dim %2418, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2420 = stablehlo.broadcast_in_dim %2419, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2421 = stablehlo.multiply %2420, %2417 : tensor<1x27x4096xf32>
    %2422 = stablehlo.convert %148 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2423 = stablehlo.dot_general %2421, %2422, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2424 = stablehlo.convert %149 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2425 = stablehlo.dot_general %2421, %2424, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2426 = call @silu(%2425) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %2427 = stablehlo.multiply %2423, %2426 : tensor<1x27x11008xf32>
    %2428 = stablehlo.convert %150 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %2429 = stablehlo.dot_general %2427, %2428, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %2430 = stablehlo.add %2404, %2429 : tensor<1x27x4096xf32>
    %2431 = stablehlo.multiply %2430, %2430 : tensor<1x27x4096xf32>
    %2432 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2433 = stablehlo.reduce(%2431 init: %2432) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2434 = stablehlo.broadcast_in_dim %2433, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2435 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2436 = stablehlo.broadcast_in_dim %2435, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2437 = stablehlo.divide %2434, %2436 : tensor<1x27x1xf32>
    %2438 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2439 = stablehlo.broadcast_in_dim %2438, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2440 = stablehlo.add %2437, %2439 : tensor<1x27x1xf32>
    %2441 = stablehlo.sqrt %2440 : tensor<1x27x1xf32>
    %2442 = stablehlo.broadcast_in_dim %2441, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2443 = stablehlo.divide %2430, %2442 : tensor<1x27x4096xf32>
    %2444 = stablehlo.convert %151 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2445 = stablehlo.broadcast_in_dim %2444, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2446 = stablehlo.broadcast_in_dim %2445, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2447 = stablehlo.multiply %2446, %2443 : tensor<1x27x4096xf32>
    %2448 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %2449 = stablehlo.broadcast_in_dim %2448, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %2450 = stablehlo.broadcast_in_dim %2449, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %2451 = stablehlo.broadcast_in_dim %2449, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %2452 = stablehlo.broadcast_in_dim %2450, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %2453 = stablehlo.broadcast_in_dim %2451, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %2454 = stablehlo.compare  GE, %2452, %2453,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %2455 = stablehlo.broadcast_in_dim %2454, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %2456 = stablehlo.convert %152 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2457 = stablehlo.dot_general %2447, %2456, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2458 = stablehlo.convert %153 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2459 = stablehlo.dot_general %2447, %2458, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2460 = stablehlo.convert %154 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2461 = stablehlo.dot_general %2447, %2460, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2462 = stablehlo.reshape %2457 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2463 = stablehlo.reshape %2459 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2464 = stablehlo.reshape %2461 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2465 = stablehlo.constant dense<0> : tensor<i32>
    %2466 = stablehlo.broadcast_in_dim %2465, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2467 = stablehlo.compare  LT, %324, %2466,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %2468 = stablehlo.constant dense<4096> : tensor<i32>
    %2469 = stablehlo.broadcast_in_dim %2468, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2470 = stablehlo.add %324, %2469 : tensor<1x27xi32>
    %2471 = stablehlo.select %2467, %2470, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %2472 = stablehlo.broadcast_in_dim %2471, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %2473 = "stablehlo.gather"(%155, %2472) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %2474 = stablehlo.slice %2473 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2475 = stablehlo.slice %2473 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2476 = stablehlo.broadcast_in_dim %2475, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2477 = stablehlo.multiply %2463, %2476 : tensor<1x27x32x128xf32>
    %2478 = stablehlo.constant dense<64> : tensor<i32>
    %2479 = stablehlo.broadcast_in_dim %2478, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2480 = "stablehlo.gather"(%2463, %2479) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2481 = stablehlo.negate %2480 : tensor<1x27x32x64xf32>
    %2482 = stablehlo.constant dense<0> : tensor<i32>
    %2483 = stablehlo.broadcast_in_dim %2482, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2484 = "stablehlo.gather"(%2463, %2483) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2485 = stablehlo.concatenate %2481, %2484, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2486 = stablehlo.broadcast_in_dim %2474, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2487 = stablehlo.multiply %2485, %2486 : tensor<1x27x32x128xf32>
    %2488 = stablehlo.add %2477, %2487 : tensor<1x27x32x128xf32>
    %2489 = stablehlo.broadcast_in_dim %2475, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2490 = stablehlo.multiply %2462, %2489 : tensor<1x27x32x128xf32>
    %2491 = stablehlo.constant dense<64> : tensor<i32>
    %2492 = stablehlo.broadcast_in_dim %2491, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2493 = "stablehlo.gather"(%2462, %2492) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2494 = stablehlo.negate %2493 : tensor<1x27x32x64xf32>
    %2495 = stablehlo.constant dense<0> : tensor<i32>
    %2496 = stablehlo.broadcast_in_dim %2495, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2497 = "stablehlo.gather"(%2462, %2496) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2498 = stablehlo.concatenate %2494, %2497, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2499 = stablehlo.broadcast_in_dim %2474, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2500 = stablehlo.multiply %2498, %2499 : tensor<1x27x32x128xf32>
    %2501 = stablehlo.add %2490, %2500 : tensor<1x27x32x128xf32>
    %2502 = stablehlo.slice %2455 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %2503 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %2504 = stablehlo.reshape %2503 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %2505 = stablehlo.broadcast_in_dim %2504, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %2506 = stablehlo.constant dense<0> : tensor<i32>
    %2507 = stablehlo.broadcast_in_dim %2506, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %2508 = stablehlo.compare  NE, %2505, %2507,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %2509 = stablehlo.and %2508, %2502 : tensor<1x1x27x27xi1>
    %2510 = stablehlo.convert %2509 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %2511 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2512 = stablehlo.broadcast_in_dim %2511, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2513 = stablehlo.compare  GT, %2510, %2512,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %2514 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2515 = stablehlo.broadcast_in_dim %2514, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2516 = stablehlo.convert %2515 : tensor<1x1x27x27xf32>
    %2517 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %2518 = stablehlo.broadcast_in_dim %2517, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2519 = stablehlo.select %2513, %2516, %2518 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %2520 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2521 = stablehlo.sqrt %2520 : tensor<f32>
    %2522 = stablehlo.convert %2521 : tensor<f32>
    %2523 = stablehlo.broadcast_in_dim %2522, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %2524 = stablehlo.divide %2501, %2523 : tensor<1x27x32x128xf32>
    %2525 = stablehlo.dot_general %2524, %2488, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %2526 = stablehlo.broadcast_in_dim %2519, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %2527 = stablehlo.add %2525, %2526 : tensor<1x32x27x27xf32>
    %2528 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2529 = stablehlo.reduce(%2527 init: %2528) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2530 = stablehlo.broadcast_in_dim %2529, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2531 = stablehlo.broadcast_in_dim %2530, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2532 = stablehlo.subtract %2527, %2531 : tensor<1x32x27x27xf32>
    %2533 = stablehlo.exponential %2532 : tensor<1x32x27x27xf32>
    %2534 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2535 = stablehlo.reduce(%2533 init: %2534) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2536 = stablehlo.broadcast_in_dim %2535, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2537 = stablehlo.broadcast_in_dim %2536, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2538 = stablehlo.divide %2533, %2537 : tensor<1x32x27x27xf32>
    %2539 = stablehlo.dot_general %2464, %2538, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %2540 = stablehlo.transpose %2539, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %2541 = stablehlo.reshape %2540 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %2542 = stablehlo.convert %156 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2543 = stablehlo.dot_general %2541, %2542, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2544 = stablehlo.add %2430, %2543 : tensor<1x27x4096xf32>
    %2545 = stablehlo.multiply %2544, %2544 : tensor<1x27x4096xf32>
    %2546 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2547 = stablehlo.reduce(%2545 init: %2546) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2548 = stablehlo.broadcast_in_dim %2547, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2549 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2550 = stablehlo.broadcast_in_dim %2549, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2551 = stablehlo.divide %2548, %2550 : tensor<1x27x1xf32>
    %2552 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2553 = stablehlo.broadcast_in_dim %2552, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2554 = stablehlo.add %2551, %2553 : tensor<1x27x1xf32>
    %2555 = stablehlo.sqrt %2554 : tensor<1x27x1xf32>
    %2556 = stablehlo.broadcast_in_dim %2555, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2557 = stablehlo.divide %2544, %2556 : tensor<1x27x4096xf32>
    %2558 = stablehlo.convert %157 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2559 = stablehlo.broadcast_in_dim %2558, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2560 = stablehlo.broadcast_in_dim %2559, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2561 = stablehlo.multiply %2560, %2557 : tensor<1x27x4096xf32>
    %2562 = stablehlo.convert %158 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2563 = stablehlo.dot_general %2561, %2562, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2564 = stablehlo.convert %159 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2565 = stablehlo.dot_general %2561, %2564, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2566 = call @silu(%2565) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %2567 = stablehlo.multiply %2563, %2566 : tensor<1x27x11008xf32>
    %2568 = stablehlo.convert %160 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %2569 = stablehlo.dot_general %2567, %2568, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %2570 = stablehlo.add %2544, %2569 : tensor<1x27x4096xf32>
    %2571 = stablehlo.multiply %2570, %2570 : tensor<1x27x4096xf32>
    %2572 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2573 = stablehlo.reduce(%2571 init: %2572) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2574 = stablehlo.broadcast_in_dim %2573, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2575 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2576 = stablehlo.broadcast_in_dim %2575, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2577 = stablehlo.divide %2574, %2576 : tensor<1x27x1xf32>
    %2578 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2579 = stablehlo.broadcast_in_dim %2578, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2580 = stablehlo.add %2577, %2579 : tensor<1x27x1xf32>
    %2581 = stablehlo.sqrt %2580 : tensor<1x27x1xf32>
    %2582 = stablehlo.broadcast_in_dim %2581, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2583 = stablehlo.divide %2570, %2582 : tensor<1x27x4096xf32>
    %2584 = stablehlo.convert %161 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2585 = stablehlo.broadcast_in_dim %2584, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2586 = stablehlo.broadcast_in_dim %2585, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2587 = stablehlo.multiply %2586, %2583 : tensor<1x27x4096xf32>
    %2588 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %2589 = stablehlo.broadcast_in_dim %2588, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %2590 = stablehlo.broadcast_in_dim %2589, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %2591 = stablehlo.broadcast_in_dim %2589, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %2592 = stablehlo.broadcast_in_dim %2590, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %2593 = stablehlo.broadcast_in_dim %2591, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %2594 = stablehlo.compare  GE, %2592, %2593,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %2595 = stablehlo.broadcast_in_dim %2594, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %2596 = stablehlo.convert %162 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2597 = stablehlo.dot_general %2587, %2596, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2598 = stablehlo.convert %163 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2599 = stablehlo.dot_general %2587, %2598, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2600 = stablehlo.convert %164 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2601 = stablehlo.dot_general %2587, %2600, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2602 = stablehlo.reshape %2597 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2603 = stablehlo.reshape %2599 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2604 = stablehlo.reshape %2601 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2605 = stablehlo.constant dense<0> : tensor<i32>
    %2606 = stablehlo.broadcast_in_dim %2605, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2607 = stablehlo.compare  LT, %324, %2606,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %2608 = stablehlo.constant dense<4096> : tensor<i32>
    %2609 = stablehlo.broadcast_in_dim %2608, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2610 = stablehlo.add %324, %2609 : tensor<1x27xi32>
    %2611 = stablehlo.select %2607, %2610, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %2612 = stablehlo.broadcast_in_dim %2611, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %2613 = "stablehlo.gather"(%165, %2612) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %2614 = stablehlo.slice %2613 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2615 = stablehlo.slice %2613 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2616 = stablehlo.broadcast_in_dim %2615, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2617 = stablehlo.multiply %2603, %2616 : tensor<1x27x32x128xf32>
    %2618 = stablehlo.constant dense<64> : tensor<i32>
    %2619 = stablehlo.broadcast_in_dim %2618, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2620 = "stablehlo.gather"(%2603, %2619) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2621 = stablehlo.negate %2620 : tensor<1x27x32x64xf32>
    %2622 = stablehlo.constant dense<0> : tensor<i32>
    %2623 = stablehlo.broadcast_in_dim %2622, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2624 = "stablehlo.gather"(%2603, %2623) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2625 = stablehlo.concatenate %2621, %2624, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2626 = stablehlo.broadcast_in_dim %2614, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2627 = stablehlo.multiply %2625, %2626 : tensor<1x27x32x128xf32>
    %2628 = stablehlo.add %2617, %2627 : tensor<1x27x32x128xf32>
    %2629 = stablehlo.broadcast_in_dim %2615, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2630 = stablehlo.multiply %2602, %2629 : tensor<1x27x32x128xf32>
    %2631 = stablehlo.constant dense<64> : tensor<i32>
    %2632 = stablehlo.broadcast_in_dim %2631, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2633 = "stablehlo.gather"(%2602, %2632) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2634 = stablehlo.negate %2633 : tensor<1x27x32x64xf32>
    %2635 = stablehlo.constant dense<0> : tensor<i32>
    %2636 = stablehlo.broadcast_in_dim %2635, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2637 = "stablehlo.gather"(%2602, %2636) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2638 = stablehlo.concatenate %2634, %2637, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2639 = stablehlo.broadcast_in_dim %2614, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2640 = stablehlo.multiply %2638, %2639 : tensor<1x27x32x128xf32>
    %2641 = stablehlo.add %2630, %2640 : tensor<1x27x32x128xf32>
    %2642 = stablehlo.slice %2595 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %2643 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %2644 = stablehlo.reshape %2643 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %2645 = stablehlo.broadcast_in_dim %2644, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %2646 = stablehlo.constant dense<0> : tensor<i32>
    %2647 = stablehlo.broadcast_in_dim %2646, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %2648 = stablehlo.compare  NE, %2645, %2647,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %2649 = stablehlo.and %2648, %2642 : tensor<1x1x27x27xi1>
    %2650 = stablehlo.convert %2649 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %2651 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2652 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2653 = stablehlo.compare  GT, %2650, %2652,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %2654 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2655 = stablehlo.broadcast_in_dim %2654, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2656 = stablehlo.convert %2655 : tensor<1x1x27x27xf32>
    %2657 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %2658 = stablehlo.broadcast_in_dim %2657, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2659 = stablehlo.select %2653, %2656, %2658 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %2660 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2661 = stablehlo.sqrt %2660 : tensor<f32>
    %2662 = stablehlo.convert %2661 : tensor<f32>
    %2663 = stablehlo.broadcast_in_dim %2662, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %2664 = stablehlo.divide %2641, %2663 : tensor<1x27x32x128xf32>
    %2665 = stablehlo.dot_general %2664, %2628, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %2666 = stablehlo.broadcast_in_dim %2659, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %2667 = stablehlo.add %2665, %2666 : tensor<1x32x27x27xf32>
    %2668 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2669 = stablehlo.reduce(%2667 init: %2668) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2670 = stablehlo.broadcast_in_dim %2669, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2671 = stablehlo.broadcast_in_dim %2670, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2672 = stablehlo.subtract %2667, %2671 : tensor<1x32x27x27xf32>
    %2673 = stablehlo.exponential %2672 : tensor<1x32x27x27xf32>
    %2674 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2675 = stablehlo.reduce(%2673 init: %2674) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2676 = stablehlo.broadcast_in_dim %2675, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2677 = stablehlo.broadcast_in_dim %2676, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2678 = stablehlo.divide %2673, %2677 : tensor<1x32x27x27xf32>
    %2679 = stablehlo.dot_general %2604, %2678, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %2680 = stablehlo.transpose %2679, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %2681 = stablehlo.reshape %2680 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %2682 = stablehlo.convert %166 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2683 = stablehlo.dot_general %2681, %2682, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2684 = stablehlo.add %2570, %2683 : tensor<1x27x4096xf32>
    %2685 = stablehlo.multiply %2684, %2684 : tensor<1x27x4096xf32>
    %2686 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2687 = stablehlo.reduce(%2685 init: %2686) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2688 = stablehlo.broadcast_in_dim %2687, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2689 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2690 = stablehlo.broadcast_in_dim %2689, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2691 = stablehlo.divide %2688, %2690 : tensor<1x27x1xf32>
    %2692 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2693 = stablehlo.broadcast_in_dim %2692, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2694 = stablehlo.add %2691, %2693 : tensor<1x27x1xf32>
    %2695 = stablehlo.sqrt %2694 : tensor<1x27x1xf32>
    %2696 = stablehlo.broadcast_in_dim %2695, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2697 = stablehlo.divide %2684, %2696 : tensor<1x27x4096xf32>
    %2698 = stablehlo.convert %167 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2699 = stablehlo.broadcast_in_dim %2698, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2700 = stablehlo.broadcast_in_dim %2699, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2701 = stablehlo.multiply %2700, %2697 : tensor<1x27x4096xf32>
    %2702 = stablehlo.convert %168 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2703 = stablehlo.dot_general %2701, %2702, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2704 = stablehlo.convert %169 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2705 = stablehlo.dot_general %2701, %2704, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2706 = call @silu(%2705) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %2707 = stablehlo.multiply %2703, %2706 : tensor<1x27x11008xf32>
    %2708 = stablehlo.convert %170 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %2709 = stablehlo.dot_general %2707, %2708, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %2710 = stablehlo.add %2684, %2709 : tensor<1x27x4096xf32>
    %2711 = stablehlo.multiply %2710, %2710 : tensor<1x27x4096xf32>
    %2712 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2713 = stablehlo.reduce(%2711 init: %2712) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2714 = stablehlo.broadcast_in_dim %2713, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2715 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2716 = stablehlo.broadcast_in_dim %2715, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2717 = stablehlo.divide %2714, %2716 : tensor<1x27x1xf32>
    %2718 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2719 = stablehlo.broadcast_in_dim %2718, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2720 = stablehlo.add %2717, %2719 : tensor<1x27x1xf32>
    %2721 = stablehlo.sqrt %2720 : tensor<1x27x1xf32>
    %2722 = stablehlo.broadcast_in_dim %2721, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2723 = stablehlo.divide %2710, %2722 : tensor<1x27x4096xf32>
    %2724 = stablehlo.convert %171 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2725 = stablehlo.broadcast_in_dim %2724, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2726 = stablehlo.broadcast_in_dim %2725, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2727 = stablehlo.multiply %2726, %2723 : tensor<1x27x4096xf32>
    %2728 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %2729 = stablehlo.broadcast_in_dim %2728, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %2730 = stablehlo.broadcast_in_dim %2729, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %2731 = stablehlo.broadcast_in_dim %2729, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %2732 = stablehlo.broadcast_in_dim %2730, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %2733 = stablehlo.broadcast_in_dim %2731, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %2734 = stablehlo.compare  GE, %2732, %2733,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %2735 = stablehlo.broadcast_in_dim %2734, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %2736 = stablehlo.convert %172 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2737 = stablehlo.dot_general %2727, %2736, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2738 = stablehlo.convert %173 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2739 = stablehlo.dot_general %2727, %2738, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2740 = stablehlo.convert %174 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2741 = stablehlo.dot_general %2727, %2740, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2742 = stablehlo.reshape %2737 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2743 = stablehlo.reshape %2739 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2744 = stablehlo.reshape %2741 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2745 = stablehlo.constant dense<0> : tensor<i32>
    %2746 = stablehlo.broadcast_in_dim %2745, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2747 = stablehlo.compare  LT, %324, %2746,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %2748 = stablehlo.constant dense<4096> : tensor<i32>
    %2749 = stablehlo.broadcast_in_dim %2748, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2750 = stablehlo.add %324, %2749 : tensor<1x27xi32>
    %2751 = stablehlo.select %2747, %2750, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %2752 = stablehlo.broadcast_in_dim %2751, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %2753 = "stablehlo.gather"(%175, %2752) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %2754 = stablehlo.slice %2753 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2755 = stablehlo.slice %2753 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2756 = stablehlo.broadcast_in_dim %2755, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2757 = stablehlo.multiply %2743, %2756 : tensor<1x27x32x128xf32>
    %2758 = stablehlo.constant dense<64> : tensor<i32>
    %2759 = stablehlo.broadcast_in_dim %2758, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2760 = "stablehlo.gather"(%2743, %2759) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2761 = stablehlo.negate %2760 : tensor<1x27x32x64xf32>
    %2762 = stablehlo.constant dense<0> : tensor<i32>
    %2763 = stablehlo.broadcast_in_dim %2762, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2764 = "stablehlo.gather"(%2743, %2763) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2765 = stablehlo.concatenate %2761, %2764, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2766 = stablehlo.broadcast_in_dim %2754, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2767 = stablehlo.multiply %2765, %2766 : tensor<1x27x32x128xf32>
    %2768 = stablehlo.add %2757, %2767 : tensor<1x27x32x128xf32>
    %2769 = stablehlo.broadcast_in_dim %2755, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2770 = stablehlo.multiply %2742, %2769 : tensor<1x27x32x128xf32>
    %2771 = stablehlo.constant dense<64> : tensor<i32>
    %2772 = stablehlo.broadcast_in_dim %2771, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2773 = "stablehlo.gather"(%2742, %2772) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2774 = stablehlo.negate %2773 : tensor<1x27x32x64xf32>
    %2775 = stablehlo.constant dense<0> : tensor<i32>
    %2776 = stablehlo.broadcast_in_dim %2775, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2777 = "stablehlo.gather"(%2742, %2776) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2778 = stablehlo.concatenate %2774, %2777, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2779 = stablehlo.broadcast_in_dim %2754, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2780 = stablehlo.multiply %2778, %2779 : tensor<1x27x32x128xf32>
    %2781 = stablehlo.add %2770, %2780 : tensor<1x27x32x128xf32>
    %2782 = stablehlo.slice %2735 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %2783 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %2784 = stablehlo.reshape %2783 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %2785 = stablehlo.broadcast_in_dim %2784, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %2786 = stablehlo.constant dense<0> : tensor<i32>
    %2787 = stablehlo.broadcast_in_dim %2786, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %2788 = stablehlo.compare  NE, %2785, %2787,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %2789 = stablehlo.and %2788, %2782 : tensor<1x1x27x27xi1>
    %2790 = stablehlo.convert %2789 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %2791 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2792 = stablehlo.broadcast_in_dim %2791, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2793 = stablehlo.compare  GT, %2790, %2792,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %2794 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2795 = stablehlo.broadcast_in_dim %2794, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2796 = stablehlo.convert %2795 : tensor<1x1x27x27xf32>
    %2797 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %2798 = stablehlo.broadcast_in_dim %2797, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2799 = stablehlo.select %2793, %2796, %2798 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %2800 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2801 = stablehlo.sqrt %2800 : tensor<f32>
    %2802 = stablehlo.convert %2801 : tensor<f32>
    %2803 = stablehlo.broadcast_in_dim %2802, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %2804 = stablehlo.divide %2781, %2803 : tensor<1x27x32x128xf32>
    %2805 = stablehlo.dot_general %2804, %2768, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %2806 = stablehlo.broadcast_in_dim %2799, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %2807 = stablehlo.add %2805, %2806 : tensor<1x32x27x27xf32>
    %2808 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2809 = stablehlo.reduce(%2807 init: %2808) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2810 = stablehlo.broadcast_in_dim %2809, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2811 = stablehlo.broadcast_in_dim %2810, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2812 = stablehlo.subtract %2807, %2811 : tensor<1x32x27x27xf32>
    %2813 = stablehlo.exponential %2812 : tensor<1x32x27x27xf32>
    %2814 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2815 = stablehlo.reduce(%2813 init: %2814) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2816 = stablehlo.broadcast_in_dim %2815, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2817 = stablehlo.broadcast_in_dim %2816, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2818 = stablehlo.divide %2813, %2817 : tensor<1x32x27x27xf32>
    %2819 = stablehlo.dot_general %2744, %2818, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %2820 = stablehlo.transpose %2819, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %2821 = stablehlo.reshape %2820 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %2822 = stablehlo.convert %176 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2823 = stablehlo.dot_general %2821, %2822, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2824 = stablehlo.add %2710, %2823 : tensor<1x27x4096xf32>
    %2825 = stablehlo.multiply %2824, %2824 : tensor<1x27x4096xf32>
    %2826 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2827 = stablehlo.reduce(%2825 init: %2826) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2828 = stablehlo.broadcast_in_dim %2827, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2829 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2830 = stablehlo.broadcast_in_dim %2829, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2831 = stablehlo.divide %2828, %2830 : tensor<1x27x1xf32>
    %2832 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2833 = stablehlo.broadcast_in_dim %2832, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2834 = stablehlo.add %2831, %2833 : tensor<1x27x1xf32>
    %2835 = stablehlo.sqrt %2834 : tensor<1x27x1xf32>
    %2836 = stablehlo.broadcast_in_dim %2835, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2837 = stablehlo.divide %2824, %2836 : tensor<1x27x4096xf32>
    %2838 = stablehlo.convert %177 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2839 = stablehlo.broadcast_in_dim %2838, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2840 = stablehlo.broadcast_in_dim %2839, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2841 = stablehlo.multiply %2840, %2837 : tensor<1x27x4096xf32>
    %2842 = stablehlo.convert %178 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2843 = stablehlo.dot_general %2841, %2842, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2844 = stablehlo.convert %179 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2845 = stablehlo.dot_general %2841, %2844, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2846 = call @silu(%2845) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %2847 = stablehlo.multiply %2843, %2846 : tensor<1x27x11008xf32>
    %2848 = stablehlo.convert %180 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %2849 = stablehlo.dot_general %2847, %2848, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %2850 = stablehlo.add %2824, %2849 : tensor<1x27x4096xf32>
    %2851 = stablehlo.multiply %2850, %2850 : tensor<1x27x4096xf32>
    %2852 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2853 = stablehlo.reduce(%2851 init: %2852) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2854 = stablehlo.broadcast_in_dim %2853, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2855 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2856 = stablehlo.broadcast_in_dim %2855, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2857 = stablehlo.divide %2854, %2856 : tensor<1x27x1xf32>
    %2858 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2859 = stablehlo.broadcast_in_dim %2858, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2860 = stablehlo.add %2857, %2859 : tensor<1x27x1xf32>
    %2861 = stablehlo.sqrt %2860 : tensor<1x27x1xf32>
    %2862 = stablehlo.broadcast_in_dim %2861, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2863 = stablehlo.divide %2850, %2862 : tensor<1x27x4096xf32>
    %2864 = stablehlo.convert %181 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2865 = stablehlo.broadcast_in_dim %2864, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2866 = stablehlo.broadcast_in_dim %2865, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2867 = stablehlo.multiply %2866, %2863 : tensor<1x27x4096xf32>
    %2868 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %2869 = stablehlo.broadcast_in_dim %2868, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %2870 = stablehlo.broadcast_in_dim %2869, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %2871 = stablehlo.broadcast_in_dim %2869, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %2872 = stablehlo.broadcast_in_dim %2870, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %2873 = stablehlo.broadcast_in_dim %2871, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %2874 = stablehlo.compare  GE, %2872, %2873,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %2875 = stablehlo.broadcast_in_dim %2874, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %2876 = stablehlo.convert %182 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2877 = stablehlo.dot_general %2867, %2876, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2878 = stablehlo.convert %183 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2879 = stablehlo.dot_general %2867, %2878, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2880 = stablehlo.convert %184 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2881 = stablehlo.dot_general %2867, %2880, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2882 = stablehlo.reshape %2877 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2883 = stablehlo.reshape %2879 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2884 = stablehlo.reshape %2881 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %2885 = stablehlo.constant dense<0> : tensor<i32>
    %2886 = stablehlo.broadcast_in_dim %2885, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2887 = stablehlo.compare  LT, %324, %2886,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %2888 = stablehlo.constant dense<4096> : tensor<i32>
    %2889 = stablehlo.broadcast_in_dim %2888, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2890 = stablehlo.add %324, %2889 : tensor<1x27xi32>
    %2891 = stablehlo.select %2887, %2890, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %2892 = stablehlo.broadcast_in_dim %2891, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %2893 = "stablehlo.gather"(%185, %2892) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %2894 = stablehlo.slice %2893 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2895 = stablehlo.slice %2893 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %2896 = stablehlo.broadcast_in_dim %2895, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2897 = stablehlo.multiply %2883, %2896 : tensor<1x27x32x128xf32>
    %2898 = stablehlo.constant dense<64> : tensor<i32>
    %2899 = stablehlo.broadcast_in_dim %2898, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2900 = "stablehlo.gather"(%2883, %2899) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2901 = stablehlo.negate %2900 : tensor<1x27x32x64xf32>
    %2902 = stablehlo.constant dense<0> : tensor<i32>
    %2903 = stablehlo.broadcast_in_dim %2902, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2904 = "stablehlo.gather"(%2883, %2903) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2905 = stablehlo.concatenate %2901, %2904, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2906 = stablehlo.broadcast_in_dim %2894, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2907 = stablehlo.multiply %2905, %2906 : tensor<1x27x32x128xf32>
    %2908 = stablehlo.add %2897, %2907 : tensor<1x27x32x128xf32>
    %2909 = stablehlo.broadcast_in_dim %2895, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2910 = stablehlo.multiply %2882, %2909 : tensor<1x27x32x128xf32>
    %2911 = stablehlo.constant dense<64> : tensor<i32>
    %2912 = stablehlo.broadcast_in_dim %2911, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2913 = "stablehlo.gather"(%2882, %2912) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2914 = stablehlo.negate %2913 : tensor<1x27x32x64xf32>
    %2915 = stablehlo.constant dense<0> : tensor<i32>
    %2916 = stablehlo.broadcast_in_dim %2915, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2917 = "stablehlo.gather"(%2882, %2916) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %2918 = stablehlo.concatenate %2914, %2917, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %2919 = stablehlo.broadcast_in_dim %2894, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %2920 = stablehlo.multiply %2918, %2919 : tensor<1x27x32x128xf32>
    %2921 = stablehlo.add %2910, %2920 : tensor<1x27x32x128xf32>
    %2922 = stablehlo.slice %2875 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %2923 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %2924 = stablehlo.reshape %2923 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %2925 = stablehlo.broadcast_in_dim %2924, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %2926 = stablehlo.constant dense<0> : tensor<i32>
    %2927 = stablehlo.broadcast_in_dim %2926, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %2928 = stablehlo.compare  NE, %2925, %2927,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %2929 = stablehlo.and %2928, %2922 : tensor<1x1x27x27xi1>
    %2930 = stablehlo.convert %2929 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %2931 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2932 = stablehlo.broadcast_in_dim %2931, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2933 = stablehlo.compare  GT, %2930, %2932,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %2934 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2935 = stablehlo.broadcast_in_dim %2934, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2936 = stablehlo.convert %2935 : tensor<1x1x27x27xf32>
    %2937 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %2938 = stablehlo.broadcast_in_dim %2937, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %2939 = stablehlo.select %2933, %2936, %2938 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %2940 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %2941 = stablehlo.sqrt %2940 : tensor<f32>
    %2942 = stablehlo.convert %2941 : tensor<f32>
    %2943 = stablehlo.broadcast_in_dim %2942, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %2944 = stablehlo.divide %2921, %2943 : tensor<1x27x32x128xf32>
    %2945 = stablehlo.dot_general %2944, %2908, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %2946 = stablehlo.broadcast_in_dim %2939, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %2947 = stablehlo.add %2945, %2946 : tensor<1x32x27x27xf32>
    %2948 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2949 = stablehlo.reduce(%2947 init: %2948) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2950 = stablehlo.broadcast_in_dim %2949, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2951 = stablehlo.broadcast_in_dim %2950, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2952 = stablehlo.subtract %2947, %2951 : tensor<1x32x27x27xf32>
    %2953 = stablehlo.exponential %2952 : tensor<1x32x27x27xf32>
    %2954 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2955 = stablehlo.reduce(%2953 init: %2954) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %2956 = stablehlo.broadcast_in_dim %2955, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %2957 = stablehlo.broadcast_in_dim %2956, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %2958 = stablehlo.divide %2953, %2957 : tensor<1x32x27x27xf32>
    %2959 = stablehlo.dot_general %2884, %2958, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %2960 = stablehlo.transpose %2959, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %2961 = stablehlo.reshape %2960 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %2962 = stablehlo.convert %186 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %2963 = stablehlo.dot_general %2961, %2962, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %2964 = stablehlo.add %2850, %2963 : tensor<1x27x4096xf32>
    %2965 = stablehlo.multiply %2964, %2964 : tensor<1x27x4096xf32>
    %2966 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2967 = stablehlo.reduce(%2965 init: %2966) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2968 = stablehlo.broadcast_in_dim %2967, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2969 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2970 = stablehlo.broadcast_in_dim %2969, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2971 = stablehlo.divide %2968, %2970 : tensor<1x27x1xf32>
    %2972 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2973 = stablehlo.broadcast_in_dim %2972, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2974 = stablehlo.add %2971, %2973 : tensor<1x27x1xf32>
    %2975 = stablehlo.sqrt %2974 : tensor<1x27x1xf32>
    %2976 = stablehlo.broadcast_in_dim %2975, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %2977 = stablehlo.divide %2964, %2976 : tensor<1x27x4096xf32>
    %2978 = stablehlo.convert %187 : (tensor<4096xf16>) -> tensor<4096xf32>
    %2979 = stablehlo.broadcast_in_dim %2978, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2980 = stablehlo.broadcast_in_dim %2979, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %2981 = stablehlo.multiply %2980, %2977 : tensor<1x27x4096xf32>
    %2982 = stablehlo.convert %188 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2983 = stablehlo.dot_general %2981, %2982, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2984 = stablehlo.convert %189 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %2985 = stablehlo.dot_general %2981, %2984, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %2986 = call @silu(%2985) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %2987 = stablehlo.multiply %2983, %2986 : tensor<1x27x11008xf32>
    %2988 = stablehlo.convert %190 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %2989 = stablehlo.dot_general %2987, %2988, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %2990 = stablehlo.add %2964, %2989 : tensor<1x27x4096xf32>
    %2991 = stablehlo.multiply %2990, %2990 : tensor<1x27x4096xf32>
    %2992 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2993 = stablehlo.reduce(%2991 init: %2992) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %2994 = stablehlo.broadcast_in_dim %2993, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %2995 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %2996 = stablehlo.broadcast_in_dim %2995, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %2997 = stablehlo.divide %2994, %2996 : tensor<1x27x1xf32>
    %2998 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2999 = stablehlo.broadcast_in_dim %2998, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3000 = stablehlo.add %2997, %2999 : tensor<1x27x1xf32>
    %3001 = stablehlo.sqrt %3000 : tensor<1x27x1xf32>
    %3002 = stablehlo.broadcast_in_dim %3001, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3003 = stablehlo.divide %2990, %3002 : tensor<1x27x4096xf32>
    %3004 = stablehlo.convert %191 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3005 = stablehlo.broadcast_in_dim %3004, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3006 = stablehlo.broadcast_in_dim %3005, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3007 = stablehlo.multiply %3006, %3003 : tensor<1x27x4096xf32>
    %3008 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %3009 = stablehlo.broadcast_in_dim %3008, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %3010 = stablehlo.broadcast_in_dim %3009, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %3011 = stablehlo.broadcast_in_dim %3009, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %3012 = stablehlo.broadcast_in_dim %3010, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %3013 = stablehlo.broadcast_in_dim %3011, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %3014 = stablehlo.compare  GE, %3012, %3013,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %3015 = stablehlo.broadcast_in_dim %3014, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %3016 = stablehlo.convert %192 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3017 = stablehlo.dot_general %3007, %3016, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3018 = stablehlo.convert %193 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3019 = stablehlo.dot_general %3007, %3018, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3020 = stablehlo.convert %194 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3021 = stablehlo.dot_general %3007, %3020, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3022 = stablehlo.reshape %3017 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3023 = stablehlo.reshape %3019 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3024 = stablehlo.reshape %3021 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3025 = stablehlo.constant dense<0> : tensor<i32>
    %3026 = stablehlo.broadcast_in_dim %3025, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3027 = stablehlo.compare  LT, %324, %3026,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %3028 = stablehlo.constant dense<4096> : tensor<i32>
    %3029 = stablehlo.broadcast_in_dim %3028, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3030 = stablehlo.add %324, %3029 : tensor<1x27xi32>
    %3031 = stablehlo.select %3027, %3030, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %3032 = stablehlo.broadcast_in_dim %3031, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %3033 = "stablehlo.gather"(%195, %3032) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %3034 = stablehlo.slice %3033 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3035 = stablehlo.slice %3033 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3036 = stablehlo.broadcast_in_dim %3035, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3037 = stablehlo.multiply %3023, %3036 : tensor<1x27x32x128xf32>
    %3038 = stablehlo.constant dense<64> : tensor<i32>
    %3039 = stablehlo.broadcast_in_dim %3038, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3040 = "stablehlo.gather"(%3023, %3039) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3041 = stablehlo.negate %3040 : tensor<1x27x32x64xf32>
    %3042 = stablehlo.constant dense<0> : tensor<i32>
    %3043 = stablehlo.broadcast_in_dim %3042, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3044 = "stablehlo.gather"(%3023, %3043) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3045 = stablehlo.concatenate %3041, %3044, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3046 = stablehlo.broadcast_in_dim %3034, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3047 = stablehlo.multiply %3045, %3046 : tensor<1x27x32x128xf32>
    %3048 = stablehlo.add %3037, %3047 : tensor<1x27x32x128xf32>
    %3049 = stablehlo.broadcast_in_dim %3035, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3050 = stablehlo.multiply %3022, %3049 : tensor<1x27x32x128xf32>
    %3051 = stablehlo.constant dense<64> : tensor<i32>
    %3052 = stablehlo.broadcast_in_dim %3051, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3053 = "stablehlo.gather"(%3022, %3052) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3054 = stablehlo.negate %3053 : tensor<1x27x32x64xf32>
    %3055 = stablehlo.constant dense<0> : tensor<i32>
    %3056 = stablehlo.broadcast_in_dim %3055, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3057 = "stablehlo.gather"(%3022, %3056) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3058 = stablehlo.concatenate %3054, %3057, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3059 = stablehlo.broadcast_in_dim %3034, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3060 = stablehlo.multiply %3058, %3059 : tensor<1x27x32x128xf32>
    %3061 = stablehlo.add %3050, %3060 : tensor<1x27x32x128xf32>
    %3062 = stablehlo.slice %3015 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %3063 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %3064 = stablehlo.reshape %3063 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %3065 = stablehlo.broadcast_in_dim %3064, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %3066 = stablehlo.constant dense<0> : tensor<i32>
    %3067 = stablehlo.broadcast_in_dim %3066, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %3068 = stablehlo.compare  NE, %3065, %3067,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %3069 = stablehlo.and %3068, %3062 : tensor<1x1x27x27xi1>
    %3070 = stablehlo.convert %3069 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %3071 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3072 = stablehlo.broadcast_in_dim %3071, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3073 = stablehlo.compare  GT, %3070, %3072,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %3074 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3075 = stablehlo.broadcast_in_dim %3074, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3076 = stablehlo.convert %3075 : tensor<1x1x27x27xf32>
    %3077 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %3078 = stablehlo.broadcast_in_dim %3077, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3079 = stablehlo.select %3073, %3076, %3078 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %3080 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %3081 = stablehlo.sqrt %3080 : tensor<f32>
    %3082 = stablehlo.convert %3081 : tensor<f32>
    %3083 = stablehlo.broadcast_in_dim %3082, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %3084 = stablehlo.divide %3061, %3083 : tensor<1x27x32x128xf32>
    %3085 = stablehlo.dot_general %3084, %3048, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %3086 = stablehlo.broadcast_in_dim %3079, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %3087 = stablehlo.add %3085, %3086 : tensor<1x32x27x27xf32>
    %3088 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3089 = stablehlo.reduce(%3087 init: %3088) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3090 = stablehlo.broadcast_in_dim %3089, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3091 = stablehlo.broadcast_in_dim %3090, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3092 = stablehlo.subtract %3087, %3091 : tensor<1x32x27x27xf32>
    %3093 = stablehlo.exponential %3092 : tensor<1x32x27x27xf32>
    %3094 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3095 = stablehlo.reduce(%3093 init: %3094) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3096 = stablehlo.broadcast_in_dim %3095, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3097 = stablehlo.broadcast_in_dim %3096, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3098 = stablehlo.divide %3093, %3097 : tensor<1x32x27x27xf32>
    %3099 = stablehlo.dot_general %3024, %3098, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %3100 = stablehlo.transpose %3099, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %3101 = stablehlo.reshape %3100 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %3102 = stablehlo.convert %196 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3103 = stablehlo.dot_general %3101, %3102, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3104 = stablehlo.add %2990, %3103 : tensor<1x27x4096xf32>
    %3105 = stablehlo.multiply %3104, %3104 : tensor<1x27x4096xf32>
    %3106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3107 = stablehlo.reduce(%3105 init: %3106) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3108 = stablehlo.broadcast_in_dim %3107, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3109 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3110 = stablehlo.broadcast_in_dim %3109, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3111 = stablehlo.divide %3108, %3110 : tensor<1x27x1xf32>
    %3112 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3113 = stablehlo.broadcast_in_dim %3112, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3114 = stablehlo.add %3111, %3113 : tensor<1x27x1xf32>
    %3115 = stablehlo.sqrt %3114 : tensor<1x27x1xf32>
    %3116 = stablehlo.broadcast_in_dim %3115, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3117 = stablehlo.divide %3104, %3116 : tensor<1x27x4096xf32>
    %3118 = stablehlo.convert %197 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3119 = stablehlo.broadcast_in_dim %3118, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3120 = stablehlo.broadcast_in_dim %3119, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3121 = stablehlo.multiply %3120, %3117 : tensor<1x27x4096xf32>
    %3122 = stablehlo.convert %198 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3123 = stablehlo.dot_general %3121, %3122, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3124 = stablehlo.convert %199 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3125 = stablehlo.dot_general %3121, %3124, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3126 = call @silu(%3125) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %3127 = stablehlo.multiply %3123, %3126 : tensor<1x27x11008xf32>
    %3128 = stablehlo.convert %200 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %3129 = stablehlo.dot_general %3127, %3128, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %3130 = stablehlo.add %3104, %3129 : tensor<1x27x4096xf32>
    %3131 = stablehlo.multiply %3130, %3130 : tensor<1x27x4096xf32>
    %3132 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3133 = stablehlo.reduce(%3131 init: %3132) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3134 = stablehlo.broadcast_in_dim %3133, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3135 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3136 = stablehlo.broadcast_in_dim %3135, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3137 = stablehlo.divide %3134, %3136 : tensor<1x27x1xf32>
    %3138 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3139 = stablehlo.broadcast_in_dim %3138, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3140 = stablehlo.add %3137, %3139 : tensor<1x27x1xf32>
    %3141 = stablehlo.sqrt %3140 : tensor<1x27x1xf32>
    %3142 = stablehlo.broadcast_in_dim %3141, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3143 = stablehlo.divide %3130, %3142 : tensor<1x27x4096xf32>
    %3144 = stablehlo.convert %201 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3145 = stablehlo.broadcast_in_dim %3144, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3146 = stablehlo.broadcast_in_dim %3145, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3147 = stablehlo.multiply %3146, %3143 : tensor<1x27x4096xf32>
    %3148 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %3149 = stablehlo.broadcast_in_dim %3148, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %3150 = stablehlo.broadcast_in_dim %3149, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %3151 = stablehlo.broadcast_in_dim %3149, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %3152 = stablehlo.broadcast_in_dim %3150, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %3153 = stablehlo.broadcast_in_dim %3151, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %3154 = stablehlo.compare  GE, %3152, %3153,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %3155 = stablehlo.broadcast_in_dim %3154, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %3156 = stablehlo.convert %202 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3157 = stablehlo.dot_general %3147, %3156, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3158 = stablehlo.convert %203 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3159 = stablehlo.dot_general %3147, %3158, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3160 = stablehlo.convert %204 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3161 = stablehlo.dot_general %3147, %3160, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3162 = stablehlo.reshape %3157 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3163 = stablehlo.reshape %3159 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3164 = stablehlo.reshape %3161 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3165 = stablehlo.constant dense<0> : tensor<i32>
    %3166 = stablehlo.broadcast_in_dim %3165, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3167 = stablehlo.compare  LT, %324, %3166,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %3168 = stablehlo.constant dense<4096> : tensor<i32>
    %3169 = stablehlo.broadcast_in_dim %3168, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3170 = stablehlo.add %324, %3169 : tensor<1x27xi32>
    %3171 = stablehlo.select %3167, %3170, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %3172 = stablehlo.broadcast_in_dim %3171, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %3173 = "stablehlo.gather"(%205, %3172) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %3174 = stablehlo.slice %3173 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3175 = stablehlo.slice %3173 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3176 = stablehlo.broadcast_in_dim %3175, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3177 = stablehlo.multiply %3163, %3176 : tensor<1x27x32x128xf32>
    %3178 = stablehlo.constant dense<64> : tensor<i32>
    %3179 = stablehlo.broadcast_in_dim %3178, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3180 = "stablehlo.gather"(%3163, %3179) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3181 = stablehlo.negate %3180 : tensor<1x27x32x64xf32>
    %3182 = stablehlo.constant dense<0> : tensor<i32>
    %3183 = stablehlo.broadcast_in_dim %3182, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3184 = "stablehlo.gather"(%3163, %3183) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3185 = stablehlo.concatenate %3181, %3184, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3186 = stablehlo.broadcast_in_dim %3174, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3187 = stablehlo.multiply %3185, %3186 : tensor<1x27x32x128xf32>
    %3188 = stablehlo.add %3177, %3187 : tensor<1x27x32x128xf32>
    %3189 = stablehlo.broadcast_in_dim %3175, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3190 = stablehlo.multiply %3162, %3189 : tensor<1x27x32x128xf32>
    %3191 = stablehlo.constant dense<64> : tensor<i32>
    %3192 = stablehlo.broadcast_in_dim %3191, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3193 = "stablehlo.gather"(%3162, %3192) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3194 = stablehlo.negate %3193 : tensor<1x27x32x64xf32>
    %3195 = stablehlo.constant dense<0> : tensor<i32>
    %3196 = stablehlo.broadcast_in_dim %3195, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3197 = "stablehlo.gather"(%3162, %3196) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3198 = stablehlo.concatenate %3194, %3197, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3199 = stablehlo.broadcast_in_dim %3174, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3200 = stablehlo.multiply %3198, %3199 : tensor<1x27x32x128xf32>
    %3201 = stablehlo.add %3190, %3200 : tensor<1x27x32x128xf32>
    %3202 = stablehlo.slice %3155 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %3203 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %3204 = stablehlo.reshape %3203 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %3205 = stablehlo.broadcast_in_dim %3204, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %3206 = stablehlo.constant dense<0> : tensor<i32>
    %3207 = stablehlo.broadcast_in_dim %3206, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %3208 = stablehlo.compare  NE, %3205, %3207,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %3209 = stablehlo.and %3208, %3202 : tensor<1x1x27x27xi1>
    %3210 = stablehlo.convert %3209 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %3211 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3212 = stablehlo.broadcast_in_dim %3211, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3213 = stablehlo.compare  GT, %3210, %3212,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %3214 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3215 = stablehlo.broadcast_in_dim %3214, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3216 = stablehlo.convert %3215 : tensor<1x1x27x27xf32>
    %3217 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %3218 = stablehlo.broadcast_in_dim %3217, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3219 = stablehlo.select %3213, %3216, %3218 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %3220 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %3221 = stablehlo.sqrt %3220 : tensor<f32>
    %3222 = stablehlo.convert %3221 : tensor<f32>
    %3223 = stablehlo.broadcast_in_dim %3222, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %3224 = stablehlo.divide %3201, %3223 : tensor<1x27x32x128xf32>
    %3225 = stablehlo.dot_general %3224, %3188, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %3226 = stablehlo.broadcast_in_dim %3219, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %3227 = stablehlo.add %3225, %3226 : tensor<1x32x27x27xf32>
    %3228 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3229 = stablehlo.reduce(%3227 init: %3228) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3230 = stablehlo.broadcast_in_dim %3229, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3231 = stablehlo.broadcast_in_dim %3230, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3232 = stablehlo.subtract %3227, %3231 : tensor<1x32x27x27xf32>
    %3233 = stablehlo.exponential %3232 : tensor<1x32x27x27xf32>
    %3234 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3235 = stablehlo.reduce(%3233 init: %3234) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3236 = stablehlo.broadcast_in_dim %3235, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3237 = stablehlo.broadcast_in_dim %3236, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3238 = stablehlo.divide %3233, %3237 : tensor<1x32x27x27xf32>
    %3239 = stablehlo.dot_general %3164, %3238, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %3240 = stablehlo.transpose %3239, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %3241 = stablehlo.reshape %3240 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %3242 = stablehlo.convert %206 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3243 = stablehlo.dot_general %3241, %3242, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3244 = stablehlo.add %3130, %3243 : tensor<1x27x4096xf32>
    %3245 = stablehlo.multiply %3244, %3244 : tensor<1x27x4096xf32>
    %3246 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3247 = stablehlo.reduce(%3245 init: %3246) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3248 = stablehlo.broadcast_in_dim %3247, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3249 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3250 = stablehlo.broadcast_in_dim %3249, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3251 = stablehlo.divide %3248, %3250 : tensor<1x27x1xf32>
    %3252 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3253 = stablehlo.broadcast_in_dim %3252, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3254 = stablehlo.add %3251, %3253 : tensor<1x27x1xf32>
    %3255 = stablehlo.sqrt %3254 : tensor<1x27x1xf32>
    %3256 = stablehlo.broadcast_in_dim %3255, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3257 = stablehlo.divide %3244, %3256 : tensor<1x27x4096xf32>
    %3258 = stablehlo.convert %207 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3259 = stablehlo.broadcast_in_dim %3258, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3260 = stablehlo.broadcast_in_dim %3259, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3261 = stablehlo.multiply %3260, %3257 : tensor<1x27x4096xf32>
    %3262 = stablehlo.convert %208 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3263 = stablehlo.dot_general %3261, %3262, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3264 = stablehlo.convert %209 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3265 = stablehlo.dot_general %3261, %3264, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3266 = call @silu(%3265) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %3267 = stablehlo.multiply %3263, %3266 : tensor<1x27x11008xf32>
    %3268 = stablehlo.convert %210 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %3269 = stablehlo.dot_general %3267, %3268, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %3270 = stablehlo.add %3244, %3269 : tensor<1x27x4096xf32>
    %3271 = stablehlo.multiply %3270, %3270 : tensor<1x27x4096xf32>
    %3272 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3273 = stablehlo.reduce(%3271 init: %3272) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3274 = stablehlo.broadcast_in_dim %3273, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3275 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3276 = stablehlo.broadcast_in_dim %3275, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3277 = stablehlo.divide %3274, %3276 : tensor<1x27x1xf32>
    %3278 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3279 = stablehlo.broadcast_in_dim %3278, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3280 = stablehlo.add %3277, %3279 : tensor<1x27x1xf32>
    %3281 = stablehlo.sqrt %3280 : tensor<1x27x1xf32>
    %3282 = stablehlo.broadcast_in_dim %3281, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3283 = stablehlo.divide %3270, %3282 : tensor<1x27x4096xf32>
    %3284 = stablehlo.convert %211 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3285 = stablehlo.broadcast_in_dim %3284, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3286 = stablehlo.broadcast_in_dim %3285, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3287 = stablehlo.multiply %3286, %3283 : tensor<1x27x4096xf32>
    %3288 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %3289 = stablehlo.broadcast_in_dim %3288, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %3290 = stablehlo.broadcast_in_dim %3289, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %3291 = stablehlo.broadcast_in_dim %3289, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %3292 = stablehlo.broadcast_in_dim %3290, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %3293 = stablehlo.broadcast_in_dim %3291, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %3294 = stablehlo.compare  GE, %3292, %3293,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %3295 = stablehlo.broadcast_in_dim %3294, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %3296 = stablehlo.convert %212 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3297 = stablehlo.dot_general %3287, %3296, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3298 = stablehlo.convert %213 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3299 = stablehlo.dot_general %3287, %3298, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3300 = stablehlo.convert %214 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3301 = stablehlo.dot_general %3287, %3300, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3302 = stablehlo.reshape %3297 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3303 = stablehlo.reshape %3299 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3304 = stablehlo.reshape %3301 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3305 = stablehlo.constant dense<0> : tensor<i32>
    %3306 = stablehlo.broadcast_in_dim %3305, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3307 = stablehlo.compare  LT, %324, %3306,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %3308 = stablehlo.constant dense<4096> : tensor<i32>
    %3309 = stablehlo.broadcast_in_dim %3308, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3310 = stablehlo.add %324, %3309 : tensor<1x27xi32>
    %3311 = stablehlo.select %3307, %3310, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %3312 = stablehlo.broadcast_in_dim %3311, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %3313 = "stablehlo.gather"(%215, %3312) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %3314 = stablehlo.slice %3313 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3315 = stablehlo.slice %3313 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3316 = stablehlo.broadcast_in_dim %3315, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3317 = stablehlo.multiply %3303, %3316 : tensor<1x27x32x128xf32>
    %3318 = stablehlo.constant dense<64> : tensor<i32>
    %3319 = stablehlo.broadcast_in_dim %3318, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3320 = "stablehlo.gather"(%3303, %3319) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3321 = stablehlo.negate %3320 : tensor<1x27x32x64xf32>
    %3322 = stablehlo.constant dense<0> : tensor<i32>
    %3323 = stablehlo.broadcast_in_dim %3322, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3324 = "stablehlo.gather"(%3303, %3323) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3325 = stablehlo.concatenate %3321, %3324, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3326 = stablehlo.broadcast_in_dim %3314, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3327 = stablehlo.multiply %3325, %3326 : tensor<1x27x32x128xf32>
    %3328 = stablehlo.add %3317, %3327 : tensor<1x27x32x128xf32>
    %3329 = stablehlo.broadcast_in_dim %3315, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3330 = stablehlo.multiply %3302, %3329 : tensor<1x27x32x128xf32>
    %3331 = stablehlo.constant dense<64> : tensor<i32>
    %3332 = stablehlo.broadcast_in_dim %3331, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3333 = "stablehlo.gather"(%3302, %3332) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3334 = stablehlo.negate %3333 : tensor<1x27x32x64xf32>
    %3335 = stablehlo.constant dense<0> : tensor<i32>
    %3336 = stablehlo.broadcast_in_dim %3335, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3337 = "stablehlo.gather"(%3302, %3336) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3338 = stablehlo.concatenate %3334, %3337, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3339 = stablehlo.broadcast_in_dim %3314, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3340 = stablehlo.multiply %3338, %3339 : tensor<1x27x32x128xf32>
    %3341 = stablehlo.add %3330, %3340 : tensor<1x27x32x128xf32>
    %3342 = stablehlo.slice %3295 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %3343 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %3344 = stablehlo.reshape %3343 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %3345 = stablehlo.broadcast_in_dim %3344, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %3346 = stablehlo.constant dense<0> : tensor<i32>
    %3347 = stablehlo.broadcast_in_dim %3346, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %3348 = stablehlo.compare  NE, %3345, %3347,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %3349 = stablehlo.and %3348, %3342 : tensor<1x1x27x27xi1>
    %3350 = stablehlo.convert %3349 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %3351 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3352 = stablehlo.broadcast_in_dim %3351, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3353 = stablehlo.compare  GT, %3350, %3352,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %3354 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3355 = stablehlo.broadcast_in_dim %3354, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3356 = stablehlo.convert %3355 : tensor<1x1x27x27xf32>
    %3357 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %3358 = stablehlo.broadcast_in_dim %3357, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3359 = stablehlo.select %3353, %3356, %3358 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %3360 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %3361 = stablehlo.sqrt %3360 : tensor<f32>
    %3362 = stablehlo.convert %3361 : tensor<f32>
    %3363 = stablehlo.broadcast_in_dim %3362, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %3364 = stablehlo.divide %3341, %3363 : tensor<1x27x32x128xf32>
    %3365 = stablehlo.dot_general %3364, %3328, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %3366 = stablehlo.broadcast_in_dim %3359, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %3367 = stablehlo.add %3365, %3366 : tensor<1x32x27x27xf32>
    %3368 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3369 = stablehlo.reduce(%3367 init: %3368) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3370 = stablehlo.broadcast_in_dim %3369, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3371 = stablehlo.broadcast_in_dim %3370, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3372 = stablehlo.subtract %3367, %3371 : tensor<1x32x27x27xf32>
    %3373 = stablehlo.exponential %3372 : tensor<1x32x27x27xf32>
    %3374 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3375 = stablehlo.reduce(%3373 init: %3374) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3376 = stablehlo.broadcast_in_dim %3375, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3377 = stablehlo.broadcast_in_dim %3376, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3378 = stablehlo.divide %3373, %3377 : tensor<1x32x27x27xf32>
    %3379 = stablehlo.dot_general %3304, %3378, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %3380 = stablehlo.transpose %3379, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %3381 = stablehlo.reshape %3380 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %3382 = stablehlo.convert %216 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3383 = stablehlo.dot_general %3381, %3382, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3384 = stablehlo.add %3270, %3383 : tensor<1x27x4096xf32>
    %3385 = stablehlo.multiply %3384, %3384 : tensor<1x27x4096xf32>
    %3386 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3387 = stablehlo.reduce(%3385 init: %3386) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3388 = stablehlo.broadcast_in_dim %3387, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3389 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3390 = stablehlo.broadcast_in_dim %3389, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3391 = stablehlo.divide %3388, %3390 : tensor<1x27x1xf32>
    %3392 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3393 = stablehlo.broadcast_in_dim %3392, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3394 = stablehlo.add %3391, %3393 : tensor<1x27x1xf32>
    %3395 = stablehlo.sqrt %3394 : tensor<1x27x1xf32>
    %3396 = stablehlo.broadcast_in_dim %3395, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3397 = stablehlo.divide %3384, %3396 : tensor<1x27x4096xf32>
    %3398 = stablehlo.convert %217 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3399 = stablehlo.broadcast_in_dim %3398, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3400 = stablehlo.broadcast_in_dim %3399, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3401 = stablehlo.multiply %3400, %3397 : tensor<1x27x4096xf32>
    %3402 = stablehlo.convert %218 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3403 = stablehlo.dot_general %3401, %3402, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3404 = stablehlo.convert %219 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3405 = stablehlo.dot_general %3401, %3404, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3406 = call @silu(%3405) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %3407 = stablehlo.multiply %3403, %3406 : tensor<1x27x11008xf32>
    %3408 = stablehlo.convert %220 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %3409 = stablehlo.dot_general %3407, %3408, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %3410 = stablehlo.add %3384, %3409 : tensor<1x27x4096xf32>
    %3411 = stablehlo.multiply %3410, %3410 : tensor<1x27x4096xf32>
    %3412 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3413 = stablehlo.reduce(%3411 init: %3412) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3414 = stablehlo.broadcast_in_dim %3413, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3415 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3416 = stablehlo.broadcast_in_dim %3415, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3417 = stablehlo.divide %3414, %3416 : tensor<1x27x1xf32>
    %3418 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3419 = stablehlo.broadcast_in_dim %3418, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3420 = stablehlo.add %3417, %3419 : tensor<1x27x1xf32>
    %3421 = stablehlo.sqrt %3420 : tensor<1x27x1xf32>
    %3422 = stablehlo.broadcast_in_dim %3421, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3423 = stablehlo.divide %3410, %3422 : tensor<1x27x4096xf32>
    %3424 = stablehlo.convert %221 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3425 = stablehlo.broadcast_in_dim %3424, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3426 = stablehlo.broadcast_in_dim %3425, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3427 = stablehlo.multiply %3426, %3423 : tensor<1x27x4096xf32>
    %3428 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %3429 = stablehlo.broadcast_in_dim %3428, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %3430 = stablehlo.broadcast_in_dim %3429, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %3431 = stablehlo.broadcast_in_dim %3429, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %3432 = stablehlo.broadcast_in_dim %3430, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %3433 = stablehlo.broadcast_in_dim %3431, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %3434 = stablehlo.compare  GE, %3432, %3433,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %3435 = stablehlo.broadcast_in_dim %3434, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %3436 = stablehlo.convert %222 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3437 = stablehlo.dot_general %3427, %3436, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3438 = stablehlo.convert %223 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3439 = stablehlo.dot_general %3427, %3438, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3440 = stablehlo.convert %224 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3441 = stablehlo.dot_general %3427, %3440, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3442 = stablehlo.reshape %3437 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3443 = stablehlo.reshape %3439 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3444 = stablehlo.reshape %3441 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3445 = stablehlo.constant dense<0> : tensor<i32>
    %3446 = stablehlo.broadcast_in_dim %3445, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3447 = stablehlo.compare  LT, %324, %3446,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %3448 = stablehlo.constant dense<4096> : tensor<i32>
    %3449 = stablehlo.broadcast_in_dim %3448, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3450 = stablehlo.add %324, %3449 : tensor<1x27xi32>
    %3451 = stablehlo.select %3447, %3450, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %3452 = stablehlo.broadcast_in_dim %3451, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %3453 = "stablehlo.gather"(%225, %3452) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %3454 = stablehlo.slice %3453 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3455 = stablehlo.slice %3453 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3456 = stablehlo.broadcast_in_dim %3455, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3457 = stablehlo.multiply %3443, %3456 : tensor<1x27x32x128xf32>
    %3458 = stablehlo.constant dense<64> : tensor<i32>
    %3459 = stablehlo.broadcast_in_dim %3458, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3460 = "stablehlo.gather"(%3443, %3459) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3461 = stablehlo.negate %3460 : tensor<1x27x32x64xf32>
    %3462 = stablehlo.constant dense<0> : tensor<i32>
    %3463 = stablehlo.broadcast_in_dim %3462, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3464 = "stablehlo.gather"(%3443, %3463) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3465 = stablehlo.concatenate %3461, %3464, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3466 = stablehlo.broadcast_in_dim %3454, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3467 = stablehlo.multiply %3465, %3466 : tensor<1x27x32x128xf32>
    %3468 = stablehlo.add %3457, %3467 : tensor<1x27x32x128xf32>
    %3469 = stablehlo.broadcast_in_dim %3455, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3470 = stablehlo.multiply %3442, %3469 : tensor<1x27x32x128xf32>
    %3471 = stablehlo.constant dense<64> : tensor<i32>
    %3472 = stablehlo.broadcast_in_dim %3471, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3473 = "stablehlo.gather"(%3442, %3472) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3474 = stablehlo.negate %3473 : tensor<1x27x32x64xf32>
    %3475 = stablehlo.constant dense<0> : tensor<i32>
    %3476 = stablehlo.broadcast_in_dim %3475, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3477 = "stablehlo.gather"(%3442, %3476) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3478 = stablehlo.concatenate %3474, %3477, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3479 = stablehlo.broadcast_in_dim %3454, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3480 = stablehlo.multiply %3478, %3479 : tensor<1x27x32x128xf32>
    %3481 = stablehlo.add %3470, %3480 : tensor<1x27x32x128xf32>
    %3482 = stablehlo.slice %3435 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %3483 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %3484 = stablehlo.reshape %3483 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %3485 = stablehlo.broadcast_in_dim %3484, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %3486 = stablehlo.constant dense<0> : tensor<i32>
    %3487 = stablehlo.broadcast_in_dim %3486, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %3488 = stablehlo.compare  NE, %3485, %3487,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %3489 = stablehlo.and %3488, %3482 : tensor<1x1x27x27xi1>
    %3490 = stablehlo.convert %3489 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %3491 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3492 = stablehlo.broadcast_in_dim %3491, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3493 = stablehlo.compare  GT, %3490, %3492,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %3494 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3495 = stablehlo.broadcast_in_dim %3494, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3496 = stablehlo.convert %3495 : tensor<1x1x27x27xf32>
    %3497 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %3498 = stablehlo.broadcast_in_dim %3497, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3499 = stablehlo.select %3493, %3496, %3498 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %3500 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %3501 = stablehlo.sqrt %3500 : tensor<f32>
    %3502 = stablehlo.convert %3501 : tensor<f32>
    %3503 = stablehlo.broadcast_in_dim %3502, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %3504 = stablehlo.divide %3481, %3503 : tensor<1x27x32x128xf32>
    %3505 = stablehlo.dot_general %3504, %3468, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %3506 = stablehlo.broadcast_in_dim %3499, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %3507 = stablehlo.add %3505, %3506 : tensor<1x32x27x27xf32>
    %3508 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3509 = stablehlo.reduce(%3507 init: %3508) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3510 = stablehlo.broadcast_in_dim %3509, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3511 = stablehlo.broadcast_in_dim %3510, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3512 = stablehlo.subtract %3507, %3511 : tensor<1x32x27x27xf32>
    %3513 = stablehlo.exponential %3512 : tensor<1x32x27x27xf32>
    %3514 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3515 = stablehlo.reduce(%3513 init: %3514) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3516 = stablehlo.broadcast_in_dim %3515, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3517 = stablehlo.broadcast_in_dim %3516, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3518 = stablehlo.divide %3513, %3517 : tensor<1x32x27x27xf32>
    %3519 = stablehlo.dot_general %3444, %3518, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %3520 = stablehlo.transpose %3519, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %3521 = stablehlo.reshape %3520 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %3522 = stablehlo.convert %226 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3523 = stablehlo.dot_general %3521, %3522, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3524 = stablehlo.add %3410, %3523 : tensor<1x27x4096xf32>
    %3525 = stablehlo.multiply %3524, %3524 : tensor<1x27x4096xf32>
    %3526 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3527 = stablehlo.reduce(%3525 init: %3526) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3528 = stablehlo.broadcast_in_dim %3527, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3529 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3530 = stablehlo.broadcast_in_dim %3529, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3531 = stablehlo.divide %3528, %3530 : tensor<1x27x1xf32>
    %3532 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3533 = stablehlo.broadcast_in_dim %3532, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3534 = stablehlo.add %3531, %3533 : tensor<1x27x1xf32>
    %3535 = stablehlo.sqrt %3534 : tensor<1x27x1xf32>
    %3536 = stablehlo.broadcast_in_dim %3535, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3537 = stablehlo.divide %3524, %3536 : tensor<1x27x4096xf32>
    %3538 = stablehlo.convert %227 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3539 = stablehlo.broadcast_in_dim %3538, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3540 = stablehlo.broadcast_in_dim %3539, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3541 = stablehlo.multiply %3540, %3537 : tensor<1x27x4096xf32>
    %3542 = stablehlo.convert %228 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3543 = stablehlo.dot_general %3541, %3542, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3544 = stablehlo.convert %229 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3545 = stablehlo.dot_general %3541, %3544, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3546 = call @silu(%3545) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %3547 = stablehlo.multiply %3543, %3546 : tensor<1x27x11008xf32>
    %3548 = stablehlo.convert %230 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %3549 = stablehlo.dot_general %3547, %3548, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %3550 = stablehlo.add %3524, %3549 : tensor<1x27x4096xf32>
    %3551 = stablehlo.multiply %3550, %3550 : tensor<1x27x4096xf32>
    %3552 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3553 = stablehlo.reduce(%3551 init: %3552) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3554 = stablehlo.broadcast_in_dim %3553, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3555 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3556 = stablehlo.broadcast_in_dim %3555, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3557 = stablehlo.divide %3554, %3556 : tensor<1x27x1xf32>
    %3558 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3559 = stablehlo.broadcast_in_dim %3558, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3560 = stablehlo.add %3557, %3559 : tensor<1x27x1xf32>
    %3561 = stablehlo.sqrt %3560 : tensor<1x27x1xf32>
    %3562 = stablehlo.broadcast_in_dim %3561, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3563 = stablehlo.divide %3550, %3562 : tensor<1x27x4096xf32>
    %3564 = stablehlo.convert %231 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3565 = stablehlo.broadcast_in_dim %3564, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3566 = stablehlo.broadcast_in_dim %3565, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3567 = stablehlo.multiply %3566, %3563 : tensor<1x27x4096xf32>
    %3568 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %3569 = stablehlo.broadcast_in_dim %3568, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %3570 = stablehlo.broadcast_in_dim %3569, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %3571 = stablehlo.broadcast_in_dim %3569, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %3572 = stablehlo.broadcast_in_dim %3570, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %3573 = stablehlo.broadcast_in_dim %3571, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %3574 = stablehlo.compare  GE, %3572, %3573,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %3575 = stablehlo.broadcast_in_dim %3574, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %3576 = stablehlo.convert %232 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3577 = stablehlo.dot_general %3567, %3576, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3578 = stablehlo.convert %233 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3579 = stablehlo.dot_general %3567, %3578, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3580 = stablehlo.convert %234 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3581 = stablehlo.dot_general %3567, %3580, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3582 = stablehlo.reshape %3577 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3583 = stablehlo.reshape %3579 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3584 = stablehlo.reshape %3581 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3585 = stablehlo.constant dense<0> : tensor<i32>
    %3586 = stablehlo.broadcast_in_dim %3585, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3587 = stablehlo.compare  LT, %324, %3586,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %3588 = stablehlo.constant dense<4096> : tensor<i32>
    %3589 = stablehlo.broadcast_in_dim %3588, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3590 = stablehlo.add %324, %3589 : tensor<1x27xi32>
    %3591 = stablehlo.select %3587, %3590, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %3592 = stablehlo.broadcast_in_dim %3591, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %3593 = "stablehlo.gather"(%235, %3592) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %3594 = stablehlo.slice %3593 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3595 = stablehlo.slice %3593 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3596 = stablehlo.broadcast_in_dim %3595, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3597 = stablehlo.multiply %3583, %3596 : tensor<1x27x32x128xf32>
    %3598 = stablehlo.constant dense<64> : tensor<i32>
    %3599 = stablehlo.broadcast_in_dim %3598, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3600 = "stablehlo.gather"(%3583, %3599) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3601 = stablehlo.negate %3600 : tensor<1x27x32x64xf32>
    %3602 = stablehlo.constant dense<0> : tensor<i32>
    %3603 = stablehlo.broadcast_in_dim %3602, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3604 = "stablehlo.gather"(%3583, %3603) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3605 = stablehlo.concatenate %3601, %3604, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3606 = stablehlo.broadcast_in_dim %3594, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3607 = stablehlo.multiply %3605, %3606 : tensor<1x27x32x128xf32>
    %3608 = stablehlo.add %3597, %3607 : tensor<1x27x32x128xf32>
    %3609 = stablehlo.broadcast_in_dim %3595, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3610 = stablehlo.multiply %3582, %3609 : tensor<1x27x32x128xf32>
    %3611 = stablehlo.constant dense<64> : tensor<i32>
    %3612 = stablehlo.broadcast_in_dim %3611, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3613 = "stablehlo.gather"(%3582, %3612) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3614 = stablehlo.negate %3613 : tensor<1x27x32x64xf32>
    %3615 = stablehlo.constant dense<0> : tensor<i32>
    %3616 = stablehlo.broadcast_in_dim %3615, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3617 = "stablehlo.gather"(%3582, %3616) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3618 = stablehlo.concatenate %3614, %3617, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3619 = stablehlo.broadcast_in_dim %3594, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3620 = stablehlo.multiply %3618, %3619 : tensor<1x27x32x128xf32>
    %3621 = stablehlo.add %3610, %3620 : tensor<1x27x32x128xf32>
    %3622 = stablehlo.slice %3575 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %3623 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %3624 = stablehlo.reshape %3623 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %3625 = stablehlo.broadcast_in_dim %3624, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %3626 = stablehlo.constant dense<0> : tensor<i32>
    %3627 = stablehlo.broadcast_in_dim %3626, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %3628 = stablehlo.compare  NE, %3625, %3627,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %3629 = stablehlo.and %3628, %3622 : tensor<1x1x27x27xi1>
    %3630 = stablehlo.convert %3629 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %3631 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3632 = stablehlo.broadcast_in_dim %3631, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3633 = stablehlo.compare  GT, %3630, %3632,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %3634 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3635 = stablehlo.broadcast_in_dim %3634, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3636 = stablehlo.convert %3635 : tensor<1x1x27x27xf32>
    %3637 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %3638 = stablehlo.broadcast_in_dim %3637, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3639 = stablehlo.select %3633, %3636, %3638 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %3640 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %3641 = stablehlo.sqrt %3640 : tensor<f32>
    %3642 = stablehlo.convert %3641 : tensor<f32>
    %3643 = stablehlo.broadcast_in_dim %3642, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %3644 = stablehlo.divide %3621, %3643 : tensor<1x27x32x128xf32>
    %3645 = stablehlo.dot_general %3644, %3608, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %3646 = stablehlo.broadcast_in_dim %3639, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %3647 = stablehlo.add %3645, %3646 : tensor<1x32x27x27xf32>
    %3648 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3649 = stablehlo.reduce(%3647 init: %3648) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3650 = stablehlo.broadcast_in_dim %3649, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3651 = stablehlo.broadcast_in_dim %3650, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3652 = stablehlo.subtract %3647, %3651 : tensor<1x32x27x27xf32>
    %3653 = stablehlo.exponential %3652 : tensor<1x32x27x27xf32>
    %3654 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3655 = stablehlo.reduce(%3653 init: %3654) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3656 = stablehlo.broadcast_in_dim %3655, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3657 = stablehlo.broadcast_in_dim %3656, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3658 = stablehlo.divide %3653, %3657 : tensor<1x32x27x27xf32>
    %3659 = stablehlo.dot_general %3584, %3658, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %3660 = stablehlo.transpose %3659, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %3661 = stablehlo.reshape %3660 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %3662 = stablehlo.convert %236 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3663 = stablehlo.dot_general %3661, %3662, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3664 = stablehlo.add %3550, %3663 : tensor<1x27x4096xf32>
    %3665 = stablehlo.multiply %3664, %3664 : tensor<1x27x4096xf32>
    %3666 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3667 = stablehlo.reduce(%3665 init: %3666) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3668 = stablehlo.broadcast_in_dim %3667, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3669 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3670 = stablehlo.broadcast_in_dim %3669, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3671 = stablehlo.divide %3668, %3670 : tensor<1x27x1xf32>
    %3672 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3673 = stablehlo.broadcast_in_dim %3672, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3674 = stablehlo.add %3671, %3673 : tensor<1x27x1xf32>
    %3675 = stablehlo.sqrt %3674 : tensor<1x27x1xf32>
    %3676 = stablehlo.broadcast_in_dim %3675, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3677 = stablehlo.divide %3664, %3676 : tensor<1x27x4096xf32>
    %3678 = stablehlo.convert %237 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3679 = stablehlo.broadcast_in_dim %3678, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3680 = stablehlo.broadcast_in_dim %3679, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3681 = stablehlo.multiply %3680, %3677 : tensor<1x27x4096xf32>
    %3682 = stablehlo.convert %238 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3683 = stablehlo.dot_general %3681, %3682, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3684 = stablehlo.convert %239 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3685 = stablehlo.dot_general %3681, %3684, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3686 = call @silu(%3685) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %3687 = stablehlo.multiply %3683, %3686 : tensor<1x27x11008xf32>
    %3688 = stablehlo.convert %240 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %3689 = stablehlo.dot_general %3687, %3688, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %3690 = stablehlo.add %3664, %3689 : tensor<1x27x4096xf32>
    %3691 = stablehlo.multiply %3690, %3690 : tensor<1x27x4096xf32>
    %3692 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3693 = stablehlo.reduce(%3691 init: %3692) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3694 = stablehlo.broadcast_in_dim %3693, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3695 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3696 = stablehlo.broadcast_in_dim %3695, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3697 = stablehlo.divide %3694, %3696 : tensor<1x27x1xf32>
    %3698 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3699 = stablehlo.broadcast_in_dim %3698, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3700 = stablehlo.add %3697, %3699 : tensor<1x27x1xf32>
    %3701 = stablehlo.sqrt %3700 : tensor<1x27x1xf32>
    %3702 = stablehlo.broadcast_in_dim %3701, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3703 = stablehlo.divide %3690, %3702 : tensor<1x27x4096xf32>
    %3704 = stablehlo.convert %241 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3705 = stablehlo.broadcast_in_dim %3704, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3706 = stablehlo.broadcast_in_dim %3705, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3707 = stablehlo.multiply %3706, %3703 : tensor<1x27x4096xf32>
    %3708 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %3709 = stablehlo.broadcast_in_dim %3708, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %3710 = stablehlo.broadcast_in_dim %3709, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %3711 = stablehlo.broadcast_in_dim %3709, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %3712 = stablehlo.broadcast_in_dim %3710, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %3713 = stablehlo.broadcast_in_dim %3711, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %3714 = stablehlo.compare  GE, %3712, %3713,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %3715 = stablehlo.broadcast_in_dim %3714, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %3716 = stablehlo.convert %242 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3717 = stablehlo.dot_general %3707, %3716, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3718 = stablehlo.convert %243 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3719 = stablehlo.dot_general %3707, %3718, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3720 = stablehlo.convert %244 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3721 = stablehlo.dot_general %3707, %3720, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3722 = stablehlo.reshape %3717 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3723 = stablehlo.reshape %3719 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3724 = stablehlo.reshape %3721 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3725 = stablehlo.constant dense<0> : tensor<i32>
    %3726 = stablehlo.broadcast_in_dim %3725, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3727 = stablehlo.compare  LT, %324, %3726,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %3728 = stablehlo.constant dense<4096> : tensor<i32>
    %3729 = stablehlo.broadcast_in_dim %3728, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3730 = stablehlo.add %324, %3729 : tensor<1x27xi32>
    %3731 = stablehlo.select %3727, %3730, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %3732 = stablehlo.broadcast_in_dim %3731, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %3733 = "stablehlo.gather"(%245, %3732) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %3734 = stablehlo.slice %3733 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3735 = stablehlo.slice %3733 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3736 = stablehlo.broadcast_in_dim %3735, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3737 = stablehlo.multiply %3723, %3736 : tensor<1x27x32x128xf32>
    %3738 = stablehlo.constant dense<64> : tensor<i32>
    %3739 = stablehlo.broadcast_in_dim %3738, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3740 = "stablehlo.gather"(%3723, %3739) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3741 = stablehlo.negate %3740 : tensor<1x27x32x64xf32>
    %3742 = stablehlo.constant dense<0> : tensor<i32>
    %3743 = stablehlo.broadcast_in_dim %3742, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3744 = "stablehlo.gather"(%3723, %3743) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3745 = stablehlo.concatenate %3741, %3744, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3746 = stablehlo.broadcast_in_dim %3734, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3747 = stablehlo.multiply %3745, %3746 : tensor<1x27x32x128xf32>
    %3748 = stablehlo.add %3737, %3747 : tensor<1x27x32x128xf32>
    %3749 = stablehlo.broadcast_in_dim %3735, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3750 = stablehlo.multiply %3722, %3749 : tensor<1x27x32x128xf32>
    %3751 = stablehlo.constant dense<64> : tensor<i32>
    %3752 = stablehlo.broadcast_in_dim %3751, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3753 = "stablehlo.gather"(%3722, %3752) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3754 = stablehlo.negate %3753 : tensor<1x27x32x64xf32>
    %3755 = stablehlo.constant dense<0> : tensor<i32>
    %3756 = stablehlo.broadcast_in_dim %3755, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3757 = "stablehlo.gather"(%3722, %3756) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3758 = stablehlo.concatenate %3754, %3757, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3759 = stablehlo.broadcast_in_dim %3734, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3760 = stablehlo.multiply %3758, %3759 : tensor<1x27x32x128xf32>
    %3761 = stablehlo.add %3750, %3760 : tensor<1x27x32x128xf32>
    %3762 = stablehlo.slice %3715 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %3763 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %3764 = stablehlo.reshape %3763 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %3765 = stablehlo.broadcast_in_dim %3764, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %3766 = stablehlo.constant dense<0> : tensor<i32>
    %3767 = stablehlo.broadcast_in_dim %3766, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %3768 = stablehlo.compare  NE, %3765, %3767,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %3769 = stablehlo.and %3768, %3762 : tensor<1x1x27x27xi1>
    %3770 = stablehlo.convert %3769 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %3771 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3772 = stablehlo.broadcast_in_dim %3771, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3773 = stablehlo.compare  GT, %3770, %3772,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %3774 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3775 = stablehlo.broadcast_in_dim %3774, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3776 = stablehlo.convert %3775 : tensor<1x1x27x27xf32>
    %3777 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %3778 = stablehlo.broadcast_in_dim %3777, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3779 = stablehlo.select %3773, %3776, %3778 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %3780 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %3781 = stablehlo.sqrt %3780 : tensor<f32>
    %3782 = stablehlo.convert %3781 : tensor<f32>
    %3783 = stablehlo.broadcast_in_dim %3782, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %3784 = stablehlo.divide %3761, %3783 : tensor<1x27x32x128xf32>
    %3785 = stablehlo.dot_general %3784, %3748, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %3786 = stablehlo.broadcast_in_dim %3779, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %3787 = stablehlo.add %3785, %3786 : tensor<1x32x27x27xf32>
    %3788 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3789 = stablehlo.reduce(%3787 init: %3788) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3790 = stablehlo.broadcast_in_dim %3789, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3791 = stablehlo.broadcast_in_dim %3790, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3792 = stablehlo.subtract %3787, %3791 : tensor<1x32x27x27xf32>
    %3793 = stablehlo.exponential %3792 : tensor<1x32x27x27xf32>
    %3794 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3795 = stablehlo.reduce(%3793 init: %3794) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3796 = stablehlo.broadcast_in_dim %3795, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3797 = stablehlo.broadcast_in_dim %3796, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3798 = stablehlo.divide %3793, %3797 : tensor<1x32x27x27xf32>
    %3799 = stablehlo.dot_general %3724, %3798, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %3800 = stablehlo.transpose %3799, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %3801 = stablehlo.reshape %3800 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %3802 = stablehlo.convert %246 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3803 = stablehlo.dot_general %3801, %3802, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3804 = stablehlo.add %3690, %3803 : tensor<1x27x4096xf32>
    %3805 = stablehlo.multiply %3804, %3804 : tensor<1x27x4096xf32>
    %3806 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3807 = stablehlo.reduce(%3805 init: %3806) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3808 = stablehlo.broadcast_in_dim %3807, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3809 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3810 = stablehlo.broadcast_in_dim %3809, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3811 = stablehlo.divide %3808, %3810 : tensor<1x27x1xf32>
    %3812 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3813 = stablehlo.broadcast_in_dim %3812, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3814 = stablehlo.add %3811, %3813 : tensor<1x27x1xf32>
    %3815 = stablehlo.sqrt %3814 : tensor<1x27x1xf32>
    %3816 = stablehlo.broadcast_in_dim %3815, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3817 = stablehlo.divide %3804, %3816 : tensor<1x27x4096xf32>
    %3818 = stablehlo.convert %247 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3819 = stablehlo.broadcast_in_dim %3818, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3820 = stablehlo.broadcast_in_dim %3819, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3821 = stablehlo.multiply %3820, %3817 : tensor<1x27x4096xf32>
    %3822 = stablehlo.convert %248 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3823 = stablehlo.dot_general %3821, %3822, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3824 = stablehlo.convert %249 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3825 = stablehlo.dot_general %3821, %3824, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3826 = call @silu(%3825) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %3827 = stablehlo.multiply %3823, %3826 : tensor<1x27x11008xf32>
    %3828 = stablehlo.convert %250 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %3829 = stablehlo.dot_general %3827, %3828, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %3830 = stablehlo.add %3804, %3829 : tensor<1x27x4096xf32>
    %3831 = stablehlo.multiply %3830, %3830 : tensor<1x27x4096xf32>
    %3832 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3833 = stablehlo.reduce(%3831 init: %3832) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3834 = stablehlo.broadcast_in_dim %3833, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3835 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3836 = stablehlo.broadcast_in_dim %3835, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3837 = stablehlo.divide %3834, %3836 : tensor<1x27x1xf32>
    %3838 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3839 = stablehlo.broadcast_in_dim %3838, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3840 = stablehlo.add %3837, %3839 : tensor<1x27x1xf32>
    %3841 = stablehlo.sqrt %3840 : tensor<1x27x1xf32>
    %3842 = stablehlo.broadcast_in_dim %3841, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3843 = stablehlo.divide %3830, %3842 : tensor<1x27x4096xf32>
    %3844 = stablehlo.convert %251 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3845 = stablehlo.broadcast_in_dim %3844, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3846 = stablehlo.broadcast_in_dim %3845, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3847 = stablehlo.multiply %3846, %3843 : tensor<1x27x4096xf32>
    %3848 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %3849 = stablehlo.broadcast_in_dim %3848, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %3850 = stablehlo.broadcast_in_dim %3849, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %3851 = stablehlo.broadcast_in_dim %3849, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %3852 = stablehlo.broadcast_in_dim %3850, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %3853 = stablehlo.broadcast_in_dim %3851, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %3854 = stablehlo.compare  GE, %3852, %3853,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %3855 = stablehlo.broadcast_in_dim %3854, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %3856 = stablehlo.convert %252 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3857 = stablehlo.dot_general %3847, %3856, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3858 = stablehlo.convert %253 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3859 = stablehlo.dot_general %3847, %3858, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3860 = stablehlo.convert %254 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3861 = stablehlo.dot_general %3847, %3860, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3862 = stablehlo.reshape %3857 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3863 = stablehlo.reshape %3859 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3864 = stablehlo.reshape %3861 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %3865 = stablehlo.constant dense<0> : tensor<i32>
    %3866 = stablehlo.broadcast_in_dim %3865, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3867 = stablehlo.compare  LT, %324, %3866,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %3868 = stablehlo.constant dense<4096> : tensor<i32>
    %3869 = stablehlo.broadcast_in_dim %3868, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %3870 = stablehlo.add %324, %3869 : tensor<1x27xi32>
    %3871 = stablehlo.select %3867, %3870, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %3872 = stablehlo.broadcast_in_dim %3871, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %3873 = "stablehlo.gather"(%255, %3872) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %3874 = stablehlo.slice %3873 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3875 = stablehlo.slice %3873 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %3876 = stablehlo.broadcast_in_dim %3875, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3877 = stablehlo.multiply %3863, %3876 : tensor<1x27x32x128xf32>
    %3878 = stablehlo.constant dense<64> : tensor<i32>
    %3879 = stablehlo.broadcast_in_dim %3878, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3880 = "stablehlo.gather"(%3863, %3879) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3881 = stablehlo.negate %3880 : tensor<1x27x32x64xf32>
    %3882 = stablehlo.constant dense<0> : tensor<i32>
    %3883 = stablehlo.broadcast_in_dim %3882, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3884 = "stablehlo.gather"(%3863, %3883) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3885 = stablehlo.concatenate %3881, %3884, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3886 = stablehlo.broadcast_in_dim %3874, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3887 = stablehlo.multiply %3885, %3886 : tensor<1x27x32x128xf32>
    %3888 = stablehlo.add %3877, %3887 : tensor<1x27x32x128xf32>
    %3889 = stablehlo.broadcast_in_dim %3875, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3890 = stablehlo.multiply %3862, %3889 : tensor<1x27x32x128xf32>
    %3891 = stablehlo.constant dense<64> : tensor<i32>
    %3892 = stablehlo.broadcast_in_dim %3891, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3893 = "stablehlo.gather"(%3862, %3892) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3894 = stablehlo.negate %3893 : tensor<1x27x32x64xf32>
    %3895 = stablehlo.constant dense<0> : tensor<i32>
    %3896 = stablehlo.broadcast_in_dim %3895, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3897 = "stablehlo.gather"(%3862, %3896) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %3898 = stablehlo.concatenate %3894, %3897, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %3899 = stablehlo.broadcast_in_dim %3874, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %3900 = stablehlo.multiply %3898, %3899 : tensor<1x27x32x128xf32>
    %3901 = stablehlo.add %3890, %3900 : tensor<1x27x32x128xf32>
    %3902 = stablehlo.slice %3855 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %3903 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %3904 = stablehlo.reshape %3903 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %3905 = stablehlo.broadcast_in_dim %3904, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %3906 = stablehlo.constant dense<0> : tensor<i32>
    %3907 = stablehlo.broadcast_in_dim %3906, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %3908 = stablehlo.compare  NE, %3905, %3907,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %3909 = stablehlo.and %3908, %3902 : tensor<1x1x27x27xi1>
    %3910 = stablehlo.convert %3909 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %3911 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3912 = stablehlo.broadcast_in_dim %3911, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3913 = stablehlo.compare  GT, %3910, %3912,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %3914 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3915 = stablehlo.broadcast_in_dim %3914, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3916 = stablehlo.convert %3915 : tensor<1x1x27x27xf32>
    %3917 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %3918 = stablehlo.broadcast_in_dim %3917, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %3919 = stablehlo.select %3913, %3916, %3918 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %3920 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %3921 = stablehlo.sqrt %3920 : tensor<f32>
    %3922 = stablehlo.convert %3921 : tensor<f32>
    %3923 = stablehlo.broadcast_in_dim %3922, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %3924 = stablehlo.divide %3901, %3923 : tensor<1x27x32x128xf32>
    %3925 = stablehlo.dot_general %3924, %3888, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %3926 = stablehlo.broadcast_in_dim %3919, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %3927 = stablehlo.add %3925, %3926 : tensor<1x32x27x27xf32>
    %3928 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3929 = stablehlo.reduce(%3927 init: %3928) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3930 = stablehlo.broadcast_in_dim %3929, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3931 = stablehlo.broadcast_in_dim %3930, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3932 = stablehlo.subtract %3927, %3931 : tensor<1x32x27x27xf32>
    %3933 = stablehlo.exponential %3932 : tensor<1x32x27x27xf32>
    %3934 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3935 = stablehlo.reduce(%3933 init: %3934) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %3936 = stablehlo.broadcast_in_dim %3935, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %3937 = stablehlo.broadcast_in_dim %3936, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %3938 = stablehlo.divide %3933, %3937 : tensor<1x32x27x27xf32>
    %3939 = stablehlo.dot_general %3864, %3938, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %3940 = stablehlo.transpose %3939, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %3941 = stablehlo.reshape %3940 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %3942 = stablehlo.convert %256 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3943 = stablehlo.dot_general %3941, %3942, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3944 = stablehlo.add %3830, %3943 : tensor<1x27x4096xf32>
    %3945 = stablehlo.multiply %3944, %3944 : tensor<1x27x4096xf32>
    %3946 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3947 = stablehlo.reduce(%3945 init: %3946) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3948 = stablehlo.broadcast_in_dim %3947, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3949 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3950 = stablehlo.broadcast_in_dim %3949, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3951 = stablehlo.divide %3948, %3950 : tensor<1x27x1xf32>
    %3952 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3953 = stablehlo.broadcast_in_dim %3952, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3954 = stablehlo.add %3951, %3953 : tensor<1x27x1xf32>
    %3955 = stablehlo.sqrt %3954 : tensor<1x27x1xf32>
    %3956 = stablehlo.broadcast_in_dim %3955, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3957 = stablehlo.divide %3944, %3956 : tensor<1x27x4096xf32>
    %3958 = stablehlo.convert %257 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3959 = stablehlo.broadcast_in_dim %3958, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3960 = stablehlo.broadcast_in_dim %3959, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3961 = stablehlo.multiply %3960, %3957 : tensor<1x27x4096xf32>
    %3962 = stablehlo.convert %258 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3963 = stablehlo.dot_general %3961, %3962, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3964 = stablehlo.convert %259 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %3965 = stablehlo.dot_general %3961, %3964, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %3966 = call @silu(%3965) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %3967 = stablehlo.multiply %3963, %3966 : tensor<1x27x11008xf32>
    %3968 = stablehlo.convert %260 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %3969 = stablehlo.dot_general %3967, %3968, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %3970 = stablehlo.add %3944, %3969 : tensor<1x27x4096xf32>
    %3971 = stablehlo.multiply %3970, %3970 : tensor<1x27x4096xf32>
    %3972 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3973 = stablehlo.reduce(%3971 init: %3972) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %3974 = stablehlo.broadcast_in_dim %3973, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %3975 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %3976 = stablehlo.broadcast_in_dim %3975, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3977 = stablehlo.divide %3974, %3976 : tensor<1x27x1xf32>
    %3978 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3979 = stablehlo.broadcast_in_dim %3978, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %3980 = stablehlo.add %3977, %3979 : tensor<1x27x1xf32>
    %3981 = stablehlo.sqrt %3980 : tensor<1x27x1xf32>
    %3982 = stablehlo.broadcast_in_dim %3981, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %3983 = stablehlo.divide %3970, %3982 : tensor<1x27x4096xf32>
    %3984 = stablehlo.convert %261 : (tensor<4096xf16>) -> tensor<4096xf32>
    %3985 = stablehlo.broadcast_in_dim %3984, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3986 = stablehlo.broadcast_in_dim %3985, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %3987 = stablehlo.multiply %3986, %3983 : tensor<1x27x4096xf32>
    %3988 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %3989 = stablehlo.broadcast_in_dim %3988, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %3990 = stablehlo.broadcast_in_dim %3989, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %3991 = stablehlo.broadcast_in_dim %3989, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %3992 = stablehlo.broadcast_in_dim %3990, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %3993 = stablehlo.broadcast_in_dim %3991, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %3994 = stablehlo.compare  GE, %3992, %3993,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %3995 = stablehlo.broadcast_in_dim %3994, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %3996 = stablehlo.convert %262 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3997 = stablehlo.dot_general %3987, %3996, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %3998 = stablehlo.convert %263 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %3999 = stablehlo.dot_general %3987, %3998, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4000 = stablehlo.convert %264 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4001 = stablehlo.dot_general %3987, %4000, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4002 = stablehlo.reshape %3997 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4003 = stablehlo.reshape %3999 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4004 = stablehlo.reshape %4001 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4005 = stablehlo.constant dense<0> : tensor<i32>
    %4006 = stablehlo.broadcast_in_dim %4005, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %4007 = stablehlo.compare  LT, %324, %4006,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %4008 = stablehlo.constant dense<4096> : tensor<i32>
    %4009 = stablehlo.broadcast_in_dim %4008, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %4010 = stablehlo.add %324, %4009 : tensor<1x27xi32>
    %4011 = stablehlo.select %4007, %4010, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %4012 = stablehlo.broadcast_in_dim %4011, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %4013 = "stablehlo.gather"(%265, %4012) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %4014 = stablehlo.slice %4013 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %4015 = stablehlo.slice %4013 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %4016 = stablehlo.broadcast_in_dim %4015, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4017 = stablehlo.multiply %4003, %4016 : tensor<1x27x32x128xf32>
    %4018 = stablehlo.constant dense<64> : tensor<i32>
    %4019 = stablehlo.broadcast_in_dim %4018, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4020 = "stablehlo.gather"(%4003, %4019) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4021 = stablehlo.negate %4020 : tensor<1x27x32x64xf32>
    %4022 = stablehlo.constant dense<0> : tensor<i32>
    %4023 = stablehlo.broadcast_in_dim %4022, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4024 = "stablehlo.gather"(%4003, %4023) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4025 = stablehlo.concatenate %4021, %4024, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %4026 = stablehlo.broadcast_in_dim %4014, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4027 = stablehlo.multiply %4025, %4026 : tensor<1x27x32x128xf32>
    %4028 = stablehlo.add %4017, %4027 : tensor<1x27x32x128xf32>
    %4029 = stablehlo.broadcast_in_dim %4015, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4030 = stablehlo.multiply %4002, %4029 : tensor<1x27x32x128xf32>
    %4031 = stablehlo.constant dense<64> : tensor<i32>
    %4032 = stablehlo.broadcast_in_dim %4031, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4033 = "stablehlo.gather"(%4002, %4032) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4034 = stablehlo.negate %4033 : tensor<1x27x32x64xf32>
    %4035 = stablehlo.constant dense<0> : tensor<i32>
    %4036 = stablehlo.broadcast_in_dim %4035, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4037 = "stablehlo.gather"(%4002, %4036) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4038 = stablehlo.concatenate %4034, %4037, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %4039 = stablehlo.broadcast_in_dim %4014, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4040 = stablehlo.multiply %4038, %4039 : tensor<1x27x32x128xf32>
    %4041 = stablehlo.add %4030, %4040 : tensor<1x27x32x128xf32>
    %4042 = stablehlo.slice %3995 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %4043 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %4044 = stablehlo.reshape %4043 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %4045 = stablehlo.broadcast_in_dim %4044, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %4046 = stablehlo.constant dense<0> : tensor<i32>
    %4047 = stablehlo.broadcast_in_dim %4046, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %4048 = stablehlo.compare  NE, %4045, %4047,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %4049 = stablehlo.and %4048, %4042 : tensor<1x1x27x27xi1>
    %4050 = stablehlo.convert %4049 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %4051 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4052 = stablehlo.broadcast_in_dim %4051, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4053 = stablehlo.compare  GT, %4050, %4052,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %4054 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4055 = stablehlo.broadcast_in_dim %4054, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4056 = stablehlo.convert %4055 : tensor<1x1x27x27xf32>
    %4057 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %4058 = stablehlo.broadcast_in_dim %4057, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4059 = stablehlo.select %4053, %4056, %4058 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %4060 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %4061 = stablehlo.sqrt %4060 : tensor<f32>
    %4062 = stablehlo.convert %4061 : tensor<f32>
    %4063 = stablehlo.broadcast_in_dim %4062, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %4064 = stablehlo.divide %4041, %4063 : tensor<1x27x32x128xf32>
    %4065 = stablehlo.dot_general %4064, %4028, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %4066 = stablehlo.broadcast_in_dim %4059, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %4067 = stablehlo.add %4065, %4066 : tensor<1x32x27x27xf32>
    %4068 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4069 = stablehlo.reduce(%4067 init: %4068) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %4070 = stablehlo.broadcast_in_dim %4069, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %4071 = stablehlo.broadcast_in_dim %4070, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %4072 = stablehlo.subtract %4067, %4071 : tensor<1x32x27x27xf32>
    %4073 = stablehlo.exponential %4072 : tensor<1x32x27x27xf32>
    %4074 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4075 = stablehlo.reduce(%4073 init: %4074) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %4076 = stablehlo.broadcast_in_dim %4075, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %4077 = stablehlo.broadcast_in_dim %4076, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %4078 = stablehlo.divide %4073, %4077 : tensor<1x32x27x27xf32>
    %4079 = stablehlo.dot_general %4004, %4078, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %4080 = stablehlo.transpose %4079, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %4081 = stablehlo.reshape %4080 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %4082 = stablehlo.convert %266 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4083 = stablehlo.dot_general %4081, %4082, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4084 = stablehlo.add %3970, %4083 : tensor<1x27x4096xf32>
    %4085 = stablehlo.multiply %4084, %4084 : tensor<1x27x4096xf32>
    %4086 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4087 = stablehlo.reduce(%4085 init: %4086) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %4088 = stablehlo.broadcast_in_dim %4087, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %4089 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4090 = stablehlo.broadcast_in_dim %4089, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4091 = stablehlo.divide %4088, %4090 : tensor<1x27x1xf32>
    %4092 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4093 = stablehlo.broadcast_in_dim %4092, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4094 = stablehlo.add %4091, %4093 : tensor<1x27x1xf32>
    %4095 = stablehlo.sqrt %4094 : tensor<1x27x1xf32>
    %4096 = stablehlo.broadcast_in_dim %4095, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %4097 = stablehlo.divide %4084, %4096 : tensor<1x27x4096xf32>
    %4098 = stablehlo.convert %267 : (tensor<4096xf16>) -> tensor<4096xf32>
    %4099 = stablehlo.broadcast_in_dim %4098, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4100 = stablehlo.broadcast_in_dim %4099, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %4101 = stablehlo.multiply %4100, %4097 : tensor<1x27x4096xf32>
    %4102 = stablehlo.convert %268 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %4103 = stablehlo.dot_general %4101, %4102, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %4104 = stablehlo.convert %269 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %4105 = stablehlo.dot_general %4101, %4104, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %4106 = call @silu(%4105) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %4107 = stablehlo.multiply %4103, %4106 : tensor<1x27x11008xf32>
    %4108 = stablehlo.convert %270 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %4109 = stablehlo.dot_general %4107, %4108, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %4110 = stablehlo.add %4084, %4109 : tensor<1x27x4096xf32>
    %4111 = stablehlo.multiply %4110, %4110 : tensor<1x27x4096xf32>
    %4112 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4113 = stablehlo.reduce(%4111 init: %4112) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %4114 = stablehlo.broadcast_in_dim %4113, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %4115 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4116 = stablehlo.broadcast_in_dim %4115, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4117 = stablehlo.divide %4114, %4116 : tensor<1x27x1xf32>
    %4118 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4119 = stablehlo.broadcast_in_dim %4118, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4120 = stablehlo.add %4117, %4119 : tensor<1x27x1xf32>
    %4121 = stablehlo.sqrt %4120 : tensor<1x27x1xf32>
    %4122 = stablehlo.broadcast_in_dim %4121, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %4123 = stablehlo.divide %4110, %4122 : tensor<1x27x4096xf32>
    %4124 = stablehlo.convert %271 : (tensor<4096xf16>) -> tensor<4096xf32>
    %4125 = stablehlo.broadcast_in_dim %4124, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4126 = stablehlo.broadcast_in_dim %4125, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %4127 = stablehlo.multiply %4126, %4123 : tensor<1x27x4096xf32>
    %4128 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %4129 = stablehlo.broadcast_in_dim %4128, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %4130 = stablehlo.broadcast_in_dim %4129, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %4131 = stablehlo.broadcast_in_dim %4129, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %4132 = stablehlo.broadcast_in_dim %4130, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %4133 = stablehlo.broadcast_in_dim %4131, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %4134 = stablehlo.compare  GE, %4132, %4133,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %4135 = stablehlo.broadcast_in_dim %4134, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %4136 = stablehlo.convert %272 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4137 = stablehlo.dot_general %4127, %4136, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4138 = stablehlo.convert %273 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4139 = stablehlo.dot_general %4127, %4138, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4140 = stablehlo.convert %274 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4141 = stablehlo.dot_general %4127, %4140, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4142 = stablehlo.reshape %4137 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4143 = stablehlo.reshape %4139 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4144 = stablehlo.reshape %4141 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4145 = stablehlo.constant dense<0> : tensor<i32>
    %4146 = stablehlo.broadcast_in_dim %4145, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %4147 = stablehlo.compare  LT, %324, %4146,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %4148 = stablehlo.constant dense<4096> : tensor<i32>
    %4149 = stablehlo.broadcast_in_dim %4148, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %4150 = stablehlo.add %324, %4149 : tensor<1x27xi32>
    %4151 = stablehlo.select %4147, %4150, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %4152 = stablehlo.broadcast_in_dim %4151, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %4153 = "stablehlo.gather"(%275, %4152) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %4154 = stablehlo.slice %4153 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %4155 = stablehlo.slice %4153 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %4156 = stablehlo.broadcast_in_dim %4155, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4157 = stablehlo.multiply %4143, %4156 : tensor<1x27x32x128xf32>
    %4158 = stablehlo.constant dense<64> : tensor<i32>
    %4159 = stablehlo.broadcast_in_dim %4158, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4160 = "stablehlo.gather"(%4143, %4159) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4161 = stablehlo.negate %4160 : tensor<1x27x32x64xf32>
    %4162 = stablehlo.constant dense<0> : tensor<i32>
    %4163 = stablehlo.broadcast_in_dim %4162, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4164 = "stablehlo.gather"(%4143, %4163) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4165 = stablehlo.concatenate %4161, %4164, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %4166 = stablehlo.broadcast_in_dim %4154, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4167 = stablehlo.multiply %4165, %4166 : tensor<1x27x32x128xf32>
    %4168 = stablehlo.add %4157, %4167 : tensor<1x27x32x128xf32>
    %4169 = stablehlo.broadcast_in_dim %4155, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4170 = stablehlo.multiply %4142, %4169 : tensor<1x27x32x128xf32>
    %4171 = stablehlo.constant dense<64> : tensor<i32>
    %4172 = stablehlo.broadcast_in_dim %4171, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4173 = "stablehlo.gather"(%4142, %4172) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4174 = stablehlo.negate %4173 : tensor<1x27x32x64xf32>
    %4175 = stablehlo.constant dense<0> : tensor<i32>
    %4176 = stablehlo.broadcast_in_dim %4175, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4177 = "stablehlo.gather"(%4142, %4176) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4178 = stablehlo.concatenate %4174, %4177, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %4179 = stablehlo.broadcast_in_dim %4154, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4180 = stablehlo.multiply %4178, %4179 : tensor<1x27x32x128xf32>
    %4181 = stablehlo.add %4170, %4180 : tensor<1x27x32x128xf32>
    %4182 = stablehlo.slice %4135 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %4183 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %4184 = stablehlo.reshape %4183 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %4185 = stablehlo.broadcast_in_dim %4184, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %4186 = stablehlo.constant dense<0> : tensor<i32>
    %4187 = stablehlo.broadcast_in_dim %4186, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %4188 = stablehlo.compare  NE, %4185, %4187,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %4189 = stablehlo.and %4188, %4182 : tensor<1x1x27x27xi1>
    %4190 = stablehlo.convert %4189 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %4191 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4192 = stablehlo.broadcast_in_dim %4191, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4193 = stablehlo.compare  GT, %4190, %4192,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %4194 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4195 = stablehlo.broadcast_in_dim %4194, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4196 = stablehlo.convert %4195 : tensor<1x1x27x27xf32>
    %4197 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %4198 = stablehlo.broadcast_in_dim %4197, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4199 = stablehlo.select %4193, %4196, %4198 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %4200 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %4201 = stablehlo.sqrt %4200 : tensor<f32>
    %4202 = stablehlo.convert %4201 : tensor<f32>
    %4203 = stablehlo.broadcast_in_dim %4202, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %4204 = stablehlo.divide %4181, %4203 : tensor<1x27x32x128xf32>
    %4205 = stablehlo.dot_general %4204, %4168, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %4206 = stablehlo.broadcast_in_dim %4199, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %4207 = stablehlo.add %4205, %4206 : tensor<1x32x27x27xf32>
    %4208 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4209 = stablehlo.reduce(%4207 init: %4208) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %4210 = stablehlo.broadcast_in_dim %4209, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %4211 = stablehlo.broadcast_in_dim %4210, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %4212 = stablehlo.subtract %4207, %4211 : tensor<1x32x27x27xf32>
    %4213 = stablehlo.exponential %4212 : tensor<1x32x27x27xf32>
    %4214 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4215 = stablehlo.reduce(%4213 init: %4214) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %4216 = stablehlo.broadcast_in_dim %4215, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %4217 = stablehlo.broadcast_in_dim %4216, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %4218 = stablehlo.divide %4213, %4217 : tensor<1x32x27x27xf32>
    %4219 = stablehlo.dot_general %4144, %4218, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %4220 = stablehlo.transpose %4219, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %4221 = stablehlo.reshape %4220 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %4222 = stablehlo.convert %276 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4223 = stablehlo.dot_general %4221, %4222, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4224 = stablehlo.add %4110, %4223 : tensor<1x27x4096xf32>
    %4225 = stablehlo.multiply %4224, %4224 : tensor<1x27x4096xf32>
    %4226 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4227 = stablehlo.reduce(%4225 init: %4226) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %4228 = stablehlo.broadcast_in_dim %4227, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %4229 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4230 = stablehlo.broadcast_in_dim %4229, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4231 = stablehlo.divide %4228, %4230 : tensor<1x27x1xf32>
    %4232 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4233 = stablehlo.broadcast_in_dim %4232, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4234 = stablehlo.add %4231, %4233 : tensor<1x27x1xf32>
    %4235 = stablehlo.sqrt %4234 : tensor<1x27x1xf32>
    %4236 = stablehlo.broadcast_in_dim %4235, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %4237 = stablehlo.divide %4224, %4236 : tensor<1x27x4096xf32>
    %4238 = stablehlo.convert %277 : (tensor<4096xf16>) -> tensor<4096xf32>
    %4239 = stablehlo.broadcast_in_dim %4238, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4240 = stablehlo.broadcast_in_dim %4239, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %4241 = stablehlo.multiply %4240, %4237 : tensor<1x27x4096xf32>
    %4242 = stablehlo.convert %278 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %4243 = stablehlo.dot_general %4241, %4242, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %4244 = stablehlo.convert %279 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %4245 = stablehlo.dot_general %4241, %4244, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %4246 = call @silu(%4245) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %4247 = stablehlo.multiply %4243, %4246 : tensor<1x27x11008xf32>
    %4248 = stablehlo.convert %280 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %4249 = stablehlo.dot_general %4247, %4248, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %4250 = stablehlo.add %4224, %4249 : tensor<1x27x4096xf32>
    %4251 = stablehlo.multiply %4250, %4250 : tensor<1x27x4096xf32>
    %4252 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4253 = stablehlo.reduce(%4251 init: %4252) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %4254 = stablehlo.broadcast_in_dim %4253, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %4255 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4256 = stablehlo.broadcast_in_dim %4255, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4257 = stablehlo.divide %4254, %4256 : tensor<1x27x1xf32>
    %4258 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4259 = stablehlo.broadcast_in_dim %4258, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4260 = stablehlo.add %4257, %4259 : tensor<1x27x1xf32>
    %4261 = stablehlo.sqrt %4260 : tensor<1x27x1xf32>
    %4262 = stablehlo.broadcast_in_dim %4261, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %4263 = stablehlo.divide %4250, %4262 : tensor<1x27x4096xf32>
    %4264 = stablehlo.convert %281 : (tensor<4096xf16>) -> tensor<4096xf32>
    %4265 = stablehlo.broadcast_in_dim %4264, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4266 = stablehlo.broadcast_in_dim %4265, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %4267 = stablehlo.multiply %4266, %4263 : tensor<1x27x4096xf32>
    %4268 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %4269 = stablehlo.broadcast_in_dim %4268, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %4270 = stablehlo.broadcast_in_dim %4269, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %4271 = stablehlo.broadcast_in_dim %4269, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %4272 = stablehlo.broadcast_in_dim %4270, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %4273 = stablehlo.broadcast_in_dim %4271, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %4274 = stablehlo.compare  GE, %4272, %4273,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %4275 = stablehlo.broadcast_in_dim %4274, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %4276 = stablehlo.convert %282 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4277 = stablehlo.dot_general %4267, %4276, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4278 = stablehlo.convert %283 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4279 = stablehlo.dot_general %4267, %4278, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4280 = stablehlo.convert %284 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4281 = stablehlo.dot_general %4267, %4280, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4282 = stablehlo.reshape %4277 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4283 = stablehlo.reshape %4279 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4284 = stablehlo.reshape %4281 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4285 = stablehlo.constant dense<0> : tensor<i32>
    %4286 = stablehlo.broadcast_in_dim %4285, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %4287 = stablehlo.compare  LT, %324, %4286,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %4288 = stablehlo.constant dense<4096> : tensor<i32>
    %4289 = stablehlo.broadcast_in_dim %4288, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %4290 = stablehlo.add %324, %4289 : tensor<1x27xi32>
    %4291 = stablehlo.select %4287, %4290, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %4292 = stablehlo.broadcast_in_dim %4291, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %4293 = "stablehlo.gather"(%285, %4292) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %4294 = stablehlo.slice %4293 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %4295 = stablehlo.slice %4293 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %4296 = stablehlo.broadcast_in_dim %4295, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4297 = stablehlo.multiply %4283, %4296 : tensor<1x27x32x128xf32>
    %4298 = stablehlo.constant dense<64> : tensor<i32>
    %4299 = stablehlo.broadcast_in_dim %4298, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4300 = "stablehlo.gather"(%4283, %4299) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4301 = stablehlo.negate %4300 : tensor<1x27x32x64xf32>
    %4302 = stablehlo.constant dense<0> : tensor<i32>
    %4303 = stablehlo.broadcast_in_dim %4302, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4304 = "stablehlo.gather"(%4283, %4303) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4305 = stablehlo.concatenate %4301, %4304, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %4306 = stablehlo.broadcast_in_dim %4294, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4307 = stablehlo.multiply %4305, %4306 : tensor<1x27x32x128xf32>
    %4308 = stablehlo.add %4297, %4307 : tensor<1x27x32x128xf32>
    %4309 = stablehlo.broadcast_in_dim %4295, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4310 = stablehlo.multiply %4282, %4309 : tensor<1x27x32x128xf32>
    %4311 = stablehlo.constant dense<64> : tensor<i32>
    %4312 = stablehlo.broadcast_in_dim %4311, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4313 = "stablehlo.gather"(%4282, %4312) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4314 = stablehlo.negate %4313 : tensor<1x27x32x64xf32>
    %4315 = stablehlo.constant dense<0> : tensor<i32>
    %4316 = stablehlo.broadcast_in_dim %4315, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4317 = "stablehlo.gather"(%4282, %4316) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4318 = stablehlo.concatenate %4314, %4317, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %4319 = stablehlo.broadcast_in_dim %4294, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4320 = stablehlo.multiply %4318, %4319 : tensor<1x27x32x128xf32>
    %4321 = stablehlo.add %4310, %4320 : tensor<1x27x32x128xf32>
    %4322 = stablehlo.slice %4275 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %4323 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %4324 = stablehlo.reshape %4323 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %4325 = stablehlo.broadcast_in_dim %4324, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %4326 = stablehlo.constant dense<0> : tensor<i32>
    %4327 = stablehlo.broadcast_in_dim %4326, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %4328 = stablehlo.compare  NE, %4325, %4327,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %4329 = stablehlo.and %4328, %4322 : tensor<1x1x27x27xi1>
    %4330 = stablehlo.convert %4329 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %4331 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4332 = stablehlo.broadcast_in_dim %4331, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4333 = stablehlo.compare  GT, %4330, %4332,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %4334 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4335 = stablehlo.broadcast_in_dim %4334, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4336 = stablehlo.convert %4335 : tensor<1x1x27x27xf32>
    %4337 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %4338 = stablehlo.broadcast_in_dim %4337, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4339 = stablehlo.select %4333, %4336, %4338 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %4340 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %4341 = stablehlo.sqrt %4340 : tensor<f32>
    %4342 = stablehlo.convert %4341 : tensor<f32>
    %4343 = stablehlo.broadcast_in_dim %4342, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %4344 = stablehlo.divide %4321, %4343 : tensor<1x27x32x128xf32>
    %4345 = stablehlo.dot_general %4344, %4308, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %4346 = stablehlo.broadcast_in_dim %4339, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %4347 = stablehlo.add %4345, %4346 : tensor<1x32x27x27xf32>
    %4348 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4349 = stablehlo.reduce(%4347 init: %4348) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %4350 = stablehlo.broadcast_in_dim %4349, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %4351 = stablehlo.broadcast_in_dim %4350, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %4352 = stablehlo.subtract %4347, %4351 : tensor<1x32x27x27xf32>
    %4353 = stablehlo.exponential %4352 : tensor<1x32x27x27xf32>
    %4354 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4355 = stablehlo.reduce(%4353 init: %4354) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %4356 = stablehlo.broadcast_in_dim %4355, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %4357 = stablehlo.broadcast_in_dim %4356, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %4358 = stablehlo.divide %4353, %4357 : tensor<1x32x27x27xf32>
    %4359 = stablehlo.dot_general %4284, %4358, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %4360 = stablehlo.transpose %4359, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %4361 = stablehlo.reshape %4360 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %4362 = stablehlo.convert %286 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4363 = stablehlo.dot_general %4361, %4362, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4364 = stablehlo.add %4250, %4363 : tensor<1x27x4096xf32>
    %4365 = stablehlo.multiply %4364, %4364 : tensor<1x27x4096xf32>
    %4366 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4367 = stablehlo.reduce(%4365 init: %4366) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %4368 = stablehlo.broadcast_in_dim %4367, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %4369 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4370 = stablehlo.broadcast_in_dim %4369, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4371 = stablehlo.divide %4368, %4370 : tensor<1x27x1xf32>
    %4372 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4373 = stablehlo.broadcast_in_dim %4372, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4374 = stablehlo.add %4371, %4373 : tensor<1x27x1xf32>
    %4375 = stablehlo.sqrt %4374 : tensor<1x27x1xf32>
    %4376 = stablehlo.broadcast_in_dim %4375, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %4377 = stablehlo.divide %4364, %4376 : tensor<1x27x4096xf32>
    %4378 = stablehlo.convert %287 : (tensor<4096xf16>) -> tensor<4096xf32>
    %4379 = stablehlo.broadcast_in_dim %4378, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4380 = stablehlo.broadcast_in_dim %4379, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %4381 = stablehlo.multiply %4380, %4377 : tensor<1x27x4096xf32>
    %4382 = stablehlo.convert %288 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %4383 = stablehlo.dot_general %4381, %4382, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %4384 = stablehlo.convert %289 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %4385 = stablehlo.dot_general %4381, %4384, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %4386 = call @silu(%4385) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %4387 = stablehlo.multiply %4383, %4386 : tensor<1x27x11008xf32>
    %4388 = stablehlo.convert %290 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %4389 = stablehlo.dot_general %4387, %4388, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %4390 = stablehlo.add %4364, %4389 : tensor<1x27x4096xf32>
    %4391 = stablehlo.multiply %4390, %4390 : tensor<1x27x4096xf32>
    %4392 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4393 = stablehlo.reduce(%4391 init: %4392) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %4394 = stablehlo.broadcast_in_dim %4393, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %4395 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4396 = stablehlo.broadcast_in_dim %4395, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4397 = stablehlo.divide %4394, %4396 : tensor<1x27x1xf32>
    %4398 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4399 = stablehlo.broadcast_in_dim %4398, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4400 = stablehlo.add %4397, %4399 : tensor<1x27x1xf32>
    %4401 = stablehlo.sqrt %4400 : tensor<1x27x1xf32>
    %4402 = stablehlo.broadcast_in_dim %4401, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %4403 = stablehlo.divide %4390, %4402 : tensor<1x27x4096xf32>
    %4404 = stablehlo.convert %291 : (tensor<4096xf16>) -> tensor<4096xf32>
    %4405 = stablehlo.broadcast_in_dim %4404, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4406 = stablehlo.broadcast_in_dim %4405, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %4407 = stablehlo.multiply %4406, %4403 : tensor<1x27x4096xf32>
    %4408 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %4409 = stablehlo.broadcast_in_dim %4408, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %4410 = stablehlo.broadcast_in_dim %4409, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %4411 = stablehlo.broadcast_in_dim %4409, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %4412 = stablehlo.broadcast_in_dim %4410, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %4413 = stablehlo.broadcast_in_dim %4411, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %4414 = stablehlo.compare  GE, %4412, %4413,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %4415 = stablehlo.broadcast_in_dim %4414, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %4416 = stablehlo.convert %292 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4417 = stablehlo.dot_general %4407, %4416, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4418 = stablehlo.convert %293 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4419 = stablehlo.dot_general %4407, %4418, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4420 = stablehlo.convert %294 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4421 = stablehlo.dot_general %4407, %4420, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4422 = stablehlo.reshape %4417 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4423 = stablehlo.reshape %4419 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4424 = stablehlo.reshape %4421 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4425 = stablehlo.constant dense<0> : tensor<i32>
    %4426 = stablehlo.broadcast_in_dim %4425, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %4427 = stablehlo.compare  LT, %324, %4426,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %4428 = stablehlo.constant dense<4096> : tensor<i32>
    %4429 = stablehlo.broadcast_in_dim %4428, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %4430 = stablehlo.add %324, %4429 : tensor<1x27xi32>
    %4431 = stablehlo.select %4427, %4430, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %4432 = stablehlo.broadcast_in_dim %4431, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %4433 = "stablehlo.gather"(%295, %4432) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %4434 = stablehlo.slice %4433 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %4435 = stablehlo.slice %4433 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %4436 = stablehlo.broadcast_in_dim %4435, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4437 = stablehlo.multiply %4423, %4436 : tensor<1x27x32x128xf32>
    %4438 = stablehlo.constant dense<64> : tensor<i32>
    %4439 = stablehlo.broadcast_in_dim %4438, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4440 = "stablehlo.gather"(%4423, %4439) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4441 = stablehlo.negate %4440 : tensor<1x27x32x64xf32>
    %4442 = stablehlo.constant dense<0> : tensor<i32>
    %4443 = stablehlo.broadcast_in_dim %4442, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4444 = "stablehlo.gather"(%4423, %4443) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4445 = stablehlo.concatenate %4441, %4444, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %4446 = stablehlo.broadcast_in_dim %4434, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4447 = stablehlo.multiply %4445, %4446 : tensor<1x27x32x128xf32>
    %4448 = stablehlo.add %4437, %4447 : tensor<1x27x32x128xf32>
    %4449 = stablehlo.broadcast_in_dim %4435, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4450 = stablehlo.multiply %4422, %4449 : tensor<1x27x32x128xf32>
    %4451 = stablehlo.constant dense<64> : tensor<i32>
    %4452 = stablehlo.broadcast_in_dim %4451, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4453 = "stablehlo.gather"(%4422, %4452) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4454 = stablehlo.negate %4453 : tensor<1x27x32x64xf32>
    %4455 = stablehlo.constant dense<0> : tensor<i32>
    %4456 = stablehlo.broadcast_in_dim %4455, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4457 = "stablehlo.gather"(%4422, %4456) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4458 = stablehlo.concatenate %4454, %4457, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %4459 = stablehlo.broadcast_in_dim %4434, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4460 = stablehlo.multiply %4458, %4459 : tensor<1x27x32x128xf32>
    %4461 = stablehlo.add %4450, %4460 : tensor<1x27x32x128xf32>
    %4462 = stablehlo.slice %4415 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %4463 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %4464 = stablehlo.reshape %4463 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %4465 = stablehlo.broadcast_in_dim %4464, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %4466 = stablehlo.constant dense<0> : tensor<i32>
    %4467 = stablehlo.broadcast_in_dim %4466, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %4468 = stablehlo.compare  NE, %4465, %4467,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %4469 = stablehlo.and %4468, %4462 : tensor<1x1x27x27xi1>
    %4470 = stablehlo.convert %4469 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %4471 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4472 = stablehlo.broadcast_in_dim %4471, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4473 = stablehlo.compare  GT, %4470, %4472,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %4474 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4475 = stablehlo.broadcast_in_dim %4474, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4476 = stablehlo.convert %4475 : tensor<1x1x27x27xf32>
    %4477 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %4478 = stablehlo.broadcast_in_dim %4477, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4479 = stablehlo.select %4473, %4476, %4478 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %4480 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %4481 = stablehlo.sqrt %4480 : tensor<f32>
    %4482 = stablehlo.convert %4481 : tensor<f32>
    %4483 = stablehlo.broadcast_in_dim %4482, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %4484 = stablehlo.divide %4461, %4483 : tensor<1x27x32x128xf32>
    %4485 = stablehlo.dot_general %4484, %4448, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %4486 = stablehlo.broadcast_in_dim %4479, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %4487 = stablehlo.add %4485, %4486 : tensor<1x32x27x27xf32>
    %4488 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4489 = stablehlo.reduce(%4487 init: %4488) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %4490 = stablehlo.broadcast_in_dim %4489, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %4491 = stablehlo.broadcast_in_dim %4490, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %4492 = stablehlo.subtract %4487, %4491 : tensor<1x32x27x27xf32>
    %4493 = stablehlo.exponential %4492 : tensor<1x32x27x27xf32>
    %4494 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4495 = stablehlo.reduce(%4493 init: %4494) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %4496 = stablehlo.broadcast_in_dim %4495, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %4497 = stablehlo.broadcast_in_dim %4496, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %4498 = stablehlo.divide %4493, %4497 : tensor<1x32x27x27xf32>
    %4499 = stablehlo.dot_general %4424, %4498, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %4500 = stablehlo.transpose %4499, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %4501 = stablehlo.reshape %4500 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %4502 = stablehlo.convert %296 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4503 = stablehlo.dot_general %4501, %4502, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4504 = stablehlo.add %4390, %4503 : tensor<1x27x4096xf32>
    %4505 = stablehlo.multiply %4504, %4504 : tensor<1x27x4096xf32>
    %4506 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4507 = stablehlo.reduce(%4505 init: %4506) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %4508 = stablehlo.broadcast_in_dim %4507, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %4509 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4510 = stablehlo.broadcast_in_dim %4509, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4511 = stablehlo.divide %4508, %4510 : tensor<1x27x1xf32>
    %4512 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4513 = stablehlo.broadcast_in_dim %4512, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4514 = stablehlo.add %4511, %4513 : tensor<1x27x1xf32>
    %4515 = stablehlo.sqrt %4514 : tensor<1x27x1xf32>
    %4516 = stablehlo.broadcast_in_dim %4515, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %4517 = stablehlo.divide %4504, %4516 : tensor<1x27x4096xf32>
    %4518 = stablehlo.convert %297 : (tensor<4096xf16>) -> tensor<4096xf32>
    %4519 = stablehlo.broadcast_in_dim %4518, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4520 = stablehlo.broadcast_in_dim %4519, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %4521 = stablehlo.multiply %4520, %4517 : tensor<1x27x4096xf32>
    %4522 = stablehlo.convert %298 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %4523 = stablehlo.dot_general %4521, %4522, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %4524 = stablehlo.convert %299 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %4525 = stablehlo.dot_general %4521, %4524, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %4526 = call @silu(%4525) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %4527 = stablehlo.multiply %4523, %4526 : tensor<1x27x11008xf32>
    %4528 = stablehlo.convert %300 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %4529 = stablehlo.dot_general %4527, %4528, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %4530 = stablehlo.add %4504, %4529 : tensor<1x27x4096xf32>
    %4531 = stablehlo.multiply %4530, %4530 : tensor<1x27x4096xf32>
    %4532 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4533 = stablehlo.reduce(%4531 init: %4532) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %4534 = stablehlo.broadcast_in_dim %4533, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %4535 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4536 = stablehlo.broadcast_in_dim %4535, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4537 = stablehlo.divide %4534, %4536 : tensor<1x27x1xf32>
    %4538 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4539 = stablehlo.broadcast_in_dim %4538, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4540 = stablehlo.add %4537, %4539 : tensor<1x27x1xf32>
    %4541 = stablehlo.sqrt %4540 : tensor<1x27x1xf32>
    %4542 = stablehlo.broadcast_in_dim %4541, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %4543 = stablehlo.divide %4530, %4542 : tensor<1x27x4096xf32>
    %4544 = stablehlo.convert %301 : (tensor<4096xf16>) -> tensor<4096xf32>
    %4545 = stablehlo.broadcast_in_dim %4544, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4546 = stablehlo.broadcast_in_dim %4545, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %4547 = stablehlo.multiply %4546, %4543 : tensor<1x27x4096xf32>
    %4548 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %4549 = stablehlo.broadcast_in_dim %4548, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %4550 = stablehlo.broadcast_in_dim %4549, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %4551 = stablehlo.broadcast_in_dim %4549, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %4552 = stablehlo.broadcast_in_dim %4550, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %4553 = stablehlo.broadcast_in_dim %4551, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %4554 = stablehlo.compare  GE, %4552, %4553,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %4555 = stablehlo.broadcast_in_dim %4554, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %4556 = stablehlo.convert %302 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4557 = stablehlo.dot_general %4547, %4556, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4558 = stablehlo.convert %303 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4559 = stablehlo.dot_general %4547, %4558, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4560 = stablehlo.convert %304 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4561 = stablehlo.dot_general %4547, %4560, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4562 = stablehlo.reshape %4557 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4563 = stablehlo.reshape %4559 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4564 = stablehlo.reshape %4561 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4565 = stablehlo.constant dense<0> : tensor<i32>
    %4566 = stablehlo.broadcast_in_dim %4565, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %4567 = stablehlo.compare  LT, %324, %4566,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %4568 = stablehlo.constant dense<4096> : tensor<i32>
    %4569 = stablehlo.broadcast_in_dim %4568, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %4570 = stablehlo.add %324, %4569 : tensor<1x27xi32>
    %4571 = stablehlo.select %4567, %4570, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %4572 = stablehlo.broadcast_in_dim %4571, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %4573 = "stablehlo.gather"(%305, %4572) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %4574 = stablehlo.slice %4573 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %4575 = stablehlo.slice %4573 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %4576 = stablehlo.broadcast_in_dim %4575, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4577 = stablehlo.multiply %4563, %4576 : tensor<1x27x32x128xf32>
    %4578 = stablehlo.constant dense<64> : tensor<i32>
    %4579 = stablehlo.broadcast_in_dim %4578, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4580 = "stablehlo.gather"(%4563, %4579) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4581 = stablehlo.negate %4580 : tensor<1x27x32x64xf32>
    %4582 = stablehlo.constant dense<0> : tensor<i32>
    %4583 = stablehlo.broadcast_in_dim %4582, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4584 = "stablehlo.gather"(%4563, %4583) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4585 = stablehlo.concatenate %4581, %4584, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %4586 = stablehlo.broadcast_in_dim %4574, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4587 = stablehlo.multiply %4585, %4586 : tensor<1x27x32x128xf32>
    %4588 = stablehlo.add %4577, %4587 : tensor<1x27x32x128xf32>
    %4589 = stablehlo.broadcast_in_dim %4575, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4590 = stablehlo.multiply %4562, %4589 : tensor<1x27x32x128xf32>
    %4591 = stablehlo.constant dense<64> : tensor<i32>
    %4592 = stablehlo.broadcast_in_dim %4591, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4593 = "stablehlo.gather"(%4562, %4592) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4594 = stablehlo.negate %4593 : tensor<1x27x32x64xf32>
    %4595 = stablehlo.constant dense<0> : tensor<i32>
    %4596 = stablehlo.broadcast_in_dim %4595, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4597 = "stablehlo.gather"(%4562, %4596) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4598 = stablehlo.concatenate %4594, %4597, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %4599 = stablehlo.broadcast_in_dim %4574, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4600 = stablehlo.multiply %4598, %4599 : tensor<1x27x32x128xf32>
    %4601 = stablehlo.add %4590, %4600 : tensor<1x27x32x128xf32>
    %4602 = stablehlo.slice %4555 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %4603 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %4604 = stablehlo.reshape %4603 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %4605 = stablehlo.broadcast_in_dim %4604, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %4606 = stablehlo.constant dense<0> : tensor<i32>
    %4607 = stablehlo.broadcast_in_dim %4606, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %4608 = stablehlo.compare  NE, %4605, %4607,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %4609 = stablehlo.and %4608, %4602 : tensor<1x1x27x27xi1>
    %4610 = stablehlo.convert %4609 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %4611 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4612 = stablehlo.broadcast_in_dim %4611, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4613 = stablehlo.compare  GT, %4610, %4612,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %4614 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4615 = stablehlo.broadcast_in_dim %4614, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4616 = stablehlo.convert %4615 : tensor<1x1x27x27xf32>
    %4617 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %4618 = stablehlo.broadcast_in_dim %4617, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4619 = stablehlo.select %4613, %4616, %4618 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %4620 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %4621 = stablehlo.sqrt %4620 : tensor<f32>
    %4622 = stablehlo.convert %4621 : tensor<f32>
    %4623 = stablehlo.broadcast_in_dim %4622, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %4624 = stablehlo.divide %4601, %4623 : tensor<1x27x32x128xf32>
    %4625 = stablehlo.dot_general %4624, %4588, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %4626 = stablehlo.broadcast_in_dim %4619, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %4627 = stablehlo.add %4625, %4626 : tensor<1x32x27x27xf32>
    %4628 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4629 = stablehlo.reduce(%4627 init: %4628) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %4630 = stablehlo.broadcast_in_dim %4629, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %4631 = stablehlo.broadcast_in_dim %4630, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %4632 = stablehlo.subtract %4627, %4631 : tensor<1x32x27x27xf32>
    %4633 = stablehlo.exponential %4632 : tensor<1x32x27x27xf32>
    %4634 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4635 = stablehlo.reduce(%4633 init: %4634) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %4636 = stablehlo.broadcast_in_dim %4635, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %4637 = stablehlo.broadcast_in_dim %4636, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %4638 = stablehlo.divide %4633, %4637 : tensor<1x32x27x27xf32>
    %4639 = stablehlo.dot_general %4564, %4638, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %4640 = stablehlo.transpose %4639, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %4641 = stablehlo.reshape %4640 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %4642 = stablehlo.convert %306 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4643 = stablehlo.dot_general %4641, %4642, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4644 = stablehlo.add %4530, %4643 : tensor<1x27x4096xf32>
    %4645 = stablehlo.multiply %4644, %4644 : tensor<1x27x4096xf32>
    %4646 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4647 = stablehlo.reduce(%4645 init: %4646) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %4648 = stablehlo.broadcast_in_dim %4647, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %4649 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4650 = stablehlo.broadcast_in_dim %4649, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4651 = stablehlo.divide %4648, %4650 : tensor<1x27x1xf32>
    %4652 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4653 = stablehlo.broadcast_in_dim %4652, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4654 = stablehlo.add %4651, %4653 : tensor<1x27x1xf32>
    %4655 = stablehlo.sqrt %4654 : tensor<1x27x1xf32>
    %4656 = stablehlo.broadcast_in_dim %4655, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %4657 = stablehlo.divide %4644, %4656 : tensor<1x27x4096xf32>
    %4658 = stablehlo.convert %307 : (tensor<4096xf16>) -> tensor<4096xf32>
    %4659 = stablehlo.broadcast_in_dim %4658, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4660 = stablehlo.broadcast_in_dim %4659, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %4661 = stablehlo.multiply %4660, %4657 : tensor<1x27x4096xf32>
    %4662 = stablehlo.convert %308 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %4663 = stablehlo.dot_general %4661, %4662, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %4664 = stablehlo.convert %309 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %4665 = stablehlo.dot_general %4661, %4664, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %4666 = call @silu(%4665) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %4667 = stablehlo.multiply %4663, %4666 : tensor<1x27x11008xf32>
    %4668 = stablehlo.convert %310 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %4669 = stablehlo.dot_general %4667, %4668, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %4670 = stablehlo.add %4644, %4669 : tensor<1x27x4096xf32>
    %4671 = stablehlo.multiply %4670, %4670 : tensor<1x27x4096xf32>
    %4672 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4673 = stablehlo.reduce(%4671 init: %4672) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %4674 = stablehlo.broadcast_in_dim %4673, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %4675 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4676 = stablehlo.broadcast_in_dim %4675, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4677 = stablehlo.divide %4674, %4676 : tensor<1x27x1xf32>
    %4678 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4679 = stablehlo.broadcast_in_dim %4678, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4680 = stablehlo.add %4677, %4679 : tensor<1x27x1xf32>
    %4681 = stablehlo.sqrt %4680 : tensor<1x27x1xf32>
    %4682 = stablehlo.broadcast_in_dim %4681, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %4683 = stablehlo.divide %4670, %4682 : tensor<1x27x4096xf32>
    %4684 = stablehlo.convert %311 : (tensor<4096xf16>) -> tensor<4096xf32>
    %4685 = stablehlo.broadcast_in_dim %4684, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4686 = stablehlo.broadcast_in_dim %4685, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %4687 = stablehlo.multiply %4686, %4683 : tensor<1x27x4096xf32>
    %4688 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %4689 = stablehlo.broadcast_in_dim %4688, dims = [1] : (tensor<4096xi32>) -> tensor<1x4096xi32>
    %4690 = stablehlo.broadcast_in_dim %4689, dims = [0, 1] : (tensor<1x4096xi32>) -> tensor<1x4096x1xi32>
    %4691 = stablehlo.broadcast_in_dim %4689, dims = [0, 2] : (tensor<1x4096xi32>) -> tensor<1x1x4096xi32>
    %4692 = stablehlo.broadcast_in_dim %4690, dims = [0, 1, 2] : (tensor<1x4096x1xi32>) -> tensor<1x4096x4096xi32>
    %4693 = stablehlo.broadcast_in_dim %4691, dims = [0, 1, 2] : (tensor<1x1x4096xi32>) -> tensor<1x4096x4096xi32>
    %4694 = stablehlo.compare  GE, %4692, %4693,  SIGNED : (tensor<1x4096x4096xi32>, tensor<1x4096x4096xi32>) -> tensor<1x4096x4096xi1>
    %4695 = stablehlo.broadcast_in_dim %4694, dims = [0, 2, 3] : (tensor<1x4096x4096xi1>) -> tensor<1x1x4096x4096xi1>
    %4696 = stablehlo.convert %312 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4697 = stablehlo.dot_general %4687, %4696, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4698 = stablehlo.convert %313 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4699 = stablehlo.dot_general %4687, %4698, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4700 = stablehlo.convert %314 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4701 = stablehlo.dot_general %4687, %4700, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4702 = stablehlo.reshape %4697 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4703 = stablehlo.reshape %4699 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4704 = stablehlo.reshape %4701 : (tensor<1x27x4096xf32>) -> tensor<1x27x32x128xf32>
    %4705 = stablehlo.constant dense<0> : tensor<i32>
    %4706 = stablehlo.broadcast_in_dim %4705, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %4707 = stablehlo.compare  LT, %324, %4706,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %4708 = stablehlo.constant dense<4096> : tensor<i32>
    %4709 = stablehlo.broadcast_in_dim %4708, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %4710 = stablehlo.add %324, %4709 : tensor<1x27xi32>
    %4711 = stablehlo.select %4707, %4710, %324 : tensor<1x27xi1>, tensor<1x27xi32>
    %4712 = stablehlo.broadcast_in_dim %4711, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %4713 = "stablehlo.gather"(%315, %4712) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 256>} : (tensor<4096x1x256xf32>, tensor<1x27x1xi32>) -> tensor<1x27x1x256xf32>
    %4714 = stablehlo.slice %4713 [0:1, 0:27, 0:1, 0:128] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %4715 = stablehlo.slice %4713 [0:1, 0:27, 0:1, 128:256] : (tensor<1x27x1x256xf32>) -> tensor<1x27x1x128xf32>
    %4716 = stablehlo.broadcast_in_dim %4715, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4717 = stablehlo.multiply %4703, %4716 : tensor<1x27x32x128xf32>
    %4718 = stablehlo.constant dense<64> : tensor<i32>
    %4719 = stablehlo.broadcast_in_dim %4718, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4720 = "stablehlo.gather"(%4703, %4719) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4721 = stablehlo.negate %4720 : tensor<1x27x32x64xf32>
    %4722 = stablehlo.constant dense<0> : tensor<i32>
    %4723 = stablehlo.broadcast_in_dim %4722, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4724 = "stablehlo.gather"(%4703, %4723) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4725 = stablehlo.concatenate %4721, %4724, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %4726 = stablehlo.broadcast_in_dim %4714, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4727 = stablehlo.multiply %4725, %4726 : tensor<1x27x32x128xf32>
    %4728 = stablehlo.add %4717, %4727 : tensor<1x27x32x128xf32>
    %4729 = stablehlo.broadcast_in_dim %4715, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4730 = stablehlo.multiply %4702, %4729 : tensor<1x27x32x128xf32>
    %4731 = stablehlo.constant dense<64> : tensor<i32>
    %4732 = stablehlo.broadcast_in_dim %4731, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4733 = "stablehlo.gather"(%4702, %4732) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4734 = stablehlo.negate %4733 : tensor<1x27x32x64xf32>
    %4735 = stablehlo.constant dense<0> : tensor<i32>
    %4736 = stablehlo.broadcast_in_dim %4735, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4737 = "stablehlo.gather"(%4702, %4736) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 27, 32, 64>} : (tensor<1x27x32x128xf32>, tensor<1xi32>) -> tensor<1x27x32x64xf32>
    %4738 = stablehlo.concatenate %4734, %4737, dim = 3 : (tensor<1x27x32x64xf32>, tensor<1x27x32x64xf32>) -> tensor<1x27x32x128xf32>
    %4739 = stablehlo.broadcast_in_dim %4714, dims = [0, 1, 2, 3] : (tensor<1x27x1x128xf32>) -> tensor<1x27x32x128xf32>
    %4740 = stablehlo.multiply %4738, %4739 : tensor<1x27x32x128xf32>
    %4741 = stablehlo.add %4730, %4740 : tensor<1x27x32x128xf32>
    %4742 = stablehlo.slice %4695 [0:1, 0:1, 0:27, 0:27] : (tensor<1x1x4096x4096xi1>) -> tensor<1x1x27x27xi1>
    %4743 = stablehlo.broadcast_in_dim %328, dims = [0, 3] : (tensor<1x27xi32>) -> tensor<1x1x1x27xi32>
    %4744 = stablehlo.reshape %4743 : (tensor<1x1x1x27xi32>) -> tensor<1x1x27xi32>
    %4745 = stablehlo.broadcast_in_dim %4744, dims = [0, 1, 3] : (tensor<1x1x27xi32>) -> tensor<1x1x27x27xi32>
    %4746 = stablehlo.constant dense<0> : tensor<i32>
    %4747 = stablehlo.broadcast_in_dim %4746, dims = [] : (tensor<i32>) -> tensor<1x1x27x27xi32>
    %4748 = stablehlo.compare  NE, %4745, %4747,  SIGNED : (tensor<1x1x27x27xi32>, tensor<1x1x27x27xi32>) -> tensor<1x1x27x27xi1>
    %4749 = stablehlo.and %4748, %4742 : tensor<1x1x27x27xi1>
    %4750 = stablehlo.convert %4749 : (tensor<1x1x27x27xi1>) -> tensor<1x1x27x27xf32>
    %4751 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4752 = stablehlo.broadcast_in_dim %4751, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4753 = stablehlo.compare  GT, %4750, %4752,  FLOAT : (tensor<1x1x27x27xf32>, tensor<1x1x27x27xf32>) -> tensor<1x1x27x27xi1>
    %4754 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4755 = stablehlo.broadcast_in_dim %4754, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4756 = stablehlo.convert %4755 : tensor<1x1x27x27xf32>
    %4757 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %4758 = stablehlo.broadcast_in_dim %4757, dims = [] : (tensor<f32>) -> tensor<1x1x27x27xf32>
    %4759 = stablehlo.select %4753, %4756, %4758 : tensor<1x1x27x27xi1>, tensor<1x1x27x27xf32>
    %4760 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %4761 = stablehlo.sqrt %4760 : tensor<f32>
    %4762 = stablehlo.convert %4761 : tensor<f32>
    %4763 = stablehlo.broadcast_in_dim %4762, dims = [] : (tensor<f32>) -> tensor<1x27x32x128xf32>
    %4764 = stablehlo.divide %4741, %4763 : tensor<1x27x32x128xf32>
    %4765 = stablehlo.dot_general %4764, %4728, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x27x32x128xf32>, tensor<1x27x32x128xf32>) -> tensor<1x32x27x27xf32>
    %4766 = stablehlo.broadcast_in_dim %4759, dims = [0, 1, 2, 3] : (tensor<1x1x27x27xf32>) -> tensor<1x32x27x27xf32>
    %4767 = stablehlo.add %4765, %4766 : tensor<1x32x27x27xf32>
    %4768 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4769 = stablehlo.reduce(%4767 init: %4768) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %4770 = stablehlo.broadcast_in_dim %4769, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %4771 = stablehlo.broadcast_in_dim %4770, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %4772 = stablehlo.subtract %4767, %4771 : tensor<1x32x27x27xf32>
    %4773 = stablehlo.exponential %4772 : tensor<1x32x27x27xf32>
    %4774 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4775 = stablehlo.reduce(%4773 init: %4774) applies stablehlo.add across dimensions = [3] : (tensor<1x32x27x27xf32>, tensor<f32>) -> tensor<1x32x27xf32>
    %4776 = stablehlo.broadcast_in_dim %4775, dims = [0, 1, 2] : (tensor<1x32x27xf32>) -> tensor<1x32x27x1xf32>
    %4777 = stablehlo.broadcast_in_dim %4776, dims = [0, 1, 2, 3] : (tensor<1x32x27x1xf32>) -> tensor<1x32x27x27xf32>
    %4778 = stablehlo.divide %4773, %4777 : tensor<1x32x27x27xf32>
    %4779 = stablehlo.dot_general %4704, %4778, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<1x27x32x128xf32>, tensor<1x32x27x27xf32>) -> tensor<1x32x128x27xf32>
    %4780 = stablehlo.transpose %4779, dims = [0, 3, 1, 2] : (tensor<1x32x128x27xf32>) -> tensor<1x27x32x128xf32>
    %4781 = stablehlo.reshape %4780 : (tensor<1x27x32x128xf32>) -> tensor<1x27x4096xf32>
    %4782 = stablehlo.convert %316 : (tensor<4096x4096xf16>) -> tensor<4096x4096xf32>
    %4783 = stablehlo.dot_general %4781, %4782, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x27x4096xf32>
    %4784 = stablehlo.add %4670, %4783 : tensor<1x27x4096xf32>
    %4785 = stablehlo.multiply %4784, %4784 : tensor<1x27x4096xf32>
    %4786 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4787 = stablehlo.reduce(%4785 init: %4786) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %4788 = stablehlo.broadcast_in_dim %4787, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %4789 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4790 = stablehlo.broadcast_in_dim %4789, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4791 = stablehlo.divide %4788, %4790 : tensor<1x27x1xf32>
    %4792 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4793 = stablehlo.broadcast_in_dim %4792, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4794 = stablehlo.add %4791, %4793 : tensor<1x27x1xf32>
    %4795 = stablehlo.sqrt %4794 : tensor<1x27x1xf32>
    %4796 = stablehlo.broadcast_in_dim %4795, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %4797 = stablehlo.divide %4784, %4796 : tensor<1x27x4096xf32>
    %4798 = stablehlo.convert %317 : (tensor<4096xf16>) -> tensor<4096xf32>
    %4799 = stablehlo.broadcast_in_dim %4798, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4800 = stablehlo.broadcast_in_dim %4799, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %4801 = stablehlo.multiply %4800, %4797 : tensor<1x27x4096xf32>
    %4802 = stablehlo.convert %318 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %4803 = stablehlo.dot_general %4801, %4802, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %4804 = stablehlo.convert %319 : (tensor<4096x11008xf16>) -> tensor<4096x11008xf32>
    %4805 = stablehlo.dot_general %4801, %4804, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x27x11008xf32>
    %4806 = call @silu(%4805) : (tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32>
    %4807 = stablehlo.multiply %4803, %4806 : tensor<1x27x11008xf32>
    %4808 = stablehlo.convert %320 : (tensor<11008x4096xf16>) -> tensor<11008x4096xf32>
    %4809 = stablehlo.dot_general %4807, %4808, contracting_dims = [2] x [0] : (tensor<1x27x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x27x4096xf32>
    %4810 = stablehlo.add %4784, %4809 : tensor<1x27x4096xf32>
    %4811 = stablehlo.multiply %4810, %4810 : tensor<1x27x4096xf32>
    %4812 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4813 = stablehlo.reduce(%4811 init: %4812) applies stablehlo.add across dimensions = [2] : (tensor<1x27x4096xf32>, tensor<f32>) -> tensor<1x27xf32>
    %4814 = stablehlo.broadcast_in_dim %4813, dims = [0, 1] : (tensor<1x27xf32>) -> tensor<1x27x1xf32>
    %4815 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4816 = stablehlo.broadcast_in_dim %4815, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4817 = stablehlo.divide %4814, %4816 : tensor<1x27x1xf32>
    %4818 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4819 = stablehlo.broadcast_in_dim %4818, dims = [] : (tensor<f32>) -> tensor<1x27x1xf32>
    %4820 = stablehlo.add %4817, %4819 : tensor<1x27x1xf32>
    %4821 = stablehlo.sqrt %4820 : tensor<1x27x1xf32>
    %4822 = stablehlo.broadcast_in_dim %4821, dims = [0, 1, 2] : (tensor<1x27x1xf32>) -> tensor<1x27x4096xf32>
    %4823 = stablehlo.divide %4810, %4822 : tensor<1x27x4096xf32>
    %4824 = stablehlo.convert %321 : (tensor<4096xf16>) -> tensor<4096xf32>
    %4825 = stablehlo.broadcast_in_dim %4824, dims = [2] : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4826 = stablehlo.broadcast_in_dim %4825, dims = [0, 1, 2] : (tensor<1x1x4096xf32>) -> tensor<1x27x4096xf32>
    %4827 = stablehlo.multiply %4826, %4823 : tensor<1x27x4096xf32>
    %4828 = stablehlo.convert %322 : (tensor<4096x32000xf16>) -> tensor<4096x32000xf32>
    %4829 = stablehlo.dot_general %4827, %4828, contracting_dims = [2] x [0] : (tensor<1x27x4096xf32>, tensor<4096x32000xf32>) -> tensor<1x27x32000xf32>
    return %4829 : tensor<1x27x32000xf32>
  }
  func.func private @_take(%arg0: tensor<32000x4096xf32>, %arg1: tensor<1x27xi32>) -> tensor<1x27x4096xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi1>
    %3 = stablehlo.constant dense<32000> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<1x27xi32>
    %5 = stablehlo.add %arg1, %4 : tensor<1x27xi32>
    %6 = call @_where(%2, %5, %arg1) : (tensor<1x27xi1>, tensor<1x27xi32>, tensor<1x27xi32>) -> tensor<1x27xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x27xi32>) -> tensor<1x27x1xi32>
    %8 = stablehlo.constant dense<0> : tensor<1xi32>
    %9 = stablehlo.constant dense<0> : tensor<1xi32>
    %10 = stablehlo.constant dense<32000> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.constant dense<4096> : tensor<i32>
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
    %26 = stablehlo.constant dense<4096> : tensor<i32>
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
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1x27x1xi32>
    %41 = stablehlo.compare  GE, %7, %40,  SIGNED : (tensor<1x27x1xi32>, tensor<1x27x1xi32>) -> tensor<1x27x1xi1>
    %42 = stablehlo.broadcast_in_dim %38, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x27x1xi32>
    %44 = stablehlo.compare  LE, %7, %43,  SIGNED : (tensor<1x27x1xi32>, tensor<1x27x1xi32>) -> tensor<1x27x1xi1>
    %45 = stablehlo.and %41, %44 : tensor<1x27x1xi1>
    %46 = stablehlo.constant dense<true> : tensor<i1>
    %47 = stablehlo.reduce(%45 init: %46) applies stablehlo.and across dimensions = [2] : (tensor<1x27x1xi1>, tensor<i1>) -> tensor<1x27xi1>
    %48 = "stablehlo.gather"(%arg0, %7) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4096>} : (tensor<32000x4096xf32>, tensor<1x27x1xi32>) -> tensor<1x27x4096xf32>
    %49 = stablehlo.broadcast_in_dim %47, dims = [0, 1] : (tensor<1x27xi1>) -> tensor<1x27x4096xi1>
    %50 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [] : (tensor<f32>) -> tensor<1x27x4096xf32>
    %52 = stablehlo.select %49, %48, %51 : tensor<1x27x4096xi1>, tensor<1x27x4096xf32>
    return %52 : tensor<1x27x4096xf32>
  }
  func.func private @_where(%arg0: tensor<1x27xi1>, %arg1: tensor<1x27xi32>, %arg2: tensor<1x27xi32>) -> tensor<1x27xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x27xi1>, tensor<1x27xi32>
    return %0 : tensor<1x27xi32>
  }
  func.func private @silu(%arg0: tensor<1x27x11008xf32>) -> tensor<1x27x11008xf32> {
    %0 = stablehlo.negate %arg0 : tensor<1x27x11008xf32>
    %1 = stablehlo.exponential %0 : tensor<1x27x11008xf32>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x27x11008xf32>
    %4 = stablehlo.add %3, %1 : tensor<1x27x11008xf32>
    %5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x27x11008xf32>
    %7 = stablehlo.divide %6, %4 : tensor<1x27x11008xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x27x11008xf32>
    return %8 : tensor<1x27x11008xf32>
  }
}
