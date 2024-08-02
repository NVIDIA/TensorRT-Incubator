module @jit_generate attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x9xi32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<1x9xi32> {mhlo.sharding = "{replicated}"}) -> tensor<1x20xi32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<1x1x3072xf32>
    %1 = stablehlo.constant dense<-3.40282347E+38> : tensor<1x1x1x20xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x1x20xf32>
    %3 = stablehlo.constant dense<0> : tensor<1x1x1x20xi32>
    %4 = stablehlo.constant dense<2048> : tensor<1x1xi32>
    %5 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %6 = stablehlo.constant dense<7.680000e+02> : tensor<1x1x1xf32>
    %7 = stablehlo.constant dense<0x7FC00000> : tensor<1x1x768xf32>
    %8 = stablehlo.constant dense<0> : tensor<1x1x1xi32>
    %9 = stablehlo.constant dense<32000> : tensor<1x1xi32>
    %10 = stablehlo.constant dense<0> : tensor<1x1xi32>
    %11 = stablehlo.constant dense<1> : tensor<1x1xi32>
    %12 = stablehlo.constant dense<29> : tensor<i32>
    %13 = stablehlo.constant dense<false> : tensor<1xi1>
    %14 = stablehlo.constant dense<0xFF800000> : tensor<1xf32>
    %15 = stablehlo.constant dense<8> : tensor<i32>
    %16 = stablehlo.constant dense<-1> : tensor<i32>
    %17 = stablehlo.constant dense<1.000000e+00> : tensor<1x9x3072xf32>
    %18 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %19 = stablehlo.constant dense<-3.40282347E+38> : tensor<1x1x9x20xf32>
    %20 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x9x20xf32>
    %21 = stablehlo.constant dense<0> : tensor<1x1x9x20xi32>
    %22 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %23 = stablehlo.constant dense<32> : tensor<1xi32>
    %24 = stablehlo.constant dense<2048> : tensor<1x9xi32>
    %25 = stablehlo.constant dense<2048> : tensor<i32>
    %26 = stablehlo.constant dense<1> : tensor<1x9xi32>
    %27 = stablehlo.constant dense<> : tensor<1x0xi32>
    %28 = stablehlo.constant dense<20> : tensor<i32>
    %29 = stablehlo.constant dense<false> : tensor<i1>
    %30 = stablehlo.constant dense<9.99999997E-7> : tensor<1x9x1xf32>
    %31 = stablehlo.constant dense<7.680000e+02> : tensor<1x9x1xf32>
    %32 = stablehlo.constant dense<0.000000e+00> : tensor<1x20x12x64xf32>
    %33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.constant dense<0x7FC00000> : tensor<1x9x768xf32>
    %35 = stablehlo.constant dense<true> : tensor<i1>
    %36 = stablehlo.constant dense<1> : tensor<1xi32>
    %37 = stablehlo.constant dense<2> : tensor<1xi32>
    %38 = stablehlo.constant dense<0> : tensor<1xi32>
    %39 = stablehlo.constant dense<768> : tensor<1xi32>
    %40 = stablehlo.constant dense<32000> : tensor<1xi32>
    %41 = stablehlo.constant dense<0> : tensor<1x9x1xi32>
    %42 = stablehlo.constant dense<32000> : tensor<1x9xi32>
    %43 = stablehlo.constant dense<32000> : tensor<i32>
    %44 = stablehlo.constant dense<0> : tensor<1x9xi32>
    %45 = stablehlo.constant dense_resource<__elided__> : tensor<32000x768xf32>
    %46 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %47 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %48 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %49 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %50 = stablehlo.constant dense_resource<__elided__> : tensor<2048x1x128xf32>
    %51 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %52 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %53 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %54 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %55 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %56 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %57 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %58 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %59 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %60 = stablehlo.constant dense_resource<__elided__> : tensor<768x768xf32>
    %61 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %62 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %63 = stablehlo.constant dense_resource<__elided__> : tensor<768x3072xf32>
    %64 = stablehlo.constant dense_resource<__elided__> : tensor<3072x768xf32>
    %65 = stablehlo.constant dense_resource<__elided__> : tensor<768xf32>
    %66 = stablehlo.constant dense_resource<__elided__> : tensor<768x32000xf32>
    %67 = stablehlo.constant dense<1> : tensor<i32>
    %68 = stablehlo.constant dense<2> : tensor<i32>
    %69 = stablehlo.constant dense<9> : tensor<i32>
    %70 = stablehlo.constant dense<10> : tensor<i32>
    %71 = stablehlo.constant dense<1> : tensor<1x20xi32>
    %72 = stablehlo.constant dense<0> : tensor<i32>
    %73 = stablehlo.dynamic_update_slice %71, %arg1, %72, %72 : (tensor<1x20xi32>, tensor<1x9xi32>, tensor<i32>, tensor<i32>) -> tensor<1x20xi32>
    %74 = stablehlo.maximum %69, %72 : tensor<i32>
    %75 = stablehlo.minimum %74, %67 : tensor<i32>
    %76 = stablehlo.subtract %67, %75 : tensor<i32>
    %77 = stablehlo.compare  NE, %76, %72 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %78 = stablehlo.broadcast_in_dim %77, dims = [] : (tensor<i1>) -> tensor<1x32000xi1>
    %79 = stablehlo.broadcast_in_dim %65, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %80 = stablehlo.reshape %79 : (tensor<1x1x768xf32>) -> tensor<1x768xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 2] : (tensor<1x768xf32>) -> tensor<1x9x768xf32>
    %82 = stablehlo.compare  LT, %arg1, %44 : (tensor<1x9xi32>, tensor<1x9xi32>) -> tensor<1x9xi1>
    %83 = stablehlo.add %arg1, %42 : tensor<1x9xi32>
    %84 = stablehlo.select %82, %83, %arg1 : tensor<1x9xi1>, tensor<1x9xi32>
    %85 = stablehlo.broadcast_in_dim %84, dims = [0, 1] : (tensor<1x9xi32>) -> tensor<1x9x1xi32>
    %86 = stablehlo.compare  GE, %85, %41 : (tensor<1x9x1xi32>, tensor<1x9x1xi32>) -> tensor<1x9x1xi1>
    %87 = stablehlo.concatenate %40, %39, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %88 = stablehlo.compare  LT, %38, %38 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %89 = stablehlo.select %88, %37, %38 : tensor<1xi1>, tensor<1xi32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %91 = "stablehlo.gather"(%87, %90) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %92 = stablehlo.concatenate %36, %39, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %93 = stablehlo.compare  LT, %38, %38 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %94 = stablehlo.select %93, %37, %38 : tensor<1xi1>, tensor<1xi32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %96 = "stablehlo.gather"(%92, %95) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %97 = stablehlo.subtract %91, %96 : tensor<1xi32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %99 = stablehlo.reshape %98 : (tensor<1x1x1xi32>) -> tensor<1x1xi32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [0, 2] : (tensor<1x1xi32>) -> tensor<1x9x1xi32>
    %101 = stablehlo.compare  LE, %85, %100 : (tensor<1x9x1xi32>, tensor<1x9x1xi32>) -> tensor<1x9x1xi1>
    %102 = stablehlo.and %86, %101 : tensor<1x9x1xi1>
    %103 = stablehlo.reduce(%102 init: %35) applies stablehlo.and across dimensions = [2] : (tensor<1x9x1xi1>, tensor<i1>) -> tensor<1x9xi1>
    %104 = stablehlo.broadcast_in_dim %103, dims = [0, 1] : (tensor<1x9xi1>) -> tensor<1x9x768xi1>
    %105 = "stablehlo.gather"(%45, %85) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>} : (tensor<32000x768xf32>, tensor<1x9x1xi32>) -> tensor<1x9x768xf32>
    %106 = stablehlo.select %104, %105, %34 : tensor<1x9x768xi1>, tensor<1x9x768xf32>
    %107 = stablehlo.broadcast_in_dim %46, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %108 = stablehlo.reshape %107 : (tensor<1x1x768xf32>) -> tensor<1x768xf32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [0, 2] : (tensor<1x768xf32>) -> tensor<1x9x768xf32>
    %110 = stablehlo.multiply %106, %106 : tensor<1x9x768xf32>
    %111 = stablehlo.reduce(%110 init: %33) applies stablehlo.add across dimensions = [2] : (tensor<1x9x768xf32>, tensor<f32>) -> tensor<1x9xf32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [0, 1] : (tensor<1x9xf32>) -> tensor<1x9x1xf32>
    %113 = stablehlo.divide %112, %31 : tensor<1x9x1xf32>
    %114 = stablehlo.add %113, %30 : tensor<1x9x1xf32>
    %115 = stablehlo.sqrt %114 : tensor<1x9x1xf32>
    %116 = stablehlo.reshape %115 : (tensor<1x9x1xf32>) -> tensor<1x9xf32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [0, 1] : (tensor<1x9xf32>) -> tensor<1x9x768xf32>
    %118 = stablehlo.divide %106, %117 : tensor<1x9x768xf32>
    %119 = stablehlo.multiply %109, %118 : tensor<1x9x768xf32>
    %120 = stablehlo.dot_general %119, %49, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x768xf32>) -> tensor<1x9x768xf32>
    %121 = stablehlo.reshape %120 : (tensor<1x9x768xf32>) -> tensor<1x9x12x64xf32>
    %122 = stablehlo.select %29, %28, %72 : tensor<i1>, tensor<i32>
    %123 = stablehlo.dynamic_update_slice %32, %121, %72, %122, %72, %72 : (tensor<1x20x12x64xf32>, tensor<1x9x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %124 = stablehlo.dot_general %119, %47, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x768xf32>) -> tensor<1x9x768xf32>
    %125 = stablehlo.reshape %124 : (tensor<1x9x768xf32>) -> tensor<1x9x12x64xf32>
    %126 = stablehlo.slice %arg0 [0:1, 0:1] : (tensor<1x9xi32>) -> tensor<1x1xi32>
    %127 = stablehlo.slice %arg0 [0:1, 0:8:2] : (tensor<1x9xi32>) -> tensor<1x4xi32>
    %128 = stablehlo.slice %arg0 [0:1, 1:9:2] : (tensor<1x9xi32>) -> tensor<1x4xi32>
    %129 = stablehlo.add %127, %128 : tensor<1x4xi32>
    %130 = stablehlo.slice %129 [0:1, 0:1] : (tensor<1x4xi32>) -> tensor<1x1xi32>
    %131 = stablehlo.slice %129 [0:1, 0:3:2] : (tensor<1x4xi32>) -> tensor<1x2xi32>
    %132 = stablehlo.slice %129 [0:1, 1:4:2] : (tensor<1x4xi32>) -> tensor<1x2xi32>
    %133 = stablehlo.add %131, %132 : tensor<1x2xi32>
    %134 = stablehlo.slice %133 [0:1, 0:1] : (tensor<1x2xi32>) -> tensor<1x1xi32>
    %135 = stablehlo.concatenate %134, %27, dim = 1 : (tensor<1x1xi32>, tensor<1x0xi32>) -> tensor<1x1xi32>
    %136 = stablehlo.pad %135, %72, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<1x1xi32>, tensor<i32>) -> tensor<1x2xi32>
    %137 = stablehlo.slice %133 [0:1, 0:1:2] : (tensor<1x2xi32>) -> tensor<1x1xi32>
    %138 = stablehlo.slice %133 [0:1, 1:2:2] : (tensor<1x2xi32>) -> tensor<1x1xi32>
    %139 = stablehlo.add %137, %138 : tensor<1x1xi32>
    %140 = stablehlo.pad %139, %72, low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<1x1xi32>, tensor<i32>) -> tensor<1x2xi32>
    %141 = stablehlo.add %136, %140 : tensor<1x2xi32>
    %142 = stablehlo.slice %141 [0:1, 0:1] : (tensor<1x2xi32>) -> tensor<1x1xi32>
    %143 = stablehlo.slice %129 [0:1, 2:4:2] : (tensor<1x4xi32>) -> tensor<1x1xi32>
    %144 = stablehlo.add %142, %143 : tensor<1x1xi32>
    %145 = stablehlo.concatenate %130, %144, dim = 1 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
    %146 = stablehlo.pad %145, %72, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<1x2xi32>, tensor<i32>) -> tensor<1x4xi32>
    %147 = stablehlo.pad %141, %72, low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<1x2xi32>, tensor<i32>) -> tensor<1x4xi32>
    %148 = stablehlo.add %146, %147 : tensor<1x4xi32>
    %149 = stablehlo.slice %arg0 [0:1, 2:9:2] : (tensor<1x9xi32>) -> tensor<1x4xi32>
    %150 = stablehlo.add %148, %149 : tensor<1x4xi32>
    %151 = stablehlo.concatenate %126, %150, dim = 1 : (tensor<1x1xi32>, tensor<1x4xi32>) -> tensor<1x5xi32>
    %152 = stablehlo.pad %151, %72, low = [0, 0], high = [0, 0], interior = [0, 1] : (tensor<1x5xi32>, tensor<i32>) -> tensor<1x9xi32>
    %153 = stablehlo.pad %148, %72, low = [0, 1], high = [0, 1], interior = [0, 1] : (tensor<1x4xi32>, tensor<i32>) -> tensor<1x9xi32>
    %154 = stablehlo.add %152, %153 : tensor<1x9xi32>
    %155 = stablehlo.subtract %154, %26 : tensor<1x9xi32>
    %156 = stablehlo.compare  LT, %155, %44 : (tensor<1x9xi32>, tensor<1x9xi32>) -> tensor<1x9xi1>
    %157 = stablehlo.add %155, %24 : tensor<1x9xi32>
    %158 = stablehlo.select %156, %157, %155 : tensor<1x9xi1>, tensor<1x9xi32>
    %159 = stablehlo.broadcast_in_dim %158, dims = [0, 1] : (tensor<1x9xi32>) -> tensor<1x9x1xi32>
    %160 = "stablehlo.gather"(%50, %159) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 128>} : (tensor<2048x1x128xf32>, tensor<1x9x1xi32>) -> tensor<1x9x1x128xf32>
    %161 = stablehlo.slice %160 [0:1, 0:9, 0:1, 64:128] : (tensor<1x9x1x128xf32>) -> tensor<1x9x1x64xf32>
    %162 = stablehlo.reshape %161 : (tensor<1x9x1x64xf32>) -> tensor<1x9x64xf32>
    %163 = stablehlo.broadcast_in_dim %162, dims = [0, 1, 3] : (tensor<1x9x64xf32>) -> tensor<1x9x12x64xf32>
    %164 = stablehlo.multiply %125, %163 : tensor<1x9x12x64xf32>
    %165 = "stablehlo.gather"(%125, %23) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 9, 12, 32>} : (tensor<1x9x12x64xf32>, tensor<1xi32>) -> tensor<1x9x12x32xf32>
    %166 = stablehlo.negate %165 : tensor<1x9x12x32xf32>
    %167 = "stablehlo.gather"(%125, %38) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 9, 12, 32>} : (tensor<1x9x12x64xf32>, tensor<1xi32>) -> tensor<1x9x12x32xf32>
    %168 = stablehlo.concatenate %166, %167, dim = 3 : (tensor<1x9x12x32xf32>, tensor<1x9x12x32xf32>) -> tensor<1x9x12x64xf32>
    %169 = stablehlo.slice %160 [0:1, 0:9, 0:1, 0:64] : (tensor<1x9x1x128xf32>) -> tensor<1x9x1x64xf32>
    %170 = stablehlo.reshape %169 : (tensor<1x9x1x64xf32>) -> tensor<1x9x64xf32>
    %171 = stablehlo.broadcast_in_dim %170, dims = [0, 1, 3] : (tensor<1x9x64xf32>) -> tensor<1x9x12x64xf32>
    %172 = stablehlo.multiply %168, %171 : tensor<1x9x12x64xf32>
    %173 = stablehlo.add %164, %172 : tensor<1x9x12x64xf32>
    %174 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<1x9x12x64xf32>
    %175 = stablehlo.divide %173, %174 : tensor<1x9x12x64xf32>
    %176 = stablehlo.dot_general %119, %48, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x768xf32>) -> tensor<1x9x768xf32>
    %177 = stablehlo.reshape %176 : (tensor<1x9x768xf32>) -> tensor<1x9x12x64xf32>
    %178 = stablehlo.reshape %161 : (tensor<1x9x1x64xf32>) -> tensor<1x9x64xf32>
    %179 = stablehlo.broadcast_in_dim %178, dims = [0, 1, 3] : (tensor<1x9x64xf32>) -> tensor<1x9x12x64xf32>
    %180 = stablehlo.multiply %177, %179 : tensor<1x9x12x64xf32>
    %181 = "stablehlo.gather"(%177, %23) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 9, 12, 32>} : (tensor<1x9x12x64xf32>, tensor<1xi32>) -> tensor<1x9x12x32xf32>
    %182 = stablehlo.negate %181 : tensor<1x9x12x32xf32>
    %183 = "stablehlo.gather"(%177, %38) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 9, 12, 32>} : (tensor<1x9x12x64xf32>, tensor<1xi32>) -> tensor<1x9x12x32xf32>
    %184 = stablehlo.concatenate %182, %183, dim = 3 : (tensor<1x9x12x32xf32>, tensor<1x9x12x32xf32>) -> tensor<1x9x12x64xf32>
    %185 = stablehlo.reshape %169 : (tensor<1x9x1x64xf32>) -> tensor<1x9x64xf32>
    %186 = stablehlo.broadcast_in_dim %185, dims = [0, 1, 3] : (tensor<1x9x64xf32>) -> tensor<1x9x12x64xf32>
    %187 = stablehlo.multiply %184, %186 : tensor<1x9x12x64xf32>
    %188 = stablehlo.add %180, %187 : tensor<1x9x12x64xf32>
    %189 = stablehlo.select %29, %28, %72 : tensor<i1>, tensor<i32>
    %190 = stablehlo.dynamic_update_slice %32, %188, %72, %189, %72, %72 : (tensor<1x20x12x64xf32>, tensor<1x9x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %191 = stablehlo.dot_general %175, %190, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x9x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x9x20xf32>
    %192 = stablehlo.iota dim = 0 : tensor<20xi32>
    %193 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %194 = stablehlo.compare  LT, %192, %193 : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %195 = stablehlo.broadcast_in_dim %194, dims = [3] : (tensor<20xi1>) -> tensor<1x1x9x20xi1>
    %196 = stablehlo.dynamic_update_slice %71, %arg0, %72, %72 : (tensor<1x20xi32>, tensor<1x9xi32>, tensor<i32>, tensor<i32>) -> tensor<1x20xi32>
    %197 = stablehlo.broadcast_in_dim %196, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %198 = stablehlo.reshape %197 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %199 = stablehlo.broadcast_in_dim %198, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x9x20xi32>
    %200 = stablehlo.compare  NE, %199, %21 : (tensor<1x1x9x20xi32>, tensor<1x1x9x20xi32>) -> tensor<1x1x9x20xi1>
    %201 = stablehlo.iota dim = 0 : tensor<2048xi32>
    %202 = stablehlo.broadcast_in_dim %201, dims = [1] : (tensor<2048xi32>) -> tensor<1x2048xi32>
    %203 = stablehlo.broadcast_in_dim %202, dims = [0, 1] : (tensor<1x2048xi32>) -> tensor<1x2048x1xi32>
    %204 = stablehlo.reshape %203 : (tensor<1x2048x1xi32>) -> tensor<1x2048xi32>
    %205 = stablehlo.broadcast_in_dim %204, dims = [0, 1] : (tensor<1x2048xi32>) -> tensor<1x2048x2048xi32>
    %206 = stablehlo.broadcast_in_dim %202, dims = [0, 2] : (tensor<1x2048xi32>) -> tensor<1x1x2048xi32>
    %207 = stablehlo.reshape %206 : (tensor<1x1x2048xi32>) -> tensor<1x2048xi32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0, 2] : (tensor<1x2048xi32>) -> tensor<1x2048x2048xi32>
    %209 = stablehlo.compare  GE, %205, %208 : (tensor<1x2048x2048xi32>, tensor<1x2048x2048xi32>) -> tensor<1x2048x2048xi1>
    %210 = stablehlo.broadcast_in_dim %209, dims = [0, 2, 3] : (tensor<1x2048x2048xi1>) -> tensor<1x1x2048x2048xi1>
    %211 = stablehlo.select %29, %25, %72 : tensor<i1>, tensor<i32>
    %212 = stablehlo.dynamic_slice %210, %72, %72, %211, %72, sizes = [1, 1, 9, 20] : (tensor<1x1x2048x2048xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x9x20xi1>
    %213 = stablehlo.and %200, %212 : tensor<1x1x9x20xi1>
    %214 = stablehlo.convert %213 : (tensor<1x1x9x20xi1>) -> tensor<1x1x9x20xf32>
    %215 = stablehlo.compare  NE, %214, %20 : (tensor<1x1x9x20xf32>, tensor<1x1x9x20xf32>) -> tensor<1x1x9x20xi1>
    %216 = stablehlo.and %195, %215 : tensor<1x1x9x20xi1>
    %217 = stablehlo.convert %216 : (tensor<1x1x9x20xi1>) -> tensor<1x1x9x20xf32>
    %218 = stablehlo.compare  GT, %217, %20 : (tensor<1x1x9x20xf32>, tensor<1x1x9x20xf32>) -> tensor<1x1x9x20xi1>
    %219 = stablehlo.select %218, %20, %19 : tensor<1x1x9x20xi1>, tensor<1x1x9x20xf32>
    %220 = stablehlo.reshape %219 : (tensor<1x1x9x20xf32>) -> tensor<1x9x20xf32>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0, 2, 3] : (tensor<1x9x20xf32>) -> tensor<1x12x9x20xf32>
    %222 = stablehlo.add %191, %221 : tensor<1x12x9x20xf32>
    %223 = stablehlo.reduce(%222 init: %18) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x9x20xf32>, tensor<f32>) -> tensor<1x12x9xf32>
    %224 = stablehlo.broadcast_in_dim %223, dims = [0, 1, 2] : (tensor<1x12x9xf32>) -> tensor<1x12x9x1xf32>
    %225 = stablehlo.reshape %224 : (tensor<1x12x9x1xf32>) -> tensor<1x12x9xf32>
    %226 = stablehlo.broadcast_in_dim %225, dims = [0, 1, 2] : (tensor<1x12x9xf32>) -> tensor<1x12x9x20xf32>
    %227 = stablehlo.subtract %222, %226 : tensor<1x12x9x20xf32>
    %228 = stablehlo.exponential %227 : tensor<1x12x9x20xf32>
    %229 = stablehlo.reduce(%228 init: %33) applies stablehlo.add across dimensions = [3] : (tensor<1x12x9x20xf32>, tensor<f32>) -> tensor<1x12x9xf32>
    %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2] : (tensor<1x12x9xf32>) -> tensor<1x12x9x1xf32>
    %231 = stablehlo.reshape %230 : (tensor<1x12x9x1xf32>) -> tensor<1x12x9xf32>
    %232 = stablehlo.broadcast_in_dim %231, dims = [0, 1, 2] : (tensor<1x12x9xf32>) -> tensor<1x12x9x20xf32>
    %233 = stablehlo.divide %228, %232 : tensor<1x12x9x20xf32>
    %234 = stablehlo.dot_general %123, %233, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x9x20xf32>) -> tensor<1x12x64x9xf32>
    %235 = stablehlo.transpose %234, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,9,12,64]{1,3,2,0}"} : (tensor<1x12x64x9xf32>) -> tensor<1x9x12x64xf32>
    %236 = stablehlo.reshape %235 : (tensor<1x9x12x64xf32>) -> tensor<1x9x768xf32>
    %237 = stablehlo.dot_general %236, %51, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x768xf32>) -> tensor<1x9x768xf32>
    %238 = stablehlo.add %106, %237 : tensor<1x9x768xf32>
    %239 = stablehlo.broadcast_in_dim %52, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %240 = stablehlo.reshape %239 : (tensor<1x1x768xf32>) -> tensor<1x768xf32>
    %241 = stablehlo.broadcast_in_dim %240, dims = [0, 2] : (tensor<1x768xf32>) -> tensor<1x9x768xf32>
    %242 = stablehlo.multiply %238, %238 : tensor<1x9x768xf32>
    %243 = stablehlo.reduce(%242 init: %33) applies stablehlo.add across dimensions = [2] : (tensor<1x9x768xf32>, tensor<f32>) -> tensor<1x9xf32>
    %244 = stablehlo.broadcast_in_dim %243, dims = [0, 1] : (tensor<1x9xf32>) -> tensor<1x9x1xf32>
    %245 = stablehlo.divide %244, %31 : tensor<1x9x1xf32>
    %246 = stablehlo.add %245, %30 : tensor<1x9x1xf32>
    %247 = stablehlo.sqrt %246 : tensor<1x9x1xf32>
    %248 = stablehlo.reshape %247 : (tensor<1x9x1xf32>) -> tensor<1x9xf32>
    %249 = stablehlo.broadcast_in_dim %248, dims = [0, 1] : (tensor<1x9xf32>) -> tensor<1x9x768xf32>
    %250 = stablehlo.divide %238, %249 : tensor<1x9x768xf32>
    %251 = stablehlo.multiply %241, %250 : tensor<1x9x768xf32>
    %252 = stablehlo.dot_general %251, %53, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x3072xf32>) -> tensor<1x9x3072xf32>
    %253 = stablehlo.dot_general %251, %54, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x3072xf32>) -> tensor<1x9x3072xf32>
    %254 = stablehlo.negate %253 : tensor<1x9x3072xf32>
    %255 = stablehlo.exponential %254 : tensor<1x9x3072xf32>
    %256 = stablehlo.add %17, %255 : tensor<1x9x3072xf32>
    %257 = stablehlo.divide %17, %256 : tensor<1x9x3072xf32>
    %258 = stablehlo.multiply %253, %257 : tensor<1x9x3072xf32>
    %259 = stablehlo.multiply %252, %258 : tensor<1x9x3072xf32>
    %260 = stablehlo.dot_general %259, %55, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x3072xf32>, tensor<3072x768xf32>) -> tensor<1x9x768xf32>
    %261 = stablehlo.add %238, %260 : tensor<1x9x768xf32>
    %262 = stablehlo.broadcast_in_dim %56, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %263 = stablehlo.reshape %262 : (tensor<1x1x768xf32>) -> tensor<1x768xf32>
    %264 = stablehlo.broadcast_in_dim %263, dims = [0, 2] : (tensor<1x768xf32>) -> tensor<1x9x768xf32>
    %265 = stablehlo.multiply %261, %261 : tensor<1x9x768xf32>
    %266 = stablehlo.reduce(%265 init: %33) applies stablehlo.add across dimensions = [2] : (tensor<1x9x768xf32>, tensor<f32>) -> tensor<1x9xf32>
    %267 = stablehlo.broadcast_in_dim %266, dims = [0, 1] : (tensor<1x9xf32>) -> tensor<1x9x1xf32>
    %268 = stablehlo.divide %267, %31 : tensor<1x9x1xf32>
    %269 = stablehlo.add %268, %30 : tensor<1x9x1xf32>
    %270 = stablehlo.sqrt %269 : tensor<1x9x1xf32>
    %271 = stablehlo.reshape %270 : (tensor<1x9x1xf32>) -> tensor<1x9xf32>
    %272 = stablehlo.broadcast_in_dim %271, dims = [0, 1] : (tensor<1x9xf32>) -> tensor<1x9x768xf32>
    %273 = stablehlo.divide %261, %272 : tensor<1x9x768xf32>
    %274 = stablehlo.multiply %264, %273 : tensor<1x9x768xf32>
    %275 = stablehlo.dot_general %274, %59, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x768xf32>) -> tensor<1x9x768xf32>
    %276 = stablehlo.reshape %275 : (tensor<1x9x768xf32>) -> tensor<1x9x12x64xf32>
    %277 = stablehlo.select %29, %28, %72 : tensor<i1>, tensor<i32>
    %278 = stablehlo.dynamic_update_slice %32, %276, %72, %277, %72, %72 : (tensor<1x20x12x64xf32>, tensor<1x9x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %279 = stablehlo.dot_general %274, %57, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x768xf32>) -> tensor<1x9x768xf32>
    %280 = stablehlo.reshape %279 : (tensor<1x9x768xf32>) -> tensor<1x9x12x64xf32>
    %281 = stablehlo.compare  LT, %155, %44 : (tensor<1x9xi32>, tensor<1x9xi32>) -> tensor<1x9xi1>
    %282 = stablehlo.add %155, %24 : tensor<1x9xi32>
    %283 = stablehlo.select %281, %282, %155 : tensor<1x9xi1>, tensor<1x9xi32>
    %284 = stablehlo.broadcast_in_dim %283, dims = [0, 1] : (tensor<1x9xi32>) -> tensor<1x9x1xi32>
    %285 = "stablehlo.gather"(%50, %284) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 128>} : (tensor<2048x1x128xf32>, tensor<1x9x1xi32>) -> tensor<1x9x1x128xf32>
    %286 = stablehlo.slice %285 [0:1, 0:9, 0:1, 64:128] : (tensor<1x9x1x128xf32>) -> tensor<1x9x1x64xf32>
    %287 = stablehlo.reshape %286 : (tensor<1x9x1x64xf32>) -> tensor<1x9x64xf32>
    %288 = stablehlo.broadcast_in_dim %287, dims = [0, 1, 3] : (tensor<1x9x64xf32>) -> tensor<1x9x12x64xf32>
    %289 = stablehlo.multiply %280, %288 : tensor<1x9x12x64xf32>
    %290 = "stablehlo.gather"(%280, %23) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 9, 12, 32>} : (tensor<1x9x12x64xf32>, tensor<1xi32>) -> tensor<1x9x12x32xf32>
    %291 = stablehlo.negate %290 : tensor<1x9x12x32xf32>
    %292 = "stablehlo.gather"(%280, %38) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 9, 12, 32>} : (tensor<1x9x12x64xf32>, tensor<1xi32>) -> tensor<1x9x12x32xf32>
    %293 = stablehlo.concatenate %291, %292, dim = 3 : (tensor<1x9x12x32xf32>, tensor<1x9x12x32xf32>) -> tensor<1x9x12x64xf32>
    %294 = stablehlo.slice %285 [0:1, 0:9, 0:1, 0:64] : (tensor<1x9x1x128xf32>) -> tensor<1x9x1x64xf32>
    %295 = stablehlo.reshape %294 : (tensor<1x9x1x64xf32>) -> tensor<1x9x64xf32>
    %296 = stablehlo.broadcast_in_dim %295, dims = [0, 1, 3] : (tensor<1x9x64xf32>) -> tensor<1x9x12x64xf32>
    %297 = stablehlo.multiply %293, %296 : tensor<1x9x12x64xf32>
    %298 = stablehlo.add %289, %297 : tensor<1x9x12x64xf32>
    %299 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<1x9x12x64xf32>
    %300 = stablehlo.divide %298, %299 : tensor<1x9x12x64xf32>
    %301 = stablehlo.dot_general %274, %58, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x768xf32>) -> tensor<1x9x768xf32>
    %302 = stablehlo.reshape %301 : (tensor<1x9x768xf32>) -> tensor<1x9x12x64xf32>
    %303 = stablehlo.reshape %286 : (tensor<1x9x1x64xf32>) -> tensor<1x9x64xf32>
    %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 3] : (tensor<1x9x64xf32>) -> tensor<1x9x12x64xf32>
    %305 = stablehlo.multiply %302, %304 : tensor<1x9x12x64xf32>
    %306 = "stablehlo.gather"(%302, %23) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 9, 12, 32>} : (tensor<1x9x12x64xf32>, tensor<1xi32>) -> tensor<1x9x12x32xf32>
    %307 = stablehlo.negate %306 : tensor<1x9x12x32xf32>
    %308 = "stablehlo.gather"(%302, %38) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 9, 12, 32>} : (tensor<1x9x12x64xf32>, tensor<1xi32>) -> tensor<1x9x12x32xf32>
    %309 = stablehlo.concatenate %307, %308, dim = 3 : (tensor<1x9x12x32xf32>, tensor<1x9x12x32xf32>) -> tensor<1x9x12x64xf32>
    %310 = stablehlo.reshape %294 : (tensor<1x9x1x64xf32>) -> tensor<1x9x64xf32>
    %311 = stablehlo.broadcast_in_dim %310, dims = [0, 1, 3] : (tensor<1x9x64xf32>) -> tensor<1x9x12x64xf32>
    %312 = stablehlo.multiply %309, %311 : tensor<1x9x12x64xf32>
    %313 = stablehlo.add %305, %312 : tensor<1x9x12x64xf32>
    %314 = stablehlo.select %29, %28, %72 : tensor<i1>, tensor<i32>
    %315 = stablehlo.dynamic_update_slice %32, %313, %72, %314, %72, %72 : (tensor<1x20x12x64xf32>, tensor<1x9x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
    %316 = stablehlo.dot_general %300, %315, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x9x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x9x20xf32>
    %317 = stablehlo.iota dim = 0 : tensor<20xi32>
    %318 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<i32>) -> tensor<20xi32>
    %319 = stablehlo.compare  LT, %317, %318 : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %320 = stablehlo.broadcast_in_dim %319, dims = [3] : (tensor<20xi1>) -> tensor<1x1x9x20xi1>
    %321 = stablehlo.broadcast_in_dim %196, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
    %322 = stablehlo.reshape %321 : (tensor<1x1x1x20xi32>) -> tensor<1x1x20xi32>
    %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 3] : (tensor<1x1x20xi32>) -> tensor<1x1x9x20xi32>
    %324 = stablehlo.compare  NE, %323, %21 : (tensor<1x1x9x20xi32>, tensor<1x1x9x20xi32>) -> tensor<1x1x9x20xi1>
    %325 = stablehlo.iota dim = 0 : tensor<2048xi32>
    %326 = stablehlo.broadcast_in_dim %325, dims = [1] : (tensor<2048xi32>) -> tensor<1x2048xi32>
    %327 = stablehlo.broadcast_in_dim %326, dims = [0, 1] : (tensor<1x2048xi32>) -> tensor<1x2048x1xi32>
    %328 = stablehlo.reshape %327 : (tensor<1x2048x1xi32>) -> tensor<1x2048xi32>
    %329 = stablehlo.broadcast_in_dim %328, dims = [0, 1] : (tensor<1x2048xi32>) -> tensor<1x2048x2048xi32>
    %330 = stablehlo.broadcast_in_dim %326, dims = [0, 2] : (tensor<1x2048xi32>) -> tensor<1x1x2048xi32>
    %331 = stablehlo.reshape %330 : (tensor<1x1x2048xi32>) -> tensor<1x2048xi32>
    %332 = stablehlo.broadcast_in_dim %331, dims = [0, 2] : (tensor<1x2048xi32>) -> tensor<1x2048x2048xi32>
    %333 = stablehlo.compare  GE, %329, %332 : (tensor<1x2048x2048xi32>, tensor<1x2048x2048xi32>) -> tensor<1x2048x2048xi1>
    %334 = stablehlo.broadcast_in_dim %333, dims = [0, 2, 3] : (tensor<1x2048x2048xi1>) -> tensor<1x1x2048x2048xi1>
    %335 = stablehlo.select %29, %25, %72 : tensor<i1>, tensor<i32>
    %336 = stablehlo.dynamic_slice %334, %72, %72, %335, %72, sizes = [1, 1, 9, 20] : (tensor<1x1x2048x2048xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x9x20xi1>
    %337 = stablehlo.and %324, %336 : tensor<1x1x9x20xi1>
    %338 = stablehlo.convert %337 : (tensor<1x1x9x20xi1>) -> tensor<1x1x9x20xf32>
    %339 = stablehlo.compare  NE, %338, %20 : (tensor<1x1x9x20xf32>, tensor<1x1x9x20xf32>) -> tensor<1x1x9x20xi1>
    %340 = stablehlo.and %320, %339 : tensor<1x1x9x20xi1>
    %341 = stablehlo.convert %340 : (tensor<1x1x9x20xi1>) -> tensor<1x1x9x20xf32>
    %342 = stablehlo.compare  GT, %341, %20 : (tensor<1x1x9x20xf32>, tensor<1x1x9x20xf32>) -> tensor<1x1x9x20xi1>
    %343 = stablehlo.select %342, %20, %19 : tensor<1x1x9x20xi1>, tensor<1x1x9x20xf32>
    %344 = stablehlo.reshape %343 : (tensor<1x1x9x20xf32>) -> tensor<1x9x20xf32>
    %345 = stablehlo.broadcast_in_dim %344, dims = [0, 2, 3] : (tensor<1x9x20xf32>) -> tensor<1x12x9x20xf32>
    %346 = stablehlo.add %316, %345 : tensor<1x12x9x20xf32>
    %347 = stablehlo.reduce(%346 init: %18) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x9x20xf32>, tensor<f32>) -> tensor<1x12x9xf32>
    %348 = stablehlo.broadcast_in_dim %347, dims = [0, 1, 2] : (tensor<1x12x9xf32>) -> tensor<1x12x9x1xf32>
    %349 = stablehlo.reshape %348 : (tensor<1x12x9x1xf32>) -> tensor<1x12x9xf32>
    %350 = stablehlo.broadcast_in_dim %349, dims = [0, 1, 2] : (tensor<1x12x9xf32>) -> tensor<1x12x9x20xf32>
    %351 = stablehlo.subtract %346, %350 : tensor<1x12x9x20xf32>
    %352 = stablehlo.exponential %351 : tensor<1x12x9x20xf32>
    %353 = stablehlo.reduce(%352 init: %33) applies stablehlo.add across dimensions = [3] : (tensor<1x12x9x20xf32>, tensor<f32>) -> tensor<1x12x9xf32>
    %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<1x12x9xf32>) -> tensor<1x12x9x1xf32>
    %355 = stablehlo.reshape %354 : (tensor<1x12x9x1xf32>) -> tensor<1x12x9xf32>
    %356 = stablehlo.broadcast_in_dim %355, dims = [0, 1, 2] : (tensor<1x12x9xf32>) -> tensor<1x12x9x20xf32>
    %357 = stablehlo.divide %352, %356 : tensor<1x12x9x20xf32>
    %358 = stablehlo.dot_general %278, %357, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x9x20xf32>) -> tensor<1x12x64x9xf32>
    %359 = stablehlo.transpose %358, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,9,12,64]{1,3,2,0}"} : (tensor<1x12x64x9xf32>) -> tensor<1x9x12x64xf32>
    %360 = stablehlo.reshape %359 : (tensor<1x9x12x64xf32>) -> tensor<1x9x768xf32>
    %361 = stablehlo.dot_general %360, %60, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x768xf32>) -> tensor<1x9x768xf32>
    %362 = stablehlo.add %261, %361 : tensor<1x9x768xf32>
    %363 = stablehlo.broadcast_in_dim %61, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %364 = stablehlo.reshape %363 : (tensor<1x1x768xf32>) -> tensor<1x768xf32>
    %365 = stablehlo.broadcast_in_dim %364, dims = [0, 2] : (tensor<1x768xf32>) -> tensor<1x9x768xf32>
    %366 = stablehlo.multiply %362, %362 : tensor<1x9x768xf32>
    %367 = stablehlo.reduce(%366 init: %33) applies stablehlo.add across dimensions = [2] : (tensor<1x9x768xf32>, tensor<f32>) -> tensor<1x9xf32>
    %368 = stablehlo.broadcast_in_dim %367, dims = [0, 1] : (tensor<1x9xf32>) -> tensor<1x9x1xf32>
    %369 = stablehlo.divide %368, %31 : tensor<1x9x1xf32>
    %370 = stablehlo.add %369, %30 : tensor<1x9x1xf32>
    %371 = stablehlo.sqrt %370 : tensor<1x9x1xf32>
    %372 = stablehlo.reshape %371 : (tensor<1x9x1xf32>) -> tensor<1x9xf32>
    %373 = stablehlo.broadcast_in_dim %372, dims = [0, 1] : (tensor<1x9xf32>) -> tensor<1x9x768xf32>
    %374 = stablehlo.divide %362, %373 : tensor<1x9x768xf32>
    %375 = stablehlo.multiply %365, %374 : tensor<1x9x768xf32>
    %376 = stablehlo.dot_general %375, %62, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x3072xf32>) -> tensor<1x9x3072xf32>
    %377 = stablehlo.dot_general %375, %63, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x3072xf32>) -> tensor<1x9x3072xf32>
    %378 = stablehlo.negate %377 : tensor<1x9x3072xf32>
    %379 = stablehlo.exponential %378 : tensor<1x9x3072xf32>
    %380 = stablehlo.add %17, %379 : tensor<1x9x3072xf32>
    %381 = stablehlo.divide %17, %380 : tensor<1x9x3072xf32>
    %382 = stablehlo.multiply %377, %381 : tensor<1x9x3072xf32>
    %383 = stablehlo.multiply %376, %382 : tensor<1x9x3072xf32>
    %384 = stablehlo.dot_general %383, %64, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x3072xf32>, tensor<3072x768xf32>) -> tensor<1x9x768xf32>
    %385 = stablehlo.add %362, %384 : tensor<1x9x768xf32>
    %386 = stablehlo.multiply %385, %385 : tensor<1x9x768xf32>
    %387 = stablehlo.reduce(%386 init: %33) applies stablehlo.add across dimensions = [2] : (tensor<1x9x768xf32>, tensor<f32>) -> tensor<1x9xf32>
    %388 = stablehlo.broadcast_in_dim %387, dims = [0, 1] : (tensor<1x9xf32>) -> tensor<1x9x1xf32>
    %389 = stablehlo.divide %388, %31 : tensor<1x9x1xf32>
    %390 = stablehlo.add %389, %30 : tensor<1x9x1xf32>
    %391 = stablehlo.sqrt %390 : tensor<1x9x1xf32>
    %392 = stablehlo.reshape %391 : (tensor<1x9x1xf32>) -> tensor<1x9xf32>
    %393 = stablehlo.broadcast_in_dim %392, dims = [0, 1] : (tensor<1x9xf32>) -> tensor<1x9x768xf32>
    %394 = stablehlo.divide %385, %393 : tensor<1x9x768xf32>
    %395 = stablehlo.multiply %81, %394 : tensor<1x9x768xf32>
    %396 = stablehlo.dot_general %395, %66, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x9x768xf32>, tensor<768x32000xf32>) -> tensor<1x9x32000xf32>
    %397 = stablehlo.select %29, %67, %72 : tensor<i1>, tensor<i32>
    %398 = stablehlo.select %35, %15, %16 : tensor<i1>, tensor<i32>
    %399 = stablehlo.select %29, %43, %72 : tensor<i1>, tensor<i32>
    %400 = stablehlo.dynamic_slice %396, %397, %398, %399, sizes = [1, 1, 32000] : (tensor<1x9x32000xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x32000xf32>
    %401 = stablehlo.reshape %400 : (tensor<1x1x32000xf32>) -> tensor<1x32000xf32>
    %402 = "stablehlo.scatter"(%401, %37, %14) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      stablehlo.return %arg3 : tensor<f32>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x32000xf32>, tensor<1xi32>, tensor<1xf32>) -> tensor<1x32000xf32>
    %403 = stablehlo.select %78, %402, %401 : tensor<1x32000xi1>, tensor<1x32000xf32>
    %404 = stablehlo.iota dim = 1 : tensor<1x32000xi32>
    %405:2 = stablehlo.reduce(%403 init: %18), (%404 init: %72) across dimensions = [1] : (tensor<1x32000xf32>, tensor<1x32000xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
     reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<i32>, %arg5: tensor<i32>)  {
      %420 = stablehlo.compare  GT, %arg2, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %421 = stablehlo.compare  NE, %arg2, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %422 = stablehlo.or %420, %421 : tensor<i1>
      %423 = stablehlo.select %422, %arg2, %arg4 : tensor<i1>, tensor<f32>
      %424 = stablehlo.compare  EQ, %arg2, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %425 = stablehlo.compare  LT, %arg3, %arg5 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %426 = stablehlo.and %424, %425 : tensor<i1>
      %427 = stablehlo.or %422, %426 : tensor<i1>
      %428 = stablehlo.select %427, %arg3, %arg5 : tensor<i1>, tensor<i32>
      stablehlo.return %423, %428 : tensor<f32>, tensor<i32>
    }
    %406 = stablehlo.not %13 : tensor<1xi1>
    %407 = stablehlo.convert %406 : (tensor<1xi1>) -> tensor<1xi32>
    %408 = stablehlo.multiply %405#1, %407 : tensor<1xi32>
    %409 = stablehlo.convert %13 : (tensor<1xi1>) -> tensor<1xi32>
    %410 = stablehlo.multiply %36, %409 : tensor<1xi32>
    %411 = stablehlo.add %408, %410 : tensor<1xi32>
    %412 = stablehlo.broadcast_in_dim %411, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %413 = stablehlo.select %29, %12, %69 : tensor<i1>, tensor<i32>
    %414 = stablehlo.dynamic_update_slice %73, %412, %72, %413 : (tensor<1x20xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x20xi32>
    %415 = stablehlo.compare  EQ, %411, %37 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %416 = stablehlo.or %13, %415 : tensor<1xi1>
    %417 = stablehlo.slice %155 [0:1, 8:9] : (tensor<1x9xi32>) -> tensor<1x1xi32>
    %418 = stablehlo.add %417, %11 : tensor<1x1xi32>
    %419:37 = stablehlo.while(%iterArg = %45, %iterArg_0 = %46, %iterArg_1 = %47, %iterArg_2 = %48, %iterArg_3 = %49, %iterArg_4 = %50, %iterArg_5 = %51, %iterArg_6 = %52, %iterArg_7 = %53, %iterArg_8 = %54, %iterArg_9 = %55, %iterArg_10 = %56, %iterArg_11 = %57, %iterArg_12 = %58, %iterArg_13 = %59, %iterArg_14 = %50, %iterArg_15 = %60, %iterArg_16 = %61, %iterArg_17 = %62, %iterArg_18 = %63, %iterArg_19 = %64, %iterArg_20 = %65, %iterArg_21 = %66, %iterArg_22 = %67, %iterArg_23 = %68, %iterArg_24 = %70, %iterArg_25 = %414, %iterArg_26 = %412, %iterArg_27 = %416, %iterArg_28 = %196, %iterArg_29 = %69, %iterArg_30 = %190, %iterArg_31 = %123, %iterArg_32 = %69, %iterArg_33 = %315, %iterArg_34 = %278, %iterArg_35 = %418) : tensor<32000x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<2048x1x128xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<2048x1x128xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768x32000xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x20xi32>, tensor<1x1xi32>, tensor<1xi1>, tensor<1x20xi32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<1x1xi32>
     cond {
      %420 = stablehlo.compare  EQ, %iterArg_24, %28 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %421 = stablehlo.compare  NE, %420, %29 : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %422 = stablehlo.reduce(%iterArg_27 init: %35) applies stablehlo.and across dimensions = [0] : (tensor<1xi1>, tensor<i1>) -> tensor<i1>
      %423 = stablehlo.or %421, %422 : tensor<i1>
      %424 = stablehlo.not %423 : tensor<i1>
      stablehlo.return %424 : tensor<i1>
    } do {
      %420 = stablehlo.add %iterArg_24, %67 : tensor<i32>
      %421 = stablehlo.subtract %iterArg_24, %72 : tensor<i32>
      %422 = stablehlo.maximum %421, %72 : tensor<i32>
      %423 = stablehlo.minimum %422, %67 : tensor<i32>
      %424 = stablehlo.subtract %67, %423 : tensor<i32>
      %425 = stablehlo.compare  NE, %424, %72 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %426 = stablehlo.broadcast_in_dim %425, dims = [] : (tensor<i1>) -> tensor<1x32000xi1>
      %427 = stablehlo.broadcast_in_dim %iterArg_20, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %428 = stablehlo.compare  LT, %iterArg_26, %10 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
      %429 = stablehlo.add %iterArg_26, %9 : tensor<1x1xi32>
      %430 = stablehlo.select %428, %429, %iterArg_26 : tensor<1x1xi1>, tensor<1x1xi32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
      %432 = stablehlo.compare  GE, %431, %8 : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi1>
      %433 = stablehlo.concatenate %40, %39, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %434 = stablehlo.compare  LT, %38, %38 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      %435 = stablehlo.select %434, %37, %38 : tensor<1xi1>, tensor<1xi32>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
      %437 = "stablehlo.gather"(%433, %436) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
      %438 = stablehlo.concatenate %36, %39, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %439 = stablehlo.compare  LT, %38, %38 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      %440 = stablehlo.select %439, %37, %38 : tensor<1xi1>, tensor<1xi32>
      %441 = stablehlo.broadcast_in_dim %440, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
      %442 = "stablehlo.gather"(%438, %441) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
      %443 = stablehlo.subtract %437, %442 : tensor<1xi32>
      %444 = stablehlo.broadcast_in_dim %443, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
      %445 = stablehlo.compare  LE, %431, %444 : (tensor<1x1x1xi32>, tensor<1x1x1xi32>) -> tensor<1x1x1xi1>
      %446 = stablehlo.and %432, %445 : tensor<1x1x1xi1>
      %447 = stablehlo.reduce(%446 init: %35) applies stablehlo.and across dimensions = [2] : (tensor<1x1x1xi1>, tensor<i1>) -> tensor<1x1xi1>
      %448 = stablehlo.broadcast_in_dim %447, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<1x1x768xi1>
      %449 = "stablehlo.gather"(%iterArg, %431) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>} : (tensor<32000x768xf32>, tensor<1x1x1xi32>) -> tensor<1x1x768xf32>
      %450 = stablehlo.select %448, %449, %7 : tensor<1x1x768xi1>, tensor<1x1x768xf32>
      %451 = stablehlo.broadcast_in_dim %iterArg_0, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %452 = stablehlo.multiply %450, %450 : tensor<1x1x768xf32>
      %453 = stablehlo.reduce(%452 init: %33) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %454 = stablehlo.broadcast_in_dim %453, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %455 = stablehlo.divide %454, %6 : tensor<1x1x1xf32>
      %456 = stablehlo.add %455, %5 : tensor<1x1x1xf32>
      %457 = stablehlo.sqrt %456 : tensor<1x1x1xf32>
      %458 = stablehlo.reshape %457 : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x768xf32>
      %460 = stablehlo.divide %450, %459 : tensor<1x1x768xf32>
      %461 = stablehlo.multiply %451, %460 : tensor<1x1x768xf32>
      %462 = stablehlo.dot_general %461, %iterArg_3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %463 = stablehlo.reshape %462 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %464 = stablehlo.compare  LT, %iterArg_29, %72 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %465 = stablehlo.add %iterArg_29, %28 : tensor<i32>
      %466 = stablehlo.select %464, %465, %iterArg_29 : tensor<i1>, tensor<i32>
      %467 = stablehlo.dynamic_update_slice %iterArg_31, %463, %72, %466, %72, %72 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %468 = stablehlo.dot_general %461, %iterArg_1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %469 = stablehlo.reshape %468 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %470 = stablehlo.compare  LT, %iterArg_35, %10 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
      %471 = stablehlo.add %iterArg_35, %4 : tensor<1x1xi32>
      %472 = stablehlo.select %470, %471, %iterArg_35 : tensor<1x1xi1>, tensor<1x1xi32>
      %473 = stablehlo.broadcast_in_dim %472, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
      %474 = "stablehlo.gather"(%iterArg_4, %473) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 128>} : (tensor<2048x1x128xf32>, tensor<1x1x1xi32>) -> tensor<1x1x1x128xf32>
      %475 = stablehlo.slice %474 [0:1, 0:1, 0:1, 64:128] : (tensor<1x1x1x128xf32>) -> tensor<1x1x1x64xf32>
      %476 = stablehlo.reshape %475 : (tensor<1x1x1x64xf32>) -> tensor<1x1x64xf32>
      %477 = stablehlo.broadcast_in_dim %476, dims = [0, 1, 3] : (tensor<1x1x64xf32>) -> tensor<1x1x12x64xf32>
      %478 = stablehlo.multiply %469, %477 : tensor<1x1x12x64xf32>
      %479 = "stablehlo.gather"(%469, %23) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 12, 32>} : (tensor<1x1x12x64xf32>, tensor<1xi32>) -> tensor<1x1x12x32xf32>
      %480 = stablehlo.negate %479 : tensor<1x1x12x32xf32>
      %481 = "stablehlo.gather"(%469, %38) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 12, 32>} : (tensor<1x1x12x64xf32>, tensor<1xi32>) -> tensor<1x1x12x32xf32>
      %482 = stablehlo.concatenate %480, %481, dim = 3 : (tensor<1x1x12x32xf32>, tensor<1x1x12x32xf32>) -> tensor<1x1x12x64xf32>
      %483 = stablehlo.slice %474 [0:1, 0:1, 0:1, 0:64] : (tensor<1x1x1x128xf32>) -> tensor<1x1x1x64xf32>
      %484 = stablehlo.reshape %483 : (tensor<1x1x1x64xf32>) -> tensor<1x1x64xf32>
      %485 = stablehlo.broadcast_in_dim %484, dims = [0, 1, 3] : (tensor<1x1x64xf32>) -> tensor<1x1x12x64xf32>
      %486 = stablehlo.multiply %482, %485 : tensor<1x1x12x64xf32>
      %487 = stablehlo.add %478, %486 : tensor<1x1x12x64xf32>
      %488 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %489 = stablehlo.divide %487, %488 : tensor<1x1x12x64xf32>
      %490 = stablehlo.dot_general %461, %iterArg_2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %491 = stablehlo.reshape %490 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %492 = stablehlo.reshape %475 : (tensor<1x1x1x64xf32>) -> tensor<1x1x64xf32>
      %493 = stablehlo.broadcast_in_dim %492, dims = [0, 1, 3] : (tensor<1x1x64xf32>) -> tensor<1x1x12x64xf32>
      %494 = stablehlo.multiply %491, %493 : tensor<1x1x12x64xf32>
      %495 = "stablehlo.gather"(%491, %23) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 12, 32>} : (tensor<1x1x12x64xf32>, tensor<1xi32>) -> tensor<1x1x12x32xf32>
      %496 = stablehlo.negate %495 : tensor<1x1x12x32xf32>
      %497 = "stablehlo.gather"(%491, %38) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 12, 32>} : (tensor<1x1x12x64xf32>, tensor<1xi32>) -> tensor<1x1x12x32xf32>
      %498 = stablehlo.concatenate %496, %497, dim = 3 : (tensor<1x1x12x32xf32>, tensor<1x1x12x32xf32>) -> tensor<1x1x12x64xf32>
      %499 = stablehlo.reshape %483 : (tensor<1x1x1x64xf32>) -> tensor<1x1x64xf32>
      %500 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 3] : (tensor<1x1x64xf32>) -> tensor<1x1x12x64xf32>
      %501 = stablehlo.multiply %498, %500 : tensor<1x1x12x64xf32>
      %502 = stablehlo.add %494, %501 : tensor<1x1x12x64xf32>
      %503 = stablehlo.compare  LT, %iterArg_29, %72 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %504 = stablehlo.add %iterArg_29, %28 : tensor<i32>
      %505 = stablehlo.select %503, %504, %iterArg_29 : tensor<i1>, tensor<i32>
      %506 = stablehlo.dynamic_update_slice %iterArg_30, %502, %72, %505, %72, %72 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %507 = stablehlo.dot_general %489, %506, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %508 = stablehlo.iota dim = 0 : tensor<20xi32>
      %509 = stablehlo.add %iterArg_29, %67 : tensor<i32>
      %510 = stablehlo.broadcast_in_dim %509, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %511 = stablehlo.compare  LT, %508, %510 : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %512 = stablehlo.broadcast_in_dim %511, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %513 = stablehlo.broadcast_in_dim %iterArg_28, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %514 = stablehlo.compare  NE, %513, %3 : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %515 = stablehlo.iota dim = 0 : tensor<2048xi32>
      %516 = stablehlo.broadcast_in_dim %515, dims = [1] : (tensor<2048xi32>) -> tensor<1x2048xi32>
      %517 = stablehlo.broadcast_in_dim %516, dims = [0, 1] : (tensor<1x2048xi32>) -> tensor<1x2048x1xi32>
      %518 = stablehlo.reshape %517 : (tensor<1x2048x1xi32>) -> tensor<1x2048xi32>
      %519 = stablehlo.broadcast_in_dim %518, dims = [0, 1] : (tensor<1x2048xi32>) -> tensor<1x2048x2048xi32>
      %520 = stablehlo.broadcast_in_dim %516, dims = [0, 2] : (tensor<1x2048xi32>) -> tensor<1x1x2048xi32>
      %521 = stablehlo.reshape %520 : (tensor<1x1x2048xi32>) -> tensor<1x2048xi32>
      %522 = stablehlo.broadcast_in_dim %521, dims = [0, 2] : (tensor<1x2048xi32>) -> tensor<1x2048x2048xi32>
      %523 = stablehlo.compare  GE, %519, %522 : (tensor<1x2048x2048xi32>, tensor<1x2048x2048xi32>) -> tensor<1x2048x2048xi1>
      %524 = stablehlo.broadcast_in_dim %523, dims = [0, 2, 3] : (tensor<1x2048x2048xi1>) -> tensor<1x1x2048x2048xi1>
      %525 = stablehlo.compare  LT, %iterArg_29, %72 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %526 = stablehlo.add %iterArg_29, %25 : tensor<i32>
      %527 = stablehlo.select %525, %526, %iterArg_29 : tensor<i1>, tensor<i32>
      %528 = stablehlo.dynamic_slice %524, %72, %72, %527, %72, sizes = [1, 1, 1, 20] : (tensor<1x1x2048x2048xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %529 = stablehlo.and %514, %528 : tensor<1x1x1x20xi1>
      %530 = stablehlo.convert %529 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %531 = stablehlo.compare  NE, %530, %2 : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %532 = stablehlo.and %512, %531 : tensor<1x1x1x20xi1>
      %533 = stablehlo.convert %532 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %534 = stablehlo.compare  GT, %533, %2 : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %535 = stablehlo.select %534, %2, %1 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %536 = stablehlo.reshape %535 : (tensor<1x1x1x20xf32>) -> tensor<1x1x20xf32>
      %537 = stablehlo.broadcast_in_dim %536, dims = [0, 2, 3] : (tensor<1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %538 = stablehlo.add %507, %537 : tensor<1x12x1x20xf32>
      %539 = stablehlo.reduce(%538 init: %18) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %540 = stablehlo.broadcast_in_dim %539, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %541 = stablehlo.reshape %540 : (tensor<1x12x1x1xf32>) -> tensor<1x12x1xf32>
      %542 = stablehlo.broadcast_in_dim %541, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x20xf32>
      %543 = stablehlo.subtract %538, %542 : tensor<1x12x1x20xf32>
      %544 = stablehlo.exponential %543 : tensor<1x12x1x20xf32>
      %545 = stablehlo.reduce(%544 init: %33) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %546 = stablehlo.broadcast_in_dim %545, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %547 = stablehlo.reshape %546 : (tensor<1x12x1x1xf32>) -> tensor<1x12x1xf32>
      %548 = stablehlo.broadcast_in_dim %547, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x20xf32>
      %549 = stablehlo.divide %544, %548 : tensor<1x12x1x20xf32>
      %550 = stablehlo.dot_general %467, %549, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %551 = stablehlo.transpose %550, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,1,12,64]{1,3,2,0}"} : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %552 = stablehlo.reshape %551 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %553 = stablehlo.dot_general %552, %iterArg_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %554 = stablehlo.add %450, %553 : tensor<1x1x768xf32>
      %555 = stablehlo.broadcast_in_dim %iterArg_6, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %556 = stablehlo.multiply %554, %554 : tensor<1x1x768xf32>
      %557 = stablehlo.reduce(%556 init: %33) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %558 = stablehlo.broadcast_in_dim %557, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %559 = stablehlo.divide %558, %6 : tensor<1x1x1xf32>
      %560 = stablehlo.add %559, %5 : tensor<1x1x1xf32>
      %561 = stablehlo.sqrt %560 : tensor<1x1x1xf32>
      %562 = stablehlo.reshape %561 : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
      %563 = stablehlo.broadcast_in_dim %562, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x768xf32>
      %564 = stablehlo.divide %554, %563 : tensor<1x1x768xf32>
      %565 = stablehlo.multiply %555, %564 : tensor<1x1x768xf32>
      %566 = stablehlo.dot_general %565, %iterArg_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %567 = stablehlo.dot_general %565, %iterArg_8, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %568 = stablehlo.negate %567 : tensor<1x1x3072xf32>
      %569 = stablehlo.exponential %568 : tensor<1x1x3072xf32>
      %570 = stablehlo.add %0, %569 : tensor<1x1x3072xf32>
      %571 = stablehlo.divide %0, %570 : tensor<1x1x3072xf32>
      %572 = stablehlo.multiply %567, %571 : tensor<1x1x3072xf32>
      %573 = stablehlo.multiply %566, %572 : tensor<1x1x3072xf32>
      %574 = stablehlo.dot_general %573, %iterArg_9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %575 = stablehlo.add %554, %574 : tensor<1x1x768xf32>
      %576 = stablehlo.broadcast_in_dim %iterArg_10, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %577 = stablehlo.multiply %575, %575 : tensor<1x1x768xf32>
      %578 = stablehlo.reduce(%577 init: %33) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %579 = stablehlo.broadcast_in_dim %578, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %580 = stablehlo.divide %579, %6 : tensor<1x1x1xf32>
      %581 = stablehlo.add %580, %5 : tensor<1x1x1xf32>
      %582 = stablehlo.sqrt %581 : tensor<1x1x1xf32>
      %583 = stablehlo.reshape %582 : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
      %584 = stablehlo.broadcast_in_dim %583, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x768xf32>
      %585 = stablehlo.divide %575, %584 : tensor<1x1x768xf32>
      %586 = stablehlo.multiply %576, %585 : tensor<1x1x768xf32>
      %587 = stablehlo.dot_general %586, %iterArg_13, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %588 = stablehlo.reshape %587 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %589 = stablehlo.compare  LT, %iterArg_32, %72 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %590 = stablehlo.add %iterArg_32, %28 : tensor<i32>
      %591 = stablehlo.select %589, %590, %iterArg_32 : tensor<i1>, tensor<i32>
      %592 = stablehlo.dynamic_update_slice %iterArg_34, %588, %72, %591, %72, %72 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %593 = stablehlo.dot_general %586, %iterArg_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %594 = stablehlo.reshape %593 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %595 = stablehlo.compare  LT, %iterArg_35, %10 : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
      %596 = stablehlo.add %iterArg_35, %4 : tensor<1x1xi32>
      %597 = stablehlo.select %595, %596, %iterArg_35 : tensor<1x1xi1>, tensor<1x1xi32>
      %598 = stablehlo.broadcast_in_dim %597, dims = [0, 1] : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
      %599 = "stablehlo.gather"(%iterArg_14, %598) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 128>} : (tensor<2048x1x128xf32>, tensor<1x1x1xi32>) -> tensor<1x1x1x128xf32>
      %600 = stablehlo.slice %599 [0:1, 0:1, 0:1, 64:128] : (tensor<1x1x1x128xf32>) -> tensor<1x1x1x64xf32>
      %601 = stablehlo.reshape %600 : (tensor<1x1x1x64xf32>) -> tensor<1x1x64xf32>
      %602 = stablehlo.broadcast_in_dim %601, dims = [0, 1, 3] : (tensor<1x1x64xf32>) -> tensor<1x1x12x64xf32>
      %603 = stablehlo.multiply %594, %602 : tensor<1x1x12x64xf32>
      %604 = "stablehlo.gather"(%594, %23) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 12, 32>} : (tensor<1x1x12x64xf32>, tensor<1xi32>) -> tensor<1x1x12x32xf32>
      %605 = stablehlo.negate %604 : tensor<1x1x12x32xf32>
      %606 = "stablehlo.gather"(%594, %38) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 12, 32>} : (tensor<1x1x12x64xf32>, tensor<1xi32>) -> tensor<1x1x12x32xf32>
      %607 = stablehlo.concatenate %605, %606, dim = 3 : (tensor<1x1x12x32xf32>, tensor<1x1x12x32xf32>) -> tensor<1x1x12x64xf32>
      %608 = stablehlo.slice %599 [0:1, 0:1, 0:1, 0:64] : (tensor<1x1x1x128xf32>) -> tensor<1x1x1x64xf32>
      %609 = stablehlo.reshape %608 : (tensor<1x1x1x64xf32>) -> tensor<1x1x64xf32>
      %610 = stablehlo.broadcast_in_dim %609, dims = [0, 1, 3] : (tensor<1x1x64xf32>) -> tensor<1x1x12x64xf32>
      %611 = stablehlo.multiply %607, %610 : tensor<1x1x12x64xf32>
      %612 = stablehlo.add %603, %611 : tensor<1x1x12x64xf32>
      %613 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<1x1x12x64xf32>
      %614 = stablehlo.divide %612, %613 : tensor<1x1x12x64xf32>
      %615 = stablehlo.dot_general %586, %iterArg_12, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %616 = stablehlo.reshape %615 : (tensor<1x1x768xf32>) -> tensor<1x1x12x64xf32>
      %617 = stablehlo.reshape %600 : (tensor<1x1x1x64xf32>) -> tensor<1x1x64xf32>
      %618 = stablehlo.broadcast_in_dim %617, dims = [0, 1, 3] : (tensor<1x1x64xf32>) -> tensor<1x1x12x64xf32>
      %619 = stablehlo.multiply %616, %618 : tensor<1x1x12x64xf32>
      %620 = "stablehlo.gather"(%616, %23) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 12, 32>} : (tensor<1x1x12x64xf32>, tensor<1xi32>) -> tensor<1x1x12x32xf32>
      %621 = stablehlo.negate %620 : tensor<1x1x12x32xf32>
      %622 = "stablehlo.gather"(%616, %38) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [3]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 12, 32>} : (tensor<1x1x12x64xf32>, tensor<1xi32>) -> tensor<1x1x12x32xf32>
      %623 = stablehlo.concatenate %621, %622, dim = 3 : (tensor<1x1x12x32xf32>, tensor<1x1x12x32xf32>) -> tensor<1x1x12x64xf32>
      %624 = stablehlo.reshape %608 : (tensor<1x1x1x64xf32>) -> tensor<1x1x64xf32>
      %625 = stablehlo.broadcast_in_dim %624, dims = [0, 1, 3] : (tensor<1x1x64xf32>) -> tensor<1x1x12x64xf32>
      %626 = stablehlo.multiply %623, %625 : tensor<1x1x12x64xf32>
      %627 = stablehlo.add %619, %626 : tensor<1x1x12x64xf32>
      %628 = stablehlo.compare  LT, %iterArg_32, %72 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %629 = stablehlo.add %iterArg_32, %28 : tensor<i32>
      %630 = stablehlo.select %628, %629, %iterArg_32 : tensor<i1>, tensor<i32>
      %631 = stablehlo.dynamic_update_slice %iterArg_33, %627, %72, %630, %72, %72 : (tensor<1x20x12x64xf32>, tensor<1x1x12x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x20x12x64xf32>
      %632 = stablehlo.dot_general %614, %631, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x12x64xf32>, tensor<1x20x12x64xf32>) -> tensor<1x12x1x20xf32>
      %633 = stablehlo.iota dim = 0 : tensor<20xi32>
      %634 = stablehlo.add %iterArg_32, %67 : tensor<i32>
      %635 = stablehlo.broadcast_in_dim %634, dims = [] : (tensor<i32>) -> tensor<20xi32>
      %636 = stablehlo.compare  LT, %633, %635 : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
      %637 = stablehlo.broadcast_in_dim %636, dims = [3] : (tensor<20xi1>) -> tensor<1x1x1x20xi1>
      %638 = stablehlo.broadcast_in_dim %iterArg_28, dims = [0, 3] : (tensor<1x20xi32>) -> tensor<1x1x1x20xi32>
      %639 = stablehlo.compare  NE, %638, %3 : (tensor<1x1x1x20xi32>, tensor<1x1x1x20xi32>) -> tensor<1x1x1x20xi1>
      %640 = stablehlo.iota dim = 0 : tensor<2048xi32>
      %641 = stablehlo.broadcast_in_dim %640, dims = [1] : (tensor<2048xi32>) -> tensor<1x2048xi32>
      %642 = stablehlo.broadcast_in_dim %641, dims = [0, 1] : (tensor<1x2048xi32>) -> tensor<1x2048x1xi32>
      %643 = stablehlo.reshape %642 : (tensor<1x2048x1xi32>) -> tensor<1x2048xi32>
      %644 = stablehlo.broadcast_in_dim %643, dims = [0, 1] : (tensor<1x2048xi32>) -> tensor<1x2048x2048xi32>
      %645 = stablehlo.broadcast_in_dim %641, dims = [0, 2] : (tensor<1x2048xi32>) -> tensor<1x1x2048xi32>
      %646 = stablehlo.reshape %645 : (tensor<1x1x2048xi32>) -> tensor<1x2048xi32>
      %647 = stablehlo.broadcast_in_dim %646, dims = [0, 2] : (tensor<1x2048xi32>) -> tensor<1x2048x2048xi32>
      %648 = stablehlo.compare  GE, %644, %647 : (tensor<1x2048x2048xi32>, tensor<1x2048x2048xi32>) -> tensor<1x2048x2048xi1>
      %649 = stablehlo.broadcast_in_dim %648, dims = [0, 2, 3] : (tensor<1x2048x2048xi1>) -> tensor<1x1x2048x2048xi1>
      %650 = stablehlo.compare  LT, %iterArg_32, %72 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %651 = stablehlo.add %iterArg_32, %25 : tensor<i32>
      %652 = stablehlo.select %650, %651, %iterArg_32 : tensor<i1>, tensor<i32>
      %653 = stablehlo.dynamic_slice %649, %72, %72, %652, %72, sizes = [1, 1, 1, 20] : (tensor<1x1x2048x2048xi1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x20xi1>
      %654 = stablehlo.and %639, %653 : tensor<1x1x1x20xi1>
      %655 = stablehlo.convert %654 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %656 = stablehlo.compare  NE, %655, %2 : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %657 = stablehlo.and %637, %656 : tensor<1x1x1x20xi1>
      %658 = stablehlo.convert %657 : (tensor<1x1x1x20xi1>) -> tensor<1x1x1x20xf32>
      %659 = stablehlo.compare  GT, %658, %2 : (tensor<1x1x1x20xf32>, tensor<1x1x1x20xf32>) -> tensor<1x1x1x20xi1>
      %660 = stablehlo.select %659, %2, %1 : tensor<1x1x1x20xi1>, tensor<1x1x1x20xf32>
      %661 = stablehlo.reshape %660 : (tensor<1x1x1x20xf32>) -> tensor<1x1x20xf32>
      %662 = stablehlo.broadcast_in_dim %661, dims = [0, 2, 3] : (tensor<1x1x20xf32>) -> tensor<1x12x1x20xf32>
      %663 = stablehlo.add %632, %662 : tensor<1x12x1x20xf32>
      %664 = stablehlo.reduce(%663 init: %18) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %666 = stablehlo.reshape %665 : (tensor<1x12x1x1xf32>) -> tensor<1x12x1xf32>
      %667 = stablehlo.broadcast_in_dim %666, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x20xf32>
      %668 = stablehlo.subtract %663, %667 : tensor<1x12x1x20xf32>
      %669 = stablehlo.exponential %668 : tensor<1x12x1x20xf32>
      %670 = stablehlo.reduce(%669 init: %33) applies stablehlo.add across dimensions = [3] : (tensor<1x12x1x20xf32>, tensor<f32>) -> tensor<1x12x1xf32>
      %671 = stablehlo.broadcast_in_dim %670, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x1xf32>
      %672 = stablehlo.reshape %671 : (tensor<1x12x1x1xf32>) -> tensor<1x12x1xf32>
      %673 = stablehlo.broadcast_in_dim %672, dims = [0, 1, 2] : (tensor<1x12x1xf32>) -> tensor<1x12x1x20xf32>
      %674 = stablehlo.divide %669, %673 : tensor<1x12x1x20xf32>
      %675 = stablehlo.dot_general %592, %674, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x20x12x64xf32>, tensor<1x12x1x20xf32>) -> tensor<1x12x64x1xf32>
      %676 = stablehlo.transpose %675, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,1,12,64]{1,3,2,0}"} : (tensor<1x12x64x1xf32>) -> tensor<1x1x12x64xf32>
      %677 = stablehlo.reshape %676 : (tensor<1x1x12x64xf32>) -> tensor<1x1x768xf32>
      %678 = stablehlo.dot_general %677, %iterArg_15, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x768xf32>) -> tensor<1x1x768xf32>
      %679 = stablehlo.add %575, %678 : tensor<1x1x768xf32>
      %680 = stablehlo.broadcast_in_dim %iterArg_16, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %681 = stablehlo.multiply %679, %679 : tensor<1x1x768xf32>
      %682 = stablehlo.reduce(%681 init: %33) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %684 = stablehlo.divide %683, %6 : tensor<1x1x1xf32>
      %685 = stablehlo.add %684, %5 : tensor<1x1x1xf32>
      %686 = stablehlo.sqrt %685 : tensor<1x1x1xf32>
      %687 = stablehlo.reshape %686 : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
      %688 = stablehlo.broadcast_in_dim %687, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x768xf32>
      %689 = stablehlo.divide %679, %688 : tensor<1x1x768xf32>
      %690 = stablehlo.multiply %680, %689 : tensor<1x1x768xf32>
      %691 = stablehlo.dot_general %690, %iterArg_17, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %692 = stablehlo.dot_general %690, %iterArg_18, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x3072xf32>) -> tensor<1x1x3072xf32>
      %693 = stablehlo.negate %692 : tensor<1x1x3072xf32>
      %694 = stablehlo.exponential %693 : tensor<1x1x3072xf32>
      %695 = stablehlo.add %0, %694 : tensor<1x1x3072xf32>
      %696 = stablehlo.divide %0, %695 : tensor<1x1x3072xf32>
      %697 = stablehlo.multiply %692, %696 : tensor<1x1x3072xf32>
      %698 = stablehlo.multiply %691, %697 : tensor<1x1x3072xf32>
      %699 = stablehlo.dot_general %698, %iterArg_19, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x1x768xf32>
      %700 = stablehlo.add %679, %699 : tensor<1x1x768xf32>
      %701 = stablehlo.multiply %700, %700 : tensor<1x1x768xf32>
      %702 = stablehlo.reduce(%701 init: %33) applies stablehlo.add across dimensions = [2] : (tensor<1x1x768xf32>, tensor<f32>) -> tensor<1x1xf32>
      %703 = stablehlo.broadcast_in_dim %702, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %704 = stablehlo.divide %703, %6 : tensor<1x1x1xf32>
      %705 = stablehlo.add %704, %5 : tensor<1x1x1xf32>
      %706 = stablehlo.sqrt %705 : tensor<1x1x1xf32>
      %707 = stablehlo.reshape %706 : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
      %708 = stablehlo.broadcast_in_dim %707, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x768xf32>
      %709 = stablehlo.divide %700, %708 : tensor<1x1x768xf32>
      %710 = stablehlo.multiply %427, %709 : tensor<1x1x768xf32>
      %711 = stablehlo.dot_general %710, %iterArg_21, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x768xf32>, tensor<768x32000xf32>) -> tensor<1x1x32000xf32>
      %712 = stablehlo.select %29, %67, %72 : tensor<i1>, tensor<i32>
      %713 = stablehlo.select %35, %72, %16 : tensor<i1>, tensor<i32>
      %714 = stablehlo.select %29, %43, %72 : tensor<i1>, tensor<i32>
      %715 = stablehlo.dynamic_slice %711, %712, %713, %714, sizes = [1, 1, 32000] : (tensor<1x1x32000xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x32000xf32>
      %716 = stablehlo.reshape %715 : (tensor<1x1x32000xf32>) -> tensor<1x32000xf32>
      %717 = "stablehlo.scatter"(%716, %37, %14) ({
      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        stablehlo.return %arg3 : tensor<f32>
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x32000xf32>, tensor<1xi32>, tensor<1xf32>) -> tensor<1x32000xf32>
      %718 = stablehlo.select %426, %717, %716 : tensor<1x32000xi1>, tensor<1x32000xf32>
      %719 = stablehlo.iota dim = 1 : tensor<1x32000xi32>
      %720:2 = stablehlo.reduce(%718 init: %18), (%719 init: %72) across dimensions = [1] : (tensor<1x32000xf32>, tensor<1x32000xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
       reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<i32>, %arg5: tensor<i32>)  {
        %739 = stablehlo.compare  GT, %arg2, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
        %740 = stablehlo.compare  NE, %arg2, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
        %741 = stablehlo.or %739, %740 : tensor<i1>
        %742 = stablehlo.select %741, %arg2, %arg4 : tensor<i1>, tensor<f32>
        %743 = stablehlo.compare  EQ, %arg2, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
        %744 = stablehlo.compare  LT, %arg3, %arg5 : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %745 = stablehlo.and %743, %744 : tensor<i1>
        %746 = stablehlo.or %741, %745 : tensor<i1>
        %747 = stablehlo.select %746, %arg3, %arg5 : tensor<i1>, tensor<i32>
        stablehlo.return %742, %747 : tensor<f32>, tensor<i32>
      }
      %721 = stablehlo.not %iterArg_27 : tensor<1xi1>
      %722 = stablehlo.convert %721 : (tensor<1xi1>) -> tensor<1xi32>
      %723 = stablehlo.multiply %720#1, %722 : tensor<1xi32>
      %724 = stablehlo.broadcast_in_dim %iterArg_22, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %725 = stablehlo.convert %iterArg_27 : (tensor<1xi1>) -> tensor<1xi32>
      %726 = stablehlo.multiply %724, %725 : tensor<1xi32>
      %727 = stablehlo.add %723, %726 : tensor<1xi32>
      %728 = stablehlo.broadcast_in_dim %727, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
      %729 = stablehlo.compare  LT, %iterArg_24, %72 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %730 = stablehlo.add %iterArg_24, %28 : tensor<i32>
      %731 = stablehlo.select %729, %730, %iterArg_24 : tensor<i1>, tensor<i32>
      %732 = stablehlo.dynamic_update_slice %iterArg_25, %728, %72, %731 : (tensor<1x20xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x20xi32>
      %733 = stablehlo.broadcast_in_dim %iterArg_23, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %734 = stablehlo.compare  EQ, %727, %733 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      %735 = stablehlo.or %iterArg_27, %734 : tensor<1xi1>
      %736 = stablehlo.add %iterArg_29, %67 : tensor<i32>
      %737 = stablehlo.add %iterArg_32, %67 : tensor<i32>
      %738 = stablehlo.add %iterArg_35, %11 : tensor<1x1xi32>
      stablehlo.return %iterArg, %iterArg_0, %iterArg_1, %iterArg_2, %iterArg_3, %iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11, %iterArg_12, %iterArg_13, %iterArg_14, %iterArg_15, %iterArg_16, %iterArg_17, %iterArg_18, %iterArg_19, %iterArg_20, %iterArg_21, %iterArg_22, %iterArg_23, %420, %732, %728, %735, %iterArg_28, %736, %506, %467, %737, %631, %592, %738 : tensor<32000x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<2048x1x128xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<2048x1x128xf32>, tensor<768x768xf32>, tensor<768xf32>, tensor<768x3072xf32>, tensor<768x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>, tensor<768x32000xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x20xi32>, tensor<1x1xi32>, tensor<1xi1>, tensor<1x20xi32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<i32>, tensor<1x20x12x64xf32>, tensor<1x20x12x64xf32>, tensor<1x1xi32>
    }
    return %419#26 : tensor<1x20xi32>
  }
}


