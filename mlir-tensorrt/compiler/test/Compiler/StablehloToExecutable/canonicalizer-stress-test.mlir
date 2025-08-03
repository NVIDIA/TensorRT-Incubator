// RUN: mlir-tensorrt-opt %s -stablehlo-simplification-pipeline | FileCheck %s

// CHECK-LABEL: @main() -> tensor<4xi32> {
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.constant dense<[1, 16, 1, 128]> : tensor<4xi32>
//  CHECK-NEXT:     return %[[v0]] : tensor<4xi32>

func.func @main() -> tensor<?xi32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<1024> : tensor<1xi32>
  %2 = stablehlo.dynamic_broadcast_in_dim %0, %1, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  %3 = stablehlo.constant dense<1> : tensor<i32>
  %4 = stablehlo.constant dense<2> : tensor<1xi32>
  %5 = stablehlo.dynamic_broadcast_in_dim %3, %4, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %6 = stablehlo.get_dimension_size %2, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %7 = stablehlo.constant dense<1> : tensor<1xi32>
  %8 = stablehlo.dynamic_reshape %6, %7 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %9 = stablehlo.concatenate %8, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %10 = stablehlo.concatenate %5, %9, dim = 0 : (tensor<?xi32>, tensor<1xi32>) -> tensor<3xi32>
  %11 = stablehlo.dynamic_broadcast_in_dim %2, %10, dims = [2] : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %12 = stablehlo.get_dimension_size %11, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %13 = stablehlo.constant dense<1> : tensor<1xi32>
  %14 = stablehlo.dynamic_reshape %12, %13 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %15 = stablehlo.get_dimension_size %11, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %16 = stablehlo.constant dense<1> : tensor<1xi32>
  %17 = stablehlo.dynamic_reshape %15, %16 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %18 = stablehlo.get_dimension_size %11, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %19 = stablehlo.constant dense<1> : tensor<1xi32>
  %20 = stablehlo.dynamic_reshape %18, %19 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %21 = stablehlo.concatenate %14, %17, %20, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %22 = stablehlo.constant dense<[51200, 1024]> : tensor<2xi32>
  %23 = stablehlo.dynamic_iota %22, dim = 0 : (tensor<2xi32>) -> tensor<?x?xf32>
  %24 = stablehlo.constant dense<1> : tensor<1x128xi32>
  %25 = stablehlo.constant dense<1> : tensor<1xi32>
  %26 = stablehlo.get_dimension_size %23, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %27 = stablehlo.constant dense<1> : tensor<1xi32>
  %28 = stablehlo.dynamic_reshape %26, %27 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %29 = stablehlo.get_dimension_size %23, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %30 = stablehlo.constant dense<1> : tensor<1xi32>
  %31 = stablehlo.dynamic_reshape %29, %30 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %32 = stablehlo.concatenate %28, %31, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %33 = stablehlo.constant dense<1> : tensor<1xi32>
  %34 = stablehlo.constant dense<2> : tensor<1xi32>
  %35 = stablehlo.real_dynamic_slice %32, %33, %34, %25 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %36 = stablehlo.concatenate %25, %35, dim = 0 : (tensor<1xi32>, tensor<?xi32>) -> tensor<2xi32>
  %37 = "stablehlo.dynamic_gather"(%23, %24, %36) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>} : (tensor<?x?xf32>, tensor<1x128xi32>, tensor<2xi32>) -> tensor<1x128x?xf32>
  %38 = stablehlo.get_dimension_size %37, dim = 0 : (tensor<1x128x?xf32>) -> tensor<i32>
  %39 = stablehlo.constant dense<1> : tensor<1xi32>
  %40 = stablehlo.dynamic_reshape %38, %39 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %41 = stablehlo.get_dimension_size %37, dim = 1 : (tensor<1x128x?xf32>) -> tensor<i32>
  %42 = stablehlo.constant dense<1> : tensor<1xi32>
  %43 = stablehlo.dynamic_reshape %41, %42 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %44 = stablehlo.get_dimension_size %37, dim = 2 : (tensor<1x128x?xf32>) -> tensor<i32>
  %45 = stablehlo.constant dense<1> : tensor<1xi32>
  %46 = stablehlo.dynamic_reshape %44, %45 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %47 = stablehlo.concatenate %40, %43, %46, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %48 = stablehlo.constant dense<1024> : tensor<2xi32>
  %49 = stablehlo.dynamic_iota %48, dim = 0 : (tensor<2xi32>) -> tensor<?x?xf32>
  %50 = stablehlo.constant dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B0000008C0000008D0000008E0000008F000000900000009100000092000000930000009400000095000000960000009700000098000000990000009A0000009B0000009C0000009D0000009E0000009F000000A0000000A1000000A2000000A3000000A4000000A5000000A6000000A7000000A8000000A9000000AA000000AB000000AC000000AD000000AE000000AF000000B0000000B1000000B2000000B3000000B4000000B5000000B6000000B7000000B8000000B9000000BA000000BB000000BC000000BD000000BE000000BF000000C0000000C1000000C2000000C3000000C4000000C5000000C6000000C7000000C8000000C9000000CA000000CB000000CC000000CD000000CE000000CF000000D0000000D1000000D2000000D3000000D4000000D5000000D6000000D7000000D8000000D9000000DA000000DB000000DC000000DD000000DE000000DF000000E0000000E1000000E2000000E3000000E4000000E5000000E6000000E7000000E8000000E9000000EA000000EB000000EC000000ED000000EE000000EF000000F0000000F1000000F2000000F3000000F4000000F5000000F6000000F7000000F8000000F9000000FA000000FB000000FC000000FD000000FE000000FF000000000100000101000002010000030100000401000005010000060100000701000008010000090100000A0100000B0100000C0100000D0100000E0100000F010000100100001101000012010000130100001401000015010000160100001701000018010000190100001A0100001B0100001C0100001D0100001E0100001F010000200100002101000022010000230100002401000025010000260100002701000028010000290100002A0100002B0100002C0100002D0100002E0100002F010000300100003101000032010000330100003401000035010000360100003701000038010000390100003A0100003B0100003C0100003D0100003E0100003F010000400100004101000042010000430100004401000045010000460100004701000048010000490100004A0100004B0100004C0100004D0100004E0100004F010000500100005101000052010000530100005401000055010000560100005701000058010000590100005A0100005B0100005C0100005D0100005E0100005F010000600100006101000062010000630100006401000065010000660100006701000068010000690100006A0100006B0100006C0100006D0100006E0100006F010000700100007101000072010000730100007401000075010000760100007701000078010000790100007A0100007B0100007C0100007D0100007E0100007F010000800100008101000082010000830100008401000085010000860100008701000088010000890100008A0100008B0100008C0100008D0100008E0100008F010000900100009101000092010000930100009401000095010000960100009701000098010000990100009A0100009B0100009C0100009D0100009E0100009F010000A0010000A1010000A2010000A3010000A4010000A5010000A6010000A7010000A8010000A9010000AA010000AB010000AC010000AD010000AE010000AF010000B0010000B1010000B2010000B3010000B4010000B5010000B6010000B7010000B8010000B9010000BA010000BB010000BC010000BD010000BE010000BF010000C0010000C1010000C2010000C3010000C4010000C5010000C6010000C7010000C8010000C9010000CA010000CB010000CC010000CD010000CE010000CF010000D0010000D1010000D2010000D3010000D4010000D5010000D6010000D7010000D8010000D9010000DA010000DB010000DC010000DD010000DE010000DF010000E0010000E1010000E2010000E3010000E4010000E5010000E6010000E7010000E8010000E9010000EA010000EB010000EC010000ED010000EE010000EF010000F0010000F1010000F2010000F3010000F4010000F5010000F6010000F7010000F8010000F9010000FA010000FB010000FC010000FD010000FE010000FF010000000200000102000002020000030200000402000005020000060200000702000008020000090200000A0200000B0200000C0200000D0200000E0200000F020000100200001102000012020000130200001402000015020000160200001702000018020000190200001A0200001B0200001C0200001D0200001E0200001F020000200200002102000022020000230200002402000025020000260200002702000028020000290200002A0200002B0200002C0200002D0200002E0200002F020000300200003102000032020000330200003402000035020000360200003702000038020000390200003A0200003B0200003C0200003D0200003E0200003F020000400200004102000042020000430200004402000045020000460200004702000048020000490200004A0200004B0200004C0200004D0200004E0200004F020000500200005102000052020000530200005402000055020000560200005702000058020000590200005A0200005B0200005C0200005D0200005E0200005F020000600200006102000062020000630200006402000065020000660200006702000068020000690200006A0200006B0200006C0200006D0200006E0200006F020000700200007102000072020000730200007402000075020000760200007702000078020000790200007A0200007B0200007C0200007D0200007E0200007F020000800200008102000082020000830200008402000085020000860200008702000088020000890200008A0200008B0200008C0200008D0200008E0200008F020000900200009102000092020000930200009402000095020000960200009702000098020000990200009A0200009B0200009C0200009D0200009E0200009F020000A0020000A1020000A2020000A3020000A4020000A5020000A6020000A7020000A8020000A9020000AA020000AB020000AC020000AD020000AE020000AF020000B0020000B1020000B2020000B3020000B4020000B5020000B6020000B7020000B8020000B9020000BA020000BB020000BC020000BD020000BE020000BF020000C0020000C1020000C2020000C3020000C4020000C5020000C6020000C7020000C8020000C9020000CA020000CB020000CC020000CD020000CE020000CF020000D0020000D1020000D2020000D3020000D4020000D5020000D6020000D7020000D8020000D9020000DA020000DB020000DC020000DD020000DE020000DF020000E0020000E1020000E2020000E3020000E4020000E5020000E6020000E7020000E8020000E9020000EA020000EB020000EC020000ED020000EE020000EF020000F0020000F1020000F2020000F3020000F4020000F5020000F6020000F7020000F8020000F9020000FA020000FB020000FC020000FD020000FE020000FF020000000300000103000002030000030300000403000005030000060300000703000008030000090300000A0300000B0300000C0300000D0300000E0300000F030000100300001103000012030000130300001403000015030000160300001703000018030000190300001A0300001B0300001C0300001D0300001E0300001F030000200300002103000022030000230300002403000025030000260300002703000028030000290300002A0300002B0300002C0300002D0300002E0300002F030000300300003103000032030000330300003403000035030000360300003703000038030000390300003A0300003B0300003C0300003D0300003E0300003F030000400300004103000042030000430300004403000045030000460300004703000048030000490300004A0300004B0300004C0300004D0300004E0300004F030000500300005103000052030000530300005403000055030000560300005703000058030000590300005A0300005B0300005C0300005D0300005E0300005F030000600300006103000062030000630300006403000065030000660300006703000068030000690300006A0300006B0300006C0300006D0300006E0300006F030000700300007103000072030000730300007403000075030000760300007703000078030000790300007A0300007B0300007C0300007D0300007E0300007F030000800300008103000082030000830300008403000085030000860300008703000088030000890300008A0300008B0300008C0300008D0300008E0300008F030000900300009103000092030000930300009403000095030000960300009703000098030000990300009A0300009B0300009C0300009D0300009E0300009F030000A0030000A1030000A2030000A3030000A4030000A5030000A6030000A7030000A8030000A9030000AA030000AB030000AC030000AD030000AE030000AF030000B0030000B1030000B2030000B3030000B4030000B5030000B6030000B7030000B8030000B9030000BA030000BB030000BC030000BD030000BE030000BF030000C0030000C1030000C2030000C3030000C4030000C5030000C6030000C7030000C8030000C9030000CA030000CB030000CC030000CD030000CE030000CF030000D0030000D1030000D2030000D3030000D4030000D5030000D6030000D7030000D8030000D9030000DA030000DB030000DC030000DD030000DE030000DF030000E0030000E1030000E2030000E3030000E4030000E5030000E6030000E7030000E8030000E9030000EA030000EB030000EC030000ED030000EE030000EF030000F0030000F1030000F2030000F3030000F4030000F5030000F6030000F7030000F8030000F9030000FA030000FB030000FC030000FD030000FE030000FF030000"> : tensor<1x1024xi32>
  %51 = stablehlo.constant dense<0> : tensor<i32>
  %52 = stablehlo.reshape %51 : (tensor<i32>) -> tensor<1xi32>
  %53 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<1x128xi32>) -> tensor<i32>
  %54 = stablehlo.constant dense<1> : tensor<1xi32>
  %55 = stablehlo.dynamic_reshape %53, %54 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %56 = stablehlo.get_dimension_size %24, dim = 1 : (tensor<1x128xi32>) -> tensor<i32>
  %57 = stablehlo.constant dense<1> : tensor<1xi32>
  %58 = stablehlo.dynamic_reshape %56, %57 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %59 = stablehlo.concatenate %55, %58, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %60 = stablehlo.constant dense<1> : tensor<i32>
  %61 = stablehlo.reshape %60 : (tensor<i32>) -> tensor<1xi32>
  %62 = stablehlo.concatenate %61, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %63 = stablehlo.constant dense<2> : tensor<i32>
  %64 = stablehlo.reshape %63 : (tensor<i32>) -> tensor<1xi32>
  %65 = stablehlo.get_dimension_size %59, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %66 = stablehlo.constant dense<1> : tensor<1xi32>
  %67 = stablehlo.dynamic_reshape %65, %66 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %68 = stablehlo.concatenate %67, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %69 = stablehlo.constant dense<0> : tensor<1xi32>
  %70 = stablehlo.constant dense<1> : tensor<1xi32>
  %71 = stablehlo.constant dense<1> : tensor<1xi32>
  %72 = stablehlo.real_dynamic_slice %68, %69, %70, %71 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %73 = stablehlo.minimum %64, %72 : tensor<1xi32>
  %74 = stablehlo.concatenate %73, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %75 = stablehlo.constant dense<1> : tensor<i32>
  %76 = stablehlo.reshape %75 : (tensor<i32>) -> tensor<1xi32>
  %77 = stablehlo.concatenate %76, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %78 = stablehlo.real_dynamic_slice %59, %62, %74, %77 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %79 = stablehlo.constant dense<> : tensor<0xi32>
  %80 = stablehlo.dynamic_reshape %78, %79 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %81 = stablehlo.constant dense<> : tensor<0xi32>
  %82 = stablehlo.constant dense<> : tensor<0xi32>
  %83 = stablehlo.maximum %81, %82 : tensor<0xi32>
  %84 = stablehlo.dynamic_broadcast_in_dim %80, %83, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %85 = stablehlo.constant dense<0> : tensor<i32>
  %86 = stablehlo.dynamic_broadcast_in_dim %85, %83, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %87 = stablehlo.compare  GE, %84, %86 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %88 = stablehlo.constant dense<1> : tensor<i32>
  %89 = stablehlo.constant dense<1> : tensor<1xi32>
  %90 = stablehlo.dynamic_broadcast_in_dim %88, %89, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %91 = stablehlo.constant dense<> : tensor<0xi32>
  %92 = stablehlo.concatenate %90, %91, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %93 = stablehlo.dynamic_broadcast_in_dim %87, %92, dims = [] : (tensor<i1>, tensor<1xi32>) -> tensor<?xi1>
  %94 = stablehlo.get_dimension_size %93, dim = 0 : (tensor<?xi1>) -> tensor<i32>
  %95 = stablehlo.constant dense<1> : tensor<1xi32>
  %96 = stablehlo.dynamic_reshape %94, %95 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %97 = stablehlo.concatenate %96, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %98 = stablehlo.constant dense<1> : tensor<i32>
  %99 = stablehlo.constant dense<1> : tensor<1xi32>
  %100 = stablehlo.dynamic_broadcast_in_dim %98, %99, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %101 = stablehlo.constant dense<> : tensor<0xi32>
  %102 = stablehlo.concatenate %100, %101, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %103 = stablehlo.dynamic_broadcast_in_dim %80, %102, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %104 = stablehlo.get_dimension_size %103, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %105 = stablehlo.constant dense<1> : tensor<1xi32>
  %106 = stablehlo.dynamic_reshape %104, %105 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %107 = stablehlo.concatenate %106, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %108 = stablehlo.maximum %97, %107 : tensor<1xi32>
  %109 = stablehlo.get_dimension_size %103, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %110 = stablehlo.constant dense<1> : tensor<1xi32>
  %111 = stablehlo.dynamic_reshape %109, %110 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %112 = stablehlo.concatenate %111, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %113 = stablehlo.get_dimension_size %50, dim = 0 : (tensor<1x1024xi32>) -> tensor<i32>
  %114 = stablehlo.constant dense<1> : tensor<1xi32>
  %115 = stablehlo.dynamic_reshape %113, %114 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %116 = stablehlo.get_dimension_size %50, dim = 1 : (tensor<1x1024xi32>) -> tensor<i32>
  %117 = stablehlo.constant dense<1> : tensor<1xi32>
  %118 = stablehlo.dynamic_reshape %116, %117 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %119 = stablehlo.concatenate %115, %118, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %120 = stablehlo.constant dense<1> : tensor<i32>
  %121 = stablehlo.reshape %120 : (tensor<i32>) -> tensor<1xi32>
  %122 = stablehlo.concatenate %121, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %123 = stablehlo.constant dense<2> : tensor<i32>
  %124 = stablehlo.reshape %123 : (tensor<i32>) -> tensor<1xi32>
  %125 = stablehlo.get_dimension_size %119, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %126 = stablehlo.constant dense<1> : tensor<1xi32>
  %127 = stablehlo.dynamic_reshape %125, %126 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %128 = stablehlo.concatenate %127, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %129 = stablehlo.constant dense<0> : tensor<1xi32>
  %130 = stablehlo.constant dense<1> : tensor<1xi32>
  %131 = stablehlo.constant dense<1> : tensor<1xi32>
  %132 = stablehlo.real_dynamic_slice %128, %129, %130, %131 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %133 = stablehlo.minimum %124, %132 : tensor<1xi32>
  %134 = stablehlo.concatenate %133, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %135 = stablehlo.constant dense<1> : tensor<i32>
  %136 = stablehlo.reshape %135 : (tensor<i32>) -> tensor<1xi32>
  %137 = stablehlo.concatenate %136, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %138 = stablehlo.real_dynamic_slice %119, %122, %134, %137 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %139 = stablehlo.constant dense<> : tensor<0xi32>
  %140 = stablehlo.dynamic_reshape %138, %139 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %141 = stablehlo.constant dense<1> : tensor<1xi32>
  %142 = stablehlo.dynamic_reshape %140, %141 : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %143 = stablehlo.get_dimension_size %142, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %144 = stablehlo.constant dense<1> : tensor<1xi32>
  %145 = stablehlo.dynamic_reshape %143, %144 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %146 = stablehlo.concatenate %145, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %147 = stablehlo.constant dense<1> : tensor<i32>
  %148 = stablehlo.constant dense<1> : tensor<1xi32>
  %149 = stablehlo.dynamic_broadcast_in_dim %147, %148, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %150 = stablehlo.constant dense<> : tensor<0xi32>
  %151 = stablehlo.concatenate %149, %150, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %152 = stablehlo.dynamic_broadcast_in_dim %80, %151, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %153 = stablehlo.get_dimension_size %152, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %154 = stablehlo.constant dense<1> : tensor<1xi32>
  %155 = stablehlo.dynamic_reshape %153, %154 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %156 = stablehlo.concatenate %155, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %157 = stablehlo.maximum %146, %156 : tensor<1xi32>
  %158 = stablehlo.dynamic_broadcast_in_dim %142, %157, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %159 = stablehlo.dynamic_broadcast_in_dim %152, %157, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %160 = stablehlo.add %158, %159 : tensor<?xi32>
  %161 = stablehlo.get_dimension_size %160, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %162 = stablehlo.constant dense<1> : tensor<1xi32>
  %163 = stablehlo.dynamic_reshape %161, %162 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %164 = stablehlo.concatenate %163, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %165 = stablehlo.maximum %112, %164 : tensor<1xi32>
  %166 = stablehlo.maximum %108, %165 : tensor<1xi32>
  %167 = stablehlo.dynamic_broadcast_in_dim %93, %166, dims = [0] : (tensor<?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %168 = stablehlo.dynamic_broadcast_in_dim %103, %166, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %169 = stablehlo.dynamic_broadcast_in_dim %160, %166, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %170 = stablehlo.select %167, %168, %169 : tensor<?xi1>, tensor<?xi32>
  %171 = stablehlo.reshape %170 : (tensor<?xi32>) -> tensor<1xi32>
  %172 = stablehlo.concatenate %52, %171, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %173 = stablehlo.constant dense<1> : tensor<i32>
  %174 = stablehlo.reshape %173 : (tensor<i32>) -> tensor<1xi32>
  %175 = stablehlo.get_dimension_size %50, dim = 0 : (tensor<1x1024xi32>) -> tensor<i32>
  %176 = stablehlo.constant dense<1> : tensor<1xi32>
  %177 = stablehlo.dynamic_reshape %175, %176 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %178 = stablehlo.get_dimension_size %50, dim = 1 : (tensor<1x1024xi32>) -> tensor<i32>
  %179 = stablehlo.constant dense<1> : tensor<1xi32>
  %180 = stablehlo.dynamic_reshape %178, %179 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %181 = stablehlo.concatenate %177, %180, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %182 = stablehlo.constant dense<0> : tensor<1xi32>
  %183 = stablehlo.constant dense<1> : tensor<1xi32>
  %184 = stablehlo.constant dense<1> : tensor<1xi32>
  %185 = stablehlo.real_dynamic_slice %181, %182, %183, %184 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %186 = stablehlo.minimum %174, %185 : tensor<1xi32>
  %187 = stablehlo.constant dense<> : tensor<0xi32>
  %188 = stablehlo.constant dense<> : tensor<0xi32>
  %189 = stablehlo.maximum %187, %188 : tensor<0xi32>
  %190 = stablehlo.dynamic_broadcast_in_dim %80, %189, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %191 = stablehlo.constant dense<0> : tensor<i32>
  %192 = stablehlo.dynamic_broadcast_in_dim %191, %189, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %193 = stablehlo.compare  GE, %190, %192 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %194 = stablehlo.constant dense<1> : tensor<i32>
  %195 = stablehlo.constant dense<1> : tensor<1xi32>
  %196 = stablehlo.dynamic_broadcast_in_dim %194, %195, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %197 = stablehlo.constant dense<> : tensor<0xi32>
  %198 = stablehlo.concatenate %196, %197, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %199 = stablehlo.dynamic_broadcast_in_dim %193, %198, dims = [] : (tensor<i1>, tensor<1xi32>) -> tensor<?xi1>
  %200 = stablehlo.get_dimension_size %199, dim = 0 : (tensor<?xi1>) -> tensor<i32>
  %201 = stablehlo.constant dense<1> : tensor<1xi32>
  %202 = stablehlo.dynamic_reshape %200, %201 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %203 = stablehlo.concatenate %202, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %204 = stablehlo.constant dense<1> : tensor<i32>
  %205 = stablehlo.constant dense<1> : tensor<1xi32>
  %206 = stablehlo.dynamic_broadcast_in_dim %204, %205, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %207 = stablehlo.constant dense<> : tensor<0xi32>
  %208 = stablehlo.concatenate %206, %207, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %209 = stablehlo.dynamic_broadcast_in_dim %80, %208, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %210 = stablehlo.get_dimension_size %209, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %211 = stablehlo.constant dense<1> : tensor<1xi32>
  %212 = stablehlo.dynamic_reshape %210, %211 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %213 = stablehlo.concatenate %212, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %214 = stablehlo.maximum %203, %213 : tensor<1xi32>
  %215 = stablehlo.get_dimension_size %209, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %216 = stablehlo.constant dense<1> : tensor<1xi32>
  %217 = stablehlo.dynamic_reshape %215, %216 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %218 = stablehlo.concatenate %217, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %219 = stablehlo.constant dense<1> : tensor<i32>
  %220 = stablehlo.reshape %219 : (tensor<i32>) -> tensor<1xi32>
  %221 = stablehlo.concatenate %220, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %222 = stablehlo.constant dense<2> : tensor<i32>
  %223 = stablehlo.reshape %222 : (tensor<i32>) -> tensor<1xi32>
  %224 = stablehlo.get_dimension_size %119, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %225 = stablehlo.constant dense<1> : tensor<1xi32>
  %226 = stablehlo.dynamic_reshape %224, %225 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %227 = stablehlo.concatenate %226, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %228 = stablehlo.constant dense<0> : tensor<1xi32>
  %229 = stablehlo.constant dense<1> : tensor<1xi32>
  %230 = stablehlo.constant dense<1> : tensor<1xi32>
  %231 = stablehlo.real_dynamic_slice %227, %228, %229, %230 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %232 = stablehlo.minimum %223, %231 : tensor<1xi32>
  %233 = stablehlo.concatenate %232, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %234 = stablehlo.constant dense<1> : tensor<i32>
  %235 = stablehlo.reshape %234 : (tensor<i32>) -> tensor<1xi32>
  %236 = stablehlo.concatenate %235, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %237 = stablehlo.real_dynamic_slice %119, %221, %233, %236 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %238 = stablehlo.constant dense<> : tensor<0xi32>
  %239 = stablehlo.dynamic_reshape %237, %238 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %240 = stablehlo.constant dense<1> : tensor<1xi32>
  %241 = stablehlo.dynamic_reshape %239, %240 : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %242 = stablehlo.get_dimension_size %241, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %243 = stablehlo.constant dense<1> : tensor<1xi32>
  %244 = stablehlo.dynamic_reshape %242, %243 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %245 = stablehlo.concatenate %244, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %246 = stablehlo.constant dense<1> : tensor<i32>
  %247 = stablehlo.constant dense<1> : tensor<1xi32>
  %248 = stablehlo.dynamic_broadcast_in_dim %246, %247, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %249 = stablehlo.constant dense<> : tensor<0xi32>
  %250 = stablehlo.concatenate %248, %249, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %251 = stablehlo.dynamic_broadcast_in_dim %80, %250, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %252 = stablehlo.get_dimension_size %251, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %253 = stablehlo.constant dense<1> : tensor<1xi32>
  %254 = stablehlo.dynamic_reshape %252, %253 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %255 = stablehlo.concatenate %254, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %256 = stablehlo.maximum %245, %255 : tensor<1xi32>
  %257 = stablehlo.dynamic_broadcast_in_dim %241, %256, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %258 = stablehlo.dynamic_broadcast_in_dim %251, %256, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %259 = stablehlo.add %257, %258 : tensor<?xi32>
  %260 = stablehlo.get_dimension_size %259, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %261 = stablehlo.constant dense<1> : tensor<1xi32>
  %262 = stablehlo.dynamic_reshape %260, %261 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %263 = stablehlo.concatenate %262, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %264 = stablehlo.maximum %218, %263 : tensor<1xi32>
  %265 = stablehlo.maximum %214, %264 : tensor<1xi32>
  %266 = stablehlo.dynamic_broadcast_in_dim %199, %265, dims = [0] : (tensor<?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %267 = stablehlo.dynamic_broadcast_in_dim %209, %265, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %268 = stablehlo.dynamic_broadcast_in_dim %259, %265, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %269 = stablehlo.select %266, %267, %268 : tensor<?xi1>, tensor<?xi32>
  %270 = stablehlo.get_dimension_size %269, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %271 = stablehlo.constant dense<1> : tensor<1xi32>
  %272 = stablehlo.dynamic_reshape %270, %271 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %273 = stablehlo.concatenate %272, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %274 = stablehlo.constant dense<1> : tensor<i32>
  %275 = stablehlo.constant dense<1> : tensor<i32>
  %276 = stablehlo.constant dense<1> : tensor<1xi32>
  %277 = stablehlo.dynamic_broadcast_in_dim %275, %276, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %278 = stablehlo.constant dense<> : tensor<0xi32>
  %279 = stablehlo.concatenate %277, %278, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %280 = stablehlo.dynamic_broadcast_in_dim %274, %279, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %281 = stablehlo.get_dimension_size %280, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %282 = stablehlo.constant dense<1> : tensor<1xi32>
  %283 = stablehlo.dynamic_reshape %281, %282 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %284 = stablehlo.concatenate %283, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %285 = stablehlo.maximum %273, %284 : tensor<1xi32>
  %286 = stablehlo.dynamic_broadcast_in_dim %269, %285, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %287 = stablehlo.dynamic_broadcast_in_dim %280, %285, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %288 = stablehlo.add %286, %287 : tensor<?xi32>
  %289 = stablehlo.reshape %288 : (tensor<?xi32>) -> tensor<1xi32>
  %290 = stablehlo.constant dense<1> : tensor<1xi32>
  %291 = stablehlo.constant dense<2> : tensor<1xi32>
  %292 = stablehlo.constant dense<1> : tensor<1xi32>
  %293 = stablehlo.real_dynamic_slice %181, %290, %291, %292 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %294 = stablehlo.minimum %289, %293 : tensor<1xi32>
  %295 = stablehlo.concatenate %186, %294, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %296 = stablehlo.constant dense<1> : tensor<i32>
  %297 = stablehlo.reshape %296 : (tensor<i32>) -> tensor<1xi32>
  %298 = stablehlo.constant dense<1> : tensor<i32>
  %299 = stablehlo.reshape %298 : (tensor<i32>) -> tensor<1xi32>
  %300 = stablehlo.concatenate %297, %299, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %301 = stablehlo.real_dynamic_slice %50, %172, %295, %300 : (tensor<1x1024xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %302 = stablehlo.constant dense<1> : tensor<1xi32>
  %303 = stablehlo.get_dimension_size %49, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %304 = stablehlo.constant dense<1> : tensor<1xi32>
  %305 = stablehlo.dynamic_reshape %303, %304 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %306 = stablehlo.get_dimension_size %49, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %307 = stablehlo.constant dense<1> : tensor<1xi32>
  %308 = stablehlo.dynamic_reshape %306, %307 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %309 = stablehlo.concatenate %305, %308, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %310 = stablehlo.constant dense<1> : tensor<1xi32>
  %311 = stablehlo.constant dense<2> : tensor<1xi32>
  %312 = stablehlo.real_dynamic_slice %309, %310, %311, %302 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %313 = stablehlo.concatenate %302, %312, dim = 0 : (tensor<1xi32>, tensor<?xi32>) -> tensor<2xi32>
  %314 = "stablehlo.dynamic_gather"(%49, %301, %313) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %315 = stablehlo.get_dimension_size %314, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %316 = stablehlo.constant dense<1> : tensor<1xi32>
  %317 = stablehlo.dynamic_reshape %315, %316 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %318 = stablehlo.get_dimension_size %314, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %319 = stablehlo.constant dense<1> : tensor<1xi32>
  %320 = stablehlo.dynamic_reshape %318, %319 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %321 = stablehlo.get_dimension_size %314, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %322 = stablehlo.constant dense<1> : tensor<1xi32>
  %323 = stablehlo.dynamic_reshape %321, %322 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %324 = stablehlo.concatenate %317, %320, %323, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %325 = stablehlo.maximum %47, %324 : tensor<3xi32>
  %326 = stablehlo.dynamic_broadcast_in_dim %37, %325, dims = [0, 1, 2] : (tensor<1x128x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %327 = stablehlo.dynamic_broadcast_in_dim %314, %325, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %328 = stablehlo.add %326, %327 : tensor<?x?x?xf32>
  %329 = stablehlo.get_dimension_size %328, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %330 = stablehlo.constant dense<1> : tensor<1xi32>
  %331 = stablehlo.dynamic_reshape %329, %330 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %332 = stablehlo.get_dimension_size %328, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %333 = stablehlo.constant dense<1> : tensor<1xi32>
  %334 = stablehlo.dynamic_reshape %332, %333 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %335 = stablehlo.get_dimension_size %328, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %336 = stablehlo.constant dense<1> : tensor<1xi32>
  %337 = stablehlo.dynamic_reshape %335, %336 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %338 = stablehlo.concatenate %331, %334, %337, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %339 = stablehlo.constant dense<[2, 1024]> : tensor<2xi32>
  %340 = stablehlo.dynamic_iota %339, dim = 0 : (tensor<2xi32>) -> tensor<?x?xf32>
  %341 = stablehlo.constant dense<1> : tensor<1x128xi32>
  %342 = stablehlo.constant dense<1> : tensor<1xi32>
  %343 = stablehlo.get_dimension_size %340, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %344 = stablehlo.constant dense<1> : tensor<1xi32>
  %345 = stablehlo.dynamic_reshape %343, %344 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %346 = stablehlo.get_dimension_size %340, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %347 = stablehlo.constant dense<1> : tensor<1xi32>
  %348 = stablehlo.dynamic_reshape %346, %347 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %349 = stablehlo.concatenate %345, %348, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %350 = stablehlo.constant dense<1> : tensor<1xi32>
  %351 = stablehlo.constant dense<2> : tensor<1xi32>
  %352 = stablehlo.real_dynamic_slice %349, %350, %351, %342 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %353 = stablehlo.concatenate %342, %352, dim = 0 : (tensor<1xi32>, tensor<?xi32>) -> tensor<2xi32>
  %354 = "stablehlo.dynamic_gather"(%340, %341, %353) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>} : (tensor<?x?xf32>, tensor<1x128xi32>, tensor<2xi32>) -> tensor<1x128x?xf32>
  %355 = stablehlo.get_dimension_size %354, dim = 0 : (tensor<1x128x?xf32>) -> tensor<i32>
  %356 = stablehlo.constant dense<1> : tensor<1xi32>
  %357 = stablehlo.dynamic_reshape %355, %356 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %358 = stablehlo.get_dimension_size %354, dim = 1 : (tensor<1x128x?xf32>) -> tensor<i32>
  %359 = stablehlo.constant dense<1> : tensor<1xi32>
  %360 = stablehlo.dynamic_reshape %358, %359 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %361 = stablehlo.get_dimension_size %354, dim = 2 : (tensor<1x128x?xf32>) -> tensor<i32>
  %362 = stablehlo.constant dense<1> : tensor<1xi32>
  %363 = stablehlo.dynamic_reshape %361, %362 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %364 = stablehlo.concatenate %357, %360, %363, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %365 = stablehlo.maximum %338, %364 : tensor<3xi32>
  %366 = stablehlo.dynamic_broadcast_in_dim %328, %365, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %367 = stablehlo.dynamic_broadcast_in_dim %354, %365, dims = [0, 1, 2] : (tensor<1x128x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %368 = stablehlo.add %366, %367 : tensor<?x?x?xf32>
  %369 = stablehlo.get_dimension_size %368, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %370 = stablehlo.constant dense<1> : tensor<1xi32>
  %371 = stablehlo.dynamic_reshape %369, %370 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %372 = stablehlo.get_dimension_size %368, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %373 = stablehlo.constant dense<1> : tensor<1xi32>
  %374 = stablehlo.dynamic_reshape %372, %373 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %375 = stablehlo.get_dimension_size %368, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %376 = stablehlo.constant dense<1> : tensor<1xi32>
  %377 = stablehlo.dynamic_reshape %375, %376 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %378 = stablehlo.concatenate %371, %374, %377, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %379 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %380 = stablehlo.reduce(%368 init: %379) applies stablehlo.add across dimensions = [2] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  %381 = stablehlo.get_dimension_size %380, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %382 = stablehlo.constant dense<1> : tensor<1xi32>
  %383 = stablehlo.dynamic_reshape %381, %382 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %384 = stablehlo.get_dimension_size %380, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %385 = stablehlo.constant dense<1> : tensor<1xi32>
  %386 = stablehlo.dynamic_reshape %384, %385 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %387 = stablehlo.concatenate %383, %386, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %388 = stablehlo.constant dense<0> : tensor<i32>
  %389 = stablehlo.reshape %388 : (tensor<i32>) -> tensor<1xi32>
  %390 = stablehlo.concatenate %389, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %391 = stablehlo.constant dense<2> : tensor<i32>
  %392 = stablehlo.reshape %391 : (tensor<i32>) -> tensor<1xi32>
  %393 = stablehlo.get_dimension_size %387, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %394 = stablehlo.constant dense<1> : tensor<1xi32>
  %395 = stablehlo.dynamic_reshape %393, %394 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %396 = stablehlo.concatenate %395, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %397 = stablehlo.constant dense<0> : tensor<1xi32>
  %398 = stablehlo.constant dense<1> : tensor<1xi32>
  %399 = stablehlo.constant dense<1> : tensor<1xi32>
  %400 = stablehlo.real_dynamic_slice %396, %397, %398, %399 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %401 = stablehlo.minimum %392, %400 : tensor<1xi32>
  %402 = stablehlo.concatenate %401, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %403 = stablehlo.constant dense<1> : tensor<i32>
  %404 = stablehlo.reshape %403 : (tensor<i32>) -> tensor<1xi32>
  %405 = stablehlo.concatenate %404, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %406 = stablehlo.real_dynamic_slice %387, %390, %402, %405 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %407 = stablehlo.constant dense<1> : tensor<1xi32>
  %408 = stablehlo.constant dense<2> : tensor<i32>
  %409 = stablehlo.reshape %408 : (tensor<i32>) -> tensor<1xi32>
  %410 = stablehlo.concatenate %409, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %411 = stablehlo.get_dimension_size %387, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %412 = stablehlo.constant dense<1> : tensor<1xi32>
  %413 = stablehlo.dynamic_reshape %411, %412 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %414 = stablehlo.concatenate %413, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %415 = stablehlo.constant dense<0> : tensor<i32>
  %416 = stablehlo.reshape %415 : (tensor<i32>) -> tensor<1xi32>
  %417 = stablehlo.concatenate %416, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %418 = stablehlo.constant dense<1> : tensor<i32>
  %419 = stablehlo.reshape %418 : (tensor<i32>) -> tensor<1xi32>
  %420 = stablehlo.get_dimension_size %414, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %421 = stablehlo.constant dense<1> : tensor<1xi32>
  %422 = stablehlo.dynamic_reshape %420, %421 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %423 = stablehlo.concatenate %422, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %424 = stablehlo.constant dense<0> : tensor<1xi32>
  %425 = stablehlo.constant dense<1> : tensor<1xi32>
  %426 = stablehlo.constant dense<1> : tensor<1xi32>
  %427 = stablehlo.real_dynamic_slice %423, %424, %425, %426 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %428 = stablehlo.minimum %419, %427 : tensor<1xi32>
  %429 = stablehlo.concatenate %428, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %430 = stablehlo.constant dense<1> : tensor<i32>
  %431 = stablehlo.reshape %430 : (tensor<i32>) -> tensor<1xi32>
  %432 = stablehlo.concatenate %431, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %433 = stablehlo.real_dynamic_slice %414, %417, %429, %432 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %434 = stablehlo.constant dense<> : tensor<0xi32>
  %435 = stablehlo.dynamic_reshape %433, %434 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %436 = stablehlo.constant dense<> : tensor<0xi32>
  %437 = stablehlo.constant dense<> : tensor<0xi32>
  %438 = stablehlo.maximum %436, %437 : tensor<0xi32>
  %439 = stablehlo.dynamic_broadcast_in_dim %435, %438, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %440 = stablehlo.constant dense<0> : tensor<i32>
  %441 = stablehlo.dynamic_broadcast_in_dim %440, %438, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %442 = stablehlo.compare  GE, %439, %441 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %443 = stablehlo.constant dense<1> : tensor<i32>
  %444 = stablehlo.constant dense<1> : tensor<1xi32>
  %445 = stablehlo.dynamic_broadcast_in_dim %443, %444, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %446 = stablehlo.constant dense<> : tensor<0xi32>
  %447 = stablehlo.concatenate %445, %446, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %448 = stablehlo.dynamic_broadcast_in_dim %442, %447, dims = [] : (tensor<i1>, tensor<1xi32>) -> tensor<?xi1>
  %449 = stablehlo.get_dimension_size %448, dim = 0 : (tensor<?xi1>) -> tensor<i32>
  %450 = stablehlo.constant dense<1> : tensor<1xi32>
  %451 = stablehlo.dynamic_reshape %449, %450 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %452 = stablehlo.concatenate %451, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %453 = stablehlo.constant dense<1> : tensor<i32>
  %454 = stablehlo.constant dense<1> : tensor<1xi32>
  %455 = stablehlo.dynamic_broadcast_in_dim %453, %454, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %456 = stablehlo.constant dense<> : tensor<0xi32>
  %457 = stablehlo.concatenate %455, %456, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %458 = stablehlo.dynamic_broadcast_in_dim %435, %457, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %459 = stablehlo.get_dimension_size %458, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %460 = stablehlo.constant dense<1> : tensor<1xi32>
  %461 = stablehlo.dynamic_reshape %459, %460 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %462 = stablehlo.concatenate %461, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %463 = stablehlo.maximum %452, %462 : tensor<1xi32>
  %464 = stablehlo.get_dimension_size %458, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %465 = stablehlo.constant dense<1> : tensor<1xi32>
  %466 = stablehlo.dynamic_reshape %464, %465 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %467 = stablehlo.concatenate %466, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %468 = stablehlo.constant dense<0> : tensor<i32>
  %469 = stablehlo.reshape %468 : (tensor<i32>) -> tensor<1xi32>
  %470 = stablehlo.concatenate %469, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %471 = stablehlo.constant dense<1> : tensor<i32>
  %472 = stablehlo.reshape %471 : (tensor<i32>) -> tensor<1xi32>
  %473 = stablehlo.get_dimension_size %414, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %474 = stablehlo.constant dense<1> : tensor<1xi32>
  %475 = stablehlo.dynamic_reshape %473, %474 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %476 = stablehlo.concatenate %475, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %477 = stablehlo.constant dense<0> : tensor<1xi32>
  %478 = stablehlo.constant dense<1> : tensor<1xi32>
  %479 = stablehlo.constant dense<1> : tensor<1xi32>
  %480 = stablehlo.real_dynamic_slice %476, %477, %478, %479 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %481 = stablehlo.minimum %472, %480 : tensor<1xi32>
  %482 = stablehlo.concatenate %481, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %483 = stablehlo.constant dense<1> : tensor<i32>
  %484 = stablehlo.reshape %483 : (tensor<i32>) -> tensor<1xi32>
  %485 = stablehlo.concatenate %484, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %486 = stablehlo.real_dynamic_slice %414, %470, %482, %485 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %487 = stablehlo.constant dense<> : tensor<0xi32>
  %488 = stablehlo.dynamic_reshape %486, %487 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %489 = stablehlo.constant dense<1> : tensor<1xi32>
  %490 = stablehlo.dynamic_reshape %488, %489 : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %491 = stablehlo.get_dimension_size %490, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %492 = stablehlo.constant dense<1> : tensor<1xi32>
  %493 = stablehlo.dynamic_reshape %491, %492 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %494 = stablehlo.concatenate %493, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %495 = stablehlo.constant dense<1> : tensor<i32>
  %496 = stablehlo.constant dense<1> : tensor<1xi32>
  %497 = stablehlo.dynamic_broadcast_in_dim %495, %496, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %498 = stablehlo.constant dense<> : tensor<0xi32>
  %499 = stablehlo.concatenate %497, %498, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %500 = stablehlo.dynamic_broadcast_in_dim %435, %499, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %501 = stablehlo.get_dimension_size %500, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %502 = stablehlo.constant dense<1> : tensor<1xi32>
  %503 = stablehlo.dynamic_reshape %501, %502 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %504 = stablehlo.concatenate %503, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %505 = stablehlo.maximum %494, %504 : tensor<1xi32>
  %506 = stablehlo.dynamic_broadcast_in_dim %490, %505, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %507 = stablehlo.dynamic_broadcast_in_dim %500, %505, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %508 = stablehlo.add %506, %507 : tensor<?xi32>
  %509 = stablehlo.get_dimension_size %508, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %510 = stablehlo.constant dense<1> : tensor<1xi32>
  %511 = stablehlo.dynamic_reshape %509, %510 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %512 = stablehlo.concatenate %511, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %513 = stablehlo.maximum %467, %512 : tensor<1xi32>
  %514 = stablehlo.maximum %463, %513 : tensor<1xi32>
  %515 = stablehlo.dynamic_broadcast_in_dim %448, %514, dims = [0] : (tensor<?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %516 = stablehlo.dynamic_broadcast_in_dim %458, %514, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %517 = stablehlo.dynamic_broadcast_in_dim %508, %514, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %518 = stablehlo.select %515, %516, %517 : tensor<?xi1>, tensor<?xi32>
  %519 = stablehlo.reshape %518 : (tensor<?xi32>) -> tensor<1xi32>
  %520 = stablehlo.get_dimension_size %387, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %521 = stablehlo.constant dense<1> : tensor<1xi32>
  %522 = stablehlo.dynamic_reshape %520, %521 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %523 = stablehlo.concatenate %522, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %524 = stablehlo.constant dense<0> : tensor<1xi32>
  %525 = stablehlo.constant dense<1> : tensor<1xi32>
  %526 = stablehlo.constant dense<1> : tensor<1xi32>
  %527 = stablehlo.real_dynamic_slice %523, %524, %525, %526 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %528 = stablehlo.minimum %519, %527 : tensor<1xi32>
  %529 = stablehlo.concatenate %528, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %530 = stablehlo.constant dense<1> : tensor<i32>
  %531 = stablehlo.reshape %530 : (tensor<i32>) -> tensor<1xi32>
  %532 = stablehlo.concatenate %531, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %533 = stablehlo.real_dynamic_slice %387, %410, %529, %532 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %534 = stablehlo.concatenate %406, %407, %533, dim = 0 : (tensor<?xi32>, tensor<1xi32>, tensor<?xi32>) -> tensor<3xi32>
  %535 = stablehlo.dynamic_broadcast_in_dim %380, %534, dims = [0, 1] : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %536 = stablehlo.get_dimension_size %535, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %537 = stablehlo.constant dense<1> : tensor<1xi32>
  %538 = stablehlo.dynamic_reshape %536, %537 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %539 = stablehlo.get_dimension_size %535, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %540 = stablehlo.constant dense<1> : tensor<1xi32>
  %541 = stablehlo.dynamic_reshape %539, %540 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %542 = stablehlo.get_dimension_size %535, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %543 = stablehlo.constant dense<1> : tensor<1xi32>
  %544 = stablehlo.dynamic_reshape %542, %543 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %545 = stablehlo.concatenate %538, %541, %544, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %546 = stablehlo.get_dimension_size %368, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %547 = stablehlo.constant dense<1> : tensor<1xi32>
  %548 = stablehlo.dynamic_reshape %546, %547 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %549 = stablehlo.get_dimension_size %368, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %550 = stablehlo.constant dense<1> : tensor<1xi32>
  %551 = stablehlo.dynamic_reshape %549, %550 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %552 = stablehlo.get_dimension_size %368, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %553 = stablehlo.constant dense<1> : tensor<1xi32>
  %554 = stablehlo.dynamic_reshape %552, %553 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %555 = stablehlo.concatenate %548, %551, %554, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %556 = stablehlo.constant dense<2> : tensor<i32>
  %557 = stablehlo.reshape %556 : (tensor<i32>) -> tensor<1xi32>
  %558 = stablehlo.concatenate %557, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %559 = stablehlo.constant dense<3> : tensor<i32>
  %560 = stablehlo.reshape %559 : (tensor<i32>) -> tensor<1xi32>
  %561 = stablehlo.get_dimension_size %555, dim = 0 : (tensor<3xi32>) -> tensor<i32>
  %562 = stablehlo.constant dense<1> : tensor<1xi32>
  %563 = stablehlo.dynamic_reshape %561, %562 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %564 = stablehlo.concatenate %563, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %565 = stablehlo.constant dense<0> : tensor<1xi32>
  %566 = stablehlo.constant dense<1> : tensor<1xi32>
  %567 = stablehlo.constant dense<1> : tensor<1xi32>
  %568 = stablehlo.real_dynamic_slice %564, %565, %566, %567 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %569 = stablehlo.minimum %560, %568 : tensor<1xi32>
  %570 = stablehlo.concatenate %569, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %571 = stablehlo.constant dense<1> : tensor<i32>
  %572 = stablehlo.reshape %571 : (tensor<i32>) -> tensor<1xi32>
  %573 = stablehlo.concatenate %572, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %574 = stablehlo.real_dynamic_slice %555, %558, %570, %573 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %575 = stablehlo.constant dense<> : tensor<0xi32>
  %576 = stablehlo.dynamic_reshape %574, %575 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %577 = stablehlo.constant dense<> : tensor<0xi32>
  %578 = stablehlo.constant dense<> : tensor<0xi32>
  %579 = stablehlo.maximum %577, %578 : tensor<0xi32>
  %580 = stablehlo.dynamic_broadcast_in_dim %576, %579, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %581 = stablehlo.constant dense<1> : tensor<i32>
  %582 = stablehlo.dynamic_broadcast_in_dim %581, %579, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %583 = stablehlo.multiply %580, %582 : tensor<i32>
  %584 = stablehlo.convert %583 : (tensor<i32>) -> tensor<f32>
  %585 = stablehlo.constant dense<1> : tensor<i32>
  %586 = stablehlo.constant dense<3> : tensor<1xi32>
  %587 = stablehlo.dynamic_broadcast_in_dim %585, %586, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %588 = stablehlo.constant dense<> : tensor<0xi32>
  %589 = stablehlo.concatenate %587, %588, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<3xi32>
  %590 = stablehlo.dynamic_broadcast_in_dim %584, %589, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %591 = stablehlo.get_dimension_size %590, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %592 = stablehlo.constant dense<1> : tensor<1xi32>
  %593 = stablehlo.dynamic_reshape %591, %592 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %594 = stablehlo.get_dimension_size %590, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %595 = stablehlo.constant dense<1> : tensor<1xi32>
  %596 = stablehlo.dynamic_reshape %594, %595 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %597 = stablehlo.get_dimension_size %590, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %598 = stablehlo.constant dense<1> : tensor<1xi32>
  %599 = stablehlo.dynamic_reshape %597, %598 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %600 = stablehlo.concatenate %593, %596, %599, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %601 = stablehlo.maximum %545, %600 : tensor<3xi32>
  %602 = stablehlo.dynamic_broadcast_in_dim %535, %601, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %603 = stablehlo.dynamic_broadcast_in_dim %590, %601, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %604 = stablehlo.divide %602, %603 : tensor<?x?x?xf32>
  %605 = stablehlo.get_dimension_size %604, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %606 = stablehlo.constant dense<1> : tensor<1xi32>
  %607 = stablehlo.dynamic_reshape %605, %606 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %608 = stablehlo.get_dimension_size %604, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %609 = stablehlo.constant dense<1> : tensor<1xi32>
  %610 = stablehlo.dynamic_reshape %608, %609 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %611 = stablehlo.get_dimension_size %604, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %612 = stablehlo.constant dense<1> : tensor<1xi32>
  %613 = stablehlo.dynamic_reshape %611, %612 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %614 = stablehlo.concatenate %607, %610, %613, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %615 = stablehlo.maximum %378, %614 : tensor<3xi32>
  %616 = stablehlo.dynamic_broadcast_in_dim %368, %615, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %617 = stablehlo.dynamic_broadcast_in_dim %604, %615, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %618 = stablehlo.subtract %616, %617 : tensor<?x?x?xf32>
  %619 = stablehlo.get_dimension_size %618, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %620 = stablehlo.constant dense<1> : tensor<1xi32>
  %621 = stablehlo.dynamic_reshape %619, %620 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %622 = stablehlo.get_dimension_size %618, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %623 = stablehlo.constant dense<1> : tensor<1xi32>
  %624 = stablehlo.dynamic_reshape %622, %623 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %625 = stablehlo.get_dimension_size %618, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %626 = stablehlo.constant dense<1> : tensor<1xi32>
  %627 = stablehlo.dynamic_reshape %625, %626 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %628 = stablehlo.concatenate %621, %624, %627, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %629 = stablehlo.get_dimension_size %368, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %630 = stablehlo.constant dense<1> : tensor<1xi32>
  %631 = stablehlo.dynamic_reshape %629, %630 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %632 = stablehlo.get_dimension_size %368, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %633 = stablehlo.constant dense<1> : tensor<1xi32>
  %634 = stablehlo.dynamic_reshape %632, %633 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %635 = stablehlo.get_dimension_size %368, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %636 = stablehlo.constant dense<1> : tensor<1xi32>
  %637 = stablehlo.dynamic_reshape %635, %636 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %638 = stablehlo.concatenate %631, %634, %637, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %639 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %640 = stablehlo.reduce(%368 init: %639) applies stablehlo.add across dimensions = [2] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  %641 = stablehlo.get_dimension_size %640, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %642 = stablehlo.constant dense<1> : tensor<1xi32>
  %643 = stablehlo.dynamic_reshape %641, %642 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %644 = stablehlo.get_dimension_size %640, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %645 = stablehlo.constant dense<1> : tensor<1xi32>
  %646 = stablehlo.dynamic_reshape %644, %645 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %647 = stablehlo.concatenate %643, %646, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %648 = stablehlo.constant dense<0> : tensor<i32>
  %649 = stablehlo.reshape %648 : (tensor<i32>) -> tensor<1xi32>
  %650 = stablehlo.concatenate %649, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %651 = stablehlo.constant dense<2> : tensor<i32>
  %652 = stablehlo.reshape %651 : (tensor<i32>) -> tensor<1xi32>
  %653 = stablehlo.get_dimension_size %647, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %654 = stablehlo.constant dense<1> : tensor<1xi32>
  %655 = stablehlo.dynamic_reshape %653, %654 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %656 = stablehlo.concatenate %655, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %657 = stablehlo.constant dense<0> : tensor<1xi32>
  %658 = stablehlo.constant dense<1> : tensor<1xi32>
  %659 = stablehlo.constant dense<1> : tensor<1xi32>
  %660 = stablehlo.real_dynamic_slice %656, %657, %658, %659 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %661 = stablehlo.minimum %652, %660 : tensor<1xi32>
  %662 = stablehlo.concatenate %661, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %663 = stablehlo.constant dense<1> : tensor<i32>
  %664 = stablehlo.reshape %663 : (tensor<i32>) -> tensor<1xi32>
  %665 = stablehlo.concatenate %664, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %666 = stablehlo.real_dynamic_slice %647, %650, %662, %665 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %667 = stablehlo.constant dense<1> : tensor<1xi32>
  %668 = stablehlo.constant dense<2> : tensor<i32>
  %669 = stablehlo.reshape %668 : (tensor<i32>) -> tensor<1xi32>
  %670 = stablehlo.concatenate %669, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %671 = stablehlo.get_dimension_size %647, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %672 = stablehlo.constant dense<1> : tensor<1xi32>
  %673 = stablehlo.dynamic_reshape %671, %672 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %674 = stablehlo.concatenate %673, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %675 = stablehlo.constant dense<0> : tensor<i32>
  %676 = stablehlo.reshape %675 : (tensor<i32>) -> tensor<1xi32>
  %677 = stablehlo.concatenate %676, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %678 = stablehlo.constant dense<1> : tensor<i32>
  %679 = stablehlo.reshape %678 : (tensor<i32>) -> tensor<1xi32>
  %680 = stablehlo.get_dimension_size %674, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %681 = stablehlo.constant dense<1> : tensor<1xi32>
  %682 = stablehlo.dynamic_reshape %680, %681 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %683 = stablehlo.concatenate %682, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %684 = stablehlo.constant dense<0> : tensor<1xi32>
  %685 = stablehlo.constant dense<1> : tensor<1xi32>
  %686 = stablehlo.constant dense<1> : tensor<1xi32>
  %687 = stablehlo.real_dynamic_slice %683, %684, %685, %686 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %688 = stablehlo.minimum %679, %687 : tensor<1xi32>
  %689 = stablehlo.concatenate %688, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %690 = stablehlo.constant dense<1> : tensor<i32>
  %691 = stablehlo.reshape %690 : (tensor<i32>) -> tensor<1xi32>
  %692 = stablehlo.concatenate %691, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %693 = stablehlo.real_dynamic_slice %674, %677, %689, %692 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %694 = stablehlo.constant dense<> : tensor<0xi32>
  %695 = stablehlo.dynamic_reshape %693, %694 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %696 = stablehlo.constant dense<> : tensor<0xi32>
  %697 = stablehlo.constant dense<> : tensor<0xi32>
  %698 = stablehlo.maximum %696, %697 : tensor<0xi32>
  %699 = stablehlo.dynamic_broadcast_in_dim %695, %698, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %700 = stablehlo.constant dense<0> : tensor<i32>
  %701 = stablehlo.dynamic_broadcast_in_dim %700, %698, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %702 = stablehlo.compare  GE, %699, %701 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %703 = stablehlo.constant dense<1> : tensor<i32>
  %704 = stablehlo.constant dense<1> : tensor<1xi32>
  %705 = stablehlo.dynamic_broadcast_in_dim %703, %704, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %706 = stablehlo.constant dense<> : tensor<0xi32>
  %707 = stablehlo.concatenate %705, %706, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %708 = stablehlo.dynamic_broadcast_in_dim %702, %707, dims = [] : (tensor<i1>, tensor<1xi32>) -> tensor<?xi1>
  %709 = stablehlo.get_dimension_size %708, dim = 0 : (tensor<?xi1>) -> tensor<i32>
  %710 = stablehlo.constant dense<1> : tensor<1xi32>
  %711 = stablehlo.dynamic_reshape %709, %710 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %712 = stablehlo.concatenate %711, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %713 = stablehlo.constant dense<1> : tensor<i32>
  %714 = stablehlo.constant dense<1> : tensor<1xi32>
  %715 = stablehlo.dynamic_broadcast_in_dim %713, %714, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %716 = stablehlo.constant dense<> : tensor<0xi32>
  %717 = stablehlo.concatenate %715, %716, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %718 = stablehlo.dynamic_broadcast_in_dim %695, %717, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %719 = stablehlo.get_dimension_size %718, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %720 = stablehlo.constant dense<1> : tensor<1xi32>
  %721 = stablehlo.dynamic_reshape %719, %720 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %722 = stablehlo.concatenate %721, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %723 = stablehlo.maximum %712, %722 : tensor<1xi32>
  %724 = stablehlo.get_dimension_size %718, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %725 = stablehlo.constant dense<1> : tensor<1xi32>
  %726 = stablehlo.dynamic_reshape %724, %725 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %727 = stablehlo.concatenate %726, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %728 = stablehlo.constant dense<0> : tensor<i32>
  %729 = stablehlo.reshape %728 : (tensor<i32>) -> tensor<1xi32>
  %730 = stablehlo.concatenate %729, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %731 = stablehlo.constant dense<1> : tensor<i32>
  %732 = stablehlo.reshape %731 : (tensor<i32>) -> tensor<1xi32>
  %733 = stablehlo.get_dimension_size %674, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %734 = stablehlo.constant dense<1> : tensor<1xi32>
  %735 = stablehlo.dynamic_reshape %733, %734 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %736 = stablehlo.concatenate %735, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %737 = stablehlo.constant dense<0> : tensor<1xi32>
  %738 = stablehlo.constant dense<1> : tensor<1xi32>
  %739 = stablehlo.constant dense<1> : tensor<1xi32>
  %740 = stablehlo.real_dynamic_slice %736, %737, %738, %739 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %741 = stablehlo.minimum %732, %740 : tensor<1xi32>
  %742 = stablehlo.concatenate %741, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %743 = stablehlo.constant dense<1> : tensor<i32>
  %744 = stablehlo.reshape %743 : (tensor<i32>) -> tensor<1xi32>
  %745 = stablehlo.concatenate %744, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %746 = stablehlo.real_dynamic_slice %674, %730, %742, %745 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %747 = stablehlo.constant dense<> : tensor<0xi32>
  %748 = stablehlo.dynamic_reshape %746, %747 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %749 = stablehlo.constant dense<1> : tensor<1xi32>
  %750 = stablehlo.dynamic_reshape %748, %749 : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %751 = stablehlo.get_dimension_size %750, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %752 = stablehlo.constant dense<1> : tensor<1xi32>
  %753 = stablehlo.dynamic_reshape %751, %752 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %754 = stablehlo.concatenate %753, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %755 = stablehlo.constant dense<1> : tensor<i32>
  %756 = stablehlo.constant dense<1> : tensor<1xi32>
  %757 = stablehlo.dynamic_broadcast_in_dim %755, %756, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %758 = stablehlo.constant dense<> : tensor<0xi32>
  %759 = stablehlo.concatenate %757, %758, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %760 = stablehlo.dynamic_broadcast_in_dim %695, %759, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %761 = stablehlo.get_dimension_size %760, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %762 = stablehlo.constant dense<1> : tensor<1xi32>
  %763 = stablehlo.dynamic_reshape %761, %762 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %764 = stablehlo.concatenate %763, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %765 = stablehlo.maximum %754, %764 : tensor<1xi32>
  %766 = stablehlo.dynamic_broadcast_in_dim %750, %765, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %767 = stablehlo.dynamic_broadcast_in_dim %760, %765, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %768 = stablehlo.add %766, %767 : tensor<?xi32>
  %769 = stablehlo.get_dimension_size %768, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %770 = stablehlo.constant dense<1> : tensor<1xi32>
  %771 = stablehlo.dynamic_reshape %769, %770 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %772 = stablehlo.concatenate %771, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %773 = stablehlo.maximum %727, %772 : tensor<1xi32>
  %774 = stablehlo.maximum %723, %773 : tensor<1xi32>
  %775 = stablehlo.dynamic_broadcast_in_dim %708, %774, dims = [0] : (tensor<?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %776 = stablehlo.dynamic_broadcast_in_dim %718, %774, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %777 = stablehlo.dynamic_broadcast_in_dim %768, %774, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %778 = stablehlo.select %775, %776, %777 : tensor<?xi1>, tensor<?xi32>
  %779 = stablehlo.reshape %778 : (tensor<?xi32>) -> tensor<1xi32>
  %780 = stablehlo.get_dimension_size %647, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %781 = stablehlo.constant dense<1> : tensor<1xi32>
  %782 = stablehlo.dynamic_reshape %780, %781 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %783 = stablehlo.concatenate %782, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %784 = stablehlo.constant dense<0> : tensor<1xi32>
  %785 = stablehlo.constant dense<1> : tensor<1xi32>
  %786 = stablehlo.constant dense<1> : tensor<1xi32>
  %787 = stablehlo.real_dynamic_slice %783, %784, %785, %786 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %788 = stablehlo.minimum %779, %787 : tensor<1xi32>
  %789 = stablehlo.concatenate %788, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %790 = stablehlo.constant dense<1> : tensor<i32>
  %791 = stablehlo.reshape %790 : (tensor<i32>) -> tensor<1xi32>
  %792 = stablehlo.concatenate %791, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %793 = stablehlo.real_dynamic_slice %647, %670, %789, %792 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %794 = stablehlo.concatenate %666, %667, %793, dim = 0 : (tensor<?xi32>, tensor<1xi32>, tensor<?xi32>) -> tensor<3xi32>
  %795 = stablehlo.dynamic_broadcast_in_dim %640, %794, dims = [0, 1] : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %796 = stablehlo.get_dimension_size %795, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %797 = stablehlo.constant dense<1> : tensor<1xi32>
  %798 = stablehlo.dynamic_reshape %796, %797 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %799 = stablehlo.get_dimension_size %795, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %800 = stablehlo.constant dense<1> : tensor<1xi32>
  %801 = stablehlo.dynamic_reshape %799, %800 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %802 = stablehlo.get_dimension_size %795, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %803 = stablehlo.constant dense<1> : tensor<1xi32>
  %804 = stablehlo.dynamic_reshape %802, %803 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %805 = stablehlo.concatenate %798, %801, %804, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %806 = stablehlo.get_dimension_size %368, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %807 = stablehlo.constant dense<1> : tensor<1xi32>
  %808 = stablehlo.dynamic_reshape %806, %807 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %809 = stablehlo.get_dimension_size %368, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %810 = stablehlo.constant dense<1> : tensor<1xi32>
  %811 = stablehlo.dynamic_reshape %809, %810 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %812 = stablehlo.get_dimension_size %368, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %813 = stablehlo.constant dense<1> : tensor<1xi32>
  %814 = stablehlo.dynamic_reshape %812, %813 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %815 = stablehlo.concatenate %808, %811, %814, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %816 = stablehlo.constant dense<2> : tensor<i32>
  %817 = stablehlo.reshape %816 : (tensor<i32>) -> tensor<1xi32>
  %818 = stablehlo.concatenate %817, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %819 = stablehlo.constant dense<3> : tensor<i32>
  %820 = stablehlo.reshape %819 : (tensor<i32>) -> tensor<1xi32>
  %821 = stablehlo.get_dimension_size %815, dim = 0 : (tensor<3xi32>) -> tensor<i32>
  %822 = stablehlo.constant dense<1> : tensor<1xi32>
  %823 = stablehlo.dynamic_reshape %821, %822 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %824 = stablehlo.concatenate %823, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %825 = stablehlo.constant dense<0> : tensor<1xi32>
  %826 = stablehlo.constant dense<1> : tensor<1xi32>
  %827 = stablehlo.constant dense<1> : tensor<1xi32>
  %828 = stablehlo.real_dynamic_slice %824, %825, %826, %827 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %829 = stablehlo.minimum %820, %828 : tensor<1xi32>
  %830 = stablehlo.concatenate %829, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %831 = stablehlo.constant dense<1> : tensor<i32>
  %832 = stablehlo.reshape %831 : (tensor<i32>) -> tensor<1xi32>
  %833 = stablehlo.concatenate %832, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %834 = stablehlo.real_dynamic_slice %815, %818, %830, %833 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %835 = stablehlo.constant dense<> : tensor<0xi32>
  %836 = stablehlo.dynamic_reshape %834, %835 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %837 = stablehlo.constant dense<> : tensor<0xi32>
  %838 = stablehlo.constant dense<> : tensor<0xi32>
  %839 = stablehlo.maximum %837, %838 : tensor<0xi32>
  %840 = stablehlo.dynamic_broadcast_in_dim %836, %839, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %841 = stablehlo.constant dense<1> : tensor<i32>
  %842 = stablehlo.dynamic_broadcast_in_dim %841, %839, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %843 = stablehlo.multiply %840, %842 : tensor<i32>
  %844 = stablehlo.convert %843 : (tensor<i32>) -> tensor<f32>
  %845 = stablehlo.constant dense<1> : tensor<i32>
  %846 = stablehlo.constant dense<3> : tensor<1xi32>
  %847 = stablehlo.dynamic_broadcast_in_dim %845, %846, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %848 = stablehlo.constant dense<> : tensor<0xi32>
  %849 = stablehlo.concatenate %847, %848, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<3xi32>
  %850 = stablehlo.dynamic_broadcast_in_dim %844, %849, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %851 = stablehlo.get_dimension_size %850, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %852 = stablehlo.constant dense<1> : tensor<1xi32>
  %853 = stablehlo.dynamic_reshape %851, %852 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %854 = stablehlo.get_dimension_size %850, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %855 = stablehlo.constant dense<1> : tensor<1xi32>
  %856 = stablehlo.dynamic_reshape %854, %855 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %857 = stablehlo.get_dimension_size %850, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %858 = stablehlo.constant dense<1> : tensor<1xi32>
  %859 = stablehlo.dynamic_reshape %857, %858 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %860 = stablehlo.concatenate %853, %856, %859, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %861 = stablehlo.maximum %805, %860 : tensor<3xi32>
  %862 = stablehlo.dynamic_broadcast_in_dim %795, %861, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %863 = stablehlo.dynamic_broadcast_in_dim %850, %861, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %864 = stablehlo.divide %862, %863 : tensor<?x?x?xf32>
  %865 = stablehlo.get_dimension_size %864, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %866 = stablehlo.constant dense<1> : tensor<1xi32>
  %867 = stablehlo.dynamic_reshape %865, %866 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %868 = stablehlo.get_dimension_size %864, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %869 = stablehlo.constant dense<1> : tensor<1xi32>
  %870 = stablehlo.dynamic_reshape %868, %869 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %871 = stablehlo.get_dimension_size %864, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %872 = stablehlo.constant dense<1> : tensor<1xi32>
  %873 = stablehlo.dynamic_reshape %871, %872 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %874 = stablehlo.concatenate %867, %870, %873, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %875 = stablehlo.maximum %638, %874 : tensor<3xi32>
  %876 = stablehlo.dynamic_broadcast_in_dim %368, %875, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %877 = stablehlo.dynamic_broadcast_in_dim %864, %875, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %878 = stablehlo.subtract %876, %877 : tensor<?x?x?xf32>
  %879 = stablehlo.get_dimension_size %878, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %880 = stablehlo.constant dense<1> : tensor<1xi32>
  %881 = stablehlo.dynamic_reshape %879, %880 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %882 = stablehlo.get_dimension_size %878, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %883 = stablehlo.constant dense<1> : tensor<1xi32>
  %884 = stablehlo.dynamic_reshape %882, %883 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %885 = stablehlo.get_dimension_size %878, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %886 = stablehlo.constant dense<1> : tensor<1xi32>
  %887 = stablehlo.dynamic_reshape %885, %886 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %888 = stablehlo.concatenate %881, %884, %887, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %889 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  %890 = stablehlo.constant dense<1> : tensor<i32>
  %891 = stablehlo.constant dense<3> : tensor<1xi32>
  %892 = stablehlo.dynamic_broadcast_in_dim %890, %891, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %893 = stablehlo.constant dense<> : tensor<0xi32>
  %894 = stablehlo.concatenate %892, %893, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<3xi32>
  %895 = stablehlo.dynamic_broadcast_in_dim %889, %894, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %896 = stablehlo.get_dimension_size %895, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %897 = stablehlo.constant dense<1> : tensor<1xi32>
  %898 = stablehlo.dynamic_reshape %896, %897 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %899 = stablehlo.get_dimension_size %895, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %900 = stablehlo.constant dense<1> : tensor<1xi32>
  %901 = stablehlo.dynamic_reshape %899, %900 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %902 = stablehlo.get_dimension_size %895, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %903 = stablehlo.constant dense<1> : tensor<1xi32>
  %904 = stablehlo.dynamic_reshape %902, %903 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %905 = stablehlo.concatenate %898, %901, %904, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %906 = stablehlo.maximum %888, %905 : tensor<3xi32>
  %907 = stablehlo.dynamic_broadcast_in_dim %878, %906, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %908 = stablehlo.dynamic_broadcast_in_dim %895, %906, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %909 = stablehlo.power %907, %908 : tensor<?x?x?xf32>
  %910 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %911 = stablehlo.reduce(%909 init: %910) applies stablehlo.add across dimensions = [2] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  %912 = stablehlo.get_dimension_size %911, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %913 = stablehlo.constant dense<1> : tensor<1xi32>
  %914 = stablehlo.dynamic_reshape %912, %913 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %915 = stablehlo.get_dimension_size %911, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %916 = stablehlo.constant dense<1> : tensor<1xi32>
  %917 = stablehlo.dynamic_reshape %915, %916 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %918 = stablehlo.concatenate %914, %917, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %919 = stablehlo.constant dense<0> : tensor<i32>
  %920 = stablehlo.reshape %919 : (tensor<i32>) -> tensor<1xi32>
  %921 = stablehlo.concatenate %920, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %922 = stablehlo.constant dense<2> : tensor<i32>
  %923 = stablehlo.reshape %922 : (tensor<i32>) -> tensor<1xi32>
  %924 = stablehlo.get_dimension_size %918, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %925 = stablehlo.constant dense<1> : tensor<1xi32>
  %926 = stablehlo.dynamic_reshape %924, %925 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %927 = stablehlo.concatenate %926, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %928 = stablehlo.constant dense<0> : tensor<1xi32>
  %929 = stablehlo.constant dense<1> : tensor<1xi32>
  %930 = stablehlo.constant dense<1> : tensor<1xi32>
  %931 = stablehlo.real_dynamic_slice %927, %928, %929, %930 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %932 = stablehlo.minimum %923, %931 : tensor<1xi32>
  %933 = stablehlo.concatenate %932, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %934 = stablehlo.constant dense<1> : tensor<i32>
  %935 = stablehlo.reshape %934 : (tensor<i32>) -> tensor<1xi32>
  %936 = stablehlo.concatenate %935, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %937 = stablehlo.real_dynamic_slice %918, %921, %933, %936 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %938 = stablehlo.constant dense<1> : tensor<1xi32>
  %939 = stablehlo.constant dense<2> : tensor<i32>
  %940 = stablehlo.reshape %939 : (tensor<i32>) -> tensor<1xi32>
  %941 = stablehlo.concatenate %940, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %942 = stablehlo.get_dimension_size %918, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %943 = stablehlo.constant dense<1> : tensor<1xi32>
  %944 = stablehlo.dynamic_reshape %942, %943 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %945 = stablehlo.concatenate %944, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %946 = stablehlo.constant dense<0> : tensor<i32>
  %947 = stablehlo.reshape %946 : (tensor<i32>) -> tensor<1xi32>
  %948 = stablehlo.concatenate %947, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %949 = stablehlo.constant dense<1> : tensor<i32>
  %950 = stablehlo.reshape %949 : (tensor<i32>) -> tensor<1xi32>
  %951 = stablehlo.get_dimension_size %945, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %952 = stablehlo.constant dense<1> : tensor<1xi32>
  %953 = stablehlo.dynamic_reshape %951, %952 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %954 = stablehlo.concatenate %953, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %955 = stablehlo.constant dense<0> : tensor<1xi32>
  %956 = stablehlo.constant dense<1> : tensor<1xi32>
  %957 = stablehlo.constant dense<1> : tensor<1xi32>
  %958 = stablehlo.real_dynamic_slice %954, %955, %956, %957 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %959 = stablehlo.minimum %950, %958 : tensor<1xi32>
  %960 = stablehlo.concatenate %959, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %961 = stablehlo.constant dense<1> : tensor<i32>
  %962 = stablehlo.reshape %961 : (tensor<i32>) -> tensor<1xi32>
  %963 = stablehlo.concatenate %962, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %964 = stablehlo.real_dynamic_slice %945, %948, %960, %963 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %965 = stablehlo.constant dense<> : tensor<0xi32>
  %966 = stablehlo.dynamic_reshape %964, %965 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %967 = stablehlo.constant dense<> : tensor<0xi32>
  %968 = stablehlo.constant dense<> : tensor<0xi32>
  %969 = stablehlo.maximum %967, %968 : tensor<0xi32>
  %970 = stablehlo.dynamic_broadcast_in_dim %966, %969, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %971 = stablehlo.constant dense<0> : tensor<i32>
  %972 = stablehlo.dynamic_broadcast_in_dim %971, %969, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %973 = stablehlo.compare  GE, %970, %972 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %974 = stablehlo.constant dense<1> : tensor<i32>
  %975 = stablehlo.constant dense<1> : tensor<1xi32>
  %976 = stablehlo.dynamic_broadcast_in_dim %974, %975, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %977 = stablehlo.constant dense<> : tensor<0xi32>
  %978 = stablehlo.concatenate %976, %977, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %979 = stablehlo.dynamic_broadcast_in_dim %973, %978, dims = [] : (tensor<i1>, tensor<1xi32>) -> tensor<?xi1>
  %980 = stablehlo.get_dimension_size %979, dim = 0 : (tensor<?xi1>) -> tensor<i32>
  %981 = stablehlo.constant dense<1> : tensor<1xi32>
  %982 = stablehlo.dynamic_reshape %980, %981 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %983 = stablehlo.concatenate %982, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %984 = stablehlo.constant dense<1> : tensor<i32>
  %985 = stablehlo.constant dense<1> : tensor<1xi32>
  %986 = stablehlo.dynamic_broadcast_in_dim %984, %985, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %987 = stablehlo.constant dense<> : tensor<0xi32>
  %988 = stablehlo.concatenate %986, %987, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %989 = stablehlo.dynamic_broadcast_in_dim %966, %988, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %990 = stablehlo.get_dimension_size %989, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %991 = stablehlo.constant dense<1> : tensor<1xi32>
  %992 = stablehlo.dynamic_reshape %990, %991 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %993 = stablehlo.concatenate %992, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %994 = stablehlo.maximum %983, %993 : tensor<1xi32>
  %995 = stablehlo.get_dimension_size %989, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %996 = stablehlo.constant dense<1> : tensor<1xi32>
  %997 = stablehlo.dynamic_reshape %995, %996 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %998 = stablehlo.concatenate %997, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %999 = stablehlo.constant dense<0> : tensor<i32>
  %1000 = stablehlo.reshape %999 : (tensor<i32>) -> tensor<1xi32>
  %1001 = stablehlo.concatenate %1000, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1002 = stablehlo.constant dense<1> : tensor<i32>
  %1003 = stablehlo.reshape %1002 : (tensor<i32>) -> tensor<1xi32>
  %1004 = stablehlo.get_dimension_size %945, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %1005 = stablehlo.constant dense<1> : tensor<1xi32>
  %1006 = stablehlo.dynamic_reshape %1004, %1005 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1007 = stablehlo.concatenate %1006, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1008 = stablehlo.constant dense<0> : tensor<1xi32>
  %1009 = stablehlo.constant dense<1> : tensor<1xi32>
  %1010 = stablehlo.constant dense<1> : tensor<1xi32>
  %1011 = stablehlo.real_dynamic_slice %1007, %1008, %1009, %1010 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1012 = stablehlo.minimum %1003, %1011 : tensor<1xi32>
  %1013 = stablehlo.concatenate %1012, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1014 = stablehlo.constant dense<1> : tensor<i32>
  %1015 = stablehlo.reshape %1014 : (tensor<i32>) -> tensor<1xi32>
  %1016 = stablehlo.concatenate %1015, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1017 = stablehlo.real_dynamic_slice %945, %1001, %1013, %1016 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1018 = stablehlo.constant dense<> : tensor<0xi32>
  %1019 = stablehlo.dynamic_reshape %1017, %1018 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %1020 = stablehlo.constant dense<1> : tensor<1xi32>
  %1021 = stablehlo.dynamic_reshape %1019, %1020 : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1022 = stablehlo.get_dimension_size %1021, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1023 = stablehlo.constant dense<1> : tensor<1xi32>
  %1024 = stablehlo.dynamic_reshape %1022, %1023 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1025 = stablehlo.concatenate %1024, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1026 = stablehlo.constant dense<1> : tensor<i32>
  %1027 = stablehlo.constant dense<1> : tensor<1xi32>
  %1028 = stablehlo.dynamic_broadcast_in_dim %1026, %1027, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1029 = stablehlo.constant dense<> : tensor<0xi32>
  %1030 = stablehlo.concatenate %1028, %1029, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %1031 = stablehlo.dynamic_broadcast_in_dim %966, %1030, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1032 = stablehlo.get_dimension_size %1031, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1033 = stablehlo.constant dense<1> : tensor<1xi32>
  %1034 = stablehlo.dynamic_reshape %1032, %1033 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1035 = stablehlo.concatenate %1034, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1036 = stablehlo.maximum %1025, %1035 : tensor<1xi32>
  %1037 = stablehlo.dynamic_broadcast_in_dim %1021, %1036, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1038 = stablehlo.dynamic_broadcast_in_dim %1031, %1036, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1039 = stablehlo.add %1037, %1038 : tensor<?xi32>
  %1040 = stablehlo.get_dimension_size %1039, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1041 = stablehlo.constant dense<1> : tensor<1xi32>
  %1042 = stablehlo.dynamic_reshape %1040, %1041 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1043 = stablehlo.concatenate %1042, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1044 = stablehlo.maximum %998, %1043 : tensor<1xi32>
  %1045 = stablehlo.maximum %994, %1044 : tensor<1xi32>
  %1046 = stablehlo.dynamic_broadcast_in_dim %979, %1045, dims = [0] : (tensor<?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %1047 = stablehlo.dynamic_broadcast_in_dim %989, %1045, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1048 = stablehlo.dynamic_broadcast_in_dim %1039, %1045, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1049 = stablehlo.select %1046, %1047, %1048 : tensor<?xi1>, tensor<?xi32>
  %1050 = stablehlo.reshape %1049 : (tensor<?xi32>) -> tensor<1xi32>
  %1051 = stablehlo.get_dimension_size %918, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %1052 = stablehlo.constant dense<1> : tensor<1xi32>
  %1053 = stablehlo.dynamic_reshape %1051, %1052 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1054 = stablehlo.concatenate %1053, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1055 = stablehlo.constant dense<0> : tensor<1xi32>
  %1056 = stablehlo.constant dense<1> : tensor<1xi32>
  %1057 = stablehlo.constant dense<1> : tensor<1xi32>
  %1058 = stablehlo.real_dynamic_slice %1054, %1055, %1056, %1057 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1059 = stablehlo.minimum %1050, %1058 : tensor<1xi32>
  %1060 = stablehlo.concatenate %1059, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1061 = stablehlo.constant dense<1> : tensor<i32>
  %1062 = stablehlo.reshape %1061 : (tensor<i32>) -> tensor<1xi32>
  %1063 = stablehlo.concatenate %1062, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1064 = stablehlo.real_dynamic_slice %918, %941, %1060, %1063 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1065 = stablehlo.concatenate %937, %938, %1064, dim = 0 : (tensor<?xi32>, tensor<1xi32>, tensor<?xi32>) -> tensor<3xi32>
  %1066 = stablehlo.dynamic_broadcast_in_dim %911, %1065, dims = [0, 1] : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1067 = stablehlo.get_dimension_size %1066, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1068 = stablehlo.constant dense<1> : tensor<1xi32>
  %1069 = stablehlo.dynamic_reshape %1067, %1068 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1070 = stablehlo.get_dimension_size %1066, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1071 = stablehlo.constant dense<1> : tensor<1xi32>
  %1072 = stablehlo.dynamic_reshape %1070, %1071 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1073 = stablehlo.get_dimension_size %1066, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1074 = stablehlo.constant dense<1> : tensor<1xi32>
  %1075 = stablehlo.dynamic_reshape %1073, %1074 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1076 = stablehlo.concatenate %1069, %1072, %1075, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1077 = stablehlo.get_dimension_size %909, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1078 = stablehlo.constant dense<1> : tensor<1xi32>
  %1079 = stablehlo.dynamic_reshape %1077, %1078 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1080 = stablehlo.get_dimension_size %909, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1081 = stablehlo.constant dense<1> : tensor<1xi32>
  %1082 = stablehlo.dynamic_reshape %1080, %1081 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1083 = stablehlo.get_dimension_size %909, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1084 = stablehlo.constant dense<1> : tensor<1xi32>
  %1085 = stablehlo.dynamic_reshape %1083, %1084 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1086 = stablehlo.concatenate %1079, %1082, %1085, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1087 = stablehlo.constant dense<2> : tensor<i32>
  %1088 = stablehlo.reshape %1087 : (tensor<i32>) -> tensor<1xi32>
  %1089 = stablehlo.concatenate %1088, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1090 = stablehlo.constant dense<3> : tensor<i32>
  %1091 = stablehlo.reshape %1090 : (tensor<i32>) -> tensor<1xi32>
  %1092 = stablehlo.get_dimension_size %1086, dim = 0 : (tensor<3xi32>) -> tensor<i32>
  %1093 = stablehlo.constant dense<1> : tensor<1xi32>
  %1094 = stablehlo.dynamic_reshape %1092, %1093 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1095 = stablehlo.concatenate %1094, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1096 = stablehlo.constant dense<0> : tensor<1xi32>
  %1097 = stablehlo.constant dense<1> : tensor<1xi32>
  %1098 = stablehlo.constant dense<1> : tensor<1xi32>
  %1099 = stablehlo.real_dynamic_slice %1095, %1096, %1097, %1098 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1100 = stablehlo.minimum %1091, %1099 : tensor<1xi32>
  %1101 = stablehlo.concatenate %1100, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1102 = stablehlo.constant dense<1> : tensor<i32>
  %1103 = stablehlo.reshape %1102 : (tensor<i32>) -> tensor<1xi32>
  %1104 = stablehlo.concatenate %1103, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1105 = stablehlo.real_dynamic_slice %1086, %1089, %1101, %1104 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1106 = stablehlo.constant dense<> : tensor<0xi32>
  %1107 = stablehlo.dynamic_reshape %1105, %1106 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %1108 = stablehlo.constant dense<> : tensor<0xi32>
  %1109 = stablehlo.constant dense<> : tensor<0xi32>
  %1110 = stablehlo.maximum %1108, %1109 : tensor<0xi32>
  %1111 = stablehlo.dynamic_broadcast_in_dim %1107, %1110, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %1112 = stablehlo.constant dense<1> : tensor<i32>
  %1113 = stablehlo.dynamic_broadcast_in_dim %1112, %1110, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %1114 = stablehlo.multiply %1111, %1113 : tensor<i32>
  %1115 = stablehlo.constant dense<> : tensor<0xi32>
  %1116 = stablehlo.constant dense<> : tensor<0xi32>
  %1117 = stablehlo.maximum %1115, %1116 : tensor<0xi32>
  %1118 = stablehlo.dynamic_broadcast_in_dim %1114, %1117, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %1119 = stablehlo.constant dense<1> : tensor<i32>
  %1120 = stablehlo.dynamic_broadcast_in_dim %1119, %1117, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %1121 = stablehlo.subtract %1118, %1120 : tensor<i32>
  %1122 = stablehlo.constant dense<> : tensor<0xi32>
  %1123 = stablehlo.constant dense<> : tensor<0xi32>
  %1124 = stablehlo.maximum %1122, %1123 : tensor<0xi32>
  %1125 = stablehlo.dynamic_broadcast_in_dim %1121, %1124, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %1126 = stablehlo.constant dense<0> : tensor<i32>
  %1127 = stablehlo.dynamic_broadcast_in_dim %1126, %1124, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %1128 = stablehlo.maximum %1125, %1127 : tensor<i32>
  %1129 = stablehlo.convert %1128 : (tensor<i32>) -> tensor<f32>
  %1130 = stablehlo.constant dense<1> : tensor<i32>
  %1131 = stablehlo.constant dense<3> : tensor<1xi32>
  %1132 = stablehlo.dynamic_broadcast_in_dim %1130, %1131, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1133 = stablehlo.constant dense<> : tensor<0xi32>
  %1134 = stablehlo.concatenate %1132, %1133, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<3xi32>
  %1135 = stablehlo.dynamic_broadcast_in_dim %1129, %1134, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1136 = stablehlo.get_dimension_size %1135, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1137 = stablehlo.constant dense<1> : tensor<1xi32>
  %1138 = stablehlo.dynamic_reshape %1136, %1137 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1139 = stablehlo.get_dimension_size %1135, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1140 = stablehlo.constant dense<1> : tensor<1xi32>
  %1141 = stablehlo.dynamic_reshape %1139, %1140 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1142 = stablehlo.get_dimension_size %1135, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1143 = stablehlo.constant dense<1> : tensor<1xi32>
  %1144 = stablehlo.dynamic_reshape %1142, %1143 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1145 = stablehlo.concatenate %1138, %1141, %1144, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1146 = stablehlo.maximum %1076, %1145 : tensor<3xi32>
  %1147 = stablehlo.dynamic_broadcast_in_dim %1066, %1146, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1148 = stablehlo.dynamic_broadcast_in_dim %1135, %1146, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1149 = stablehlo.divide %1147, %1148 : tensor<?x?x?xf32>
  %1150 = stablehlo.get_dimension_size %1149, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1151 = stablehlo.constant dense<1> : tensor<1xi32>
  %1152 = stablehlo.dynamic_reshape %1150, %1151 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1153 = stablehlo.get_dimension_size %1149, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1154 = stablehlo.constant dense<1> : tensor<1xi32>
  %1155 = stablehlo.dynamic_reshape %1153, %1154 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1156 = stablehlo.get_dimension_size %1149, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1157 = stablehlo.constant dense<1> : tensor<1xi32>
  %1158 = stablehlo.dynamic_reshape %1156, %1157 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1159 = stablehlo.concatenate %1152, %1155, %1158, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1160 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
  %1161 = stablehlo.constant dense<1> : tensor<i32>
  %1162 = stablehlo.constant dense<3> : tensor<1xi32>
  %1163 = stablehlo.dynamic_broadcast_in_dim %1161, %1162, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1164 = stablehlo.constant dense<> : tensor<0xi32>
  %1165 = stablehlo.concatenate %1163, %1164, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<3xi32>
  %1166 = stablehlo.dynamic_broadcast_in_dim %1160, %1165, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1167 = stablehlo.get_dimension_size %1166, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1168 = stablehlo.constant dense<1> : tensor<1xi32>
  %1169 = stablehlo.dynamic_reshape %1167, %1168 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1170 = stablehlo.get_dimension_size %1166, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1171 = stablehlo.constant dense<1> : tensor<1xi32>
  %1172 = stablehlo.dynamic_reshape %1170, %1171 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1173 = stablehlo.get_dimension_size %1166, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1174 = stablehlo.constant dense<1> : tensor<1xi32>
  %1175 = stablehlo.dynamic_reshape %1173, %1174 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1176 = stablehlo.concatenate %1169, %1172, %1175, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1177 = stablehlo.maximum %1159, %1176 : tensor<3xi32>
  %1178 = stablehlo.dynamic_broadcast_in_dim %1149, %1177, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1179 = stablehlo.dynamic_broadcast_in_dim %1166, %1177, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1180 = stablehlo.add %1178, %1179 : tensor<?x?x?xf32>
  %1181 = stablehlo.rsqrt %1180 : tensor<?x?x?xf32>
  %1182 = stablehlo.get_dimension_size %1181, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1183 = stablehlo.constant dense<1> : tensor<1xi32>
  %1184 = stablehlo.dynamic_reshape %1182, %1183 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1185 = stablehlo.get_dimension_size %1181, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1186 = stablehlo.constant dense<1> : tensor<1xi32>
  %1187 = stablehlo.dynamic_reshape %1185, %1186 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1188 = stablehlo.get_dimension_size %1181, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1189 = stablehlo.constant dense<1> : tensor<1xi32>
  %1190 = stablehlo.dynamic_reshape %1188, %1189 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1191 = stablehlo.concatenate %1184, %1187, %1190, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1192 = stablehlo.maximum %628, %1191 : tensor<3xi32>
  %1193 = stablehlo.dynamic_broadcast_in_dim %618, %1192, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1194 = stablehlo.dynamic_broadcast_in_dim %1181, %1192, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1195 = stablehlo.multiply %1193, %1194 : tensor<?x?x?xf32>
  %1196 = stablehlo.get_dimension_size %1195, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1197 = stablehlo.constant dense<1> : tensor<1xi32>
  %1198 = stablehlo.dynamic_reshape %1196, %1197 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1199 = stablehlo.get_dimension_size %1195, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1200 = stablehlo.constant dense<1> : tensor<1xi32>
  %1201 = stablehlo.dynamic_reshape %1199, %1200 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1202 = stablehlo.get_dimension_size %1195, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1203 = stablehlo.constant dense<1> : tensor<1xi32>
  %1204 = stablehlo.dynamic_reshape %1202, %1203 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1205 = stablehlo.concatenate %1198, %1201, %1204, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1206 = stablehlo.maximum %21, %1205 : tensor<3xi32>
  %1207 = stablehlo.dynamic_broadcast_in_dim %11, %1206, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1208 = stablehlo.dynamic_broadcast_in_dim %1195, %1206, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1209 = stablehlo.multiply %1207, %1208 : tensor<?x?x?xf32>
  %1210 = stablehlo.get_dimension_size %1209, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1211 = stablehlo.constant dense<1> : tensor<1xi32>
  %1212 = stablehlo.dynamic_reshape %1210, %1211 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1213 = stablehlo.get_dimension_size %1209, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1214 = stablehlo.constant dense<1> : tensor<1xi32>
  %1215 = stablehlo.dynamic_reshape %1213, %1214 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1216 = stablehlo.get_dimension_size %1209, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1217 = stablehlo.constant dense<1> : tensor<1xi32>
  %1218 = stablehlo.dynamic_reshape %1216, %1217 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1219 = stablehlo.concatenate %1212, %1215, %1218, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1220 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1221 = stablehlo.constant dense<1024> : tensor<1xi32>
  %1222 = stablehlo.dynamic_broadcast_in_dim %1220, %1221, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  %1223 = stablehlo.constant dense<1> : tensor<i32>
  %1224 = stablehlo.constant dense<2> : tensor<1xi32>
  %1225 = stablehlo.dynamic_broadcast_in_dim %1223, %1224, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1226 = stablehlo.get_dimension_size %1222, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %1227 = stablehlo.constant dense<1> : tensor<1xi32>
  %1228 = stablehlo.dynamic_reshape %1226, %1227 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1229 = stablehlo.concatenate %1228, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1230 = stablehlo.concatenate %1225, %1229, dim = 0 : (tensor<?xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1231 = stablehlo.dynamic_broadcast_in_dim %1222, %1230, dims = [2] : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1232 = stablehlo.get_dimension_size %1231, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1233 = stablehlo.constant dense<1> : tensor<1xi32>
  %1234 = stablehlo.dynamic_reshape %1232, %1233 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1235 = stablehlo.get_dimension_size %1231, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1236 = stablehlo.constant dense<1> : tensor<1xi32>
  %1237 = stablehlo.dynamic_reshape %1235, %1236 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1238 = stablehlo.get_dimension_size %1231, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1239 = stablehlo.constant dense<1> : tensor<1xi32>
  %1240 = stablehlo.dynamic_reshape %1238, %1239 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1241 = stablehlo.concatenate %1234, %1237, %1240, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1242 = stablehlo.maximum %1219, %1241 : tensor<3xi32>
  %1243 = stablehlo.dynamic_broadcast_in_dim %1209, %1242, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1244 = stablehlo.dynamic_broadcast_in_dim %1231, %1242, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1245 = stablehlo.add %1243, %1244 : tensor<?x?x?xf32>
  %1246 = stablehlo.constant dense<> : tensor<0xi32>
  %1247 = stablehlo.get_dimension_size %1245, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1248 = stablehlo.constant dense<1> : tensor<1xi32>
  %1249 = stablehlo.dynamic_reshape %1247, %1248 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1250 = stablehlo.get_dimension_size %1245, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1251 = stablehlo.constant dense<1> : tensor<1xi32>
  %1252 = stablehlo.dynamic_reshape %1250, %1251 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1253 = stablehlo.get_dimension_size %1245, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1254 = stablehlo.constant dense<1> : tensor<1xi32>
  %1255 = stablehlo.dynamic_reshape %1253, %1254 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1256 = stablehlo.concatenate %1249, %1252, %1255, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1257 = stablehlo.constant dense<0> : tensor<1xi32>
  %1258 = stablehlo.constant dense<1> : tensor<1xi32>
  %1259 = stablehlo.constant dense<1> : tensor<1xi32>
  %1260 = stablehlo.real_dynamic_slice %1256, %1257, %1258, %1259 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1261 = stablehlo.concatenate %1246, %1260, dim = 0 : (tensor<0xi32>, tensor<?xi32>) -> tensor<?xi32>
  %1262 = stablehlo.constant dense<1> : tensor<1xi32>
  %1263 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1264 = stablehlo.constant dense<[3072, 1024]> : tensor<2xi32>
  %1265 = stablehlo.dynamic_broadcast_in_dim %1263, %1264, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x?xf32>
  %1266 = stablehlo.transpose %1265, dims = [1, 0] : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1267 = stablehlo.get_dimension_size %1266, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %1268 = stablehlo.constant dense<1> : tensor<1xi32>
  %1269 = stablehlo.dynamic_reshape %1267, %1268 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1270 = stablehlo.get_dimension_size %1266, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %1271 = stablehlo.constant dense<1> : tensor<1xi32>
  %1272 = stablehlo.dynamic_reshape %1270, %1271 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1273 = stablehlo.concatenate %1269, %1272, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %1274 = stablehlo.constant dense<0> : tensor<1xi32>
  %1275 = stablehlo.constant dense<0> : tensor<1xi32>
  %1276 = stablehlo.constant dense<1> : tensor<1xi32>
  %1277 = stablehlo.real_dynamic_slice %1273, %1274, %1275, %1276 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1278 = stablehlo.concatenate %1262, %1277, dim = 0 : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
  %1279 = stablehlo.maximum %1261, %1278 : tensor<?xi32>
  %1280 = stablehlo.constant dense<3> : tensor<1xi32>
  %1281 = stablehlo.real_dynamic_slice %1256, %1258, %1280, %1259 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1282 = stablehlo.concatenate %1279, %1281, dim = 0 : (tensor<?xi32>, tensor<?xi32>) -> tensor<3xi32>
  %1283 = stablehlo.dynamic_broadcast_in_dim %1245, %1282, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1284 = stablehlo.constant dense<2> : tensor<1xi32>
  %1285 = stablehlo.real_dynamic_slice %1273, %1275, %1284, %1276 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1286 = stablehlo.concatenate %1279, %1285, dim = 0 : (tensor<?xi32>, tensor<?xi32>) -> tensor<3xi32>
  %1287 = stablehlo.dynamic_broadcast_in_dim %1266, %1286, dims = [1, 2] : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1288 = stablehlo.dot_general %1283, %1287, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1289 = stablehlo.get_dimension_size %1288, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1290 = stablehlo.constant dense<1> : tensor<1xi32>
  %1291 = stablehlo.dynamic_reshape %1289, %1290 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1292 = stablehlo.get_dimension_size %1288, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1293 = stablehlo.constant dense<1> : tensor<1xi32>
  %1294 = stablehlo.dynamic_reshape %1292, %1293 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1295 = stablehlo.get_dimension_size %1288, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1296 = stablehlo.constant dense<1> : tensor<1xi32>
  %1297 = stablehlo.dynamic_reshape %1295, %1296 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1298 = stablehlo.concatenate %1291, %1294, %1297, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1299 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1300 = stablehlo.constant dense<3072> : tensor<1xi32>
  %1301 = stablehlo.dynamic_broadcast_in_dim %1299, %1300, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  %1302 = stablehlo.get_dimension_size %1301, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %1303 = stablehlo.constant dense<1> : tensor<1xi32>
  %1304 = stablehlo.dynamic_reshape %1302, %1303 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1305 = stablehlo.concatenate %1304, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1306 = stablehlo.constant dense<0> : tensor<i32>
  %1307 = stablehlo.reshape %1306 : (tensor<i32>) -> tensor<1xi32>
  %1308 = stablehlo.concatenate %1307, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1309 = stablehlo.constant dense<0> : tensor<i32>
  %1310 = stablehlo.reshape %1309 : (tensor<i32>) -> tensor<1xi32>
  %1311 = stablehlo.get_dimension_size %1305, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %1312 = stablehlo.constant dense<1> : tensor<1xi32>
  %1313 = stablehlo.dynamic_reshape %1311, %1312 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1314 = stablehlo.concatenate %1313, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1315 = stablehlo.constant dense<0> : tensor<1xi32>
  %1316 = stablehlo.constant dense<1> : tensor<1xi32>
  %1317 = stablehlo.constant dense<1> : tensor<1xi32>
  %1318 = stablehlo.real_dynamic_slice %1314, %1315, %1316, %1317 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1319 = stablehlo.minimum %1310, %1318 : tensor<1xi32>
  %1320 = stablehlo.concatenate %1319, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1321 = stablehlo.constant dense<1> : tensor<i32>
  %1322 = stablehlo.reshape %1321 : (tensor<i32>) -> tensor<1xi32>
  %1323 = stablehlo.concatenate %1322, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1324 = stablehlo.real_dynamic_slice %1305, %1308, %1320, %1323 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1325 = stablehlo.constant dense<1> : tensor<1xi32>
  %1326 = stablehlo.constant dense<0> : tensor<i32>
  %1327 = stablehlo.reshape %1326 : (tensor<i32>) -> tensor<1xi32>
  %1328 = stablehlo.concatenate %1327, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1329 = stablehlo.get_dimension_size %1305, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %1330 = stablehlo.constant dense<1> : tensor<1xi32>
  %1331 = stablehlo.dynamic_reshape %1329, %1330 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1332 = stablehlo.concatenate %1331, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1333 = stablehlo.constant dense<0> : tensor<i32>
  %1334 = stablehlo.reshape %1333 : (tensor<i32>) -> tensor<1xi32>
  %1335 = stablehlo.concatenate %1334, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1336 = stablehlo.constant dense<1> : tensor<i32>
  %1337 = stablehlo.reshape %1336 : (tensor<i32>) -> tensor<1xi32>
  %1338 = stablehlo.get_dimension_size %1332, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %1339 = stablehlo.constant dense<1> : tensor<1xi32>
  %1340 = stablehlo.dynamic_reshape %1338, %1339 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1341 = stablehlo.concatenate %1340, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1342 = stablehlo.constant dense<0> : tensor<1xi32>
  %1343 = stablehlo.constant dense<1> : tensor<1xi32>
  %1344 = stablehlo.constant dense<1> : tensor<1xi32>
  %1345 = stablehlo.real_dynamic_slice %1341, %1342, %1343, %1344 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1346 = stablehlo.minimum %1337, %1345 : tensor<1xi32>
  %1347 = stablehlo.concatenate %1346, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1348 = stablehlo.constant dense<1> : tensor<i32>
  %1349 = stablehlo.reshape %1348 : (tensor<i32>) -> tensor<1xi32>
  %1350 = stablehlo.concatenate %1349, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1351 = stablehlo.real_dynamic_slice %1332, %1335, %1347, %1350 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1352 = stablehlo.constant dense<> : tensor<0xi32>
  %1353 = stablehlo.dynamic_reshape %1351, %1352 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %1354 = stablehlo.constant dense<> : tensor<0xi32>
  %1355 = stablehlo.constant dense<> : tensor<0xi32>
  %1356 = stablehlo.maximum %1354, %1355 : tensor<0xi32>
  %1357 = stablehlo.dynamic_broadcast_in_dim %1353, %1356, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %1358 = stablehlo.constant dense<0> : tensor<i32>
  %1359 = stablehlo.dynamic_broadcast_in_dim %1358, %1356, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %1360 = stablehlo.compare  GE, %1357, %1359 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1361 = stablehlo.constant dense<1> : tensor<i32>
  %1362 = stablehlo.constant dense<1> : tensor<1xi32>
  %1363 = stablehlo.dynamic_broadcast_in_dim %1361, %1362, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1364 = stablehlo.constant dense<> : tensor<0xi32>
  %1365 = stablehlo.concatenate %1363, %1364, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %1366 = stablehlo.dynamic_broadcast_in_dim %1360, %1365, dims = [] : (tensor<i1>, tensor<1xi32>) -> tensor<?xi1>
  %1367 = stablehlo.get_dimension_size %1366, dim = 0 : (tensor<?xi1>) -> tensor<i32>
  %1368 = stablehlo.constant dense<1> : tensor<1xi32>
  %1369 = stablehlo.dynamic_reshape %1367, %1368 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1370 = stablehlo.concatenate %1369, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1371 = stablehlo.constant dense<1> : tensor<i32>
  %1372 = stablehlo.constant dense<1> : tensor<1xi32>
  %1373 = stablehlo.dynamic_broadcast_in_dim %1371, %1372, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1374 = stablehlo.constant dense<> : tensor<0xi32>
  %1375 = stablehlo.concatenate %1373, %1374, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %1376 = stablehlo.dynamic_broadcast_in_dim %1353, %1375, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1377 = stablehlo.get_dimension_size %1376, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1378 = stablehlo.constant dense<1> : tensor<1xi32>
  %1379 = stablehlo.dynamic_reshape %1377, %1378 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1380 = stablehlo.concatenate %1379, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1381 = stablehlo.maximum %1370, %1380 : tensor<1xi32>
  %1382 = stablehlo.get_dimension_size %1376, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1383 = stablehlo.constant dense<1> : tensor<1xi32>
  %1384 = stablehlo.dynamic_reshape %1382, %1383 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1385 = stablehlo.concatenate %1384, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1386 = stablehlo.constant dense<0> : tensor<i32>
  %1387 = stablehlo.reshape %1386 : (tensor<i32>) -> tensor<1xi32>
  %1388 = stablehlo.concatenate %1387, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1389 = stablehlo.constant dense<1> : tensor<i32>
  %1390 = stablehlo.reshape %1389 : (tensor<i32>) -> tensor<1xi32>
  %1391 = stablehlo.get_dimension_size %1332, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %1392 = stablehlo.constant dense<1> : tensor<1xi32>
  %1393 = stablehlo.dynamic_reshape %1391, %1392 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1394 = stablehlo.concatenate %1393, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1395 = stablehlo.constant dense<0> : tensor<1xi32>
  %1396 = stablehlo.constant dense<1> : tensor<1xi32>
  %1397 = stablehlo.constant dense<1> : tensor<1xi32>
  %1398 = stablehlo.real_dynamic_slice %1394, %1395, %1396, %1397 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1399 = stablehlo.minimum %1390, %1398 : tensor<1xi32>
  %1400 = stablehlo.concatenate %1399, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1401 = stablehlo.constant dense<1> : tensor<i32>
  %1402 = stablehlo.reshape %1401 : (tensor<i32>) -> tensor<1xi32>
  %1403 = stablehlo.concatenate %1402, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1404 = stablehlo.real_dynamic_slice %1332, %1388, %1400, %1403 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1405 = stablehlo.constant dense<> : tensor<0xi32>
  %1406 = stablehlo.dynamic_reshape %1404, %1405 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %1407 = stablehlo.constant dense<1> : tensor<1xi32>
  %1408 = stablehlo.dynamic_reshape %1406, %1407 : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1409 = stablehlo.get_dimension_size %1408, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1410 = stablehlo.constant dense<1> : tensor<1xi32>
  %1411 = stablehlo.dynamic_reshape %1409, %1410 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1412 = stablehlo.concatenate %1411, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1413 = stablehlo.constant dense<1> : tensor<i32>
  %1414 = stablehlo.constant dense<1> : tensor<1xi32>
  %1415 = stablehlo.dynamic_broadcast_in_dim %1413, %1414, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1416 = stablehlo.constant dense<> : tensor<0xi32>
  %1417 = stablehlo.concatenate %1415, %1416, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %1418 = stablehlo.dynamic_broadcast_in_dim %1353, %1417, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1419 = stablehlo.get_dimension_size %1418, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1420 = stablehlo.constant dense<1> : tensor<1xi32>
  %1421 = stablehlo.dynamic_reshape %1419, %1420 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1422 = stablehlo.concatenate %1421, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1423 = stablehlo.maximum %1412, %1422 : tensor<1xi32>
  %1424 = stablehlo.dynamic_broadcast_in_dim %1408, %1423, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1425 = stablehlo.dynamic_broadcast_in_dim %1418, %1423, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1426 = stablehlo.add %1424, %1425 : tensor<?xi32>
  %1427 = stablehlo.get_dimension_size %1426, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1428 = stablehlo.constant dense<1> : tensor<1xi32>
  %1429 = stablehlo.dynamic_reshape %1427, %1428 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1430 = stablehlo.concatenate %1429, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1431 = stablehlo.maximum %1385, %1430 : tensor<1xi32>
  %1432 = stablehlo.maximum %1381, %1431 : tensor<1xi32>
  %1433 = stablehlo.dynamic_broadcast_in_dim %1366, %1432, dims = [0] : (tensor<?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %1434 = stablehlo.dynamic_broadcast_in_dim %1376, %1432, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1435 = stablehlo.dynamic_broadcast_in_dim %1426, %1432, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1436 = stablehlo.select %1433, %1434, %1435 : tensor<?xi1>, tensor<?xi32>
  %1437 = stablehlo.reshape %1436 : (tensor<?xi32>) -> tensor<1xi32>
  %1438 = stablehlo.get_dimension_size %1305, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %1439 = stablehlo.constant dense<1> : tensor<1xi32>
  %1440 = stablehlo.dynamic_reshape %1438, %1439 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1441 = stablehlo.concatenate %1440, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1442 = stablehlo.constant dense<0> : tensor<1xi32>
  %1443 = stablehlo.constant dense<1> : tensor<1xi32>
  %1444 = stablehlo.constant dense<1> : tensor<1xi32>
  %1445 = stablehlo.real_dynamic_slice %1441, %1442, %1443, %1444 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1446 = stablehlo.minimum %1437, %1445 : tensor<1xi32>
  %1447 = stablehlo.concatenate %1446, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1448 = stablehlo.constant dense<1> : tensor<i32>
  %1449 = stablehlo.reshape %1448 : (tensor<i32>) -> tensor<1xi32>
  %1450 = stablehlo.concatenate %1449, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1451 = stablehlo.real_dynamic_slice %1305, %1328, %1447, %1450 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1452 = stablehlo.concatenate %1324, %1325, %1451, dim = 0 : (tensor<?xi32>, tensor<1xi32>, tensor<?xi32>) -> tensor<2xi32>
  %1453 = stablehlo.dynamic_broadcast_in_dim %1301, %1452, dims = [1] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %1454 = stablehlo.constant dense<1> : tensor<i32>
  %1455 = stablehlo.constant dense<1> : tensor<1xi32>
  %1456 = stablehlo.dynamic_broadcast_in_dim %1454, %1455, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1457 = stablehlo.get_dimension_size %1453, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %1458 = stablehlo.constant dense<1> : tensor<1xi32>
  %1459 = stablehlo.dynamic_reshape %1457, %1458 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1460 = stablehlo.get_dimension_size %1453, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
  %1461 = stablehlo.constant dense<1> : tensor<1xi32>
  %1462 = stablehlo.dynamic_reshape %1460, %1461 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1463 = stablehlo.concatenate %1459, %1462, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %1464 = stablehlo.concatenate %1456, %1463, dim = 0 : (tensor<?xi32>, tensor<2xi32>) -> tensor<3xi32>
  %1465 = stablehlo.dynamic_broadcast_in_dim %1453, %1464, dims = [1, 2] : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1466 = stablehlo.get_dimension_size %1465, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1467 = stablehlo.constant dense<1> : tensor<1xi32>
  %1468 = stablehlo.dynamic_reshape %1466, %1467 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1469 = stablehlo.get_dimension_size %1465, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1470 = stablehlo.constant dense<1> : tensor<1xi32>
  %1471 = stablehlo.dynamic_reshape %1469, %1470 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1472 = stablehlo.get_dimension_size %1465, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1473 = stablehlo.constant dense<1> : tensor<1xi32>
  %1474 = stablehlo.dynamic_reshape %1472, %1473 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1475 = stablehlo.concatenate %1468, %1471, %1474, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1476 = stablehlo.maximum %1298, %1475 : tensor<3xi32>
  %1477 = stablehlo.dynamic_broadcast_in_dim %1288, %1476, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1478 = stablehlo.dynamic_broadcast_in_dim %1465, %1476, dims = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1479 = stablehlo.add %1477, %1478 : tensor<?x?x?xf32>
  %1480 = stablehlo.constant dense<0> : tensor<1xi32>
  %1481 = stablehlo.get_dimension_size %1479, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1482 = stablehlo.constant dense<1> : tensor<1xi32>
  %1483 = stablehlo.dynamic_reshape %1481, %1482 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1484 = stablehlo.get_dimension_size %1479, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1485 = stablehlo.constant dense<1> : tensor<1xi32>
  %1486 = stablehlo.dynamic_reshape %1484, %1485 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1487 = stablehlo.get_dimension_size %1479, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1488 = stablehlo.constant dense<1> : tensor<1xi32>
  %1489 = stablehlo.dynamic_reshape %1487, %1488 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1490 = stablehlo.concatenate %1483, %1486, %1489, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1491 = stablehlo.constant dense<2> : tensor<1xi32>
  %1492 = stablehlo.constant dense<3> : tensor<1xi32>
  %1493 = stablehlo.constant dense<1> : tensor<1xi32>
  %1494 = stablehlo.real_dynamic_slice %1490, %1491, %1492, %1493 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1495 = stablehlo.constant dense<3> : tensor<1xi32>
  %1496 = stablehlo.divide %1494, %1495 : tensor<1xi32>
  %1497 = stablehlo.constant dense<0> : tensor<1xi32>
  %1498 = stablehlo.multiply %1496, %1497 : tensor<1xi32>
  %1499 = stablehlo.concatenate %1480, %1480, %1498, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1500 = stablehlo.constant dense<0> : tensor<1xi32>
  %1501 = stablehlo.constant dense<1> : tensor<1xi32>
  %1502 = stablehlo.constant dense<1> : tensor<1xi32>
  %1503 = stablehlo.real_dynamic_slice %1490, %1500, %1501, %1502 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1504 = stablehlo.constant dense<1> : tensor<1xi32>
  %1505 = stablehlo.constant dense<2> : tensor<1xi32>
  %1506 = stablehlo.constant dense<1> : tensor<1xi32>
  %1507 = stablehlo.real_dynamic_slice %1490, %1504, %1505, %1506 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1508 = stablehlo.constant dense<1> : tensor<1xi32>
  %1509 = stablehlo.multiply %1496, %1508 : tensor<1xi32>
  %1510 = stablehlo.concatenate %1503, %1507, %1509, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1511 = stablehlo.constant dense<1> : tensor<1xi32>
  %1512 = stablehlo.concatenate %1511, %1511, %1511, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1513 = stablehlo.real_dynamic_slice %1479, %1499, %1510, %1512 : (tensor<?x?x?xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1514 = stablehlo.get_dimension_size %1513, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1515 = stablehlo.constant dense<1> : tensor<1xi32>
  %1516 = stablehlo.dynamic_reshape %1514, %1515 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1517 = stablehlo.get_dimension_size %1513, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1518 = stablehlo.constant dense<1> : tensor<1xi32>
  %1519 = stablehlo.dynamic_reshape %1517, %1518 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1520 = stablehlo.get_dimension_size %1513, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1521 = stablehlo.constant dense<1> : tensor<1xi32>
  %1522 = stablehlo.dynamic_reshape %1520, %1521 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1523 = stablehlo.concatenate %1516, %1519, %1522, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1524 = stablehlo.constant dense<0> : tensor<i32>
  %1525 = stablehlo.reshape %1524 : (tensor<i32>) -> tensor<1xi32>
  %1526 = stablehlo.concatenate %1525, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1527 = stablehlo.constant dense<1> : tensor<i32>
  %1528 = stablehlo.reshape %1527 : (tensor<i32>) -> tensor<1xi32>
  %1529 = stablehlo.get_dimension_size %1523, dim = 0 : (tensor<3xi32>) -> tensor<i32>
  %1530 = stablehlo.constant dense<1> : tensor<1xi32>
  %1531 = stablehlo.dynamic_reshape %1529, %1530 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1532 = stablehlo.concatenate %1531, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1533 = stablehlo.constant dense<0> : tensor<1xi32>
  %1534 = stablehlo.constant dense<1> : tensor<1xi32>
  %1535 = stablehlo.constant dense<1> : tensor<1xi32>
  %1536 = stablehlo.real_dynamic_slice %1532, %1533, %1534, %1535 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1537 = stablehlo.minimum %1528, %1536 : tensor<1xi32>
  %1538 = stablehlo.concatenate %1537, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1539 = stablehlo.constant dense<1> : tensor<i32>
  %1540 = stablehlo.reshape %1539 : (tensor<i32>) -> tensor<1xi32>
  %1541 = stablehlo.concatenate %1540, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1542 = stablehlo.real_dynamic_slice %1523, %1526, %1538, %1541 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1543 = stablehlo.constant dense<> : tensor<0xi32>
  %1544 = stablehlo.dynamic_reshape %1542, %1543 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %1545 = stablehlo.constant dense<1> : tensor<1xi32>
  %1546 = stablehlo.dynamic_broadcast_in_dim %1544, %1545, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1547 = stablehlo.get_dimension_size %1513, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1548 = stablehlo.constant dense<1> : tensor<1xi32>
  %1549 = stablehlo.dynamic_reshape %1547, %1548 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1550 = stablehlo.get_dimension_size %1513, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1551 = stablehlo.constant dense<1> : tensor<1xi32>
  %1552 = stablehlo.dynamic_reshape %1550, %1551 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1553 = stablehlo.get_dimension_size %1513, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1554 = stablehlo.constant dense<1> : tensor<1xi32>
  %1555 = stablehlo.dynamic_reshape %1553, %1554 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1556 = stablehlo.concatenate %1549, %1552, %1555, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1557 = stablehlo.constant dense<1> : tensor<i32>
  %1558 = stablehlo.reshape %1557 : (tensor<i32>) -> tensor<1xi32>
  %1559 = stablehlo.concatenate %1558, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1560 = stablehlo.constant dense<2> : tensor<i32>
  %1561 = stablehlo.reshape %1560 : (tensor<i32>) -> tensor<1xi32>
  %1562 = stablehlo.get_dimension_size %1556, dim = 0 : (tensor<3xi32>) -> tensor<i32>
  %1563 = stablehlo.constant dense<1> : tensor<1xi32>
  %1564 = stablehlo.dynamic_reshape %1562, %1563 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1565 = stablehlo.concatenate %1564, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1566 = stablehlo.constant dense<0> : tensor<1xi32>
  %1567 = stablehlo.constant dense<1> : tensor<1xi32>
  %1568 = stablehlo.constant dense<1> : tensor<1xi32>
  %1569 = stablehlo.real_dynamic_slice %1565, %1566, %1567, %1568 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1570 = stablehlo.minimum %1561, %1569 : tensor<1xi32>
  %1571 = stablehlo.concatenate %1570, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1572 = stablehlo.constant dense<1> : tensor<i32>
  %1573 = stablehlo.reshape %1572 : (tensor<i32>) -> tensor<1xi32>
  %1574 = stablehlo.concatenate %1573, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1575 = stablehlo.real_dynamic_slice %1556, %1559, %1571, %1574 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1576 = stablehlo.constant dense<> : tensor<0xi32>
  %1577 = stablehlo.dynamic_reshape %1575, %1576 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %1578 = stablehlo.constant dense<1> : tensor<1xi32>
  %1579 = stablehlo.dynamic_broadcast_in_dim %1577, %1578, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1580 = stablehlo.constant dense<16> : tensor<1xi32>
  %1581 = stablehlo.constant dense<64> : tensor<1xi32>
  %1582 = stablehlo.concatenate %1546, %1579, %1580, %1581, dim = 0 : (tensor<?xi32>, tensor<?xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %1583 = stablehlo.dynamic_reshape %1513, %1582 : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1584 = stablehlo.transpose %1583, dims = [0, 2, 1, 3] : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1585 = stablehlo.constant dense<> : tensor<0xi32>
  %1586 = stablehlo.get_dimension_size %1584, dim = 0 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1587 = stablehlo.constant dense<1> : tensor<1xi32>
  %1588 = stablehlo.dynamic_reshape %1586, %1587 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1589 = stablehlo.get_dimension_size %1584, dim = 1 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1590 = stablehlo.constant dense<1> : tensor<1xi32>
  %1591 = stablehlo.dynamic_reshape %1589, %1590 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1592 = stablehlo.get_dimension_size %1584, dim = 2 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1593 = stablehlo.constant dense<1> : tensor<1xi32>
  %1594 = stablehlo.dynamic_reshape %1592, %1593 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1595 = stablehlo.get_dimension_size %1584, dim = 3 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1596 = stablehlo.constant dense<1> : tensor<1xi32>
  %1597 = stablehlo.dynamic_reshape %1595, %1596 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1598 = stablehlo.concatenate %1588, %1591, %1594, %1597, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %1599 = stablehlo.constant dense<0> : tensor<1xi32>
  %1600 = stablehlo.constant dense<2> : tensor<1xi32>
  %1601 = stablehlo.constant dense<1> : tensor<1xi32>
  %1602 = stablehlo.real_dynamic_slice %1598, %1599, %1600, %1601 : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1603 = stablehlo.concatenate %1585, %1602, dim = 0 : (tensor<0xi32>, tensor<?xi32>) -> tensor<?xi32>
  %1604 = stablehlo.constant dense<> : tensor<0xi32>
  %1605 = stablehlo.constant dense<0> : tensor<1xi32>
  %1606 = stablehlo.constant dense<1> : tensor<1xi32>
  %1607 = stablehlo.multiply %1496, %1606 : tensor<1xi32>
  %1608 = stablehlo.concatenate %1605, %1605, %1607, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1609 = stablehlo.constant dense<0> : tensor<1xi32>
  %1610 = stablehlo.constant dense<1> : tensor<1xi32>
  %1611 = stablehlo.constant dense<1> : tensor<1xi32>
  %1612 = stablehlo.real_dynamic_slice %1490, %1609, %1610, %1611 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1613 = stablehlo.constant dense<1> : tensor<1xi32>
  %1614 = stablehlo.constant dense<2> : tensor<1xi32>
  %1615 = stablehlo.constant dense<1> : tensor<1xi32>
  %1616 = stablehlo.real_dynamic_slice %1490, %1613, %1614, %1615 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1617 = stablehlo.constant dense<2> : tensor<1xi32>
  %1618 = stablehlo.multiply %1496, %1617 : tensor<1xi32>
  %1619 = stablehlo.concatenate %1612, %1616, %1618, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1620 = stablehlo.constant dense<1> : tensor<1xi32>
  %1621 = stablehlo.concatenate %1620, %1620, %1620, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1622 = stablehlo.real_dynamic_slice %1479, %1608, %1619, %1621 : (tensor<?x?x?xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1623 = stablehlo.get_dimension_size %1622, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1624 = stablehlo.constant dense<1> : tensor<1xi32>
  %1625 = stablehlo.dynamic_reshape %1623, %1624 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1626 = stablehlo.get_dimension_size %1622, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1627 = stablehlo.constant dense<1> : tensor<1xi32>
  %1628 = stablehlo.dynamic_reshape %1626, %1627 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1629 = stablehlo.get_dimension_size %1622, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1630 = stablehlo.constant dense<1> : tensor<1xi32>
  %1631 = stablehlo.dynamic_reshape %1629, %1630 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1632 = stablehlo.concatenate %1625, %1628, %1631, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1633 = stablehlo.constant dense<0> : tensor<i32>
  %1634 = stablehlo.reshape %1633 : (tensor<i32>) -> tensor<1xi32>
  %1635 = stablehlo.concatenate %1634, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1636 = stablehlo.constant dense<1> : tensor<i32>
  %1637 = stablehlo.reshape %1636 : (tensor<i32>) -> tensor<1xi32>
  %1638 = stablehlo.get_dimension_size %1632, dim = 0 : (tensor<3xi32>) -> tensor<i32>
  %1639 = stablehlo.constant dense<1> : tensor<1xi32>
  %1640 = stablehlo.dynamic_reshape %1638, %1639 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1641 = stablehlo.concatenate %1640, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1642 = stablehlo.constant dense<0> : tensor<1xi32>
  %1643 = stablehlo.constant dense<1> : tensor<1xi32>
  %1644 = stablehlo.constant dense<1> : tensor<1xi32>
  %1645 = stablehlo.real_dynamic_slice %1641, %1642, %1643, %1644 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1646 = stablehlo.minimum %1637, %1645 : tensor<1xi32>
  %1647 = stablehlo.concatenate %1646, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1648 = stablehlo.constant dense<1> : tensor<i32>
  %1649 = stablehlo.reshape %1648 : (tensor<i32>) -> tensor<1xi32>
  %1650 = stablehlo.concatenate %1649, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1651 = stablehlo.real_dynamic_slice %1632, %1635, %1647, %1650 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1652 = stablehlo.constant dense<> : tensor<0xi32>
  %1653 = stablehlo.dynamic_reshape %1651, %1652 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %1654 = stablehlo.constant dense<1> : tensor<1xi32>
  %1655 = stablehlo.dynamic_broadcast_in_dim %1653, %1654, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1656 = stablehlo.get_dimension_size %1622, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1657 = stablehlo.constant dense<1> : tensor<1xi32>
  %1658 = stablehlo.dynamic_reshape %1656, %1657 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1659 = stablehlo.get_dimension_size %1622, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1660 = stablehlo.constant dense<1> : tensor<1xi32>
  %1661 = stablehlo.dynamic_reshape %1659, %1660 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1662 = stablehlo.get_dimension_size %1622, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %1663 = stablehlo.constant dense<1> : tensor<1xi32>
  %1664 = stablehlo.dynamic_reshape %1662, %1663 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1665 = stablehlo.concatenate %1658, %1661, %1664, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %1666 = stablehlo.constant dense<1> : tensor<i32>
  %1667 = stablehlo.reshape %1666 : (tensor<i32>) -> tensor<1xi32>
  %1668 = stablehlo.concatenate %1667, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1669 = stablehlo.constant dense<2> : tensor<i32>
  %1670 = stablehlo.reshape %1669 : (tensor<i32>) -> tensor<1xi32>
  %1671 = stablehlo.get_dimension_size %1665, dim = 0 : (tensor<3xi32>) -> tensor<i32>
  %1672 = stablehlo.constant dense<1> : tensor<1xi32>
  %1673 = stablehlo.dynamic_reshape %1671, %1672 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1674 = stablehlo.concatenate %1673, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1675 = stablehlo.constant dense<0> : tensor<1xi32>
  %1676 = stablehlo.constant dense<1> : tensor<1xi32>
  %1677 = stablehlo.constant dense<1> : tensor<1xi32>
  %1678 = stablehlo.real_dynamic_slice %1674, %1675, %1676, %1677 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1679 = stablehlo.minimum %1670, %1678 : tensor<1xi32>
  %1680 = stablehlo.concatenate %1679, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1681 = stablehlo.constant dense<1> : tensor<i32>
  %1682 = stablehlo.reshape %1681 : (tensor<i32>) -> tensor<1xi32>
  %1683 = stablehlo.concatenate %1682, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1684 = stablehlo.real_dynamic_slice %1665, %1668, %1680, %1683 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1685 = stablehlo.constant dense<> : tensor<0xi32>
  %1686 = stablehlo.dynamic_reshape %1684, %1685 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %1687 = stablehlo.constant dense<1> : tensor<1xi32>
  %1688 = stablehlo.dynamic_broadcast_in_dim %1686, %1687, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1689 = stablehlo.constant dense<16> : tensor<1xi32>
  %1690 = stablehlo.constant dense<64> : tensor<1xi32>
  %1691 = stablehlo.concatenate %1655, %1688, %1689, %1690, dim = 0 : (tensor<?xi32>, tensor<?xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %1692 = stablehlo.dynamic_reshape %1622, %1691 : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1693 = stablehlo.transpose %1692, dims = [0, 2, 1, 3] : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1694 = stablehlo.transpose %1693, dims = [0, 1, 3, 2] : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1695 = stablehlo.get_dimension_size %1694, dim = 0 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1696 = stablehlo.constant dense<1> : tensor<1xi32>
  %1697 = stablehlo.dynamic_reshape %1695, %1696 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1698 = stablehlo.get_dimension_size %1694, dim = 1 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1699 = stablehlo.constant dense<1> : tensor<1xi32>
  %1700 = stablehlo.dynamic_reshape %1698, %1699 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1701 = stablehlo.get_dimension_size %1694, dim = 2 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1702 = stablehlo.constant dense<1> : tensor<1xi32>
  %1703 = stablehlo.dynamic_reshape %1701, %1702 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1704 = stablehlo.get_dimension_size %1694, dim = 3 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1705 = stablehlo.constant dense<1> : tensor<1xi32>
  %1706 = stablehlo.dynamic_reshape %1704, %1705 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1707 = stablehlo.concatenate %1697, %1700, %1703, %1706, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %1708 = stablehlo.constant dense<0> : tensor<1xi32>
  %1709 = stablehlo.constant dense<2> : tensor<1xi32>
  %1710 = stablehlo.constant dense<1> : tensor<1xi32>
  %1711 = stablehlo.real_dynamic_slice %1707, %1708, %1709, %1710 : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1712 = stablehlo.concatenate %1604, %1711, dim = 0 : (tensor<0xi32>, tensor<?xi32>) -> tensor<?xi32>
  %1713 = stablehlo.maximum %1603, %1712 : tensor<?xi32>
  %1714 = stablehlo.constant dense<4> : tensor<1xi32>
  %1715 = stablehlo.real_dynamic_slice %1598, %1600, %1714, %1601 : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1716 = stablehlo.concatenate %1713, %1715, dim = 0 : (tensor<?xi32>, tensor<?xi32>) -> tensor<4xi32>
  %1717 = stablehlo.dynamic_broadcast_in_dim %1584, %1716, dims = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1718 = stablehlo.constant dense<4> : tensor<1xi32>
  %1719 = stablehlo.real_dynamic_slice %1707, %1709, %1718, %1710 : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1720 = stablehlo.concatenate %1713, %1719, dim = 0 : (tensor<?xi32>, tensor<?xi32>) -> tensor<4xi32>
  %1721 = stablehlo.dynamic_broadcast_in_dim %1694, %1720, dims = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1722 = stablehlo.dot_general %1717, %1721, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1723 = stablehlo.get_dimension_size %1722, dim = 0 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1724 = stablehlo.constant dense<1> : tensor<1xi32>
  %1725 = stablehlo.dynamic_reshape %1723, %1724 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1726 = stablehlo.get_dimension_size %1722, dim = 1 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1727 = stablehlo.constant dense<1> : tensor<1xi32>
  %1728 = stablehlo.dynamic_reshape %1726, %1727 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1729 = stablehlo.get_dimension_size %1722, dim = 2 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1730 = stablehlo.constant dense<1> : tensor<1xi32>
  %1731 = stablehlo.dynamic_reshape %1729, %1730 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1732 = stablehlo.get_dimension_size %1722, dim = 3 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1733 = stablehlo.constant dense<1> : tensor<1xi32>
  %1734 = stablehlo.dynamic_reshape %1732, %1733 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1735 = stablehlo.concatenate %1725, %1728, %1731, %1734, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %1736 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
  %1737 = stablehlo.constant dense<1> : tensor<i32>
  %1738 = stablehlo.constant dense<4> : tensor<1xi32>
  %1739 = stablehlo.dynamic_broadcast_in_dim %1737, %1738, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1740 = stablehlo.constant dense<> : tensor<0xi32>
  %1741 = stablehlo.concatenate %1739, %1740, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<4xi32>
  %1742 = stablehlo.dynamic_broadcast_in_dim %1736, %1741, dims = [] : (tensor<f32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1743 = stablehlo.get_dimension_size %1742, dim = 0 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1744 = stablehlo.constant dense<1> : tensor<1xi32>
  %1745 = stablehlo.dynamic_reshape %1743, %1744 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1746 = stablehlo.get_dimension_size %1742, dim = 1 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1747 = stablehlo.constant dense<1> : tensor<1xi32>
  %1748 = stablehlo.dynamic_reshape %1746, %1747 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1749 = stablehlo.get_dimension_size %1742, dim = 2 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1750 = stablehlo.constant dense<1> : tensor<1xi32>
  %1751 = stablehlo.dynamic_reshape %1749, %1750 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1752 = stablehlo.get_dimension_size %1742, dim = 3 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1753 = stablehlo.constant dense<1> : tensor<1xi32>
  %1754 = stablehlo.dynamic_reshape %1752, %1753 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1755 = stablehlo.concatenate %1745, %1748, %1751, %1754, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %1756 = stablehlo.maximum %1735, %1755 : tensor<4xi32>
  %1757 = stablehlo.dynamic_broadcast_in_dim %1722, %1756, dims = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1758 = stablehlo.dynamic_broadcast_in_dim %1742, %1756, dims = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1759 = stablehlo.divide %1757, %1758 : tensor<?x?x?x?xf32>
  %1760 = stablehlo.get_dimension_size %1759, dim = 0 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1761 = stablehlo.constant dense<1> : tensor<1xi32>
  %1762 = stablehlo.dynamic_reshape %1760, %1761 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1763 = stablehlo.get_dimension_size %1759, dim = 1 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1764 = stablehlo.constant dense<1> : tensor<1xi32>
  %1765 = stablehlo.dynamic_reshape %1763, %1764 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1766 = stablehlo.get_dimension_size %1759, dim = 2 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1767 = stablehlo.constant dense<1> : tensor<1xi32>
  %1768 = stablehlo.dynamic_reshape %1766, %1767 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1769 = stablehlo.get_dimension_size %1759, dim = 3 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %1770 = stablehlo.constant dense<1> : tensor<1xi32>
  %1771 = stablehlo.dynamic_reshape %1769, %1770 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1772 = stablehlo.concatenate %1762, %1765, %1768, %1771, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %1773 = stablehlo.constant dense<0> : tensor<i32>
  %1774 = stablehlo.reshape %1773 : (tensor<i32>) -> tensor<1xi32>
  %1775 = stablehlo.constant dense<0> : tensor<i32>
  %1776 = stablehlo.reshape %1775 : (tensor<i32>) -> tensor<1xi32>
  %1777 = stablehlo.concatenate %1774, %1776, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %1778 = stablehlo.get_dimension_size %50, dim = 0 : (tensor<1x1024xi32>) -> tensor<i32>
  %1779 = stablehlo.constant dense<1> : tensor<1xi32>
  %1780 = stablehlo.dynamic_reshape %1778, %1779 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1781 = stablehlo.get_dimension_size %50, dim = 1 : (tensor<1x1024xi32>) -> tensor<i32>
  %1782 = stablehlo.constant dense<1> : tensor<1xi32>
  %1783 = stablehlo.dynamic_reshape %1781, %1782 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1784 = stablehlo.concatenate %1780, %1783, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %1785 = stablehlo.constant dense<0> : tensor<i32>
  %1786 = stablehlo.reshape %1785 : (tensor<i32>) -> tensor<1xi32>
  %1787 = stablehlo.concatenate %1786, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1788 = stablehlo.constant dense<1> : tensor<i32>
  %1789 = stablehlo.reshape %1788 : (tensor<i32>) -> tensor<1xi32>
  %1790 = stablehlo.get_dimension_size %1784, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %1791 = stablehlo.constant dense<1> : tensor<1xi32>
  %1792 = stablehlo.dynamic_reshape %1790, %1791 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1793 = stablehlo.concatenate %1792, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1794 = stablehlo.constant dense<0> : tensor<1xi32>
  %1795 = stablehlo.constant dense<1> : tensor<1xi32>
  %1796 = stablehlo.constant dense<1> : tensor<1xi32>
  %1797 = stablehlo.real_dynamic_slice %1793, %1794, %1795, %1796 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1798 = stablehlo.minimum %1789, %1797 : tensor<1xi32>
  %1799 = stablehlo.concatenate %1798, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1800 = stablehlo.constant dense<1> : tensor<i32>
  %1801 = stablehlo.reshape %1800 : (tensor<i32>) -> tensor<1xi32>
  %1802 = stablehlo.concatenate %1801, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1803 = stablehlo.real_dynamic_slice %1784, %1787, %1799, %1802 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1804 = stablehlo.constant dense<> : tensor<0xi32>
  %1805 = stablehlo.dynamic_reshape %1803, %1804 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %1806 = stablehlo.constant dense<> : tensor<0xi32>
  %1807 = stablehlo.constant dense<> : tensor<0xi32>
  %1808 = stablehlo.maximum %1806, %1807 : tensor<0xi32>
  %1809 = stablehlo.dynamic_broadcast_in_dim %1805, %1808, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %1810 = stablehlo.constant dense<0> : tensor<i32>
  %1811 = stablehlo.dynamic_broadcast_in_dim %1810, %1808, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %1812 = stablehlo.compare  GE, %1809, %1811 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1813 = stablehlo.constant dense<1> : tensor<i32>
  %1814 = stablehlo.constant dense<1> : tensor<1xi32>
  %1815 = stablehlo.dynamic_broadcast_in_dim %1813, %1814, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1816 = stablehlo.constant dense<> : tensor<0xi32>
  %1817 = stablehlo.concatenate %1815, %1816, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %1818 = stablehlo.dynamic_broadcast_in_dim %1812, %1817, dims = [] : (tensor<i1>, tensor<1xi32>) -> tensor<?xi1>
  %1819 = stablehlo.get_dimension_size %1818, dim = 0 : (tensor<?xi1>) -> tensor<i32>
  %1820 = stablehlo.constant dense<1> : tensor<1xi32>
  %1821 = stablehlo.dynamic_reshape %1819, %1820 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1822 = stablehlo.concatenate %1821, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1823 = stablehlo.constant dense<1> : tensor<i32>
  %1824 = stablehlo.constant dense<1> : tensor<1xi32>
  %1825 = stablehlo.dynamic_broadcast_in_dim %1823, %1824, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1826 = stablehlo.constant dense<> : tensor<0xi32>
  %1827 = stablehlo.concatenate %1825, %1826, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %1828 = stablehlo.dynamic_broadcast_in_dim %1805, %1827, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1829 = stablehlo.get_dimension_size %1828, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1830 = stablehlo.constant dense<1> : tensor<1xi32>
  %1831 = stablehlo.dynamic_reshape %1829, %1830 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1832 = stablehlo.concatenate %1831, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1833 = stablehlo.maximum %1822, %1832 : tensor<1xi32>
  %1834 = stablehlo.get_dimension_size %1828, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1835 = stablehlo.constant dense<1> : tensor<1xi32>
  %1836 = stablehlo.dynamic_reshape %1834, %1835 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1837 = stablehlo.concatenate %1836, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1838 = stablehlo.constant dense<0> : tensor<i32>
  %1839 = stablehlo.reshape %1838 : (tensor<i32>) -> tensor<1xi32>
  %1840 = stablehlo.concatenate %1839, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1841 = stablehlo.constant dense<1> : tensor<i32>
  %1842 = stablehlo.reshape %1841 : (tensor<i32>) -> tensor<1xi32>
  %1843 = stablehlo.get_dimension_size %1784, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %1844 = stablehlo.constant dense<1> : tensor<1xi32>
  %1845 = stablehlo.dynamic_reshape %1843, %1844 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1846 = stablehlo.concatenate %1845, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1847 = stablehlo.constant dense<0> : tensor<1xi32>
  %1848 = stablehlo.constant dense<1> : tensor<1xi32>
  %1849 = stablehlo.constant dense<1> : tensor<1xi32>
  %1850 = stablehlo.real_dynamic_slice %1846, %1847, %1848, %1849 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1851 = stablehlo.minimum %1842, %1850 : tensor<1xi32>
  %1852 = stablehlo.concatenate %1851, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1853 = stablehlo.constant dense<1> : tensor<i32>
  %1854 = stablehlo.reshape %1853 : (tensor<i32>) -> tensor<1xi32>
  %1855 = stablehlo.concatenate %1854, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1856 = stablehlo.real_dynamic_slice %1784, %1840, %1852, %1855 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1857 = stablehlo.constant dense<> : tensor<0xi32>
  %1858 = stablehlo.dynamic_reshape %1856, %1857 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %1859 = stablehlo.constant dense<1> : tensor<1xi32>
  %1860 = stablehlo.dynamic_reshape %1858, %1859 : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1861 = stablehlo.get_dimension_size %1860, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1862 = stablehlo.constant dense<1> : tensor<1xi32>
  %1863 = stablehlo.dynamic_reshape %1861, %1862 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1864 = stablehlo.concatenate %1863, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1865 = stablehlo.constant dense<1> : tensor<i32>
  %1866 = stablehlo.constant dense<1> : tensor<1xi32>
  %1867 = stablehlo.dynamic_broadcast_in_dim %1865, %1866, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1868 = stablehlo.constant dense<> : tensor<0xi32>
  %1869 = stablehlo.concatenate %1867, %1868, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %1870 = stablehlo.dynamic_broadcast_in_dim %1805, %1869, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1871 = stablehlo.get_dimension_size %1870, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1872 = stablehlo.constant dense<1> : tensor<1xi32>
  %1873 = stablehlo.dynamic_reshape %1871, %1872 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1874 = stablehlo.concatenate %1873, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1875 = stablehlo.maximum %1864, %1874 : tensor<1xi32>
  %1876 = stablehlo.dynamic_broadcast_in_dim %1860, %1875, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1877 = stablehlo.dynamic_broadcast_in_dim %1870, %1875, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1878 = stablehlo.add %1876, %1877 : tensor<?xi32>
  %1879 = stablehlo.get_dimension_size %1878, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1880 = stablehlo.constant dense<1> : tensor<1xi32>
  %1881 = stablehlo.dynamic_reshape %1879, %1880 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1882 = stablehlo.concatenate %1881, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1883 = stablehlo.maximum %1837, %1882 : tensor<1xi32>
  %1884 = stablehlo.maximum %1833, %1883 : tensor<1xi32>
  %1885 = stablehlo.dynamic_broadcast_in_dim %1818, %1884, dims = [0] : (tensor<?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %1886 = stablehlo.dynamic_broadcast_in_dim %1828, %1884, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1887 = stablehlo.dynamic_broadcast_in_dim %1878, %1884, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1888 = stablehlo.select %1885, %1886, %1887 : tensor<?xi1>, tensor<?xi32>
  %1889 = stablehlo.reshape %1888 : (tensor<?xi32>) -> tensor<1xi32>
  %1890 = stablehlo.get_dimension_size %50, dim = 0 : (tensor<1x1024xi32>) -> tensor<i32>
  %1891 = stablehlo.constant dense<1> : tensor<1xi32>
  %1892 = stablehlo.dynamic_reshape %1890, %1891 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1893 = stablehlo.get_dimension_size %50, dim = 1 : (tensor<1x1024xi32>) -> tensor<i32>
  %1894 = stablehlo.constant dense<1> : tensor<1xi32>
  %1895 = stablehlo.dynamic_reshape %1893, %1894 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1896 = stablehlo.concatenate %1892, %1895, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %1897 = stablehlo.constant dense<0> : tensor<1xi32>
  %1898 = stablehlo.constant dense<1> : tensor<1xi32>
  %1899 = stablehlo.constant dense<1> : tensor<1xi32>
  %1900 = stablehlo.real_dynamic_slice %1896, %1897, %1898, %1899 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1901 = stablehlo.minimum %1889, %1900 : tensor<1xi32>
  %1902 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<1x128xi32>) -> tensor<i32>
  %1903 = stablehlo.constant dense<1> : tensor<1xi32>
  %1904 = stablehlo.dynamic_reshape %1902, %1903 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1905 = stablehlo.get_dimension_size %24, dim = 1 : (tensor<1x128xi32>) -> tensor<i32>
  %1906 = stablehlo.constant dense<1> : tensor<1xi32>
  %1907 = stablehlo.dynamic_reshape %1905, %1906 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1908 = stablehlo.concatenate %1904, %1907, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %1909 = stablehlo.constant dense<1> : tensor<i32>
  %1910 = stablehlo.reshape %1909 : (tensor<i32>) -> tensor<1xi32>
  %1911 = stablehlo.concatenate %1910, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1912 = stablehlo.constant dense<2> : tensor<i32>
  %1913 = stablehlo.reshape %1912 : (tensor<i32>) -> tensor<1xi32>
  %1914 = stablehlo.get_dimension_size %1908, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %1915 = stablehlo.constant dense<1> : tensor<1xi32>
  %1916 = stablehlo.dynamic_reshape %1914, %1915 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1917 = stablehlo.concatenate %1916, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1918 = stablehlo.constant dense<0> : tensor<1xi32>
  %1919 = stablehlo.constant dense<1> : tensor<1xi32>
  %1920 = stablehlo.constant dense<1> : tensor<1xi32>
  %1921 = stablehlo.real_dynamic_slice %1917, %1918, %1919, %1920 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1922 = stablehlo.minimum %1913, %1921 : tensor<1xi32>
  %1923 = stablehlo.concatenate %1922, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1924 = stablehlo.constant dense<1> : tensor<i32>
  %1925 = stablehlo.reshape %1924 : (tensor<i32>) -> tensor<1xi32>
  %1926 = stablehlo.concatenate %1925, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1927 = stablehlo.real_dynamic_slice %1908, %1911, %1923, %1926 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1928 = stablehlo.constant dense<> : tensor<0xi32>
  %1929 = stablehlo.dynamic_reshape %1927, %1928 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %1930 = stablehlo.constant dense<> : tensor<0xi32>
  %1931 = stablehlo.constant dense<> : tensor<0xi32>
  %1932 = stablehlo.maximum %1930, %1931 : tensor<0xi32>
  %1933 = stablehlo.dynamic_broadcast_in_dim %1929, %1932, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %1934 = stablehlo.constant dense<0> : tensor<i32>
  %1935 = stablehlo.dynamic_broadcast_in_dim %1934, %1932, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %1936 = stablehlo.compare  GE, %1933, %1935 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1937 = stablehlo.constant dense<1> : tensor<i32>
  %1938 = stablehlo.constant dense<1> : tensor<1xi32>
  %1939 = stablehlo.dynamic_broadcast_in_dim %1937, %1938, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1940 = stablehlo.constant dense<> : tensor<0xi32>
  %1941 = stablehlo.concatenate %1939, %1940, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %1942 = stablehlo.dynamic_broadcast_in_dim %1936, %1941, dims = [] : (tensor<i1>, tensor<1xi32>) -> tensor<?xi1>
  %1943 = stablehlo.get_dimension_size %1942, dim = 0 : (tensor<?xi1>) -> tensor<i32>
  %1944 = stablehlo.constant dense<1> : tensor<1xi32>
  %1945 = stablehlo.dynamic_reshape %1943, %1944 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1946 = stablehlo.concatenate %1945, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1947 = stablehlo.constant dense<1> : tensor<i32>
  %1948 = stablehlo.constant dense<1> : tensor<1xi32>
  %1949 = stablehlo.dynamic_broadcast_in_dim %1947, %1948, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1950 = stablehlo.constant dense<> : tensor<0xi32>
  %1951 = stablehlo.concatenate %1949, %1950, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %1952 = stablehlo.dynamic_broadcast_in_dim %1929, %1951, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1953 = stablehlo.get_dimension_size %1952, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1954 = stablehlo.constant dense<1> : tensor<1xi32>
  %1955 = stablehlo.dynamic_reshape %1953, %1954 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1956 = stablehlo.concatenate %1955, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1957 = stablehlo.maximum %1946, %1956 : tensor<1xi32>
  %1958 = stablehlo.get_dimension_size %1952, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1959 = stablehlo.constant dense<1> : tensor<1xi32>
  %1960 = stablehlo.dynamic_reshape %1958, %1959 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1961 = stablehlo.concatenate %1960, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1962 = stablehlo.constant dense<1> : tensor<i32>
  %1963 = stablehlo.reshape %1962 : (tensor<i32>) -> tensor<1xi32>
  %1964 = stablehlo.concatenate %1963, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1965 = stablehlo.constant dense<2> : tensor<i32>
  %1966 = stablehlo.reshape %1965 : (tensor<i32>) -> tensor<1xi32>
  %1967 = stablehlo.get_dimension_size %1784, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %1968 = stablehlo.constant dense<1> : tensor<1xi32>
  %1969 = stablehlo.dynamic_reshape %1967, %1968 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1970 = stablehlo.concatenate %1969, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1971 = stablehlo.constant dense<0> : tensor<1xi32>
  %1972 = stablehlo.constant dense<1> : tensor<1xi32>
  %1973 = stablehlo.constant dense<1> : tensor<1xi32>
  %1974 = stablehlo.real_dynamic_slice %1970, %1971, %1972, %1973 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1975 = stablehlo.minimum %1966, %1974 : tensor<1xi32>
  %1976 = stablehlo.concatenate %1975, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1977 = stablehlo.constant dense<1> : tensor<i32>
  %1978 = stablehlo.reshape %1977 : (tensor<i32>) -> tensor<1xi32>
  %1979 = stablehlo.concatenate %1978, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1980 = stablehlo.real_dynamic_slice %1784, %1964, %1976, %1979 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1981 = stablehlo.constant dense<> : tensor<0xi32>
  %1982 = stablehlo.dynamic_reshape %1980, %1981 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %1983 = stablehlo.constant dense<1> : tensor<1xi32>
  %1984 = stablehlo.dynamic_reshape %1982, %1983 : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1985 = stablehlo.get_dimension_size %1984, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1986 = stablehlo.constant dense<1> : tensor<1xi32>
  %1987 = stablehlo.dynamic_reshape %1985, %1986 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1988 = stablehlo.concatenate %1987, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1989 = stablehlo.constant dense<1> : tensor<i32>
  %1990 = stablehlo.constant dense<1> : tensor<1xi32>
  %1991 = stablehlo.dynamic_broadcast_in_dim %1989, %1990, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1992 = stablehlo.constant dense<> : tensor<0xi32>
  %1993 = stablehlo.concatenate %1991, %1992, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %1994 = stablehlo.dynamic_broadcast_in_dim %1929, %1993, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1995 = stablehlo.get_dimension_size %1994, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1996 = stablehlo.constant dense<1> : tensor<1xi32>
  %1997 = stablehlo.dynamic_reshape %1995, %1996 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %1998 = stablehlo.concatenate %1997, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %1999 = stablehlo.maximum %1988, %1998 : tensor<1xi32>
  %2000 = stablehlo.dynamic_broadcast_in_dim %1984, %1999, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2001 = stablehlo.dynamic_broadcast_in_dim %1994, %1999, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2002 = stablehlo.add %2000, %2001 : tensor<?xi32>
  %2003 = stablehlo.get_dimension_size %2002, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %2004 = stablehlo.constant dense<1> : tensor<1xi32>
  %2005 = stablehlo.dynamic_reshape %2003, %2004 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2006 = stablehlo.concatenate %2005, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2007 = stablehlo.maximum %1961, %2006 : tensor<1xi32>
  %2008 = stablehlo.maximum %1957, %2007 : tensor<1xi32>
  %2009 = stablehlo.dynamic_broadcast_in_dim %1942, %2008, dims = [0] : (tensor<?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %2010 = stablehlo.dynamic_broadcast_in_dim %1952, %2008, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2011 = stablehlo.dynamic_broadcast_in_dim %2002, %2008, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2012 = stablehlo.select %2009, %2010, %2011 : tensor<?xi1>, tensor<?xi32>
  %2013 = stablehlo.reshape %2012 : (tensor<?xi32>) -> tensor<1xi32>
  %2014 = stablehlo.constant dense<1> : tensor<1xi32>
  %2015 = stablehlo.constant dense<2> : tensor<1xi32>
  %2016 = stablehlo.constant dense<1> : tensor<1xi32>
  %2017 = stablehlo.real_dynamic_slice %1896, %2014, %2015, %2016 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2018 = stablehlo.minimum %2013, %2017 : tensor<1xi32>
  %2019 = stablehlo.concatenate %1901, %2018, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %2020 = stablehlo.constant dense<1> : tensor<i32>
  %2021 = stablehlo.reshape %2020 : (tensor<i32>) -> tensor<1xi32>
  %2022 = stablehlo.constant dense<1> : tensor<i32>
  %2023 = stablehlo.reshape %2022 : (tensor<i32>) -> tensor<1xi32>
  %2024 = stablehlo.concatenate %2021, %2023, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %2025 = stablehlo.real_dynamic_slice %50, %1777, %2019, %2024 : (tensor<1x1024xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %2026 = stablehlo.get_dimension_size %2025, dim = 0 : (tensor<?x?xi32>) -> tensor<i32>
  %2027 = stablehlo.constant dense<1> : tensor<1xi32>
  %2028 = stablehlo.dynamic_reshape %2026, %2027 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2029 = stablehlo.get_dimension_size %2025, dim = 1 : (tensor<?x?xi32>) -> tensor<i32>
  %2030 = stablehlo.constant dense<1> : tensor<1xi32>
  %2031 = stablehlo.dynamic_reshape %2029, %2030 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2032 = stablehlo.concatenate %2028, %2031, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %2033 = stablehlo.constant dense<1> : tensor<1xi32>
  %2034 = stablehlo.get_dimension_size %2033, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %2035 = stablehlo.constant dense<1> : tensor<1xi32>
  %2036 = stablehlo.dynamic_reshape %2034, %2035 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2037 = stablehlo.concatenate %2036, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2038 = stablehlo.constant dense<0> : tensor<i32>
  %2039 = stablehlo.reshape %2038 : (tensor<i32>) -> tensor<1xi32>
  %2040 = stablehlo.concatenate %2039, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2041 = stablehlo.constant dense<1> : tensor<i32>
  %2042 = stablehlo.reshape %2041 : (tensor<i32>) -> tensor<1xi32>
  %2043 = stablehlo.get_dimension_size %2037, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %2044 = stablehlo.constant dense<1> : tensor<1xi32>
  %2045 = stablehlo.dynamic_reshape %2043, %2044 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2046 = stablehlo.concatenate %2045, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2047 = stablehlo.constant dense<0> : tensor<1xi32>
  %2048 = stablehlo.constant dense<1> : tensor<1xi32>
  %2049 = stablehlo.constant dense<1> : tensor<1xi32>
  %2050 = stablehlo.real_dynamic_slice %2046, %2047, %2048, %2049 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2051 = stablehlo.minimum %2042, %2050 : tensor<1xi32>
  %2052 = stablehlo.concatenate %2051, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2053 = stablehlo.constant dense<1> : tensor<i32>
  %2054 = stablehlo.reshape %2053 : (tensor<i32>) -> tensor<1xi32>
  %2055 = stablehlo.concatenate %2054, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2056 = stablehlo.real_dynamic_slice %2037, %2040, %2052, %2055 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2057 = stablehlo.constant dense<1> : tensor<1xi32>
  %2058 = stablehlo.constant dense<1> : tensor<i32>
  %2059 = stablehlo.reshape %2058 : (tensor<i32>) -> tensor<1xi32>
  %2060 = stablehlo.concatenate %2059, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2061 = stablehlo.get_dimension_size %2037, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %2062 = stablehlo.constant dense<1> : tensor<1xi32>
  %2063 = stablehlo.dynamic_reshape %2061, %2062 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2064 = stablehlo.concatenate %2063, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2065 = stablehlo.constant dense<0> : tensor<i32>
  %2066 = stablehlo.reshape %2065 : (tensor<i32>) -> tensor<1xi32>
  %2067 = stablehlo.concatenate %2066, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2068 = stablehlo.constant dense<1> : tensor<i32>
  %2069 = stablehlo.reshape %2068 : (tensor<i32>) -> tensor<1xi32>
  %2070 = stablehlo.get_dimension_size %2064, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %2071 = stablehlo.constant dense<1> : tensor<1xi32>
  %2072 = stablehlo.dynamic_reshape %2070, %2071 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2073 = stablehlo.concatenate %2072, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2074 = stablehlo.constant dense<0> : tensor<1xi32>
  %2075 = stablehlo.constant dense<1> : tensor<1xi32>
  %2076 = stablehlo.constant dense<1> : tensor<1xi32>
  %2077 = stablehlo.real_dynamic_slice %2073, %2074, %2075, %2076 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2078 = stablehlo.minimum %2069, %2077 : tensor<1xi32>
  %2079 = stablehlo.concatenate %2078, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2080 = stablehlo.constant dense<1> : tensor<i32>
  %2081 = stablehlo.reshape %2080 : (tensor<i32>) -> tensor<1xi32>
  %2082 = stablehlo.concatenate %2081, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2083 = stablehlo.real_dynamic_slice %2064, %2067, %2079, %2082 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2084 = stablehlo.constant dense<> : tensor<0xi32>
  %2085 = stablehlo.dynamic_reshape %2083, %2084 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %2086 = stablehlo.constant dense<> : tensor<0xi32>
  %2087 = stablehlo.constant dense<> : tensor<0xi32>
  %2088 = stablehlo.maximum %2086, %2087 : tensor<0xi32>
  %2089 = stablehlo.dynamic_broadcast_in_dim %2085, %2088, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %2090 = stablehlo.constant dense<0> : tensor<i32>
  %2091 = stablehlo.dynamic_broadcast_in_dim %2090, %2088, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %2092 = stablehlo.compare  GE, %2089, %2091 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2093 = stablehlo.constant dense<1> : tensor<i32>
  %2094 = stablehlo.constant dense<1> : tensor<1xi32>
  %2095 = stablehlo.dynamic_broadcast_in_dim %2093, %2094, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2096 = stablehlo.constant dense<> : tensor<0xi32>
  %2097 = stablehlo.concatenate %2095, %2096, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %2098 = stablehlo.dynamic_broadcast_in_dim %2092, %2097, dims = [] : (tensor<i1>, tensor<1xi32>) -> tensor<?xi1>
  %2099 = stablehlo.get_dimension_size %2098, dim = 0 : (tensor<?xi1>) -> tensor<i32>
  %2100 = stablehlo.constant dense<1> : tensor<1xi32>
  %2101 = stablehlo.dynamic_reshape %2099, %2100 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2102 = stablehlo.concatenate %2101, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2103 = stablehlo.constant dense<1> : tensor<i32>
  %2104 = stablehlo.constant dense<1> : tensor<1xi32>
  %2105 = stablehlo.dynamic_broadcast_in_dim %2103, %2104, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2106 = stablehlo.constant dense<> : tensor<0xi32>
  %2107 = stablehlo.concatenate %2105, %2106, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %2108 = stablehlo.dynamic_broadcast_in_dim %2085, %2107, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2109 = stablehlo.get_dimension_size %2108, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %2110 = stablehlo.constant dense<1> : tensor<1xi32>
  %2111 = stablehlo.dynamic_reshape %2109, %2110 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2112 = stablehlo.concatenate %2111, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2113 = stablehlo.maximum %2102, %2112 : tensor<1xi32>
  %2114 = stablehlo.get_dimension_size %2108, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %2115 = stablehlo.constant dense<1> : tensor<1xi32>
  %2116 = stablehlo.dynamic_reshape %2114, %2115 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2117 = stablehlo.concatenate %2116, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2118 = stablehlo.constant dense<0> : tensor<i32>
  %2119 = stablehlo.reshape %2118 : (tensor<i32>) -> tensor<1xi32>
  %2120 = stablehlo.concatenate %2119, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2121 = stablehlo.constant dense<1> : tensor<i32>
  %2122 = stablehlo.reshape %2121 : (tensor<i32>) -> tensor<1xi32>
  %2123 = stablehlo.get_dimension_size %2064, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %2124 = stablehlo.constant dense<1> : tensor<1xi32>
  %2125 = stablehlo.dynamic_reshape %2123, %2124 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2126 = stablehlo.concatenate %2125, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2127 = stablehlo.constant dense<0> : tensor<1xi32>
  %2128 = stablehlo.constant dense<1> : tensor<1xi32>
  %2129 = stablehlo.constant dense<1> : tensor<1xi32>
  %2130 = stablehlo.real_dynamic_slice %2126, %2127, %2128, %2129 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2131 = stablehlo.minimum %2122, %2130 : tensor<1xi32>
  %2132 = stablehlo.concatenate %2131, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2133 = stablehlo.constant dense<1> : tensor<i32>
  %2134 = stablehlo.reshape %2133 : (tensor<i32>) -> tensor<1xi32>
  %2135 = stablehlo.concatenate %2134, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2136 = stablehlo.real_dynamic_slice %2064, %2120, %2132, %2135 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2137 = stablehlo.constant dense<> : tensor<0xi32>
  %2138 = stablehlo.dynamic_reshape %2136, %2137 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %2139 = stablehlo.constant dense<1> : tensor<1xi32>
  %2140 = stablehlo.dynamic_reshape %2138, %2139 : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2141 = stablehlo.get_dimension_size %2140, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %2142 = stablehlo.constant dense<1> : tensor<1xi32>
  %2143 = stablehlo.dynamic_reshape %2141, %2142 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2144 = stablehlo.concatenate %2143, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2145 = stablehlo.constant dense<1> : tensor<i32>
  %2146 = stablehlo.constant dense<1> : tensor<1xi32>
  %2147 = stablehlo.dynamic_broadcast_in_dim %2145, %2146, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2148 = stablehlo.constant dense<> : tensor<0xi32>
  %2149 = stablehlo.concatenate %2147, %2148, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %2150 = stablehlo.dynamic_broadcast_in_dim %2085, %2149, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2151 = stablehlo.get_dimension_size %2150, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %2152 = stablehlo.constant dense<1> : tensor<1xi32>
  %2153 = stablehlo.dynamic_reshape %2151, %2152 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2154 = stablehlo.concatenate %2153, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2155 = stablehlo.maximum %2144, %2154 : tensor<1xi32>
  %2156 = stablehlo.dynamic_broadcast_in_dim %2140, %2155, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2157 = stablehlo.dynamic_broadcast_in_dim %2150, %2155, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2158 = stablehlo.add %2156, %2157 : tensor<?xi32>
  %2159 = stablehlo.get_dimension_size %2158, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %2160 = stablehlo.constant dense<1> : tensor<1xi32>
  %2161 = stablehlo.dynamic_reshape %2159, %2160 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2162 = stablehlo.concatenate %2161, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2163 = stablehlo.maximum %2117, %2162 : tensor<1xi32>
  %2164 = stablehlo.maximum %2113, %2163 : tensor<1xi32>
  %2165 = stablehlo.dynamic_broadcast_in_dim %2098, %2164, dims = [0] : (tensor<?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %2166 = stablehlo.dynamic_broadcast_in_dim %2108, %2164, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2167 = stablehlo.dynamic_broadcast_in_dim %2158, %2164, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2168 = stablehlo.select %2165, %2166, %2167 : tensor<?xi1>, tensor<?xi32>
  %2169 = stablehlo.reshape %2168 : (tensor<?xi32>) -> tensor<1xi32>
  %2170 = stablehlo.get_dimension_size %2037, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %2171 = stablehlo.constant dense<1> : tensor<1xi32>
  %2172 = stablehlo.dynamic_reshape %2170, %2171 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2173 = stablehlo.concatenate %2172, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2174 = stablehlo.constant dense<0> : tensor<1xi32>
  %2175 = stablehlo.constant dense<1> : tensor<1xi32>
  %2176 = stablehlo.constant dense<1> : tensor<1xi32>
  %2177 = stablehlo.real_dynamic_slice %2173, %2174, %2175, %2176 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2178 = stablehlo.minimum %2169, %2177 : tensor<1xi32>
  %2179 = stablehlo.concatenate %2178, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2180 = stablehlo.constant dense<1> : tensor<i32>
  %2181 = stablehlo.reshape %2180 : (tensor<i32>) -> tensor<1xi32>
  %2182 = stablehlo.concatenate %2181, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2183 = stablehlo.real_dynamic_slice %2037, %2060, %2179, %2182 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2184 = stablehlo.concatenate %2056, %2057, %2183, dim = 0 : (tensor<?xi32>, tensor<1xi32>, tensor<?xi32>) -> tensor<2xi32>
  %2185 = stablehlo.dynamic_broadcast_in_dim %2033, %2184, dims = [0] : (tensor<1xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %2186 = stablehlo.get_dimension_size %24, dim = 0 : (tensor<1x128xi32>) -> tensor<i32>
  %2187 = stablehlo.constant dense<1> : tensor<1xi32>
  %2188 = stablehlo.dynamic_reshape %2186, %2187 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2189 = stablehlo.get_dimension_size %24, dim = 1 : (tensor<1x128xi32>) -> tensor<i32>
  %2190 = stablehlo.constant dense<1> : tensor<1xi32>
  %2191 = stablehlo.dynamic_reshape %2189, %2190 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2192 = stablehlo.concatenate %2188, %2191, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %2193 = stablehlo.dynamic_broadcast_in_dim %2185, %2192, dims = [0, 1] : (tensor<?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %2194 = stablehlo.get_dimension_size %2193, dim = 0 : (tensor<?x?xi32>) -> tensor<i32>
  %2195 = stablehlo.constant dense<1> : tensor<1xi32>
  %2196 = stablehlo.dynamic_reshape %2194, %2195 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2197 = stablehlo.get_dimension_size %2193, dim = 1 : (tensor<?x?xi32>) -> tensor<i32>
  %2198 = stablehlo.constant dense<1> : tensor<1xi32>
  %2199 = stablehlo.dynamic_reshape %2197, %2198 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2200 = stablehlo.concatenate %2196, %2199, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %2201 = stablehlo.maximum %2032, %2200 : tensor<2xi32>
  %2202 = stablehlo.dynamic_broadcast_in_dim %2025, %2201, dims = [0, 1] : (tensor<?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %2203 = stablehlo.dynamic_broadcast_in_dim %2193, %2201, dims = [0, 1] : (tensor<?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %2204 = stablehlo.compare  LT, %2202, %2203 : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi1>
  %2205 = stablehlo.get_dimension_size %2204, dim = 0 : (tensor<?x?xi1>) -> tensor<i32>
  %2206 = stablehlo.constant dense<1> : tensor<1xi32>
  %2207 = stablehlo.dynamic_reshape %2205, %2206 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2208 = stablehlo.get_dimension_size %2204, dim = 1 : (tensor<?x?xi1>) -> tensor<i32>
  %2209 = stablehlo.constant dense<1> : tensor<1xi32>
  %2210 = stablehlo.dynamic_reshape %2208, %2209 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2211 = stablehlo.concatenate %2207, %2210, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %2212 = stablehlo.constant dense<0> : tensor<i32>
  %2213 = stablehlo.reshape %2212 : (tensor<i32>) -> tensor<1xi32>
  %2214 = stablehlo.concatenate %2213, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2215 = stablehlo.constant dense<1> : tensor<i32>
  %2216 = stablehlo.reshape %2215 : (tensor<i32>) -> tensor<1xi32>
  %2217 = stablehlo.get_dimension_size %2211, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %2218 = stablehlo.constant dense<1> : tensor<1xi32>
  %2219 = stablehlo.dynamic_reshape %2217, %2218 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2220 = stablehlo.concatenate %2219, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2221 = stablehlo.constant dense<0> : tensor<1xi32>
  %2222 = stablehlo.constant dense<1> : tensor<1xi32>
  %2223 = stablehlo.constant dense<1> : tensor<1xi32>
  %2224 = stablehlo.real_dynamic_slice %2220, %2221, %2222, %2223 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2225 = stablehlo.minimum %2216, %2224 : tensor<1xi32>
  %2226 = stablehlo.concatenate %2225, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2227 = stablehlo.constant dense<1> : tensor<i32>
  %2228 = stablehlo.reshape %2227 : (tensor<i32>) -> tensor<1xi32>
  %2229 = stablehlo.concatenate %2228, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2230 = stablehlo.real_dynamic_slice %2211, %2214, %2226, %2229 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2231 = stablehlo.constant dense<> : tensor<0xi32>
  %2232 = stablehlo.dynamic_reshape %2230, %2231 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %2233 = stablehlo.constant dense<1> : tensor<1xi32>
  %2234 = stablehlo.dynamic_broadcast_in_dim %2232, %2233, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2235 = stablehlo.constant dense<1> : tensor<1xi32>
  %2236 = stablehlo.get_dimension_size %2204, dim = 0 : (tensor<?x?xi1>) -> tensor<i32>
  %2237 = stablehlo.constant dense<1> : tensor<1xi32>
  %2238 = stablehlo.dynamic_reshape %2236, %2237 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2239 = stablehlo.get_dimension_size %2204, dim = 1 : (tensor<?x?xi1>) -> tensor<i32>
  %2240 = stablehlo.constant dense<1> : tensor<1xi32>
  %2241 = stablehlo.dynamic_reshape %2239, %2240 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2242 = stablehlo.concatenate %2238, %2241, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %2243 = stablehlo.constant dense<1> : tensor<i32>
  %2244 = stablehlo.reshape %2243 : (tensor<i32>) -> tensor<1xi32>
  %2245 = stablehlo.concatenate %2244, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2246 = stablehlo.constant dense<2> : tensor<i32>
  %2247 = stablehlo.reshape %2246 : (tensor<i32>) -> tensor<1xi32>
  %2248 = stablehlo.get_dimension_size %2242, dim = 0 : (tensor<2xi32>) -> tensor<i32>
  %2249 = stablehlo.constant dense<1> : tensor<1xi32>
  %2250 = stablehlo.dynamic_reshape %2248, %2249 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2251 = stablehlo.concatenate %2250, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2252 = stablehlo.constant dense<0> : tensor<1xi32>
  %2253 = stablehlo.constant dense<1> : tensor<1xi32>
  %2254 = stablehlo.constant dense<1> : tensor<1xi32>
  %2255 = stablehlo.real_dynamic_slice %2251, %2252, %2253, %2254 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2256 = stablehlo.minimum %2247, %2255 : tensor<1xi32>
  %2257 = stablehlo.concatenate %2256, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2258 = stablehlo.constant dense<1> : tensor<i32>
  %2259 = stablehlo.reshape %2258 : (tensor<i32>) -> tensor<1xi32>
  %2260 = stablehlo.concatenate %2259, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2261 = stablehlo.real_dynamic_slice %2242, %2245, %2257, %2260 : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2262 = stablehlo.constant dense<> : tensor<0xi32>
  %2263 = stablehlo.dynamic_reshape %2261, %2262 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %2264 = stablehlo.constant dense<1> : tensor<1xi32>
  %2265 = stablehlo.dynamic_broadcast_in_dim %2263, %2264, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2266 = stablehlo.concatenate %2234, %2235, %2235, %2265, dim = 0 : (tensor<?xi32>, tensor<1xi32>, tensor<1xi32>, tensor<?xi32>) -> tensor<4xi32>
  %2267 = stablehlo.dynamic_reshape %2204, %2266 : (tensor<?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %2268 = stablehlo.constant dense<1> : tensor<1xi32>
  %2269 = stablehlo.concatenate %2234, %2235, %2268, %2265, dim = 0 : (tensor<?xi32>, tensor<1xi32>, tensor<1xi32>, tensor<?xi32>) -> tensor<4xi32>
  %2270 = stablehlo.dynamic_broadcast_in_dim %2267, %2269, dims = [0, 1, 2, 3] : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %2271 = stablehlo.convert %2270 : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xf32>
  %2272 = stablehlo.get_dimension_size %2271, dim = 0 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2273 = stablehlo.constant dense<1> : tensor<1xi32>
  %2274 = stablehlo.dynamic_reshape %2272, %2273 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2275 = stablehlo.get_dimension_size %2271, dim = 1 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2276 = stablehlo.constant dense<1> : tensor<1xi32>
  %2277 = stablehlo.dynamic_reshape %2275, %2276 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2278 = stablehlo.get_dimension_size %2271, dim = 2 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2279 = stablehlo.constant dense<1> : tensor<1xi32>
  %2280 = stablehlo.dynamic_reshape %2278, %2279 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2281 = stablehlo.get_dimension_size %2271, dim = 3 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2282 = stablehlo.constant dense<1> : tensor<1xi32>
  %2283 = stablehlo.dynamic_reshape %2281, %2282 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2284 = stablehlo.concatenate %2274, %2277, %2280, %2283, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %2285 = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
  %2286 = stablehlo.constant dense<1> : tensor<i32>
  %2287 = stablehlo.constant dense<3> : tensor<1xi32>
  %2288 = stablehlo.dynamic_broadcast_in_dim %2286, %2287, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2289 = stablehlo.get_dimension_size %2285, dim = 0 : (tensor<1xf32>) -> tensor<i32>
  %2290 = stablehlo.constant dense<1> : tensor<1xi32>
  %2291 = stablehlo.dynamic_reshape %2289, %2290 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2292 = stablehlo.concatenate %2291, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2293 = stablehlo.concatenate %2288, %2292, dim = 0 : (tensor<?xi32>, tensor<1xi32>) -> tensor<4xi32>
  %2294 = stablehlo.dynamic_broadcast_in_dim %2285, %2293, dims = [3] : (tensor<1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2295 = stablehlo.get_dimension_size %2294, dim = 0 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2296 = stablehlo.constant dense<1> : tensor<1xi32>
  %2297 = stablehlo.dynamic_reshape %2295, %2296 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2298 = stablehlo.get_dimension_size %2294, dim = 1 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2299 = stablehlo.constant dense<1> : tensor<1xi32>
  %2300 = stablehlo.dynamic_reshape %2298, %2299 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2301 = stablehlo.get_dimension_size %2294, dim = 2 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2302 = stablehlo.constant dense<1> : tensor<1xi32>
  %2303 = stablehlo.dynamic_reshape %2301, %2302 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2304 = stablehlo.get_dimension_size %2294, dim = 3 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2305 = stablehlo.constant dense<1> : tensor<1xi32>
  %2306 = stablehlo.dynamic_reshape %2304, %2305 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2307 = stablehlo.concatenate %2297, %2300, %2303, %2306, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %2308 = stablehlo.maximum %2284, %2307 : tensor<4xi32>
  %2309 = stablehlo.dynamic_broadcast_in_dim %2271, %2308, dims = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2310 = stablehlo.dynamic_broadcast_in_dim %2294, %2308, dims = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2311 = stablehlo.compare  EQ, %2309, %2310 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %2312 = stablehlo.get_dimension_size %2311, dim = 0 : (tensor<?x?x?x?xi1>) -> tensor<i32>
  %2313 = stablehlo.constant dense<1> : tensor<1xi32>
  %2314 = stablehlo.dynamic_reshape %2312, %2313 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2315 = stablehlo.get_dimension_size %2311, dim = 1 : (tensor<?x?x?x?xi1>) -> tensor<i32>
  %2316 = stablehlo.constant dense<1> : tensor<1xi32>
  %2317 = stablehlo.dynamic_reshape %2315, %2316 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2318 = stablehlo.get_dimension_size %2311, dim = 2 : (tensor<?x?x?x?xi1>) -> tensor<i32>
  %2319 = stablehlo.constant dense<1> : tensor<1xi32>
  %2320 = stablehlo.dynamic_reshape %2318, %2319 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2321 = stablehlo.get_dimension_size %2311, dim = 3 : (tensor<?x?x?x?xi1>) -> tensor<i32>
  %2322 = stablehlo.constant dense<1> : tensor<1xi32>
  %2323 = stablehlo.dynamic_reshape %2321, %2322 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2324 = stablehlo.concatenate %2314, %2317, %2320, %2323, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %2325 = stablehlo.constant dense<0xFF800000> : tensor<1xf32>
  %2326 = stablehlo.constant dense<1> : tensor<i32>
  %2327 = stablehlo.constant dense<3> : tensor<1xi32>
  %2328 = stablehlo.dynamic_broadcast_in_dim %2326, %2327, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2329 = stablehlo.get_dimension_size %2325, dim = 0 : (tensor<1xf32>) -> tensor<i32>
  %2330 = stablehlo.constant dense<1> : tensor<1xi32>
  %2331 = stablehlo.dynamic_reshape %2329, %2330 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2332 = stablehlo.concatenate %2331, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2333 = stablehlo.concatenate %2328, %2332, dim = 0 : (tensor<?xi32>, tensor<1xi32>) -> tensor<4xi32>
  %2334 = stablehlo.dynamic_broadcast_in_dim %2325, %2333, dims = [3] : (tensor<1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2335 = stablehlo.get_dimension_size %2334, dim = 0 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2336 = stablehlo.constant dense<1> : tensor<1xi32>
  %2337 = stablehlo.dynamic_reshape %2335, %2336 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2338 = stablehlo.get_dimension_size %2334, dim = 1 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2339 = stablehlo.constant dense<1> : tensor<1xi32>
  %2340 = stablehlo.dynamic_reshape %2338, %2339 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2341 = stablehlo.get_dimension_size %2334, dim = 2 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2342 = stablehlo.constant dense<1> : tensor<1xi32>
  %2343 = stablehlo.dynamic_reshape %2341, %2342 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2344 = stablehlo.get_dimension_size %2334, dim = 3 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2345 = stablehlo.constant dense<1> : tensor<1xi32>
  %2346 = stablehlo.dynamic_reshape %2344, %2345 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2347 = stablehlo.concatenate %2337, %2340, %2343, %2346, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %2348 = stablehlo.maximum %2324, %2347 : tensor<4xi32>
  %2349 = stablehlo.get_dimension_size %2334, dim = 0 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2350 = stablehlo.constant dense<1> : tensor<1xi32>
  %2351 = stablehlo.dynamic_reshape %2349, %2350 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2352 = stablehlo.get_dimension_size %2334, dim = 1 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2353 = stablehlo.constant dense<1> : tensor<1xi32>
  %2354 = stablehlo.dynamic_reshape %2352, %2353 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2355 = stablehlo.get_dimension_size %2334, dim = 2 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2356 = stablehlo.constant dense<1> : tensor<1xi32>
  %2357 = stablehlo.dynamic_reshape %2355, %2356 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2358 = stablehlo.get_dimension_size %2334, dim = 3 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2359 = stablehlo.constant dense<1> : tensor<1xi32>
  %2360 = stablehlo.dynamic_reshape %2358, %2359 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2361 = stablehlo.concatenate %2351, %2354, %2357, %2360, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %2362 = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
  %2363 = stablehlo.constant dense<1> : tensor<i32>
  %2364 = stablehlo.constant dense<3> : tensor<1xi32>
  %2365 = stablehlo.dynamic_broadcast_in_dim %2363, %2364, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2366 = stablehlo.get_dimension_size %2362, dim = 0 : (tensor<1xf32>) -> tensor<i32>
  %2367 = stablehlo.constant dense<1> : tensor<1xi32>
  %2368 = stablehlo.dynamic_reshape %2366, %2367 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2369 = stablehlo.concatenate %2368, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2370 = stablehlo.concatenate %2365, %2369, dim = 0 : (tensor<?xi32>, tensor<1xi32>) -> tensor<4xi32>
  %2371 = stablehlo.dynamic_broadcast_in_dim %2362, %2370, dims = [3] : (tensor<1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2372 = stablehlo.get_dimension_size %2371, dim = 0 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2373 = stablehlo.constant dense<1> : tensor<1xi32>
  %2374 = stablehlo.dynamic_reshape %2372, %2373 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2375 = stablehlo.get_dimension_size %2371, dim = 1 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2376 = stablehlo.constant dense<1> : tensor<1xi32>
  %2377 = stablehlo.dynamic_reshape %2375, %2376 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2378 = stablehlo.get_dimension_size %2371, dim = 2 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2379 = stablehlo.constant dense<1> : tensor<1xi32>
  %2380 = stablehlo.dynamic_reshape %2378, %2379 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2381 = stablehlo.get_dimension_size %2371, dim = 3 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2382 = stablehlo.constant dense<1> : tensor<1xi32>
  %2383 = stablehlo.dynamic_reshape %2381, %2382 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2384 = stablehlo.concatenate %2374, %2377, %2380, %2383, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %2385 = stablehlo.maximum %2361, %2384 : tensor<4xi32>
  %2386 = stablehlo.maximum %2348, %2385 : tensor<4xi32>
  %2387 = stablehlo.dynamic_broadcast_in_dim %2311, %2386, dims = [0, 1, 2, 3] : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %2388 = stablehlo.dynamic_broadcast_in_dim %2334, %2386, dims = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2389 = stablehlo.dynamic_broadcast_in_dim %2371, %2386, dims = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2390 = stablehlo.select %2387, %2388, %2389 : tensor<?x?x?x?xi1>, tensor<?x?x?x?xf32>
  %2391 = stablehlo.get_dimension_size %2390, dim = 0 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2392 = stablehlo.constant dense<1> : tensor<1xi32>
  %2393 = stablehlo.dynamic_reshape %2391, %2392 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2394 = stablehlo.get_dimension_size %2390, dim = 1 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2395 = stablehlo.constant dense<1> : tensor<1xi32>
  %2396 = stablehlo.dynamic_reshape %2394, %2395 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2397 = stablehlo.get_dimension_size %2390, dim = 2 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2398 = stablehlo.constant dense<1> : tensor<1xi32>
  %2399 = stablehlo.dynamic_reshape %2397, %2398 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2400 = stablehlo.get_dimension_size %2390, dim = 3 : (tensor<?x?x?x?xf32>) -> tensor<i32>
  %2401 = stablehlo.constant dense<1> : tensor<1xi32>
  %2402 = stablehlo.dynamic_reshape %2400, %2401 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2403 = stablehlo.concatenate %2393, %2396, %2399, %2402, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %2404 = stablehlo.maximum %1772, %2403 : tensor<4xi32>
  %2405 = stablehlo.dynamic_broadcast_in_dim %1759, %2404, dims = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2406 = stablehlo.dynamic_broadcast_in_dim %2390, %2404, dims = [0, 1, 2, 3] : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2407 = stablehlo.add %2405, %2406 : tensor<?x?x?x?xf32>
  %2408 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2409 = stablehlo.reduce(%2407 init: %2408) applies stablehlo.maximum across dimensions = [3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xf32>
  %2410 = stablehlo.get_dimension_size %2409, dim = 0 : (tensor<?x?x?xf32>) -> tensor<i32>
  %2411 = stablehlo.constant dense<1> : tensor<1xi32>
  %2412 = stablehlo.dynamic_reshape %2410, %2411 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2413 = stablehlo.get_dimension_size %2409, dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
  %2414 = stablehlo.constant dense<1> : tensor<1xi32>
  %2415 = stablehlo.dynamic_reshape %2413, %2414 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2416 = stablehlo.get_dimension_size %2409, dim = 2 : (tensor<?x?x?xf32>) -> tensor<i32>
  %2417 = stablehlo.constant dense<1> : tensor<1xi32>
  %2418 = stablehlo.dynamic_reshape %2416, %2417 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2419 = stablehlo.concatenate %2412, %2415, %2418, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %2420 = stablehlo.constant dense<0> : tensor<i32>
  %2421 = stablehlo.reshape %2420 : (tensor<i32>) -> tensor<1xi32>
  %2422 = stablehlo.concatenate %2421, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2423 = stablehlo.get_dimension_size %2419, dim = 0 : (tensor<3xi32>) -> tensor<i32>
  %2424 = stablehlo.constant dense<1> : tensor<1xi32>
  %2425 = stablehlo.dynamic_reshape %2423, %2424 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2426 = stablehlo.concatenate %2425, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2427 = stablehlo.constant dense<0> : tensor<i32>
  %2428 = stablehlo.reshape %2427 : (tensor<i32>) -> tensor<1xi32>
  %2429 = stablehlo.concatenate %2428, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2430 = stablehlo.constant dense<1> : tensor<i32>
  %2431 = stablehlo.reshape %2430 : (tensor<i32>) -> tensor<1xi32>
  %2432 = stablehlo.get_dimension_size %2426, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %2433 = stablehlo.constant dense<1> : tensor<1xi32>
  %2434 = stablehlo.dynamic_reshape %2432, %2433 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2435 = stablehlo.concatenate %2434, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2436 = stablehlo.constant dense<0> : tensor<1xi32>
  %2437 = stablehlo.constant dense<1> : tensor<1xi32>
  %2438 = stablehlo.constant dense<1> : tensor<1xi32>
  %2439 = stablehlo.real_dynamic_slice %2435, %2436, %2437, %2438 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2440 = stablehlo.minimum %2431, %2439 : tensor<1xi32>
  %2441 = stablehlo.concatenate %2440, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2442 = stablehlo.constant dense<1> : tensor<i32>
  %2443 = stablehlo.reshape %2442 : (tensor<i32>) -> tensor<1xi32>
  %2444 = stablehlo.concatenate %2443, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2445 = stablehlo.real_dynamic_slice %2426, %2429, %2441, %2444 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2446 = stablehlo.constant dense<> : tensor<0xi32>
  %2447 = stablehlo.dynamic_reshape %2445, %2446 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %2448 = stablehlo.constant dense<> : tensor<0xi32>
  %2449 = stablehlo.constant dense<> : tensor<0xi32>
  %2450 = stablehlo.maximum %2448, %2449 : tensor<0xi32>
  %2451 = stablehlo.dynamic_broadcast_in_dim %2447, %2450, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %2452 = stablehlo.constant dense<-1> : tensor<i32>
  %2453 = stablehlo.dynamic_broadcast_in_dim %2452, %2450, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %2454 = stablehlo.add %2451, %2453 : tensor<i32>
  %2455 = stablehlo.reshape %2454 : (tensor<i32>) -> tensor<1xi32>
  %2456 = stablehlo.get_dimension_size %2419, dim = 0 : (tensor<3xi32>) -> tensor<i32>
  %2457 = stablehlo.constant dense<1> : tensor<1xi32>
  %2458 = stablehlo.dynamic_reshape %2456, %2457 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2459 = stablehlo.concatenate %2458, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2460 = stablehlo.constant dense<0> : tensor<1xi32>
  %2461 = stablehlo.constant dense<1> : tensor<1xi32>
  %2462 = stablehlo.constant dense<1> : tensor<1xi32>
  %2463 = stablehlo.real_dynamic_slice %2459, %2460, %2461, %2462 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2464 = stablehlo.minimum %2455, %2463 : tensor<1xi32>
  %2465 = stablehlo.concatenate %2464, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2466 = stablehlo.constant dense<1> : tensor<i32>
  %2467 = stablehlo.reshape %2466 : (tensor<i32>) -> tensor<1xi32>
  %2468 = stablehlo.concatenate %2467, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2469 = stablehlo.real_dynamic_slice %2419, %2422, %2465, %2468 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2470 = stablehlo.constant dense<1> : tensor<1xi32>
  %2471 = stablehlo.get_dimension_size %2419, dim = 0 : (tensor<3xi32>) -> tensor<i32>
  %2472 = stablehlo.constant dense<1> : tensor<1xi32>
  %2473 = stablehlo.dynamic_reshape %2471, %2472 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2474 = stablehlo.concatenate %2473, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2475 = stablehlo.constant dense<0> : tensor<i32>
  %2476 = stablehlo.reshape %2475 : (tensor<i32>) -> tensor<1xi32>
  %2477 = stablehlo.concatenate %2476, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2478 = stablehlo.constant dense<1> : tensor<i32>
  %2479 = stablehlo.reshape %2478 : (tensor<i32>) -> tensor<1xi32>
  %2480 = stablehlo.get_dimension_size %2474, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %2481 = stablehlo.constant dense<1> : tensor<1xi32>
  %2482 = stablehlo.dynamic_reshape %2480, %2481 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2483 = stablehlo.concatenate %2482, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2484 = stablehlo.constant dense<0> : tensor<1xi32>
  %2485 = stablehlo.constant dense<1> : tensor<1xi32>
  %2486 = stablehlo.constant dense<1> : tensor<1xi32>
  %2487 = stablehlo.real_dynamic_slice %2483, %2484, %2485, %2486 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2488 = stablehlo.minimum %2479, %2487 : tensor<1xi32>
  %2489 = stablehlo.concatenate %2488, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2490 = stablehlo.constant dense<1> : tensor<i32>
  %2491 = stablehlo.reshape %2490 : (tensor<i32>) -> tensor<1xi32>
  %2492 = stablehlo.concatenate %2491, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2493 = stablehlo.real_dynamic_slice %2474, %2477, %2489, %2492 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2494 = stablehlo.constant dense<> : tensor<0xi32>
  %2495 = stablehlo.dynamic_reshape %2493, %2494 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %2496 = stablehlo.constant dense<> : tensor<0xi32>
  %2497 = stablehlo.constant dense<> : tensor<0xi32>
  %2498 = stablehlo.maximum %2496, %2497 : tensor<0xi32>
  %2499 = stablehlo.dynamic_broadcast_in_dim %2495, %2498, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %2500 = stablehlo.constant dense<-1> : tensor<i32>
  %2501 = stablehlo.dynamic_broadcast_in_dim %2500, %2498, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %2502 = stablehlo.add %2499, %2501 : tensor<i32>
  %2503 = stablehlo.reshape %2502 : (tensor<i32>) -> tensor<1xi32>
  %2504 = stablehlo.concatenate %2503, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2505 = stablehlo.constant dense<0> : tensor<i32>
  %2506 = stablehlo.reshape %2505 : (tensor<i32>) -> tensor<1xi32>
  %2507 = stablehlo.concatenate %2506, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2508 = stablehlo.constant dense<1> : tensor<i32>
  %2509 = stablehlo.reshape %2508 : (tensor<i32>) -> tensor<1xi32>
  %2510 = stablehlo.get_dimension_size %2474, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %2511 = stablehlo.constant dense<1> : tensor<1xi32>
  %2512 = stablehlo.dynamic_reshape %2510, %2511 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2513 = stablehlo.concatenate %2512, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2514 = stablehlo.constant dense<0> : tensor<1xi32>
  %2515 = stablehlo.constant dense<1> : tensor<1xi32>
  %2516 = stablehlo.constant dense<1> : tensor<1xi32>
  %2517 = stablehlo.real_dynamic_slice %2513, %2514, %2515, %2516 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2518 = stablehlo.minimum %2509, %2517 : tensor<1xi32>
  %2519 = stablehlo.concatenate %2518, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2520 = stablehlo.constant dense<1> : tensor<i32>
  %2521 = stablehlo.reshape %2520 : (tensor<i32>) -> tensor<1xi32>
  %2522 = stablehlo.concatenate %2521, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2523 = stablehlo.real_dynamic_slice %2474, %2507, %2519, %2522 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2524 = stablehlo.constant dense<> : tensor<0xi32>
  %2525 = stablehlo.dynamic_reshape %2523, %2524 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %2526 = stablehlo.constant dense<> : tensor<0xi32>
  %2527 = stablehlo.constant dense<> : tensor<0xi32>
  %2528 = stablehlo.maximum %2526, %2527 : tensor<0xi32>
  %2529 = stablehlo.dynamic_broadcast_in_dim %2525, %2528, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %2530 = stablehlo.constant dense<0> : tensor<i32>
  %2531 = stablehlo.dynamic_broadcast_in_dim %2530, %2528, dims = [] : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %2532 = stablehlo.compare  GE, %2529, %2531 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2533 = stablehlo.constant dense<1> : tensor<i32>
  %2534 = stablehlo.constant dense<1> : tensor<1xi32>
  %2535 = stablehlo.dynamic_broadcast_in_dim %2533, %2534, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2536 = stablehlo.constant dense<> : tensor<0xi32>
  %2537 = stablehlo.concatenate %2535, %2536, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %2538 = stablehlo.dynamic_broadcast_in_dim %2532, %2537, dims = [] : (tensor<i1>, tensor<1xi32>) -> tensor<?xi1>
  %2539 = stablehlo.get_dimension_size %2538, dim = 0 : (tensor<?xi1>) -> tensor<i32>
  %2540 = stablehlo.constant dense<1> : tensor<1xi32>
  %2541 = stablehlo.dynamic_reshape %2539, %2540 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2542 = stablehlo.concatenate %2541, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2543 = stablehlo.constant dense<1> : tensor<i32>
  %2544 = stablehlo.constant dense<1> : tensor<1xi32>
  %2545 = stablehlo.dynamic_broadcast_in_dim %2543, %2544, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2546 = stablehlo.constant dense<> : tensor<0xi32>
  %2547 = stablehlo.concatenate %2545, %2546, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %2548 = stablehlo.dynamic_broadcast_in_dim %2525, %2547, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2549 = stablehlo.get_dimension_size %2548, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %2550 = stablehlo.constant dense<1> : tensor<1xi32>
  %2551 = stablehlo.dynamic_reshape %2549, %2550 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2552 = stablehlo.concatenate %2551, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2553 = stablehlo.maximum %2542, %2552 : tensor<1xi32>
  %2554 = stablehlo.get_dimension_size %2548, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %2555 = stablehlo.constant dense<1> : tensor<1xi32>
  %2556 = stablehlo.dynamic_reshape %2554, %2555 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2557 = stablehlo.concatenate %2556, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2558 = stablehlo.constant dense<0> : tensor<i32>
  %2559 = stablehlo.reshape %2558 : (tensor<i32>) -> tensor<1xi32>
  %2560 = stablehlo.concatenate %2559, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2561 = stablehlo.constant dense<1> : tensor<i32>
  %2562 = stablehlo.reshape %2561 : (tensor<i32>) -> tensor<1xi32>
  %2563 = stablehlo.get_dimension_size %2474, dim = 0 : (tensor<1xi32>) -> tensor<i32>
  %2564 = stablehlo.constant dense<1> : tensor<1xi32>
  %2565 = stablehlo.dynamic_reshape %2563, %2564 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2566 = stablehlo.concatenate %2565, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2567 = stablehlo.constant dense<0> : tensor<1xi32>
  %2568 = stablehlo.constant dense<1> : tensor<1xi32>
  %2569 = stablehlo.constant dense<1> : tensor<1xi32>
  %2570 = stablehlo.real_dynamic_slice %2566, %2567, %2568, %2569 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2571 = stablehlo.minimum %2562, %2570 : tensor<1xi32>
  %2572 = stablehlo.concatenate %2571, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2573 = stablehlo.constant dense<1> : tensor<i32>
  %2574 = stablehlo.reshape %2573 : (tensor<i32>) -> tensor<1xi32>
  %2575 = stablehlo.concatenate %2574, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2576 = stablehlo.real_dynamic_slice %2474, %2560, %2572, %2575 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2577 = stablehlo.constant dense<> : tensor<0xi32>
  %2578 = stablehlo.dynamic_reshape %2576, %2577 : (tensor<?xi32>, tensor<0xi32>) -> tensor<i32>
  %2579 = stablehlo.constant dense<1> : tensor<1xi32>
  %2580 = stablehlo.dynamic_reshape %2578, %2579 : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2581 = stablehlo.get_dimension_size %2580, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %2582 = stablehlo.constant dense<1> : tensor<1xi32>
  %2583 = stablehlo.dynamic_reshape %2581, %2582 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2584 = stablehlo.concatenate %2583, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2585 = stablehlo.constant dense<1> : tensor<i32>
  %2586 = stablehlo.constant dense<1> : tensor<1xi32>
  %2587 = stablehlo.dynamic_broadcast_in_dim %2585, %2586, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2588 = stablehlo.constant dense<> : tensor<0xi32>
  %2589 = stablehlo.concatenate %2587, %2588, dim = 0 : (tensor<?xi32>, tensor<0xi32>) -> tensor<1xi32>
  %2590 = stablehlo.dynamic_broadcast_in_dim %2525, %2589, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %2591 = stablehlo.get_dimension_size %2590, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %2592 = stablehlo.constant dense<1> : tensor<1xi32>
  %2593 = stablehlo.dynamic_reshape %2591, %2592 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2594 = stablehlo.concatenate %2593, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2595 = stablehlo.maximum %2584, %2594 : tensor<1xi32>
  %2596 = stablehlo.dynamic_broadcast_in_dim %2580, %2595, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2597 = stablehlo.dynamic_broadcast_in_dim %2590, %2595, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2598 = stablehlo.add %2596, %2597 : tensor<?xi32>
  %2599 = stablehlo.get_dimension_size %2598, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %2600 = stablehlo.constant dense<1> : tensor<1xi32>
  %2601 = stablehlo.dynamic_reshape %2599, %2600 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2602 = stablehlo.concatenate %2601, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2603 = stablehlo.maximum %2557, %2602 : tensor<1xi32>
  %2604 = stablehlo.maximum %2553, %2603 : tensor<1xi32>
  %2605 = stablehlo.dynamic_broadcast_in_dim %2538, %2604, dims = [0] : (tensor<?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %2606 = stablehlo.dynamic_broadcast_in_dim %2548, %2604, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2607 = stablehlo.dynamic_broadcast_in_dim %2598, %2604, dims = [0] : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2608 = stablehlo.select %2605, %2606, %2607 : tensor<?xi1>, tensor<?xi32>
  %2609 = stablehlo.reshape %2608 : (tensor<?xi32>) -> tensor<1xi32>
  %2610 = stablehlo.get_dimension_size %2419, dim = 0 : (tensor<3xi32>) -> tensor<i32>
  %2611 = stablehlo.constant dense<1> : tensor<1xi32>
  %2612 = stablehlo.dynamic_reshape %2610, %2611 : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %2613 = stablehlo.concatenate %2612, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2614 = stablehlo.constant dense<0> : tensor<1xi32>
  %2615 = stablehlo.constant dense<1> : tensor<1xi32>
  %2616 = stablehlo.constant dense<1> : tensor<1xi32>
  %2617 = stablehlo.real_dynamic_slice %2613, %2614, %2615, %2616 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2618 = stablehlo.minimum %2609, %2617 : tensor<1xi32>
  %2619 = stablehlo.concatenate %2618, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2620 = stablehlo.constant dense<1> : tensor<i32>
  %2621 = stablehlo.reshape %2620 : (tensor<i32>) -> tensor<1xi32>
  %2622 = stablehlo.concatenate %2621, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
  %2623 = stablehlo.real_dynamic_slice %2419, %2504, %2619, %2622 : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2624 = stablehlo.concatenate %2469, %2470, %2623, dim = 0 : (tensor<?xi32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
  return %2624 : tensor<?xi32>
}

