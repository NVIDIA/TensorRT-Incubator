// RUN: mlir-tensorrt-opt -split-input-file -convert-host-to-emitc %s | \
// RUN: mlir-tensorrt-translate -split-input-file -mlir-to-cpp | FileCheck %s --check-prefix=CPP

// Verify `trtrt.enqueue` lowering:
// - materializes stable `mtrt::PtrAndShape<Rank>` locals (no temporaries)
// - packs them into `mtrt::UnrankedMemRef` arrays
// - calls `mtrt::tensorrt_enqueue` and checks status

func.func @enqueue_1in1out(
    %ctx: !trtrt.context,
    %stream: !cuda.stream,
    %in: memref<?x?xf32>,
    %out: memref<?x?xf32>) {
  trtrt.enqueue %ctx stream(%stream) (%in) outs(%out)
    : (memref<?x?xf32>) -> memref<?x?xf32>
  return
}

// CPP: #include "MTRTRuntimeTensorRT.h"

// CPP-LABEL: void enqueue_1in1out(
// CPP: mtrt::UnrankedMemRef [[IN_ARR:.+]][1];
// CPP: mtrt::UnrankedMemRef [[OUT_ARR:.+]][1];
// CPP: mtrt::make_ptr_shape_descriptor<2>(
// CPP: mtrt::make_unranked_descriptor(

// CPP: int32_t [[ST:.+]] = mtrt::tensorrt_enqueue({{.*}}, {{.*}}, {{.*}}, [[IN_ARR]], {{.*}}, [[OUT_ARR]]);
// CPP: mtrt::abort_on_error([[ST]]);

// Ensure we do NOT take the address of a temporary ranked descriptor.
// CPP-NOT: make_unranked_descriptor({{.*}}make_ptr_shape_descriptor

// -----

func.func @enqueue_alloc_1out(
    %ctx: !trtrt.context,
    %stream: !cuda.stream,
    %in: memref<?x?xf32>) -> memref<?x?xf32> {
  %out = trtrt.enqueue_alloc %ctx stream(%stream) (%in)
    : (memref<?x?xf32>) -> memref<?x?xf32>
  return %out : memref<?x?xf32>
}

// CPP-LABEL: mtrt::RankedMemRef<2> enqueue_alloc_1out(
// CPP: mtrt::UnrankedMemRef
// CPP: mtrt::UnrankedMemRefMut
// CPP: mtrt::make_unranked_descriptor_mut_ptr(
// CPP: int32_t {{.*}} = mtrt::tensorrt_enqueue_alloc(
// CPP: mtrt::abort_on_error(
