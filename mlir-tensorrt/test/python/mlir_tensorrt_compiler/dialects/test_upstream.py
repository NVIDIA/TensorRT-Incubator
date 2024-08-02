# RUN: %PYTHON %s | FileCheck %s

import mlir_tensorrt.compiler.dialects.affine as affine
import mlir_tensorrt.compiler.dialects.arith as arith
import mlir_tensorrt.compiler.dialects.bufferization as bufferization
import mlir_tensorrt.compiler.dialects.func as func
import mlir_tensorrt.compiler.dialects.quant as quant
import mlir_tensorrt.compiler.dialects.scf as scf
import mlir_tensorrt.compiler.dialects.tensor as tensor
from mlir_tensorrt.compiler.ir import *


def run(f):
    print("\n TEST: ", f.__name__)
    f()
    return f


@run
def test_multiple_dialects():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        f32Type = F32Type.get()
        indexType = IndexType.get()
        bool = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)

        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(
                RankedTensorType.get(
                    (1024, 1024),
                    f32Type,
                ),
                RankedTensorType.get(
                    (1024, 1024),
                    f32Type,
                ),
            )
            def test_func(arg0, arg1):
                c0 = arith.ConstantOp(indexType, 0)
                c1 = arith.ConstantOp(indexType, 1)
                d0 = tensor.DimOp(arg0, c0)
                d1 = tensor.DimOp(arg1, c1)
                d0PlusD1 = arith.addi(d0, d1)
                alloc = bufferization.AllocTensorOp(arg1.type, (), copy=arg1)
                return [d0PlusD1, alloc.result]

            @func.FuncOp.from_py_func(bool)
            def test_scf(cond):
                if_op = scf.IfOp(cond, [i32, i32], hasElse=True)
                with InsertionPoint(if_op.then_block):
                    x_true = arith.ConstantOp(i32, 0)
                    y_true = arith.ConstantOp(i32, 1)
                    scf.YieldOp([x_true, y_true])
                with InsertionPoint(if_op.else_block):
                    x_false = arith.ConstantOp(i32, 2)
                    y_false = arith.ConstantOp(i32, 3)
                    scf.YieldOp([x_false, y_false])
                add = arith.AddIOp(if_op.results[0], if_op.results[1])
                return add.result

            @func.FuncOp.from_py_func(indexType, indexType)
            def test_affine_apply(arg0, arg1):
                d0 = AffineDimExpr.get(0)
                s0 = AffineSymbolExpr.get(0)
                s1 = AffineSymbolExpr.get(1)
                expr = AffineExpr.get_floor_div(s0 * 3, s1)
                map = AffineMap.get(1, 2, [d0 + (expr % d0)])
                a1 = affine.AffineApplyOp(map, [arg0, arg1, arg1])
                return a1

        print(module)


@run
def test_quant_dialect():
    with Context():
        i8 = IntegerType.get_signed(8)
        f32 = F32Type.get()
        uniform = quant.UniformQuantizedType.get(
            quant.UniformQuantizedType.FLAG_SIGNED, i8, f32, 0.99872, 0, -128, 127
        )

        print(f"scale: {uniform.scale}")
        print(f"zero point: {uniform.zero_point}")
        print(f"fixed point: {uniform.is_fixed_point}")
        print(uniform)


# CHECK-LABEL: TEST:  test_multiple_dialects
# CHECK-DAG: #[[map:.+]] = affine_map<(d0)[s0, s1] -> (d0 + ((s0 * 3) floordiv s1) mod d0)>
# CHECK-LABEL: @test_func
#  CHECK-SAME: (%[[arg0:.+]]: tensor<1024x1024xf32>, %[[arg1:.+]]: tensor<1024x1024xf32>) -> (index, tensor<1024x1024xf32>)
#       CHECK:     %[[c0:.+]] = arith.constant 0 : index
#       CHECK:     %[[c1:.+]] = arith.constant 1 : index
#       CHECK:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<1024x1024xf32>
#       CHECK:     %[[dim_0:.+]] = tensor.dim %[[arg1]], %[[c1]] : tensor<1024x1024xf32>
#       CHECK:     %[[v0:.+]] = arith.addi %[[dim]], %[[dim_0]] : index
#       CHECK:     %[[v1:.+]] = bufferization.alloc_tensor() copy(%[[arg1]]) : tensor<1024x1024xf32>
#       CHECK:     return %[[v0]], %[[v1]] : index, tensor<1024x1024xf32>

# CHECK-LABEL: @test_scf
#  CHECK-SAME: (%[[arg0:.+]]: i1) -> i32 {
#       CHECK:     %[[v0]]:2 = scf.if %[[arg0]] -> (i32, i32) {
#       CHECK:       %[[c0_i32:.+]] = arith.constant 0 : i32
#       CHECK:       %[[c1_i32:.+]] = arith.constant 1 : i32
#       CHECK:       scf.yield %[[c0_i32]], %[[c1_i32]] : i32, i32
#       CHECK:     } else {
#       CHECK:       %[[c2_i32:.+]] = arith.constant 2 : i32
#       CHECK:       %[[c3_i32:.+]] = arith.constant 3 : i32
#       CHECK:       scf.yield %[[c2_i32]], %[[c3_i32]] : i32, i32
#       CHECK:     }
#       CHECK:     %[[v1:.+]] = arith.addi %[[v0]]#0, %[[v0]]#1 : i32
#       CHECK:     return %[[v1]] : i32

# CHECK-LABEL: @test_affine_apply
#  CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index) -> index {
#       CHECK:   %[[v0:.+]] = affine.apply #[[map]](%[[arg0]])[%[[arg1]], %[[arg1]]]

# CHECK-LABEL: TEST:  test_quant_dialect
#       CHECK: scale: 0.99872
#       CHECK: zero point: 0
#       CHECK: fixed point: True
#       CHECK: !quant.uniform<i8:f32, 9.987200e-01>
