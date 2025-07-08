#!/usr/bin/env python3
"""
Generate MLIR functions exercising arith.remf for a menu of test cases
in multiple element types (f64, f32, f16, bf16, f8E4M3FN).

Each test case translates to a function:
  func.func @remf_case{N}_{TYPE}() -> {TYPE}

Example MLIR (f32):
  %cst0 = arith.constant 5.5 : f32
  %cst1 = arith.constant 2.0 : f32
  %r     = arith.remf %cst0, %cst1 : f32
  return %r : f32
"""
import math
import re
import struct
import sys
from typing import Union


# Width-correct bit patterns for special values.
# Width-correct bit patterns for special values and limits
SPECIAL_HEX = {
    "f64": {
        "nan": "0x7ff8000000000000",
        "-nan": "0xfff8000000000000",
        "inf": "0x7ff0000000000000",
        "-inf": "0xfff0000000000000",
        "max": "0x7fefffffffffffff",  # 1.7976931348623157e+308
        "eps": "0x0000000000000001",  # 5e-324
        "min": "0xffefffffffffffff",  # -1.7976931348623157e+308
    },
    "f32": {
        "nan": "0x7fc00000",
        "-nan": "0xffc00000",
        "inf": "0x7f800000",
        "-inf": "0xff800000",
        "max": "0x7f7fffff",  # 3.4028235e+38
        "eps": "0x00000001",  # 1.4e-45
        "min": "0xff7fffff",  # -3.4028235e+38
    },
    "bf16": {
        "nan": "0x7fc0",
        "-nan": "0xffc0",
        "inf": "0x7f80",
        "-inf": "0xff80",
        "max": "0x7f7f",  # 3.389531e+38
        "eps": "0x0001",  # 9.183549615799121e-41
        "min": "0xff7f",  # -3.389531e+38
    },
    "f16": {
        "nan": "0x7e00",
        "-nan": "0xfe00",
        "inf": "0x7c00",
        "-inf": "0xfc00",
        "max": "0x7bff",  # 65504
        "eps": "0x0001",  # 5.96e-8
        "min": "0xfbff",  # -65504
    },
    "f8E4M3FN": {
        "nan": "0x7f",
        "-nan": "0xff",
        "inf": "448.",
        "-inf": "-448.",
        "max": "448.",  # 448
        "eps": "0x01",  # 0.0625
        "min": "-448.",  # -448
    },
}


# handle type-dependent case "max fmod 3.0"
def case_9_expected(tp: str):
    if tp == "f64":
        return 2.0
    elif tp == "f32":
        return 0.0
    elif tp == "f16":
        return 2.0
    elif tp == "bf16":
        return 0.0
    elif tp == "f8E4M3FN":
        return 1.0


def case_14_expected(tp: str):
    if tp == "f8E4M3FN":
        return 1.0
    return "nan"


# ---------------------------------------------------------------------------
# Test-case catalogue (x, y) — keep in same order as explanation table.
CASES = [
    (5.5, 2.0, 1.5),
    (-5.5, 2.0, -1.5),
    (5.5, -2.0, 1.5),
    (-5.5, -2.0, -1.5),
    (6.0, 2.0, 0.0),
    (-6.0, 2.0, -0.0),
    (1.0, 2.5, 1.0),
    (7.0, 0.5, 0.0),
    ("max", 3.0, case_9_expected),
    ("eps", "eps", 0.0),
    ("max", -2.0, 0.0),
    ("nan", 1.0, "nan"),
    (1.0, "nan", "nan"),
    ("inf", 3.0, case_14_expected),
    (5.0, "inf", 5.0),
    (1.0, 0.0, "nan"),
    (1.0, -0.0, "nan"),
    (-0.0, 1.0, -0.0),
    ("min", 2.0, -0.0),
]


# MLIR element types to cover
# TYPES = ["f64", "f32", "bf16", "f16", "f8E4M3FN"]
TYPES = ["f64", "f32", "f16", "bf16", "f8E4M3FN"]
INT_FOR_FP = {"f64": "i64", "f32": "i32", "bf16": "i16", "f16": "i16", "f8E4M3FN": "i8"}


# ---------------------------------------------------------------------------
def mlir_float_literal(val: Union[float, str], tp: str) -> str:
    """Return an MLIR-parsable literal for *val* in element type *tp*."""
    if isinstance(val, str):
        return SPECIAL_HEX[tp].get(val, val)
    assert isinstance(val, float), f"Expected float, got {type(val)}"
    if math.isnan(val):
        return (
            SPECIAL_HEX[tp]["nan"]
            if math.copysign(1.0, val) >= 0
            else SPECIAL_HEX[tp]["-nan"]
        )
    if val == math.inf:
        return SPECIAL_HEX[tp]["inf"]
    if val == -math.inf:
        return SPECIAL_HEX[tp]["-inf"]

    # Preserve the sign bit of −0.0 explicitly.
    if val == 0.0 and math.copysign(1.0, val) < 0:
        return "-0.0"

    # Ordinary finite value — decimal form is fine (MLIR canonicalises).
    return re.sub(r"^([0-9]+)e", r"\1.e", repr(val))


def expected_rem(x: float, y: float) -> float:
    """IEEE-754 remainder result that `arith.remf` MUST produce."""
    if math.isnan(x) or math.isnan(y):
        return math.copysign(float("nan"), x)
    if y == 0.0:
        return float("nan")
    if math.isinf(x):
        return float("nan")
    if math.isinf(y):
        return x
    return math.fmod(x, y)


def emit_eq_or_both_nan(tp: str):
    print(f"  func.func private @eq_or_both_nan_{tp}(%x: {tp}, %y: {tp}) -> i1 {{")
    print(f"    %cst0 = executor.constant 0.0 : {tp}")
    print(f"    %x_isnan = executor.fcmp <uno> %x, %cst0 : {tp}")
    print(f"    %y_isnan = executor.fcmp <uno> %y, %cst0 : {tp}")
    print(f"    %result0 = executor.bitwise_andi %x_isnan, %y_isnan : i1")
    print(f"    %x_oeq_y = executor.fcmp <oeq> %x, %y : {tp}")
    print(f"    %result1 = executor.bitwise_ori %result0, %x_oeq_y : i1")
    print("    return %result1 : i1")
    print("  }")
    print("")


def emit_print(tp: str):

    msg = f'executor.print "%f remf_{tp} %f = %f (expected: %f)"'
    if tp == "f64" or tp == "f32":
        signature = ", ".join([tp] * 4)
        print(f"    {msg}(%x, %y, %r, %expected : {signature})")
        return
    # Extned operands to f32.
    signature = ", ".join(["f32"] * 4)
    print(f"    %x_f32 = executor.extf %x : {tp} to f32")
    print(f"    %y_f32 = executor.extf %y : {tp} to f32")
    print(f"    %r_f32 = executor.extf %r : {tp} to f32")
    print(f"    %expected_f32 = executor.extf %expected : {tp} to f32")
    print(f"    {msg}(%x_f32, %y_f32, %r_f32, %expected_f32 : {signature})")


def emit_remf(tp: str):
    print(
        f"  func.func private @remf_{tp}(%x: {tp}, %y: {tp}, %expected: {tp}) -> () attributes {{no_inline}} {{"
    )
    print(f"    %r = executor.remf %x, %y : {tp}")
    emit_print(tp)
    print(
        f"    %eq_or_both_nan = call @eq_or_both_nan_{tp}(%r, %expected) : ({tp}, {tp}) -> i1"
    )
    print(f'    executor.assert %eq_or_both_nan, "remf {tp} failed"')
    print(f"    return")
    print("  }")
    print("")


# ---------------------------------------------------------------------------
def emit_module():
    print(
        "//===----------------------------------------------------------------------===//"
    )
    print("//  Auto-generated tests for arith.remf")
    print("//  Generated by generate_remf_tests.py")
    print(
        "//===----------------------------------------------------------------------===//\n"
    )
    print("module {")
    # Loop over element types then tests to group functions by type.
    for tp in TYPES:
        emit_eq_or_both_nan(tp)
        emit_remf(tp)

    test_case_names = []
    for tp in TYPES:
        print(f"  // ---- Element type: {tp} ----")
        elem_type = f"{tp}"
        for idx, (raw_x, raw_y, raw_expected) in enumerate(CASES, start=1):
            if callable(raw_expected):
                raw_expected = raw_expected(tp)
            expected = mlir_float_literal(raw_expected, tp)
            lit_x = mlir_float_literal(raw_x, tp)
            lit_y = mlir_float_literal(raw_y, tp)
            func_name = f"@remf_case{idx:02d}_{tp}"
            test_case_names.append(func_name)
            print(f"  func.func {func_name}() attributes {{no_inline}} {{")
            print(f'    executor.print "test {func_name}"()')
            print(f"    %c0 = executor.constant {lit_x} : {elem_type}")
            print(f"    %c1 = executor.constant {lit_y} : {elem_type}")
            print(f"    %expected = executor.constant {expected} : {elem_type}")
            print(
                f"    call @remf_{elem_type}(%c0, %c1, %expected) : ({elem_type}, {elem_type}, {elem_type}) -> ()"
            )
            print("    return")
            print("  }\n")
    print("")
    print("  // ---- Entrypoint ----")
    print("  func.func @main() -> i32{")
    for name in test_case_names:
        print(f"    call {name}() : () -> ()")
    print("    %c0 = executor.constant 0 : i32")
    print("    return %c0 : i32")
    print("  }")
    print("}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    emit_module()
