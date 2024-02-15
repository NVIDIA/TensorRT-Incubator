import pytest
from mlir import ir

import tripy
from tripy.backend.mlir import utils as mlir_utils
from tripy.common.datatype import DATA_TYPES
import os


class TestUtils:
    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_convert_dtype(self, dtype):
        if dtype in {tripy.bool}:
            pytest.skip("Bool is not working correctly yet")

        with mlir_utils.make_ir_context():
            assert (
                mlir_utils.get_mlir_dtype(dtype)
                == {
                    "float32": ir.F32Type.get(),
                    "float16": ir.F16Type.get(),
                    "float8e4m3fn": ir.Float8E4M3FNType.get(),
                    "bfloat16": ir.BF16Type.get(),
                    "int4": ir.IntegerType.get_signless(4),
                    "int8": ir.IntegerType.get_signless(8),
                    "int32": ir.IntegerType.get_signless(32),
                    "int64": ir.IntegerType.get_signless(64),
                    "uint8": ir.IntegerType.get_unsigned(8),
                    # TODO (pranavm): Figure out how to make boolean types work.
                }[dtype.name]
            )

    def test_env(self):
        ENV_VAR = "test_env_test_env_var"
        assert ENV_VAR not in os.environ

        with mlir_utils.env({ENV_VAR: "1"}):
            assert ENV_VAR in os.environ
            assert os.environ[ENV_VAR] == "1"

        assert ENV_VAR not in os.environ
