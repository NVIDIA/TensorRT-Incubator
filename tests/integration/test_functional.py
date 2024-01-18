import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import tripy.common.datatype
from tripy import jit
from tripy.backend.mlir.compiler import FlatIRCompiler
from tripy.backend.mlir.executor import FlatIRExecutor
from tripy.common.device import device
from tripy.frontend import Dim, Tensor
from tripy.frontend.trace import Trace
from tripy.frontend.tensor_ops import ones


class TestFunctional:
    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_add_two_tensors(self, kind):
        arr = np.array([2, 3], dtype=np.float32)
        a = Tensor(arr, device=device(kind))
        b = Tensor(np.ones(2, dtype=np.float32), device=device(kind))

        c = a + b
        out = c + c
        assert (out.numpy() == np.array([6.0, 8.0], dtype=np.float32)).all()

    @pytest.mark.parametrize("dim", [Dim(2, min=2, opt=3, max=4)])
    def test_add_two_tensors_dynamic(self, dim):
        arr = np.ones(2, dtype=np.float32)
        a = Tensor(arr, shape=(dim,), device=device("gpu"))
        b = Tensor(arr, shape=(dim,), device=device("gpu"))

        @jit
        def func(a, b):
            c = a + b
            return c

        out = func(a, b)
        assert (out.numpy() == np.array([2.0, 2.0], dtype=np.float32)).all()

    @pytest.mark.parametrize(
        "dim_a, dim_b",
        [
            ((1, 3), (3, 3)),  # naive broadcast at 0th dim
            ((3, 3), (3, 1)),  # naive broadcast at 1sh dim of second operand
            ((1, 3, 1), (4, 3, 7)),  # broadcast at multiple dim of same operand
            ((1, 3, 7), (4, 3, 1)),  # broadcast at differnt dim of both operand
        ],
    )
    @pytest.mark.parametrize(
        "use_jit",
        [False, True],
    )
    def test_static_broadcast_add_two_tensors(self, dim_a, dim_b, use_jit):
        np_a = np.random.rand(*dim_a).astype(np.float32)
        np_b = np.random.rand(*dim_b).astype(np.float32)
        a = Tensor(np_a, shape=dim_a, device=device("gpu"))
        b = Tensor(np_b, shape=dim_b, device=device("gpu"))

        def func(a, b):
            c = a + b
            return c

        if use_jit:
            func = tripy.jit(func)

        out = func(a, b)
        assert (out.numpy() == np.array(np_a + np_b)).all()

    def test_multi_output_trace(self):
        arr = np.ones(2, dtype=np.float32)
        a = Tensor(arr)
        b = Tensor(arr)
        c = a + b
        d = c + c
        trace = Trace([c, d])
        flat_ir = trace.to_flat_ir()
        i_tensor_info, o_tensor_info = flat_ir.io_tensor_info()
        output_devices = [o.device for o in trace.outputs]

        compiler = FlatIRCompiler()
        with FlatIRExecutor(compiler.compile(flat_ir), output_devices, i_tensor_info, o_tensor_info) as executor:
            out = executor.execute()
            assert (
                len(out) == 2
                and (out[0].data.view().get() == np.array([2.0, 2.0], dtype=np.float32)).all()
                and (out[1].data.view().get() == np.array([4.0, 4.0], dtype=np.float32)).all()
            )

    def _test_framework_interoperability(self, data, device):
        a = Tensor(data, device=device)
        b = Tensor(torch.tensor(data), device=device)

        if device.kind == "gpu":
            if isinstance(data, cp.ndarray):
                data = data.get()
            c = Tensor(jax.device_put(jnp.array(data), jax.devices("gpu")[0]), device=device)
        else:
            c = Tensor(jax.device_put(jnp.array(data), jax.devices("cpu")[0]))

        out = a + b + c
        assert (out.numpy() == np.array([3.0, 3.0], dtype=np.float32)).all()

    def test_cpu_and_gpu_framework_interoperability(self):
        from tripy.common.device import device as make_device

        self._test_framework_interoperability(np.ones(2, np.float32), device=make_device("cpu"))
        self._test_framework_interoperability(cp.ones(2, cp.float32), device=make_device("gpu"))

    def _assert_round_tripping(
        self, original_data, tensor, round_trip, compare, data_type=tripy.common.datatype.float32
    ):
        """Assert round-tripping for different frameworks."""
        round_tripped_data = tensor.numpy()
        assert (round_tripped_data == original_data).all()
        assert round_tripped_data.data == original_data.data

    def _test_round_tripping(self, data, device):
        assert isinstance(data, np.ndarray) or isinstance(data, cp.ndarray)

        # Assert round-tripping for numpy or cupy array
        xp_orig = data
        if device.kind == "gpu":
            xp_round_tripped = cp.array(Tensor(xp_orig, device=device).numpy())
        else:
            xp_round_tripped = np.array(Tensor(xp_orig, device=device).numpy())
        assert (xp_round_tripped == xp_orig).all()
        # (39): Remove explicit CPU to GPU copies. Add memory pointer checks.
        # assert xp_round_tripped.data == xp_orig.data

        # Assert round-tripping for Torch tensor
        torch_orig = torch.as_tensor(data)
        torch_round_tripped = torch.as_tensor(Tensor(torch_orig, device=device).numpy())
        assert torch.equal(torch_round_tripped, torch_orig)
        # (39): Remove explicit CPU to GPU copies. Add memory pointer checks.
        # Below fails as we do allocate a new np array from Torch tensor data.
        # assert torch_data_round_tripped.data_ptr == torch_data.data_ptr

        # Assert round-tripping for Jax data
        if device.kind == "gpu":
            if isinstance(data, cp.ndarray):
                data = data.get()
            jax_orig = jax.device_put(jnp.array(data), jax.devices("gpu")[0])
            jax_round_tripped = jnp.array(Tensor(jax_orig, device=device).numpy())
        else:
            jax_orig = jax.device_put(jnp.array(data), jax.devices("cpu")[0])
            jax_round_tripped = jnp.array(Tensor(jax_orig, device=device).numpy())
        assert jnp.array_equal(jax_round_tripped, jax_orig)
        # (39): Remove explicit CPU to GPU copies. Add memory pointer checks.
        # Figure out how to compare two Jax data memory pointers.

        # Assert round-tripping for List data
        if device.kind == "cpu":
            list_orig = data.tolist()
            list_round_tripped = Tensor(list_orig, shape=(2,)).numpy().tolist()
            assert list_round_tripped == list_orig
            # (39): Remove explicit CPU to GPU copies. Add memory pointer checks.
            # assert id(list_round_tripped) == id(list_orig)

    def test_tensor_round_tripping(self):
        from tripy.common.device import device as make_device

        self._test_round_tripping(np.ones(2, np.float32), device=make_device("cpu"))
        self._test_round_tripping(cp.ones(2, cp.float32), device=make_device("gpu"))


class TestCopyFunctional:
    @pytest.mark.parametrize("src", ["cpu", "gpu"])
    @pytest.mark.parametrize("dst", ["cpu", "gpu"])
    def test_single_copy(self, src, dst):
        a = tripy.Tensor([1, 2], device=device(src))
        out = a.to(device(dst))
        out = out.eval()
        assert out.device.kind == dst
        assert out.view().tolist() == [1, 2]

    def test_multiple_copy_1(self):
        a = tripy.Tensor([1, 2])
        a = a.to(device("gpu"))
        a = a.to(device("cpu"))
        out = a.eval()
        assert out.device.kind == "cpu"
        assert out.view().tolist() == [1, 2]

    def test_multiple_copy_2(self):
        a = tripy.Tensor([1, 2])
        a = a.to(device("cpu"))
        a = a.to(device("gpu"))
        out = a.eval()
        assert out.device.kind == "gpu"
        assert out.view().tolist() == [1, 2]

    @pytest.mark.parametrize("dst", ["cpu", "gpu"])
    def test_jit_single_copy(self, dst):
        a = tripy.Tensor([1, 2], device=device("gpu"))

        @tripy.jit
        def func(x):
            x = x.to(device(dst))
            return x

        out = func(a)
        out = out.eval()
        assert out.device.kind == dst
        assert out.view().tolist() == [1, 2]

    def test_jit_multiple_copy_1(self):
        a = tripy.Tensor([1, 2], device=device("gpu"))

        @tripy.jit
        def func(x):
            x = x.to(device("cpu"))
            x = x.to(device("gpu"))
            return x

        out = func(a)
        out = out.eval()
        assert out.device.kind == "gpu"
        assert out.view().tolist() == [1, 2]

    def test_jit_multiple_copy_2(self):
        a = tripy.Tensor([1, 2], device=device("gpu"))

        @tripy.jit
        def func(x):
            x = x.to(device("gpu"))
            x = x.to(device("cpu"))
            return x

        out = func(a)
        out = out.eval()
        assert out.device.kind == "cpu"
        assert out.view().tolist() == [1, 2]

    def test_with_ops(self):
        a = tripy.Tensor([1, 2])
        b = tripy.Tensor([2, 3])
        out = a + b
        out = out.to(device("cpu"))
        out = out.eval()
        assert out.device.kind == "cpu"
        assert out.view().tolist() == [3, 5]

    def test_jit_with_ops(self):
        a = tripy.Tensor([1, 2], device=device("gpu"))
        b = tripy.Tensor([2, 3], device=device("gpu"))

        @tripy.jit
        def func(x, y):
            out = x + y
            out = out.to(device("cpu"))
            return out

        out = func(a, b)
        out = out.eval()
        assert out.device.kind == "cpu"
        assert out.view().tolist() == [3, 5]

    def test_print_ds_tensor(self):
        arr = np.ones(4, dtype=np.float32)
        a = tripy.Tensor(arr, shape=(Dim(4, min=2, opt=4, max=6),), device=device("gpu"))
        assert (a.numpy() == arr).all()

    def test_print_static_tensor(self):
        arr = np.ones(4, dtype=np.float32)
        a = tripy.Tensor(arr, shape=(4,), device=device("gpu"))
        assert (a.numpy() == arr).all()


class TestDynamic:
    @pytest.mark.parametrize("dim", [Dim(4, min=2, opt=4, max=6)])
    def test_dynamic_jit(self, dim):
        from tripy.common.logging import set_logger_mode, LoggerModes, get_log_str

        set_logger_mode(LoggerModes.VERBOSE, True)

        a = Tensor(np.ones(4, dtype=np.float32), shape=(dim,), device=device("gpu"))
        b = Tensor(np.ones(4, dtype=np.float32), shape=(dim,), device=device("gpu"))

        @jit
        def func(a, b):
            c = a + b
            return c

        out = func(a, b)
        assert (out.numpy() == np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)).all()

        print("Re-run dynamic shape test with a different input shape.")

        a = Tensor(np.ones(3, dtype=np.float32), device=device("gpu"))
        b = Tensor(np.ones(3, dtype=np.float32), device=device("gpu"))

        out = func(a, b)
        assert (out.numpy() == np.array([2.0, 2.0, 2.0], dtype=np.float32)).all()
        # 1 compile call for stablehlo add and 2 compile calls for device copy.
        assert get_log_str().count("compile(<tripy.backend.mlir.compiler.FlatIRCompile") == 3

    @pytest.mark.parametrize("dim", [Dim(4, min=2, opt=4, max=6)])
    def test_dynamic_lazy(self, dim):
        from tripy.common.logging import set_logger_mode, LoggerModes, get_log_str

        a = Tensor(np.ones(4, dtype=np.float32), shape=(dim,), device=device("gpu"))
        b = Tensor(np.ones(4, dtype=np.float32), shape=(dim,), device=device("gpu"))

        def func(a, b):
            c = a + b
            return c

        out = func(a, b)
        assert (out.numpy() == np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)).all()


class TestReshape:
    @pytest.mark.parametrize(
        "shape, new_shape",
        [
            ((2, 4), (1, 8)),
            ((2, 4, 8, 9), (8, 8, 9)),
            ((2, 4), (8,)),  # change rank of output
        ],
    )
    def test_static_reshape(self, shape, new_shape):
        np_a = np.random.rand(*shape).astype(np.float32)
        a = Tensor(np_a, shape=shape, device=device("gpu"))
        b = a.reshape(new_shape)
        assert (b.shape.numpy() == np.array(new_shape)).all()
        assert (b.numpy() == np.array(np_a.reshape(new_shape))).all()

    def test_dynamic_reshape(self):
        dim = Dim(runtime_value=4, min=3, opt=5, max=6)
        a = ones((dim, 5, 6, 7))
        with pytest.raises(NotImplementedError):
            a = a.reshape((20, 3, 14))
            print(a)


class TestConversionToTripyType:
    @pytest.mark.parametrize(
        "reverse_direction",
        [False],
        # #84 will fix issues found with reverse direction implementation.
    )
    @pytest.mark.parametrize(
        "input0",
        [np.ones((2, 3), dtype=np.float32), np.ones((3,), dtype=np.float32)],
    )
    @pytest.mark.parametrize(
        "input1",
        [
            [
                4.0,
            ],
            (5.0,),
            np.array([4.0], dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            torch.Tensor([[4.0]]),
        ],
    )
    def test_element_wise_prod(self, reverse_direction, input0, input1):
        from tripy.common.logging import set_logger_mode, LoggerModes, get_log_str

        set_logger_mode(LoggerModes.VERBOSE, True)

        a = Tensor(input0)
        if reverse_direction:
            import copy

            out = input1 * a
            input0, input1 = input1, input0
        else:
            out = a * input1

        if isinstance(input1, torch.Tensor):
            input1 = input1.numpy()
        assert np.array_equal(out.numpy(), np.array(input0 * input1))
