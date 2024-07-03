import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import tripy as tp
from tripy.backend.utils import get_tensor_info, get_devices
from tripy.backend.mlir.compiler import Compiler
from tripy.backend.mlir.executor import Executor
from tripy.frontend.trace import Trace
from tripy.logging import logger


class TestFunctional:
    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_add_two_tensors(self, kind):
        module = cp if kind == "gpu" else np
        arr = module.array([2, 3], dtype=np.float32)
        a = tp.Tensor(arr, device=tp.device(kind))
        b = tp.Tensor(module.ones(2, dtype=module.float32), device=tp.device(kind))

        c = a + b
        out = c + c
        assert (cp.from_dlpack(out).get() == np.array([6.0, 8.0], dtype=np.float32)).all()

    @pytest.mark.parametrize("dim", [tp.dynamic_dim(2, min=2, opt=3, max=4)])
    def test_add_two_tensors_dynamic(self, dim):
        arr = cp.ones(2, dtype=np.float32)
        a = tp.Tensor(arr, shape=(dim,), device=tp.device("gpu"))
        b = tp.Tensor(arr, shape=(dim,), device=tp.device("gpu"))

        @tp.jit
        def func(a, b):
            c = a + b
            return c

        out = func(a, b)
        assert (cp.from_dlpack(out).get() == np.array([2.0, 2.0], dtype=np.float32)).all()

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
        cp_a = cp.random.rand(*dim_a).astype(np.float32)
        cp_b = cp.random.rand(*dim_b).astype(np.float32)
        a = tp.Tensor(cp_a, shape=dim_a, device=tp.device("gpu"))
        b = tp.Tensor(cp_b, shape=dim_b, device=tp.device("gpu"))

        def func(a, b):
            c = a + b
            return c

        if use_jit:
            func = tp.jit(func)

        out = func(a, b)
        assert (cp.from_dlpack(out) == cp.array(cp_a + cp_b)).all()

    def test_multi_output_trace(self):
        arr = cp.ones(2, dtype=np.float32)
        a = tp.Tensor(arr)
        b = tp.Tensor(arr)
        c = a + b
        d = c + c
        trace = Trace([c, d])
        flat_ir = trace.to_flat_ir()

        compiler = Compiler()
        output_tensor_info = get_tensor_info(flat_ir.outputs)
        executor = Executor(compiler.compile(flat_ir.to_mlir()))
        out = executor.execute(get_devices(output_tensor_info))
        assert (
            len(out) == 2
            and (cp.from_dlpack(out[0]) == cp.array([2.0, 2.0], dtype=np.float32)).all()
            and (cp.from_dlpack(out[1]) == cp.array([4.0, 4.0], dtype=np.float32)).all()
        )

    def _test_framework_interoperability(self, data, device):
        a = tp.Tensor(data, device=device)
        if device.kind == "gpu":
            b = tp.Tensor(torch.tensor(data).to("cuda"), device=device)
        else:
            b = tp.Tensor(torch.tensor(data), device=device)

        if device.kind == "gpu":
            if isinstance(data, cp.ndarray):
                data = data.get()
            c = tp.Tensor(jax.device_put(jnp.array(data), jax.devices("gpu")[0]), device=device)
        else:
            c = tp.Tensor(jax.device_put(jnp.array(data), jax.devices("cpu")[0]), device=device)

        out = a + b + c
        assert (cp.from_dlpack(out).get() == np.array([3.0, 3.0], dtype=np.float32)).all()

    def test_cpu_and_gpu_framework_interoperability(self):
        self._test_framework_interoperability(np.ones(2, np.float32), device=tp.device("cpu"))
        self._test_framework_interoperability(cp.ones(2, cp.float32), device=tp.device("gpu"))

    def _test_round_tripping(self, data, device):
        assert isinstance(data, np.ndarray) or isinstance(data, cp.ndarray)

        # Assert round-tripping for numpy or cupy array
        xp_orig = data
        if device.kind == "gpu":
            xp_round_tripped = cp.array(cp.from_dlpack(tp.Tensor(xp_orig, device=device)))
        else:
            xp_round_tripped = np.array(np.from_dlpack(tp.Tensor(xp_orig, device=device)))
        assert (xp_round_tripped == xp_orig).all()
        # (39): Remove explicit CPU to GPU copies. Add memory pointer checks.
        # assert xp_round_tripped.data == xp_orig.data

        # Assert round-tripping for Torch tensor
        torch_orig = torch.as_tensor(data)
        if device.kind == "gpu":
            torch_round_tripped = torch.as_tensor(cp.from_dlpack(tp.Tensor(torch_orig.to("cuda"), device=device)))
        else:
            torch_round_tripped = torch.as_tensor(np.from_dlpack(tp.Tensor(torch_orig, device=device)))
        assert torch.equal(torch_round_tripped, torch_orig)
        # (39): Remove explicit CPU to GPU copies. Add memory pointer checks.
        # Below fails as we do allocate a new np array from Torch tensor data.
        # assert torch_data_round_tripped.data_ptr == torch_data.data_ptr

        # Assert round-tripping for Jax data
        if device.kind == "gpu":
            if isinstance(data, cp.ndarray):
                data = data.get()
            jax_orig = jax.device_put(jnp.array(data), jax.devices("gpu")[0])
            jax_round_tripped = jnp.array(cp.from_dlpack(tp.Tensor(jax_orig, device=device)).get())
        else:
            jax_orig = jax.device_put(jnp.array(data), jax.devices("cpu")[0])
            jax_round_tripped = jnp.array(np.from_dlpack(tp.Tensor(jax_orig, device=device)))
        assert jnp.array_equal(jax_round_tripped, jax_orig)
        # (39): Remove explicit CPU to GPU copies. Add memory pointer checks.
        # Figure out how to compare two Jax data memory pointers.

        # Assert round-tripping for List data
        if device.kind == "cpu":
            list_orig = data.tolist()
            list_round_tripped = np.from_dlpack(tp.Tensor(list_orig, shape=(2,), device=device)).tolist()
            assert list_round_tripped == list_orig
            # (39): Remove explicit CPU to GPU copies. Add memory pointer checks.
            # assert id(list_round_tripped) == id(list_orig)

    def test_tensor_round_tripping(self):
        self._test_round_tripping(np.ones(2, np.float32), device=tp.device("cpu"))
        self._test_round_tripping(cp.ones(2, cp.float32), device=tp.device("gpu"))

    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_weights_loading_from_torch(self, kind):
        with torch.no_grad():
            if kind == "gpu":
                inp = torch.randn((2, 2), dtype=torch.float32).to("cuda")
            else:
                inp = torch.randn((2, 2), dtype=torch.float32)

            if kind == "gpu":
                torch_linear = torch.nn.Linear(2, 3).to("cuda")
            else:
                torch_linear = torch.nn.Linear(2, 3)
            torch_out = torch_linear(inp)

            tripy_linear = tp.Linear(2, 3)
            if kind == "gpu":
                tripy_linear.weight = tp.Parameter(tp.Tensor(torch_linear.weight.to("cuda"), device=tp.device(kind)))
                tripy_linear.bias = tp.Parameter(tp.Tensor(torch_linear.bias.to("cuda"), device=tp.device(kind)))
            else:
                tripy_linear.weight = tp.Parameter(tp.Tensor(torch_linear.weight, device=tp.device(kind)))
                tripy_linear.bias = tp.Parameter(tp.Tensor(torch_linear.bias, device=tp.device(kind)))

            tripy_out = tripy_linear(tp.Tensor(inp, device=tp.device(kind)))
            assert np.allclose(cp.from_dlpack(tripy_out).get(), torch_out.cpu().numpy())


class TestCopyFunctional:
    @pytest.mark.parametrize("src", ["cpu", "gpu"])
    @pytest.mark.parametrize("dst", ["cpu", "gpu"])
    def test_single_copy(self, src, dst):
        a = tp.Tensor([1, 2], device=tp.device(src))
        out = tp.copy(a, tp.device(dst))
        out = out.eval()
        assert out.device.kind == dst
        assert out.data() == [1, 2]

    def test_multiple_copy_1(self):
        a = tp.Tensor([1, 2])
        a = tp.copy(a, tp.device("gpu"))
        a = tp.copy(a, tp.device("cpu"))
        out = a.eval()
        assert out.device.kind == "cpu"
        assert out.data() == [1, 2]

    def test_multiple_copy_2(self):
        a = tp.Tensor([1, 2])
        a = tp.copy(a, tp.device("cpu"))
        a = tp.copy(a, tp.device("gpu"))
        out = a.eval()
        assert out.device.kind == "gpu"
        assert out.data() == [1, 2]

    @pytest.mark.parametrize("dst", ["cpu", "gpu"])
    def test_jit_single_copy(self, dst):
        a = tp.Tensor([1, 2], device=tp.device("gpu"))

        @tp.jit
        def func(x):
            x = tp.copy(x, tp.device(dst))
            return x

        out = func(a)
        out = out.eval()
        assert out.device.kind == dst
        assert out.data() == [1, 2]

    def test_jit_multiple_copy_1(self):
        a = tp.Tensor([1, 2], device=tp.device("gpu"))

        @tp.jit
        def func(x):
            x = tp.copy(x, tp.device("cpu"))
            x = tp.copy(x, tp.device("gpu"))
            return x

        out = func(a)
        out = out.eval()
        assert out.device.kind == "gpu"
        assert out.data() == [1, 2]

    def test_jit_multiple_copy_2(self):
        a = tp.Tensor([1, 2], device=tp.device("gpu"))

        @tp.jit
        def func(x):
            x = tp.copy(x, tp.device("gpu"))
            x = tp.copy(x, tp.device("cpu"))
            return x

        out = func(a)
        out = out.eval()
        assert out.device.kind == "cpu"
        assert out.data() == [1, 2]

    @pytest.mark.skip("Remove copy op in Tripy: https://gitlab-master.nvidia.com/initialdl/mlir-tensorrt/-/issues/756")
    def test_with_ops(self):
        a = tp.Tensor([1, 2])
        b = tp.Tensor([2, 3])
        out = a + b
        out = tp.copy(out, tp.device("cpu"))
        out = out.eval()
        assert out.device.kind == "cpu"
        assert out.data() == [3, 5]

    def test_jit_with_ops(self):
        a = tp.Tensor([1, 2], device=tp.device("gpu"))
        b = tp.Tensor([2, 3], device=tp.device("gpu"))

        @tp.jit
        def func(x, y):
            out = x + y
            out = tp.copy(out, tp.device("cpu"))
            return out

        out = func(a, b)
        out = out.eval()
        assert out.device.kind == "cpu"
        assert out.data() == [3, 5]

    def test_print_dynamic_tensor(self):
        arr = cp.ones(4, dtype=np.float32)
        a = tp.Tensor(arr, shape=(tp.dynamic_dim(4, min=2, opt=4, max=6),), device=tp.device("gpu"))
        assert cp.array_equal(cp.from_dlpack(a), arr)

    def test_print_static_tensor(self):
        arr = cp.ones(4, dtype=np.float32)
        a = tp.Tensor(arr, shape=(4,), device=tp.device("gpu"))
        assert cp.array_equal(cp.from_dlpack(a), arr)


class TestDynamic:
    @pytest.mark.parametrize(
        "dims_a, dims_b",
        [
            ((tp.dynamic_dim(4, min=2, opt=4, max=6), 2), (tp.dynamic_dim(4, min=2, opt=4, max=6), 2)),
            (
                (tp.dynamic_dim(4, min=2, opt=4, max=6), 2),
                (tp.dynamic_dim(4, min=2, opt=4, max=6), 1),
            ),  # use DynamicBroadcast static dim
            ((tp.dynamic_dim(4, min=2, opt=4, max=6), 2), (1, 2)),  # use DynamicBroadcast dynamic dim
            ((1, 2), (tp.dynamic_dim(4, min=2, opt=4, max=6), 2)),  # use DynamicBroadcast dynamic dim
        ],
    )
    def test_dynamic_jit(self, dims_a, dims_b, capsys):
        with logger.use_verbosity({"mlir"}):

            def get_np_dims(dims, dim_func):
                return [dim_func(d) if isinstance(d, tp.dynamic_dim) else d for d in dims]

            a_cp = cp.random.rand(*get_np_dims(dims_a, lambda x: x.runtime_value)).astype(cp.float32)
            b_cp = cp.random.rand(*get_np_dims(dims_b, lambda x: x.runtime_value)).astype(cp.float32)

            a = tp.Tensor(a_cp, shape=dims_a, device=tp.device("gpu"))
            b = tp.Tensor(b_cp, shape=dims_b, device=tp.device("gpu"))

            @tp.jit
            def func(a, b):
                c = a + b
                return c

            out = func(a, b)
            assert cp.array_equal(cp.from_dlpack(out), cp.array(a_cp + b_cp))
            print("Re-run dynamic shape test with a different input shape.")

            a_cp = cp.random.rand(*get_np_dims(dims_a, lambda x: x.max)).astype(np.float32)
            b_cp = cp.random.rand(*get_np_dims(dims_b, lambda x: x.max)).astype(np.float32)

            a = tp.Tensor(a_cp, device=tp.device("gpu"))
            b = tp.Tensor(b_cp, device=tp.device("gpu"))

            out = func(a, b)
            assert cp.array_equal(cp.from_dlpack(out), cp.array(a_cp + b_cp))
            # 1 compile call for stablehlo add.
            captured = capsys.readouterr()
            assert "stablehlo.add" in captured.out.strip()

    @pytest.mark.parametrize("dim", [tp.dynamic_dim(4, min=2, opt=4, max=6)])
    def test_dynamic_lazy(self, dim):
        a = tp.Tensor(cp.ones(4, dtype=cp.float32), shape=(dim,), device=tp.device("gpu"))
        b = tp.Tensor(cp.ones(4, dtype=cp.float32), shape=(dim,), device=tp.device("gpu"))

        def func(a, b):
            c = a + b
            return c

        out = func(a, b)
        assert cp.array_equal(cp.from_dlpack(out), cp.array([2.0, 2.0, 2.0, 2.0], dtype=cp.float32))


class TestConversionToTripyType:

    @pytest.mark.parametrize(
        "reverse_direction",
        [False, True],
    )
    @pytest.mark.parametrize(
        "input0",
        [cp.ones((2, 3), dtype=cp.float32), cp.ones((3,), dtype=np.float32)],
    )
    @pytest.mark.parametrize(
        "input1",
        [
            [
                4.0,
            ],
            (5.0,),
            cp.array([4.0], dtype=cp.float32),
            cp.ones((1, 3), dtype=cp.float32),
            torch.Tensor([[4.0]]),
        ],
    )
    def test_element_wise_prod(self, reverse_direction, input0, input1):
        a = tp.Tensor(input0)
        if isinstance(input1, torch.Tensor):
            input1 = input1.to("cuda")
        if reverse_direction:
            out = input1 * a
            input0, input1 = input1, input0
        else:
            out = a * input1
        assert cp.array_equal(cp.from_dlpack(out), cp.array(input0) * cp.array(input1))
