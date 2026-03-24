---
name: tripy-testing
description: 'Write tests for nvtripy following project conventions. Use when: adding tests for ops, modules, trace operations, or compilation, using pytest parametrize, testing error cases with helper.raises, testing dtype combinations, understanding test directory structure.'
---

# Testing Patterns for nvtripy

## When to Use

- Adding tests for new operations, modules, or features
- Understanding the test directory structure and conventions
- Testing error cases and dtype validation
- Writing parametrized tests for multiple configurations

## Test Directory Structure

The test directory mirrors the source tree:

```
tests/
├── frontend/
│   ├── ops/
│   │   ├── test_binary.py
│   │   ├── test_softmax.py
│   │   └── ...
│   ├── module/
│   │   ├── test_linear.py
│   │   ├── test_layernorm.py
│   │   └── ...
│   ├── test_tensor.py
│   └── ...
├── trace/
│   ├── ops/
│   │   ├── test_binary.py
│   │   └── ...
│   └── ...
├── backend/
│   └── ...
├── integration/
│   └── ...
├── helper.py          # Test utilities
└── conftest.py        # Shared fixtures
```

## Test Utilities (`tests/helper.py`)

### `helper.raises` — Error Testing

```python
from tests import helper
import nvtripy as tp

# Basic error test
with helper.raises(tp.TripyException):
    result = bad_operation()

# With message matching (regex)
with helper.raises(tp.TripyException, match="Expected.*one of"):
    result = bad_operation()

# With stack info verification
a = tp.Tensor([1.0])
with helper.raises(tp.TripyException, has_stack_info_for=[a]):
    result = bad_operation(a)
```

### `helper.config` — Temporary Config Changes

```python
with helper.config("enable_input_validation", False):
    # Validation disabled in this block
    result = operation()
# Automatically restored
```

### `NUMPY_TO_TRIPY` — Dtype Mapping

```python
from tests.helper import NUMPY_TO_TRIPY

# Maps numpy dtypes to tripy dtypes:
# bool → tp.bool, np.int8 → tp.int8, np.int32 → tp.int32,
# np.int64 → tp.int64, np.float16 → tp.float16, np.float32 → tp.float32
```

## Common Test Patterns

### Basic Op Test

```python
import cupy as cp
import numpy as np
import nvtripy as tp


class TestMyOp:
    def test_basic(self):
        input = tp.Tensor([1.0, 2.0, 3.0])
        output = tp.my_op(input)

        expected = np.array([...])  # Compute expected result
        assert np.array_equal(cp.from_dlpack(output).get(), expected)

    def test_with_specific_dim(self):
        input = tp.iota([2, 3], dtype=tp.float32)
        output = tp.my_op(input, dim=1)

        assert output.shape == [2, 3]
```

### Parametrized Dtype Tests

```python
import pytest

class TestMyOp:
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16, tp.bfloat16])
    def test_supported_dtypes(self, dtype):
        input = tp.ones([2, 3], dtype=dtype)
        output = tp.my_op(input)
        assert output.dtype == dtype

    @pytest.mark.parametrize(
        "dtype",
        [tp.int8, tp.int32],
        ids=["int8", "int32"],
    )
    def test_unsupported_dtypes_fail(self, dtype):
        input = tp.ones([2, 3], dtype=dtype)
        with helper.raises(tp.TripyException, match="Expected.*one of"):
            tp.my_op(input)
```

### Testing from NumPy Data

```python
from tests.helper import NUMPY_TO_TRIPY

class TestTensor:
    @pytest.mark.parametrize("dtype", list(NUMPY_TO_TRIPY.keys()))
    def test_dtype_from_numpy(self, dtype):
        np_array = np.array([1, 2, 3], dtype=dtype)
        tensor = tp.Tensor(np_array)
        assert tensor.dtype == NUMPY_TO_TRIPY[dtype]
```

### Mismatched Dtype Error Tests

```python
class TestBinaryOps:
    def test_mismatched_dtypes_fails(self):
        a = tp.Tensor([1.0, 2.0])
        b = tp.ones((2,), dtype=tp.float16)
        with helper.raises(tp.TripyException):
            c = a + b
```

### Module Tests

```python
class TestLinear:
    def test_basic(self):
        linear = tp.Linear(3, 4)
        linear.weight = tp.iota(linear.weight.shape)
        linear.bias = tp.iota(linear.bias.shape)

        input = tp.iota((2, 3))
        output = linear(input)

        assert cp.from_dlpack(output).get().shape == (2, 4)

    def test_no_bias(self):
        linear = tp.Linear(3, 4, bias=False)
        linear.weight = tp.iota(linear.weight.shape)

        input = tp.iota((2, 3))
        output = linear(input)
        assert output.shape == [2, 4]

    def test_state_dict(self):
        linear = tp.Linear(3, 4)
        sd = linear.state_dict()
        assert "weight" in sd
        assert "bias" in sd
```

### Trace Op Tests

```python
from nvtripy.trace.ops.my_op import MyOp
from nvtripy.trace.ops.base import TraceOp

class TestMyTraceOp:
    def test_creates_correct_op(self):
        input = tp.Tensor([1.0, 2.0])
        output = tp.my_op(input)

        assert isinstance(output.trace_tensor.producer, MyOp)

    def test_infer_rank(self):
        input = tp.Tensor([1.0, 2.0])
        output = tp.my_op(input)

        assert output.trace_tensor.rank == 1
```

### Allclose Comparisons

```python
class TestSoftmax:
    def test_matches_torch(self):
        input = tp.iota([2, 3], dtype=tp.float32)
        output = tp.softmax(input, dim=1)

        torch_input = torch.from_dlpack(input)
        torch_output = torch.softmax(torch_input, dim=1)

        assert tp.allclose(output, tp.Tensor(torch_output))
```

### Parametrize with IDs

```python
@pytest.mark.parametrize(
    "tensor_a, tensor_b, rtol, atol, expected",
    [
        (tp.Tensor([1.0]), tp.Tensor([1.0]), 1e-5, 1e-8, True),
        (tp.Tensor([1.0]), tp.Tensor([2.0]), 1e-5, 1e-8, False),
    ],
    ids=["equal", "not_equal"],
)
def test_allclose(self, tensor_a, tensor_b, rtol, atol, expected):
    result = tp.allclose(tensor_a, tensor_b, rtol=rtol, atol=atol)
    assert result == expected
```

## Verifying Outputs

| Method | Use When |
|--------|----------|
| `cp.from_dlpack(output).get()` | Convert tripy tensor → numpy array (via cupy) |
| `np.array_equal(a, b)` | Exact equality for integer/bool results |
| `np.allclose(a, b)` | Approximate equality for float results |
| `tp.allclose(a, b)` | Compare two tripy tensors directly |
| `torch.from_dlpack(tensor)` | Convert tripy tensor → torch tensor |
| `output.shape` | Check output shape |
| `output.dtype` | Check output dtype |

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/frontend/ops/test_softmax.py

# Run specific test
pytest tests/frontend/ops/test_softmax.py::TestSoftmax::test_basic

# Run with verbose output
pytest -v tests/frontend/ops/test_softmax.py

# Run tests matching a pattern
pytest -k "softmax"
```

## Checklist

- [ ] Test file created at `tests/<mirror_of_source_path>/test_<name>.py`
- [ ] Tests organized in a class (e.g., `TestMyOp`)
- [ ] Basic functionality test with assertion
- [ ] Parametrized dtype tests for all supported dtypes
- [ ] Error case tests using `helper.raises(tp.TripyException)`
- [ ] Shape validation tests
- [ ] Comparison against reference implementation (torch/numpy) where applicable
- [ ] Test IDs provided for parametrized tests to make failures readable
