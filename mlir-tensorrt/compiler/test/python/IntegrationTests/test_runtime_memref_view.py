# RUN: %pick-one-gpu %PYTHON %s
import gc

import mlir_tensorrt.runtime.api as runtime
import numpy as np


def assert_error(expected_substr, fn):
    try:
        fn()
    except Exception as exc:
        msg = str(exc)
        assert expected_substr in msg, (
            f"expected '{expected_substr}' in '{msg}'"
        )
    else:
        assert False, f"expected error containing '{expected_substr}'"


def assert_error_any(expected_substrs, fn):
    try:
        fn()
    except Exception as exc:
        msg = str(exc)
        assert any(substr in msg for substr in expected_substrs), (
            f"expected one of {expected_substrs} in '{msg}'"
        )
    else:
        assert False, f"expected error containing one of {expected_substrs}"


def make_host_view(client, array, shape):
    return client.create_host_memref_view(
        ptr=int(array.ctypes.data),
        shape=shape,
        dtype=runtime.ScalarTypeCode.f32,
    )


def main():
    client = runtime.RuntimeClient()

    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    memref = make_host_view(client, data, [2, 3, 4])

    view = memref.slice(slice(None, None, 2))
    assert view.shape == [1, 3, 4]
    assert view.strides == [24, 4, 1]
    assert np.asarray(view).tolist() == data[0:2:2, :, :].tolist()

    view = memref.slice(1)
    assert view.shape == [3, 4]
    assert view.strides == [4, 1]
    assert np.asarray(view).tolist() == data[1, :, :].tolist()

    view = memref.slice((slice(0, 2), ..., slice(1, 4, 2)))
    assert view.shape == [2, 3, 2]
    assert view.strides == [12, 4, 2]
    assert np.asarray(view).tolist() == data[0:2, :, 1:4:2].tolist()

    view = memref.slice((-1, slice(None), 1))
    assert view.shape == [3]
    assert view.strides == [4]
    assert np.asarray(view).tolist() == data[1, :, 1].tolist()

    view = memref.slice((1, slice(None), 1), squeeze_unit_dims=False)
    assert view.shape == [1, 3, 1]
    assert view.strides == [12, 4, 1]
    assert np.asarray(view).tolist() == data[1:2, :, 1:2].tolist()

    view = memref.slice(
        (slice(0, 1), slice(None), slice(2, 3)), squeeze_unit_dims=True
    )
    assert view.shape == [3]
    assert view.strides == [4]
    assert np.asarray(view).tolist() == data[0, :, 2].tolist()

    view = memref[(slice(0, 1), slice(None), slice(2, 3))]
    assert view.shape == [1, 3, 1]
    assert view.strides == [12, 4, 1]
    assert np.asarray(view).tolist() == data[0:1, :, 2:3].tolist()

    view_data = np.arange(10, dtype=np.float32)
    base = make_host_view(client, view_data, [10])
    view = base.slice_view(
        offsets=[3],
        sizes=[4],
        strides=[1],
        squeeze_unit_dims=False,
    )
    ref_count_before = view.ref_count()
    assert ref_count_before >= 2
    del base
    gc.collect()
    assert view.ref_count() == ref_count_before - 1
    assert np.asarray(view).tolist() == [3.0, 4.0, 5.0, 6.0]

    dyn_data = np.zeros(1, dtype=np.float32)
    dyn_memref = make_host_view(client, dyn_data, [-1, 3, 4])
    assert_error(
        "slicing requires static dimensions",
        lambda: dyn_memref.slice((slice(None), slice(None), slice(None))),
    )

    assert_error(
        "only one ellipsis is supported",
        lambda: memref.slice((..., ..., slice(None))),
    )
    assert_error(
        "newaxis (None) is not supported",
        lambda: memref.slice((None, slice(None), slice(None))),
    )
    assert_error(
        "too many indices for memref rank",
        lambda: memref.slice(
            (slice(None), slice(None), slice(None), slice(None))
        ),
    )
    assert_error(
        "too many indices for memref rank",
        lambda: memref.slice(
            (..., slice(None), slice(None), slice(None), slice(None))
        ),
    )

    assert_error(
        "indices must be integers or slices",
        lambda: memref.slice((0, 1.5, slice(None))),
    )
    assert_error(
        "index is out of bounds",
        lambda: memref.slice((2, slice(None), slice(None))),
    )
    assert_error(
        "index is out of bounds",
        lambda: memref.slice((-3, slice(None), slice(None))),
    )
    assert_error(
        "invalid slice for dimension",
        lambda: memref.slice((slice("bad"), slice(None), slice(None))),
    )
    assert_error_any(
        ["slice step must be non-zero", "invalid slice for dimension"],
        lambda: memref.slice(
            (slice(None, None, 0), slice(None), slice(None))
        ),
    )


if __name__ == "__main__":
    main()
