window.BENCHMARK_DATA = {
  "lastUpdate": 1728324952675,
  "repoUrl": "https://github.com/NVIDIA/TensorRT-Incubator",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "name": "NVIDIA",
            "username": "NVIDIA"
          },
          "committer": {
            "name": "NVIDIA",
            "username": "NVIDIA"
          },
          "id": "d181ee2d53cf642915c285c7ad5873a7e07ee78d",
          "message": "testing: Do not merge!!",
          "timestamp": "2024-10-04T19:42:03Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/221/commits/d181ee2d53cf642915c285c7ad5873a7e07ee78d"
        },
        "date": 1728323688159,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6138.66113359083,
            "unit": "iter/sec",
            "range": "stddev: 0.00004083209986898708",
            "extra": "mean: 162.9019713318247 usec\nrounds: 6510"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6828.517183547142,
            "unit": "iter/sec",
            "range": "stddev: 0.00009466167290389844",
            "extra": "mean: 146.44467797627186 usec\nrounds: 7438"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "NVIDIA",
            "username": "NVIDIA"
          },
          "committer": {
            "name": "NVIDIA",
            "username": "NVIDIA"
          },
          "id": "a22524ea2d8a6276be9c6dc1b8d4f218f03dcdfa",
          "message": "testing: Do not merge!!",
          "timestamp": "2024-10-04T19:42:03Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/221/commits/a22524ea2d8a6276be9c6dc1b8d4f218f03dcdfa"
        },
        "date": 1728324951899,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5963.606152667102,
            "unit": "iter/sec",
            "range": "stddev: 0.00004274076107186598",
            "extra": "mean: 167.68377629243176 usec\nrounds: 6449"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6615.575139595522,
            "unit": "iter/sec",
            "range": "stddev: 0.0001021827292225658",
            "extra": "mean: 151.15843730876892 usec\nrounds: 7284"
          }
        ]
      }
    ]
  }
}