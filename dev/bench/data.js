window.BENCHMARK_DATA = {
  "lastUpdate": 1729108673853,
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
          "id": "7269ac03a03b97575d455613d8180ab921910412",
          "message": "Adds performance testing",
          "timestamp": "2024-10-08T17:27:12Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/221/commits/7269ac03a03b97575d455613d8180ab921910412"
        },
        "date": 1728410630114,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6078.349413403918,
            "unit": "iter/sec",
            "range": "stddev: 0.00003917498053986113",
            "extra": "mean: 164.5183473320585 usec\nrounds: 6449"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6847.916989914084,
            "unit": "iter/sec",
            "range": "stddev: 0.00009487584053039735",
            "extra": "mean: 146.02980752728814 usec\nrounds: 7513"
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
          "id": "ce83abee83b808a852a918548e5ded7b40e6b2b1",
          "message": "Update pyproject.toml to use 0.1.34 MLIR-TRT",
          "timestamp": "2024-10-09T16:45:52Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/261/commits/ce83abee83b808a852a918548e5ded7b40e6b2b1"
        },
        "date": 1728495811030,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6012.32472136634,
            "unit": "iter/sec",
            "range": "stddev: 0.000033529936354029256",
            "extra": "mean: 166.3250150887964 usec\nrounds: 6388"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6700.311722038496,
            "unit": "iter/sec",
            "range": "stddev: 0.00006691655825094381",
            "extra": "mean: 149.2467875353956 usec\nrounds: 7245"
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
          "id": "f633dbbc40c373012f60c03b99a3d434c7c37297",
          "message": "Add tp.pad",
          "timestamp": "2024-10-09T19:44:26Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/203/commits/f633dbbc40c373012f60c03b99a3d434c7c37297"
        },
        "date": 1728504599645,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6169.062462876753,
            "unit": "iter/sec",
            "range": "stddev: 0.00003473009090299235",
            "extra": "mean: 162.09918541393415 usec\nrounds: 6548"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6910.185661918285,
            "unit": "iter/sec",
            "range": "stddev: 0.00007220392277263378",
            "extra": "mean: 144.71391203147462 usec\nrounds: 7583"
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
          "id": "a08c1d57731dfd56c6c54c0d084599068dc3d0f3",
          "message": "Add tp.pad",
          "timestamp": "2024-10-09T19:44:26Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/203/commits/a08c1d57731dfd56c6c54c0d084599068dc3d0f3"
        },
        "date": 1728510246600,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6086.786595945146,
            "unit": "iter/sec",
            "range": "stddev: 0.00003158266898115851",
            "extra": "mean: 164.29030067625064 usec\nrounds: 6451"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6699.72787578988,
            "unit": "iter/sec",
            "range": "stddev: 0.00007383972969742347",
            "extra": "mean: 149.25979361245365 usec\nrounds: 7298"
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
          "id": "6d14909ba384471f62afc63e5d769c01634f4825",
          "message": "Add tp.resize",
          "timestamp": "2024-10-09T19:44:26Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/247/commits/6d14909ba384471f62afc63e5d769c01634f4825"
        },
        "date": 1728517368485,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6001.0783179067075,
            "unit": "iter/sec",
            "range": "stddev: 0.000036721583413811784",
            "extra": "mean: 166.6367187737052 usec\nrounds: 6413"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6792.456231986853,
            "unit": "iter/sec",
            "range": "stddev: 0.00007176781986870017",
            "extra": "mean: 147.2221484903836 usec\nrounds: 7431"
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
          "id": "4b7beb695d47a34d176330ff2dfeef19d44dc188",
          "message": "Add tp.resize",
          "timestamp": "2024-10-10T18:22:52Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/247/commits/4b7beb695d47a34d176330ff2dfeef19d44dc188"
        },
        "date": 1728585071425,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6054.353959795353,
            "unit": "iter/sec",
            "range": "stddev: 0.000032101822324042623",
            "extra": "mean: 165.17038921751472 usec\nrounds: 6484"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6890.676404992283,
            "unit": "iter/sec",
            "range": "stddev: 0.00007237946638184328",
            "extra": "mean: 145.1236339113968 usec\nrounds: 7483"
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
          "id": "f9bf8da4e570869370cc1c6c8bed46c5696766ad",
          "message": "Add tp.resize",
          "timestamp": "2024-10-10T18:22:52Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/247/commits/f9bf8da4e570869370cc1c6c8bed46c5696766ad"
        },
        "date": 1728585651803,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6099.666125475276,
            "unit": "iter/sec",
            "range": "stddev: 0.00003424217082131857",
            "extra": "mean: 163.94339943025676 usec\nrounds: 6483"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6901.0874384667695,
            "unit": "iter/sec",
            "range": "stddev: 0.00007083938939371836",
            "extra": "mean: 144.90469928347585 usec\nrounds: 7508"
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
          "id": "5b16caa66a6f28d944e1115f9cdadd2cfce49dfe",
          "message": "Updates exception throwing logic to correctly exclude decorators",
          "timestamp": "2024-10-11T05:06:30Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/268/commits/5b16caa66a6f28d944e1115f9cdadd2cfce49dfe"
        },
        "date": 1728934960469,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6148.854756278693,
            "unit": "iter/sec",
            "range": "stddev: 0.00003170542564436263",
            "extra": "mean: 162.63191108537475 usec\nrounds: 6508"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6782.720735791541,
            "unit": "iter/sec",
            "range": "stddev: 0.00006933956267390598",
            "extra": "mean: 147.43346202109268 usec\nrounds: 7489"
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
          "id": "af99d76a393b7865095c358114288d84b079eea3",
          "message": "Updates exception throwing logic to correctly exclude decorators",
          "timestamp": "2024-10-11T05:06:30Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/268/commits/af99d76a393b7865095c358114288d84b079eea3"
        },
        "date": 1728943146297,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6025.616340255541,
            "unit": "iter/sec",
            "range": "stddev: 0.000034485055611267415",
            "extra": "mean: 165.9581266930763 usec\nrounds: 6446"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6773.176106661651,
            "unit": "iter/sec",
            "range": "stddev: 0.00007052055895121178",
            "extra": "mean: 147.6412224121067 usec\nrounds: 7394"
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
          "id": "1c1a13f14a1353e3cfcb2cb3b0ab98f6ff2cdb6f",
          "message": "Add checks for non-canonical strides",
          "timestamp": "2024-10-15T07:03:52Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/273/commits/1c1a13f14a1353e3cfcb2cb3b0ab98f6ff2cdb6f"
        },
        "date": 1729029345357,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6173.779465920269,
            "unit": "iter/sec",
            "range": "stddev: 0.0000321432830493542",
            "extra": "mean: 161.97533545214498 usec\nrounds: 6545"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6967.933739771023,
            "unit": "iter/sec",
            "range": "stddev: 0.00007165824897296668",
            "extra": "mean: 143.5145679259662 usec\nrounds: 7585"
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
          "id": "c9c2fc226711f26da3f482bacb134c4daf7eaca8",
          "message": "Adds dtype constraint enforcement, various bug fixes and improvements",
          "timestamp": "2024-10-15T22:05:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/274/commits/c9c2fc226711f26da3f482bacb134c4daf7eaca8"
        },
        "date": 1729031107813,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6226.324438759435,
            "unit": "iter/sec",
            "range": "stddev: 0.00003160990012362032",
            "extra": "mean: 160.60839903794752 usec\nrounds: 6582"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6858.967925109602,
            "unit": "iter/sec",
            "range": "stddev: 0.00007663718318583941",
            "extra": "mean: 145.79452928175348 usec\nrounds: 7568"
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
          "id": "6b839c2f90420d098afa9040effa5fe4f3eea998",
          "message": "Updates the compiler to emit errors if evaluating while tracing",
          "timestamp": "2024-10-15T22:05:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/275/commits/6b839c2f90420d098afa9040effa5fe4f3eea998"
        },
        "date": 1729034708291,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6110.391835779459,
            "unit": "iter/sec",
            "range": "stddev: 0.00003390249076856899",
            "extra": "mean: 163.6556258380175 usec\nrounds: 6507"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6798.101122680909,
            "unit": "iter/sec",
            "range": "stddev: 0.00006929137231129753",
            "extra": "mean: 147.09990068603724 usec\nrounds: 7446"
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
          "id": "4d6646048fc1d6aceabd6a2576d94f3733cc743b",
          "message": "Updates exception throwing logic to correctly exclude decorators",
          "timestamp": "2024-10-15T22:05:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/268/commits/4d6646048fc1d6aceabd6a2576d94f3733cc743b"
        },
        "date": 1729035427346,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6050.194891831476,
            "unit": "iter/sec",
            "range": "stddev: 0.00003213076691904024",
            "extra": "mean: 165.28393182013454 usec\nrounds: 6412"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6844.208170300476,
            "unit": "iter/sec",
            "range": "stddev: 0.00007189429946369173",
            "extra": "mean: 146.10893986821821 usec\nrounds: 7391"
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
          "id": "38b6e65791e4e70567c5ff5e49515df0c85b8299",
          "message": "[Tripy] Disallow passing both Shapes and Tensors with the `convert_inputs_to_tensors decorator`",
          "timestamp": "2024-10-15T22:05:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/277/commits/38b6e65791e4e70567c5ff5e49515df0c85b8299"
        },
        "date": 1729108673026,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6054.6223563069425,
            "unit": "iter/sec",
            "range": "stddev: 0.00003192532015890508",
            "extra": "mean: 165.1630673477638 usec\nrounds: 6389"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6682.843523577972,
            "unit": "iter/sec",
            "range": "stddev: 0.00007068040167462425",
            "extra": "mean: 149.6369017876695 usec\nrounds: 7310"
          }
        ]
      }
    ]
  }
}