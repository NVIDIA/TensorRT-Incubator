window.BENCHMARK_DATA = {
  "lastUpdate": 1736806683660,
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
          "id": "76bc5c68295e11885e9b13c657216b7bc7aa31ed",
          "message": "[Tripy] Rename `infer_shape_output_idxs` to `infer_tensor_variants`",
          "timestamp": "2024-10-15T22:05:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/278/commits/76bc5c68295e11885e9b13c657216b7bc7aa31ed"
        },
        "date": 1729109193031,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6046.71921295299,
            "unit": "iter/sec",
            "range": "stddev: 0.000032044978902927075",
            "extra": "mean: 165.3789376986198 usec\nrounds: 6472"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6659.6316166235665,
            "unit": "iter/sec",
            "range": "stddev: 0.00006976907146034105",
            "extra": "mean: 150.1584558376819 usec\nrounds: 7272"
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
          "id": "29cf0b5abe9cf13cdac7a126c830503f828892cd",
          "message": "[Tripy] Disallow passing both Shapes and Tensors with the `convert_inputs_to_tensors decorator`",
          "timestamp": "2024-10-15T22:05:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/277/commits/29cf0b5abe9cf13cdac7a126c830503f828892cd"
        },
        "date": 1729109435083,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6113.164165148713,
            "unit": "iter/sec",
            "range": "stddev: 0.00003163239819681569",
            "extra": "mean: 163.58140775950736 usec\nrounds: 6513"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6792.879643596127,
            "unit": "iter/sec",
            "range": "stddev: 0.00006918099261127552",
            "extra": "mean: 147.21297188633886 usec\nrounds: 7455"
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
          "id": "fc6e225dcc40f3f13b1d2ed1d9501f664f615d77",
          "message": "[Tripy] Disallow passing both Shapes and Tensors with the `convert_inputs_to_tensors decorator`",
          "timestamp": "2024-10-15T22:05:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/277/commits/fc6e225dcc40f3f13b1d2ed1d9501f664f615d77"
        },
        "date": 1729109674884,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6059.545879409825,
            "unit": "iter/sec",
            "range": "stddev: 0.00003151448089558046",
            "extra": "mean: 165.02886848302828 usec\nrounds: 6428"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6663.409965673202,
            "unit": "iter/sec",
            "range": "stddev: 0.00007232546937192076",
            "extra": "mean: 150.0733115854399 usec\nrounds: 7427"
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
          "id": "873986a0cb609992f6fbb031b5f23b5c1ba2c975",
          "message": "[Tripy] Rename `infer_shape_output_idxs` to `infer_tensor_variants`",
          "timestamp": "2024-10-15T22:05:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/278/commits/873986a0cb609992f6fbb031b5f23b5c1ba2c975"
        },
        "date": 1729109916373,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6116.591793932413,
            "unit": "iter/sec",
            "range": "stddev: 0.0000348394187680136",
            "extra": "mean: 163.48973966057181 usec\nrounds: 6512"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6979.635393588755,
            "unit": "iter/sec",
            "range": "stddev: 0.00007112252783823312",
            "extra": "mean: 143.2739596854249 usec\nrounds: 7574"
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
          "id": "9ff5b9923c335412105d2828ac95a255ea5f66ca",
          "message": "[Tripy] Rename `infer_shape_output_idxs` to `infer_tensor_variants`",
          "timestamp": "2024-10-15T22:05:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/278/commits/9ff5b9923c335412105d2828ac95a255ea5f66ca"
        },
        "date": 1729112602594,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6083.249365006933,
            "unit": "iter/sec",
            "range": "stddev: 0.00003517970727879972",
            "extra": "mean: 164.38583066352078 usec\nrounds: 6496"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6954.500732767296,
            "unit": "iter/sec",
            "range": "stddev: 0.00007055140515075352",
            "extra": "mean: 143.7917743380675 usec\nrounds: 7488"
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
          "id": "6c258a908b7b73de2e2bfedb31afa56883993489",
          "message": "Updates exception throwing logic to correctly exclude decorators",
          "timestamp": "2024-10-15T22:05:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/268/commits/6c258a908b7b73de2e2bfedb31afa56883993489"
        },
        "date": 1729113673419,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6098.249681901309,
            "unit": "iter/sec",
            "range": "stddev: 0.00003426582879907012",
            "extra": "mean: 163.981478647529 usec\nrounds: 6493"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6710.157881013954,
            "unit": "iter/sec",
            "range": "stddev: 0.00006988449930710818",
            "extra": "mean: 149.02779006578197 usec\nrounds: 7368"
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
          "id": "834b30e02510162b38bd7ea3bf5f49b112572e89",
          "message": "Fix unnecessary calls to compute shape of trace tensor; fix #246",
          "timestamp": "2024-10-17T13:27:42Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/280/commits/834b30e02510162b38bd7ea3bf5f49b112572e89"
        },
        "date": 1729197066227,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6101.7061007035845,
            "unit": "iter/sec",
            "range": "stddev: 0.00003180892578688663",
            "extra": "mean: 163.88858845310338 usec\nrounds: 6458"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6800.176819967019,
            "unit": "iter/sec",
            "range": "stddev: 0.00006924089138300179",
            "extra": "mean: 147.05499966761894 usec\nrounds: 7420"
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
          "id": "574090c1ee277ec2a8f03cf0b63f70a5618986aa",
          "message": "Updates the compiler to emit errors if evaluating while tracing",
          "timestamp": "2024-10-17T20:32:19Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/275/commits/574090c1ee277ec2a8f03cf0b63f70a5618986aa"
        },
        "date": 1729200164718,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6114.105373577508,
            "unit": "iter/sec",
            "range": "stddev: 0.000033947751687278664",
            "extra": "mean: 163.55622595605942 usec\nrounds: 6500"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6611.587419035638,
            "unit": "iter/sec",
            "range": "stddev: 0.00006991283321808812",
            "extra": "mean: 151.24960718523772 usec\nrounds: 7320"
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
          "id": "fe1bbb9465a71234a95dcff16c96e5d0fb590a95",
          "message": "Updates the compiler to emit errors if evaluating while tracing",
          "timestamp": "2024-10-17T20:32:19Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/275/commits/fe1bbb9465a71234a95dcff16c96e5d0fb590a95"
        },
        "date": 1729200406970,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6172.989994249732,
            "unit": "iter/sec",
            "range": "stddev: 0.000031052125013631276",
            "extra": "mean: 161.9960506871906 usec\nrounds: 6533"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6982.932653039478,
            "unit": "iter/sec",
            "range": "stddev: 0.00007244482127713629",
            "extra": "mean: 143.20630739073897 usec\nrounds: 7620"
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
          "id": "c893d1799b2d7b208efe70b282017e7b02cf32e6",
          "message": "Force assign output shape of reshape operator if its known statically",
          "timestamp": "2024-10-17T20:32:19Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/281/commits/c893d1799b2d7b208efe70b282017e7b02cf32e6"
        },
        "date": 1729202858563,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6158.32081495556,
            "unit": "iter/sec",
            "range": "stddev: 0.00003122764612753409",
            "extra": "mean: 162.38192683490723 usec\nrounds: 6532"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6936.9664818381825,
            "unit": "iter/sec",
            "range": "stddev: 0.0000727983657527738",
            "extra": "mean: 144.15523018860202 usec\nrounds: 7570"
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
          "id": "b9cb106c794db6cbf77973cff9ec878beab6728d",
          "message": "Updates the compiler to emit errors if evaluating while tracing",
          "timestamp": "2024-10-20T09:51:08Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/275/commits/b9cb106c794db6cbf77973cff9ec878beab6728d"
        },
        "date": 1729532963212,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6169.74180308348,
            "unit": "iter/sec",
            "range": "stddev: 0.000031428322921612056",
            "extra": "mean: 162.0813369370215 usec\nrounds: 6516"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6903.345430701209,
            "unit": "iter/sec",
            "range": "stddev: 0.00007187132916196603",
            "extra": "mean: 144.85730288864087 usec\nrounds: 7514"
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
          "id": "8966b2bc71afd2b842aba5413d178c2a9ede0704",
          "message": "Updates the compiler to emit errors if evaluating while tracing",
          "timestamp": "2024-10-20T09:51:08Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/275/commits/8966b2bc71afd2b842aba5413d178c2a9ede0704"
        },
        "date": 1729533359644,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6063.225483810628,
            "unit": "iter/sec",
            "range": "stddev: 0.00003423211167138175",
            "extra": "mean: 164.92871701210726 usec\nrounds: 6441"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6681.919622997409,
            "unit": "iter/sec",
            "range": "stddev: 0.0000708669322535347",
            "extra": "mean: 149.65759189294394 usec\nrounds: 7321"
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
          "id": "1c9d4560c65ad4ac2a2bf60b956c3d16176c3848",
          "message": "Avoids passing stream when creating host memrefs to not trip an assertion",
          "timestamp": "2024-10-20T09:51:08Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/292/commits/1c9d4560c65ad4ac2a2bf60b956c3d16176c3848"
        },
        "date": 1729533594469,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5980.503265325588,
            "unit": "iter/sec",
            "range": "stddev: 0.000038081308427477585",
            "extra": "mean: 167.21000819411117 usec\nrounds: 6415"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6706.668022220729,
            "unit": "iter/sec",
            "range": "stddev: 0.00007409738131551353",
            "extra": "mean: 149.105337656012 usec\nrounds: 7353"
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
          "id": "0f21d30ae7f5a075448e0acdfaba3125ee64c8d7",
          "message": "Avoids passing stream when creating host memrefs to not trip an assertion",
          "timestamp": "2024-10-20T09:51:08Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/292/commits/0f21d30ae7f5a075448e0acdfaba3125ee64c8d7"
        },
        "date": 1729533870780,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6075.822134220327,
            "unit": "iter/sec",
            "range": "stddev: 0.00003731710245966861",
            "extra": "mean: 164.58677984791336 usec\nrounds: 6489"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 7018.408379005334,
            "unit": "iter/sec",
            "range": "stddev: 0.00007255684025516411",
            "extra": "mean: 142.48244701624537 usec\nrounds: 7679"
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
          "id": "cf04956a4ab6b57c57a348bccff61c8e79b17ce3",
          "message": "Avoids passing stream when creating host memrefs to not trip an assertion",
          "timestamp": "2024-10-20T09:51:08Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/292/commits/cf04956a4ab6b57c57a348bccff61c8e79b17ce3"
        },
        "date": 1729534105503,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6036.43706047739,
            "unit": "iter/sec",
            "range": "stddev: 0.000033930776818550067",
            "extra": "mean: 165.66063556718592 usec\nrounds: 6419"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6666.782781158063,
            "unit": "iter/sec",
            "range": "stddev: 0.00006687091540419909",
            "extra": "mean: 149.9973874694465 usec\nrounds: 7219"
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
          "id": "dec091c3d13e05fcc49e4d1e542ca937c76cf316",
          "message": "Avoids passing stream when creating host memrefs to not trip an assertion",
          "timestamp": "2024-10-21T18:03:22Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/292/commits/dec091c3d13e05fcc49e4d1e542ca937c76cf316"
        },
        "date": 1729534966552,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6134.744754396316,
            "unit": "iter/sec",
            "range": "stddev: 0.00003475698462448287",
            "extra": "mean: 163.00596683886062 usec\nrounds: 6520"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6657.745745713886,
            "unit": "iter/sec",
            "range": "stddev: 0.0000713336083578554",
            "extra": "mean: 150.20098967338586 usec\nrounds: 7392"
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
          "id": "a13b1672a29a61e19224f74ceac104cc38240759",
          "message": "[Tripy] Rename `infer_shape_output_idxs` to `infer_tensor_variants`",
          "timestamp": "2024-10-22T21:28:13Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/278/commits/a13b1672a29a61e19224f74ceac104cc38240759"
        },
        "date": 1729635526568,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6130.417052039198,
            "unit": "iter/sec",
            "range": "stddev: 0.00003433367850125287",
            "extra": "mean: 163.12103915790917 usec\nrounds: 6515"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6799.747843774323,
            "unit": "iter/sec",
            "range": "stddev: 0.000069383877552323",
            "extra": "mean: 147.0642769372065 usec\nrounds: 7458"
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
          "id": "c655d8c0bebe39715bec92d4e746c8f14361fa2e",
          "message": "Adds dtype constraint enforcement, various bug fixes and improvements",
          "timestamp": "2024-10-23T08:26:56Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/274/commits/c655d8c0bebe39715bec92d4e746c8f14361fa2e"
        },
        "date": 1729703607178,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6067.352628948021,
            "unit": "iter/sec",
            "range": "stddev: 0.00003572238402468232",
            "extra": "mean: 164.81652891392656 usec\nrounds: 6555"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6706.227894656573,
            "unit": "iter/sec",
            "range": "stddev: 0.0000690178613888604",
            "extra": "mean: 149.11512339101773 usec\nrounds: 7356"
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
          "id": "29ce768a9da763327de23f526ab96f3ab07027a0",
          "message": "Add ndim; support start_dim negative",
          "timestamp": "2024-10-24T16:58:29Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/299/commits/29ce768a9da763327de23f526ab96f3ab07027a0"
        },
        "date": 1729794911561,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6005.652957150817,
            "unit": "iter/sec",
            "range": "stddev: 0.00003366805436196966",
            "extra": "mean: 166.50978788398336 usec\nrounds: 6358"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6584.896570866722,
            "unit": "iter/sec",
            "range": "stddev: 0.00006777422025125233",
            "extra": "mean: 151.86267380785563 usec\nrounds: 7205"
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
          "id": "585a5ea6d3f82e7446da27c787bdb756560cfef1",
          "message": "Replaces `convert_inputs_to_tensors` decorator with a better decorator",
          "timestamp": "2024-10-24T20:57:14Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/301/commits/585a5ea6d3f82e7446da27c787bdb756560cfef1"
        },
        "date": 1729811733658,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6009.889472587219,
            "unit": "iter/sec",
            "range": "stddev: 0.00003475092155276744",
            "extra": "mean: 166.39241113522615 usec\nrounds: 6356"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6588.652798437231,
            "unit": "iter/sec",
            "range": "stddev: 0.00006740025452351762",
            "extra": "mean: 151.7760960536866 usec\nrounds: 7111"
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
          "id": "d84fff177ba1ff7edd865d7f87d7749fbdacc9fa",
          "message": "Replaces `convert_inputs_to_tensors` decorator with a better decorator",
          "timestamp": "2024-10-24T20:57:14Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/301/commits/d84fff177ba1ff7edd865d7f87d7749fbdacc9fa"
        },
        "date": 1729812449352,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6149.32166027973,
            "unit": "iter/sec",
            "range": "stddev: 0.00003408559289217265",
            "extra": "mean: 162.61956281443088 usec\nrounds: 6486"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6911.762097920046,
            "unit": "iter/sec",
            "range": "stddev: 0.00007144580721588068",
            "extra": "mean: 144.68090565514828 usec\nrounds: 7459"
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
          "id": "8c51976a32b8e4d6aa55740c4c0aed345c9f27b5",
          "message": "Fix compile when no args are provided",
          "timestamp": "2024-10-25T17:02:34Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/305/commits/8c51976a32b8e4d6aa55740c4c0aed345c9f27b5"
        },
        "date": 1729882861856,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6088.890351108544,
            "unit": "iter/sec",
            "range": "stddev: 0.00003432757494799273",
            "extra": "mean: 164.23353720238038 usec\nrounds: 6484"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6775.75368161733,
            "unit": "iter/sec",
            "range": "stddev: 0.00006924790317610594",
            "extra": "mean: 147.58505798594885 usec\nrounds: 7387"
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
          "id": "9239fa98a980cd6fa7b542e989fb84c9f233a45f",
          "message": "Fix compile when no args are provided",
          "timestamp": "2024-10-25T17:02:34Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/305/commits/9239fa98a980cd6fa7b542e989fb84c9f233a45f"
        },
        "date": 1729887083527,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6066.911187035078,
            "unit": "iter/sec",
            "range": "stddev: 0.00003357734202438837",
            "extra": "mean: 164.82852133009445 usec\nrounds: 6448"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6864.280617337224,
            "unit": "iter/sec",
            "range": "stddev: 0.00006987314044461517",
            "extra": "mean: 145.68168985899032 usec\nrounds: 7481"
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
          "id": "80cdc2653cebad1e7ffd36e06ae0c6fc9f17a67b",
          "message": "Fix build mlir tensorrt noexcept",
          "timestamp": "2024-10-25T20:11:53Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/304/commits/80cdc2653cebad1e7ffd36e06ae0c6fc9f17a67b"
        },
        "date": 1729894032158,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6003.056131880063,
            "unit": "iter/sec",
            "range": "stddev: 0.000035002775089291",
            "extra": "mean: 166.58181733290166 usec\nrounds: 6408"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6651.044207157274,
            "unit": "iter/sec",
            "range": "stddev: 0.00007082608420665363",
            "extra": "mean: 150.35233098043267 usec\nrounds: 7264"
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
          "id": "b934df3e0069ed35ae6fd412d2226ee678266150",
          "message": "Shape refactor",
          "timestamp": "2024-10-26T01:41:06Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/306/commits/b934df3e0069ed35ae6fd412d2226ee678266150"
        },
        "date": 1730136504993,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6083.154270256986,
            "unit": "iter/sec",
            "range": "stddev: 0.000034508893509389966",
            "extra": "mean: 164.38840042071703 usec\nrounds: 6468"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6786.457137423555,
            "unit": "iter/sec",
            "range": "stddev: 0.00006996049643864201",
            "extra": "mean: 147.3522899725622 usec\nrounds: 7383"
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
          "id": "b005b75f26b1980bb2cb0cdb8cfde0e881aac1b8",
          "message": "Removes `Shape` tensors",
          "timestamp": "2024-10-26T01:41:06Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/306/commits/b005b75f26b1980bb2cb0cdb8cfde0e881aac1b8"
        },
        "date": 1730147275923,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6114.466674018444,
            "unit": "iter/sec",
            "range": "stddev: 0.00003425146925200358",
            "extra": "mean: 163.54656150947625 usec\nrounds: 6501"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6780.218214955523,
            "unit": "iter/sec",
            "range": "stddev: 0.00007465256872594293",
            "extra": "mean: 147.48787845710356 usec\nrounds: 7378"
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
          "id": "89140e32cbfba34053b1b95685c973ac8ba18461",
          "message": "Removes `Shape` tensors",
          "timestamp": "2024-10-29T18:44:25Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/306/commits/89140e32cbfba34053b1b95685c973ac8ba18461"
        },
        "date": 1730233619172,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5998.5272154476015,
            "unit": "iter/sec",
            "range": "stddev: 0.0000335929668868222",
            "extra": "mean: 166.70758739324674 usec\nrounds: 6360"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6482.688554105842,
            "unit": "iter/sec",
            "range": "stddev: 0.00006836264805337887",
            "extra": "mean: 154.25698638054195 usec\nrounds: 7138"
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
          "id": "99bd0cc41cccaf7ea66dfd3c316b57e3e0afee95",
          "message": "[tripy] Batchnorm implementation from feature branch",
          "timestamp": "2024-10-29T18:44:25Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/319/commits/99bd0cc41cccaf7ea66dfd3c316b57e3e0afee95"
        },
        "date": 1730234937232,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6123.229350002889,
            "unit": "iter/sec",
            "range": "stddev: 0.00003326099591080592",
            "extra": "mean: 163.31251743812734 usec\nrounds: 6516"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6752.7684943717395,
            "unit": "iter/sec",
            "range": "stddev: 0.00007092974515426243",
            "extra": "mean: 148.08741049444748 usec\nrounds: 7398"
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
          "id": "7c6cc5d2c1cbf8342db9396a99d44045deea5420",
          "message": "[tripy] Sequential module feature branch",
          "timestamp": "2024-10-29T18:44:25Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/321/commits/7c6cc5d2c1cbf8342db9396a99d44045deea5420"
        },
        "date": 1730238677590,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6081.735237152243,
            "unit": "iter/sec",
            "range": "stddev: 0.0000340284991052951",
            "extra": "mean: 164.4267566748347 usec\nrounds: 6505"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6783.7396075613315,
            "unit": "iter/sec",
            "range": "stddev: 0.00007030557703660041",
            "extra": "mean: 147.4113185130771 usec\nrounds: 7469"
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
          "id": "27489c3dd16052849a7805cb9e8ef0277d0ff15b",
          "message": "[tripy] Batchnorm implementation from feature branch",
          "timestamp": "2024-10-29T21:48:51Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/319/commits/27489c3dd16052849a7805cb9e8ef0277d0ff15b"
        },
        "date": 1730238911682,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6036.873842976998,
            "unit": "iter/sec",
            "range": "stddev: 0.000034667514555350855",
            "extra": "mean: 165.64864961744243 usec\nrounds: 6446"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6595.366828926385,
            "unit": "iter/sec",
            "range": "stddev: 0.0000720522571317278",
            "extra": "mean: 151.62158920624938 usec\nrounds: 7511"
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
          "id": "0bb60b42992dad161b201d3dec85c81f2f5d7541",
          "message": "[tripy] Sequential module feature branch",
          "timestamp": "2024-10-29T21:48:51Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/321/commits/0bb60b42992dad161b201d3dec85c81f2f5d7541"
        },
        "date": 1730239146162,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6087.577316037465,
            "unit": "iter/sec",
            "range": "stddev: 0.00003430247748283543",
            "extra": "mean: 164.26896088293486 usec\nrounds: 6445"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6688.098906067791,
            "unit": "iter/sec",
            "range": "stddev: 0.00007064485051300751",
            "extra": "mean: 149.51931992105378 usec\nrounds: 7339"
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
          "id": "a5507c5392ade2aed6d80223fb51d0e2311ccd91",
          "message": "[tripy] Batchnorm implementation from feature branch",
          "timestamp": "2024-10-29T21:48:51Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/319/commits/a5507c5392ade2aed6d80223fb51d0e2311ccd91"
        },
        "date": 1730312764649,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6143.708521145655,
            "unit": "iter/sec",
            "range": "stddev: 0.00003312402591643835",
            "extra": "mean: 162.76813858570293 usec\nrounds: 6526"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6877.292618972406,
            "unit": "iter/sec",
            "range": "stddev: 0.00006956880100671279",
            "extra": "mean: 145.40605662776326 usec\nrounds: 7446"
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
          "id": "6a3181f47966a5518bb00094c38f0d76804dca4d",
          "message": "Removes `Shape` tensors",
          "timestamp": "2024-10-29T21:48:51Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/306/commits/6a3181f47966a5518bb00094c38f0d76804dca4d"
        },
        "date": 1730312988862,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6016.49768217544,
            "unit": "iter/sec",
            "range": "stddev: 0.000033601391028947475",
            "extra": "mean: 166.20965432474347 usec\nrounds: 6418"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6704.959948974344,
            "unit": "iter/sec",
            "range": "stddev: 0.0000737484250074903",
            "extra": "mean: 149.14332190052374 usec\nrounds: 7335"
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
          "id": "3938de2ede8e2bbf3998c28d7ba12e76395586fe",
          "message": "[tripy] Batchnorm implementation from feature branch",
          "timestamp": "2024-10-29T21:48:51Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/319/commits/3938de2ede8e2bbf3998c28d7ba12e76395586fe"
        },
        "date": 1730313789860,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6152.57851026835,
            "unit": "iter/sec",
            "range": "stddev: 0.00003322952809141614",
            "extra": "mean: 162.53348060996692 usec\nrounds: 6513"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6886.076640357745,
            "unit": "iter/sec",
            "range": "stddev: 0.00007255270236706523",
            "extra": "mean: 145.2205736629803 usec\nrounds: 7540"
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
          "id": "5d5b49262753db81d325efaefa3213c61a4878df",
          "message": "[tripy] Batchnorm implementation from feature branch",
          "timestamp": "2024-10-29T21:48:51Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/319/commits/5d5b49262753db81d325efaefa3213c61a4878df"
        },
        "date": 1730314025002,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6154.161381908674,
            "unit": "iter/sec",
            "range": "stddev: 0.00003353831993830127",
            "extra": "mean: 162.491676435345 usec\nrounds: 6493"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6566.133225669398,
            "unit": "iter/sec",
            "range": "stddev: 0.00007047378983735142",
            "extra": "mean: 152.29663572628058 usec\nrounds: 7213"
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
          "id": "89833a1a1e46e853f5c88dd5f16ea5b5402d3c33",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-10-30T18:47:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/89833a1a1e46e853f5c88dd5f16ea5b5402d3c33"
        },
        "date": 1730314392615,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6072.439610954923,
            "unit": "iter/sec",
            "range": "stddev: 0.000033913326260747326",
            "extra": "mean: 164.67845941126532 usec\nrounds: 6470"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6708.794229947114,
            "unit": "iter/sec",
            "range": "stddev: 0.0000697922204592727",
            "extra": "mean: 149.05808193313496 usec\nrounds: 7353"
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
          "id": "1e752ad96088568ff2102f589eb1ff2d2d9bd402",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-10-30T18:47:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/1e752ad96088568ff2102f589eb1ff2d2d9bd402"
        },
        "date": 1730314629320,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6140.206866665535,
            "unit": "iter/sec",
            "range": "stddev: 0.00002668323375408667",
            "extra": "mean: 162.86096245859127 usec\nrounds: 6486"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6842.741453379775,
            "unit": "iter/sec",
            "range": "stddev: 0.00007725567918864919",
            "extra": "mean: 146.14025779186483 usec\nrounds: 7513"
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
          "id": "84bc58378ff28a1c498c67a32e72f3381033c27f",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-10-30T18:47:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/84bc58378ff28a1c498c67a32e72f3381033c27f"
        },
        "date": 1730314935389,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6099.673933262066,
            "unit": "iter/sec",
            "range": "stddev: 0.00003388539873312206",
            "extra": "mean: 163.94318957721833 usec\nrounds: 6430"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6528.388750030106,
            "unit": "iter/sec",
            "range": "stddev: 0.00007095249256694502",
            "extra": "mean: 153.17715263132706 usec\nrounds: 7247"
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
          "id": "d01fc1ca9952d446f7b7871349dbae65bdba0415",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-10-30T18:47:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/d01fc1ca9952d446f7b7871349dbae65bdba0415"
        },
        "date": 1730315170881,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5994.02955831263,
            "unit": "iter/sec",
            "range": "stddev: 0.000037640395628274924",
            "extra": "mean: 166.83267746205584 usec\nrounds: 6363"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6593.81683075319,
            "unit": "iter/sec",
            "range": "stddev: 0.00006859309692977086",
            "extra": "mean: 151.6572306552491 usec\nrounds: 7221"
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
          "id": "ea28a883becd93d7f3735e95c534348d509c6d6b",
          "message": "Removes `Shape` tensors",
          "timestamp": "2024-10-30T22:04:27Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/306/commits/ea28a883becd93d7f3735e95c534348d509c6d6b"
        },
        "date": 1730401934629,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6090.73651239144,
            "unit": "iter/sec",
            "range": "stddev: 0.000033232833079497323",
            "extra": "mean: 164.18375642510998 usec\nrounds: 6499"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6828.73499184416,
            "unit": "iter/sec",
            "range": "stddev: 0.00006987335871361767",
            "extra": "mean: 146.44000699900369 usec\nrounds: 7425"
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
          "id": "121a6f1e3bfefb5266a149fe02d8cb5d3e9eaff1",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-10-31T19:27:25Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/121a6f1e3bfefb5266a149fe02d8cb5d3e9eaff1"
        },
        "date": 1730403563477,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6127.973266589262,
            "unit": "iter/sec",
            "range": "stddev: 0.00003416004415567859",
            "extra": "mean: 163.18609048968406 usec\nrounds: 6542"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6931.835883342362,
            "unit": "iter/sec",
            "range": "stddev: 0.00007343104303885312",
            "extra": "mean: 144.2619266856942 usec\nrounds: 7611"
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
          "id": "bf91c7beb476960197994e518e829a973ea87acb",
          "message": "Removes useless tests",
          "timestamp": "2024-10-31T19:27:25Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/325/commits/bf91c7beb476960197994e518e829a973ea87acb"
        },
        "date": 1730407407368,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6016.749243879364,
            "unit": "iter/sec",
            "range": "stddev: 0.000034475554030813114",
            "extra": "mean: 166.20270505992357 usec\nrounds: 6408"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6591.309034878399,
            "unit": "iter/sec",
            "range": "stddev: 0.00007278371488846627",
            "extra": "mean: 151.71493169390573 usec\nrounds: 7248"
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
          "id": "bf91c7beb476960197994e518e829a973ea87acb",
          "message": "Removes useless tests",
          "timestamp": "2024-10-31T19:27:25Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/325/commits/bf91c7beb476960197994e518e829a973ea87acb"
        },
        "date": 1730407632788,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6092.280571396463,
            "unit": "iter/sec",
            "range": "stddev: 0.000033642480850409445",
            "extra": "mean: 164.14214484720975 usec\nrounds: 6503"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6879.957085348208,
            "unit": "iter/sec",
            "range": "stddev: 0.0000729114145112315",
            "extra": "mean: 145.34974384209957 usec\nrounds: 7527"
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
          "id": "7b276439b8838c63dc93eb85fb6253cccb842db6",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-10-31T19:27:25Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/7b276439b8838c63dc93eb85fb6253cccb842db6"
        },
        "date": 1730408404492,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6105.9708732766985,
            "unit": "iter/sec",
            "range": "stddev: 0.000032641133479440216",
            "extra": "mean: 163.77411893276224 usec\nrounds: 6440"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6866.259644884378,
            "unit": "iter/sec",
            "range": "stddev: 0.00007278273905792622",
            "extra": "mean: 145.63970075687973 usec\nrounds: 7549"
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
          "id": "42e1c6f5663dcc0cf0a6e0708961221d1602a7b0",
          "message": "Removes useless tests",
          "timestamp": "2024-10-31T19:27:25Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/325/commits/42e1c6f5663dcc0cf0a6e0708961221d1602a7b0"
        },
        "date": 1730410542581,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6017.746670468524,
            "unit": "iter/sec",
            "range": "stddev: 0.00003496931666830955",
            "extra": "mean: 166.17515737367236 usec\nrounds: 6405"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6853.049848319158,
            "unit": "iter/sec",
            "range": "stddev: 0.00007197429333379573",
            "extra": "mean: 145.92043281945033 usec\nrounds: 7368"
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
          "id": "ec9d021781169b2e5e97770e571897b1a01e04c0",
          "message": "Makes rank inference happen entirely in the frontend",
          "timestamp": "2024-10-31T21:39:12Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/328/commits/ec9d021781169b2e5e97770e571897b1a01e04c0"
        },
        "date": 1730414341826,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6034.109461279974,
            "unit": "iter/sec",
            "range": "stddev: 0.00003231732023095559",
            "extra": "mean: 165.72453755054633 usec\nrounds: 6364"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6783.052008479547,
            "unit": "iter/sec",
            "range": "stddev: 0.00006996892539035734",
            "extra": "mean: 147.42626162233344 usec\nrounds: 7301"
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
          "id": "8a23edcdf11e3cd037dbc2259626108f6cebbda3",
          "message": "Makes rank inference happen entirely in the frontend",
          "timestamp": "2024-10-31T21:39:12Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/328/commits/8a23edcdf11e3cd037dbc2259626108f6cebbda3"
        },
        "date": 1730414532450,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6094.851172798653,
            "unit": "iter/sec",
            "range": "stddev: 0.000033315153840755234",
            "extra": "mean: 164.07291526050778 usec\nrounds: 6468"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6712.593824108619,
            "unit": "iter/sec",
            "range": "stddev: 0.00006982765455477599",
            "extra": "mean: 148.97370915076817 usec\nrounds: 7409"
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
          "id": "34da357deb1f36f64b65ab126c42af18fec6a320",
          "message": "Makes rank inference happen entirely in the frontend",
          "timestamp": "2024-10-31T21:39:12Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/328/commits/34da357deb1f36f64b65ab126c42af18fec6a320"
        },
        "date": 1730414868507,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6145.393683679024,
            "unit": "iter/sec",
            "range": "stddev: 0.000033437415856448966",
            "extra": "mean: 162.72350503041105 usec\nrounds: 6473"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6747.788216396663,
            "unit": "iter/sec",
            "range": "stddev: 0.00006892448645613479",
            "extra": "mean: 148.1967080072354 usec\nrounds: 7404"
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
          "id": "1057dda579c51cf8b34a028cf9c9d89a98993edd",
          "message": "[Tripy] Handle variadic arguments in `preprocess_args` for the `convert_to_tensors` decorator",
          "timestamp": "2024-10-31T21:39:12Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/329/commits/1057dda579c51cf8b34a028cf9c9d89a98993edd"
        },
        "date": 1730415346887,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6038.786158312559,
            "unit": "iter/sec",
            "range": "stddev: 0.000027985989843047016",
            "extra": "mean: 165.5961933050853 usec\nrounds: 6399"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6525.213284832084,
            "unit": "iter/sec",
            "range": "stddev: 0.00007800682324979184",
            "extra": "mean: 153.25169559200597 usec\nrounds: 7148"
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
          "id": "1cdc30fa00251ee44a94705ca299dd58f4fba58c",
          "message": "Makes rank inference happen entirely in the frontend",
          "timestamp": "2024-11-01T10:27:12Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/328/commits/1cdc30fa00251ee44a94705ca299dd58f4fba58c"
        },
        "date": 1730482732396,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6044.894174952387,
            "unit": "iter/sec",
            "range": "stddev: 0.0000337063388387711",
            "extra": "mean: 165.42886791030986 usec\nrounds: 6413"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6709.508790118361,
            "unit": "iter/sec",
            "range": "stddev: 0.00007433135715827687",
            "extra": "mean: 149.04220730327998 usec\nrounds: 7283"
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
          "id": "257bc79a4c1be8f3cb11b38bc30c5e398db65288",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-01T10:27:12Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/257bc79a4c1be8f3cb11b38bc30c5e398db65288"
        },
        "date": 1730487763636,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6062.908798838923,
            "unit": "iter/sec",
            "range": "stddev: 0.000033686862280866774",
            "extra": "mean: 164.9373317625205 usec\nrounds: 6476"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6845.303904435496,
            "unit": "iter/sec",
            "range": "stddev: 0.00007005394223460206",
            "extra": "mean: 146.0855520749105 usec\nrounds: 7383"
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
          "id": "7fe01a422183d22bc95fe7376cc36595dd9b188d",
          "message": "[Tripy] Handle variadic arguments in `preprocess_args` for the `convert_to_tensors` decorator",
          "timestamp": "2024-11-01T10:27:12Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/329/commits/7fe01a422183d22bc95fe7376cc36595dd9b188d"
        },
        "date": 1730490016453,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6034.81416812779,
            "unit": "iter/sec",
            "range": "stddev: 0.00003439723501724592",
            "extra": "mean: 165.70518530320126 usec\nrounds: 6428"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6729.472715647485,
            "unit": "iter/sec",
            "range": "stddev: 0.00007254632282134162",
            "extra": "mean: 148.60005267199955 usec\nrounds: 7276"
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
          "id": "fa7c783bb2cf04d113d30009e3873a735cbc66dd",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-01T10:27:12Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/fa7c783bb2cf04d113d30009e3873a735cbc66dd"
        },
        "date": 1730490385098,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6094.188257385046,
            "unit": "iter/sec",
            "range": "stddev: 0.00003316927328022222",
            "extra": "mean: 164.0907628326352 usec\nrounds: 6450"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6724.788428345738,
            "unit": "iter/sec",
            "range": "stddev: 0.00006899443517186175",
            "extra": "mean: 148.70356304220485 usec\nrounds: 7237"
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
          "id": "788c8a3f30bd7cda85c2db653bc2c5c8a85d96ec",
          "message": "[Tripy] Handle variadic arguments in `preprocess_args` for the `convert_to_tensors` decorator",
          "timestamp": "2024-11-01T10:27:12Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/329/commits/788c8a3f30bd7cda85c2db653bc2c5c8a85d96ec"
        },
        "date": 1730492762818,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6108.319761653958,
            "unit": "iter/sec",
            "range": "stddev: 0.000033962642493378186",
            "extra": "mean: 163.7111413645491 usec\nrounds: 6492"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6726.307762706072,
            "unit": "iter/sec",
            "range": "stddev: 0.0000712262905247931",
            "extra": "mean: 148.6699739706362 usec\nrounds: 7402"
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
          "id": "be47039a01a7286ba2d39fa992bad47476f26fb6",
          "message": "[Tripy] Handle variadic arguments in `preprocess_args` for the `convert_to_tensors` decorator",
          "timestamp": "2024-11-01T10:27:12Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/329/commits/be47039a01a7286ba2d39fa992bad47476f26fb6"
        },
        "date": 1730492956629,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6035.661910477973,
            "unit": "iter/sec",
            "range": "stddev: 0.00003520760690793041",
            "extra": "mean: 165.68191108650228 usec\nrounds: 6397"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6845.086533725807,
            "unit": "iter/sec",
            "range": "stddev: 0.00007192017724730891",
            "extra": "mean: 146.09019112804353 usec\nrounds: 7379"
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
          "id": "c9f905d40861ccf25e71bab9d2c7ecafe8fd1ddf",
          "message": "[Tripy] Handle variadic arguments in `preprocess_args` for the `convert_to_tensors` decorator",
          "timestamp": "2024-11-01T20:27:58Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/329/commits/c9f905d40861ccf25e71bab9d2c7ecafe8fd1ddf"
        },
        "date": 1730496964038,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6053.887387818783,
            "unit": "iter/sec",
            "range": "stddev: 0.000032716700458695607",
            "extra": "mean: 165.18311886873406 usec\nrounds: 6381"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6921.642735713468,
            "unit": "iter/sec",
            "range": "stddev: 0.00006768503263437921",
            "extra": "mean: 144.4743738130717 usec\nrounds: 7415"
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
          "id": "31e345226aa704fd8b3f56210ff2840dd2c3d861",
          "message": "[Tripy] Eliminate need for `skip_num_stack_entries` argument in `convert_to_tensors`",
          "timestamp": "2024-11-01T20:27:58Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/333/commits/31e345226aa704fd8b3f56210ff2840dd2c3d861"
        },
        "date": 1730604011826,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6011.2658749014645,
            "unit": "iter/sec",
            "range": "stddev: 0.00003373349167145282",
            "extra": "mean: 166.3543121882613 usec\nrounds: 6471"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6861.509557096872,
            "unit": "iter/sec",
            "range": "stddev: 0.0000690879248908526",
            "extra": "mean: 145.74052425033761 usec\nrounds: 7449"
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
          "id": "be7dcb5cc1c10d935431984d29a3968e78339747",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-01T20:27:58Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/be7dcb5cc1c10d935431984d29a3968e78339747"
        },
        "date": 1730736205822,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6070.28341761545,
            "unit": "iter/sec",
            "range": "stddev: 0.00003338217509737737",
            "extra": "mean: 164.73695397781336 usec\nrounds: 6468"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6783.064329946104,
            "unit": "iter/sec",
            "range": "stddev: 0.00006902135392816443",
            "extra": "mean: 147.42599382187277 usec\nrounds: 7471"
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
          "id": "4dc1022a8b31e335982b2cebb6a5f3605221861f",
          "message": "Miscellaneous fixes",
          "timestamp": "2024-11-01T20:27:58Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/330/commits/4dc1022a8b31e335982b2cebb6a5f3605221861f"
        },
        "date": 1730746377698,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6076.951043286503,
            "unit": "iter/sec",
            "range": "stddev: 0.00003369654524859038",
            "extra": "mean: 164.55620472782115 usec\nrounds: 6427"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6826.988353410189,
            "unit": "iter/sec",
            "range": "stddev: 0.00007748646062376169",
            "extra": "mean: 146.47747267658426 usec\nrounds: 7342"
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
          "id": "02f07f8bcd50af4943e50d63d2ccedc693493d51",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-04T20:47:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/02f07f8bcd50af4943e50d63d2ccedc693493d51"
        },
        "date": 1730760869181,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6144.197509855673,
            "unit": "iter/sec",
            "range": "stddev: 0.00003263694334989696",
            "extra": "mean: 162.75518461051067 usec\nrounds: 6514"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6821.341655548628,
            "unit": "iter/sec",
            "range": "stddev: 0.0000699897569250616",
            "extra": "mean: 146.5987265403395 usec\nrounds: 7460"
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
          "id": "ee7bb974752fe17d9e2b0505fe80be460659429f",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-04T20:47:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/ee7bb974752fe17d9e2b0505fe80be460659429f"
        },
        "date": 1730762465489,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6035.448139491303,
            "unit": "iter/sec",
            "range": "stddev: 0.00003392545309234572",
            "extra": "mean: 165.6877794138887 usec\nrounds: 6441"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6743.495389400325,
            "unit": "iter/sec",
            "range": "stddev: 0.0000713566632603054",
            "extra": "mean: 148.29104822579652 usec\nrounds: 7440"
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
          "id": "ed0d1c5b047a254e35c85c86e7412d9a3b6d2216",
          "message": "[Tripy] Eliminate need for `skip_num_stack_entries` argument in `convert_to_tensors`",
          "timestamp": "2024-11-04T20:47:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/333/commits/ed0d1c5b047a254e35c85c86e7412d9a3b6d2216"
        },
        "date": 1730777210071,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6140.516947137764,
            "unit": "iter/sec",
            "range": "stddev: 0.00003328785595862787",
            "extra": "mean: 162.852738394627 usec\nrounds: 6493"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6955.5970447690515,
            "unit": "iter/sec",
            "range": "stddev: 0.00006974146138571431",
            "extra": "mean: 143.76911048233433 usec\nrounds: 7463"
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
          "id": "d6a4ebbfd59568117bf1c5b72ef76df75c32a8b6",
          "message": "[Tripy] Eliminate need for `skip_num_stack_entries` argument in `convert_to_tensors`",
          "timestamp": "2024-11-04T20:47:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/333/commits/d6a4ebbfd59568117bf1c5b72ef76df75c32a8b6"
        },
        "date": 1730777384900,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6051.139519262163,
            "unit": "iter/sec",
            "range": "stddev: 0.00003389169113984487",
            "extra": "mean: 165.25812978146857 usec\nrounds: 6421"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6698.099115963208,
            "unit": "iter/sec",
            "range": "stddev: 0.00007060409571028444",
            "extra": "mean: 149.29608873908055 usec\nrounds: 7246"
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
          "id": "ad1255a5641060d5723d928a68ab2621c24b0c38",
          "message": "[Tripy] Handle variadic arguments in `preprocess_args` for the `convert_to_tensors` decorator",
          "timestamp": "2024-11-04T20:47:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/329/commits/ad1255a5641060d5723d928a68ab2621c24b0c38"
        },
        "date": 1730777715456,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6123.115406931558,
            "unit": "iter/sec",
            "range": "stddev: 0.00003358309260722583",
            "extra": "mean: 163.31555646786745 usec\nrounds: 6499"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6894.216590496278,
            "unit": "iter/sec",
            "range": "stddev: 0.00007233979821531977",
            "extra": "mean: 145.0491128141385 usec\nrounds: 7507"
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
          "id": "3ece9ccc2071f0438ce54ca38804aa27e8a11ca3",
          "message": "Removes some dead code",
          "timestamp": "2024-11-05T17:21:33Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/337/commits/3ece9ccc2071f0438ce54ca38804aa27e8a11ca3"
        },
        "date": 1730830071569,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6087.070652222527,
            "unit": "iter/sec",
            "range": "stddev: 0.00003328599694967904",
            "extra": "mean: 164.282633985015 usec\nrounds: 6457"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6756.765435070431,
            "unit": "iter/sec",
            "range": "stddev: 0.0000703487463827655",
            "extra": "mean: 147.99980991046144 usec\nrounds: 7341"
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
          "id": "df4433cd1db9f08739f02476f01cd1f821abfdc8",
          "message": "[tripy] Sequential module feature branch",
          "timestamp": "2024-11-05T19:06:44Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/321/commits/df4433cd1db9f08739f02476f01cd1f821abfdc8"
        },
        "date": 1730842550605,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6016.460595510388,
            "unit": "iter/sec",
            "range": "stddev: 0.0000336718273588889",
            "extra": "mean: 166.21067887425733 usec\nrounds: 6426"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6665.021460007536,
            "unit": "iter/sec",
            "range": "stddev: 0.00007122943811355846",
            "extra": "mean: 150.03702628721456 usec\nrounds: 7294"
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
          "id": "750eaf6443572adf55d5f581b939eea297ec327c",
          "message": "[tripy] Sequential module feature branch",
          "timestamp": "2024-11-05T19:06:44Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/321/commits/750eaf6443572adf55d5f581b939eea297ec327c"
        },
        "date": 1730842724995,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6150.558829506483,
            "unit": "iter/sec",
            "range": "stddev: 0.00003201227721280346",
            "extra": "mean: 162.58685230399453 usec\nrounds: 6490"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6891.533294068448,
            "unit": "iter/sec",
            "range": "stddev: 0.00007080363205779684",
            "extra": "mean: 145.1055893266454 usec\nrounds: 7459"
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
          "id": "42809b37faf5c651c506349492be3edde0dc9446",
          "message": "Fixes arange type annotation",
          "timestamp": "2024-11-05T19:06:44Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/339/commits/42809b37faf5c651c506349492be3edde0dc9446"
        },
        "date": 1730843779365,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6013.545318430329,
            "unit": "iter/sec",
            "range": "stddev: 0.000037359926198235246",
            "extra": "mean: 166.29125533239062 usec\nrounds: 6435"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6714.759227198997,
            "unit": "iter/sec",
            "range": "stddev: 0.00007126881733121566",
            "extra": "mean: 148.9256674981541 usec\nrounds: 7162"
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
          "id": "9ce8c3bb801fac645a160178090becd351505439",
          "message": "[tripy] Sequential module feature branch",
          "timestamp": "2024-11-05T22:06:14Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/321/commits/9ce8c3bb801fac645a160178090becd351505439"
        },
        "date": 1730844888785,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6033.974247849202,
            "unit": "iter/sec",
            "range": "stddev: 0.00003367964278623998",
            "extra": "mean: 165.7282512195752 usec\nrounds: 6433"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6700.136154032191,
            "unit": "iter/sec",
            "range": "stddev: 0.00007087170065092947",
            "extra": "mean: 149.25069834561387 usec\nrounds: 7257"
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
          "id": "6e2de9d28d4698170b4fe6c9184b0bfd0711fbae",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-05T23:24:44Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/6e2de9d28d4698170b4fe6c9184b0bfd0711fbae"
        },
        "date": 1730912248748,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6108.42781018864,
            "unit": "iter/sec",
            "range": "stddev: 0.00003222055767681326",
            "extra": "mean: 163.7082455704945 usec\nrounds: 6475"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6871.742751899451,
            "unit": "iter/sec",
            "range": "stddev: 0.00007098219418887855",
            "extra": "mean: 145.5234917988723 usec\nrounds: 7380"
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
          "id": "9000015c589bc5ffadab7af6eceec008764f1362",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-05T23:24:44Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/9000015c589bc5ffadab7af6eceec008764f1362"
        },
        "date": 1730913730309,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6089.418978611332,
            "unit": "iter/sec",
            "range": "stddev: 0.000027102026306858248",
            "extra": "mean: 164.21927995305163 usec\nrounds: 6516"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6746.6843078713355,
            "unit": "iter/sec",
            "range": "stddev: 0.00007413890449670957",
            "extra": "mean: 148.22095630490716 usec\nrounds: 7451"
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
          "id": "3c809613e98ccec731febe5c3adaf9a4e03d2433",
          "message": "Convert state_dict param, add warnings for missing/unexpected keys",
          "timestamp": "2024-11-05T23:24:44Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/340/commits/3c809613e98ccec731febe5c3adaf9a4e03d2433"
        },
        "date": 1730921159837,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6078.528908892163,
            "unit": "iter/sec",
            "range": "stddev: 0.00003382539114064846",
            "extra": "mean: 164.51348919919084 usec\nrounds: 6449"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6717.637923999131,
            "unit": "iter/sec",
            "range": "stddev: 0.00006936732393352157",
            "extra": "mean: 148.86184866073907 usec\nrounds: 7389"
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
          "id": "ac0863250f491b3acff83f3ebfcb352f664e7e63",
          "message": "Convert state_dict param, add warnings for missing/unexpected keys",
          "timestamp": "2024-11-05T23:24:44Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/340/commits/ac0863250f491b3acff83f3ebfcb352f664e7e63"
        },
        "date": 1730921333194,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6135.76326025852,
            "unit": "iter/sec",
            "range": "stddev: 0.000033738403631744254",
            "extra": "mean: 162.97890866765067 usec\nrounds: 6487"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6813.235232022758,
            "unit": "iter/sec",
            "range": "stddev: 0.00006980163561975327",
            "extra": "mean: 146.77315048509098 usec\nrounds: 7403"
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
          "id": "7e8a1e3bc76f4463c229aa59a47fd115936a2622",
          "message": "[Tripy] Eliminate need for `skip_num_stack_entries` argument in `convert_to_tensors`",
          "timestamp": "2024-11-06T19:30:04Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/333/commits/7e8a1e3bc76f4463c229aa59a47fd115936a2622"
        },
        "date": 1730932813509,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6124.466325737729,
            "unit": "iter/sec",
            "range": "stddev: 0.00003303004471783471",
            "extra": "mean: 163.27953274843816 usec\nrounds: 6517"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6876.544682145301,
            "unit": "iter/sec",
            "range": "stddev: 0.00006947086760179482",
            "extra": "mean: 145.42187191722374 usec\nrounds: 7391"
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
          "id": "6db25c7709dedde5a77b60b6e8b51315ac8ca2fe",
          "message": "[Tripy] Eliminate need for `skip_num_stack_entries` argument in `convert_to_tensors`",
          "timestamp": "2024-11-06T19:30:04Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/333/commits/6db25c7709dedde5a77b60b6e8b51315ac8ca2fe"
        },
        "date": 1730932987717,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6118.243284909433,
            "unit": "iter/sec",
            "range": "stddev: 0.00003339263139178577",
            "extra": "mean: 163.44560904704247 usec\nrounds: 6468"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6757.482712937446,
            "unit": "iter/sec",
            "range": "stddev: 0.0000716186647382972",
            "extra": "mean: 147.98410036409916 usec\nrounds: 7379"
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
          "id": "737b7f79b32473108cb464ad65975dd6233d4481",
          "message": "[Tripy] Eliminate need for `skip_num_stack_entries` argument in `convert_to_tensors`",
          "timestamp": "2024-11-06T19:30:04Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/333/commits/737b7f79b32473108cb464ad65975dd6233d4481"
        },
        "date": 1730933164384,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6071.040413532976,
            "unit": "iter/sec",
            "range": "stddev: 0.000033574044069173814",
            "extra": "mean: 164.71641298432087 usec\nrounds: 6437"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6758.853539643399,
            "unit": "iter/sec",
            "range": "stddev: 0.00007157502879566817",
            "extra": "mean: 147.95408631576305 usec\nrounds: 7270"
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
          "id": "c024be553a6925319f4a20c532dfde5839d15bb7",
          "message": "Adds a warning if a tensor is evaluated while compiling",
          "timestamp": "2024-11-06T19:30:04Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/344/commits/c024be553a6925319f4a20c532dfde5839d15bb7"
        },
        "date": 1730939542078,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6110.780475079928,
            "unit": "iter/sec",
            "range": "stddev: 0.000033201032953922546",
            "extra": "mean: 163.6452175099483 usec\nrounds: 6488"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6864.586988533826,
            "unit": "iter/sec",
            "range": "stddev: 0.00007241764009880738",
            "extra": "mean: 145.67518798586676 usec\nrounds: 7531"
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
          "id": "cdca7acedbed7b235f829b8e8a65b152e35f0e77",
          "message": "[Tripy] Eliminate need for `skip_num_stack_entries` argument in `convert_to_tensors`",
          "timestamp": "2024-11-06T19:30:04Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/333/commits/cdca7acedbed7b235f829b8e8a65b152e35f0e77"
        },
        "date": 1730941059444,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5965.061819907184,
            "unit": "iter/sec",
            "range": "stddev: 0.00003321504140666314",
            "extra": "mean: 167.64285604932087 usec\nrounds: 6406"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6621.727691474784,
            "unit": "iter/sec",
            "range": "stddev: 0.00007207760828830007",
            "extra": "mean: 151.01798904951363 usec\nrounds: 7273"
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
          "id": "79f61d6e93fdec9654bb53d05bfb5f4398624628",
          "message": "[Tripy] Check type annotations for variadic arguments in the function registry",
          "timestamp": "2024-11-06T19:30:04Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/345/commits/79f61d6e93fdec9654bb53d05bfb5f4398624628"
        },
        "date": 1730952234074,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6063.646847320709,
            "unit": "iter/sec",
            "range": "stddev: 0.0000339576498341227",
            "extra": "mean: 164.91725609677638 usec\nrounds: 6452"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6888.700172508907,
            "unit": "iter/sec",
            "range": "stddev: 0.00006933495372696392",
            "extra": "mean: 145.16526702537465 usec\nrounds: 7407"
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
          "id": "33b130cb4b6e676e7ffdc93861a8a9034d98bbb5",
          "message": "[Tripy] Check type annotations for variadic arguments in the function registry",
          "timestamp": "2024-11-07T04:01:36Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/345/commits/33b130cb4b6e676e7ffdc93861a8a9034d98bbb5"
        },
        "date": 1730953374700,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6105.3713341961275,
            "unit": "iter/sec",
            "range": "stddev: 0.000032918013190985175",
            "extra": "mean: 163.790201326332 usec\nrounds: 6480"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6675.690695724152,
            "unit": "iter/sec",
            "range": "stddev: 0.00007146832070258873",
            "extra": "mean: 149.79723381140326 usec\nrounds: 7244"
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
          "id": "b839326d470f02deba5aef554e33c31d98de604d",
          "message": "[Tripy] Eliminate need for `skip_num_stack_entries` argument in `convert_to_tensors`",
          "timestamp": "2024-11-07T04:01:36Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/333/commits/b839326d470f02deba5aef554e33c31d98de604d"
        },
        "date": 1730953974042,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6054.344752178689,
            "unit": "iter/sec",
            "range": "stddev: 0.000032974174755829856",
            "extra": "mean: 165.17064041325767 usec\nrounds: 6434"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6921.479964334455,
            "unit": "iter/sec",
            "range": "stddev: 0.00007021863594685683",
            "extra": "mean: 144.47777139468417 usec\nrounds: 7518"
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
          "id": "8ad1ef6a947fa6e8c2c85e7edcdf345e7755246a",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-07T04:01:36Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/8ad1ef6a947fa6e8c2c85e7edcdf345e7755246a"
        },
        "date": 1730994717196,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6021.015949836911,
            "unit": "iter/sec",
            "range": "stddev: 0.000025892558097749224",
            "extra": "mean: 166.08492791438073 usec\nrounds: 6388"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6653.897646489402,
            "unit": "iter/sec",
            "range": "stddev: 0.0000769381628437848",
            "extra": "mean: 150.28785429658663 usec\nrounds: 7332"
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
          "id": "73c1f85b80c8bf189a8155c94470f10eb008f71d",
          "message": "Pin to 1.8 version for pytest-virtualenv",
          "timestamp": "2024-11-07T04:01:36Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/346/commits/73c1f85b80c8bf189a8155c94470f10eb008f71d"
        },
        "date": 1731000850951,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6021.5623742939215,
            "unit": "iter/sec",
            "range": "stddev: 0.00003579193783994499",
            "extra": "mean: 166.0698565988463 usec\nrounds: 6455"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6763.857595842079,
            "unit": "iter/sec",
            "range": "stddev: 0.00007070096863536507",
            "extra": "mean: 147.84462650643715 usec\nrounds: 7427"
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
          "id": "d5236c139af4c7b905f3795289167ffd3b35559f",
          "message": "[Tripy] Check type annotations for variadic arguments in the function registry",
          "timestamp": "2024-11-07T20:50:02Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/345/commits/d5236c139af4c7b905f3795289167ffd3b35559f"
        },
        "date": 1731015707523,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6024.268103572687,
            "unit": "iter/sec",
            "range": "stddev: 0.0000352379224480397",
            "extra": "mean: 165.99526827282983 usec\nrounds: 6430"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6918.4656344386985,
            "unit": "iter/sec",
            "range": "stddev: 0.00007215052758373254",
            "extra": "mean: 144.54071940781287 usec\nrounds: 7537"
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
          "id": "de75ca94a8e7177af985324d99a1bae61cedb2a0",
          "message": "[Tripy] Check type annotations for variadic arguments in the function registry",
          "timestamp": "2024-11-07T20:50:02Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/345/commits/de75ca94a8e7177af985324d99a1bae61cedb2a0"
        },
        "date": 1731015884940,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5971.808317611848,
            "unit": "iter/sec",
            "range": "stddev: 0.000032256885202814",
            "extra": "mean: 167.45346582053463 usec\nrounds: 6381"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6654.441279994069,
            "unit": "iter/sec",
            "range": "stddev: 0.00007012240585080379",
            "extra": "mean: 150.27557655462417 usec\nrounds: 7250"
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
          "id": "c9488fadace2cc9c39efb8f48253e28a65f7c2b8",
          "message": "Adds release packaging pipeline and updates version to 0.0.3",
          "timestamp": "2024-11-07T20:50:02Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/347/commits/c9488fadace2cc9c39efb8f48253e28a65f7c2b8"
        },
        "date": 1731020793139,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5993.768045979091,
            "unit": "iter/sec",
            "range": "stddev: 0.0000344082789625233",
            "extra": "mean: 166.83995648961562 usec\nrounds: 6378"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6595.488194302583,
            "unit": "iter/sec",
            "range": "stddev: 0.00006808524138772972",
            "extra": "mean: 151.61879917605427 usec\nrounds: 7245"
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
          "id": "0824585f24223f317937414fb2b04bf69eb0d045",
          "message": "Adds release packaging pipeline and updates version to 0.0.3",
          "timestamp": "2024-11-07T20:50:02Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/347/commits/0824585f24223f317937414fb2b04bf69eb0d045"
        },
        "date": 1731021389103,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6020.094300980138,
            "unit": "iter/sec",
            "range": "stddev: 0.00003342081939981167",
            "extra": "mean: 166.11035475593613 usec\nrounds: 6480"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6808.173424805536,
            "unit": "iter/sec",
            "range": "stddev: 0.00006901806222070683",
            "extra": "mean: 146.88227481933794 usec\nrounds: 7495"
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
          "id": "d798a020b8aa64ee664ab7b5747e6e29b58be117",
          "message": "Fixes release pipeline, adds separate build flow to include stubs",
          "timestamp": "2024-11-08T00:28:01Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/348/commits/d798a020b8aa64ee664ab7b5747e6e29b58be117"
        },
        "date": 1731025955018,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6038.992766112448,
            "unit": "iter/sec",
            "range": "stddev: 0.00003481870909703711",
            "extra": "mean: 165.59052787932742 usec\nrounds: 6444"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6716.088272295698,
            "unit": "iter/sec",
            "range": "stddev: 0.0000738775691611465",
            "extra": "mean: 148.89619663354716 usec\nrounds: 7399"
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
          "id": "ec884f4864a987e5f404b505c1d0696e9718d022",
          "message": "Pins versions for release pipeline dependencies",
          "timestamp": "2024-11-08T02:39:16Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/350/commits/ec884f4864a987e5f404b505c1d0696e9718d022"
        },
        "date": 1731088385575,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5974.9329851867815,
            "unit": "iter/sec",
            "range": "stddev: 0.00003284363077402403",
            "extra": "mean: 167.365893890229 usec\nrounds: 6323"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6624.5872775155285,
            "unit": "iter/sec",
            "range": "stddev: 0.00006647900675737686",
            "extra": "mean: 150.95280024373653 usec\nrounds: 7191"
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
          "id": "248ebcdd85a7bdc0729b78d29eae51505ae18cba",
          "message": "Pins versions for release pipeline dependencies",
          "timestamp": "2024-11-08T02:39:16Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/350/commits/248ebcdd85a7bdc0729b78d29eae51505ae18cba"
        },
        "date": 1731088562724,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6120.624615010083,
            "unit": "iter/sec",
            "range": "stddev: 0.000034509701851897493",
            "extra": "mean: 163.38201783321628 usec\nrounds: 6523"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6783.801414489377,
            "unit": "iter/sec",
            "range": "stddev: 0.00007046811734490962",
            "extra": "mean: 147.40997545478282 usec\nrounds: 7495"
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
          "id": "fe055ae6b0da1fbf16d8a37dab9d40c835756a61",
          "message": "Fixes nanogpt, various other fixes",
          "timestamp": "2024-11-08T18:32:59Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/352/commits/fe055ae6b0da1fbf16d8a37dab9d40c835756a61"
        },
        "date": 1731094809798,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5985.702402405311,
            "unit": "iter/sec",
            "range": "stddev: 0.000033960754133664424",
            "extra": "mean: 167.06477081088383 usec\nrounds: 6409"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6630.6890405863605,
            "unit": "iter/sec",
            "range": "stddev: 0.00007019443705934374",
            "extra": "mean: 150.81388885514207 usec\nrounds: 7318"
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
          "id": "0eca683c982d290ed408c0e47aa6c37c70fd8116",
          "message": "[tripy] Sequential module feature branch",
          "timestamp": "2024-11-08T18:32:59Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/321/commits/0eca683c982d290ed408c0e47aa6c37c70fd8116"
        },
        "date": 1731101432778,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6064.9157937116315,
            "unit": "iter/sec",
            "range": "stddev: 0.00003333578725966919",
            "extra": "mean: 164.8827508927401 usec\nrounds: 6481"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6780.323996956074,
            "unit": "iter/sec",
            "range": "stddev: 0.00006975192904537262",
            "extra": "mean: 147.48557745159894 usec\nrounds: 7412"
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
          "id": "9349bc79fee4c933d76d45ba9c6e87df613f5e56",
          "message": "Fixes nanogpt, various other fixes",
          "timestamp": "2024-11-08T18:32:59Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/352/commits/9349bc79fee4c933d76d45ba9c6e87df613f5e56"
        },
        "date": 1731104389688,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6065.648107891346,
            "unit": "iter/sec",
            "range": "stddev: 0.000026742205822861425",
            "extra": "mean: 164.86284436761343 usec\nrounds: 6490"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6585.524734255947,
            "unit": "iter/sec",
            "range": "stddev: 0.00007706555968053383",
            "extra": "mean: 151.84818831494118 usec\nrounds: 7402"
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
          "id": "78890d7044eab7a99abfb98bd6cf22f06cc5fe2d",
          "message": "Fixes nanogpt, various other fixes",
          "timestamp": "2024-11-08T18:32:59Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/352/commits/78890d7044eab7a99abfb98bd6cf22f06cc5fe2d"
        },
        "date": 1731104584140,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6009.920645719515,
            "unit": "iter/sec",
            "range": "stddev: 0.00003473838574480714",
            "extra": "mean: 166.39154806681793 usec\nrounds: 6458"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6723.507747505038,
            "unit": "iter/sec",
            "range": "stddev: 0.00007183399466780923",
            "extra": "mean: 148.73188781125157 usec\nrounds: 7477"
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
          "id": "d5fe92a40aaa747ab79b34735c9d2d1a0c82ce97",
          "message": "Adds a new workflow to update package index",
          "timestamp": "2024-11-08T23:27:48Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/354/commits/d5fe92a40aaa747ab79b34735c9d2d1a0c82ce97"
        },
        "date": 1731115025342,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6094.637647595853,
            "unit": "iter/sec",
            "range": "stddev: 0.00003315874047411444",
            "extra": "mean: 164.0786635435938 usec\nrounds: 6466"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6672.552131033608,
            "unit": "iter/sec",
            "range": "stddev: 0.00007474825165790366",
            "extra": "mean: 149.8676938542098 usec\nrounds: 7351"
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
          "id": "a921bc6e90fc9dc4fc44d6fae3a93e5ae7276643",
          "message": "Improves various guides, hides incomplete Executable APIs",
          "timestamp": "2024-11-09T01:18:08Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/355/commits/a921bc6e90fc9dc4fc44d6fae3a93e5ae7276643"
        },
        "date": 1731115395409,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6027.613734540237,
            "unit": "iter/sec",
            "range": "stddev: 0.00003471567734526849",
            "extra": "mean: 165.90313249000454 usec\nrounds: 6473"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6764.051926164581,
            "unit": "iter/sec",
            "range": "stddev: 0.0000702016755129134",
            "extra": "mean: 147.8403789497562 usec\nrounds: 7450"
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
          "id": "06a2aec1389bbd0e2f4621d9d2cc7f3654c11500",
          "message": "Updates MLIR-TRT to 0.1.36",
          "timestamp": "2024-11-09T01:41:38Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/356/commits/06a2aec1389bbd0e2f4621d9d2cc7f3654c11500"
        },
        "date": 1731116948896,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6118.92655104443,
            "unit": "iter/sec",
            "range": "stddev: 0.00003272040316060096",
            "extra": "mean: 163.42735799456713 usec\nrounds: 6509"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6841.752733536502,
            "unit": "iter/sec",
            "range": "stddev: 0.00007107087302187712",
            "extra": "mean: 146.16137690832622 usec\nrounds: 7429"
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
          "id": "e2adde6d6e28f50859561a22648933d6f851b995",
          "message": "Fix nanoGPT output text",
          "timestamp": "2024-11-10T19:26:14Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/357/commits/e2adde6d6e28f50859561a22648933d6f851b995"
        },
        "date": 1731357519038,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6090.2245466119275,
            "unit": "iter/sec",
            "range": "stddev: 0.000032630272546297194",
            "extra": "mean: 164.19755829139555 usec\nrounds: 6534"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6869.699704040582,
            "unit": "iter/sec",
            "range": "stddev: 0.00007197612874213601",
            "extra": "mean: 145.56677046768513 usec\nrounds: 7550"
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
          "id": "a50c1c8012ecceefaf58a766d73f964106628531",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-10T19:26:14Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/a50c1c8012ecceefaf58a766d73f964106628531"
        },
        "date": 1731361149306,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6115.434838380278,
            "unit": "iter/sec",
            "range": "stddev: 0.000032902062575164286",
            "extra": "mean: 163.52066965443427 usec\nrounds: 6542"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6597.142927521046,
            "unit": "iter/sec",
            "range": "stddev: 0.00006991469298199878",
            "extra": "mean: 151.58076927943137 usec\nrounds: 7325"
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
          "id": "39462cd4545f9c25a8b05c1413ef526a4c4cab8d",
          "message": "[tripy] Sequential module feature branch",
          "timestamp": "2024-11-10T19:26:14Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/321/commits/39462cd4545f9c25a8b05c1413ef526a4c4cab8d"
        },
        "date": 1731361347674,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6105.12135243812,
            "unit": "iter/sec",
            "range": "stddev: 0.00003289725366043228",
            "extra": "mean: 163.79690791906103 usec\nrounds: 6493"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6818.8665421083715,
            "unit": "iter/sec",
            "range": "stddev: 0.00007402804091368996",
            "extra": "mean: 146.6519389732481 usec\nrounds: 7417"
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
          "id": "51971a1b576128b0ff95638f05f789a3220ca264",
          "message": "SAM2: image pipeline",
          "timestamp": "2024-11-11T21:28:12Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/358/commits/51971a1b576128b0ff95638f05f789a3220ca264"
        },
        "date": 1731424814694,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6016.898352083342,
            "unit": "iter/sec",
            "range": "stddev: 0.000033869177813671004",
            "extra": "mean: 166.19858629550413 usec\nrounds: 6441"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6764.167540771782,
            "unit": "iter/sec",
            "range": "stddev: 0.00006953969080670325",
            "extra": "mean: 147.83785203018516 usec\nrounds: 7438"
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
          "id": "49cd2bf8bc3f2fdcb65c6b5b1d45d5795f7fbeb7",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-12T19:14:18Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/49cd2bf8bc3f2fdcb65c6b5b1d45d5795f7fbeb7"
        },
        "date": 1731441481956,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6104.426775850327,
            "unit": "iter/sec",
            "range": "stddev: 0.00003905081305269111",
            "extra": "mean: 163.8155451312958 usec\nrounds: 6472"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6763.339349204668,
            "unit": "iter/sec",
            "range": "stddev: 0.00007064551368343021",
            "extra": "mean: 147.85595522685026 usec\nrounds: 7389"
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
          "id": "5054b4d425ec54b27736b0b6dd980f6f3ac542f9",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-12T19:14:18Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/5054b4d425ec54b27736b0b6dd980f6f3ac542f9"
        },
        "date": 1731441737150,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6098.9784429969695,
            "unit": "iter/sec",
            "range": "stddev: 0.00003286110174316183",
            "extra": "mean: 163.9618846576233 usec\nrounds: 6501"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6738.791877490131,
            "unit": "iter/sec",
            "range": "stddev: 0.00007032663974720751",
            "extra": "mean: 148.39455175049133 usec\nrounds: 7439"
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
          "id": "8448cf30242b2ce44972b07e8b30686172d6f5cd",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-12T19:14:18Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/8448cf30242b2ce44972b07e8b30686172d6f5cd"
        },
        "date": 1731441994980,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6018.109175784364,
            "unit": "iter/sec",
            "range": "stddev: 0.00004972568296216093",
            "extra": "mean: 166.16514768854555 usec\nrounds: 6543"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6878.376189367292,
            "unit": "iter/sec",
            "range": "stddev: 0.00007281595319832837",
            "extra": "mean: 145.38315039323038 usec\nrounds: 7528"
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
          "id": "ce0d553a31da458956887ff5df0d1ed4c447997e",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-12T19:14:18Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/ce0d553a31da458956887ff5df0d1ed4c447997e"
        },
        "date": 1731442250654,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5895.863453839607,
            "unit": "iter/sec",
            "range": "stddev: 0.00005231728148065346",
            "extra": "mean: 169.61044091832935 usec\nrounds: 6415"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6729.812845962461,
            "unit": "iter/sec",
            "range": "stddev: 0.00007087953532173605",
            "extra": "mean: 148.5925423022645 usec\nrounds: 7403"
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
          "id": "3dae41863e7ab76c4cc68ffd8afef3abf0a20577",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-12T19:14:18Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/3dae41863e7ab76c4cc68ffd8afef3abf0a20577"
        },
        "date": 1731442507401,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6124.7147910384765,
            "unit": "iter/sec",
            "range": "stddev: 0.00003272667869307182",
            "extra": "mean: 163.27290888110804 usec\nrounds: 6453"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6644.009141081506,
            "unit": "iter/sec",
            "range": "stddev: 0.00007494002745818961",
            "extra": "mean: 150.51153283591375 usec\nrounds: 7410"
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
          "id": "298b062a36c2d3cbe56f06d1ffb1c44f7221e959",
          "message": "[tripy] Sequential module feature branch",
          "timestamp": "2024-11-12T19:14:18Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/321/commits/298b062a36c2d3cbe56f06d1ffb1c44f7221e959"
        },
        "date": 1731445828409,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6058.780668689163,
            "unit": "iter/sec",
            "range": "stddev: 0.00003244421700826522",
            "extra": "mean: 165.0497112674576 usec\nrounds: 6409"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6774.663231788182,
            "unit": "iter/sec",
            "range": "stddev: 0.00007261911308642566",
            "extra": "mean: 147.60881327765256 usec\nrounds: 7394"
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
          "id": "815f1e53c709bf1e81755114df2ce89d93379bbe",
          "message": "[tripy] Sequential module feature branch",
          "timestamp": "2024-11-12T19:14:18Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/321/commits/815f1e53c709bf1e81755114df2ce89d93379bbe"
        },
        "date": 1731446024922,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5991.455848179062,
            "unit": "iter/sec",
            "range": "stddev: 0.00003325868767826281",
            "extra": "mean: 166.90434267389662 usec\nrounds: 6416"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6559.0236170218,
            "unit": "iter/sec",
            "range": "stddev: 0.00006965645702094348",
            "extra": "mean: 152.46171661965465 usec\nrounds: 7243"
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
          "id": "08b1641717d768e7cb69735934aa793d9821646e",
          "message": "[tripy] Sequential module feature branch",
          "timestamp": "2024-11-12T19:14:18Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/321/commits/08b1641717d768e7cb69735934aa793d9821646e"
        },
        "date": 1731446529163,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6034.780374375293,
            "unit": "iter/sec",
            "range": "stddev: 0.00003326413416827637",
            "extra": "mean: 165.70611322429738 usec\nrounds: 6465"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6855.433140322868,
            "unit": "iter/sec",
            "range": "stddev: 0.00006848373126388087",
            "extra": "mean: 145.86970356666674 usec\nrounds: 7465"
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
          "id": "745f4586e60ee9967bab0cccf91b99cec84fb4f4",
          "message": "Improves docstrings for overloaded functions, updates README",
          "timestamp": "2024-11-12T21:33:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/364/commits/745f4586e60ee9967bab0cccf91b99cec84fb4f4"
        },
        "date": 1731463758121,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6004.614404982412,
            "unit": "iter/sec",
            "range": "stddev: 0.00003291163060046717",
            "extra": "mean: 166.53858725220326 usec\nrounds: 6381"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6490.774466846982,
            "unit": "iter/sec",
            "range": "stddev: 0.00007078863086780687",
            "extra": "mean: 154.06482001611886 usec\nrounds: 7165"
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
          "id": "cd81822187ec0902f41f9a8109318fb83b4092b6",
          "message": "Improves docstrings for overloaded functions, updates README",
          "timestamp": "2024-11-12T21:33:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/364/commits/cd81822187ec0902f41f9a8109318fb83b4092b6"
        },
        "date": 1731463955282,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6113.751801447159,
            "unit": "iter/sec",
            "range": "stddev: 0.00003253797480698422",
            "extra": "mean: 163.56568478348998 usec\nrounds: 6482"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6643.891027373311,
            "unit": "iter/sec",
            "range": "stddev: 0.00008347751141177091",
            "extra": "mean: 150.51420859853476 usec\nrounds: 7342"
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
          "id": "e0ed951bc7e5683e1368533b78f542a8c60068bb",
          "message": "Improves docstrings for overloaded functions, updates README",
          "timestamp": "2024-11-12T21:33:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/364/commits/e0ed951bc7e5683e1368533b78f542a8c60068bb"
        },
        "date": 1731466158472,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6023.01323438026,
            "unit": "iter/sec",
            "range": "stddev: 0.000033255724115397315",
            "extra": "mean: 166.02985268102194 usec\nrounds: 6473"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6757.854849232651,
            "unit": "iter/sec",
            "range": "stddev: 0.00006997774056571819",
            "extra": "mean: 147.9759512907486 usec\nrounds: 7441"
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
          "id": "289ad3770a81d920845e18bb656466bc6426a0b5",
          "message": "Improves docstrings for overloaded functions, updates README",
          "timestamp": "2024-11-12T21:33:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/364/commits/289ad3770a81d920845e18bb656466bc6426a0b5"
        },
        "date": 1731466399154,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5933.906447844364,
            "unit": "iter/sec",
            "range": "stddev: 0.000050026024835273966",
            "extra": "mean: 168.52304780829064 usec\nrounds: 6416"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6609.418349975783,
            "unit": "iter/sec",
            "range": "stddev: 0.000068321627610034",
            "extra": "mean: 151.29924405581986 usec\nrounds: 7233"
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
          "id": "d09537c5392ba4ae7d51b0947cef0ffea56debd2",
          "message": "Improves docstrings for overloaded functions, updates README",
          "timestamp": "2024-11-12T21:33:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/364/commits/d09537c5392ba4ae7d51b0947cef0ffea56debd2"
        },
        "date": 1731466596162,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6074.711521166805,
            "unit": "iter/sec",
            "range": "stddev: 0.00003280578101346193",
            "extra": "mean: 164.61687053213748 usec\nrounds: 6467"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6618.865563250613,
            "unit": "iter/sec",
            "range": "stddev: 0.00007240088795033316",
            "extra": "mean: 151.0832922113146 usec\nrounds: 7344"
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
          "id": "3e284c978cfc38eb73a3d8fb13299cda3e269c11",
          "message": "[Tripy] Mixed module list/dict for `tp.Module`",
          "timestamp": "2024-11-12T21:33:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/365/commits/3e284c978cfc38eb73a3d8fb13299cda3e269c11"
        },
        "date": 1731513950104,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6031.695930049013,
            "unit": "iter/sec",
            "range": "stddev: 0.000032561890612025143",
            "extra": "mean: 165.790850798388 usec\nrounds: 6422"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6591.109519258459,
            "unit": "iter/sec",
            "range": "stddev: 0.00007480424889432792",
            "extra": "mean: 151.71952416783787 usec\nrounds: 7372"
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
          "id": "6b2842176eac3cb885950fc8eb1abc6afec22ec0",
          "message": "[Tripy] Mixed module list/dict for `tp.Module`",
          "timestamp": "2024-11-12T21:33:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/365/commits/6b2842176eac3cb885950fc8eb1abc6afec22ec0"
        },
        "date": 1731514152201,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6108.04190744003,
            "unit": "iter/sec",
            "range": "stddev: 0.0000329714467747296",
            "extra": "mean: 163.7185885679548 usec\nrounds: 6543"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6703.927860089493,
            "unit": "iter/sec",
            "range": "stddev: 0.00006994940085960487",
            "extra": "mean: 149.16628294187083 usec\nrounds: 7390"
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
          "id": "fe0323c18e85cba7273ad6d417d67538e23811f8",
          "message": "[Tripy] Mixed module list/dict for `tp.Module`",
          "timestamp": "2024-11-13T22:38:23Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/365/commits/fe0323c18e85cba7273ad6d417d67538e23811f8"
        },
        "date": 1731547536929,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6127.890106717352,
            "unit": "iter/sec",
            "range": "stddev: 0.00003311444934002678",
            "extra": "mean: 163.18830504218846 usec\nrounds: 6479"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6759.665725770445,
            "unit": "iter/sec",
            "range": "stddev: 0.00007469260868457537",
            "extra": "mean: 147.9363093632893 usec\nrounds: 7373"
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
          "id": "eea8bc38ae541f4c7cb6bf995351e37bf67f49a0",
          "message": "[Tripy] Mixed module list/dict for `tp.Module`",
          "timestamp": "2024-11-13T22:38:23Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/365/commits/eea8bc38ae541f4c7cb6bf995351e37bf67f49a0"
        },
        "date": 1731547738515,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6077.340377191497,
            "unit": "iter/sec",
            "range": "stddev: 0.00003318426058918356",
            "extra": "mean: 164.54566272987444 usec\nrounds: 6480"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6748.050938125683,
            "unit": "iter/sec",
            "range": "stddev: 0.00007385868477382335",
            "extra": "mean: 148.19093826783657 usec\nrounds: 7360"
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
          "id": "e2b04cdebe6f68100ace8d81de6b0cae9f291871",
          "message": "Adds tp.equal, improves Module prints, fixes nanoGPT",
          "timestamp": "2024-11-14T20:29:06Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/375/commits/e2b04cdebe6f68100ace8d81de6b0cae9f291871"
        },
        "date": 1731617921205,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6115.588459741236,
            "unit": "iter/sec",
            "range": "stddev: 0.00003190602280210883",
            "extra": "mean: 163.51656207460243 usec\nrounds: 6472"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6737.0871760918135,
            "unit": "iter/sec",
            "range": "stddev: 0.00007026944877795406",
            "extra": "mean: 148.43210038141447 usec\nrounds: 7315"
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
          "id": "ddfed6393211de00ee5776be90ff2297f81ee683",
          "message": "[Tripy] Mixed module list/dict for `tp.Module`",
          "timestamp": "2024-11-14T20:29:06Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/365/commits/ddfed6393211de00ee5776be90ff2297f81ee683"
        },
        "date": 1731618300380,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6045.794315364413,
            "unit": "iter/sec",
            "range": "stddev: 0.000033621264565151656",
            "extra": "mean: 165.4042376960561 usec\nrounds: 6442"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6846.1826220515895,
            "unit": "iter/sec",
            "range": "stddev: 0.00007020663427162471",
            "extra": "mean: 146.06680177928573 usec\nrounds: 7485"
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
          "id": "a2d231e7f4a5ab01615eebdeb1286c83163db703",
          "message": "Adds tp.equal, improves Module prints, fixes nanoGPT",
          "timestamp": "2024-11-14T20:29:06Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/375/commits/a2d231e7f4a5ab01615eebdeb1286c83163db703"
        },
        "date": 1731618503783,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6093.098903223994,
            "unit": "iter/sec",
            "range": "stddev: 0.00003253175605022139",
            "extra": "mean: 164.12009978549304 usec\nrounds: 6494"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6864.920886031284,
            "unit": "iter/sec",
            "range": "stddev: 0.00006929899122691099",
            "extra": "mean: 145.6681026047651 usec\nrounds: 7495"
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
          "id": "1c99c62c4d92562a147c6245f3c78a6b7d59516e",
          "message": "[Tripy] Mixed module list/dict for `tp.Module`",
          "timestamp": "2024-11-14T20:29:06Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/365/commits/1c99c62c4d92562a147c6245f3c78a6b7d59516e"
        },
        "date": 1731619744280,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6069.36897672967,
            "unit": "iter/sec",
            "range": "stddev: 0.00003215026074329993",
            "extra": "mean: 164.7617740549406 usec\nrounds: 6475"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6724.055903293359,
            "unit": "iter/sec",
            "range": "stddev: 0.00006821160690242873",
            "extra": "mean: 148.71976294994997 usec\nrounds: 7415"
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
          "id": "eaf5af5b5ee8c3ef12c83854eaffee10e09b38c3",
          "message": "[Tripy] Mixed module list/dict for `tp.Module`",
          "timestamp": "2024-11-14T22:14:17Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/365/commits/eaf5af5b5ee8c3ef12c83854eaffee10e09b38c3"
        },
        "date": 1731622981793,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5999.080700035261,
            "unit": "iter/sec",
            "range": "stddev: 0.00003363114136426906",
            "extra": "mean: 166.69220668995538 usec\nrounds: 6395"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6498.988549494412,
            "unit": "iter/sec",
            "range": "stddev: 0.00006934045691951362",
            "extra": "mean: 153.87009722886722 usec\nrounds: 7169"
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
          "id": "e6b673af7d771356632a1e3bfd6107dcfd990d9f",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-14T22:56:14Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/e6b673af7d771356632a1e3bfd6107dcfd990d9f"
        },
        "date": 1731625593255,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6131.507641762626,
            "unit": "iter/sec",
            "range": "stddev: 0.00003404296096429291",
            "extra": "mean: 163.0920253917403 usec\nrounds: 6505"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6775.3061602304615,
            "unit": "iter/sec",
            "range": "stddev: 0.00006861710490401794",
            "extra": "mean: 147.5948062494618 usec\nrounds: 7487"
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
          "id": "c405be638c603684b597c180c238d309e5e03bae",
          "message": "[Tripy] Eliminate need for `skip_num_stack_entries` argument in `convert_to_tensors`",
          "timestamp": "2024-11-14T23:17:20Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/333/commits/c405be638c603684b597c180c238d309e5e03bae"
        },
        "date": 1731646657637,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6121.713813674413,
            "unit": "iter/sec",
            "range": "stddev: 0.000033358992395351505",
            "extra": "mean: 163.35294828161426 usec\nrounds: 6525"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6707.038152316176,
            "unit": "iter/sec",
            "range": "stddev: 0.00007037500454083222",
            "extra": "mean: 149.09710922915576 usec\nrounds: 7363"
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
          "id": "486890f00fce9222d14da2274fbc1b9b2fff9d3d",
          "message": "[Tripy] Return the shape immediately if it is statically known instead of producing a trace operator.",
          "timestamp": "2024-11-14T23:17:20Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/379/commits/486890f00fce9222d14da2274fbc1b9b2fff9d3d"
        },
        "date": 1731662648260,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6016.645649410355,
            "unit": "iter/sec",
            "range": "stddev: 0.00003385205053121335",
            "extra": "mean: 166.20556673434845 usec\nrounds: 6443"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6612.218922698683,
            "unit": "iter/sec",
            "range": "stddev: 0.00007132883475191575",
            "extra": "mean: 151.23516200698694 usec\nrounds: 7297"
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
          "id": "e35c7217c0b4b85a9e0487e40d3f72d6aedc71ce",
          "message": "[Tripy] Mixed module list/dict for `tp.Module`",
          "timestamp": "2024-11-15T15:50:05Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/365/commits/e35c7217c0b4b85a9e0487e40d3f72d6aedc71ce"
        },
        "date": 1731687569500,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6113.570009155232,
            "unit": "iter/sec",
            "range": "stddev: 0.000032473170395725754",
            "extra": "mean: 163.57054855059707 usec\nrounds: 6532"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6716.887457811925,
            "unit": "iter/sec",
            "range": "stddev: 0.00006980322173051444",
            "extra": "mean: 148.87848073693306 usec\nrounds: 7530"
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
          "id": "8e3a4f4b7ac71505501ba8964f280bef960975fc",
          "message": "[Tripy] Return the shape immediately if it is statically known instead of producing a trace operator.",
          "timestamp": "2024-11-16T22:10:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/379/commits/8e3a4f4b7ac71505501ba8964f280bef960975fc"
        },
        "date": 1731817081063,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6168.293526200989,
            "unit": "iter/sec",
            "range": "stddev: 0.0000941534080965553",
            "extra": "mean: 162.1193926249313 usec\nrounds: 6539"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6999.08153130217,
            "unit": "iter/sec",
            "range": "stddev: 0.0000880928166420028",
            "extra": "mean: 142.87588957603575 usec\nrounds: 7484"
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
          "id": "ede832ced3d42e5986992db3ef265e408f5cd19b",
          "message": "[Tripy] Mixed module list/dict for `tp.Module`",
          "timestamp": "2024-11-16T22:10:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/365/commits/ede832ced3d42e5986992db3ef265e408f5cd19b"
        },
        "date": 1731944137183,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6075.844556427811,
            "unit": "iter/sec",
            "range": "stddev: 0.00003367446331286808",
            "extra": "mean: 164.58617245927914 usec\nrounds: 6480"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6801.305949149613,
            "unit": "iter/sec",
            "range": "stddev: 0.00006925610968367942",
            "extra": "mean: 147.03058610751555 usec\nrounds: 7367"
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
          "id": "aa72ea8ca155b6837f7298ffd3f28d9ef3a893bf",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-18T15:36:07Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/aa72ea8ca155b6837f7298ffd3f28d9ef3a893bf"
        },
        "date": 1731944962034,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5964.269084430337,
            "unit": "iter/sec",
            "range": "stddev: 0.00004907779746619435",
            "extra": "mean: 167.66513814919747 usec\nrounds: 6473"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6823.498312018523,
            "unit": "iter/sec",
            "range": "stddev: 0.00006850105960693115",
            "extra": "mean: 146.55239208290806 usec\nrounds: 7439"
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
          "id": "fae79046aa2cd7d42e007f0c4d220f51668f0bd3",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-18T15:36:07Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/fae79046aa2cd7d42e007f0c4d220f51668f0bd3"
        },
        "date": 1731945225833,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5916.103426404342,
            "unit": "iter/sec",
            "range": "stddev: 0.0000503988087876254",
            "extra": "mean: 169.03017542541082 usec\nrounds: 6438"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6818.548072536014,
            "unit": "iter/sec",
            "range": "stddev: 0.00007574819784149692",
            "extra": "mean: 146.65878855174975 usec\nrounds: 7389"
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
          "id": "fd7c2883fb8824d3803e1d55c5bb5f295825b33d",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-18T15:36:07Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/fd7c2883fb8824d3803e1d55c5bb5f295825b33d"
        },
        "date": 1731961611955,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6030.4385677781775,
            "unit": "iter/sec",
            "range": "stddev: 0.00003237730457116714",
            "extra": "mean: 165.82541862596815 usec\nrounds: 6412"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6695.396556511451,
            "unit": "iter/sec",
            "range": "stddev: 0.00006807420746345792",
            "extra": "mean: 149.35635127205026 usec\nrounds: 7273"
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
          "id": "5d5fb9233508c3d6c3be05ebc26f4d0b96d682bb",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-18T15:36:07Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/5d5fb9233508c3d6c3be05ebc26f4d0b96d682bb"
        },
        "date": 1731962360908,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6013.551795060394,
            "unit": "iter/sec",
            "range": "stddev: 0.00003328532067476138",
            "extra": "mean: 166.29107623574677 usec\nrounds: 6426"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6679.526841236356,
            "unit": "iter/sec",
            "range": "stddev: 0.00006810329696218501",
            "extra": "mean: 149.7112031688316 usec\nrounds: 7313"
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
          "id": "079bdbd10e7499615575cbd3011d5e1c7bcfc04c",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-19T18:24:49Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/079bdbd10e7499615575cbd3011d5e1c7bcfc04c"
        },
        "date": 1732042300535,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5998.01318309236,
            "unit": "iter/sec",
            "range": "stddev: 0.000034841202915554416",
            "extra": "mean: 166.72187430645758 usec\nrounds: 6449"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6714.920372157459,
            "unit": "iter/sec",
            "range": "stddev: 0.00007473930146997055",
            "extra": "mean: 148.92209357334593 usec\nrounds: 7332"
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
          "id": "1b6ad08356a042f435653527ce33c3ae98e140c1",
          "message": "[Tripy] Resnet50 feature branch PR",
          "timestamp": "2024-11-19T18:24:49Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/1b6ad08356a042f435653527ce33c3ae98e140c1"
        },
        "date": 1732051911565,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6060.880449258911,
            "unit": "iter/sec",
            "range": "stddev: 0.00003335869568619598",
            "extra": "mean: 164.99253010711905 usec\nrounds: 6485"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6790.469729083481,
            "unit": "iter/sec",
            "range": "stddev: 0.00007101817786885722",
            "extra": "mean: 147.26521726722598 usec\nrounds: 7476"
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
          "id": "fa96f1c4211d403765cae638bd747db054d9d650",
          "message": "[Tripy] Return the shape immediately if it is statically known instead of producing a trace operator.",
          "timestamp": "2024-11-19T18:24:49Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/379/commits/fa96f1c4211d403765cae638bd747db054d9d650"
        },
        "date": 1732074863584,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6147.735694381763,
            "unit": "iter/sec",
            "range": "stddev: 0.00003201218371453429",
            "extra": "mean: 162.66151469619473 usec\nrounds: 6514"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6632.362278958066,
            "unit": "iter/sec",
            "range": "stddev: 0.00007069182609035583",
            "extra": "mean: 150.7758409356822 usec\nrounds: 7373"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 7092.321277615057,
            "unit": "iter/sec",
            "range": "stddev: 0.000023803509906882457",
            "extra": "mean: 140.99756072193495 usec\nrounds: 7692"
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
          "id": "b81852f965e286858a0ee995b320b8add5157e16",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-20T18:21:22Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/b81852f965e286858a0ee995b320b8add5157e16"
        },
        "date": 1732129655005,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6037.298845127797,
            "unit": "iter/sec",
            "range": "stddev: 0.000034932830422290194",
            "extra": "mean: 165.63698860244713 usec\nrounds: 6458"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6794.729595237767,
            "unit": "iter/sec",
            "range": "stddev: 0.00006895596971539191",
            "extra": "mean: 147.17289128045235 usec\nrounds: 7336"
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
          "id": "ece2ff9b668d8d7381ad917d06fe57010a795874",
          "message": "Update tripy version to 0.0.4",
          "timestamp": "2024-11-20T18:21:22Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/397/commits/ece2ff9b668d8d7381ad917d06fe57010a795874"
        },
        "date": 1732139543340,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6125.067127346832,
            "unit": "iter/sec",
            "range": "stddev: 0.000031805543665235164",
            "extra": "mean: 163.26351682502545 usec\nrounds: 6545"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6877.5699627006625,
            "unit": "iter/sec",
            "range": "stddev: 0.00007145515062814178",
            "extra": "mean: 145.40019300760747 usec\nrounds: 7561"
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
          "id": "5203fde4c4d754fc4c912aff027473d233d780ea",
          "message": "Add compile fixture to test integration ops with compile mode",
          "timestamp": "2024-11-20T21:57:31Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/387/commits/5203fde4c4d754fc4c912aff027473d233d780ea"
        },
        "date": 1732148600450,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6077.31636845702,
            "unit": "iter/sec",
            "range": "stddev: 0.000032419022025175225",
            "extra": "mean: 164.5463127755338 usec\nrounds: 6459"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6727.780894778269,
            "unit": "iter/sec",
            "range": "stddev: 0.00006980507974065436",
            "extra": "mean: 148.6374208137701 usec\nrounds: 7458"
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
          "id": "fa1f52318628fb31816a16fa100e764782a7a9f7",
          "message": "[Tripy] Return the shape immediately if it is statically known instead of producing a trace operator.",
          "timestamp": "2024-11-21T00:28:11Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/379/commits/fa1f52318628fb31816a16fa100e764782a7a9f7"
        },
        "date": 1732160756703,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6104.398759566138,
            "unit": "iter/sec",
            "range": "stddev: 0.00003213767535818591",
            "extra": "mean: 163.81629696666042 usec\nrounds: 6517"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6879.262277240794,
            "unit": "iter/sec",
            "range": "stddev: 0.00007246949884337945",
            "extra": "mean: 145.3644242215301 usec\nrounds: 7678"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 6899.9829369006875,
            "unit": "iter/sec",
            "range": "stddev: 0.000023864329748574885",
            "extra": "mean: 144.9278946259506 usec\nrounds: 7541"
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
          "id": "cc302397650d1066613cdacf8c1ff448bc938f4f",
          "message": "[Tripy] Return the shape immediately if it is statically known instead of producing a trace operator.",
          "timestamp": "2024-11-21T00:28:11Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/379/commits/cc302397650d1066613cdacf8c1ff448bc938f4f"
        },
        "date": 1732161016705,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6066.191816647702,
            "unit": "iter/sec",
            "range": "stddev: 0.00003254663431128011",
            "extra": "mean: 164.84806782002153 usec\nrounds: 6473"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6605.507035123587,
            "unit": "iter/sec",
            "range": "stddev: 0.00007155253041608748",
            "extra": "mean: 151.38883278492946 usec\nrounds: 7460"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 7068.517697787177,
            "unit": "iter/sec",
            "range": "stddev: 0.000022247563521774548",
            "extra": "mean: 141.4723769190043 usec\nrounds: 7663"
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
          "id": "6fd693dacbf6826bd5b4928aa47f318b33d3d314",
          "message": "[Tripy] Change function registry to also register methods of registered classes",
          "timestamp": "2024-11-21T00:28:11Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/388/commits/6fd693dacbf6826bd5b4928aa47f318b33d3d314"
        },
        "date": 1732161282911,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5006.772846310616,
            "unit": "iter/sec",
            "range": "stddev: 0.00004982263509561768",
            "extra": "mean: 199.72945262273657 usec\nrounds: 5556"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 4940.8803897906,
            "unit": "iter/sec",
            "range": "stddev: 0.00004406755814134844",
            "extra": "mean: 202.3930799997328 usec\nrounds: 5469"
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
          "id": "c590a8d9228b310e6a349bbf48c4fc3b9cb627f9",
          "message": "Update mlir-tensorrt dependency version in Tripy",
          "timestamp": "2024-11-21T00:28:11Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/399/commits/c590a8d9228b310e6a349bbf48c4fc3b9cb627f9"
        },
        "date": 1732214516811,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5941.452997509103,
            "unit": "iter/sec",
            "range": "stddev: 0.00004918934627931883",
            "extra": "mean: 168.30899788641605 usec\nrounds: 6422"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6648.044613147404,
            "unit": "iter/sec",
            "range": "stddev: 0.00006878873343127624",
            "extra": "mean: 150.42016986804893 usec\nrounds: 7313"
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
          "id": "659e6243ab219bc920165fde67b6d9ca88e9781f",
          "message": "Updates various guides",
          "timestamp": "2024-11-21T18:43:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/400/commits/659e6243ab219bc920165fde67b6d9ca88e9781f"
        },
        "date": 1732223201268,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6068.506104584422,
            "unit": "iter/sec",
            "range": "stddev: 0.000032524146872054356",
            "extra": "mean: 164.78520129436058 usec\nrounds: 6485"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6693.593169993333,
            "unit": "iter/sec",
            "range": "stddev: 0.0000687448870176629",
            "extra": "mean: 149.3965908300035 usec\nrounds: 7451"
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
          "id": "6fd693dacbf6826bd5b4928aa47f318b33d3d314",
          "message": "[Tripy] Change function registry to also register methods of registered classes",
          "timestamp": "2024-11-21T00:28:11Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/388/commits/6fd693dacbf6826bd5b4928aa47f318b33d3d314"
        },
        "date": 1732225934057,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 4765.925245274582,
            "unit": "iter/sec",
            "range": "stddev: 0.00005221329162265114",
            "extra": "mean: 209.8228462545653 usec\nrounds: 5449"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 4748.863123514764,
            "unit": "iter/sec",
            "range": "stddev: 0.000044662403847402216",
            "extra": "mean: 210.57671572977924 usec\nrounds: 5300"
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
          "id": "02f66fc6af7838f9f54a4c9133e605065f62d0f0",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-21T18:43:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/02f66fc6af7838f9f54a4c9133e605065f62d0f0"
        },
        "date": 1732231444026,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5985.497764437115,
            "unit": "iter/sec",
            "range": "stddev: 0.00003496872908041358",
            "extra": "mean: 167.07048258233567 usec\nrounds: 6464"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6780.524348658512,
            "unit": "iter/sec",
            "range": "stddev: 0.0000753959178386608",
            "extra": "mean: 147.48121953102998 usec\nrounds: 7367"
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
          "id": "8cc42f4b943c77dacfb7f017214ef2cf7a93c170",
          "message": "Updates various guides",
          "timestamp": "2024-11-21T18:43:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/400/commits/8cc42f4b943c77dacfb7f017214ef2cf7a93c170"
        },
        "date": 1732232126292,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6138.1301651212025,
            "unit": "iter/sec",
            "range": "stddev: 0.00003233253468058102",
            "extra": "mean: 162.9160628887143 usec\nrounds: 6533"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6859.419191501311,
            "unit": "iter/sec",
            "range": "stddev: 0.00007117577593023368",
            "extra": "mean: 145.78493777417495 usec\nrounds: 7602"
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
          "id": "376bffd50d81ea6ec5b646fa5f12b18cb33d7ca8",
          "message": "[Tripy] Change function registry to also register methods of registered classes",
          "timestamp": "2024-11-21T23:55:01Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/388/commits/376bffd50d81ea6ec5b646fa5f12b18cb33d7ca8"
        },
        "date": 1732243770657,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5924.524007225087,
            "unit": "iter/sec",
            "range": "stddev: 0.000050375208495339094",
            "extra": "mean: 168.7899312721964 usec\nrounds: 6425"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6558.159414729823,
            "unit": "iter/sec",
            "range": "stddev: 0.00006778548070412637",
            "extra": "mean: 152.48180728177636 usec\nrounds: 7350"
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
          "id": "71eb847eb2c46910bfbe589675fc121aacf79d22",
          "message": "[Tripy] Change function registry to also register methods of registered classes",
          "timestamp": "2024-11-21T23:55:01Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/388/commits/71eb847eb2c46910bfbe589675fc121aacf79d22"
        },
        "date": 1732244048015,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5972.723788154643,
            "unit": "iter/sec",
            "range": "stddev: 0.00003458319892120024",
            "extra": "mean: 167.4277993539969 usec\nrounds: 6375"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6565.032252989789,
            "unit": "iter/sec",
            "range": "stddev: 0.00006544544310042284",
            "extra": "mean: 152.32217626114308 usec\nrounds: 7211"
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
          "id": "addfef9984f61cf5d6f449dbde149f8dccbd1187",
          "message": "[Tripy] Change function registry to also register methods of registered classes",
          "timestamp": "2024-11-21T23:55:01Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/388/commits/addfef9984f61cf5d6f449dbde149f8dccbd1187"
        },
        "date": 1732244316681,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6003.63763630939,
            "unit": "iter/sec",
            "range": "stddev: 0.00003535927908326296",
            "extra": "mean: 166.56568243761112 usec\nrounds: 6427"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6713.481395370431,
            "unit": "iter/sec",
            "range": "stddev: 0.00007161409712916484",
            "extra": "mean: 148.95401373862344 usec\nrounds: 7414"
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
          "id": "fa8ef9bfae47f0d261e9f3fccea62f0445f24297",
          "message": "[Tripy] Change function registry to also register methods of registered classes",
          "timestamp": "2024-11-21T23:55:01Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/388/commits/fa8ef9bfae47f0d261e9f3fccea62f0445f24297"
        },
        "date": 1732244585240,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6111.879718330844,
            "unit": "iter/sec",
            "range": "stddev: 0.00003368092640862569",
            "extra": "mean: 163.6157853370027 usec\nrounds: 6450"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6595.019577160748,
            "unit": "iter/sec",
            "range": "stddev: 0.00007072187577445942",
            "extra": "mean: 151.6295726343415 usec\nrounds: 7216"
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
          "id": "c3221dbca936b4c3adcfffd82ce0e4bb893d9a06",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-22T02:55:05Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/c3221dbca936b4c3adcfffd82ce0e4bb893d9a06"
        },
        "date": 1732298325589,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6032.925576734166,
            "unit": "iter/sec",
            "range": "stddev: 0.000031782215590981565",
            "extra": "mean: 165.7570588731405 usec\nrounds: 6506"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6717.0600683639195,
            "unit": "iter/sec",
            "range": "stddev: 0.00006919643651408026",
            "extra": "mean: 148.8746549565353 usec\nrounds: 7384"
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
          "id": "6ff72f31cdf78d35ebef9ed28ab5e232d0b5bcfe",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-22T02:55:05Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/6ff72f31cdf78d35ebef9ed28ab5e232d0b5bcfe"
        },
        "date": 1732565421509,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6089.969075598452,
            "unit": "iter/sec",
            "range": "stddev: 0.000033183576421377186",
            "extra": "mean: 164.20444629297754 usec\nrounds: 6509"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6857.358874140043,
            "unit": "iter/sec",
            "range": "stddev: 0.00006962944459578629",
            "extra": "mean: 145.8287393665694 usec\nrounds: 7398"
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
          "id": "ad478132d217f3aa78b1226ccb704fc1253ccaf5",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-22T02:55:05Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/ad478132d217f3aa78b1226ccb704fc1253ccaf5"
        },
        "date": 1732570198556,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6137.405828538043,
            "unit": "iter/sec",
            "range": "stddev: 0.00003229474017169136",
            "extra": "mean: 162.93529024105683 usec\nrounds: 6489"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6900.523890006239,
            "unit": "iter/sec",
            "range": "stddev: 0.00007027833667012434",
            "extra": "mean: 144.9165332864453 usec\nrounds: 7471"
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
          "id": "41c10dcaf94fe426f01a8cc96d65c1363af2d1c8",
          "message": "[Tripy] Return the shape immediately if it is statically known instead of producing a trace operator.",
          "timestamp": "2024-11-25T22:22:19Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/379/commits/41c10dcaf94fe426f01a8cc96d65c1363af2d1c8"
        },
        "date": 1732577234021,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6167.772273125027,
            "unit": "iter/sec",
            "range": "stddev: 0.00003245690021382608",
            "extra": "mean: 162.1330937196437 usec\nrounds: 6511"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6888.792956026189,
            "unit": "iter/sec",
            "range": "stddev: 0.00007039367428126436",
            "extra": "mean: 145.1633118288478 usec\nrounds: 7471"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 6985.480800644266,
            "unit": "iter/sec",
            "range": "stddev: 0.00002334229840100444",
            "extra": "mean: 143.15406892361233 usec\nrounds: 7627"
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
          "id": "fef9ec96383fb76ab2b04bf0202702031c2d5e8e",
          "message": "[Tripy]Add notebook test dep for testing res50 notebook L1 test",
          "timestamp": "2024-11-25T22:22:19Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/406/commits/fef9ec96383fb76ab2b04bf0202702031c2d5e8e"
        },
        "date": 1732578091345,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6026.297466243008,
            "unit": "iter/sec",
            "range": "stddev: 0.00003292634034314243",
            "extra": "mean: 165.93936917346247 usec\nrounds: 6389"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6684.65849344794,
            "unit": "iter/sec",
            "range": "stddev: 0.00006799412323239995",
            "extra": "mean: 149.59627346410647 usec\nrounds: 7177"
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
          "id": "861e6d5246c35692711b7e3732c78ce5eead5ec6",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-25T23:41:55Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/861e6d5246c35692711b7e3732c78ce5eead5ec6"
        },
        "date": 1732578653024,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6136.486241515054,
            "unit": "iter/sec",
            "range": "stddev: 0.00003295465951457224",
            "extra": "mean: 162.95970701192468 usec\nrounds: 6485"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6826.388206622778,
            "unit": "iter/sec",
            "range": "stddev: 0.00007084938177678163",
            "extra": "mean: 146.49035034805476 usec\nrounds: 7415"
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
          "id": "f6f6aebccef80d62f9b204703cb296a85bf52591",
          "message": "[Tripy] Return the shape immediately if it is statically known instead of producing a trace operator.",
          "timestamp": "2024-11-26T01:23:53Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/379/commits/f6f6aebccef80d62f9b204703cb296a85bf52591"
        },
        "date": 1732587688247,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6123.975430117609,
            "unit": "iter/sec",
            "range": "stddev: 0.00003212577565173818",
            "extra": "mean: 163.2926211757834 usec\nrounds: 6518"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6840.973592401408,
            "unit": "iter/sec",
            "range": "stddev: 0.00007094135621775035",
            "extra": "mean: 146.1780237115295 usec\nrounds: 7466"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 6939.565621697052,
            "unit": "iter/sec",
            "range": "stddev: 0.00002132623748756068",
            "extra": "mean: 144.1012383935715 usec\nrounds: 7532"
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
          "id": "c0d57f5ead4fe80498561ca9c4ad9502996aae55",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-26T01:23:53Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/c0d57f5ead4fe80498561ca9c4ad9502996aae55"
        },
        "date": 1732644611003,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5916.629943760107,
            "unit": "iter/sec",
            "range": "stddev: 0.00004932537252955969",
            "extra": "mean: 169.01513353131645 usec\nrounds: 6424"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6663.411584912192,
            "unit": "iter/sec",
            "range": "stddev: 0.00006705299232645387",
            "extra": "mean: 150.07327511695013 usec\nrounds: 7253"
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
          "id": "e1a6c0120de2a70955580ddb586a768cbd7eee88",
          "message": "add notebook dependency for notebook testing",
          "timestamp": "2024-11-26T01:23:53Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/407/commits/e1a6c0120de2a70955580ddb586a768cbd7eee88"
        },
        "date": 1732646425098,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6072.9849630075505,
            "unit": "iter/sec",
            "range": "stddev: 0.000032361240699695375",
            "extra": "mean: 164.6636713397633 usec\nrounds: 6499"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6788.526495045962,
            "unit": "iter/sec",
            "range": "stddev: 0.00006851396256905044",
            "extra": "mean: 147.30737233326943 usec\nrounds: 7495"
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
          "id": "d663a3e88de22b4d1cc76c565c3859d9fc0762f5",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-26T18:40:44Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/d663a3e88de22b4d1cc76c565c3859d9fc0762f5"
        },
        "date": 1732647428905,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6085.027794677594,
            "unit": "iter/sec",
            "range": "stddev: 0.00003305518687527409",
            "extra": "mean: 164.33778673528366 usec\nrounds: 6443"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6807.817928296441,
            "unit": "iter/sec",
            "range": "stddev: 0.00007364662714752904",
            "extra": "mean: 146.8899448446671 usec\nrounds: 7376"
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
          "id": "91639443b367df62399eb10958948427959f7d9e",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-26T18:40:44Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/91639443b367df62399eb10958948427959f7d9e"
        },
        "date": 1732650024814,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6115.053698194188,
            "unit": "iter/sec",
            "range": "stddev: 0.000032665090147117676",
            "extra": "mean: 163.53086160066036 usec\nrounds: 6500"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6741.808918489042,
            "unit": "iter/sec",
            "range": "stddev: 0.0000726759924200299",
            "extra": "mean: 148.3281433945057 usec\nrounds: 7353"
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
          "id": "f41e4fafbf9fb8acd7a1c865ddf440ae2f8f1a74",
          "message": "Removes dead code",
          "timestamp": "2024-11-26T20:57:25Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/412/commits/f41e4fafbf9fb8acd7a1c865ddf440ae2f8f1a74"
        },
        "date": 1732657679977,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5952.839469347794,
            "unit": "iter/sec",
            "range": "stddev: 0.00005015312179841794",
            "extra": "mean: 167.9870598139214 usec\nrounds: 6463"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6801.774686613314,
            "unit": "iter/sec",
            "range": "stddev: 0.0000691501376027541",
            "extra": "mean: 147.0204536425055 usec\nrounds: 7406"
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
          "id": "10f482bfd7c8f7f2b64ee60c0fc2df0a42e9b951",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-26T20:57:25Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/10f482bfd7c8f7f2b64ee60c0fc2df0a42e9b951"
        },
        "date": 1732658887733,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5908.837413418636,
            "unit": "iter/sec",
            "range": "stddev: 0.000050799237361916586",
            "extra": "mean: 169.23802941828393 usec\nrounds: 6411"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6593.448379677744,
            "unit": "iter/sec",
            "range": "stddev: 0.00006799712324263033",
            "extra": "mean: 151.66570547244888 usec\nrounds: 7247"
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
          "id": "9568cac8d11c489e7e1ed3ba5fa4f1573af4b449",
          "message": "[Tripy] Add tripy for setup tools to pass L1 test",
          "timestamp": "2024-11-26T20:57:25Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/413/commits/9568cac8d11c489e7e1ed3ba5fa4f1573af4b449"
        },
        "date": 1732660113564,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6143.282039726387,
            "unit": "iter/sec",
            "range": "stddev: 0.00003498574692738126",
            "extra": "mean: 162.77943834148604 usec\nrounds: 6525"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6887.462918930193,
            "unit": "iter/sec",
            "range": "stddev: 0.00006903612677861432",
            "extra": "mean: 145.19134429769485 usec\nrounds: 7425"
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
          "id": "0dcee9d120a52720429cda218827ad5a250134e1",
          "message": "[Tripy] Return the shape immediately if it is statically known instead of producing a trace operator.",
          "timestamp": "2024-11-26T20:57:25Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/379/commits/0dcee9d120a52720429cda218827ad5a250134e1"
        },
        "date": 1732661904027,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6097.84997577812,
            "unit": "iter/sec",
            "range": "stddev: 0.00003461725631390008",
            "extra": "mean: 163.9922274198611 usec\nrounds: 6519"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6666.943028532187,
            "unit": "iter/sec",
            "range": "stddev: 0.00007023642982195218",
            "extra": "mean: 149.99378211578372 usec\nrounds: 7351"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 6039.621211052994,
            "unit": "iter/sec",
            "range": "stddev: 0.0017619480848306923",
            "extra": "mean: 165.57329757202643 usec\nrounds: 7620"
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
          "id": "901c72e099105f75b563aa0a441ffba44f29a2d5",
          "message": "[Tripy] Add tripy for setup tools to pass L1 test",
          "timestamp": "2024-11-26T22:31:32Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/413/commits/901c72e099105f75b563aa0a441ffba44f29a2d5"
        },
        "date": 1732662230214,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5959.0230533275935,
            "unit": "iter/sec",
            "range": "stddev: 0.00009613419280756623",
            "extra": "mean: 167.81274229868725 usec\nrounds: 6416"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6450.442253615068,
            "unit": "iter/sec",
            "range": "stddev: 0.00009464041792690496",
            "extra": "mean: 155.02812996109884 usec\nrounds: 7135"
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
          "id": "a8fb924bd11d56b8a9a2d2e999ba41ceee82e16d",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-27T00:20:35Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/a8fb924bd11d56b8a9a2d2e999ba41ceee82e16d"
        },
        "date": 1732670937719,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6176.824053432418,
            "unit": "iter/sec",
            "range": "stddev: 0.00003457696433121466",
            "extra": "mean: 161.89549699805795 usec\nrounds: 6547"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6907.672001368799,
            "unit": "iter/sec",
            "range": "stddev: 0.00007416033347580836",
            "extra": "mean: 144.76657255901029 usec\nrounds: 7540"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 6124.4179041346,
            "unit": "iter/sec",
            "range": "stddev: 0.0017789780834857544",
            "extra": "mean: 163.28082368854993 usec\nrounds: 7567"
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
          "id": "2386beaf1fa7c2a257560d76a637d9b28b648efa",
          "message": "Revert \"[Tripy] Add tripy for setup tools to pass L1 test (#413)\"",
          "timestamp": "2024-11-27T00:20:35Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/415/commits/2386beaf1fa7c2a257560d76a637d9b28b648efa"
        },
        "date": 1732671325794,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6140.721356876769,
            "unit": "iter/sec",
            "range": "stddev: 0.00003210359517190384",
            "extra": "mean: 162.84731742145186 usec\nrounds: 6507"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6854.140628997753,
            "unit": "iter/sec",
            "range": "stddev: 0.00007678444559287144",
            "extra": "mean: 145.8972107705682 usec\nrounds: 7525"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 7029.863983935583,
            "unit": "iter/sec",
            "range": "stddev: 0.000020644051423066266",
            "extra": "mean: 142.25026291905044 usec\nrounds: 7495"
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
          "id": "fcedf59d89d4471bc65cd57a87894c4a5ca58157",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-27T00:20:35Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/fcedf59d89d4471bc65cd57a87894c4a5ca58157"
        },
        "date": 1732674896451,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6154.21458156012,
            "unit": "iter/sec",
            "range": "stddev: 0.00003305242661600221",
            "extra": "mean: 162.4902717881013 usec\nrounds: 6527"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6696.78635984711,
            "unit": "iter/sec",
            "range": "stddev: 0.00006965467492829088",
            "extra": "mean: 149.32535491886742 usec\nrounds: 7481"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 7000.741408834906,
            "unit": "iter/sec",
            "range": "stddev: 0.000021111300118485194",
            "extra": "mean: 142.8420136670102 usec\nrounds: 7621"
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
          "id": "742e97dba1e5a5f5112d8d4f05e0eb19f2be5dca",
          "message": "[Tripy] Change function registry to also register methods of registered classes",
          "timestamp": "2024-11-27T02:31:56Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/388/commits/742e97dba1e5a5f5112d8d4f05e0eb19f2be5dca"
        },
        "date": 1732682186649,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6210.957764735892,
            "unit": "iter/sec",
            "range": "stddev: 0.000031803155119696595",
            "extra": "mean: 161.00576398663097 usec\nrounds: 6569"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6839.4212247034175,
            "unit": "iter/sec",
            "range": "stddev: 0.00007015858380973088",
            "extra": "mean: 146.21120225613296 usec\nrounds: 7494"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 5620.687838033026,
            "unit": "iter/sec",
            "range": "stddev: 0.0000254833208739137",
            "extra": "mean: 177.91416794816212 usec\nrounds: 6122"
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
          "id": "bbb8edfa50118dc642cad478321fa5be51b7a82c",
          "message": "[Tripy] Change function registry to also register methods of registered classes",
          "timestamp": "2024-11-27T02:31:56Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/388/commits/bbb8edfa50118dc642cad478321fa5be51b7a82c"
        },
        "date": 1732684089096,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6158.326707773475,
            "unit": "iter/sec",
            "range": "stddev: 0.00003249396055989394",
            "extra": "mean: 162.38177145388036 usec\nrounds: 6545"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6742.229762502721,
            "unit": "iter/sec",
            "range": "stddev: 0.00007023341500385606",
            "extra": "mean: 148.31888488309232 usec\nrounds: 7511"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21941.294821521715,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025223892710202686",
            "extra": "mean: 45.576161668413604 usec\nrounds: 23013"
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
          "id": "a27c75362d4f70df1ed93de7cd122f4b8510f9bf",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-27T17:38:50Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/a27c75362d4f70df1ed93de7cd122f4b8510f9bf"
        },
        "date": 1732730882031,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6124.710242152573,
            "unit": "iter/sec",
            "range": "stddev: 0.00003575132413269831",
            "extra": "mean: 163.27303014559314 usec\nrounds: 6558"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6907.982533983466,
            "unit": "iter/sec",
            "range": "stddev: 0.00007179481219373558",
            "extra": "mean: 144.760064907598 usec\nrounds: 7516"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 7011.1885267878115,
            "unit": "iter/sec",
            "range": "stddev: 0.000021850626854627138",
            "extra": "mean: 142.6291699587419 usec\nrounds: 7656"
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
          "id": "79e1f01dcfed5f45bcb3b9801fdbfa70277b9bc4",
          "message": "SAM2: image pipeline",
          "timestamp": "2024-11-27T20:27:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/358/commits/79e1f01dcfed5f45bcb3b9801fdbfa70277b9bc4"
        },
        "date": 1732741533525,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6179.1204951422,
            "unit": "iter/sec",
            "range": "stddev: 0.00003185807284425833",
            "extra": "mean: 161.83532928127292 usec\nrounds: 6545"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6907.361264136902,
            "unit": "iter/sec",
            "range": "stddev: 0.00007135536623438171",
            "extra": "mean: 144.77308508417119 usec\nrounds: 7498"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21965.108057397727,
            "unit": "iter/sec",
            "range": "stddev: 0.000002180214644657418",
            "extra": "mean: 45.526750762476006 usec\nrounds: 22695"
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
          "id": "beab289e910b6264f88686a940ccb98a2d29a5e1",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-27T20:27:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/beab289e910b6264f88686a940ccb98a2d29a5e1"
        },
        "date": 1732746133169,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6198.856796568265,
            "unit": "iter/sec",
            "range": "stddev: 0.00003463954598975608",
            "extra": "mean: 161.3200680089283 usec\nrounds: 6582"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6696.32768245108,
            "unit": "iter/sec",
            "range": "stddev: 0.00007060838939788801",
            "extra": "mean: 149.33558323626815 usec\nrounds: 7388"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22382.694930472466,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020293554346285195",
            "extra": "mean: 44.6773725463492 usec\nrounds: 23637"
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
          "id": "2c8d4cbb05c5cfa3fbe0bbbdd27abd392b9e7c15",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-27T20:27:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/2c8d4cbb05c5cfa3fbe0bbbdd27abd392b9e7c15"
        },
        "date": 1732749631546,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6247.116111692672,
            "unit": "iter/sec",
            "range": "stddev: 0.00003112282471291657",
            "extra": "mean: 160.0738616220545 usec\nrounds: 6545"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 7004.628306145341,
            "unit": "iter/sec",
            "range": "stddev: 0.00007198762747179018",
            "extra": "mean: 142.76275004095135 usec\nrounds: 7553"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22285.541370926083,
            "unit": "iter/sec",
            "range": "stddev: 0.000002327752615823869",
            "extra": "mean: 44.87214303461387 usec\nrounds: 23669"
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
          "id": "011444bfd8c45f0be9de0f6ed1db762eb480c693",
          "message": "Removes Parameter API",
          "timestamp": "2024-11-27T23:53:28Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/417/commits/011444bfd8c45f0be9de0f6ed1db762eb480c693"
        },
        "date": 1732752542315,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6109.509935423799,
            "unit": "iter/sec",
            "range": "stddev: 0.00003383541645801625",
            "extra": "mean: 163.67924932928895 usec\nrounds: 6462"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6629.14193485815,
            "unit": "iter/sec",
            "range": "stddev: 0.00007404768728920906",
            "extra": "mean: 150.8490857227962 usec\nrounds: 7351"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22320.651232212018,
            "unit": "iter/sec",
            "range": "stddev: 0.000002415672591325962",
            "extra": "mean: 44.80156020523502 usec\nrounds: 23248"
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
          "id": "349dd19f7296cc9705b77a1a70a35482f3233c83",
          "message": "[Tripy] Resnet50 example",
          "timestamp": "2024-11-27T23:53:28Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/322/commits/349dd19f7296cc9705b77a1a70a35482f3233c83"
        },
        "date": 1732753057929,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6174.223285677285,
            "unit": "iter/sec",
            "range": "stddev: 0.00003169898226488948",
            "extra": "mean: 161.9636922298809 usec\nrounds: 6568"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6929.054746158128,
            "unit": "iter/sec",
            "range": "stddev: 0.00007160184921183427",
            "extra": "mean: 144.3198295632544 usec\nrounds: 7655"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21987.66670119674,
            "unit": "iter/sec",
            "range": "stddev: 0.000002352948677552058",
            "extra": "mean: 45.48004177021531 usec\nrounds: 22868"
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
          "id": "c338036df33d58b152131d5c324fad9b4376ee62",
          "message": "Removes Parameter API",
          "timestamp": "2024-11-27T23:53:28Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/417/commits/c338036df33d58b152131d5c324fad9b4376ee62"
        },
        "date": 1732754211590,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6226.74007887881,
            "unit": "iter/sec",
            "range": "stddev: 0.00003257947106206366",
            "extra": "mean: 160.59767829269347 usec\nrounds: 6547"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6937.592953603697,
            "unit": "iter/sec",
            "range": "stddev: 0.00007356689696780417",
            "extra": "mean: 144.14221282333307 usec\nrounds: 7507"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22091.36103180903,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021169873682067533",
            "extra": "mean: 45.266563638162204 usec\nrounds: 23283"
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
          "id": "9478a246152189a5268db44719993f4c6fbb13d3",
          "message": "SAM2: image pipeline",
          "timestamp": "2024-11-28T00:32:43Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/358/commits/9478a246152189a5268db44719993f4c6fbb13d3"
        },
        "date": 1733156335092,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6087.878653685483,
            "unit": "iter/sec",
            "range": "stddev: 0.000032821284405722276",
            "extra": "mean: 164.2608299024849 usec\nrounds: 6499"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6835.0116981351875,
            "unit": "iter/sec",
            "range": "stddev: 0.00006898858173388671",
            "extra": "mean: 146.30552867566158 usec\nrounds: 7427"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21374.89300310812,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023883194021897836",
            "extra": "mean: 46.78385991708076 usec\nrounds: 22471"
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
          "id": "cb10dbbaab4d91381a6ebe3b0d9b079da6836682",
          "message": "Removes Parameter API",
          "timestamp": "2024-11-28T00:32:43Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/417/commits/cb10dbbaab4d91381a6ebe3b0d9b079da6836682"
        },
        "date": 1733163640856,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6119.232997887716,
            "unit": "iter/sec",
            "range": "stddev: 0.00003422918703014509",
            "extra": "mean: 163.41917366852803 usec\nrounds: 6516"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6650.310877593449,
            "unit": "iter/sec",
            "range": "stddev: 0.00007663811624729523",
            "extra": "mean: 150.3689103270719 usec\nrounds: 7329"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22314.37252790056,
            "unit": "iter/sec",
            "range": "stddev: 0.000002238514418492257",
            "extra": "mean: 44.81416623970312 usec\nrounds: 23182"
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
          "id": "02f1451576ae3de1b8194f381db8b05a277217a6",
          "message": "Removes Parameter API",
          "timestamp": "2024-11-28T00:32:43Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/417/commits/02f1451576ae3de1b8194f381db8b05a277217a6"
        },
        "date": 1733163908906,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6128.214613643735,
            "unit": "iter/sec",
            "range": "stddev: 0.00003382335782940885",
            "extra": "mean: 163.1796637431104 usec\nrounds: 6486"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6585.543195667495,
            "unit": "iter/sec",
            "range": "stddev: 0.0000739877566112119",
            "extra": "mean: 151.84776263526464 usec\nrounds: 7244"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22751.93383921299,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021071729959029413",
            "extra": "mean: 43.952307837520976 usec\nrounds: 23709"
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
          "id": "f78c57fea91fa6e3cb09483fcbf549061bf9ffd9",
          "message": "SAM2: image pipeline",
          "timestamp": "2024-12-02T21:10:54Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/358/commits/f78c57fea91fa6e3cb09483fcbf549061bf9ffd9"
        },
        "date": 1733181222616,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6116.397093055047,
            "unit": "iter/sec",
            "range": "stddev: 0.00003323666766359829",
            "extra": "mean: 163.49494396553564 usec\nrounds: 6445"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6739.1006069416835,
            "unit": "iter/sec",
            "range": "stddev: 0.00007384344936802804",
            "extra": "mean: 148.3877535482909 usec\nrounds: 7302"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22159.680432473197,
            "unit": "iter/sec",
            "range": "stddev: 0.000002520470207795771",
            "extra": "mean: 45.12700456341337 usec\nrounds: 23385"
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
          "id": "ebc1aaba005b1cab15da4f7bff25c4ed1e9b2225",
          "message": "SAM2: image pipeline",
          "timestamp": "2024-12-02T21:10:54Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/358/commits/ebc1aaba005b1cab15da4f7bff25c4ed1e9b2225"
        },
        "date": 1733200935454,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6108.95334614742,
            "unit": "iter/sec",
            "range": "stddev: 0.000032711759560225594",
            "extra": "mean: 163.69416221367035 usec\nrounds: 6457"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6727.714444248002,
            "unit": "iter/sec",
            "range": "stddev: 0.00006924207309266992",
            "extra": "mean: 148.63888892534234 usec\nrounds: 7348"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22198.465642146068,
            "unit": "iter/sec",
            "range": "stddev: 0.00000204794911270408",
            "extra": "mean: 45.04815855837339 usec\nrounds: 23407"
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
          "id": "4a99068ead6df79466e0e39c7fb7994a53f11ffd",
          "message": "Combine the functionality of the `convert_to_tensors` decorator and the `dtypes` constraint.",
          "timestamp": "2024-12-02T21:10:54Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/420/commits/4a99068ead6df79466e0e39c7fb7994a53f11ffd"
        },
        "date": 1733215188299,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6148.682408425366,
            "unit": "iter/sec",
            "range": "stddev: 0.00003259917750632441",
            "extra": "mean: 162.6364696653918 usec\nrounds: 6496"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6708.094990082307,
            "unit": "iter/sec",
            "range": "stddev: 0.00007010937682825735",
            "extra": "mean: 149.07361948190453 usec\nrounds: 7329"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22285.487061955275,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021467666799508176",
            "extra": "mean: 44.872252386493834 usec\nrounds: 23477"
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
          "id": "f1bda90c6656a1c302294ae8e8280539b9665957",
          "message": "Fix get_stack_info for python3.12",
          "timestamp": "2024-12-03T17:40:01Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/421/commits/f1bda90c6656a1c302294ae8e8280539b9665957"
        },
        "date": 1733252506497,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6087.637955493162,
            "unit": "iter/sec",
            "range": "stddev: 0.00003346180114428679",
            "extra": "mean: 164.26732458648482 usec\nrounds: 6504"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6775.541439250639,
            "unit": "iter/sec",
            "range": "stddev: 0.00006983097610802722",
            "extra": "mean: 147.58968105589477 usec\nrounds: 7406"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22562.10362556317,
            "unit": "iter/sec",
            "range": "stddev: 0.000002665423325995152",
            "extra": "mean: 44.32210828368799 usec\nrounds: 23706"
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
          "id": "570d0818893bcac6bf14dcb5624b1b1b2d9c316d",
          "message": "Fix get_stack_info for python3.12",
          "timestamp": "2024-12-03T17:40:01Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/421/commits/570d0818893bcac6bf14dcb5624b1b1b2d9c316d"
        },
        "date": 1733253330147,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6144.775214280892,
            "unit": "iter/sec",
            "range": "stddev: 0.00003272643658016002",
            "extra": "mean: 162.73988309221292 usec\nrounds: 6526"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6813.818413945756,
            "unit": "iter/sec",
            "range": "stddev: 0.00006836340992257972",
            "extra": "mean: 146.76058844675296 usec\nrounds: 7471"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22911.961805385876,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025277716323819273",
            "extra": "mean: 43.64532415399417 usec\nrounds: 23951"
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
          "id": "2d8e99a1d0e4decab5e069814de27f36cdda12d5",
          "message": "Fix get_stack_info for python3.12",
          "timestamp": "2024-12-03T17:40:01Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/421/commits/2d8e99a1d0e4decab5e069814de27f36cdda12d5"
        },
        "date": 1733253761708,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6024.557350319133,
            "unit": "iter/sec",
            "range": "stddev: 0.000031544329062313966",
            "extra": "mean: 165.98729862651703 usec\nrounds: 6370"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6599.775232912501,
            "unit": "iter/sec",
            "range": "stddev: 0.00006738635077593927",
            "extra": "mean: 151.52031163320345 usec\nrounds: 7143"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22660.86480172364,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021571783191777317",
            "extra": "mean: 44.1289425072576 usec\nrounds: 23619"
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
          "id": "6762f238b7f5757930d63bc481b593fac4077e07",
          "message": "Fix get_stack_info for python3.12",
          "timestamp": "2024-12-03T17:40:01Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/421/commits/6762f238b7f5757930d63bc481b593fac4077e07"
        },
        "date": 1733254515842,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6189.9941793688495,
            "unit": "iter/sec",
            "range": "stddev: 0.00003245995603102282",
            "extra": "mean: 161.55104044087534 usec\nrounds: 6533"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6763.433547692884,
            "unit": "iter/sec",
            "range": "stddev: 0.00006922745852476967",
            "extra": "mean: 147.8538959462559 usec\nrounds: 7442"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22201.718357871076,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023738332444785566",
            "extra": "mean: 45.04155867041141 usec\nrounds: 23030"
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
          "id": "233c5f39701dbc2ab2f0ada7b827328c937435ef",
          "message": "Allows `DimensionSize`s to be used in `InputInfo`",
          "timestamp": "2024-12-03T17:40:01Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/422/commits/233c5f39701dbc2ab2f0ada7b827328c937435ef"
        },
        "date": 1733258412799,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6130.026066162811,
            "unit": "iter/sec",
            "range": "stddev: 0.00009759280916929351",
            "extra": "mean: 163.13144335876638 usec\nrounds: 6462"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6859.709793514103,
            "unit": "iter/sec",
            "range": "stddev: 0.00008958957405475076",
            "extra": "mean: 145.77876179915162 usec\nrounds: 7372"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22278.340373497285,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026050008923371304",
            "extra": "mean: 44.88664699591439 usec\nrounds: 23555"
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
          "id": "3e8084c6d1b0eb7d65985c090da14d573324c284",
          "message": "Combine the functionality of the `convert_to_tensors` decorator and the `dtypes` constraint.",
          "timestamp": "2024-12-04T00:12:11Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/420/commits/3e8084c6d1b0eb7d65985c090da14d573324c284"
        },
        "date": 1733301103595,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6211.24039251772,
            "unit": "iter/sec",
            "range": "stddev: 0.00003194373287358352",
            "extra": "mean: 160.99843780070654 usec\nrounds: 6544"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6800.650649403317,
            "unit": "iter/sec",
            "range": "stddev: 0.00007254807809353988",
            "extra": "mean: 147.04475373804695 usec\nrounds: 7518"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21876.15173489279,
            "unit": "iter/sec",
            "range": "stddev: 0.000004059469193270545",
            "extra": "mean: 45.71187895012563 usec\nrounds: 23359"
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
          "id": "f4c2068ca1fc967798fb6f80b64f8f4d13728a1a",
          "message": "Combine the functionality of the `convert_to_tensors` decorator and the `dtypes` constraint.",
          "timestamp": "2024-12-04T00:12:11Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/420/commits/f4c2068ca1fc967798fb6f80b64f8f4d13728a1a"
        },
        "date": 1733335905108,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6245.153569614931,
            "unit": "iter/sec",
            "range": "stddev: 0.000032321141402095055",
            "extra": "mean: 160.12416489890398 usec\nrounds: 6558"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 7011.096789959951,
            "unit": "iter/sec",
            "range": "stddev: 0.00007328737707647484",
            "extra": "mean: 142.63103619279977 usec\nrounds: 7678"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22547.902830948336,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018541942004657274",
            "extra": "mean: 44.35002259400553 usec\nrounds: 23892"
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
          "id": "f046a23b38a0be6e9b659b98bf57430ed0718714",
          "message": "Combine the functionality of the `convert_to_tensors` decorator and the `dtypes` constraint.",
          "timestamp": "2024-12-04T00:12:11Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/420/commits/f046a23b38a0be6e9b659b98bf57430ed0718714"
        },
        "date": 1733369990353,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6144.078671323437,
            "unit": "iter/sec",
            "range": "stddev: 0.00003279999851213242",
            "extra": "mean: 162.75833261500208 usec\nrounds: 6515"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6919.213954465175,
            "unit": "iter/sec",
            "range": "stddev: 0.00007025404898751359",
            "extra": "mean: 144.52508718200136 usec\nrounds: 7577"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22556.522937117254,
            "unit": "iter/sec",
            "range": "stddev: 0.00000253903221164618",
            "extra": "mean: 44.33307397544318 usec\nrounds: 23392"
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
          "id": "e80787a1a33ff07f248baa1f4146ef8c7f7cafc6",
          "message": "Combine the functionality of the `convert_to_tensors` decorator and the `dtypes` constraint.",
          "timestamp": "2024-12-05T19:18:15Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/420/commits/e80787a1a33ff07f248baa1f4146ef8c7f7cafc6"
        },
        "date": 1733455240439,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6149.901992692187,
            "unit": "iter/sec",
            "range": "stddev: 0.00003197235431882706",
            "extra": "mean: 162.60421730106938 usec\nrounds: 6533"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6837.835105370186,
            "unit": "iter/sec",
            "range": "stddev: 0.00007057414517182246",
            "extra": "mean: 146.2451177295335 usec\nrounds: 7435"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22706.69702613586,
            "unit": "iter/sec",
            "range": "stddev: 0.00000198669213665841",
            "extra": "mean: 44.03987065353363 usec\nrounds: 23797"
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
          "id": "18aa6a3d741e27a0ad9db83f554467042fd2806b",
          "message": "SAM2: image pipeline",
          "timestamp": "2024-12-05T19:18:15Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/358/commits/18aa6a3d741e27a0ad9db83f554467042fd2806b"
        },
        "date": 1733497887804,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6180.695707729578,
            "unit": "iter/sec",
            "range": "stddev: 0.0000323993988691048",
            "extra": "mean: 161.79408391670214 usec\nrounds: 6506"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6737.739015369252,
            "unit": "iter/sec",
            "range": "stddev: 0.00007133735421484864",
            "extra": "mean: 148.41774039020066 usec\nrounds: 7361"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22372.753904262878,
            "unit": "iter/sec",
            "range": "stddev: 0.000002360894133958137",
            "extra": "mean: 44.697224323799546 usec\nrounds: 23052"
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
          "id": "3f8e12ad9cb92541ffc8977859eda7f40c51e4ae",
          "message": "SAM2: image pipeline",
          "timestamp": "2024-12-05T19:18:15Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/358/commits/3f8e12ad9cb92541ffc8977859eda7f40c51e4ae"
        },
        "date": 1733500002201,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6097.123468637626,
            "unit": "iter/sec",
            "range": "stddev: 0.000033427134362244275",
            "extra": "mean: 164.01176803189216 usec\nrounds: 6533"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6785.129294429432,
            "unit": "iter/sec",
            "range": "stddev: 0.00006960077610533517",
            "extra": "mean: 147.38112666784357 usec\nrounds: 7440"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22987.929231469898,
            "unit": "iter/sec",
            "range": "stddev: 0.000002223468957588698",
            "extra": "mean: 43.50109093911013 usec\nrounds: 23832"
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
          "id": "dda9249eb5d2e1a277bcb53ba45d91257563a3c3",
          "message": "SAM2: image pipeline",
          "timestamp": "2024-12-05T19:18:15Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/358/commits/dda9249eb5d2e1a277bcb53ba45d91257563a3c3"
        },
        "date": 1733501292597,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6117.9994533014105,
            "unit": "iter/sec",
            "range": "stddev: 0.000032685031790814647",
            "extra": "mean: 163.45212313812442 usec\nrounds: 6491"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6829.196856358746,
            "unit": "iter/sec",
            "range": "stddev: 0.00006970951939314299",
            "extra": "mean: 146.4301031341465 usec\nrounds: 7419"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22153.971525978595,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021714034031563905",
            "extra": "mean: 45.13863344219621 usec\nrounds: 23256"
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
          "id": "9d3ea4edb2bf855bf320a2ca77c12f9385761c7c",
          "message": "Fix weight loader for nanogpt by not asserting on exactly same state …",
          "timestamp": "2024-12-05T19:18:15Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/428/commits/9d3ea4edb2bf855bf320a2ca77c12f9385761c7c"
        },
        "date": 1733505461189,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6027.4491756331945,
            "unit": "iter/sec",
            "range": "stddev: 0.00003410529269679822",
            "extra": "mean: 165.9076619082314 usec\nrounds: 6463"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6744.355634997746,
            "unit": "iter/sec",
            "range": "stddev: 0.00007393515804248291",
            "extra": "mean: 148.27213363583758 usec\nrounds: 7345"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21604.14240184177,
            "unit": "iter/sec",
            "range": "stddev: 0.000002571126746902297",
            "extra": "mean: 46.287419393919066 usec\nrounds: 23059"
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
          "id": "a5150328ae9a5766fbfe3ce638d344a51598e908",
          "message": "Fix weight loader for nanogpt by not asserting on exactly same state …",
          "timestamp": "2024-12-05T19:18:15Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/428/commits/a5150328ae9a5766fbfe3ce638d344a51598e908"
        },
        "date": 1733513171955,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6191.275990961083,
            "unit": "iter/sec",
            "range": "stddev: 0.00009275965784287219",
            "extra": "mean: 161.51759370119248 usec\nrounds: 6534"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6861.646301947599,
            "unit": "iter/sec",
            "range": "stddev: 0.00008574136733989484",
            "extra": "mean: 145.73761980651227 usec\nrounds: 7611"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22948.34237685874,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025086933865775928",
            "extra": "mean: 43.57613214836844 usec\nrounds: 23918"
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
          "id": "a92c2fa523c0d142cb4bb1e3f9ebd88bdfa87fac",
          "message": "Fixes type checking, nanoGPT",
          "timestamp": "2024-12-05T19:18:15Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/429/commits/a92c2fa523c0d142cb4bb1e3f9ebd88bdfa87fac"
        },
        "date": 1733514324324,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6085.26228021505,
            "unit": "iter/sec",
            "range": "stddev: 0.00003252624787606345",
            "extra": "mean: 164.3314542499326 usec\nrounds: 6477"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6879.620208472937,
            "unit": "iter/sec",
            "range": "stddev: 0.000068843617808557",
            "extra": "mean: 145.35686123608986 usec\nrounds: 7455"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22609.406149377322,
            "unit": "iter/sec",
            "range": "stddev: 0.000002583748850509184",
            "extra": "mean: 44.22937928546791 usec\nrounds: 23380"
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
          "id": "4c42bcadb469b427259e6efa1a900ba97b72cc05",
          "message": "Fixes type checking, nanoGPT",
          "timestamp": "2024-12-05T19:18:15Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/429/commits/4c42bcadb469b427259e6efa1a900ba97b72cc05"
        },
        "date": 1733517854434,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6169.054733438091,
            "unit": "iter/sec",
            "range": "stddev: 0.000031842109433090225",
            "extra": "mean: 162.09938851404672 usec\nrounds: 6529"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6654.4383625818045,
            "unit": "iter/sec",
            "range": "stddev: 0.00006865579164199413",
            "extra": "mean: 150.2756424378417 usec\nrounds: 7382"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22933.297021157236,
            "unit": "iter/sec",
            "range": "stddev: 0.000002472705162830085",
            "extra": "mean: 43.60472020562262 usec\nrounds: 23991"
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
          "id": "7b8dec66aa631dc07e270001d88f79ff33c6fd94",
          "message": "Fixes type checking, nanoGPT",
          "timestamp": "2024-12-06T20:45:59Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/429/commits/7b8dec66aa631dc07e270001d88f79ff33c6fd94"
        },
        "date": 1733518362771,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6154.445910049964,
            "unit": "iter/sec",
            "range": "stddev: 0.000033050319557346374",
            "extra": "mean: 162.48416423110322 usec\nrounds: 6461"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6739.622927930387,
            "unit": "iter/sec",
            "range": "stddev: 0.00007069609785257781",
            "extra": "mean: 148.3762534927279 usec\nrounds: 7315"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22215.56152031996,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026006400429746015",
            "extra": "mean: 45.01349196532024 usec\nrounds: 23166"
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
          "id": "a0028f5ec7a6eca77d256454d18ea56fe8d986e0",
          "message": "Upgrade tripy version to 0.0.6",
          "timestamp": "2024-12-06T23:41:10Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/424/commits/a0028f5ec7a6eca77d256454d18ea56fe8d986e0"
        },
        "date": 1733531643305,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6108.698915914094,
            "unit": "iter/sec",
            "range": "stddev: 0.000032035580680909045",
            "extra": "mean: 163.70098015386668 usec\nrounds: 6476"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6775.8583487476035,
            "unit": "iter/sec",
            "range": "stddev: 0.0000705360875815992",
            "extra": "mean: 147.58277822983595 usec\nrounds: 7395"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22631.613200654447,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026991408754619072",
            "extra": "mean: 44.185979635383774 usec\nrounds: 23658"
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
          "id": "9ea5a4fd78c5950ae4c9be518f8c500c4cedeb88",
          "message": "Update mlir-tensorrt version to 0.1.38",
          "timestamp": "2024-12-09T17:59:20Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/431/commits/9ea5a4fd78c5950ae4c9be518f8c500c4cedeb88"
        },
        "date": 1733771481240,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6058.816163906684,
            "unit": "iter/sec",
            "range": "stddev: 0.00003222634277892916",
            "extra": "mean: 165.04874433344858 usec\nrounds: 6366"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6718.081788981925,
            "unit": "iter/sec",
            "range": "stddev: 0.00006804606161474162",
            "extra": "mean: 148.85201332917126 usec\nrounds: 7245"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22545.22848533403,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019306096069527457",
            "extra": "mean: 44.355283453903034 usec\nrounds: 23650"
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
          "id": "dffb097facb6be1e34f8b92fd0958e8e99ed9255",
          "message": "Replace tp.Parameter with tp.Tensor in resnet notebook",
          "timestamp": "2024-12-09T19:13:10Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/433/commits/dffb097facb6be1e34f8b92fd0958e8e99ed9255"
        },
        "date": 1733783797566,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6134.668586050505,
            "unit": "iter/sec",
            "range": "stddev: 0.000032661339945669905",
            "extra": "mean: 163.00799072893346 usec\nrounds: 6462"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6761.161031391094,
            "unit": "iter/sec",
            "range": "stddev: 0.0000695184935252952",
            "extra": "mean: 147.90359161054505 usec\nrounds: 7343"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22855.55644333644,
            "unit": "iter/sec",
            "range": "stddev: 0.000002102158860114655",
            "extra": "mean: 43.75303670594075 usec\nrounds: 23769"
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
          "id": "efd985add03589d18e80d66ef8909631cc0de6f6",
          "message": "Combine the functionality of the `convert_to_tensors` decorator and the `dtypes` constraint.",
          "timestamp": "2024-12-09T23:17:43Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/420/commits/efd985add03589d18e80d66ef8909631cc0de6f6"
        },
        "date": 1733818095654,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6105.963960007558,
            "unit": "iter/sec",
            "range": "stddev: 0.00003208040026028898",
            "extra": "mean: 163.7743043604146 usec\nrounds: 6452"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6852.824639131749,
            "unit": "iter/sec",
            "range": "stddev: 0.00007091255195744412",
            "extra": "mean: 145.92522830508324 usec\nrounds: 7377"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22775.13192460843,
            "unit": "iter/sec",
            "range": "stddev: 0.000002119776629192323",
            "extra": "mean: 43.90753929813703 usec\nrounds: 23625"
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
          "id": "efb5600e2010128e4bd56c37ab43602f5bddbbf5",
          "message": "Adds explicit notebook tests, updates CI",
          "timestamp": "2024-12-10T09:08:51Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/436/commits/efb5600e2010128e4bd56c37ab43602f5bddbbf5"
        },
        "date": 1733856499882,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6167.388444508928,
            "unit": "iter/sec",
            "range": "stddev: 0.000033015939627757425",
            "extra": "mean: 162.14318410417945 usec\nrounds: 6477"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6930.544420346633,
            "unit": "iter/sec",
            "range": "stddev: 0.00006921117643032442",
            "extra": "mean: 144.28880898075028 usec\nrounds: 7409"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22620.386390591295,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023861349091889466",
            "extra": "mean: 44.207909747109326 usec\nrounds: 23751"
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
          "id": "03a70502b4a30b8e68a2017b40d05738d2ac80f0",
          "message": "Adds explicit notebook tests, updates CI",
          "timestamp": "2024-12-10T09:08:51Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/436/commits/03a70502b4a30b8e68a2017b40d05738d2ac80f0"
        },
        "date": 1733857302360,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6130.04521436906,
            "unit": "iter/sec",
            "range": "stddev: 0.000031824972752848946",
            "extra": "mean: 163.13093379082454 usec\nrounds: 6485"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6640.914068486239,
            "unit": "iter/sec",
            "range": "stddev: 0.00007041540111943775",
            "extra": "mean: 150.5816804264032 usec\nrounds: 7286"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22493.189264991244,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025146781790019024",
            "extra": "mean: 44.457901821704574 usec\nrounds: 23437"
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
          "id": "8548438774e4e2e28df3d60769fb52d14d563b9b",
          "message": "Adds explicit notebook tests, updates CI",
          "timestamp": "2024-12-10T09:08:51Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/436/commits/8548438774e4e2e28df3d60769fb52d14d563b9b"
        },
        "date": 1733857562566,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6171.2601813102065,
            "unit": "iter/sec",
            "range": "stddev: 0.00003251333901104617",
            "extra": "mean: 162.04145840885488 usec\nrounds: 6517"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6897.3168619492335,
            "unit": "iter/sec",
            "range": "stddev: 0.0000696269184299295",
            "extra": "mean: 144.98391476209383 usec\nrounds: 7479"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 23150.90103393734,
            "unit": "iter/sec",
            "range": "stddev: 0.000003256980575882889",
            "extra": "mean: 43.19486306533302 usec\nrounds: 23967"
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
          "id": "82a0c12444762c2b990f2ff77bd950e89ddd194b",
          "message": "Adds explicit notebook tests, updates CI",
          "timestamp": "2024-12-10T09:08:51Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/436/commits/82a0c12444762c2b990f2ff77bd950e89ddd194b"
        },
        "date": 1733863791520,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6105.469305983342,
            "unit": "iter/sec",
            "range": "stddev: 0.00003247261413181957",
            "extra": "mean: 163.7875730568333 usec\nrounds: 6487"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6725.762451549135,
            "unit": "iter/sec",
            "range": "stddev: 0.00007001111100139524",
            "extra": "mean: 148.6820278301193 usec\nrounds: 7382"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22612.015571459262,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024126112511235836",
            "extra": "mean: 44.224275223929766 usec\nrounds: 23580"
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
          "id": "a7361608dcd648a9faa8c48414eb93bb3a488ff1",
          "message": "Adds explicit notebook tests, updates CI",
          "timestamp": "2024-12-10T09:08:51Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/436/commits/a7361608dcd648a9faa8c48414eb93bb3a488ff1"
        },
        "date": 1733865904286,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6090.906613059506,
            "unit": "iter/sec",
            "range": "stddev: 0.000033110279363100707",
            "extra": "mean: 164.17917126752545 usec\nrounds: 6469"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6724.5613130467445,
            "unit": "iter/sec",
            "range": "stddev: 0.00007619778192478129",
            "extra": "mean: 148.7085853555736 usec\nrounds: 7358"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22360.050432273663,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023448981335929",
            "extra": "mean: 44.72261827087103 usec\nrounds: 23624"
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
          "id": "2ed86f79bbc3024e1735173ea2fb2cc6c26224c8",
          "message": "Adds explicit notebook tests, updates CI",
          "timestamp": "2024-12-10T09:08:51Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/436/commits/2ed86f79bbc3024e1735173ea2fb2cc6c26224c8"
        },
        "date": 1733867041964,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6128.853559267355,
            "unit": "iter/sec",
            "range": "stddev: 0.00003330518059153099",
            "extra": "mean: 163.1626519266256 usec\nrounds: 6450"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6775.671944619598,
            "unit": "iter/sec",
            "range": "stddev: 0.00007099324666586943",
            "extra": "mean: 147.5868383495273 usec\nrounds: 7422"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21992.348363385274,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024349393602632627",
            "extra": "mean: 45.47036012147228 usec\nrounds: 22849"
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
          "id": "a84d00499ab26b0b18fcd04f5bc4d50d78f34366",
          "message": "Remove plugin WAR due to mlir-tensorrt issue #915",
          "timestamp": "2024-12-10T21:45:18Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/438/commits/a84d00499ab26b0b18fcd04f5bc4d50d78f34366"
        },
        "date": 1733935043206,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6171.396128821182,
            "unit": "iter/sec",
            "range": "stddev: 0.00003257028058408008",
            "extra": "mean: 162.0378888546591 usec\nrounds: 6466"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6861.851690791069,
            "unit": "iter/sec",
            "range": "stddev: 0.00007227061226166651",
            "extra": "mean: 145.73325759022853 usec\nrounds: 7511"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22523.75990380411,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020145544734379284",
            "extra": "mean: 44.397560810045164 usec\nrounds: 23570"
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
          "id": "1f4d8acda9210d9f07a1409337e169852b1e8014",
          "message": "Always construct memref value in storage op",
          "timestamp": "2024-12-11T17:36:32Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/439/commits/1f4d8acda9210d9f07a1409337e169852b1e8014"
        },
        "date": 1733940929375,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6182.284317702397,
            "unit": "iter/sec",
            "range": "stddev: 0.00003384463393123629",
            "extra": "mean: 161.75250904210162 usec\nrounds: 6517"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6824.379787695854,
            "unit": "iter/sec",
            "range": "stddev: 0.00007054309321349694",
            "extra": "mean: 146.53346254306788 usec\nrounds: 7485"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21878.96317170391,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025779073775656464",
            "extra": "mean: 45.70600499448261 usec\nrounds: 23087"
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
          "id": "cd05778d1d8f6a27656e1ae378ab0fbd97e5a622",
          "message": "Always construct memref value in storage op",
          "timestamp": "2024-12-11T17:36:32Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/439/commits/cd05778d1d8f6a27656e1ae378ab0fbd97e5a622"
        },
        "date": 1733943962638,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6168.6359361257955,
            "unit": "iter/sec",
            "range": "stddev: 0.00003279577722565196",
            "extra": "mean: 162.11039366801225 usec\nrounds: 6509"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6709.668131752259,
            "unit": "iter/sec",
            "range": "stddev: 0.00007043580386974699",
            "extra": "mean: 149.03866783927592 usec\nrounds: 7365"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22685.06739309137,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021606797181280327",
            "extra": "mean: 44.08186154670827 usec\nrounds: 23632"
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
          "id": "dbeccd0636d00b192a7b65e49c1853e4f7036634",
          "message": "[Tripy] Permit `eval` while tracing, but do not update trace.",
          "timestamp": "2024-12-11T23:44:09Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/443/commits/dbeccd0636d00b192a7b65e49c1853e4f7036634"
        },
        "date": 1733986457056,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6077.2854790873225,
            "unit": "iter/sec",
            "range": "stddev: 0.000033676122917947245",
            "extra": "mean: 164.54714912457567 usec\nrounds: 6396"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6669.520302786347,
            "unit": "iter/sec",
            "range": "stddev: 0.00007204066847923626",
            "extra": "mean: 149.93582065898002 usec\nrounds: 7346"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22176.540655593184,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024607992524534507",
            "extra": "mean: 45.09269572428954 usec\nrounds: 23158"
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
          "id": "d1d4735813ce52d8228711999c4ee6e6609dda37",
          "message": "Enable sam2 video pipeline",
          "timestamp": "2024-12-11T23:44:09Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/444/commits/d1d4735813ce52d8228711999c4ee6e6609dda37"
        },
        "date": 1734021091956,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6117.551544579054,
            "unit": "iter/sec",
            "range": "stddev: 0.000033249716435644215",
            "extra": "mean: 163.4640906109128 usec\nrounds: 6436"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6629.966972624003,
            "unit": "iter/sec",
            "range": "stddev: 0.00007305411817406021",
            "extra": "mean: 150.83031395618264 usec\nrounds: 7283"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22541.347702226256,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021056083503534874",
            "extra": "mean: 44.36291978678972 usec\nrounds: 23512"
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
          "id": "f635a1d7de24e471dc9dd1a83440af32e36bf844",
          "message": "Enable sam2 video pipeline",
          "timestamp": "2024-12-11T23:44:09Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/444/commits/f635a1d7de24e471dc9dd1a83440af32e36bf844"
        },
        "date": 1734022745085,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6146.200138266757,
            "unit": "iter/sec",
            "range": "stddev: 0.00003261223148143564",
            "extra": "mean: 162.70215377041114 usec\nrounds: 6469"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6763.9366468246935,
            "unit": "iter/sec",
            "range": "stddev: 0.00006963798056637759",
            "extra": "mean: 147.84289862759826 usec\nrounds: 7367"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22578.447137680447,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026861436307439436",
            "extra": "mean: 44.290025523107474 usec\nrounds: 23726"
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
          "id": "8d92d7ba9bca0f83c9867fa4ae4eaa9d1c214e98",
          "message": "Enable sam2 video pipeline",
          "timestamp": "2024-12-11T23:44:09Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/444/commits/8d92d7ba9bca0f83c9867fa4ae4eaa9d1c214e98"
        },
        "date": 1734028625908,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6196.491789624677,
            "unit": "iter/sec",
            "range": "stddev: 0.00003433461727814949",
            "extra": "mean: 161.38163882898812 usec\nrounds: 6537"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6813.947584591058,
            "unit": "iter/sec",
            "range": "stddev: 0.00007091311485468496",
            "extra": "mean: 146.75780633554953 usec\nrounds: 7492"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22306.895629180046,
            "unit": "iter/sec",
            "range": "stddev: 0.000002576448771211035",
            "extra": "mean: 44.82918719949011 usec\nrounds: 23162"
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
          "id": "c59c4b349265e28315e671900c54cc49b9425d39",
          "message": "Enable sam2 video pipeline",
          "timestamp": "2024-12-11T23:44:09Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/444/commits/c59c4b349265e28315e671900c54cc49b9425d39"
        },
        "date": 1734039148426,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6002.932811326325,
            "unit": "iter/sec",
            "range": "stddev: 0.00004321580830907167",
            "extra": "mean: 166.58523948713892 usec\nrounds: 6427"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6370.06358014093,
            "unit": "iter/sec",
            "range": "stddev: 0.00008364478610232122",
            "extra": "mean: 156.98430438238674 usec\nrounds: 7271"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21621.214757378584,
            "unit": "iter/sec",
            "range": "stddev: 0.000006457944361183524",
            "extra": "mean: 46.25087032442218 usec\nrounds: 23630"
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
          "id": "6765a6f2409a1029c7ef5b492b3e852cb781321c",
          "message": "Enable sam2 video pipeline",
          "timestamp": "2024-12-11T23:44:09Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/444/commits/6765a6f2409a1029c7ef5b492b3e852cb781321c"
        },
        "date": 1734039715125,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6154.653765605335,
            "unit": "iter/sec",
            "range": "stddev: 0.00003362551781080541",
            "extra": "mean: 162.47867680037496 usec\nrounds: 6466"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6703.200252595214,
            "unit": "iter/sec",
            "range": "stddev: 0.00006875155053651712",
            "extra": "mean: 149.18247438793725 usec\nrounds: 7386"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22664.311021329653,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026066826563463913",
            "extra": "mean: 44.12223248520055 usec\nrounds: 23699"
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
          "id": "9adb1cf500fc8b10b1b373c9b98d0ee91e79c659",
          "message": "Enable sam2 video pipeline",
          "timestamp": "2024-12-11T23:44:09Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/444/commits/9adb1cf500fc8b10b1b373c9b98d0ee91e79c659"
        },
        "date": 1734044430711,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6092.381102347679,
            "unit": "iter/sec",
            "range": "stddev: 0.000033515617367836324",
            "extra": "mean: 164.13943632230314 usec\nrounds: 6507"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6798.4375137224415,
            "unit": "iter/sec",
            "range": "stddev: 0.00007010661883968796",
            "extra": "mean: 147.09262208875643 usec\nrounds: 7385"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22878.779943075864,
            "unit": "iter/sec",
            "range": "stddev: 0.000002139605600660698",
            "extra": "mean: 43.70862443225013 usec\nrounds: 24180"
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
          "id": "cc405ad66b37b5e05d5baabc2c9f2f3a82f99b12",
          "message": "Enable sam2 video pipeline",
          "timestamp": "2024-12-11T23:44:09Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/444/commits/cc405ad66b37b5e05d5baabc2c9f2f3a82f99b12"
        },
        "date": 1734114289922,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6150.2210230431765,
            "unit": "iter/sec",
            "range": "stddev: 0.00003263992001477148",
            "extra": "mean: 162.59578253420108 usec\nrounds: 6479"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6791.501724792081,
            "unit": "iter/sec",
            "range": "stddev: 0.00007104940724529028",
            "extra": "mean: 147.2428397315344 usec\nrounds: 7411"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22200.354445311707,
            "unit": "iter/sec",
            "range": "stddev: 0.000004742635137395679",
            "extra": "mean: 45.044325867111596 usec\nrounds: 23479"
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
          "id": "7e53ba9e40d9fc735a2a1952d6733baf5fc66812",
          "message": "Enable sam2 video pipeline",
          "timestamp": "2024-12-11T23:44:09Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/444/commits/7e53ba9e40d9fc735a2a1952d6733baf5fc66812"
        },
        "date": 1734114551458,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6068.989263356213,
            "unit": "iter/sec",
            "range": "stddev: 0.0000334846233393643",
            "extra": "mean: 164.77208256700553 usec\nrounds: 6403"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6600.388969441847,
            "unit": "iter/sec",
            "range": "stddev: 0.00007105428920289098",
            "extra": "mean: 151.5062225316948 usec\nrounds: 7220"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22175.504982281127,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028945739609838035",
            "extra": "mean: 45.09480171022167 usec\nrounds: 23136"
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
          "id": "2cda795247c340fd86aee1b1eff7d04dfafa1263",
          "message": "Enable sam2 video pipeline",
          "timestamp": "2024-12-11T23:44:09Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/444/commits/2cda795247c340fd86aee1b1eff7d04dfafa1263"
        },
        "date": 1734114807487,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6070.748415058108,
            "unit": "iter/sec",
            "range": "stddev: 0.000037394411567721216",
            "extra": "mean: 164.72433572103947 usec\nrounds: 6418"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6737.59358817918,
            "unit": "iter/sec",
            "range": "stddev: 0.00007441907801126953",
            "extra": "mean: 148.42094390413476 usec\nrounds: 7292"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22300.62012161977,
            "unit": "iter/sec",
            "range": "stddev: 0.00000216782453550478",
            "extra": "mean: 44.841802360039786 usec\nrounds: 23315"
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
          "id": "279e8d673adb30bbe63c44162e181f794b9f7f94",
          "message": "Renames package to `nvtripy` to avoid name collisions on PyPI",
          "timestamp": "2024-12-13T18:34:47Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/447/commits/279e8d673adb30bbe63c44162e181f794b9f7f94"
        },
        "date": 1734378167458,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6219.734814301458,
            "unit": "iter/sec",
            "range": "stddev: 0.000033158234610608986",
            "extra": "mean: 160.7785588704895 usec\nrounds: 6565"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6806.599072485636,
            "unit": "iter/sec",
            "range": "stddev: 0.00007056267177925717",
            "extra": "mean: 146.9162483864089 usec\nrounds: 7418"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22759.41703285516,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023119483925093976",
            "extra": "mean: 43.93785651699315 usec\nrounds: 23945"
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
          "id": "c429bead299b9cbb334c6031498ceced3c569295",
          "message": "Renames package to `nvtripy` to avoid name collisions on PyPI",
          "timestamp": "2024-12-13T18:34:47Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/447/commits/c429bead299b9cbb334c6031498ceced3c569295"
        },
        "date": 1734378430977,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6148.432847333997,
            "unit": "iter/sec",
            "range": "stddev: 0.00003355243402377117",
            "extra": "mean: 162.64307097923447 usec\nrounds: 6488"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6805.252158933403,
            "unit": "iter/sec",
            "range": "stddev: 0.00007126631424342855",
            "extra": "mean: 146.94532643985545 usec\nrounds: 7416"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22666.527741287977,
            "unit": "iter/sec",
            "range": "stddev: 0.00000224664080928088",
            "extra": "mean: 44.11791746022309 usec\nrounds: 23935"
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
          "id": "88537ac1d5330f466341d44b20a0a713f77367c3",
          "message": "Renames package to `nvtripy` to avoid name collisions on PyPI",
          "timestamp": "2024-12-13T18:34:47Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/447/commits/88537ac1d5330f466341d44b20a0a713f77367c3"
        },
        "date": 1734379137874,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6139.16209753273,
            "unit": "iter/sec",
            "range": "stddev: 0.00003227843073124693",
            "extra": "mean: 162.88867831033335 usec\nrounds: 6435"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6713.849219267543,
            "unit": "iter/sec",
            "range": "stddev: 0.00007553559372405586",
            "extra": "mean: 148.94585316723817 usec\nrounds: 7365"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22784.34776560659,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023715907720417183",
            "extra": "mean: 43.8897795226563 usec\nrounds: 23885"
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
          "id": "6442b8168dc360e9e7aa0ea07cb86a82056042f7",
          "message": "Renames package to `nvtripy` to avoid name collisions on PyPI",
          "timestamp": "2024-12-13T18:34:47Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/447/commits/6442b8168dc360e9e7aa0ea07cb86a82056042f7"
        },
        "date": 1734385835758,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5985.206251652282,
            "unit": "iter/sec",
            "range": "stddev: 0.00003263622759166793",
            "extra": "mean: 167.0786198427062 usec\nrounds: 6382"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6585.956259644864,
            "unit": "iter/sec",
            "range": "stddev: 0.00007089606306676495",
            "extra": "mean: 151.8382389095799 usec\nrounds: 7145"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22473.76168003352,
            "unit": "iter/sec",
            "range": "stddev: 0.000004667063177878455",
            "extra": "mean: 44.49633373519464 usec\nrounds: 23851"
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
          "id": "77d5d204371140ff2a38774bf70b2bd6b4d29ef4",
          "message": "Use singleton class to manage sharing of models across image and vide…",
          "timestamp": "2024-12-13T18:34:47Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/449/commits/77d5d204371140ff2a38774bf70b2bd6b4d29ef4"
        },
        "date": 1734387495467,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6154.005995655653,
            "unit": "iter/sec",
            "range": "stddev: 0.000033393581168983025",
            "extra": "mean: 162.49577928684795 usec\nrounds: 6508"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6815.117446447019,
            "unit": "iter/sec",
            "range": "stddev: 0.0000694427978710383",
            "extra": "mean: 146.73261434714354 usec\nrounds: 7514"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22529.656158446218,
            "unit": "iter/sec",
            "range": "stddev: 0.000002338180272291587",
            "extra": "mean: 44.38594148828617 usec\nrounds: 23771"
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
          "id": "f16e83bc56f82abca1abbb454ef049e8a43f867a",
          "message": "Updates version to 0.0.7",
          "timestamp": "2024-12-16T23:37:08Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/450/commits/f16e83bc56f82abca1abbb454ef049e8a43f867a"
        },
        "date": 1734393019718,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6101.352105162769,
            "unit": "iter/sec",
            "range": "stddev: 0.00003254289025278708",
            "extra": "mean: 163.89809713716278 usec\nrounds: 6440"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6754.938031529105,
            "unit": "iter/sec",
            "range": "stddev: 0.00007267383058077125",
            "extra": "mean: 148.03984808334823 usec\nrounds: 7372"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22824.82196330068,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035119722448690034",
            "extra": "mean: 43.811951813156256 usec\nrounds: 24121"
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
          "id": "c655cd8077a5bc51717a7d996d73f939a5325075",
          "message": "Updates version to 0.0.7",
          "timestamp": "2024-12-16T23:37:08Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/450/commits/c655cd8077a5bc51717a7d996d73f939a5325075"
        },
        "date": 1734393279157,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6143.215447577695,
            "unit": "iter/sec",
            "range": "stddev: 0.00003218940570556401",
            "extra": "mean: 162.7812028624693 usec\nrounds: 6457"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6662.302593348292,
            "unit": "iter/sec",
            "range": "stddev: 0.00007172173868480599",
            "extra": "mean: 150.0982559691014 usec\nrounds: 7318"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22827.834512115543,
            "unit": "iter/sec",
            "range": "stddev: 0.000002167414003196636",
            "extra": "mean: 43.80617002761539 usec\nrounds: 23628"
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
          "id": "2eb4935e62cac9c7fa994373a6bb852aa1b328e4",
          "message": "Add text to segmentation demo code",
          "timestamp": "2024-12-16T23:37:08Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/451/commits/2eb4935e62cac9c7fa994373a6bb852aa1b328e4"
        },
        "date": 1734394750763,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6115.635369667406,
            "unit": "iter/sec",
            "range": "stddev: 0.00009608805574558989",
            "extra": "mean: 163.51530782228178 usec\nrounds: 6497"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6948.01467930213,
            "unit": "iter/sec",
            "range": "stddev: 0.00008972445530081366",
            "extra": "mean: 143.92600565150812 usec\nrounds: 7451"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22492.291988864486,
            "unit": "iter/sec",
            "range": "stddev: 0.000002228107178504487",
            "extra": "mean: 44.45967536323472 usec\nrounds: 23361"
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
          "id": "6e9c193caac78c0285e1d0ad1ea77ed08590e827",
          "message": "Updates release pipeline to only run L0 tests",
          "timestamp": "2024-12-16T23:55:10Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/452/commits/6e9c193caac78c0285e1d0ad1ea77ed08590e827"
        },
        "date": 1734397092534,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6174.2449202082635,
            "unit": "iter/sec",
            "range": "stddev: 0.00003274607702045044",
            "extra": "mean: 161.9631247097125 usec\nrounds: 6529"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6673.351639502546,
            "unit": "iter/sec",
            "range": "stddev: 0.00006758337613871858",
            "extra": "mean: 149.84973878501376 usec\nrounds: 7388"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22819.72471172687,
            "unit": "iter/sec",
            "range": "stddev: 0.000002128439024463257",
            "extra": "mean: 43.82173810738866 usec\nrounds: 24053"
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
          "id": "f57327ee298a3cc5ba9b5f7dbe8a55d900bd8f68",
          "message": "Updates release pipeline to only run L0 tests",
          "timestamp": "2024-12-16T23:55:10Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/452/commits/f57327ee298a3cc5ba9b5f7dbe8a55d900bd8f68"
        },
        "date": 1734397351235,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6127.9677093507,
            "unit": "iter/sec",
            "range": "stddev: 0.00003337319179212724",
            "extra": "mean: 163.1862384774147 usec\nrounds: 6444"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6789.627980053438,
            "unit": "iter/sec",
            "range": "stddev: 0.00007115656759056439",
            "extra": "mean: 147.28347457884274 usec\nrounds: 7320"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22718.323581950346,
            "unit": "iter/sec",
            "range": "stddev: 0.000002209142841842912",
            "extra": "mean: 44.01733237017971 usec\nrounds: 23769"
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
          "id": "f89fd44b9a1d4f5e8458d56414ab1e99f5c637b8",
          "message": "[Tripy] Permit `eval` while tracing, but do not update trace.",
          "timestamp": "2024-12-16T23:55:10Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/443/commits/f89fd44b9a1d4f5e8458d56414ab1e99f5c637b8"
        },
        "date": 1734410634330,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6103.87593510876,
            "unit": "iter/sec",
            "range": "stddev: 0.00003297631277675591",
            "extra": "mean: 163.83032857010087 usec\nrounds: 6391"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6623.269018577241,
            "unit": "iter/sec",
            "range": "stddev: 0.00007063515644251624",
            "extra": "mean: 150.98284505659595 usec\nrounds: 7177"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 23306.799636585936,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025387819549561813",
            "extra": "mean: 42.90593370143562 usec\nrounds: 24649"
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
          "id": "80a676131a2e81fb4f4bfdd9f0a0a1db62dcf33c",
          "message": "Updates install instructions to point to PyPI",
          "timestamp": "2024-12-17T18:44:57Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/453/commits/80a676131a2e81fb4f4bfdd9f0a0a1db62dcf33c"
        },
        "date": 1734462617198,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6030.31679982974,
            "unit": "iter/sec",
            "range": "stddev: 0.000033136115628677344",
            "extra": "mean: 165.82876707708522 usec\nrounds: 6396"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6667.295956219318,
            "unit": "iter/sec",
            "range": "stddev: 0.00006910911719300194",
            "extra": "mean: 149.98584232145723 usec\nrounds: 7294"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22979.65337201387,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019705504733428637",
            "extra": "mean: 43.51675735970264 usec\nrounds: 24030"
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
          "id": "ab675a4e96770ce7c8e08398c2c7300c53b734ef",
          "message": "Updates install instructions to point to PyPI",
          "timestamp": "2024-12-17T23:45:10Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/453/commits/ab675a4e96770ce7c8e08398c2c7300c53b734ef"
        },
        "date": 1734544585362,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6094.041105105067,
            "unit": "iter/sec",
            "range": "stddev: 0.00003282471601764977",
            "extra": "mean: 164.09472511799854 usec\nrounds: 6427"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6621.726543753477,
            "unit": "iter/sec",
            "range": "stddev: 0.00007137895174295675",
            "extra": "mean: 151.01801522494725 usec\nrounds: 7173"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22637.679211818555,
            "unit": "iter/sec",
            "range": "stddev: 0.000002046324404623096",
            "extra": "mean: 44.17413952389278 usec\nrounds: 24066"
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
          "id": "30719f8f6d3ecb69a7477f2b3149eb90dcf02f5f",
          "message": "TESTING: DO NOT MERGE",
          "timestamp": "2024-12-18T19:16:23Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/459/commits/30719f8f6d3ecb69a7477f2b3149eb90dcf02f5f"
        },
        "date": 1734550729683,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6108.658565506243,
            "unit": "iter/sec",
            "range": "stddev: 0.00003315042811269311",
            "extra": "mean: 163.70206147167877 usec\nrounds: 6480"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6720.735494689581,
            "unit": "iter/sec",
            "range": "stddev: 0.00006972568626487124",
            "extra": "mean: 148.79323859570943 usec\nrounds: 7305"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22785.937568886955,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022457742112014756",
            "extra": "mean: 43.88671727800437 usec\nrounds: 23708"
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
          "id": "78d194c278eb71e0287379f2ff3566f551f98a04",
          "message": "Reduce nanogpt quantization calib size to speed up test",
          "timestamp": "2024-12-18T19:16:23Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/461/commits/78d194c278eb71e0287379f2ff3566f551f98a04"
        },
        "date": 1734551456807,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6149.806894431873,
            "unit": "iter/sec",
            "range": "stddev: 0.00003460238247242626",
            "extra": "mean: 162.60673175045787 usec\nrounds: 6491"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6675.7546774079965,
            "unit": "iter/sec",
            "range": "stddev: 0.00007030665244104279",
            "extra": "mean: 149.79579812664286 usec\nrounds: 7403"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22601.789031581786,
            "unit": "iter/sec",
            "range": "stddev: 0.000002305125137640534",
            "extra": "mean: 44.24428520249819 usec\nrounds: 23560"
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
          "id": "37161ad78e3180f3a201f5584cd702c132fb2e90",
          "message": "Fixes a flaky test by increasing tolerance",
          "timestamp": "2024-12-18T19:16:23Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/462/commits/37161ad78e3180f3a201f5584cd702c132fb2e90"
        },
        "date": 1734554587316,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6129.680871859039,
            "unit": "iter/sec",
            "range": "stddev: 0.00009730665801041375",
            "extra": "mean: 163.1406301412744 usec\nrounds: 6458"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6692.632678294369,
            "unit": "iter/sec",
            "range": "stddev: 0.00008950934402020765",
            "extra": "mean: 149.41803144870218 usec\nrounds: 7384"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22717.410106213727,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023205235402113147",
            "extra": "mean: 44.019102323925445 usec\nrounds: 23522"
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
          "id": "9411b6c52993c3e60a26f7955e09093516e750b3",
          "message": "Fixes a flaky test by increasing tolerance",
          "timestamp": "2024-12-18T20:47:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/462/commits/9411b6c52993c3e60a26f7955e09093516e750b3"
        },
        "date": 1734555944738,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6022.065252034321,
            "unit": "iter/sec",
            "range": "stddev: 0.00003283495436703077",
            "extra": "mean: 166.05598879258056 usec\nrounds: 6364"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6609.346749103772,
            "unit": "iter/sec",
            "range": "stddev: 0.00006833369544344614",
            "extra": "mean: 151.30088312216333 usec\nrounds: 7255"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22452.24779355823,
            "unit": "iter/sec",
            "range": "stddev: 0.000001839268022702166",
            "extra": "mean: 44.53897040486565 usec\nrounds: 23414"
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
          "id": "bde61740075c9a4ea806cd0ce02328fb7b76b9ee",
          "message": "Fixes a flaky test by increasing tolerance",
          "timestamp": "2024-12-18T20:47:41Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/462/commits/bde61740075c9a4ea806cd0ce02328fb7b76b9ee"
        },
        "date": 1734558324084,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6173.285188551154,
            "unit": "iter/sec",
            "range": "stddev: 0.00003217585376106324",
            "extra": "mean: 161.9883043560954 usec\nrounds: 6534"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6719.168567625084,
            "unit": "iter/sec",
            "range": "stddev: 0.00007004788976190519",
            "extra": "mean: 148.82793755440102 usec\nrounds: 7404"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22177.64419341094,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033150058337951135",
            "extra": "mean: 45.09045195598835 usec\nrounds: 23618"
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
          "id": "59d34c3a52a18120de70a500c818c7210d6d98aa",
          "message": "Tripy eager cache",
          "timestamp": "2024-12-19T17:45:47Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/418/commits/59d34c3a52a18120de70a500c818c7210d6d98aa"
        },
        "date": 1734632949760,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6053.478750106258,
            "unit": "iter/sec",
            "range": "stddev: 0.00003216091835166712",
            "extra": "mean: 165.19426949048872 usec\nrounds: 6378"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6635.822341516807,
            "unit": "iter/sec",
            "range": "stddev: 0.00006831220428864831",
            "extra": "mean: 150.69722312237514 usec\nrounds: 7277"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22900.39085449322,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021511331895619615",
            "extra": "mean: 43.66737696111387 usec\nrounds: 23882"
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
          "id": "64677a5b7787b89cf74e72e3f2962fdfa520f4cd",
          "message": "Tripy eager cache",
          "timestamp": "2024-12-19T17:45:47Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/418/commits/64677a5b7787b89cf74e72e3f2962fdfa520f4cd"
        },
        "date": 1734633497621,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6106.188013950015,
            "unit": "iter/sec",
            "range": "stddev: 0.00003417141698496617",
            "extra": "mean: 163.76829500097767 usec\nrounds: 6482"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6570.936414570767,
            "unit": "iter/sec",
            "range": "stddev: 0.00007023361941085688",
            "extra": "mean: 152.1853107241372 usec\nrounds: 7288"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 23013.99174545048,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019337154309364564",
            "extra": "mean: 43.451827525647964 usec\nrounds: 23999"
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
          "id": "af23b58f95e2aa4c11c3be308fb51ce3fb06e7b4",
          "message": "Tripy eager cache",
          "timestamp": "2024-12-19T17:45:47Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/418/commits/af23b58f95e2aa4c11c3be308fb51ce3fb06e7b4"
        },
        "date": 1734633988749,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6067.239970256139,
            "unit": "iter/sec",
            "range": "stddev: 0.000032660752990039545",
            "extra": "mean: 164.81958928645824 usec\nrounds: 6375"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6451.599613211359,
            "unit": "iter/sec",
            "range": "stddev: 0.00006849535864589221",
            "extra": "mean: 155.00031929325485 usec\nrounds: 7117"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22092.07435996037,
            "unit": "iter/sec",
            "range": "stddev: 0.000002402318130057048",
            "extra": "mean: 45.26510203190326 usec\nrounds: 23089"
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
          "id": "b95e9bd8174e265e5859684c94d3e0b158abb823",
          "message": "Tripy eager cache",
          "timestamp": "2024-12-19T17:45:47Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/418/commits/b95e9bd8174e265e5859684c94d3e0b158abb823"
        },
        "date": 1734635644383,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6078.9714476635945,
            "unit": "iter/sec",
            "range": "stddev: 0.00003327191706581412",
            "extra": "mean: 164.50151289727515 usec\nrounds: 6498"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6564.534199703216,
            "unit": "iter/sec",
            "range": "stddev: 0.00007088761171322254",
            "extra": "mean: 152.33373299284668 usec\nrounds: 7177"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 23122.378573279715,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023600625368407426",
            "extra": "mean: 43.24814580951472 usec\nrounds: 24119"
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
          "id": "1edf2393ed98a4fd25a35e905a26184a154e8bcb",
          "message": "Tripy eager cache",
          "timestamp": "2024-12-19T20:22:59Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/418/commits/1edf2393ed98a4fd25a35e905a26184a154e8bcb"
        },
        "date": 1734654241943,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6087.877257895945,
            "unit": "iter/sec",
            "range": "stddev: 0.000035842944053597727",
            "extra": "mean: 164.26086756315678 usec\nrounds: 6458"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6504.323007705574,
            "unit": "iter/sec",
            "range": "stddev: 0.00007103474218942946",
            "extra": "mean: 153.74390214251582 usec\nrounds: 7138"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22667.817972755554,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022901014806758634",
            "extra": "mean: 44.11540630871043 usec\nrounds: 23682"
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
          "id": "ffaafc6ef229a5ff7eb4847ddf89562554ac7284",
          "message": "Tripy eager cache",
          "timestamp": "2024-12-19T20:22:59Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/418/commits/ffaafc6ef229a5ff7eb4847ddf89562554ac7284"
        },
        "date": 1734654705674,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6122.868454118604,
            "unit": "iter/sec",
            "range": "stddev: 0.00003389684112385712",
            "extra": "mean: 163.32214345179028 usec\nrounds: 6453"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6465.942462456488,
            "unit": "iter/sec",
            "range": "stddev: 0.00007251367002543698",
            "extra": "mean: 154.65649529149817 usec\nrounds: 7126"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22854.4527054988,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025226889651160152",
            "extra": "mean: 43.755149724473576 usec\nrounds: 23799"
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
          "id": "a461a762bbe01f7d8baa83a9e700de84a3fc32d5",
          "message": "Support small and tiny config for SAM2",
          "timestamp": "2024-12-30T02:59:49Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/469/commits/a461a762bbe01f7d8baa83a9e700de84a3fc32d5"
        },
        "date": 1735933281057,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6101.363481887397,
            "unit": "iter/sec",
            "range": "stddev: 0.000032528952181808644",
            "extra": "mean: 163.8977915294861 usec\nrounds: 6414"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6575.91587503839,
            "unit": "iter/sec",
            "range": "stddev: 0.00006831316613422493",
            "extra": "mean: 152.07007191133843 usec\nrounds: 7202"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22573.905816013215,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021968941143562592",
            "extra": "mean: 44.29893560070724 usec\nrounds: 23492"
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
          "id": "bf92b05ada8682677a0966c4139d2f69dff5a721",
          "message": "Fix sam2 sample artifact removal",
          "timestamp": "2025-01-06T18:06:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/470/commits/bf92b05ada8682677a0966c4139d2f69dff5a721"
        },
        "date": 1736273950771,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6070.825612579375,
            "unit": "iter/sec",
            "range": "stddev: 0.00003399040980934122",
            "extra": "mean: 164.7222410618907 usec\nrounds: 6436"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6700.328796814354,
            "unit": "iter/sec",
            "range": "stddev: 0.00007048711719970723",
            "extra": "mean: 149.2464072025012 usec\nrounds: 7288"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21443.343946229714,
            "unit": "iter/sec",
            "range": "stddev: 0.000004659250606825831",
            "extra": "mean: 46.634517569067185 usec\nrounds: 23257"
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
          "id": "bf92b05ada8682677a0966c4139d2f69dff5a721",
          "message": "Fix sam2 sample artifact removal",
          "timestamp": "2025-01-06T18:06:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/470/commits/bf92b05ada8682677a0966c4139d2f69dff5a721"
        },
        "date": 1736275642145,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6063.264971613102,
            "unit": "iter/sec",
            "range": "stddev: 0.00003602656378101738",
            "extra": "mean: 164.92764289236644 usec\nrounds: 6451"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6622.156195963155,
            "unit": "iter/sec",
            "range": "stddev: 0.00007074801969625277",
            "extra": "mean: 151.00821702296855 usec\nrounds: 7258"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22417.262713590426,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022576375367325827",
            "extra": "mean: 44.60847931240739 usec\nrounds: 23764"
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
          "id": "0c23bf7ea13c678ceb0fd06ddc0ab40f462341e0",
          "message": "Refactors some guides",
          "timestamp": "2025-01-07T19:20:40Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/471/commits/0c23bf7ea13c678ceb0fd06ddc0ab40f462341e0"
        },
        "date": 1736359383649,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6033.759059499891,
            "unit": "iter/sec",
            "range": "stddev: 0.000033533344755533534",
            "extra": "mean: 165.73416176198543 usec\nrounds: 6438"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6695.253958455445,
            "unit": "iter/sec",
            "range": "stddev: 0.00007212920050583569",
            "extra": "mean: 149.35953232021896 usec\nrounds: 7311"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22868.433438759752,
            "unit": "iter/sec",
            "range": "stddev: 0.000002316553101075333",
            "extra": "mean: 43.72839979082686 usec\nrounds: 24081"
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
          "id": "4d60d9ff4c55aad040afb4c6023ad447bf5c1020",
          "message": "Refactors some guides",
          "timestamp": "2025-01-07T19:20:40Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/471/commits/4d60d9ff4c55aad040afb4c6023ad447bf5c1020"
        },
        "date": 1736359984881,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6053.0787025968975,
            "unit": "iter/sec",
            "range": "stddev: 0.000032605779669319256",
            "extra": "mean: 165.20518716714835 usec\nrounds: 6420"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6604.3963766394845,
            "unit": "iter/sec",
            "range": "stddev: 0.0000667433958382332",
            "extra": "mean: 151.41429177950553 usec\nrounds: 7224"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 23173.942960997516,
            "unit": "iter/sec",
            "range": "stddev: 0.00000228418373531128",
            "extra": "mean: 43.15191427212157 usec\nrounds: 24241"
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
          "id": "a258bde2c22afbf0af85bc7102ccded969909de7",
          "message": "Refactors some guides",
          "timestamp": "2025-01-07T19:20:40Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/471/commits/a258bde2c22afbf0af85bc7102ccded969909de7"
        },
        "date": 1736361867117,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6126.86819086472,
            "unit": "iter/sec",
            "range": "stddev: 0.00003339417983684773",
            "extra": "mean: 163.21552363261534 usec\nrounds: 6508"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6657.728168799669,
            "unit": "iter/sec",
            "range": "stddev: 0.00006861881015424913",
            "extra": "mean: 150.20138621554617 usec\nrounds: 7306"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22876.20131465431,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023577591266931744",
            "extra": "mean: 43.713551312359186 usec\nrounds: 24239"
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
          "id": "18b6e33efcd8a1f77dd6b97b1043f506d2ff2daf",
          "message": "Uses OpenCV package that includes binary dependencies",
          "timestamp": "2025-01-08T20:29:10Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/472/commits/18b6e33efcd8a1f77dd6b97b1043f506d2ff2daf"
        },
        "date": 1736373592212,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6058.173149447841,
            "unit": "iter/sec",
            "range": "stddev: 0.00003688304919286635",
            "extra": "mean: 165.0662626061031 usec\nrounds: 6454"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6600.034055002554,
            "unit": "iter/sec",
            "range": "stddev: 0.0000704842315067983",
            "extra": "mean: 151.51436972390184 usec\nrounds: 7244"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 23240.765749594953,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022363881039325422",
            "extra": "mean: 43.02784214489268 usec\nrounds: 24378"
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
          "id": "84c2f93df0290f767fbd6efb4a6e6a14b29b5875",
          "message": "Sandboxes notebook and example tests",
          "timestamp": "2025-01-08T22:31:34Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/473/commits/84c2f93df0290f767fbd6efb4a6e6a14b29b5875"
        },
        "date": 1736457601195,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5846.799234050513,
            "unit": "iter/sec",
            "range": "stddev: 0.00006330106701896317",
            "extra": "mean: 171.03375025710017 usec\nrounds: 6321"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6434.096923391043,
            "unit": "iter/sec",
            "range": "stddev: 0.00008860145724043948",
            "extra": "mean: 155.42196704630268 usec\nrounds: 7125"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22172.960897407433,
            "unit": "iter/sec",
            "range": "stddev: 0.000002444739108690613",
            "extra": "mean: 45.099975805077285 usec\nrounds: 23191"
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
          "id": "eedcd99edd4b01fc8306e101cc21ff7a9755ebd3",
          "message": "Sandboxes notebook and example tests",
          "timestamp": "2025-01-08T22:31:34Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/473/commits/eedcd99edd4b01fc8306e101cc21ff7a9755ebd3"
        },
        "date": 1736459170890,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5879.112821087677,
            "unit": "iter/sec",
            "range": "stddev: 0.00004942006914842389",
            "extra": "mean: 170.09369107752434 usec\nrounds: 6361"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6510.420067474605,
            "unit": "iter/sec",
            "range": "stddev: 0.0000872057925741007",
            "extra": "mean: 153.59991976491625 usec\nrounds: 7243"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22554.88417332879,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029067054215460115",
            "extra": "mean: 44.33629507095863 usec\nrounds: 23801"
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
          "id": "19f956d7539f9ad9afece30df9492b945d4c8ec9",
          "message": "Sandboxes notebook and example tests",
          "timestamp": "2025-01-08T22:31:34Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/473/commits/19f956d7539f9ad9afece30df9492b945d4c8ec9"
        },
        "date": 1736460200918,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5891.971462140408,
            "unit": "iter/sec",
            "range": "stddev: 0.000049046856091876376",
            "extra": "mean: 169.72247853297046 usec\nrounds: 6315"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6326.6385789712685,
            "unit": "iter/sec",
            "range": "stddev: 0.00009185950774943633",
            "extra": "mean: 158.0618186921313 usec\nrounds: 7245"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22074.451955132216,
            "unit": "iter/sec",
            "range": "stddev: 0.000004793425905181472",
            "extra": "mean: 45.301237921220704 usec\nrounds: 24253"
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
          "id": "4d7e318e1957a5cf2b90cde5b46e60c2172a6ddc",
          "message": "Sandboxes notebook and example tests",
          "timestamp": "2025-01-08T22:31:34Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/473/commits/4d7e318e1957a5cf2b90cde5b46e60c2172a6ddc"
        },
        "date": 1736462100866,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6124.695521147494,
            "unit": "iter/sec",
            "range": "stddev: 0.00003232901276439587",
            "extra": "mean: 163.27342258030237 usec\nrounds: 6415"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6639.4669005324,
            "unit": "iter/sec",
            "range": "stddev: 0.00006698312710671788",
            "extra": "mean: 150.6145018841517 usec\nrounds: 7291"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22534.982865521273,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021383837189401614",
            "extra": "mean: 44.37544976038162 usec\nrounds: 23856"
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
          "id": "8b1adb46d4755a9a2be6ccfc21cd8f56bbb863c6",
          "message": "Refactors several more docs",
          "timestamp": "2025-01-10T00:03:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/474/commits/8b1adb46d4755a9a2be6ccfc21cd8f56bbb863c6"
        },
        "date": 1736555156576,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6065.215840532495,
            "unit": "iter/sec",
            "range": "stddev: 0.000033125147172974815",
            "extra": "mean: 164.87459412692638 usec\nrounds: 6418"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6648.793367343343,
            "unit": "iter/sec",
            "range": "stddev: 0.00007140499306326952",
            "extra": "mean: 150.4032302931336 usec\nrounds: 7319"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22656.04706793943,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024758328858011215",
            "extra": "mean: 44.138326381529275 usec\nrounds: 23546"
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
          "id": "89306445be8ea4a3f9f426eef4e34244ca6d0ee9",
          "message": "Refactors several more docs",
          "timestamp": "2025-01-10T00:03:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/474/commits/89306445be8ea4a3f9f426eef4e34244ca6d0ee9"
        },
        "date": 1736555452608,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6102.58867827637,
            "unit": "iter/sec",
            "range": "stddev: 0.0000965681533409574",
            "extra": "mean: 163.86488631615305 usec\nrounds: 6439"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6643.088624082203,
            "unit": "iter/sec",
            "range": "stddev: 0.00009378562814966726",
            "extra": "mean: 150.5323888612367 usec\nrounds: 7275"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22812.790570037134,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024024731143324278",
            "extra": "mean: 43.83505809733878 usec\nrounds: 23924"
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
          "id": "e13e1b6417c9e02d93e78588069cfb53cb244e6b",
          "message": "Refactors several more docs",
          "timestamp": "2025-01-10T00:03:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/474/commits/e13e1b6417c9e02d93e78588069cfb53cb244e6b"
        },
        "date": 1736555702417,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6002.807456812159,
            "unit": "iter/sec",
            "range": "stddev: 0.00003303073180019967",
            "extra": "mean: 166.5887182280303 usec\nrounds: 6310"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6559.566452033395,
            "unit": "iter/sec",
            "range": "stddev: 0.00006732046734462165",
            "extra": "mean: 152.4490996946926 usec\nrounds: 7141"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21893.893219103742,
            "unit": "iter/sec",
            "range": "stddev: 0.00000254418242930538",
            "extra": "mean: 45.67483681373945 usec\nrounds: 22977"
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
          "id": "67edeaf9cbda9f3fbf5fe705c526747d1f832f3a",
          "message": "Refactors several more docs",
          "timestamp": "2025-01-10T00:03:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/474/commits/67edeaf9cbda9f3fbf5fe705c526747d1f832f3a"
        },
        "date": 1736555948551,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6078.55859743808,
            "unit": "iter/sec",
            "range": "stddev: 0.00003620176047578938",
            "extra": "mean: 164.51268569187906 usec\nrounds: 6468"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6664.84819514661,
            "unit": "iter/sec",
            "range": "stddev: 0.00007372341426805583",
            "extra": "mean: 150.0409267728269 usec\nrounds: 7418"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22075.017090597117,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021866555039651418",
            "extra": "mean: 45.300078178691486 usec\nrounds: 22959"
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
          "id": "cabceca81ecb01de40a7766627a445a5df543dec",
          "message": "Refactors several more docs",
          "timestamp": "2025-01-10T00:03:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/474/commits/cabceca81ecb01de40a7766627a445a5df543dec"
        },
        "date": 1736556196286,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6055.539111377302,
            "unit": "iter/sec",
            "range": "stddev: 0.000036062576459545166",
            "extra": "mean: 165.13806311995813 usec\nrounds: 6447"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6648.295632923673,
            "unit": "iter/sec",
            "range": "stddev: 0.00007114007885933234",
            "extra": "mean: 150.41449045192914 usec\nrounds: 7251"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22556.75291198331,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023778601312985135",
            "extra": "mean: 44.332621982517196 usec\nrounds: 23372"
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
          "id": "55317d829a5f40b856570ad4c4cc65fea8ff8dc6",
          "message": "Refactors several more docs",
          "timestamp": "2025-01-10T00:03:45Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/474/commits/55317d829a5f40b856570ad4c4cc65fea8ff8dc6"
        },
        "date": 1736556440288,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6136.026668965782,
            "unit": "iter/sec",
            "range": "stddev: 0.0000344716615631417",
            "extra": "mean: 162.97191227308477 usec\nrounds: 6501"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6695.9516740205745,
            "unit": "iter/sec",
            "range": "stddev: 0.00006945400562008936",
            "extra": "mean: 149.34396911493113 usec\nrounds: 7392"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21812.394747448725,
            "unit": "iter/sec",
            "range": "stddev: 0.00000247047975270972",
            "extra": "mean: 45.84549342602396 usec\nrounds: 22719"
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
          "id": "3f4f0d7148b56b52f778358ed27e19480d9e3205",
          "message": "Refactors several more docs",
          "timestamp": "2025-01-11T03:51:04Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/474/commits/3f4f0d7148b56b52f778358ed27e19480d9e3205"
        },
        "date": 1736790168544,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6087.40482385857,
            "unit": "iter/sec",
            "range": "stddev: 0.00003191224228784401",
            "extra": "mean: 164.27361559406506 usec\nrounds: 6421"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6555.483031856791,
            "unit": "iter/sec",
            "range": "stddev: 0.00006667666232176293",
            "extra": "mean: 152.5440604667018 usec\nrounds: 7194"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21918.01740118639,
            "unit": "iter/sec",
            "range": "stddev: 0.000002312098723798423",
            "extra": "mean: 45.624564562389274 usec\nrounds: 23326"
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
          "id": "f0893d54c2926411824783616f1b850d4b2a739f",
          "message": "Refactors several more docs",
          "timestamp": "2025-01-11T03:51:04Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/474/commits/f0893d54c2926411824783616f1b850d4b2a739f"
        },
        "date": 1736790463368,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6042.178134157351,
            "unit": "iter/sec",
            "range": "stddev: 0.000035267295248874495",
            "extra": "mean: 165.50323042395058 usec\nrounds: 6351"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6412.673646689441,
            "unit": "iter/sec",
            "range": "stddev: 0.00006996730540767388",
            "extra": "mean: 155.9411963083842 usec\nrounds: 7060"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 21609.40377797542,
            "unit": "iter/sec",
            "range": "stddev: 0.000002248703416455296",
            "extra": "mean: 46.27614950761449 usec\nrounds: 22383"
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
          "id": "d9fa8ea0c680d47499a2e4f791a6891a0ebc670e",
          "message": "Refactors several more docs",
          "timestamp": "2025-01-11T03:51:04Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/474/commits/d9fa8ea0c680d47499a2e4f791a6891a0ebc670e"
        },
        "date": 1736791081346,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 5964.812164168693,
            "unit": "iter/sec",
            "range": "stddev: 0.00003565745741268269",
            "extra": "mean: 167.64987269961557 usec\nrounds: 6424"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6630.283427138132,
            "unit": "iter/sec",
            "range": "stddev: 0.00006756993868383746",
            "extra": "mean: 150.82311502807596 usec\nrounds: 7226"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22682.211901429633,
            "unit": "iter/sec",
            "range": "stddev: 0.000002279594079217841",
            "extra": "mean: 44.087411066685746 usec\nrounds: 23533"
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
          "id": "4f9ea4a2c71cdd455f191811f9e63fca9ae70544",
          "message": "Minor docstring fixes",
          "timestamp": "2025-01-13T18:36:13Z",
          "url": "https://github.com/NVIDIA/TensorRT-Incubator/pull/476/commits/4f9ea4a2c71cdd455f191811f9e63fca9ae70544"
        },
        "date": 1736806682818,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float32]",
            "value": 6096.180114539049,
            "unit": "iter/sec",
            "range": "stddev: 0.00003318550401699513",
            "extra": "mean: 164.03714805195077 usec\nrounds: 6455"
          },
          {
            "name": "tests/performance/test_perf.py::test_perf_regression[linear_block-float16]",
            "value": 6602.604363251533,
            "unit": "iter/sec",
            "range": "stddev: 0.00007040646790606222",
            "extra": "mean: 151.4553871447687 usec\nrounds: 7216"
          },
          {
            "name": "tests/performance/test_perf.py::test_tripy_param_update",
            "value": 22532.91737999406,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022294809020316023",
            "extra": "mean: 44.37951744712178 usec\nrounds: 23505"
          }
        ]
      }
    ]
  }
}