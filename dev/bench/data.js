window.BENCHMARK_DATA = {
  "lastUpdate": 1730490385874,
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
      }
    ]
  }
}