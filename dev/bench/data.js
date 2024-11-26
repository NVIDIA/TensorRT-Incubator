window.BENCHMARK_DATA = {
  "lastUpdate": 1732650025934,
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
      }
    ]
  }
}