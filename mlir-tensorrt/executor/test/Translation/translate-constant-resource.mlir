// REQUIRES: host-has-at-least-1-gpus
// RUN: executor-translate -mlir-to-runtime-executable %s | executor-runner -dump-function-signature -input-type=rtexe | FileCheck %s

executor.data_segment @__constant_64xf32_initializer dense_resource<torch_tensor_64_torch.float32_7> : vector<64xf32>

func.func @load_from_const() -> !executor.ptr<host> {
  %2 = executor.load_data_segment @__constant_64xf32_initializer : !executor.ptr<host>
  return %2: !executor.ptr<host>
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_64_torch.float32_7: "0x04000000083DCCBDED76F3BE5AFD41BDC6238ABE11E3AABDDCF7EBBB583E423DCD36D13D5B1D8BBCDAAD16BE95316C3EBC22143EAFC735BEC061DF3C929E1F3E0A0A3CBE537FFD3DBCDB1BBE421C633D788393BED26F85BECC336D3ED902A6BEAE23B3BC526887BE18F26B3DD0617CBEE836453DEDB6B4BE768EC23DE3B03EBE1ACB2EBEF08483BCE15485BE7804A1BE1CBFDDBD190D873D7D2412BE0E6D69BDB76EF8BB4AE799BE935B98BD7CD08BBDAB13E2BDC9F1CE3C9461483D6AC980BEE998EB3D99B3C1BE453DAD3DE1E716BD4D56133E99B53FBDB5109EBEB8A7C4BC0A070F3E14F2B9BDB4C637BE1932E7BDB54EA2BD8FA518BE617D733DACFBE2BEEBB061BC"
    }
  }
#-}

// CHECK-LABEL: Function<load_from_const, Signature<args=[], results=[], num_output_args=0, arg_bounds=[], result_bounds=[], cconv=packed>>
