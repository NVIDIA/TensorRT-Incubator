// RUN: rm -rf %t/artifacts && mkdir -p %t/artifacts
// RUN: executor-opt %s -executor-serialize-artifacts="artifacts-directory=%t/artifacts create-manifest=true" -o /dev/null
// RUN: test -f %t/artifacts/blob.bin
// RUN: test -f %t/artifacts/manifest.json
// RUN: cat %t/artifacts/manifest.json | FileCheck %s --check-prefix=MANIFEST

module @my_module {
  executor.file_artifact "blob.bin" data(dense<[1, 2, 3, 4]> : tensor<4xi8>)
}

// MANIFEST: "artifacts":
// MANIFEST: "kind": "ConstantBlob"
// MANIFEST: "relpath": "blob.bin"
// MANIFEST-DAG: "schema_version": 1
// MANIFEST-DAG: "module_name": "my_module"
