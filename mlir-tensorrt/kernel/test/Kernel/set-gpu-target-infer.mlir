// REQUIRES: host-has-at-least-1-gpus
// RUN: kernel-opt %s -kernel-set-gpu-target="infer-target-from-host populate-host-system-spec" | FileCheck %s

gpu.module @no_existing_target {

}

// CHECK-LABEL: module attributes {dlti.target_system_spec = #dlti.target_system_spec<"GPU:0" = 
// CHECK-LABEL: gpu.module @no_existing_target 
//  CHECK-SAME: [#nvvm.target<chip = "sm_{{.*}}"

