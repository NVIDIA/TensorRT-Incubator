// RUN: kernel-opt %s -split-input-file -verify-diagnostics -kernel-set-gpu-target="chip=sm_86 features=+sm70 populate-host-system-spec" | FileCheck %s

gpu.module @no_existing_target {

}

// CHECK-LABEL: module attributes {dlti.target_system_spec = #dlti.target_system_spec<"GPU:0" =
// CHECK-LABEL: gpu.module @no_existing_target 
//  CHECK-SAME: [#nvvm.target<chip = "sm_86", features = "+sm70", 
//  CHECK-SAME: flags = {spec = #dlti.target_device_spec<"maxSharedMemoryPerBlockKb" = 48 : i64, "maxRegisterPerBlock" = 65536 : i64>}>]

// -----

// expected-error @below {{GPU module already has a target spec}}
gpu.module @existing_target_incompatible [#nvvm.target<chip = "sm_80", features = "+sm70">] {

}

