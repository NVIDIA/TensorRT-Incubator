//===- TransformScheduleBase.h --------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// Declarations for the transform schedule generator base class and
/// schedule generator registration infrastructure.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_KERNEL_IR_TRANSFORMSCHEDULEBASE
#define MLIR_KERNEL_KERNEL_IR_TRANSFORMSCHEDULEBASE

#include "mlir-kernel/Kernel/IR/Enums.h"
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::kernel {

/// Options common to all transform schedule generators. This struct should only
/// be used to communicate options which are common to all generators that would
/// be invoked within a particular pass/module scope. For example, the data
/// layout or device information are included here since they do not depend on
/// which kernel we are generating.
struct TransformScheduleOptions {
  // The number of shared memory per block of the CUDA device.
  uint64_t sharedMemoryPerBlockLimitBytes;
  // The number of SMs on the device.
  uint64_t numMultiProcessors;
  // The number of 4-byte registers per block.
  uint64_t registersPerBlockLimit;

  // The data layout of the device module.
  const DataLayout &dataLayout;

  // Device target information.
  gpu::TargetAttrInterface gpuTargetInfo;

  // Whether to enumerate both power-of-2 and non-power-of-2 tile
  // configurations. If true, enumerateParameters will generate 2 configs per
  // kernel (when possible). If false, only generates the default
  // (non-power-of-2) configuration.
  bool enumeratePowerOfTwoConfigs = false;

  // Whether to restrict to ONLY power-of-2 tile configurations.
  // If true, only generates power-of-2 configs (no non-power-of-2).
  // Takes precedence over enumeratePowerOfTwoConfigs.
  bool onlyPowerOfTwoConfigs = false;

  // Number of stages to load into shared memory for strip mining reduction
  // dimensions. This multiplies the shared memory usage by this factor when
  // calculating feasible CTA blocking shapes.
  uint64_t numStages = 1;

  // Maximum number of configurations to return from enumerateParameters.
  // If 0, returns all valid configurations. Otherwise, returns up to maxConfigs
  // configurations sorted by expected performance (higher usage score first).
  // This is useful for controlling CP-SAT problem size.
  uint64_t maxConfigsPerKernel = 0;
};

/// TransformScheduleBase is the base class for all Transform IR generators.
/// Typically, concrete generators inherit from one of the classes specified
/// under the `TransformSchedules` folder.
class TransformScheduleBase : public mlir::Pattern {
public:
  TransformScheduleBase(mlir::TypeID transformScheduleID, StringRef rootName,
                        PatternBenefit benefit, MLIRContext *context,
                        ArrayRef<StringRef> generatedNames = {})
      : mlir::Pattern(rootName, benefit, context, generatedNames),
        transformScheduleID(transformScheduleID) {}
  TransformScheduleBase(mlir::TypeID transformScheduleID,
                        MatchInterfaceOpTypeTag tag, TypeID interfaceID,
                        PatternBenefit benefit, MLIRContext *context,
                        ArrayRef<StringRef> generatedNames = {})
      : mlir::Pattern(tag, interfaceID, benefit, context, generatedNames),
        transformScheduleID(transformScheduleID) {}

  virtual ~TransformScheduleBase() = default;

  /// Determine the parameters for this transform schedule for targeting
  /// `rootOp`. If this transform schedule cannot support `rootOp`, then return
  /// failure.
  virtual FailureOr<CodegenScheduleAttrInterface>
  decideParameters(Operation *rootOp,
                   const TransformScheduleOptions &options) const = 0;

  /// Determine a space of possible parameters for this transform schedule for
  /// compiling `rootOp`. If this transform schedule cannot support `rootOp`,
  /// then return failure.
  virtual FailureOr<SmallVector<CodegenScheduleAttrInterface>>
  enumerateParameters(TilingInterface rootOp,
                      const TransformScheduleOptions &options) const = 0;

  /// Generate the schedule body (`transform.sequence` body) targeting `rootOp`
  /// using the provided parameters.
  virtual FailureOr<Value>
  generateSchedule(OpBuilder &b, Location loc, TilingInterface rootOp,
                   Value funcHandle, Attribute parameters,
                   const TransformScheduleOptions &options) const = 0;

  /// Return whether the codegen schedule can support operands/results with the
  /// given element type;
  /// If `requiresVectorType` is true, then the element type must be a valid
  /// vector element type.
  static bool isElementTypeSupported(const mlir::DataLayout &dataLayout,
                                     Type type, bool doesSupportComplex,
                                     bool requiresVectorType);

  /// Returns whether all operand/result types are supported. Not all schedules
  /// handle complex element type so it is treated specially.
  /// If `requiresVectorType` is true, then the element type of all tensors must
  /// be a valid vector element type.
  static bool allTypesAreSupported(const mlir::DataLayout &dataLayout,
                                   Operation *op, bool doesSupportComplex,
                                   bool requiresVectorType);

  /// Return whether the schedule is supported for the given operation.
  virtual bool isSupported(Operation *op,
                           const TransformScheduleOptions &options) const = 0;

  /// Return the TypeID for this transform schedule.
  mlir::TypeID getTypeID() const { return transformScheduleID; }

  /// Return the mnemonic for this transform schedule.
  virtual StringRef getMnemonic() const = 0;

protected:
  using Pattern::Pattern;

  mlir::TypeID transformScheduleID;
};

/// A TransformScheduleRegistry holds a set of transform schedule generators.
/// Typically, users use the MLIRContext-scoped registry that is attached to the
/// KernelDialect (which is unique in each MLIRContext).
///
/// Each TransformSchedule should be associated with a unique TypeID and a
/// unique mnemonic (see the schedules under 'Kernel/TransformSchedules'
/// for example implementations).
///
/// The registry allows for implementation hiding of the schedule generator
/// implementations and it also allows dialect users to declare additional
/// schedule generators outside of the Kernel project.
///
/// All that is required to register a new schedule is to implement the
/// TransformSheduleBase interface and register the schedule as a KernelDialect
/// extension (see registration helper methods below this class declaration).
class TransformScheduleRegistry {
public:
  /// Parse a transform schedule from a string. This matches one of the
  /// pre-defined generators defined in this file.
  FailureOr<const TransformScheduleBase *>
  parseTransformScheduleGenerator(llvm::StringRef str) const;

  void registerTransformScheduleGenerator(
      std::unique_ptr<TransformScheduleBase> generator);

  llvm::DenseMap<mlir::TypeID, std::unique_ptr<TransformScheduleBase>>
      schedules;
  llvm::StringMap<mlir::TypeID> scheduleGeneratorNameMap;
};

class KernelDialect;

namespace detail {
/// Register a transform schedule generator with the given mnemonic and type ID.
/// This is the internal implementation of the public interface and is not
/// intended to be used directly.
void registerTransformScheduleGenerator(
    KernelDialect *dialect, std::unique_ptr<TransformScheduleBase> generator);
} // namespace detail

/// Add an extension function that requires the KernelDialect.
/// Note: This bare functor overload is provided in addition to the
/// std::function variant to enable dialect type deduction, e.g.:
///  registry.addExtension(+[](MLIRContext *ctx, MyDialect *dialect) {
///  ... })
///
/// is equivalent to:
///  registry.addExtension<MyDialect>(
///     [](MLIRContext *ctx, MyDialect *dialect){ ... }
///  )
template <typename ScheduleGenerator>
bool addTransformScheduleGeneratorExtension(DialectRegistry &registry) {
  struct Extension : public DialectExtension<Extension, kernel::KernelDialect> {
    Extension(const Extension &) = default;
    Extension() : DialectExtension<Extension, kernel::KernelDialect>() {}
    ~Extension() override = default;
    void apply(MLIRContext *context, KernelDialect *kernelDialect) const final {
      detail::registerTransformScheduleGenerator(
          kernelDialect, std::make_unique<ScheduleGenerator>(context));
    }
  };
  return registry.addExtension(mlir::TypeID::get<ScheduleGenerator>(),
                               std::make_unique<Extension>());
}

} // namespace mlir::kernel

#endif // MLIR_KERNEL_KERNEL_IR_TRANSFORMSCHEDULEBASE
