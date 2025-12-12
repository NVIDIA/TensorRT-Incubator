//===- TransformSchedules.h -----------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Declarations for the help functions for deciding and generating the
/// transform schedules.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_KERNEL_TRANSFORMSCHEDULES_TRANSFORMSCHEDULES_H
#define MLIR_TENSORRT_DIALECT_KERNEL_TRANSFORMSCHEDULES_TRANSFORMSCHEDULES_H

#include "mlir-kernel/Kernel/IR/Dialect.h"
#include "mlir-kernel/Kernel/IR/Enums.h"
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir-kernel/Kernel/IR/TransformScheduleBase.h"
#include "mlir-kernel/Kernel/TransformSchedules/TransformBuilder.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"

namespace mlir::kernel {

constexpr StringRef kParametersAttrName = "kernel.parameters";
constexpr StringRef kTargetFuncAttrName = "kernel.target_func";
constexpr StringRef kRootGenericAttrName = "kernel.root";

namespace detail {
/// A transform schedule generator specialized to a particular parameter
/// attribute type and operation or interface type.
///
/// Derived classes typically should use the specialized
/// `LinalgTransformSchedule` or `TransformSchedule` classes to declare
/// the root operation type as LinalgOp interface or a different specific op
/// type.
template <typename DerivedType, typename ParamsAttrTy, typename BaseOpTy>
class RootedTransformSchedule : public TransformScheduleBase {
public:
  RootedTransformSchedule(MatchInterfaceOpTypeTag tag, TypeID interfaceID,
                          PatternBenefit benefit, MLIRContext *context,
                          ArrayRef<StringRef> generatedNames = {})
      : TransformScheduleBase(mlir::TypeID::get<DerivedType>(), tag,
                              interfaceID, benefit, context, generatedNames) {}
  RootedTransformSchedule(StringRef rootName, PatternBenefit benefit,
                          MLIRContext *context,
                          ArrayRef<StringRef> generatedNames = {})
      : TransformScheduleBase(mlir::TypeID::get<DerivedType>(), rootName,
                              benefit, context, generatedNames) {}

  virtual ~RootedTransformSchedule() {}

  FailureOr<CodegenScheduleAttrInterface>
  decideParameters(Operation *rootOp,
                   const TransformScheduleOptions &options) const final {
    FailureOr<ParamsAttrTy> params =
        decideParameters(cast<BaseOpTy>(rootOp), options);
    if (failed(params))
      return failure();
    return *params ? cast<CodegenScheduleAttrInterface>(*params)
                   : CodegenScheduleAttrInterface{};
  }

  FailureOr<SmallVector<CodegenScheduleAttrInterface>>
  enumerateParameters(TilingInterface rootOp,
                      const TransformScheduleOptions &options) const final {
    FailureOr<SmallVector<ParamsAttrTy>> attrs =
        enumerateParameters(cast<BaseOpTy>(*rootOp), options);
    if (failed(attrs) || attrs->empty())
      return failure();
    return llvm::map_to_vector(
        *attrs, [](ParamsAttrTy attr) -> CodegenScheduleAttrInterface {
          return cast<CodegenScheduleAttrInterface>(attr);
        });
  }

  FailureOr<Value>
  generateSchedule(OpBuilder &b, Location loc, TilingInterface rootOp,
                   Value funcHandle, Attribute parameters,
                   const TransformScheduleOptions &options) const final {
    TransformIRBuilder b_(loc, b);
    return generateSchedule(b_, cast<BaseOpTy>(*rootOp), funcHandle,
                            dyn_cast_if_present<ParamsAttrTy>(parameters),
                            options);
  }

  bool isSupported(Operation *op,
                   const TransformScheduleOptions &options) const final {
    return isSupported(options, cast<BaseOpTy>(op));
  }
  virtual bool isSupported(const TransformScheduleOptions &options,
                           BaseOpTy op) const = 0;

  virtual FailureOr<ParamsAttrTy>
  decideParameters(BaseOpTy rootOp,
                   const TransformScheduleOptions &options) const {
    llvm_unreachable("must provide override of decideParameters");
    return failure();
  }
  virtual FailureOr<SmallVector<ParamsAttrTy>>
  enumerateParameters(BaseOpTy rootOp,
                      const TransformScheduleOptions &options) const {
    llvm_unreachable("must provide override of enumerateParameters");
    return failure();
  }
  virtual FailureOr<Value>
  generateSchedule(TransformIRBuilder &b, BaseOpTy rootOp, Value funcHandle,
                   ParamsAttrTy params,
                   const TransformScheduleOptions &options) const {
    llvm_unreachable("must provide override of generateSchedule");
    return failure();
  }

protected:
  using TransformScheduleBase::TransformScheduleBase;
};
} // namespace detail

//===----------------------------------------------------------------------===//
// Helper methods for concrete schedule generators
//===----------------------------------------------------------------------===//

/// Return an ArrayAttr containing a GPUBlockMappingAttr for each non-zero
/// element in `tileShape`.
FailureOr<ArrayAttr> getCTADistributionMappingAttr(Location loc,
                                                   ArrayRef<int64_t> tileShape);

/// Return an ArrayAttr containing a GPUThreadMappingAttr for each non-zero
/// element in `tileShape`.
FailureOr<ArrayAttr>
getThreadDistributionMappingAttr(Location loc, ArrayRef<int64_t> tileShape);

/// Return an ArrayAttr containing a GPUWarpMappingAttr for each non-zero
/// element in `tileShape`.
FailureOr<ArrayAttr>
getWarpDistributionMappingAttr(Location loc, ArrayRef<int64_t> tileShape);

/// LinalgTransformSchedule is a TransformSchedule specialized to the LinalgOps
/// interface.
///
/// Derived classes must provide all the virtual methods of
/// `detail::RootedTransformSchedule`.
template <typename DerivedType, typename ParamsAttrTy>
class LinalgTransformSchedule
    : public detail::RootedTransformSchedule<DerivedType, ParamsAttrTy,
                                             linalg::LinalgOp> {
public:
  LinalgTransformSchedule(MLIRContext *context, PatternBenefit benefit = 1)
      : detail::RootedTransformSchedule<DerivedType, ParamsAttrTy,
                                        linalg::LinalgOp>(
            Pattern::MatchInterfaceOpTypeTag(),
            linalg::LinalgOp::getInterfaceID(), benefit, context) {}
};

/// A TransformSchedule is a transform IR generator whose match scope is limited
/// to operations of a specific type.
///
/// Derived classes must provide all the virtual methods of
/// `detail::RootedTransformSchedule`.
template <typename DerivedType, typename OpType, typename ParamsAttrTy>
class TransformSchedule
    : public detail::RootedTransformSchedule<DerivedType, ParamsAttrTy,
                                             OpType> {
public:
  TransformSchedule(MLIRContext *context, PatternBenefit benefit = 1)
      : detail::RootedTransformSchedule<DerivedType, ParamsAttrTy, OpType>(
            OpType::getOperationName(), benefit, context) {}
};

///===----------------------------------------------------------------------===//
// TransformScheduleSelector
//===----------------------------------------------------------------------===//

/// The `TransformScheduleSelector` is a convenience wrapper for repeatedly
/// querying the same `TransformScheduleRegistry` with different operations
/// in order to enumerate schedules and schedule options. It stores a single
/// "TransformScheduleOptions" to use for all queries.
///
/// The available schedules are by default retrieved from the given
/// `TransformScheduleRegistry` that is attached to the KernelDialect.
class TransformScheduleSelector {
public:
  /// A function that can be used to pre-filter the set of schedule generators
  /// allowed to be selected.
  using ScheduleFilterFunc = std::function<bool(StringRef mnemonic)>;

  /// Construct a TransformScheduleSelector from the registry attached to the
  /// KernelDialect.
  TransformScheduleSelector(MLIRContext *context,
                            const TransformScheduleOptions &options,
                            ScheduleFilterFunc scheduleFilter = {});

  /// Construct a TransformScheduleSelector from a given registry and
  /// TransformScheduleOptions.
  TransformScheduleSelector(const TransformScheduleRegistry &registry,
                            TransformScheduleOptions options,
                            ScheduleFilterFunc scheduleFilter = {});

  /// Return the schedule generator for the given operation.
  FailureOr<const TransformScheduleBase *>
  getScheduleGenerator(Operation *op) const;

  /// Return all the schedule generators.
  const std::vector<std::unique_ptr<TransformScheduleBase>> &
  getScheduleGenerators() const;

  /// Return the options for this selector.
  const TransformScheduleOptions &getOptions() const { return options; }

private:
  const TransformScheduleRegistry &registry;
  TransformScheduleOptions options;
  llvm::DenseMap<mlir::OperationName,
                 std::vector<const TransformScheduleBase *>>
      schedules;

  ScheduleFilterFunc scheduleFilter;
};

//===----------------------------------------------------------------------===//
// Builtin Generators
//===----------------------------------------------------------------------===//

/// Create a Fallback schedule.
std::unique_ptr<TransformScheduleBase>
createFallbackTransformSchedule(MLIRContext *context, PatternBenefit benefit);

/// Create a Scatter schedule.
std::unique_ptr<TransformScheduleBase>
createScatterTransformSchedule(MLIRContext *context, PatternBenefit benefit);

/// Registration methods for builtin transform schedules.
void registerFallbackTransformSchedule(DialectRegistry &registry);
void registerScatterTransformSchedule(DialectRegistry &registry);

inline void
registerBuiltinTransformScheduleGenerators(DialectRegistry &registry) {
  registerFallbackTransformSchedule(registry);
  registerScatterTransformSchedule(registry);
}

} // namespace mlir::kernel

#endif // MLIR_TENSORRT_DIALECT_KERNEL_TRANSFORMSCHEDULES_TRANSFORMSCHEDULES_H
