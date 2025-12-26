#ifndef MLIR_TENSORRT_COMPILER_PIPELINE
#define MLIR_TENSORRT_COMPILER_PIPELINE

#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mtrt::compiler {

/// Phase describes the overall phases of the compilation pipeline. Not all
/// phases are applicable to all tasks.
enum class Phase {
  ConstantFolding,
  PreClustering,
  PostClustering,
  PreBufferization,
  PostBufferization,
  ExecutorLowering
};

/// PipelineOptions is just like PipelineOptionsBase, but it
/// allows specifying any of the handy "OptionsProviders" defined in Options.h
/// in order to automatically incorporate the options associated with each
/// provider.
template <typename... Providers>
class PipelineOptions : public PipelineOptionsBase {
public:
  PipelineOptions(bool enableDebugOptions = false)
      : PipelineOptionsBase(enableDebugOptions),
        optionProviders(std::make_unique<Providers>(this)...) {}

  /// Access provider value.
  template <typename OptionsProviderT>
  const OptionsProviderT &get() const {
    if constexpr (std::is_same_v<OptionsProviderT, DebugOptions>) {
      if (hasDebugOptions())
        return *this->debugOptions;
      llvm::report_fatal_error(
          "debug options are not enabled on the pipeline options instance");
    } else {
      return *std::get<std::unique_ptr<OptionsProviderT>>(optionProviders);
    }
  }

  /// Access provider value.
  template <typename OptionsProviderT>
  OptionsProviderT &get() {
    if constexpr (std::is_same_v<OptionsProviderT, DebugOptions>) {
      if (hasDebugOptions())
        return *debugOptions;
      llvm::report_fatal_error(
          "debug options are not enabled on the pipeline options instance");
    } else {
      return *std::get<std::unique_ptr<OptionsProviderT>>(optionProviders);
    }
  }

private:
  std::tuple<std::unique_ptr<Providers>...> optionProviders;
};

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//

/// Base class for all "pipelines". A "pipeline" is like a MLIR
/// PassManager, but it has some additional functionality. Primarily, it is
/// associated with a CompilationOptions object which should capture all options
/// that may modify the behavior of the compiler pipeline and a list of
/// extensions. Each concrete subclass type is associated with a unique name
/// (static string literal ID).
///
/// Extensions are loaded from a global registry and will update the owned
/// options object. It is up to the derived class to actually invoke the
/// extension hooks for populating passes inside `populatePassManager`.
///
/// A PipelineBase should never be constructed directly. Instead, pipelines
/// are constructed in a "default" state through the CompilerClient API. They
/// are then lazily initialized using the `initialize` method, which loads
/// extensions and populates the passes.
///
/// Concrete subclasses should implement `populatePassManager`. Subclasses
/// should be implemented by deriving from the `Pipeline` CRTP template
/// rather than directly inheriting from this class.
///
/// The additional methods `openOutputFile` and `translateToTargetFormat` are
/// provided to simplify implementation of the final compilation steps.
class PipelineBase : public mlir::PassManager {
public:
  virtual ~PipelineBase();

  /// Initialize the Pipeline using the given command line options.
  /// This should be called after all extensions have been constructed.
  /// It can only be called once, otherwise it will return an error.
  Status initialize(llvm::ArrayRef<llvm::StringRef> options);

  /// Return the options for the pipeline.
  const PipelineOptionsBase &getOptions() const { return *options; }

  /// Return the (mutable) options for the pipeline.
  PipelineOptionsBase &getOptions() { return *options; }

  /// Open an output file with the given name. If the file cannot be opened,
  /// then the error message is passed through `errorMessage`. If
  /// `outputFileName` is `-`, then the output file corresponds to stdout. If
  /// the `outputFileName` is an absolute path, then that file is opened or
  /// created. If it is a relative path, then it is appended to the artifacts
  /// directory path if the artifacts directory is set and exists. Otherwise,
  /// the output file is opened relative to the current directory. Any existing
  /// file is overwritten.
  std::unique_ptr<llvm::ToolOutputFile>
  openOutputFile(llvm::StringRef outputFileName, std::string &errorMessage,
                 std::optional<llvm::StringRef> overrideExtension = {});

  /// Translate to the final target format.
  mlir::LogicalResult translateToTargetFormat(mlir::ModuleOp module,
                                              llvm::raw_ostream &os);

protected:
  PipelineBase(llvm::StringRef taskName, mlir::MLIRContext *context,
               std::unique_ptr<PipelineOptionsBase> options);

  /// Populate the pass manager with the appropriate passes. This should be
  /// implemented by concrete subclasses. The pass manager is empty when this
  /// method is called, and it will only be called once.
  virtual void populatePassManager() = 0;

  /// Populate pass manager instrumentation (e.g. dumping IR after passes,
  /// timing, debug actions, etc) based on the given options. If the
  /// DebugOptions is nullptr, then the instrumentation and timing are populated
  /// from global CL options.
  void setupPassManagerInstrumentation(const DebugOptions *options);

  const llvm::StringRef name;

  /// Options for the pipeline.
  std::unique_ptr<PipelineOptionsBase> options;

  /// The Extensions associated with this pipeline.
  ExtensionList extensions;

  /// A flag to indicate whether the pipeline has been fully initialized.
  bool initialized{false};
};

/// CRTP base class for pipelines.
template <typename DerivedTaskT, typename OptionsT>
class Pipeline : public PipelineBase {
public:
  using OptionsType = OptionsT;

  Pipeline(mlir::MLIRContext *context, std::unique_ptr<OptionsT> options)
      : PipelineBase(DerivedTaskT::getName(), context, std::move(options)) {}

  const OptionsT &getOptions() {
    return static_cast<const OptionsT &>(*this->options);
  }

  using Base = Pipeline;
};

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_PIPELINE
