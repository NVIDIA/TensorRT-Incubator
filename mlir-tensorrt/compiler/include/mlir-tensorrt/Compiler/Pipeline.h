#ifndef MLIR_TENSORRT_COMPILER_PIPELINE
#define MLIR_TENSORRT_COMPILER_PIPELINE

#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
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

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//

/// A compilation pipeline is an MLIR `PassManager` plus a shared `MainOptions`
/// instance that drives pass selection and configuration.
///
/// Pipelines are constructed with fully-populated options and immediately
/// initialized, which sets up pass manager instrumentation and populates the
/// pass list.
class Pipeline : public mlir::PassManager {
public:
  Pipeline(mlir::MLIRContext *context,
           llvm::IntrusiveRefCntPtr<MainOptions> options);

  virtual ~Pipeline();

  /// Initialize the Pipeline.
  Status initialize();

  /// Return the options for the pipeline.
  const MainOptions &getOptions() const { return *options; }

  /// Return the (mutable) options for the pipeline.
  MainOptions &getOptions() { return *options; }

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

  mlir::PassManager &getPassManager() { return *this; }

protected:
  void setupPassManagerInstrumentation();

  /// Populate the pass manager with the appropriate passes.
  void populatePassManager();

  /// Options for the pipeline.
  llvm::IntrusiveRefCntPtr<MainOptions> options;

  /// A flag to indicate whether the pipeline has been fully initialized.
  bool initialized{false};
};

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_PIPELINE
