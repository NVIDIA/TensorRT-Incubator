//===- ArtifactManager.h ---------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Shared utilities for managing compiler-produced side artifacts (constants,
/// PTX, TensorRT engines, etc.) and emitting a manifest describing them.
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTOR_SUPPORT_ARTIFACTMANAGER
#define MLIR_EXECUTOR_SUPPORT_ARTIFACTMANAGER

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/VirtualOutputBackend.h"
#include "llvm/Support/VirtualOutputFile.h"

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace llvm::vfs {
class FileSystem;
} // namespace llvm::vfs

namespace llvm {
class raw_pwrite_stream;
} // namespace llvm

namespace llvm::sys::fs {
class TempFile;
} // namespace llvm::sys::fs

namespace mlir {
class DataLayout;
} // namespace mlir

namespace mtrt::compiler {

/// High-level classification of artifacts emitted by the compiler.
///
/// This is used for:
/// - manifest reporting (human + tooling)
/// - choosing file extensions/layout under the artifacts directory
/// - runtime requirements (e.g. CUDA/PTX vs TensorRT engines)
enum class ArtifactKind {
  ConstantBlob,
  PTXModule,
  TRTEngine,
  Manifest,
  Unknown,
};

/// Identifies the compiler component responsible for producing an artifact.
///
/// This information is recorded into `manifest.json` and is intended for:
/// - debugging ("where did this file come from?")
/// - tracing and reproducibility ("which op / symbol produced it?")
/// - user-facing diagnostics
///
/// The fields are stored as plain strings to keep the manifest stable across
/// builds and to avoid requiring MLIR contexts at manifest read time.
struct ArtifactProducerInfo {
  std::string pass;
  std::string opName;
  std::string symbol;
  std::string loc;

  ArtifactProducerInfo() = default;

  /// Construct with explicit strings (typically for non-MLIR producers).
  ArtifactProducerInfo(llvm::StringRef pass, llvm::StringRef opName,
                       llvm::StringRef symbol, llvm::StringRef loc)
      : pass(pass.str()), opName(opName.str()), symbol(symbol.str()),
        loc(loc.str()) {}

  /// Construct producer info from an MLIR operation that is also a symbol.
  /// The `pass` name is a short textual identifier, usually the pass argument
  /// (e.g. `"convert-host-to-emitc"`).
  ///
  /// This is a convenience helper used throughout conversions so we record
  /// consistent fields without each pass re-implementing location printing.
  ArtifactProducerInfo(llvm::StringRef pass, mlir::Operation *producerSymbolOp);
};

/// A record serialized into the manifest describing a produced artifact.
///
/// Notes:
/// - `attributes` is intentionally schema-less in v1 so individual producers
///   can evolve independently (e.g. constants may record element type/shape).
/// - The record is "move-heavy" because JSON values are move-only in LLVM.
struct ArtifactRecord {
  std::string kind;
  std::string relPath;
  uint64_t sizeBytes{0};
  ArtifactProducerInfo producer;
  llvm::json::Object attributes;

  /// Construct a record anchored to a producer; other fields are filled by
  /// `ArtifactManager` as it finalizes the artifact.
  ArtifactRecord(ArtifactProducerInfo producerInfo)
      : producer(std::move(producerInfo)) {}

  ArtifactRecord &setProducer(ArtifactProducerInfo producerInfo) {
    this->producer = std::move(producerInfo);
    return *this;
  }

  ArtifactRecord &setKind(std::string kind) {
    this->kind = std::move(kind);
    return *this;
  }

  ArtifactRecord &setRelPath(std::string relPath) {
    this->relPath = std::move(relPath);
    return *this;
  }

  ArtifactRecord &setSizeBytes(uint64_t sizeBytes) {
    this->sizeBytes = sizeBytes;
    return *this;
  }

  ArtifactRecord &setAttributes(llvm::json::Object attributes) {
    this->attributes = std::move(attributes);
    return *this;
  }
};

/// Construct a canonical path for an artifact.
std::string createCanonicalArtifactRelativePath(
    llvm::ArrayRef<llvm::StringRef> fqnComponents, llvm::StringRef extension);

/// Construct a canonical path for an artifact.
///
/// This is a convenience helper used throughout conversions so we record
/// consistent fields without each pass re-implementing location printing.
std::string createCanonicalArtifactRelativePath(mlir::Operation *op,
                                                ArtifactKind kind);

/// A reference to an artifact as emitted by `ArtifactManager`.
///
/// This is the *only* information producers should embed back into
/// generated IR/code: treat `relPath` as the runtime reference and keep
/// `sha256` for debugging/validation.
struct FinalizedArtifactRef {
  /// Relative path (relative to the compilation task artifacts directory).
  llvm::StringRef relPath;
};

/// A reference to an artifact that is being written.
struct UnfinalizedArtifactRef {
  /// ArtifactRecord information.
  std::unique_ptr<ArtifactRecord> record;

  /// Stream for writing the artifact contents.
  std::unique_ptr<llvm::raw_pwrite_stream> stream;
};

class ArtifactManager;

class ArtifactManager : public llvm::ThreadSafeRefCountedBase<ArtifactManager> {
public:
  /// Configuration knobs for artifact management.
  ///
  /// These options are intended to be configured by drivers (e.g.
  /// `mlir-tensorrt-compiler`) and then shared across all passes in a pipeline.
  struct Options {
    /// The directory to store artifacts.
    std::string artifactsDirectory;

    Options(std::string artifactsDirectory = "./artifacts")
        : artifactsDirectory(std::move(artifactsDirectory)) {}
  };

  /// Construct an artifact manager that uses the default virtual filesystem
  /// and output backend.
  explicit ArtifactManager(Options options = Options());

  /// Construct an artifact manager that uses the provided virtual filesystem
  /// for *reading* input files (e.g. staging a pre-existing PTX file).
  ///
  /// Notes:
  /// - Output artifacts are still committed to the host filesystem under
  ///   `artifactsDirectory`, because `llvm::vfs::FileSystem` does not provide a
  ///   uniform write/rename API across all implementations.
  /// - Passing an overlay or in-memory filesystem is useful for tests that want
  ///   to stage inputs without requiring them on disk.
  explicit ArtifactManager(
      Options options, llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
      llvm::IntrusiveRefCntPtr<llvm::vfs::OutputBackend> outputBackend);
  ~ArtifactManager();

  /// Return the root directory for this compilation's artifacts.
  llvm::StringRef getArtifactsDirectory() const {
    return options.artifactsDirectory;
  }

  /// Write an ElementsAttr payload to a managed artifact file and return the
  /// relative path to reference from generated code.
  ///
  /// The payload is written to a temporary file and then atomically committed
  /// into the final location under `artifacts/by-id`.
  llvm::Expected<FinalizedArtifactRef>
  addElementsAttr(llvm::StringRef relPath, mlir::ElementsAttr attr,
                  const mlir::DataLayout &dataLayout, ArtifactKind kind,
                  const ArtifactProducerInfo &producer,
                  std::optional<mlir::Location> loc = std::nullopt,
                  llvm::json::Object attributes = {});

  /// Write (or overwrite) the manifest into
  /// `<artifactsDirectory>/manifest.json`.
  ///
  /// This is intended to be called *once per compilation* by a driver after
  /// the full pipeline has run so all passes can contribute entries.
  ///
  /// Thread safety:
  /// - `ArtifactManager` is safe to use from multiple passes when MLIR runs
  ///   pipelines in parallel; internal state is protected by a mutex.
  llvm::Error writeManifest(mlir::ModuleOp module);

  /// Sets whether to keep all artifacts on teardown.
  void setShouldKeep(bool shouldKeep) { this->shouldKeep = shouldKeep; }

  llvm::vfs::OutputBackend &getOutputBackend() { return *outputBackend; }

private:
  void setArtifactsDirectory(std::string artifactsDirectory);

  /// Flush all output files. Should be called when done compiling a module
  /// after optionally writing the manifest.
  llvm::Error flushFilesAndReset();

  /// Create an output file for an artifact. This returns an
  /// 'UnfinalizedArtifactRef' that can be used to write the artifact contents
  /// to a temporary file. The file is not committed to the filesystem until
  /// `finalizeOutputFile` is called, at which point it is moved to its final
  /// location in the artifacts directory. If the `finalizeOutputFile` is not
  /// called, then the file is deleted when the `ArtifactManager` is destroyed
  /// or when a signal is received.
  llvm::Expected<std::unique_ptr<UnfinalizedArtifactRef>>
  addOutputFile(llvm::StringRef relPath, ArtifactKind kind);

  /// Finalize the artifact.
  llvm::Expected<FinalizedArtifactRef>
  finalizeOutputFile(std::unique_ptr<UnfinalizedArtifactRef> ref);

  Options options;

  // Requirements and bookkeeping.
  llvm::SmallVector<ArtifactRecord, 8> records;

  mutable std::mutex mutex;

  /// Return true if we should keep all artifacts on teardown. Defaults to false
  /// to avoid accidentally keeping artifacts when not intended.
  bool shouldKeep{false};

  /// Virtual filesystem used for reading producer-supplied input files.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs;

  /// Virtual output backend used for writing artifacts.
  llvm::IntrusiveRefCntPtr<llvm::vfs::OutputBackend> outputBackend;

  /// List of VFS output files.
  std::list<llvm::vfs::OutputFile> outputFiles;
};

/// Construct an ResourceElementsAttr from the given file. The `name` is used as
/// the key for the new resource blob. The Location is used for the MLIRContext
/// as well as for error locations if the file cannot be read.
mlir::FailureOr<mlir::ElementsAttr>
createElementsAttrFromFile(mlir::Location loc, llvm::StringRef name,
                           llvm::StringRef filePath);

} // namespace mtrt::compiler

#endif // MLIR_EXECUTOR_SUPPORT_ARTIFACTMANAGER
