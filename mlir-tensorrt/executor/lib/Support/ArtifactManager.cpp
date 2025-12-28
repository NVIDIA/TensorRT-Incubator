//===- ArtifactManager.cpp -----------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "mlir-executor/Support/ArtifactManager.h"
#include "mlir-executor/Utils/SerializationUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/VirtualOutputBackends.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;
using namespace mtrt::compiler;

#define DEBUG_TYPE "artifact-manager"

template <typename... Ts>
static void debugAM(const ArtifactManager *self, const char *fmt,
                    Ts &&...args) {
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("ArtifactManager[{0}] ",
                                           static_cast<const void *>(self))
                          << llvm::formatv(fmt, std::forward<Ts>(args)...)
                          << "\n");
}

static std::string sha256Hex(ArrayRef<uint8_t> bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  auto digest = hasher.final();
  return llvm::toHex(digest, /*LowerCase=*/true);
}

/// Modifies the `name` in a way that it becomes suitable for artifact
/// relative paths.
static std::string sanitizeName(StringRef name) {
  std::string processedStr = name.str();
  std::replace_if(
      processedStr.begin(), processedStr.end(),
      [](char c) {
        return !llvm::isAlnum(c) || c == '/' || c == '.' || c == '\\';
      },
      '_');
  return processedStr;
}

static std::string safeModuleName(mlir::ModuleOp module) {
  if (auto name = module.getSymName())
    return sanitizeName(*name);
  return "unnamed_module";
}

ArtifactProducerInfo::ArtifactProducerInfo(llvm::StringRef pass,
                                           mlir::Operation *producerSymbolOp)
    : pass(pass.str()),
      opName(producerSymbolOp->getName().getStringRef().str()),
      symbol(mlir::SymbolTable::getSymbolName(producerSymbolOp).str()) {
  llvm::raw_string_ostream os(loc);
  producerSymbolOp->getLoc().print(os);
  os.flush();
  loc = os.str();
}

std::string mtrt::compiler::createCanonicalArtifactRelativePath(
    llvm::ArrayRef<llvm::StringRef> fqnComponents, llvm::StringRef extension) {
  llvm::SmallString<256> path;
  for (llvm::StringRef component : fqnComponents.drop_back()) {
    llvm::sys::path::append(path, sanitizeName(component));
  }

  std::string extensionStr = extension.str();
  if (!extension.empty() && !extension.starts_with("."))
    extensionStr = "." + extensionStr;

  llvm::sys::path::append(path,
                          sanitizeName(fqnComponents.back()) + extensionStr);
  return path.str().str();
}

/// Return pairs of (sanitized op name, symbol name) for `op` and all parent
/// operations. Op names are sanitized by replacing periods with underscores.
/// The pairs are returned in order of outer-most to inner-most (ancestors of
/// `op` first, `op` last). This information is used to construct the directory
/// tree for the `FileTreeIRPrinterConfig` below.
/// The counter for `op` will be incremented by this call.
static SmallVector<llvm::StringRef> getOpAndSymbolNames(Operation *op) {
  SmallVector<llvm::StringRef> pathElements;
  Operation *iter = op;
  while (iter) {
    StringAttr symbolNameAttr =
        iter->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    llvm::StringRef symbolName =
        symbolNameAttr ? symbolNameAttr.strref() : "no_symbol_name";
    pathElements.push_back(symbolName);
    iter = iter->getParentOp();
  }
  // Return in the order of top level (module) down to `op`.
  std::reverse(pathElements.begin(), pathElements.end());
  return pathElements;
}

static std::string guessExtension(ArtifactKind kind) {
  switch (kind) {
  case ArtifactKind::ConstantBlob:
    return ".bin";
  case ArtifactKind::PTXModule:
    return ".ptx";
  case ArtifactKind::TRTEngine:
    return ".trt_plan.bin";
  case ArtifactKind::Manifest:
    return ".json";
  case ArtifactKind::Unknown:
    break;
  }
  return "";
}

std::string
mtrt::compiler::createCanonicalArtifactRelativePath(Operation *op,
                                                    ArtifactKind kind) {
  SmallVector<llvm::StringRef> pathElements = getOpAndSymbolNames(op);
  return createCanonicalArtifactRelativePath(pathElements,
                                             guessExtension(kind));
}

//===----------------------------------------------------------------------===//
// ArtifactManager
//===----------------------------------------------------------------------===//

ArtifactManager::ArtifactManager(Options options)
    : options(options), vfs(llvm::vfs::getRealFileSystem()),
      outputBackend(
          llvm::makeIntrusiveRefCnt<llvm::vfs::OnDiskOutputBackend>()) {
  debugAM(this, "construct (default backends) artifactsDir='{0}'",
          this->options.artifactsDirectory);
  setArtifactsDirectory(this->options.artifactsDirectory);
}

ArtifactManager::ArtifactManager(
    Options options, llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    llvm::IntrusiveRefCntPtr<llvm::vfs::OutputBackend> outputBackend)
    : options(options), vfs(std::move(fs)),
      outputBackend(std::move(outputBackend)) {
  assert(vfs && outputBackend &&
         "filesystem and output backend must be provided");
  debugAM(this, "construct (injected backends) artifactsDir='{0}'",
          this->options.artifactsDirectory);
  setArtifactsDirectory(this->options.artifactsDirectory);
}

void ArtifactManager::setArtifactsDirectory(std::string artifactsDirectory) {
  this->options.artifactsDirectory = std::move(artifactsDirectory);

  std::string cwd = [&]() {
    llvm::ErrorOr<std::string> cwd = vfs->getCurrentWorkingDirectory();
    if (cwd)
      return *cwd;
    return std::string("");
  }();

  if (this->options.artifactsDirectory.empty()) {
    this->options.artifactsDirectory = cwd;
  } else if (llvm::sys::path::is_relative(this->options.artifactsDirectory)) {
    llvm::SmallString<256> absPath(this->options.artifactsDirectory);
    llvm::sys::path::make_absolute(cwd, absPath);
    this->options.artifactsDirectory = absPath.str().str();
  }
}

llvm::Error ArtifactManager::flushFilesAndReset() {
  debugAM(this, "flushFilesAndReset shouldKeep={0} outputFiles={1}", shouldKeep,
          outputFiles.size());
  if (shouldKeep) {
    for (llvm::vfs::OutputFile &file : outputFiles) {
      debugAM(this, "keep output file path='{0}'", file.getPath());
      llvm::Error error = llvm::Error::success();
      llvm::handleAllErrors(file.keep(), [&](const llvm::ErrorInfoBase &EIB) {
        error = llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "failed to keep output file: " + file.getPath() + ": " +
                EIB.message());
      });
      if (error)
        return error;
    }
  }
  return llvm::Error::success();
}

ArtifactManager::~ArtifactManager() {
  debugAM(this, "destroy");
  llvm::logAllUnhandledErrors(flushFilesAndReset(), llvm::errs());
}

static std::string kindToString(ArtifactKind kind) {
  switch (kind) {
  case ArtifactKind::ConstantBlob:
    return "ConstantBlob";
  case ArtifactKind::PTXModule:
    return "PTXModule";
  case ArtifactKind::TRTEngine:
    return "TRTEngine";
  case ArtifactKind::Manifest:
    return "Manifest";
  case ArtifactKind::Unknown:
    break;
  }
  return "Unknown";
}

[[maybe_unused]] static llvm::Expected<std::pair<std::string, uint64_t>>
computeFileSha256AndSize(llvm::vfs::FileSystem *vfs, llvm::StringRef absPath) {
  llvm::SmallString<256> absPathStorage(absPath.str());
  if (llvm::sys::path::is_relative(absPathStorage)) {
    if (std::error_code st = vfs->makeAbsolute(absPathStorage))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "failed to make absolute path: " +
                                         absPath.str());
  }

  // Hashing of temp/committed output files is done via the real filesystem, as
  // these are always materialized on disk.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> mb =
      vfs->getBufferForFile(absPathStorage);
  if (!mb)
    return llvm::createStringError(mb.getError(),
                                   "failed to read file for hashing: " +
                                       absPathStorage.str());
  llvm::StringRef buf = (*mb)->getBuffer();
  std::string hex = sha256Hex(llvm::arrayRefFromStringRef(buf));
  return std::make_pair(hex, static_cast<uint64_t>(buf.size()));
}

llvm::Expected<std::unique_ptr<UnfinalizedArtifactRef>>
ArtifactManager::addOutputFile(llvm::StringRef relPath, ArtifactKind kind) {
  if (llvm::sys::path::is_absolute(relPath)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "ArtifactManager::addOutputFile requires a "
                                   "relative path, got absolute path: " +
                                       relPath.str());
  }

  // Make the path absolute.
  llvm::SmallString<256> absPath(relPath);
  llvm::sys::path::make_absolute(options.artifactsDirectory, absPath);

  debugAM(this, "addOutputFile relPath='{0}' absPath='{1}' kind='{2}'", relPath,
          absPath.str(), kindToString(kind));

  llvm::Expected<llvm::vfs::OutputFile> file = outputBackend->createFile(
      absPath, llvm::vfs::OutputConfig().setAtomicWrite().setDiscardOnSignal());
  if (!file)
    return file.takeError();

  auto artifactRecord =
      std::make_unique<ArtifactRecord>(ArtifactProducerInfo());
  artifactRecord->setKind(kindToString(kind));
  artifactRecord->setRelPath(relPath.str());

  debugAM(this, "addOutputFile created relPath='{0}'", relPath);

  std::lock_guard<std::mutex> lock(mutex);
  outputFiles.push_back(std::move(*file));
  llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>> proxy =
      outputFiles.back().createProxy();
  if (!proxy)
    return proxy.takeError();
  return std::unique_ptr<UnfinalizedArtifactRef>(
      new UnfinalizedArtifactRef{std::move(artifactRecord), std::move(*proxy)});
}

llvm::Expected<FinalizedArtifactRef> ArtifactManager::finalizeOutputFile(
    std::unique_ptr<UnfinalizedArtifactRef> ref) {
  debugAM(this, "finalizeOutputFile relPath='{0}'",
          ref && ref->record ? llvm::StringRef(ref->record->relPath) : "");

  ref->stream->flush();
  ref->stream.reset(nullptr);

  std::string relPath;
  {
    std::lock_guard<std::mutex> lock(mutex);
    records.push_back(std::move(*ref->record));
    relPath = records.back().relPath;
  }

  return FinalizedArtifactRef{relPath};
}

namespace {
struct SerializationInterfaceImpl : public SerializationInterface {
  SerializationInterfaceImpl(const DataLayout &dataLayout,
                             llvm::raw_pwrite_stream &stream)
      : SerializationInterface(dataLayout), stream(stream) {}

  llvm::raw_pwrite_stream &stream;

  LogicalResult serialize(const char *data, size_t size, Type elementType,
                          uint64_t align) override {
    stream.write(data, size);
    return success();
  }
};
} // namespace

llvm::Expected<FinalizedArtifactRef> ArtifactManager::addElementsAttr(
    llvm::StringRef relPath, ElementsAttr attr, const DataLayout &dataLayout,
    ArtifactKind kind, const ArtifactProducerInfo &producer,
    std::optional<mlir::Location> loc, llvm::json::Object attributes) {
  debugAM(this, "addElementsAttr relPath='{0}' kind='{1}'", relPath,
          kindToString(kind));

  // Create temp file and serialize directly into its FD.
  auto stream = addOutputFile(relPath, kind);
  if (!stream)
    return stream.takeError();

  mlir::Location location =
      loc.has_value() ? *loc : UnknownLoc::get(attr.getContext());

  SerializationInterfaceImpl serializer(dataLayout, *(*stream)->stream);
  if (failed(mlir::serializeElementsAttr(location, attr, dataLayout, serializer,
                                         {})))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to serialize elements attribute");

  // Update record information.
  (*stream)
      ->record->setSizeBytes((*stream)->stream->tell())
      .setAttributes(std::move(attributes))
      .setProducer(producer);

  return finalizeOutputFile(std::move(*stream));
}

llvm::Error ArtifactManager::writeManifest(mlir::ModuleOp module) {
  size_t numRecords = 0;
  {
    std::lock_guard<std::mutex> lock(mutex);
    numRecords = records.size();
  }
  debugAM(this, "writeManifest module='{0}' records={1}",
          safeModuleName(module), numRecords);
  llvm::json::Object root;
  root["schema_version"] = 1;
  {
    std::lock_guard<std::mutex> lock(mutex);
    root["module_name"] = safeModuleName(module);
  }

  llvm::json::Array artifacts;
  llvm::SmallVector<ArtifactRecord, 8> snapshot;
  {
    std::lock_guard<std::mutex> lock(mutex);
    snapshot = std::move(records);
    records.clear();
  }

  for (ArtifactRecord &rec : snapshot) {
    llvm::json::Object obj;
    obj["kind"] = rec.kind;
    obj["relpath"] = rec.relPath;
    obj["size_bytes"] = static_cast<int64_t>(rec.sizeBytes);
    llvm::json::Object producer;
    producer["pass"] = rec.producer.pass;
    producer["op_name"] = rec.producer.opName;
    producer["symbol"] = rec.producer.symbol;
    producer["loc"] = rec.producer.loc;
    obj["producer"] = std::move(producer);
    obj["attributes"] = std::move(rec.attributes);
    artifacts.push_back(std::move(obj));
  }
  root["artifacts"] = std::move(artifacts);

  std::string err;
  llvm::Expected<std::unique_ptr<UnfinalizedArtifactRef>> of =
      addOutputFile("manifest.json", ArtifactKind::Manifest);
  if (!of)
    return of.takeError();

  *(*of)->stream << llvm::formatv("{0:2}\n",
                                  llvm::json::Value(std::move(root)));

  auto finalized = finalizeOutputFile(std::move(*of));
  if (!finalized)
    return finalized.takeError();

  debugAM(this, "writeManifest done relPath='{0}'", finalized->relPath);
  return llvm::Error::success();
}

mlir::FailureOr<mlir::ElementsAttr>
mtrt::compiler::createElementsAttrFromFile(Location loc, StringRef name,
                                           llvm::StringRef filePath) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(filePath);
  if (!buffer)
    return emitError(loc, "failed to read file ") << filePath;

  auto resourceAttr = mlir::DenseResourceElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>((*buffer)->getBufferSize())},
                            IntegerType::get(loc.getContext(), 8)),
      name,
      mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign(llvm::ArrayRef<char>(
          (*buffer)->getBufferStart(), (*buffer)->getBufferSize())));

  return ElementsAttr(resourceAttr);
}
