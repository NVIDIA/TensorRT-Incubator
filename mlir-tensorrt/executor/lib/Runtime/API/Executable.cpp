#include "mlir-executor/Runtime/API/Executable.h"
#include "mlir-executor/Runtime/API/ExecutableFlatbuffer.h"
#include "llvm/Support/FormatAdapters.h"

using namespace mtrt;

//===----------------------------------------------------------------------===//
// Scalar Type
//===----------------------------------------------------------------------===//

ScalarTypeCode mtrt::parseElementType(std::string_view str) {
  const char *const *names = mtrt::flat::EnumNamesScalarTypeCode();
  const ScalarTypeCode *values = mtrt::flat::EnumValuesScalarTypeCode();
  // Flatbuffers' 'enum::MAX' is inclusive (equals largest value).
  constexpr unsigned maxEnum = static_cast<unsigned>(mtrt::ScalarTypeCode::MAX);
  for (unsigned i = 0; i <= maxEnum; i++) {
    if (str == names[i])
      return values[i];
  }
  return ScalarTypeCode::unknown;
}

int64_t mtrt::getBitsPerElement(ScalarTypeCode elType) {
  switch (elType) {
  case ScalarTypeCode::i64:
  case ScalarTypeCode::f64:
    return 64;
  case ScalarTypeCode::f32:
  case ScalarTypeCode::i32:
    return 32;
  case ScalarTypeCode::f16:
  case ScalarTypeCode::bf16:
  case ScalarTypeCode::i16:
    return 16;
  case ScalarTypeCode::i8:
  case ScalarTypeCode::ui8:
  case ScalarTypeCode::f8e4m3fn:
    return 8;
  case ScalarTypeCode::i4:
    return 4;
  case ScalarTypeCode::f4e2m1fn:
    return 4;
  // We treat i1 types as having byte-level storage currently.
  case ScalarTypeCode::i1:
    return 8;
  case ScalarTypeCode::complex32:
    return 64;
  case ScalarTypeCode::complex64:
    return 128;
  default:
    llvm_unreachable("unhandled element type bit width conversion");
  }
}

ScalarType::ScalarType(ScalarTypeCode code) : code(code) {
  assert(code != ScalarTypeCode::unknown && "expected known element type code");
}

StatusOr<ScalarType> mtrt::ScalarType::fromString(std::string_view str) {
  auto code = parseElementType(str);
  assert(code != ScalarTypeCode::unknown && "expected known element type code");
  if (code != ScalarTypeCode::unknown)
    return ScalarType(code);
  return getStatusWithMsg(StatusCode::InvalidArgument, "unknown element type (",
                          str, ")");
}

int64_t mtrt::ScalarType::getBitWidth() const {
  int64_t result = getBitsPerElement(code);
  assert(result != 0 && "expected positive bitwidth");
  return result;
}

StatusOr<ScalarTypeCode>
mtrt::ScalarType::getIntegerTypeWithBitWidth(int64_t bitWidth) {
  switch (bitWidth) {
  case 4:
    return ScalarTypeCode::i4;
  case 8:
    return ScalarTypeCode::i8;
  case 16:
    return ScalarTypeCode::i16;
  case 32:
    return ScalarTypeCode::i32;
  case 64:
    return ScalarTypeCode::i64;
  }
  return getInvalidArgStatus("unknown integer type with bit width ({0})",
                             bitWidth);
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

PointerType mtrt::parsePointerType(llvm::StringRef str) {
  if (str == "host")
    return PointerType::host;
  if (str == "pinned_host")
    return PointerType::pinned_host;
  if (str == "device")
    return PointerType::device;
  if (str == "unified")
    return PointerType::unified;
  return PointerType::unknown;
}

llvm::raw_ostream &mtrt::operator<<(llvm::raw_ostream &os,
                                    PointerType ptrType) {
  return os << mtrt::flat::EnumNamePointerType(ptrType);
}

llvm::StringRef mtrt::stringifyPointerType(PointerType ptrType) {
  return mtrt::flat::EnumNamePointerType(ptrType);
}

bool mtrt::isDeviceVisible(PointerType type) {
  return type == PointerType::device || type == PointerType::unified;
}

bool mtrt::isHostVisible(PointerType type) {
  return !isDeviceVisible(type) || type == PointerType::unified;
}

//===----------------------------------------------------------------------===//
// ExecutableView
//===----------------------------------------------------------------------===//

StatusOr<FunctionView>
ExecutableView::getFunction(std::string_view name) const {
  const flatbuffers::Vector<flatbuffers::Offset<mtrt::flat::Function>>
      &functions = *view->functions();
  auto it = std::find_if(functions.begin(), functions.end(),
                         [&](const mtrt::flat::Function *x) {
                           return x->name()->string_view() == name;
                         });
  if (it == view->functions()->end())
    return getStatusWithMsg(StatusCode::InvalidArgument, "Function with name (",
                            name, ") is not present in the executable");
  return FunctionView(*it);
}

llvm::SmallVector<DataSegmentInfo> ExecutableView::getDataSegments() const {
  llvm::SmallVector<DataSegmentInfo> views;
  views.reserve(view->data_segments()->size());
  for (unsigned i = 0; i < view->data_segments()->size(); i++)
    views.push_back(view->data_segments()->Get(i));
  return views;
}

llvm::SmallVector<FunctionView> ExecutableView::getFunctions() const {
  llvm::SmallVector<FunctionView> views;
  views.reserve(view->functions()->size());
  for (unsigned i = 0; i < view->functions()->size(); i++)
    views.push_back(view->functions()->Get(i));
  return views;
}

//===----------------------------------------------------------------------===//
// ExecutableStorage (Implementations)
//===----------------------------------------------------------------------===//

namespace {
class ExecutableStorageMemBuffer : public ExecutableStorage {
public:
  ExecutableStorageMemBuffer(std::unique_ptr<llvm::MemoryBuffer> storage)
      : storage(std::move(storage)) {}

  std::unique_ptr<ExecutableStorage> getCopy() const final {
    return std::make_unique<ExecutableStorageMemBuffer>(
        llvm::MemoryBuffer::getMemBufferCopy(
            storage->getBuffer(), storage->getBufferIdentifier() + "_copy"));
  }
  const void *data() const final { return storage->getBuffer().data(); }
  size_t size() const final { return storage->getBufferSize(); }

private:
  std::unique_ptr<llvm::MemoryBuffer> storage;
};
} // namespace

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

Executable::Executable(Executable &&other)
    : ExecutableView(nullptr), storage(std::move(other.storage)) {
  other.storage.reset();
  this->view = mtrt::flat::GetExecutable(this->storage->data());
}

StatusOr<std::unique_ptr<Executable>>
Executable::loadFromFile(std::string_view path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFileOrSTDIN(path);
  if (!buffer)
    return getStatusWithMsg(
        StatusCode::InternalError,
        "error loading executable from file: ", buffer.getError().message());

  auto result = std::unique_ptr<Executable>(new Executable(
      std::make_unique<ExecutableStorageMemBuffer>(std::move(*buffer))));

  Status verifyResult = result->verify();
  if (!verifyResult.isOk())
    return verifyResult;
  return result;
}

StatusOr<std::unique_ptr<Executable>>
Executable::loadFromBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer) {
  auto result = std::make_unique<Executable>(
      std::make_unique<ExecutableStorageMemBuffer>(std::move(buffer)));
  Status verifyResult = result->verify();
  if (!verifyResult.isOk())
    return verifyResult;
  return result;
}

StatusOr<std::unique_ptr<Executable>>
Executable::loadFromUnalignedRef(llvm::ArrayRef<char> data) {
  const llvm::Align alignment(16);
  std::unique_ptr<llvm::WritableMemoryBuffer> alignedBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(data.size(), "",
                                                        alignment);
  // `getNewUninitMemBuffer` will return null on failure.
  if (!alignedBuffer)
    return getInvalidArgStatus(
        "failed to create uninitialized memory buffer of "
        "size {0} with alignment {1}",
        data.size(), alignment.value());

  llvm::copy(data, alignedBuffer->getBuffer().begin());
  auto result = std::make_unique<Executable>(
      std::make_unique<ExecutableStorageMemBuffer>(std::move(alignedBuffer)));

  Status verifyResult = result->verify();
  if (!verifyResult.isOk())
    return verifyResult;
  return result;
}

mtrt::Executable::Executable(std::unique_ptr<ExecutableStorage> storage_)
    : ExecutableView(nullptr), storage(std::move(storage_)) {
  assert(this->storage && "expected valid storage pointer");
  this->view = mtrt::flat::GetExecutable(this->storage->data());
}

Executable::~Executable() {}

std::unique_ptr<Executable> Executable::getCopy() const {
  std::unique_ptr<llvm::WritableMemoryBuffer> alignedBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(storage->size(), "",
                                                        llvm::Align(16));
  std::copy_n(reinterpret_cast<const char *>(storage->data()), storage->size(),
              alignedBuffer->getBuffer().begin());
  return std::make_unique<Executable>(
      std::make_unique<ExecutableStorageMemBuffer>(std::move(alignedBuffer)));
}

Status Executable::verify() const {
  flatbuffers::Verifier::Options options{};
  options.max_size = FLATBUFFERS_MAX_64_BUFFER_SIZE;
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t *>(getStorage()->data()),
      getStorage()->size(), options);
  if (!mtrt::flat::VerifyExecutableBuffer(verifier))
    return getStatusWithMsg(
        StatusCode::InvalidArgument,
        "failed to verify that the provided buffer contains "
        "a valid MLIR-TRT Executable");
  return getOkStatus();
}

//===----------------------------------------------------------------------===//
// Print Utilities
//===----------------------------------------------------------------------===//

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const TypeUnionView &arg) {
  if (arg.isa<MemRefTypeView>())
    return print(os, arg.get<MemRefTypeView>());
  if (arg.isa<ScalarTypeView>())
    return print(os, arg.get<ScalarTypeView>());
  return os << "UNK";
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const DimensionBoundsView &exe) {
  os << "dim_bounds<min = [";
  llvm::interleave(exe.getMin(), os, [&](const auto &x) { os << x; }, ",");
  os << "], max = [";
  llvm::interleave(exe.getMax(), os, [&](const auto &x) { os << x; }, ",");
  return os << "]>";
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const ValueBoundsView &exe) {
  os << "value_bounds<min = [";
  llvm::interleave(exe.getMin(), os, [&](const auto &x) { os << x; }, ",");
  os << "], max = [";
  llvm::interleave(exe.getMax(), os, [&](const auto &x) { os << x; }, ",");
  return os << "]>";
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const BoundsUnionView &bounds) {
  if (bounds.isa<DimensionBoundsView>())
    return print(os, bounds.get<DimensionBoundsView>());
  if (bounds.isa<ValueBoundsView>())
    return print(os, bounds.get<ValueBoundsView>());
  return os << "UNK";
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os, const Executable &exe) {
  os << llvm::formatv("RuntimeExecutable<name={0}, functions={1}, "
                      "data_segments={2}, source={3} bytes>",
                      exe.getName(), llvm::iterator_range(exe.getFunctions()),
                      llvm::iterator_range(exe.getDataSegments()),
                      exe.getCode().size());
  return os;
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const DataSegmentInfo &segment) {
  os << llvm::formatv(
      "DataSegment<{0}, size={1}, alignment={2}, constant={3}, "
      "uninitialized={4}, address_space={5}>",
      segment.getName(), segment.size(), segment.getAlignment(),
      segment.isConstant(), segment.isUninitialized(),
      mtrt::flat::EnumNamePointerType(segment.getAddressSpace()));
  return os;
}

namespace {
struct format_shape : public llvm::FormatAdapter<llvm::ArrayRef<int64_t>> {
  format_shape(llvm::ArrayRef<int64_t> &&N)
      : llvm::FormatAdapter<llvm::ArrayRef<int64_t>>(std::move(N)) {}

  void format(llvm::raw_ostream &os, llvm::StringRef style) override {
    llvm::interleave(
        this->Item, os,
        [&](int64_t x) {
          if (x == kDynamicSize)
            os << "?";
          else
            os << x;
        },
        style);
  }
};

} // namespace

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const MemRefTypeView &memref) {
  os << llvm::formatv(
      "MemRef<{0:x}x{1}, strides=[{2:, }], {3}>",
      format_shape(memref.getShape()),
      flat::EnumNameScalarTypeCode(memref.getElementType()),
      format_shape(memref.getStrides()),
      mtrt::flat::EnumNamePointerType(memref.getAddressSpace()));
  return os;
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const ScalarTypeView &scalarType) {
  return os << mtrt::flat::EnumNameScalarTypeCode(scalarType);
}
llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const FunctionSignatureView &signature) {
  llvm::SmallVector<TypeUnionView> args = signature.getArgs();
  llvm::SmallVector<TypeUnionView> results = signature.getResults();
  llvm::SmallVector<BoundsUnionView> arg_bounds = signature.getArgBounds();
  llvm::SmallVector<BoundsUnionView> result_bounds =
      signature.getResultBounds();

  os << llvm::formatv(
      "Signature<args=[{0}], results=[{1}], num_output_args={2}, "
      "arg_bounds=[{3}], "
      "result_bounds=[{4}], cconv={5}>",
      llvm::iterator_range(args), llvm::iterator_range(results),
      signature.getNumOutputArgs(), llvm::iterator_range(arg_bounds),
      llvm::iterator_range(result_bounds),
      mtrt::flat::EnumNameCallingConvention(signature.getCConv()));
  return os;
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const FunctionView &func) {
  os << "Function<" << func.getName() << ", ";
  print(os, func.getSignature());
  return os << ">";
}
