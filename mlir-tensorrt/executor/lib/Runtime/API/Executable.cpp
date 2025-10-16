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
// DimensionBoundsView
//===----------------------------------------------------------------------===//

DimensionBoundsView::DimensionBoundsView(
    const mtrt::flat::DimensionBounds *view)
    : FlatbufferBoundsObjectView(view) {}

llvm::ArrayRef<int64_t> DimensionBoundsView::getMin() const {
  return llvm::ArrayRef<int64_t>(view->min()->data(), view->min()->size());
}

llvm::ArrayRef<int64_t> DimensionBoundsView::getMax() const {
  return llvm::ArrayRef<int64_t>(view->max()->data(), view->max()->size());
}

//===----------------------------------------------------------------------===//
// ValueBoundsView
//===----------------------------------------------------------------------===//

ValueBoundsView::ValueBoundsView(const mtrt::flat::ValueBounds *view)
    : FlatbufferBoundsObjectView(view) {}

llvm::ArrayRef<int64_t> ValueBoundsView::getMin() const {
  return llvm::ArrayRef<int64_t>(view->min()->data(), view->min()->size());
}

llvm::ArrayRef<int64_t> ValueBoundsView::getMax() const {
  return llvm::ArrayRef<int64_t>(view->max()->data(), view->max()->size());
}

//===----------------------------------------------------------------------===//
// MemRefTypeView
//===----------------------------------------------------------------------===//

MemRefTypeView::MemRefTypeView(const mtrt::flat::MemRefType *view)
    : FlatbufferTypeObjectView(view) {}

int64_t MemRefTypeView::getRank() const { return view->shape()->size(); }

ScalarType MemRefTypeView::getElementType() const {
  return view->element_type();
}

llvm::ArrayRef<int64_t> MemRefTypeView::getShape() const {
  return llvm::ArrayRef<int64_t>(view->shape()->data(), view->shape()->size());
}

llvm::ArrayRef<int64_t> MemRefTypeView::getStrides() const {
  return llvm::ArrayRef<int64_t>(view->strides()->data(),
                                 view->strides()->size());
}

PointerType MemRefTypeView::getAddressSpace() const {
  return PointerType(view->address_space());
}

//===----------------------------------------------------------------------===//
// FunctionSignatureView
//===----------------------------------------------------------------------===//

FunctionSignatureView::FunctionSignatureView(
    const mtrt::flat::FunctionSignature *view)
    : view(view) {
  assert(view != nullptr && "expected valid view");
  if (view->abi_version() >= 1) {
    assert(view->result_bounds_indices()->size() == view->results()->size() &&
           "expected valid result bounds indices");
    assert(view->arg_bounds_indices()->size() == view->args()->size() &&
           "expected valid argument bounds indices");
    assert(view->results()->size() == view->num_output_args() &&
           "expected valid number of output arguments");
  }
}

uint32_t FunctionSignatureView::getNumArgs() const {
  return view->args() ? view->args()->size() : 0;
}

uint32_t FunctionSignatureView::getNumResults() const {
  return view->results() ? view->results()->size() : 0;
}

uint32_t FunctionSignatureView::getNumInputArgs() const {
  assert(getNumArgs() >= getNumOutputArgs() &&
         "invalid number of output arguments specified");
  return getNumArgs() - getNumOutputArgs();
}

uint32_t FunctionSignatureView::getNumOutputArgs() const {
  return view->num_output_args();
}

TypeUnionView FunctionSignatureView::getArg(int64_t idx) const {
  assert(idx < getNumArgs() && "expected valid argument index");
  return TypeUnionView{view->args_type()->Get(idx), view->args()->Get(idx)};
}

TypeUnionView FunctionSignatureView::getResult(int64_t idx) const {
  assert(idx < getNumResults() && "expected valid result index");
  return TypeUnionView{view->results_type()->Get(idx),
                       view->results()->Get(idx)};
}

BoundsUnionView FunctionSignatureView::getArgBound(int64_t idx) const {
  assert(idx < getNumArgs() && "expected valid argument index");
  int32_t boundsIdx = view->arg_bounds_indices()->Get(idx);
  if (boundsIdx < 0)
    return BoundsUnionView{mtrt::flat::Bounds::NONE, nullptr};
  return BoundsUnionView{view->bounds_values_type()->Get(boundsIdx),
                         view->bounds_values()->Get(boundsIdx)};
}

BoundsUnionView FunctionSignatureView::getResultBound(int64_t idx) const {
  assert(idx < getNumResults() && "expected valid result index");
  assert(idx < view->result_bounds_indices()->size() &&
         "expected valid result bounds index");
  int32_t boundsIdx = view->result_bounds_indices()->Get(idx);
  if (boundsIdx < 0)
    return BoundsUnionView{mtrt::flat::Bounds::NONE, nullptr};
  assert(static_cast<unsigned>(boundsIdx) < view->bounds_values()->size() &&
         "expected valid bounds value index");
  return BoundsUnionView{view->bounds_values_type()->Get(boundsIdx),
                         view->bounds_values()->Get(boundsIdx)};
}

TypeUnionView FunctionSignatureView::getOutputArg(int64_t idx) const {
  assert(idx < getNumOutputArgs() && "expected valid output argument index");
  // Starting in ABI version 1, outputs are in the results array.
  if (getAbiVersion() >= 1)
    return getResult(idx);
  // For ABI version 0, outputs are at the end of the args array
  unsigned offset = getNumInputArgs() + idx;
  return TypeUnionView{view->args_type()->Get(offset),
                       view->args()->Get(offset)};
}

llvm::SmallVector<TypeUnionView> FunctionSignatureView::getArgs() const {
  llvm::SmallVector<TypeUnionView> args;
  unsigned numArgs = getNumArgs();
  args.reserve(numArgs);
  for (unsigned i = 0; i < numArgs; i++)
    args.push_back(getArg(i));
  return args;
}

llvm::SmallVector<TypeUnionView> FunctionSignatureView::getResults() const {
  llvm::SmallVector<TypeUnionView> results;
  unsigned numResults = getNumResults();
  results.reserve(numResults);
  for (unsigned i = 0; i < numResults; i++)
    results.push_back(getResult(i));
  return results;
}

llvm::SmallVector<BoundsUnionView> FunctionSignatureView::getArgBounds() const {
  llvm::SmallVector<BoundsUnionView> bounds;
  unsigned numArgs = getNumArgs();
  bounds.reserve(numArgs);
  for (unsigned i = 0; i < numArgs; i++)
    bounds.push_back(getArgBound(i));
  return bounds;
}

llvm::SmallVector<BoundsUnionView>
FunctionSignatureView::getResultBounds() const {
  llvm::SmallVector<BoundsUnionView> bounds;
  unsigned numResults = getNumResults();
  bounds.reserve(numResults);
  for (unsigned i = 0; i < numResults; i++)
    bounds.push_back(getResultBound(i));
  return bounds;
}

std::optional<std::string_view>
FunctionSignatureView::getShapeFunctionName() const {
  const flatbuffers::String *name = view->shape_function_name();
  if (!name || name->size() == 0)
    return std::nullopt;
  return view->shape_function_name()->string_view();
}

CallingConvention FunctionSignatureView::getCConv() const {
  return view->calling_convention();
}

llvm::ArrayRef<uint8_t> FunctionSignatureView::getUndef() const {
  return llvm::ArrayRef<uint8_t>(view->undef()->data(), view->undef()->size());
}

uint32_t FunctionSignatureView::getAbiVersion() const {
  return view->abi_version();
}

//===----------------------------------------------------------------------===//
// FunctionView
//===----------------------------------------------------------------------===//

FunctionView::FunctionView(const mtrt::flat::Function *view) : view(view) {
  assert(view != nullptr);
}

FunctionView::FunctionView() : view(nullptr) {}

FunctionSignatureView FunctionView::getSignature() const {
  return FunctionSignatureView(view->signature());
}

std::string_view FunctionView::getName() const {
  return view->name()->string_view();
}

FunctionView::operator bool() const { return view != nullptr; }

FunctionView::operator const mtrt::flat::Function *() const { return view; }

uint32_t FunctionView::getAbiVersion() const {
  return view ? view->abi_version() : 0;
}

//===----------------------------------------------------------------------===//
// DataSegmentInfo
//===----------------------------------------------------------------------===//

DataSegmentInfo::DataSegmentInfo(const mtrt::flat::DataSegment *view)
    : view(view) {}

std::string_view DataSegmentInfo::getName() const {
  return view->name()->string_view();
}

const int8_t *DataSegmentInfo::data() const {
  return view->data() ? view->data()->data() : nullptr;
}

size_t DataSegmentInfo::size() const {
  return view->data() ? view->data()->size() : getUninitializedSize();
}

uint32_t DataSegmentInfo::getAlignment() const { return view->alignment(); }

bool DataSegmentInfo::isConstant() const { return view->constant(); }

bool DataSegmentInfo::isUninitialized() const {
  return view->uninitialized_size() > 0;
}

uint64_t DataSegmentInfo::getUninitializedSize() const {
  return view->uninitialized_size();
}

PointerType DataSegmentInfo::getAddressSpace() const {
  return view->address_space();
}

//===----------------------------------------------------------------------===//
// ExecutableView
//===----------------------------------------------------------------------===//

ExecutableView::ExecutableView(const mtrt::flat::Executable *view)
    : view(view) {}

std::string_view ExecutableView::getCode() const {
  return view->source()->string_view();
}

size_t ExecutableView::getNumFunctions() const {
  return view->functions()->size();
}

FunctionView ExecutableView::getFunction(int64_t idx) const {
  return FunctionView(view->functions()->Get(idx));
}

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

size_t ExecutableView::getNumDataSegments() const {
  if (!view || !view->data_segments())
    return 0;
  return view->data_segments()->size();
}

DataSegmentInfo ExecutableView::getDataSegments(int64_t idx) const {
  assert(view->data_segments() && "expected valid data segment pointer");
  return view->data_segments()->Get(idx);
}

std::string_view ExecutableView::getName() const {
  if (!view->name())
    return "unnamed-executable";
  return view->name()->string_view();
}

llvm::ArrayRef<uint32_t> ExecutableView::getProcessorGridShape() const {
  assert(view->process_grid_shape() && "expected valid process grid shape");
  return llvm::ArrayRef<uint32_t>(view->process_grid_shape()->data(),
                                  view->process_grid_shape()->size());
}

uint32_t ExecutableView::getAbiVersion() const { return view->abi_version(); }

ExecutableView::operator bool() const { return view != nullptr; }

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
                               const DimensionBoundsView &dimBounds) {
  os << "dim_bounds<min = [";
  llvm::interleave(
      dimBounds.getMin(), os, [&](const auto &x) { os << x; }, ",");
  os << "], max = [";
  llvm::interleave(
      dimBounds.getMax(), os, [&](const auto &x) { os << x; }, ",");
  return os << "]>";
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const ValueBoundsView &valBounds) {
  os << "value_bounds<min = [";
  llvm::interleave(
      valBounds.getMin(), os, [&](const auto &x) { os << x; }, ",");
  os << "], max = [";
  llvm::interleave(
      valBounds.getMax(), os, [&](const auto &x) { os << x; }, ",");
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
      "result_bounds=[{4}], cconv={5}, undef=[{6}], abi_version={7}>",
      llvm::iterator_range(args), llvm::iterator_range(results),
      signature.getNumOutputArgs(), llvm::iterator_range(arg_bounds),
      llvm::iterator_range(result_bounds),
      mtrt::flat::EnumNameCallingConvention(signature.getCConv()),
      llvm::iterator_range(signature.getUndef()), signature.getAbiVersion());
  return os;
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const FunctionView &func) {
  os << "Function<" << func.getName() << ", ";
  print(os, func.getSignature());
  return os << ">";
}
