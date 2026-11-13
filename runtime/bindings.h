#pragma once

#include "jsi/jsi.h"
#include <uv.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace mom::runtime {
using namespace facebook::jsi;

// Keep native failures behind a JSI exception boundary, including allocation failures.
template <typename Action> void binding(Runtime &rt, const char *name, unsigned arity, Action action) {
  rt.global().setProperty(rt, name, Function::createFromHostFunction(rt, PropNameID::forAscii(rt, name), arity,
      [action, arity](Runtime &rt, const Value &, const Value *args, size_t count) -> Value {
        if (count < arity) {throw JSError(rt, "Missing native API argument");}
        try {return action(rt, args);}
        catch (const JSError &) {throw;}
        catch (const std::string &error) {throw JSError(rt, error);}
        catch (const std::exception &error) {throw JSError(rt, error.what());}
      }));
}

inline Object codedError(Runtime &rt, const char *code, const char *message) {
  auto error = rt.global().getPropertyAsFunction(rt, "Error")
      .callAsConstructor(rt, String::createFromUtf8(rt, message)).asObject(rt);
  error.setProperty(rt, "code", String::createFromAscii(rt, code));
  return error;
}

inline void checkSystem(Runtime &rt, int status, const char *syscall, const std::string &path = {}) {
  if (status >= 0) {return;}
  auto error = codedError(rt, uv_err_name(status), uv_strerror(status));
  error.setProperty(rt, "errno", status);
  error.setProperty(rt, "syscall", String::createFromAscii(rt, syscall));
  if (!path.empty()) {error.setProperty(rt, "path", String::createFromUtf8(rt, path));}
  throw JSError(rt, std::move(error));
}

// Returned ArrayBuffers own their bytes independently of temporary OS buffers and JS views.
class OwnedBytes final : public MutableBuffer {
  std::vector<uint8_t> bytes_;
public:
  explicit OwnedBytes(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {}
  OwnedBytes(const char *bytes, size_t size) : bytes_(bytes, bytes + size) {}
  size_t size() const override {return bytes_.size();}
  uint8_t *data() override {return bytes_.data();}
};

inline Value byteBuffer(Runtime &rt, std::vector<uint8_t> bytes) {
  return ArrayBuffer(rt, std::make_shared<OwnedBytes>(std::move(bytes)));
}

// Resolve all JS properties before taking a pointer: a getter can allocate or run GC. Retain the
// ArrayBuffer handle while native synchronous code uses the view, and reject forged ranges.
struct ByteView {
  ArrayBuffer buffer;
  size_t offset, size;
  const uint8_t *data(Runtime &rt) const {return size ? buffer.data(rt) + offset : nullptr;}
};
inline ByteView byteView(Runtime &rt, const Value &value) {
  const auto view = value.asObject(rt);
  const auto storage = view.getPropertyAsObject(rt, "buffer");
  if (!storage.isArrayBuffer(rt)) {throw JSError(rt, "Native buffer storage must be an ArrayBuffer");}
  auto buffer = storage.getArrayBuffer(rt);
  const double offset = view.getProperty(rt, "byteOffset").asNumber();
  const double size = view.getProperty(rt, "byteLength").asNumber();
  if (!std::isfinite(offset) || !std::isfinite(size) || offset < 0 || size < 0 ||
      std::floor(offset) != offset || std::floor(size) != size ||
      offset > buffer.size(rt) || size > buffer.size(rt) - static_cast<size_t>(offset)) {
    throw JSError(rt, "Invalid native buffer range");
  }
  return {std::move(buffer), static_cast<size_t>(offset), static_cast<size_t>(size)};
}
} // namespace mom::runtime
