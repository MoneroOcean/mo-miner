#pragma once

#include "jsi/jsi.h"
#include <uv.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace mom::runtime {
using namespace facebook::jsi;

// Keep native failures behind a JSI exception boundary, including allocation failures. Core
// commands throw std::string; OpenSSL, libuv and allocation failures derive from std::exception.
template <typename Action> auto nativeCall(Runtime &rt, Action action) {
  try {return action();}
  catch (const JSError &) {throw;}
  catch (const std::string &error) {throw JSError(rt, error);}
  catch (const std::exception &error) {throw JSError(rt, error.what());}
}

template <typename Action> void binding(Runtime &rt, const char *name, unsigned arity, Action action) {
  rt.global().setProperty(rt, name, Function::createFromHostFunction(rt, PropNameID::forAscii(rt, name), arity,
      [action, arity](Runtime &rt, const Value &, const Value *args, size_t count) -> Value {
        if (count < arity) {throw JSError(rt, "Missing native API argument");}
        return nativeCall(rt, [&] {return action(rt, args);});
      }));
}

// A JS-realm Error, not a JSI message wrapper, so callers can attach Node's code/errno fields.
inline Object errorObject(Runtime &rt, const std::string &message) {
  return rt.global().getPropertyAsFunction(rt, "Error")
      .callAsConstructor(rt, String::createFromUtf8(rt, message)).asObject(rt);
}

inline Object codedError(Runtime &rt, const char *code, const std::string &message) {
  auto error = errorObject(rt, message);
  error.setProperty(rt, "code", String::createFromAscii(rt, code));
  return error;
}

// JS numbers cross the bridge as doubles. Admit only finite integers in [minimum, maximum] before
// any native cast: NaN, infinities, fractions and out-of-range values make such casts undefined.
inline bool integerInRange(double value, double minimum, double maximum) {
  return std::isfinite(value) && value >= minimum && value <= maximum && std::floor(value) == value;
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
  const size_t capacity = buffer.size(rt);
  // Short-circuiting keeps capacity - offset from wrapping.
  if (!integerInRange(offset, 0, static_cast<double>(capacity)) ||
      !integerInRange(size, 0, static_cast<double>(capacity - static_cast<size_t>(offset)))) {
    throw JSError(rt, "Invalid native buffer range");
  }
  return {std::move(buffer), static_cast<size_t>(offset), static_cast<size_t>(size)};
}
} // namespace mom::runtime
