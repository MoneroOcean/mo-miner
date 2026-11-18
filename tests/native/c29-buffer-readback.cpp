#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <string>

static bool fail_submit, fail_cleanup, pending;
static unsigned drains;
static void* destination;
static const void* source;
static std::size_t bytes;
static std::exception_ptr primary;
namespace sycl {
struct device {};
struct event {};
constexpr int read_only = 0;
template <int> struct range { std::size_t count; };
template <int> struct id { std::size_t offset; };
template <typename T, int> struct buffer { T* data; };
struct handler;
template <typename T> struct accessor {
  T* data;
  std::size_t count;
  accessor(buffer<T, 1>& b, handler&, range<1> extent, id<1>, int)
      : data(b.data), count(extent.count) {}
};
struct handler {
  template <typename T> void copy(accessor<T> view, T* out) {
    destination = out; source = view.data; bytes = view.count * sizeof(T); pending = true;
  }
};
template <typename T, int D>
accessor(buffer<T, D>&, handler&, range<1>, id<1>, int) -> accessor<T>;
struct queue {
  device get_device() const { return {}; }
  template <typename Fn> event submit(Fn fn) {
    handler h;
    fn(h);
    if (fail_submit) std::rethrow_exception(primary);
    return {};
  }
  void wait_and_throw() {
    ++drains;
    if (pending) std::memcpy(destination, source, bytes);
    pending = false;
    if (fail_cleanup) throw std::string("secondary cleanup error");
  }
};
}

#include "cleanup.inc"
static void sycl_wait_and_throw(sycl::event, sycl::device) {
  if (pending) std::memcpy(destination, source, bytes);
  pending = false;
}
#include "readback.inc"

int main() {
  for (bool submit_error : {false, true}) for (bool cleanup_error : {false, true}) {
    fail_submit = submit_error; fail_cleanup = cleanup_error; drains = 0;
    primary = std::make_exception_ptr(std::runtime_error("accepted copy submission error"));
    sycl::queue queue;
    std::uint32_t input[2]{17, 23}, output[2]{};
    sycl::buffer<std::uint32_t, 1> buffer{input};
    std::exception_ptr caught;
    try { c29_read_buffer(queue, buffer, output, 2); }
    catch (...) { caught = std::current_exception(); }
    if (pending || drains != unsigned(submit_error) ||
        (submit_error ? caught != primary : bool(caught)) || output[0] != 17 || output[1] != 23)
      return std::fputs("FAIL C29 accepted readback outlived its caller\n", stderr), 1;
  }
  std::puts("PASS actual-source C29 accepted readback lifetime");
}
