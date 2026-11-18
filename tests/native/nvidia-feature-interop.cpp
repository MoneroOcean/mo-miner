#include "../../sycl/nvidia-features.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static int init_result = 0, fail_attribute = 0, native_seen = -1, attribute_calls = 0;
static int major = 7, minor = 0;
static bool throws_attribute = false;

static int fake_init(unsigned) {
  return init_result;
}

static int fake_attribute(int* out, int attribute, int native) {
  ++attribute_calls;
  native_seen = native;
  if (throws_attribute)
    throw std::runtime_error("driver");
  if (attribute == fail_attribute)
    return 1;
  *out = attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ? major : minor;
  return 0;
}

static int fake_unused(...) {
  return 0;
}

void* mock_dlopen(const char*, int) {
  return reinterpret_cast<void*>(1);
}

void* mock_dlsym(void*, const char* name) {
  if (std::strcmp(name, "cuInit") == 0)
    return reinterpret_cast<void*>(&fake_init);
  if (std::strcmp(name, "cuDeviceGetAttribute") == 0)
    return reinterpret_cast<void*>(&fake_attribute);
  // The actual loader requires non-null unused symbols to instantiate; none are called.
  return reinterpret_cast<void*>(&fake_unused);
}

int main() {
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  constexpr auto kind = sycl::backend::cuda;
#else
  constexpr auto kind = sycl::backend::ext_oneapi_cuda;
#endif
  sycl::device selected{kind, 5};
  using mom::nvidia::compute_capability;
  assert(compute_capability(selected) == 70);
  assert(native_seen == 5 && attribute_calls == 2 && sycl::native_calls == 1);
  const int before = attribute_calls;
  assert(compute_capability(sycl::device{sycl::backend::other, 0}) == 0);
  assert(attribute_calls == before && sycl::native_calls == 1);
  major = 8;
  minor = 9;
  selected.native = 11;
  assert(compute_capability(selected) == 89);
  assert(native_seen == 11);
  major = 7;
  minor = 0;
  auto& driver = mom::CudaDriverApi::instance();
  const auto init = driver.init;
  const auto attribute = driver.device_attribute;
  driver.init = nullptr;
  assert(compute_capability(selected) == 0);
  driver.init = init;
  driver.device_attribute = nullptr;
  assert(compute_capability(selected) == 0);
  driver.device_attribute = attribute;
  init_result = 1;
  assert(compute_capability(selected) == 0);
  init_result = 0;
  fail_attribute = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
  assert(compute_capability(selected) == 0);
  fail_attribute = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
  assert(compute_capability(selected) == 0);
  fail_attribute = 0;
  selected.throws = true;
  assert(compute_capability(selected) == 0);
  selected.throws = false;
  throws_attribute = true;
  assert(compute_capability(selected) == 0);
  throws_attribute = false;
  major = 0;
  assert(compute_capability(selected) == 0);
  major = 7;
  minor = 10;
  assert(compute_capability(selected) == 0);
  minor = -1;
  assert(compute_capability(selected) == 0);
  std::puts("selected-device nonzero native IDs, mismatch, missing functions, init/query failures, exceptions and malformed attributes PASS");
}
