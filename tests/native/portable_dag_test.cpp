#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <stdexcept>

#include "../../xmrig/3rdparty/libethash/ethash.h"
#include "../../xmrig/3rdparty/libethash/data_sizes.h"

namespace sycl {

class device {
public:
  explicit device(const bool cpu) : cpu_(cpu) {}

  bool is_cpu() const { return cpu_; }

private:
  bool cpu_;
};

} // namespace sycl

// The Node fixture extracts these definitions from the production entry/state files into this
// include. Compiling and executing the extracted functions keeps the policy assertion tied to the
// actual runtime guard instead of merely checking source text.
#include "portable_dag_generated.inc"

namespace {

using portable_fixture::FastModData;
using namespace portable_fixture;

void require(const bool condition, const char* const message) {
  if (!condition)
    throw std::runtime_error(message);
}

void set_portable_flag(const char* const value) {
#if defined(_WIN32)
  if (_putenv_s("MOM_SYCL_PORTABLE_TEST", value ? value : "") != 0)
    throw std::runtime_error("unable to set fixture environment");
#else
  if (value) {
    if (setenv("MOM_SYCL_PORTABLE_TEST", value, 1) != 0)
      throw std::runtime_error("unable to set fixture environment");
  } else if (unsetenv("MOM_SYCL_PORTABLE_TEST") != 0) {
    throw std::runtime_error("unable to clear fixture environment");
  }
#endif
}

void check_admission(const char* const label, const bool is_test, const bool is_benchmark,
                     const sycl::device& device, const bool expected) {
  const bool kawpow = portable_fixture::kawpow_portable_test(is_test, is_benchmark, device);
  const bool etchash = portable_fixture::etchash_portable_test(is_test, is_benchmark, device);
  if (kawpow != expected || etchash != expected) {
    std::fprintf(stderr, "%s: KawPow=%d Etchash=%d expected=%d\n", label, kawpow, etchash,
                 expected);
    throw std::runtime_error("portable test admission mismatch");
  }
}

void check_fast_mod(const uint32_t divisor) {
  const FastModData mod = portable_fixture::make_fast_mod_data(divisor);
  const uint32_t samples[] = {0, 1, divisor - 1, divisor, divisor + 1, UINT32_MAX};
  for (const uint32_t value : samples)
    require(portable_fixture::fast_mod_dev(value, mod) == value % divisor,
            "portable DAG modulus mismatch");
}

void check_geometry() {
  constexpr uint64_t WORD_BYTES = sizeof(uint32_t);
  constexpr uint64_t KAWPOW_DAG_LOAD_BYTES =
      KAWPOW_DAG_LOADS * WORD_BYTES;
  constexpr uint64_t KAWPOW_DAG_ELEMENTS = KAWPOW_PORTABLE_TEST_DAG_BYTES / 256;
  constexpr uint64_t KAWPOW_DAG_LOADS_TOTAL =
      KAWPOW_PORTABLE_TEST_DAG_BYTES / KAWPOW_DAG_LOAD_BYTES;
  constexpr uint64_t KAWPOW_LAST_LOAD =
      (KAWPOW_DAG_ELEMENTS - 1) * KAWPOW_LANES + (KAWPOW_LANES - 1);

  require(KAWPOW_PORTABLE_TEST_CACHE_BYTES >= KAWPOW_CACHE_WORDS * WORD_BYTES,
          "ProgPoW fixture cache does not fill local cache");
  require(KAWPOW_PORTABLE_TEST_CACHE_BYTES >= 16u * 1024u,
          "ProgPoW fixture cache is below 16 KiB");
  require(KAWPOW_PORTABLE_TEST_DAG_BYTES % 256 == 0 && KAWPOW_DAG_ELEMENTS != 0,
          "ProgPoW fixture DAG geometry is not aligned");
  require(KAWPOW_LAST_LOAD < KAWPOW_DAG_LOADS_TOTAL,
          "ProgPoW fixture DAG load modulus can address past the allocation");

  constexpr uint64_t ETCHASH_PAGES = ETCHASH_PORTABLE_TEST_DAG_BYTES / ETHASH_MIX_BYTES;
  constexpr uint64_t ETCHASH_WORDS = ETCHASH_PORTABLE_TEST_DAG_BYTES / WORD_BYTES;
  require(ETCHASH_PORTABLE_TEST_CACHE_BYTES % ETHASH_HASH_BYTES == 0,
          "Etchash fixture cache is not node aligned");
  require(ETCHASH_PORTABLE_TEST_DAG_BYTES % ETHASH_MIX_BYTES == 0 && ETCHASH_PAGES != 0,
          "Etchash fixture DAG geometry is not page aligned");
  require(ETCHASH_PAGES * (ETHASH_MIX_BYTES / WORD_BYTES) == ETCHASH_WORDS,
          "Etchash fixture page modulus does not cover the allocation");

  check_fast_mod(static_cast<uint32_t>(KAWPOW_DAG_ELEMENTS));
  check_fast_mod(static_cast<uint32_t>(ETCHASH_PAGES));
}

void check_cache_key() {
  constexpr uint64_t compact_cache_words = KAWPOW_PORTABLE_TEST_CACHE_BYTES / sizeof(uint32_t);
  constexpr uint64_t compact_dag_words = ETCHASH_PORTABLE_TEST_DAG_BYTES / sizeof(uint32_t);
  const uint64_t full_cache_words = cache_sizes[0] / sizeof(uint32_t);
  const uint64_t full_dag_words = dag_sizes[0] / sizeof(uint32_t);
  const void* const allocated = reinterpret_cast<const void*>(1);

  require(portable_fixture::etchash_cache_key_matches(
              0, 0, 0, 0, compact_cache_words, compact_cache_words, compact_dag_words,
              compact_dag_words, allocated, allocated),
          "Etchash compact key did not reuse an unchanged compact allocation");
  require(!portable_fixture::etchash_cache_key_matches(
              0, 0, 0, 0, compact_cache_words, full_cache_words, compact_dag_words,
              full_dag_words, allocated, allocated),
          "Etchash compact key reused the full consensus dimensions");
  require(!portable_fixture::etchash_cache_key_matches(
              0, 0, 0, 0, full_cache_words, compact_cache_words, full_dag_words,
              compact_dag_words, allocated, allocated),
          "Etchash full key reused the compact dimensions");
  require(portable_fixture::etchash_cache_key_matches(
              0, 0, 0, 0, full_cache_words, full_cache_words, full_dag_words,
              full_dag_words, allocated, allocated),
          "Etchash full key did not reuse an unchanged full allocation");
}

} // namespace

int main() {
  try {
    const sycl::device cpu(true);
    const sycl::device gpu(false);

    set_portable_flag(nullptr);
    check_admission("unset", true, false, cpu, false);
    set_portable_flag("0");
    check_admission("zero", true, false, cpu, false);
    set_portable_flag("invalid");
    check_admission("invalid", true, false, cpu, false);
    set_portable_flag("1");
    check_admission("cpu test", true, false, cpu, true);
    check_admission("mining", false, false, cpu, false);
    check_admission("benchmark", true, true, cpu, false);
    check_admission("gpu test", true, false, gpu, false);
    check_admission("gpu mining", false, false, gpu, false);
    check_admission("gpu benchmark", true, true, gpu, false);

    check_geometry();
    check_cache_key();
    set_portable_flag(nullptr);
    std::puts("portable DAG fixture PASS");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "portable DAG fixture FAIL: %s\n", error.what());
    return 1;
  }
}
