#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "sycl/pearlhash/blake3.inc"

namespace fixture {
struct State {
  uint8_t* base; const uint8_t* leaves; size_t n, rd, wr, first, second;
  int cnt; size_t parent_writes, root_index; unsigned roots;
  std::vector<uint8_t> written;
};
State* active = nullptr;
[[noreturn]] void fail(const char* text) { throw std::runtime_error(text); }
size_t index_of(const uint8_t* p) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(active->base);
  const uintptr_t at = reinterpret_cast<uintptr_t>(p);
  if (at < begin || at - begin > active->n * 64 - 32 || (at - begin) % 32)
    fail("CV access outside 2*nc allocation");
  return (at - begin) / 32;
}
}

namespace pearlhash_b3 {
void checked_parent_cv(const uint8_t* left, const uint8_t* right, const uint8_t* key,
                       bool root, uint8_t* out) {
  auto& s = *fixture::active;
  const size_t li = fixture::index_of(left), ri = fixture::index_of(right);
  const size_t oi = fixture::index_of(out), parents = size_t(s.cnt / 2 + s.cnt % 2);
  if (li < s.rd || li >= s.rd + size_t(s.cnt) || ri != li + 1 ||
      oi < s.wr || oi >= s.wr + parents || oi < s.n)
    fixture::fail("parent range or cached-leaf invariant failed");
  if (s.written[oi]) fixture::fail("duplicate parent writer");
  s.written[oi] = 1;
  ++s.parent_writes;
  if (root) { ++s.roots; s.root_index = oi; }
  parent_cv(left, right, key, root, out);
}
}

namespace sycl {
template<int> struct id { size_t i; size_t operator[](int) const { return i; } };
template<int> struct range { size_t n; explicit range(size_t x) : n(x) {} };
struct queue {
  bool reverse;
  template<class F> void parallel_for(range<1> r, F body) {
    auto& s = *fixture::active;
    const size_t parents = size_t(s.cnt / 2 + s.cnt % 2);
    if (r.n != parents) fixture::fail("wrong parent launch size");
    std::array<uint8_t, 32> tail{};
    if (s.cnt & 1)
      std::memcpy(tail.data(), s.base + (s.rd + size_t(s.cnt) - 1) * 32, 32);
    s.written.assign(2 * s.n, 0); s.parent_writes = 0;
    for (size_t i = 0; i < r.n; ++i) body(id<1>{reverse ? r.n - 1 - i : i});
    if ((s.cnt & 1) && std::memcmp(tail.data(), s.base + (s.wr + parents - 1) * 32, 32))
      fixture::fail("odd carry was not copied to the next level");
    if (std::memcmp(s.base, s.leaves, s.n * 32)) fixture::fail("cached leaves changed");
    if (s.parent_writes != size_t(s.cnt / 2)) fixture::fail("paired write count changed");
    for (size_t i = 0; i < size_t(s.cnt / 2); ++i)
      if (!s.written[s.wr + i]) fixture::fail("paired parent output missing");
    if ((s.cnt & 1) && s.written[s.wr + parents - 1])
      fixture::fail("odd carry unexpectedly used parent_cv");
    s.rd = s.wr; s.wr = s.wr == s.first ? s.second : s.first; s.cnt = int(parents);
  }
};
}

#define MOM_SYCL_KERNEL_ARGS_RESTRICT
#define parent_cv checked_parent_cv
namespace extracted_tree {
#include "reduce_tree.inc"
}
#undef parent_cv

int main() try {
  std::vector<int> counts;
  for (int n = 1; n <= 129; ++n) counts.push_back(n);
  for (int n : {255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4095, 4096, 4097})
    counts.push_back(n);
  unsigned cases = 0;
  for (int n : counts) for (bool reverse : {false, true}) for (unsigned seed : {7u, 123u}) {
    std::array<uint8_t, 32> key{};
    for (unsigned i = 0; i < key.size(); ++i) key[i] = uint8_t(seed + i * 17);
    std::vector<uint8_t> storage(size_t(n) * 64 + 128, 0xee), leaves;
    std::fill_n(storage.begin(), 64, uint8_t(0xa5));
    std::fill(storage.end() - 64, storage.end(), uint8_t(0x5a));
    uint8_t* cv = storage.data() + 64;
    for (unsigned reuse = 0; reuse < 2; ++reuse) {
      std::fill(cv, cv + size_t(n) * 64, uint8_t(0x40 + reuse));
      for (size_t i = 0; i < size_t(n) * 32; ++i)
        cv[i] = uint8_t(i * 131u + seed + reuse * 37u + (i >> 4));
      leaves.assign(cv, cv + size_t(n) * 32);
      std::array<uint8_t, 32> expected{};
      auto oracle = leaves;
      pearlhash_b3::merge_root(oracle.data(), n, key.data(), expected.data());
      fixture::State s{cv, leaves.data(), size_t(n), 0, size_t(n), size_t(n),
                       size_t(n) + size_t(n / 2 + n % 2), n, 0, size_t(-1), 0, {}};
      fixture::active = &s;
      sycl::queue q{reverse};
      const size_t root = extracted_tree::reduce_tree(q, cv, key.data(), n);
      fixture::active = nullptr;
      if (root >= size_t(n) * 2 || std::memcmp(cv + root * 32, expected.data(), 32) ||
          std::memcmp(cv, leaves.data(), leaves.size()) || s.roots != unsigned(n > 1) ||
          (n > 1 && s.root_index != root)) fixture::fail("canonical root/leaf/ROOT mismatch");
      if (!std::all_of(storage.begin(), storage.begin() + 64, [](uint8_t x) { return x == 0xa5; }) ||
          !std::all_of(storage.end() - 64, storage.end(), [](uint8_t x) { return x == 0x5a; }))
        fixture::fail("allocation guard changed");
      ++cases;
    }
  }
  std::printf("PARALLEL_TREE_SOURCE %u canonical-root/leaf-cache/bounds/ROOT/layout/reuse cases PASS\n", cases);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "PARALLEL_TREE_SOURCE_FAIL %s\n", e.what()); return 2;
}
