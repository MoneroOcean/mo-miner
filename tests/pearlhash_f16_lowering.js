"use strict";
// GPU-free deferred-JIT regression using the actual WMMA helper, not copied asm.
const assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process"), test = require("node:test");
const {extractCppBlock} = require("./common/cpp_source");
const clang = "clang++-21", llc = "llc-21";
const ptxas = process.env["MOM_PEARL_PTXAS"] || "/usr/local/cuda-12.6/bin/ptxas";
const available = [clang, llc].every(command => spawnSync(command, ["--version"], {timeout: 5000}).status === 0);
const assemblerAvailable = spawnSync(ptxas, ["--version"], {timeout: 5000}).status === 0;
/** @param {string} command @param {string[]} argv */
function run(command, argv) {
  const result = spawnSync(command, argv, {encoding: "utf8", timeout: 30000, maxBuffer: 4 * 1024 * 1024});
  assert.ifError(result.error); assert.equal(result.signal, null); assert.equal(result.status, 0, result.stderr);
}
test.describe("Pearl FP16 deferred JIT lowering", () => {
  test("Volta WMMA source preserves bounded rank, aligned staging and converged barriers", () => {
    const text = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/sycl_search.inc"), "utf8");
    const dispatch = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/dispatch.inc"), "utf8");
    const helper = extractCppBlock(text, "static inline void pearlhash_wmma_m16n16k16_f16(");
    const predicate = extractCppBlock(text, "static inline bool pearlhash_uses_wmma_f16()");
    for (const query of ["compiler_backend", "target_vendor_id", "target_arch"]) {
      assert.ok(predicate.includes(`jit::knows<query::${query}>() &&`));
    }
    assert.ok(predicate.includes("== jit::compiler_backend::ptx"));
    assert.ok(predicate.includes("== jit::vendor_id::nvidia"));
    assert.ok(predicate.includes("jit::reflect<query::target_arch>() >= 70"));
    assert.ok(predicate.includes("!mom::nvidia::has_int8_mma"));
    assert.ok(predicate.includes("jit::knows<PearlPtxVersion>() && jit::reflect<PearlPtxVersion>() >= 60"));
    assert.match(predicate, /return false;/);
    assert.match(helper, /cvta\.to\.shared\.u64/);
    assert.ok(helper.includes('wmma.load.a.sync" SUFFIX ".m16n16k16.row.shared.f16'));
    assert.ok(helper.includes('wmma.load.b.sync" SUFFIX ".m16n16k16.col.shared.f16'));
    assert.ok(helper.includes('wmma.mma.sync" SUFFIX ".m16n16k16.row.col.f32.f32'));
    assert.ok(helper.includes("reflect<PearlPtxVersion>() >= 63"));
    assert.ok(helper.includes('MOM_PEARLHASH_WMMA(".aligned")'));
    assert.ok(helper.includes('MOM_PEARLHASH_WMMA("")'));
    assert.ok(helper.includes("#undef MOM_PEARLHASH_WMMA"));
    assert.match(helper, /:\s*"memory"\s*\);/);
    assert.match(text, /struct PearlF16Scratch\s*\{\s*uint32_t words\[520\];/);
    assert.match(dispatch, /rank > 0 && rank <= 1024 && rank % 16 == 0/);
    assert.ok(!text.includes("__builtin_trap"));
    const call = text.lastIndexOf("pearlhash_wmma_m16n16k16_f16(");
    assert.ok(call > text.indexOf("pearlhash_wmma_m16n16k16_f16("));
    assert.match(text.slice(Math.max(0, call - 700), call), /sycl::group_barrier\(it\.get_sub_group\(\), sycl::memory_scope::sub_group\)/);
    assert.match(text.slice(call, call + 700), /sycl::group_barrier\(it\.get_sub_group\(\), sycl::memory_scope::sub_group\)/);
    assert.match(text, /static_assert\(!FP16Matrix \|\| \(Native && ER \* EC == 8\)/);
  });
  test("Actual reflection predicate fails closed and selects only known SM70–74 PTX60+", {
    skip: available ? false : "LLVM21 clang/llc unavailable",
  }, () => {
    const text = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/sycl_search.inc"), "utf8");
    const query = extractCppBlock(text, "struct PearlPtxVersion");
    const predicate = extractCppBlock(text, "static inline bool pearlhash_uses_wmma_f16()");
    const retained = process.env["MOM_PEARL_LOWERING_OUTPUT"];
    const output = retained ? path.resolve(retained + "-reflection") : fs.mkdtempSync(path.join(os.tmpdir(), "Pearl-wmma-reflection-"));
    if (retained) {fs.mkdirSync(output, {mode: 0o700});}
    try {
      const cpp = path.join(output, "actual-predicate.cpp"), exe = path.join(output, "actual-predicate");
      fs.writeFileSync(cpp, `#include <cassert>
#define __acpp_if_target_sscp(...) __VA_ARGS__
static bool known = true;
static int version = 60, arch = 70, backend = 1, vendor = 1;
extern "C" bool __acpp_sscp_jit_reflect_knows_ptx_version() { return known; }
extern "C" int __acpp_sscp_jit_reflect_ptx_version() { assert(known); return version; }
${query};
namespace mom::nvidia { bool has_int8_mma(int sm) { return sm >= 75; } }
namespace sycl::AdaptiveCpp_jit {
namespace compiler_backend { constexpr int ptx = 1; }
namespace vendor_id { constexpr int nvidia = 1; }
namespace reflection_query {
struct compiler_backend { static bool is_known() { return true; } static int get() { return backend; } };
struct target_vendor_id { static bool is_known() { return true; } static int get() { return vendor; } };
struct target_arch { static bool is_known() { return true; } static int get() { return arch; } };
}
template<class Q> bool knows() { return Q::is_known(); }
template<class Q> int reflect() { return Q::get(); }
}
${predicate}
int main() {
  for(int sm = 69; sm <= 80; ++sm) for(int ptx = 59; ptx <= 64; ++ptx) {
    arch = sm; version = ptx;
    assert(pearlhash_uses_wmma_f16() == (sm >= 70 && sm < 75 && ptx >= 60));
  }
  arch = 70; version = 60; known = false; assert(!pearlhash_uses_wmma_f16());
  known = true; backend = 0; assert(!pearlhash_uses_wmma_f16());
  backend = 1; vendor = 0; assert(!pearlhash_uses_wmma_f16());
}
`);
      run(clang, ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", cpp, "-o", exe]);
      run(exe, []);
      // Fold the actual helper's unknown-query branch as the reflection pass would;
      // the separate packaged-translator fixture verifies that pass's field lookup.
      const unknown = path.join(output, "unknown-helper.cpp"), ir = path.join(output, "unknown-helper.ll");
      const ptx = path.join(output, "unknown-helper.ptx");
      const fixture = fs.readFileSync(cpp, "utf8"), main = fixture.indexOf("int main() {");
      assert.ok(main >= 0);
      const prefix = fixture.slice(0, main)
        .replace("static bool known = true;", "static constexpr bool known = false;");
      const helper = extractCppBlock(text, "static inline void pearlhash_wmma_m16n16k16_f16(");
      fs.writeFileSync(unknown, "#include <cstdint>\n" + prefix +
        "namespace sycl { template<class T, class U> T bit_cast(U x) { return __builtin_bit_cast(T, x); } }\n" +
        helper + '\nextern "C" void unknown_probe(float *c, const uint32_t *a, const uint32_t *b) {\n' +
        "pearlhash_wmma_m16n16k16_f16(c, a, b);\n}\n");
      run(clang, ["-std=c++17", "-O3", "-S", "-emit-llvm", unknown, "-o", ir]);
      run(llc, ["--mtriple=nvptx64-nvidia-cuda", "--march=nvptx64", "--mattr=+ptx60",
        "--mcpu=sm_70", "-O3", "-o", ptx, ir]);
      assert.ok(!fs.readFileSync(ptx, "utf8").includes("wmma."));
    } finally {if (!retained) {fs.rmSync(output, {recursive: true, force: true});}}
  });
  test("Actual scratch alignment and cache admission handle residues and malformed overrides", {
    skip: available ? false : "LLVM21 clang/llc unavailable",
  }, () => {
    const text = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/sycl_search.inc"), "utf8");
    const internal = fs.readFileSync(path.join(__dirname, "../sycl/lib-internal.h"), "utf8");
    const scratch = extractCppBlock(text, "struct PearlF16Scratch");
    const redWord = extractCppBlock(text, "        auto red_word = [&](int index) -> uint32_t&");
    const cache = extractCppBlock(text, "  if constexpr (FP16Matrix)");
    const parser = extractCppBlock(internal, "inline bool mom_parse_env_ulong(const char* const name, unsigned long& out)");
    const retained = process.env["MOM_PEARL_LOWERING_OUTPUT"];
    const output = retained ? path.resolve(retained + "-contract") : fs.mkdtempSync(path.join(os.tmpdir(), "Pearl-wmma-contract-"));
    if (retained) {fs.mkdirSync(output, {mode: 0o700});}
    try {
      const cpp = path.join(output, "actual-contract.cpp"), exe = path.join(output, "actual-contract");
      fs.writeFileSync(cpp, `#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <new>
#include <type_traits>
#define MOM_SYCL_ADAPTIVECPP_CUDA
${scratch};
static_assert(alignof(PearlF16Scratch) == alignof(uint32_t));
static_assert(sizeof(PearlF16Scratch) == 2080);
template<class T> struct MockLocal { T* pointer; T& operator[](int index) { return pointer[index]; } };
template<bool FP16Matrix> uint32_t* word(PearlF16Scratch* raw, int index) {
  using T = std::conditional_t<FP16Matrix, PearlF16Scratch, uint32_t>;
  MockLocal<T> red{reinterpret_cast<T*>(raw)};
${redWord};
  return &red_word(index);
}
${parser}
template<bool FP16Matrix> int cache_order(const char* value) {
#ifdef _WIN32
  _putenv_s("MOM_PEARLHASH_CU_BLK", value ? value : "");
#else
  if(value) setenv("MOM_PEARLHASH_CU_BLK", value, 1); else unsetenv("MOM_PEARLHASH_CU_BLK");
#endif
  int cache_block = 0;
${cache}
  return cache_block;
}
int main() {
  alignas(32) unsigned char storage[sizeof(PearlF16Scratch) + 32];
  for(unsigned offset = 0; offset < 32; offset += 4) {
    auto raw = new(storage + offset) PearlF16Scratch{};
    auto first = word<true>(raw, 0), last = word<true>(raw, 511);
    assert(reinterpret_cast<uintptr_t>(first) % 32 == 0);
    assert(reinterpret_cast<uintptr_t>(word<true>(raw, 128)) % 32 == 0);
    assert(first >= raw->words && last < raw->words + 520);
    *last = 0xabcdef01U;
    assert(*word<true>(raw, 511) == 0xabcdef01U);
    raw->~PearlF16Scratch();
  }
  assert(cache_order<true>(nullptr) == 0);
  for(const char* value : {"", "-1", "+1", " 64", "64x", "2147483648", "999999999999999999999999999"})
    assert(cache_order<true>(value) == 0);
  assert(cache_order<true>("0") == 0);
  assert(cache_order<true>("64") == 64);
  assert(cache_order<false>("64") == 0);
}
`);
      run(clang, ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", cpp, "-o", exe]);
      run(exe, []);
    } finally {
      if (!retained) {fs.rmSync(output, {recursive: true, force: true});}
    }
  });
  test("Actual WMMA helper lowers through SSCP and assembles at its PTX floor", {
    skip: available ? false : "LLVM21 clang/llc unavailable",
  }, () => {
    const text = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/sycl_search.inc"), "utf8");
    const helper = extractCppBlock(text, "static inline void pearlhash_wmma_m16n16k16_f16(");
    const begin = helper.indexOf("asm volatile(");
    const end = /:\s*"memory"\s*\);/.exec(helper.slice(begin));
    assert.ok(begin >= 0 && end, "Actual WMMA helper memory clobber was not found");
    const asm = helper.slice(begin, begin + end.index + end[0].length).replace(/ \\\n/g, "\n");
    const packingHelper = extractCppBlock(text, "static inline uint32_t pearlhash_f16x2_s8(int8_t low, int8_t high)");
    assert.ok(packingHelper.includes("if (pearlhash_uses_wmma_f16())"));
    const packingBegin = packingHelper.indexOf("asm volatile(");
    const packingTerminator = '"r"(static_cast<int>(high)));';
    const packingEnd = packingHelper.indexOf(packingTerminator, packingBegin);
    assert.ok(packingBegin >= 0 && packingEnd >= packingBegin);
    const packing = packingHelper.slice(packingBegin, packingEnd + packingTerminator.length);
    const retained = process.env["MOM_PEARL_LOWERING_OUTPUT"];
    const output = retained ? path.resolve(retained) : fs.mkdtempSync(path.join(os.tmpdir(), "Pearl-wmma-lowering-"));
    if (retained) {fs.mkdirSync(output, {mode: 0o700});}
    try {
      const cpp = path.join(output, "actual-helper.cpp"), ir = path.join(output, "actual-helper.ll");
      for (const version of [60, 63, 64]) {
        const suffix = version >= 63 ? '".aligned"' : '""';
        fs.writeFileSync(cpp, 'extern "C" void mma_probe(unsigned *bits, const unsigned *a, const unsigned *b) {\n' +
        "const unsigned long long a_address = reinterpret_cast<unsigned long long>(a);\n" +
        "const unsigned long long b_address = reinterpret_cast<unsigned long long>(b);\n" + asm.replaceAll("SUFFIX", suffix) +
        '\n}\nextern "C" unsigned packing_probe(signed char low, signed char high) {\nunsigned packed;\n' + packing + "\nreturn packed;\n}\n");
        run(clang, ["-O3", "-S", "-emit-llvm", cpp, "-o", ir]);
        fs.appendFileSync(ir, '\n!nvvm.annotations = !{!900}\n!900 = !{ptr @mma_probe, !"kernel", i32 1}\n');
        const ptx = path.join(output, "actual-helper-ptx" + version + ".ptx");
        run(llc, ["--mtriple=nvptx64-nvidia-cuda", "--march=nvptx64", "--frame-pointer=none",
          "--mattr=+ptx" + version, "--mcpu=sm_70", "-O3", "-o", ptx, ir]);
        const emitted = fs.readFileSync(ptx, "utf8");
        assert.ok(emitted.includes(".target sm_70"));
        assert.ok(emitted.includes(".version " + (version / 10).toFixed(1)));
        assert.ok(emitted.includes(".entry mma_probe"));
        assert.ok(emitted.includes("wmma.mma.sync" + (version >= 63 ? ".aligned" : "") + ".m16n16k16.row.col.f32.f32"));
        assert.ok(emitted.includes("cvta.to.shared.u64"));
        assert.ok(emitted.includes("{ .reg .b16 lo,hi;"));
        assert.ok(emitted.includes("cvt.rn.f16.s32 lo,"));
        assert.ok(emitted.includes("cvt.rn.f16.s32 hi,"));
        assert.match(emitted, /mov\.b32 %r\d+,\{lo,hi\};/);
        if (assemblerAvailable) {
          const result = spawnSync(ptxas, ["-arch=sm_70", "-v", ptx, "-o", path.join(output, "actual-helper-ptx" + version + ".cubin")],
            {encoding: "utf8", timeout: 30000, maxBuffer: 4 * 1024 * 1024});
          fs.writeFileSync(path.join(output, "ptxas" + version + ".stdout"), result.stdout || "");
          fs.writeFileSync(path.join(output, "ptxas" + version + ".stderr"), result.stderr || "");
          fs.writeFileSync(path.join(output, "ptxas" + version + ".status"), String(result.status) + "\n");
          assert.ifError(result.error);
          assert.equal(result.signal, null);
          assert.equal(result.status, 0, result.stderr);
        }
      }
      if (!assemblerAvailable) {console.log("CUDA12.6 ptxas unavailable: LLVM-only result, not assembler qualification");}
    } finally {
      if (!retained) {fs.rmSync(output, {recursive: true, force: true});}
    }
  });
});
