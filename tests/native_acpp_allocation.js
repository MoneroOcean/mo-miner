"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process"), {describe, test} = require("node:test");
const assert = require("node:assert/strict");

describe("Windows HIP device allocation placement", () => {
  test("actual allocator retries unpublished allocations and preserves errors and ownership", {timeout: 30000}, t => {
    const compiler = process.env["CXX"] || "c++";
    const probe = spawnSync(compiler, ["--version"], {encoding: "utf8", timeout: 5000});
    if (/** @type {NodeJS.ErrnoException | undefined} */ (probe.error)?.code === "ENOENT") {
      t.skip("a host C++17 compiler is required"); return;
    }
    assert.equal(probe.status, 0, probe.stderr);
    const patch = fs.readFileSync(path.join(__dirname, "../runtime/adaptivecpp-static.patch"), "utf8");
    const start = patch.indexOf("diff --git a/src/runtime/hip/hip_allocator.cpp");
    assert.ok(start >= 0);
    const end = patch.indexOf("\ndiff --git", start + 1);
    const lines = patch.slice(start, end < 0 ? undefined : end).split("\n").slice(4);
    /** Execute the actual before/after hunk, not a separately maintained policy.
     * @param {string} side */
    const body = side => lines.filter(line => line[0] === " " || line[0] === side)
      .map(line => line.slice(1)).join("\n").split("void *hip_allocator::raw_allocate_optimized_host")[0];
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hip-allocation-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const scaffold = `
#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
using std::size_t;
enum hipError_t { hipSuccess, oom, query_error, free_error, sync_error, old_error };
constexpr size_t gib = size_t{1} << 30;
struct Snapshot { size_t free = 6*gib, total = 16*gib; hipError_t error = hipSuccess; };
struct State {
  Snapshot snapshots[8];
  hipError_t allocations[3]{}, releases[2]{}, synchronizations[2]{};
  hipError_t last = hipSuccess, reported = hipSuccess;
  int mallocs = 0, infos = 0, frees = 0, syncs = 0;
  int tokens[3]{}; bool owned[3]{};
} h;
hipError_t record(hipError_t e) { if(e != hipSuccess) h.last = e; return e; }
hipError_t hipPeekAtLastError() { return h.last; }
hipError_t hipMalloc(void** p, size_t) {
  int i = h.mallocs++; assert(i < 3);
  *p = h.allocations[i] == hipSuccess ? &h.tokens[i] : nullptr;
  h.owned[i] = *p != nullptr; return record(h.allocations[i]);
}
hipError_t hipMemGetInfo(size_t* f, size_t* total) {
  assert(h.infos < 8); auto s = h.snapshots[h.infos++];
  *f = s.free; *total = s.total; return record(s.error);
}
hipError_t hipFree(void* p) {
  assert(h.frees < 2); auto e = h.releases[h.frees++];
  if(e == hipSuccess) for(int i=0;i<3;++i) if(p == &h.tokens[i]) h.owned[i] = false;
  return record(e);
}
hipError_t hipDeviceSynchronize() { assert(h.syncs < 2); return record(h.synchronizations[h.syncs++]); }
struct allocation_hints {};
struct hip_device_manager {
  static hip_device_manager& get() { static hip_device_manager m; return m; }
  void activate_device(int) {}
};
enum class error_type { memory_allocation_error };
struct error_code { const char* component; hipError_t value; };
struct error_info { const char* operation; error_code code; error_type type; };
void register_error(int, error_info info) { assert(h.reported == hipSuccess); h.reported = info.code.value; }
#define __acpp_here() 0
struct hip_allocator {
  int _dev = 0;
  void* raw_allocate(size_t, size_t, const allocation_hints&);
};
`;
    const cases = `
int main() {
  hip_allocator allocator;
  auto run = [&](size_t size = gib) { return allocator.raw_allocate(0, size, {}); };
  void* p;
  h = State{}; p = run(gib-1);
  assert(p && h.mallocs == 1 && h.infos == 0);
#if defined(BASELINE)
  h = State{}; h.snapshots[3].free -= gib; p = run();
  if(h.mallocs != 2 || h.frees != 1) { std::cout << "baseline leaves shared allocation unchanged\\n"; return 17; }
#elif !defined(_WIN32)
  h = State{}; p = run(); assert(p && h.mallocs == 1 && h.infos == 0);
#else
  h = State{}; h.last = old_error; p = run();
  assert(p && h.infos == 0 && h.last == old_error);
  h = State{}; h.snapshots[1].free -= gib; p = run();
  assert(p && h.mallocs == 1 && h.frees == 0);
  h = State{}; h.snapshots[3].free -= gib; p = run();
  assert(p == &h.tokens[1] && h.mallocs == 2 && h.frees == 1 && h.syncs == 1);
  assert(!h.owned[0] && h.owned[1] && h.reported == hipSuccess);
  h = State{}; p = run();
  assert(p == &h.tokens[2] && h.mallocs == 3 && h.frees == 2 && h.syncs == 2);
  assert(!h.owned[0] && !h.owned[1] && h.owned[2]);
  for(int query : {0, 1, 2, 3}) {
    h = State{}; h.snapshots[query].error = query_error; p = run();
    assert(h.reported == query_error && h.last == query_error);
    assert((p != nullptr) == (query == 1 || query == 3));
    if(p) assert(h.owned[query == 1 ? 0 : 1]);
  }
  h = State{}; h.releases[0] = free_error; p = run();
  assert(p == &h.tokens[0] && h.owned[0] && h.mallocs == 1);
  assert(h.reported == free_error && h.last == free_error);
  h = State{}; h.synchronizations[0] = sync_error; p = run();
  assert(!p && !h.owned[0] && h.mallocs == 1 && h.reported == sync_error);
  for(int allocation : {0, 1, 2}) {
    h = State{}; h.allocations[allocation] = oom; p = run();
    assert(!p && h.reported == oom && h.last == oom);
    for(bool owned : h.owned) assert(!owned);
  }
  h = State{}; h.snapshots[1].total -= gib; p = run();
  assert(p && h.mallocs == 1 && h.frees == 0);
  h = State{}; for(auto& s : h.snapshots) s.free = gib; p = run();
  assert(p && h.mallocs == 1 && h.frees == 0);
#endif
  std::cout << "allocation ownership, bounded retry and sticky errors passed\\n";
}
`;
    /** @type {[string, string, string[], number][]} */
    const variants = [
      ["before", "-", ["-D_WIN32", "-DBASELINE"], 17],
      ["windows", "+", ["-D_WIN32"], 0], ["linux", "+", [], 0],
    ];
    for (const [name, side, flags, expected] of variants) {
      const source = path.join(root, `${name}.cpp`), exe = path.join(root, name);
      fs.writeFileSync(source, scaffold + body(side) + cases, {flag: "wx"});
      const built = spawnSync(compiler, ["-std=c++17", "-O0", "-g0", ...flags, source, "-o", exe],
        {encoding: "utf8", timeout: 15000, maxBuffer: 1 << 20});
      assert.equal(built.status, 0, built.stderr || built.error?.message);
      const result = spawnSync(exe, [], {encoding: "utf8", timeout: 3000, maxBuffer: 1 << 20});
      assert.equal(result.status, expected, result.stderr || result.stdout || result.error?.message);
      assert.match(result.stdout, expected ? /baseline leaves shared allocation unchanged/ : /sticky errors passed/);
    }
  });
});
